#!/usr/bin/env python3
"""Read and analyze fomoe routing trace files.

Usage:
    python3 tools/read_trace.py trace.bin                # summary
    python3 tools/read_trace.py trace.bin --per-layer     # per-layer substitution rates
    python3 tools/read_trace.py trace.bin --per-token     # per-token substitution counts
    python3 tools/read_trace.py trace.bin --expert-freq   # expert selection frequency
    python3 tools/read_trace.py trace.bin --jaccard REF   # Jaccard similarity vs reference trace
    python3 tools/read_trace.py trace.bin --csv out.csv   # export all records to CSV
"""
import struct, sys, argparse, os
from collections import Counter, defaultdict

TRACE_MAGIC = 0x54524345
TRACE_VERSION = 1
SRC_NAMES = {0: "VRAM", 1: "RAM", 2: "NVMe", 3: "CAR", 4: "DROP"}


def read_trace(path):
    with open(path, "rb") as f:
        hdr = f.read(32)
        if len(hdr) < 32:
            raise ValueError("File too small for header")
        magic, version, n_layers, n_experts, top_k, _res, _res2, _res3 = \
            struct.unpack("<IIHHHHQQ", hdr)
        assert magic == TRACE_MAGIC, f"Bad magic: {magic:#x} (expected {TRACE_MAGIC:#x})"
        assert version == TRACE_VERSION, f"Unsupported version: {version}"

        rec_fmt = f"<IHBB{top_k}h{top_k}f{top_k}h{top_k}f{top_k}B"
        rec_size_raw = struct.calcsize(rec_fmt)
        pad = (8 - rec_size_raw % 8) % 8
        if pad:
            rec_fmt += f"{pad}x"
        rec_size = struct.calcsize(rec_fmt)

        records = []
        data = f.read()
        offset = 0
        while offset + rec_size <= len(data):
            fields = struct.unpack_from(rec_fmt, data, offset)
            offset += rec_size
            idx = 0
            token_id = fields[idx]; idx += 1
            layer_id = fields[idx]; idx += 1
            n_used = fields[idx]; idx += 1
            n_subs = fields[idx]; idx += 1
            orig_ids = list(fields[idx:idx+top_k]); idx += top_k
            orig_scores = list(fields[idx:idx+top_k]); idx += top_k
            final_ids = list(fields[idx:idx+top_k]); idx += top_k
            final_scores = list(fields[idx:idx+top_k]); idx += top_k
            source = list(fields[idx:idx+top_k]); idx += top_k
            records.append({
                "token": token_id, "layer": layer_id,
                "n_used": n_used, "n_subs": n_subs,
                "orig_ids": orig_ids[:n_used],
                "orig_scores": orig_scores[:n_used],
                "final_ids": final_ids[:n_used],
                "final_scores": final_scores[:n_used],
                "source": source[:n_used],
            })

    info = {
        "n_layers": n_layers, "n_experts": n_experts,
        "top_k": top_k, "records": records,
        "path": path,
    }
    n_tokens = max(r["token"] for r in records) + 1 if records else 0
    info["n_tokens"] = n_tokens
    return info


def print_summary(trace):
    recs = trace["records"]
    src_counts = Counter()
    total_subs = 0
    for r in recs:
        for s in r["source"]:
            src_counts[s] += 1
        total_subs += r["n_subs"]

    total = sum(src_counts.values())
    print(f"File: {trace['path']}")
    print(f"  {len(recs)} records, {trace['n_tokens']} tokens, "
          f"{trace['n_layers']} layers, {trace['n_experts']} experts, top-{trace['top_k']}")
    print(f"  {total_subs} total substitutions "
          f"({total_subs / trace['n_tokens']:.1f}/token)" if trace['n_tokens'] else "")
    print(f"\nSource distribution:")
    for code in sorted(src_counts):
        name = SRC_NAMES.get(code, f"?{code}")
        pct = 100 * src_counts[code] / total if total else 0
        print(f"  {name:5s}: {src_counts[code]:>8d}  ({pct:5.1f}%)")
    print(f"  Total: {total:>8d}")


def print_per_layer(trace):
    recs = trace["records"]
    layer_subs = defaultdict(int)
    layer_count = defaultdict(int)
    layer_src = defaultdict(Counter)
    for r in recs:
        layer_subs[r["layer"]] += r["n_subs"]
        layer_count[r["layer"]] += 1
        for s in r["source"]:
            layer_src[r["layer"]][s] += 1

    print(f"{'Layer':>5} {'Tokens':>6} {'Subs':>6} {'Subs/tok':>8}  "
          f"{'VRAM%':>6} {'RAM%':>6} {'NVMe%':>6} {'CAR%':>6}")
    print("-" * 70)
    for layer in sorted(layer_count):
        n = layer_count[layer]
        s = layer_subs[layer]
        sc = layer_src[layer]
        tot = sum(sc.values())
        print(f"{layer:>5d} {n:>6d} {s:>6d} {s/n:>8.2f}  "
              f"{100*sc[0]/tot:>6.1f} {100*sc[1]/tot:>6.1f} "
              f"{100*sc[2]/tot:>6.1f} {100*sc[3]/tot:>6.1f}")


def print_per_token(trace):
    recs = trace["records"]
    token_subs = defaultdict(int)
    token_src = defaultdict(Counter)
    for r in recs:
        token_subs[r["token"]] += r["n_subs"]
        for s in r["source"]:
            token_src[r["token"]][s] += 1

    print(f"{'Token':>6} {'Subs':>5}  {'VRAM':>5} {'RAM':>5} {'NVMe':>5} {'CAR':>5}")
    print("-" * 45)
    for tok in sorted(token_subs):
        s = token_subs[tok]
        sc = token_src[tok]
        print(f"{tok:>6d} {s:>5d}  {sc[0]:>5d} {sc[1]:>5d} {sc[2]:>5d} {sc[3]:>5d}")


def print_expert_freq(trace):
    recs = trace["records"]
    orig_freq = Counter()  # what the router wanted
    final_freq = Counter() # what actually ran
    for r in recs:
        for eid in r["orig_ids"]:
            orig_freq[(r["layer"], eid)] += 1
        for eid in r["final_ids"]:
            final_freq[(r["layer"], eid)] += 1

    # Per-layer entropy / concentration
    print(f"{'Layer':>5} {'Unique(orig)':>12} {'Unique(final)':>13} {'Overlap%':>9}")
    print("-" * 45)
    for layer in range(trace["n_layers"]):
        orig_set = {eid for (l, eid) in orig_freq if l == layer}
        final_set = {eid for (l, eid) in final_freq if l == layer}
        overlap = len(orig_set & final_set) / len(orig_set | final_set) * 100 if orig_set | final_set else 0
        print(f"{layer:>5d} {len(orig_set):>12d} {len(final_set):>13d} {overlap:>8.1f}%")


def compute_jaccard(trace_a, trace_b):
    """Per-token, per-layer Jaccard similarity of expert sets between two traces."""
    recs_a = {(r["token"], r["layer"]): set(r["final_ids"]) for r in trace_a["records"]}
    recs_b = {(r["token"], r["layer"]): set(r["final_ids"]) for r in trace_b["records"]}

    common_keys = sorted(set(recs_a) & set(recs_b))
    if not common_keys:
        print("No overlapping (token, layer) pairs found.")
        return

    # Per-token average Jaccard
    token_jaccard = defaultdict(list)
    for tok, layer in common_keys:
        a, b = recs_a[(tok, layer)], recs_b[(tok, layer)]
        j = len(a & b) / len(a | b) if a | b else 1.0
        token_jaccard[tok].append(j)

    print(f"{'Token':>6} {'AvgJaccard':>10} {'MinJaccard':>10} {'Layers':>6}")
    print("-" * 40)
    for tok in sorted(token_jaccard):
        vals = token_jaccard[tok]
        print(f"{tok:>6d} {sum(vals)/len(vals):>10.4f} {min(vals):>10.4f} {len(vals):>6d}")

    all_vals = [j for vals in token_jaccard.values() for j in vals]
    print(f"\nOverall: mean={sum(all_vals)/len(all_vals):.4f}, "
          f"min={min(all_vals):.4f}, records={len(all_vals)}")


def export_csv(trace, path):
    with open(path, "w") as f:
        top_k = trace["top_k"]
        cols = ["token", "layer", "n_used", "n_subs"]
        for i in range(top_k):
            cols.extend([f"orig_id_{i}", f"orig_score_{i}",
                         f"final_id_{i}", f"final_score_{i}", f"source_{i}"])
        f.write(",".join(cols) + "\n")
        for r in trace["records"]:
            row = [r["token"], r["layer"], r["n_used"], r["n_subs"]]
            for i in range(top_k):
                if i < r["n_used"]:
                    row.extend([r["orig_ids"][i], f"{r['orig_scores'][i]:.6f}",
                                r["final_ids"][i], f"{r['final_scores'][i]:.6f}",
                                SRC_NAMES.get(r["source"][i], "?")])
                else:
                    row.extend(["", "", "", "", ""])
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Exported {len(trace['records'])} records to {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze fomoe routing traces")
    parser.add_argument("trace", help="Path to .trace binary file")
    parser.add_argument("--per-layer", action="store_true", help="Per-layer substitution rates")
    parser.add_argument("--per-token", action="store_true", help="Per-token substitution counts")
    parser.add_argument("--expert-freq", action="store_true", help="Expert frequency analysis")
    parser.add_argument("--jaccard", metavar="REF", help="Jaccard similarity vs reference trace")
    parser.add_argument("--csv", metavar="OUT", help="Export to CSV")
    args = parser.parse_args()

    trace = read_trace(args.trace)
    print_summary(trace)

    if args.per_layer:
        print("\n--- Per-Layer ---")
        print_per_layer(trace)
    if args.per_token:
        print("\n--- Per-Token ---")
        print_per_token(trace)
    if args.expert_freq:
        print("\n--- Expert Frequency ---")
        print_expert_freq(trace)
    if args.jaccard:
        print(f"\n--- Jaccard vs {args.jaccard} ---")
        ref = read_trace(args.jaccard)
        compute_jaccard(trace, ref)
    if args.csv:
        export_csv(trace, args.csv)


if __name__ == "__main__":
    main()
