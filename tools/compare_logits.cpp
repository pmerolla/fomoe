#include "llama.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main() {
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    const char *model_path = "/home/paul/models/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf";
    fprintf(stderr, "Loading model...\n");
    llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.no_perf = true;
    ctx_params.embeddings = true;  // Enable embedding extraction

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int n_embd = llama_model_n_embd(model);
    fprintf(stderr, "n_vocab=%d, n_embd=%d\n", n_vocab, n_embd);

    // Process single token 760 ("The")
    {
        llama_token tok = 760;
        llama_batch batch = llama_batch_get_one(&tok, 1);
        fprintf(stderr, "Processing single token 760...\n");
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Failed to decode\n");
            return 1;
        }

        // Get logits
        const float *logits = llama_get_logits_ith(ctx, 0);

        // Save logits to binary file
        FILE *f = fopen("/tmp/ref_logits_tok760.bin", "wb");
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        fprintf(stderr, "Saved %d logits to /tmp/ref_logits_tok760.bin\n", n_vocab);

        // Get embeddings (hidden state after final norm)
        const float *embd = llama_get_embeddings_ith(ctx, 0);
        if (embd) {
            FILE *fe = fopen("/tmp/ref_embd_tok760.bin", "wb");
            fwrite(embd, sizeof(float), n_embd, fe);
            fclose(fe);
            fprintf(stderr, "Saved %d embeddings to /tmp/ref_embd_tok760.bin\n", n_embd);
            fprintf(stderr, "  embd[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    embd[0], embd[1], embd[2], embd[3], embd[4], embd[5], embd[6], embd[7]);
        } else {
            fprintf(stderr, "WARNING: embeddings not available\n");
        }

        // Print first 20 logits
        fprintf(stderr, "Single token 760 logits (first 20):\n");
        for (int i = 0; i < 20; i++) {
            fprintf(stderr, "  logit[%d] = %.6f\n", i, logits[i]);
        }
        fprintf(stderr, "  logit[11751]('Paris')=%.4f logit[264]('a')=%.4f logit[13]('.')=%.4f\n",
                logits[11751], logits[264], logits[13]);
    }

    // Fresh context for 5-token prompt
    llama_free(ctx);
    ctx_params.embeddings = true;
    ctx = llama_init_from_model(model, ctx_params);

    // Tokenize and process "The capital of France is"
    {
        const char *prompt = "The capital of France is";
        std::vector<llama_token> tokens(128);
        int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                       tokens.data(), tokens.size(), false, true);
        tokens.resize(n_tokens);
        fprintf(stderr, "\nPrompt: \"%s\" (%d tokens):", prompt, n_tokens);
        for (int i = 0; i < n_tokens; i++) fprintf(stderr, " %d", tokens[i]);
        fprintf(stderr, "\n");

        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        fprintf(stderr, "Processing...\n");
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Failed to decode\n");
            return 1;
        }

        // Get logits for last token
        const float *logits = llama_get_logits_ith(ctx, n_tokens - 1);

        // Save full logits
        FILE *f = fopen("/tmp/ref_logits_full.bin", "wb");
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        fprintf(stderr, "Saved %d logits to /tmp/ref_logits_full.bin\n", n_vocab);

        // Get embeddings for last token
        const float *embd = llama_get_embeddings_ith(ctx, n_tokens - 1);
        if (embd) {
            FILE *fe = fopen("/tmp/ref_embd_full.bin", "wb");
            fwrite(embd, sizeof(float), n_embd, fe);
            fclose(fe);
            fprintf(stderr, "Saved %d embeddings to /tmp/ref_embd_full.bin\n", n_embd);
            fprintf(stderr, "  embd[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    embd[0], embd[1], embd[2], embd[3], embd[4], embd[5], embd[6], embd[7]);
        } else {
            fprintf(stderr, "WARNING: embeddings not available\n");
        }

        // Print logit stats
        fprintf(stderr, "\nFirst 20 logits:\n");
        for (int i = 0; i < 20; i++) {
            fprintf(stderr, "  logit[%d] = %.6f\n", i, logits[i]);
        }
        fprintf(stderr, "  logit[11751]('Paris')=%.4f logit[264]('a')=%.4f logit[13]('.')=%.4f\n",
                logits[11751], logits[264], logits[13]);

        // Top 10
        typedef struct { float v; int i; } lp;
        lp top[10];
        for (int i = 0; i < 10; i++) { top[i].v = -1e30; top[i].i = 0; }
        for (int i = 0; i < n_vocab; i++) {
            for (int j = 0; j < 10; j++) {
                if (logits[i] > top[j].v) {
                    for (int k = 9; k > j; k--) top[k] = top[k-1];
                    top[j].v = logits[i]; top[j].i = i;
                    break;
                }
            }
        }
        fprintf(stderr, "\nTop 10:\n");
        for (int j = 0; j < 10; j++) {
            char buf[256];
            int n = llama_token_to_piece(vocab, top[j].i, buf, sizeof(buf)-1, 0, true);
            buf[n] = 0;
            fprintf(stderr, "  %6d (\"%s\"): %.4f\n", top[j].i, buf, top[j].v);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
