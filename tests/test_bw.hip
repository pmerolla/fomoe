/*
 * test_bw.hip — PCIe bandwidth benchmark for both GPUs
 *
 * Measures H2D and D2H bandwidth at various transfer sizes for both GPUs,
 * using both pageable and pinned host memory. Reports effective bandwidth
 * in GB/s for each configuration.
 *
 * Build: USE_GPU=1 make test_bw
 * Run via rq: ./submit.py --name "PCIe BW x8x8" "./frontrunner-tp/test_bw"
 */

#include "gpu_compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    size_t size;
    const char *label;
} test_size_t;

static void bench_h2d_d2h(int device, void *h_buf, void *d_buf, size_t size,
                           int warmup, int iters, double *h2d_gbps, double *d2h_gbps) {
    HIP_CHECK(hipSetDevice(device));

    // Warmup H2D
    for (int i = 0; i < warmup; i++)
        HIP_CHECK(hipMemcpy(d_buf, h_buf, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark H2D
    double t0 = get_time_ms();
    for (int i = 0; i < iters; i++)
        HIP_CHECK(hipMemcpy(d_buf, h_buf, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    double t1 = get_time_ms();
    *h2d_gbps = (double)size * iters / ((t1 - t0) / 1000.0) / 1e9;

    // Warmup D2H
    for (int i = 0; i < warmup; i++)
        HIP_CHECK(hipMemcpy(h_buf, d_buf, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark D2H
    t0 = get_time_ms();
    for (int i = 0; i < iters; i++)
        HIP_CHECK(hipMemcpy(h_buf, d_buf, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    t1 = get_time_ms();
    *d2h_gbps = (double)size * iters / ((t1 - t0) / 1000.0) / 1e9;
}

static void bench_h2d_d2h_async(int device, void *h_buf, void *d_buf, size_t size,
                                 int warmup, int iters, hipStream_t stream,
                                 double *h2d_gbps, double *d2h_gbps) {
    HIP_CHECK(hipSetDevice(device));

    // Warmup H2D
    for (int i = 0; i < warmup; i++)
        HIP_CHECK(hipMemcpyAsync(d_buf, h_buf, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Benchmark H2D
    double t0 = get_time_ms();
    for (int i = 0; i < iters; i++)
        HIP_CHECK(hipMemcpyAsync(d_buf, h_buf, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    double t1 = get_time_ms();
    *h2d_gbps = (double)size * iters / ((t1 - t0) / 1000.0) / 1e9;

    // Warmup D2H
    for (int i = 0; i < warmup; i++)
        HIP_CHECK(hipMemcpyAsync(h_buf, d_buf, size, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Benchmark D2H
    t0 = get_time_ms();
    for (int i = 0; i < iters; i++)
        HIP_CHECK(hipMemcpyAsync(h_buf, d_buf, size, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    t1 = get_time_ms();
    *d2h_gbps = (double)size * iters / ((t1 - t0) / 1000.0) / 1e9;
}

int main() {
    int n_devices;
    HIP_CHECK(hipGetDeviceCount(&n_devices));
    if (n_devices < 1) {
        fprintf(stderr, "No GPUs found\n");
        return 1;
    }

    fprintf(stderr, "=== PCIe Bandwidth Benchmark ===\n");
    fprintf(stderr, "GPUs found: %d\n\n", n_devices);

    // Print device info
    for (int d = 0; d < n_devices && d < 2; d++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, d));
        fprintf(stderr, "GPU %d: %s, %d MB VRAM, PCIe gen: %d\n",
                d, prop.name, (int)(prop.totalGlobalMem / (1024 * 1024)),
                prop.pciDomainID);  // Note: may not report gen correctly
    }

    // Report PCIe link info from sysfs
    fprintf(stderr, "\nPCIe link status (from sysfs):\n");
    fflush(stderr);
    system("for dev in /sys/bus/pci/devices/*/current_link_width; do "
           "dir=$(dirname $dev); bdf=$(basename $dir); "
           "class=$(cat $dir/class 2>/dev/null); "
           "if echo $class | grep -q '0x03'; then "  // display class
           "echo \"  $bdf: x$(cat $dev) @ $(cat $dir/current_link_speed)\"; "
           "fi; done 2>/dev/null");
    // Also check upstream bridges
    system("for dev in /sys/bus/pci/devices/*/current_link_width; do "
           "dir=$(dirname $dev); bdf=$(basename $dir); "
           "class=$(cat $dir/class 2>/dev/null); "
           "if echo $class | grep -q '0x0604'; then "  // PCI bridge class
           "vendor=$(cat $dir/vendor 2>/dev/null); "
           "if [ \"$vendor\" = \"0x1002\" ]; then "  // AMD
           "echo \"  $bdf (bridge): x$(cat $dev) @ $(cat $dir/current_link_speed)\"; "
           "fi; fi; done 2>/dev/null");
    fprintf(stderr, "\n");

    test_size_t sizes[] = {
        {     4096, "4 KB   (router scores)"},
        {     7424, "7.25 KB (1 expert Q4_K)"},
        {    16384, "16 KB  (hidden state relay)"},
        {    32768, "32 KB  (2 experts)"},
        {    65536, "64 KB  (8 experts batched)"},
        {   131072, "128 KB "},
        {   262144, "256 KB "},
        {   524288, "512 KB "},
        {  1048576, "1 MB   "},
        {  4194304, "4 MB   "},
        { 16777216, "16 MB  "},
        { 67108864, "64 MB  (layer weights)"},
        {268435456, "256 MB "},
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    size_t max_size = sizes[n_sizes - 1].size;

    for (int d = 0; d < n_devices && d < 2; d++) {
        HIP_CHECK(hipSetDevice(d));

        // Allocate device memory
        void *d_buf;
        HIP_CHECK(hipMalloc(&d_buf, max_size));

        // ---- Pinned host memory ----
        void *h_pinned;
        HIP_CHECK(hipHostMalloc(&h_pinned, max_size, hipHostMallocDefault));
        memset(h_pinned, 0x42, max_size);

        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));

        fprintf(stderr, "--- GPU %d: Pinned memory (hipMemcpyAsync) ---\n", d);
        fprintf(stderr, "%-30s  %12s  %12s\n", "Size", "H2D (GB/s)", "D2H (GB/s)");
        fprintf(stderr, "%-30s  %12s  %12s\n", "----", "----------", "----------");

        for (int i = 0; i < n_sizes; i++) {
            size_t sz = sizes[i].size;
            // More iterations for small transfers to get stable timing
            int iters = sz < 65536 ? 5000 : (sz < 1048576 ? 2000 : (sz < 16777216 ? 500 : 100));
            int warmup = iters / 10;

            double h2d, d2h;
            bench_h2d_d2h_async(d, h_pinned, d_buf, sz, warmup, iters, stream, &h2d, &d2h);
            fprintf(stderr, "%-30s  %10.2f    %10.2f\n", sizes[i].label, h2d, d2h);
            fflush(stderr);
        }

        // ---- Pageable (non-pinned) host memory ----
        void *h_page = malloc(max_size);
        memset(h_page, 0x42, max_size);

        fprintf(stderr, "\n--- GPU %d: Pageable memory (hipMemcpy sync) ---\n", d);
        fprintf(stderr, "%-30s  %12s  %12s\n", "Size", "H2D (GB/s)", "D2H (GB/s)");
        fprintf(stderr, "%-30s  %12s  %12s\n", "----", "----------", "----------");

        for (int i = 0; i < n_sizes; i++) {
            size_t sz = sizes[i].size;
            int iters = sz < 65536 ? 2000 : (sz < 1048576 ? 500 : (sz < 16777216 ? 200 : 50));
            int warmup = iters / 10;

            double h2d, d2h;
            bench_h2d_d2h(d, h_page, d_buf, sz, warmup, iters, &h2d, &d2h);
            fprintf(stderr, "%-30s  %10.2f    %10.2f\n", sizes[i].label, h2d, d2h);
            fflush(stderr);
        }

        free(h_page);
        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipHostFree(h_pinned));
        HIP_CHECK(hipFree(d_buf));
        fprintf(stderr, "\n");
    }

    // ---- Bidirectional (both GPUs simultaneously) ----
    if (n_devices >= 2) {
        fprintf(stderr, "--- Bidirectional: GPU0 H2D + GPU1 H2D simultaneously ---\n");
        fprintf(stderr, "%-30s  %12s  %12s  %12s\n", "Size", "GPU0 (GB/s)", "GPU1 (GB/s)", "Total (GB/s)");
        fprintf(stderr, "%-30s  %12s  %12s  %12s\n", "----", "-----------", "-----------", "-----------");

        void *d_buf0, *d_buf1;
        void *h_pin0, *h_pin1;
        hipStream_t s0, s1;

        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMalloc(&d_buf0, max_size));
        HIP_CHECK(hipHostMalloc(&h_pin0, max_size, hipHostMallocDefault));
        HIP_CHECK(hipStreamCreate(&s0));
        memset(h_pin0, 0x42, max_size);

        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipMalloc(&d_buf1, max_size));
        HIP_CHECK(hipHostMalloc(&h_pin1, max_size, hipHostMallocDefault));
        HIP_CHECK(hipStreamCreate(&s1));
        memset(h_pin1, 0x42, max_size);

        test_size_t bidir_sizes[] = {
            {    16384, "16 KB  (relay)"},
            {    65536, "64 KB  (8 experts)"},
            {  1048576, "1 MB   "},
            { 16777216, "16 MB  "},
            { 67108864, "64 MB  "},
            {268435456, "256 MB "},
        };
        int n_bidir = sizeof(bidir_sizes) / sizeof(bidir_sizes[0]);

        for (int i = 0; i < n_bidir; i++) {
            size_t sz = bidir_sizes[i].size;
            int iters = sz < 1048576 ? 2000 : (sz < 16777216 ? 500 : 100);
            int warmup = iters / 10;

            // Warmup both
            for (int w = 0; w < warmup; w++) {
                HIP_CHECK(hipSetDevice(0));
                HIP_CHECK(hipMemcpyAsync(d_buf0, h_pin0, sz, hipMemcpyHostToDevice, s0));
                HIP_CHECK(hipSetDevice(1));
                HIP_CHECK(hipMemcpyAsync(d_buf1, h_pin1, sz, hipMemcpyHostToDevice, s1));
                HIP_CHECK(hipStreamSynchronize(s0));
                HIP_CHECK(hipStreamSynchronize(s1));
            }

            // Benchmark: launch both simultaneously
            double t0 = get_time_ms();
            for (int it = 0; it < iters; it++) {
                HIP_CHECK(hipSetDevice(0));
                HIP_CHECK(hipMemcpyAsync(d_buf0, h_pin0, sz, hipMemcpyHostToDevice, s0));
                HIP_CHECK(hipSetDevice(1));
                HIP_CHECK(hipMemcpyAsync(d_buf1, h_pin1, sz, hipMemcpyHostToDevice, s1));
            }
            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipStreamSynchronize(s0));
            HIP_CHECK(hipSetDevice(1));
            HIP_CHECK(hipStreamSynchronize(s1));
            double t1 = get_time_ms();

            double elapsed_s = (t1 - t0) / 1000.0;
            double per_gpu = (double)sz * iters / elapsed_s / 1e9;
            double total = per_gpu * 2;

            fprintf(stderr, "%-30s  %10.2f    %10.2f    %10.2f\n",
                    bidir_sizes[i].label, per_gpu, per_gpu, total);
            fflush(stderr);
        }

        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipStreamDestroy(s0));
        HIP_CHECK(hipHostFree(h_pin0));
        HIP_CHECK(hipFree(d_buf0));

        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipStreamDestroy(s1));
        HIP_CHECK(hipHostFree(h_pin1));
        HIP_CHECK(hipFree(d_buf1));
    }

    // ---- NVMe sequential read bandwidth (fio) ----
    fprintf(stderr, "\n--- NVMe Sequential Read (fio, 128KB blocks, QD=32) ---\n");
    fflush(stderr);

    // Find NVMe devices
    const char *nvme_paths[] = {
        "/mnt/nvme_shard_a", "/mnt/nvme_shard_b",
        "/home/paul/nvme_shard_c", "/mnt/nvme_shard_d",
        NULL
    };
    for (int i = 0; nvme_paths[i]; i++) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "if [ -d '%s' ]; then "
                 "fio --name=seqread --filename=%s/fio_bench_tmp "
                 "--rw=read --bs=128k --direct=1 --ioengine=io_uring "
                 "--iodepth=32 --size=1G --runtime=5 --time_based "
                 "--output-format=terse 2>/dev/null | "
                 "awk -F';' '{printf \"  %s: %%.2f GB/s\\n\", $7/1024/1024}' ; "
                 "rm -f %s/fio_bench_tmp; fi",
                 nvme_paths[i], nvme_paths[i], nvme_paths[i], nvme_paths[i]);
        system(cmd);
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
