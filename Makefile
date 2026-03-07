CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter -Iinclude -DQMOE_GPU
LDFLAGS = -lm -lpthread -luring -fopenmp

OBJ_DIR = obj

# GPU support: USE_GPU=1 (AMD ROCm/HIP) or USE_CUDA=1 (NVIDIA CUDA)
ifdef USE_CUDA
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_ARCH ?= sm_80
NVCC_FLAGS = -O2 -arch=$(CUDA_ARCH) -Iinclude -DFR_GPU -DFR_CUDA -DQMOE_GPU \
             --expt-relaxed-constexpr -Wno-deprecated-gpu-targets
HIP_CFLAGS = $(NVCC_FLAGS)
CFLAGS += -DFR_GPU -DFR_CUDA
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
LINKER = $(CC)
HIPCC = $(NVCC)

else ifdef USE_GPU
ROCM_PATH ?= /opt/rocm
HIPCC = $(ROCM_PATH)/bin/hipcc
HIP_CFLAGS = -O2 --offload-arch=gfx1200 -Iinclude -DFR_GPU -DQMOE_GPU
CFLAGS += -DFR_GPU
LDFLAGS += -L$(ROCM_PATH)/lib -lamdhip64
LINKER = $(HIPCC)

else
LINKER = $(CC)
endif

# C source files (reusable modules from qwen3.5-moe)
C_SRCS = src/expert_cache.c src/freq_profile.c src/car.c src/expert_store.c \
         src/nvme_io.c src/model.c src/gguf.c src/quant.c src/tensor.c \
         src/tokenizer.c src/sampler.c src/prefetch.c src/cpu_expert.c \
         src/inference.c src/main.c
C_OBJS = $(patsubst src/%.c,$(OBJ_DIR)/%.o,$(C_SRCS))

# HIP/CUDA source files
GPU_SRCS = src/gpu_kernels.hip src/tp.hip
GPU_OBJS = $(patsubst src/%.hip,$(OBJ_DIR)/%.o,$(GPU_SRCS))

.PHONY: all clean test_tp test_bw qwen-moe

all: qwen-moe

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# C object files
$(OBJ_DIR)/%.o: src/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# GPU object files
ifdef USE_CUDA
$(OBJ_DIR)/%.o: src/%.hip | $(OBJ_DIR)
	ln -sf $(abspath $<) $(OBJ_DIR)/$*.cu
	$(NVCC) $(NVCC_FLAGS) -c $(OBJ_DIR)/$*.cu -o $@
else ifdef USE_GPU
$(OBJ_DIR)/%.o: src/%.hip | $(OBJ_DIR)
	$(HIPCC) $(HIP_CFLAGS) -c $< -o $@
endif

# Main binary
qwen-moe: $(C_OBJS) $(GPU_OBJS)
	$(LINKER) -o $@ $^ $(LDFLAGS)

# TP test binary (standalone)
ifdef USE_CUDA
$(OBJ_DIR)/test_tp.o: tests/test_tp.hip include/gpu_compat.h include/tp.h include/quant_types.h | $(OBJ_DIR)
	ln -sf $(abspath tests/test_tp.hip) $(OBJ_DIR)/test_tp.cu
	$(NVCC) $(NVCC_FLAGS) -c $(OBJ_DIR)/test_tp.cu -o $@
else ifdef USE_GPU
$(OBJ_DIR)/test_tp.o: tests/test_tp.hip include/gpu_compat.h include/tp.h include/quant_types.h | $(OBJ_DIR)
	$(HIPCC) $(HIP_CFLAGS) -c $< -o $@
endif

test_tp: $(OBJ_DIR)/tp.o $(OBJ_DIR)/test_tp.o
	$(LINKER) -o $@ $^ $(LDFLAGS)

# PCIe bandwidth test (standalone)
ifdef USE_CUDA
$(OBJ_DIR)/test_bw.o: tests/test_bw.hip include/gpu_compat.h | $(OBJ_DIR)
	ln -sf $(abspath tests/test_bw.hip) $(OBJ_DIR)/test_bw.cu
	$(NVCC) $(NVCC_FLAGS) -c $(OBJ_DIR)/test_bw.cu -o $@
else ifdef USE_GPU
$(OBJ_DIR)/test_bw.o: tests/test_bw.hip include/gpu_compat.h | $(OBJ_DIR)
	$(HIPCC) $(HIP_CFLAGS) -c $< -o $@
endif

test_bw: $(OBJ_DIR)/test_bw.o
	$(LINKER) -o $@ $^ $(LDFLAGS)

# Cache simulation test (CPU-only, validates hit rate assumptions)
test_cache_sim: tests/test_cache_sim.c src/expert_cache.c src/freq_profile.c
	$(CC) -O3 -march=native -Iinclude -o $@ $^ -lm

clean:
	rm -rf $(OBJ_DIR) test_tp test_cache_sim qwen-moe
