# Makefile for CUDA application

# Chemins spécifiques
CUDA_PATH = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/cuda/12.6
NCCL_PATH = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/nccl

INCLUDES = -I${CUDA_PATH}/include \
           -I${NCCL_PATH}/include

LIB_PATHS = -L${CUDA_PATH}/lib64 \
            -L${NCCL_PATH}/lib

# Compiler
NVCC = nvcc

# Source files
SRCS = main.cu kernel.cu

# Output binary
TARGET = app

# Compiler flags
NVCCFLAGS = -O2

# Ajout du lien avec NCCL
LDFLAGS = -lnccl

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -Xlinker --no-as-needed $(LIB_PATHS) -o $@ $^

run: all
	./app

clean:
	rm -f $(TARGET)

.PHONY: all clean run