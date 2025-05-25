# Makefile pour Jacobi MPI+NCCL simple (sans overlap)

CUDA_PATH = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/cuda/12.6
MPI_PATH  = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi
NCCL_PATH = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/nccl

INCLUDES  = -I${CUDA_PATH}/include -I${MPI_PATH}/include -I${NCCL_PATH}/include
LIB_PATHS = -L${CUDA_PATH}/lib64 -L${MPI_PATH}/lib -L${NCCL_PATH}/lib

NVCC      = ${CUDA_PATH}/bin/nvcc
MPICXX    = ${MPI_PATH}/bin/mpicxx

SRCS      = main.cpp kernel.cu
TARGET    = jacobi_mpi_nccl

CUDA_FLAGS = -O3 -std=c++14 -lcudart
MPI_FLAGS  = -O3 -std=c++14 -lmpi
NCCL_FLAGS = -O3 -std=c++14 -lnccl

all: $(TARGET)

kernel.o: kernel.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -dc kernel.cu -o kernel.o

$(TARGET): kernel.o main.cpp
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) main.cpp kernel.o $(LIB_PATHS) -lmpi -lnccl -o $@

run: all
	# adapter -np au nombre de GPUs
	mpirun -np 4 --map-by ppr:1:node ./$(TARGET) 1000 16384 16384 1

clean:
	rm -f *.o $(TARGET)

.PHONY: all run clean