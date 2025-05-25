# Makefile pour Jacobi MPI+NCCL simple (sans overlap)

CUDA_PATH = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/cuda/12.6
MPI_PATH  = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi
NCCL_PATH = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/nccl
NVSHMEM_PATH = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/nvshmem

INCLUDES  = -I${CUDA_PATH}/include -I${MPI_PATH}/include -I${NCCL_PATH}/include \
			-I${NVSHMEM_PATH}/include -I${NVSHMEM_PATH}/include/nvshmem
LIB_PATHS = -L${CUDA_PATH}/lib64 -L${MPI_PATH}/lib -L${NCCL_PATH}/lib \
			-L${NVSHMEM_PATH}/lib -L${NVSHMEM_PATH}/lib/nvshmem

NVCC      = ${CUDA_PATH}/bin/nvcc
MPICXX    = ${MPI_PATH}/bin/mpicxx

SRCS      = main.cpp kernel.cu
TARGET    = main

GENCODE_FLAGS = -gencode arch=compute_90,code=sm_90
NVCC_FLAGS = -Xptxas --optimize-float-atomics -ccbin=mpic++ -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -ldl $(GENCODE_FLAGS) -std=c++14 $(INCLUDES)
LD_FLAGS   = -ccbin=mpic++ $(LIB_PATHS) -lcuda -lcudart -ldl -lnvidia-ml -lnvshmem

all: $(TARGET)

$(TARGET): Makefile main.cu
	$(NVCC) $(NVCC_FLAGS) main.cu -c -o main.o
	$(NVCC) $(GENCODE_FLAGS) main.o -o $(TARGET) $(LD_FLAGS)

run: all
	# adapter -np au nombre de GPUs
	mpirun -np 4 --map-by ppr:1:node ./$(TARGET) 1000 16384 16384 1

clean:
	rm -f $(TARGET) *.o main.nsys-rep main.sqlite main.AnalysisSummary.html main.DiagnosticsSummary.html main.qdstrm

.PHONY: all run clean