# Makefile pour Jacobi MPI+NCCL simple (sans overlap)

CUDA_PATH = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/cuda/12.6
MPI_PATH  = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi
NCCL_PATH = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/nccl

INCLUDES  = -I${CUDA_PATH}/include -I${MPI_PATH}/include -I${NCCL_PATH}/include
LIB_PATHS = -L${CUDA_PATH}/lib64 -L${MPI_PATH}/lib -L${NCCL_PATH}/lib

NVCC      = ${CUDA_PATH}/bin/nvcc
MPICXX    = ${MPI_PATH}/bin/mpicxx

SRCS      = main.cpp kernel.cu
TARGET    = app

NVCC_FLAGS = -lineinfo -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -std=c++14
MPI_FLAGS  = -DUSE_NVTX -O3 $(INCLUDES) -std=c++14
LD_FLAGS   = $(LIB_PATHS) -lcudart -ldl -lnccl

all: $(TARGET)

$(TARGET): Makefile main.cpp kernel.o
	$(MPICXX) $(MPI_FLAGS) main.cpp kernel.o $(LD_FLAGS) -o $@

kernel.o: Makefile kernel.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) kernel.cu -c

run: all
	# adapter -np au nombre de GPUs
	mpirun -np 4 --map-by ppr:1:node ./$(TARGET) 1000 16384 16384 1

clean:
	rm -f $(TARGET) *.o main.nsys-rep main.sqlite main.AnalysisSummary.html main.DiagnosticsSummary.html main.qdstrm

.PHONY: all run clean