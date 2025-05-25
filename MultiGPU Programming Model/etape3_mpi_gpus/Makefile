# Paths for your cluster modules
CUDA_PATH   = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/cuda/12.6
MPI_PATH    = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi
NCCL_PATH   = /apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/comm_libs/nccl

INCLUDES    = -I${CUDA_PATH}/include \
			  -I${MPI_PATH}/include \
			  -I${NCCL_PATH}/include
LIB_PATHS   = -L${CUDA_PATH}/lib64 \
			  -L${MPI_PATH}/lib \
			  -L${NCCL_PATH}/lib

NVCC        = ${CUDA_PATH}/bin/nvcc
NVCCFLAGS   = -O3 -std=c++14 -lcudart \
			  -Xcompiler "-fopenmp" \
			  -I${CUDA_PATH}/include
LDFLAGS     = -lmpi -lnccl -lstdc++

SRCS        = main.cpp kernel.cu
TARGET      = app

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LIB_PATHS) $(LDFLAGS) \
		-Xlinker --no-as-needed

run: all
	# Adjust -np to number of ranks
	mpirun -np 4 ./$(TARGET) -niter 1000 -nx 4096 -ny 4096 -nccheck 1

clean:
	rm -f $(TARGET) *.o main.nsys-rep main.sqlite main.AnalysisSummary.html main.DiagnosticsSummary.html main.qdstrm

.PHONY: all run clean