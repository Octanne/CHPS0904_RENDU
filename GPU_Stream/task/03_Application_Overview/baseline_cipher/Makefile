CUDACXX=nvcc
CUDACXXFLAGS=-arch=sm_90 -O3
CXXFLAGS=-march=native -fopenmp

all: baseline

baseline: baseline.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" baseline.cu -o baseline

clean:
	rm -f baseline
