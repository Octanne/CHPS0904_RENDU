CUDACXX=nvcc
CUDACXXFLAGS=-arch=sm_90 -O3
CXXFLAGS=-march=native -fopenmp
NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true

all: baseline

baseline: baseline.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" baseline.cu -o baseline

profile: baseline
	$(NSYS) $(NSYSFLAGS) -o baseline-report ./baseline

clean:
	rm -f baseline *.qdrep *.sqlite
