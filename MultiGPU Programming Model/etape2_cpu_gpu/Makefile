# Makefile for CUDA application

# Compiler
NVCC = nvcc

# Source files
SRCS = main.cu kernel.cu

# Output binary
TARGET = app

# Compiler flags
NVCCFLAGS = -O2

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

run: all
	./app

clean:
	rm -f $(TARGET) *.o main.nsys-rep main.sqlite main.AnalysisSummary.html main.DiagnosticsSummary.html

.PHONY: all clean run