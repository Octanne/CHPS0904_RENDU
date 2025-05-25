// etape2_cpu_gpu : 1 CPU + 1 GPU : délégation du noyau Jacobi à un GPU. Montre l'accélération GPU quand le problème tient sur un seul dispositif.
// Solveur Jacobi sur grille 2D de taille N x N, T itérations.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 8192
#define T 1000

// Déclaration du kernel déplacé
__global__ void jacobi_kernel(double* A, double* B, int N_SIZE);

// Fonction pour exécuter Jacobi sur le GPU avec des flux
void jacobi_gpu_stream(double* h_A, double* h_B) {
    double *d_A, *d_B;
    size_t size = N * N * sizeof(double);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((N-2+block.x-1)/block.x, (N-2+block.y-1)/block.y);

    for (int t = 0; t < T; t++) {
        jacobi_kernel<<<grid, block, 0, stream>>>(d_A, d_B, N);
        cudaStreamSynchronize(stream);
        double* tmp = d_A; d_A = d_B; d_B = tmp;
    }

    cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
    double *A = (double*)malloc(N*N*sizeof(double));
    double *B = (double*)malloc(N*N*sizeof(double));
    for (int i = 0; i < N*N; i++) A[i] = B[i] = 0.0;

    clock_t start = clock();
    jacobi_gpu_stream(A, B);
    clock_t end = clock();

    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Terminé %s\n", "etape2_cpu_gpu (1CPU + 1GPU + 1stream)");
    printf("GPU time: %.6f seconds\n", gpu_time);

    free(A); free(B);
    return 0;
}

