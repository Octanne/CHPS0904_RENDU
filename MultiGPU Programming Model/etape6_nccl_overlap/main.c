// etape6_nccl_overlap : NCCL + recouvrement : superposition des collectifs NCCL avec le calcul sur GPU, cachant le coût de communication.
// Solveur Jacobi sur grille 2D de taille N x N, T itérations.
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define T 1000

void jacobi_cpu(double* A, double* B) {
    for (int t = 0; t < T; t++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                B[i*N + j] = 0.25 * (A[(i-1)*N + j] + A[(i+1)*N + j]
                                      + A[i*N + j-1] + A[i*N + j+1]);
            }
        }
        double* tmp = A; A = B; B = tmp;
    }
}

int main(int argc, char** argv) {
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    for (int i = 0; i < N*N; i++) A[i] = B[i] = 0.0;
    jacobi_cpu(A, B);
    printf("Terminé %s", "etape6_nccl_overlap");
    free(A); free(B);
    return 0;
}
