// etape1_cpu : Solveur Jacobi CPU de base : implémentation mono-thread. Utile pour valider la correction et les petites tailles de problème ; met en évidence la limite de calcul CPU.
// Solveur Jacobi sur grille 2D de taille N x N, T itérations.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4096
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

    clock_t start = clock();
    jacobi_cpu(A, B);
    clock_t end = clock();

    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Terminé %s\n", "etape1_cpu");
    printf("CPU time: %.6f seconds\n", cpu_time);

    free(A); free(B);
    return 0;
}
