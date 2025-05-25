// etape3_mpi_gpus : MPI + multi-GPU : résolution du problème de Jacobi avec plusieurs GPU sur un cluster.
// Solveur Jacobi sur grille 2D de taille N x N, T itérations.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <mpi.h>

#define N_GRILLE 4096
#define T 1000

__global__ void jacobi_kernel(double* A, double* B, int N_SIZE);

// Version MPI + multi-GPU
void jacobi_mpi_gpu(double* h_A, double* h_B, int N, int local_rows, int rank, int size, MPI_Comm comm) {
    double *d_A, *d_B;
    size_t local_size = (local_rows + 2) * N * sizeof(double); // +2 pour halos haut/bas

    cudaSetDevice(rank % 8); // suppose max 8 GPUs par nœud

    cudaMalloc(&d_A, local_size);
    cudaMalloc(&d_B, local_size);

    // Copier la sous-grille locale (hors halos)
    cudaMemcpy(d_A + N, h_A + N, local_rows * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B + N, h_B + N, local_rows * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N-2+block.x-1)/block.x, (local_rows+block.y-1)/block.y);

    // Buffers pour halos sur le host
    double* halo_send_top = (double*)malloc(N * sizeof(double));
    double* halo_recv_top = (double*)malloc(N * sizeof(double));
    double* halo_send_bot = (double*)malloc(N * sizeof(double));
    double* halo_recv_bot = (double*)malloc(N * sizeof(double));

    for (int t = 0; t < T; t++) {
        // Copier les halos du device vers le host
        if (rank > 0) {
            cudaMemcpy(halo_send_top, d_A + N, N * sizeof(double), cudaMemcpyDeviceToHost);
        }
        if (rank < size-1) {
            cudaMemcpy(halo_send_bot, d_A + local_rows*N, N * sizeof(double), cudaMemcpyDeviceToHost);
        }

        // Échanges d’halos haut/bas
        // Envoyer/recevoir ligne du haut
        if (rank > 0) {
            MPI_Sendrecv(
                halo_send_top, N, MPI_DOUBLE, rank-1, 0,
                halo_recv_top, N, MPI_DOUBLE, rank-1, 1,
                comm, MPI_STATUS_IGNORE
            );
            // Copier la ligne reçue dans le halo du haut sur le device
            cudaMemcpy(d_A, halo_recv_top, N * sizeof(double), cudaMemcpyHostToDevice);
        }
        // Envoyer/recevoir ligne du bas
        if (rank < size-1) {
            MPI_Sendrecv(
                halo_send_bot, N, MPI_DOUBLE, rank+1, 1,
                halo_recv_bot, N, MPI_DOUBLE, rank+1, 0,
                comm, MPI_STATUS_IGNORE
            );
            // Copier la ligne reçue dans le halo du bas sur le device
            cudaMemcpy(d_A + (local_rows+1)*N, halo_recv_bot, N * sizeof(double), cudaMemcpyHostToDevice);
        }

        jacobi_kernel<<<grid, block>>>(d_A, d_B, N);
        cudaDeviceSynchronize();
        double* tmp = d_A; d_A = d_B; d_B = tmp;
    }

    free(halo_send_top);
    free(halo_recv_top);
    free(halo_send_bot);
    free(halo_recv_bot);

    // Copier la sous-grille locale (hors halos) vers le host
    cudaMemcpy(h_A + N, d_A + N, local_rows * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (N_GRILLE < 3 || T < 1) {
        if (rank == 0) {
            fprintf(stderr, "N_GRILLE must be >= 3 and T must be >= 1\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // We set the message only on rank 0
    if (rank == 0) {
        printf("Lancement de l'étape 3 : MPI + multi-GPU\n");
        printf("Grille de taille %d x %d, T = %d itérations\n", N_GRILLE, N_GRILLE, T);
    }

    int local_rows = (N_GRILLE-2) / size;
    int rem = (N_GRILLE-2) % size;
    if (rank < rem) local_rows++;

    // Allouer la sous-grille locale (+2 lignes pour halos)
    double *A = (double*)calloc((local_rows+2)*N_GRILLE, sizeof(double));
    double *B = (double*)calloc((local_rows+2)*N_GRILLE, sizeof(double));

    clock_t start = clock();
    jacobi_mpi_gpu(A, B, N_GRILLE, local_rows, rank, size, comm);
    clock_t end = clock();

    double local_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Terminé %s rank %d/%d\n", "etape3_mpi_gpus (MPI + multi-GPU)", rank, size);
        printf("Max GPU time: %.6f seconds rank %d/%d\n", max_time, rank, size);
    }

    free(A); free(B);
    MPI_Finalize();
    return 0;
}

