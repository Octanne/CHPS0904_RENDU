// etape4_mpi_overlap : Résolution du système d'équations par la méthode de Jacobi sur plusieurs GPU avec MPI et chevauchement des flux.
// Solveur Jacobi sur grille 2D de taille N x N, T itérations.
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define N 4096
#define T 1000

__global__ void jacobi_kernel(double* A, double* B, int N_SIZE, int i_start, int i_end);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaSetDevice(rank);

    int rows_per_rank = N / size;
    int i_start = rank * rows_per_rank + 1;
    int i_end = (rank == size-1) ? N-1 : (rank+1)*rows_per_rank;
    int local_rows = i_end - i_start;

    // +2 for halos (top/bottom)
    size_t local_size = (local_rows+2) * N * sizeof(double);
    double *h_A = (double*)calloc((local_rows+2)*N, sizeof(double));
    double *h_B = (double*)calloc((local_rows+2)*N, sizeof(double));
    double *d_A, *d_B;
    cudaMalloc(&d_A, local_size);
    cudaMalloc(&d_B, local_size);

    cudaStream_t stream_compute, stream_top, stream_bottom;
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_top);
    cudaStreamCreate(&stream_bottom);

    // Initialisation (optionnel : remplir h_A/h_B)
    cudaMemcpyAsync(d_A, h_A, local_size, cudaMemcpyHostToDevice, stream_compute);
    cudaMemcpyAsync(d_B, h_B, local_size, cudaMemcpyHostToDevice, stream_compute);

    MPI_Request reqs[4];

    dim3 block(16, 16);
    dim3 grid((N-2+block.x-1)/block.x, (local_rows+block.y-1)/block.y);

    // Synchronisation avant le chronométrage
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (rank == 0) {
        printf("Début de l'itération Jacobi multi-GPU avec chevauchement des flux...\n");
    }

    for (int t = 0; t < T; t++) {
        // 1. Échanges d’halos : device->host (asynchrone)
        // Haut
        if (rank > 0) {
            cudaMemcpyAsync(h_A, d_A + 1*N, N*sizeof(double), cudaMemcpyDeviceToHost, stream_top);
        }
        // Bas
        if (rank < size-1) {
            cudaMemcpyAsync(h_A + (local_rows+1)*N, d_A + local_rows*N, N*sizeof(double), cudaMemcpyDeviceToHost, stream_bottom);
        }

        // 2. Lancer le calcul intérieur (hors bords) sur stream_compute
        jacobi_kernel<<<grid, block, 0, stream_compute>>>(d_A, d_B, N, 1, local_rows+1);

        // 3. Attendre la fin des copies device->host avant MPI
        if (rank > 0) cudaStreamSynchronize(stream_top);
        if (rank < size-1) cudaStreamSynchronize(stream_bottom);

        // 4. MPI non bloquant pour halos
        // Haut
        if (rank > 0) {
            MPI_Isend(h_A + 1*N, N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(h_A, N, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &reqs[1]);
        }
        // Bas
        if (rank < size-1) {
            MPI_Isend(h_A + local_rows*N, N, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(h_A + (local_rows+1)*N, N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[3]);
        }

        // 5. Attendre la fin des MPI
        if (rank > 0) { MPI_Wait(&reqs[0], MPI_STATUS_IGNORE); MPI_Wait(&reqs[1], MPI_STATUS_IGNORE); }
        if (rank < size-1) { MPI_Wait(&reqs[2], MPI_STATUS_IGNORE); MPI_Wait(&reqs[3], MPI_STATUS_IGNORE); }

        // 6. Copier les halos reçus host->device (asynchrone)
        if (rank > 0) {
            cudaMemcpyAsync(d_A, h_A, N*sizeof(double), cudaMemcpyHostToDevice, stream_top);
        }
        if (rank < size-1) {
            cudaMemcpyAsync(d_A + (local_rows+1)*N, h_A + (local_rows+1)*N, N*sizeof(double), cudaMemcpyHostToDevice, stream_bottom);
        }

        // 7. Synchroniser tous les streams avant itération suivante
        cudaStreamSynchronize(stream_compute);
        if (rank > 0) cudaStreamSynchronize(stream_top);
        if (rank < size-1) cudaStreamSynchronize(stream_bottom);

        // 8. Swap pointeurs
        double* tmp = d_A; d_A = d_B; d_B = tmp;
        double* tmp_h = h_A; h_A = h_B; h_B = tmp_h;
    }

    printf("Rank %d terminé\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Optionnel : rassembler les résultats (MPI_Gather) ou écrire localement
    cudaMemcpy(h_A, d_A, local_size, cudaMemcpyDeviceToHost);

    if (rank == 0) {
        printf("Temps total (Jacobi multi-GPU overlap, %d rangs): %.6f secondes\n", size, t1-t0);
    }

    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_top);
    cudaStreamDestroy(stream_bottom);

    MPI_Finalize();
    return 0;
}

