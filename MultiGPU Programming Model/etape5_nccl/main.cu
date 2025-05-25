// etape5_nccl : 2 GPU + NCCL : échange des halos entre deux GPUs.
// Solveur Jacobi sur grille 2D de taille N x N, T itérations.
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define N 4096
#define T 1000

__global__ void jacobi_kernel(double* A, double* B, int N_SIZE);

void check_nccl(ncclResult_t res, const char* msg) {
    if (res != ncclSuccess) {
        printf("NCCL error: %s (%s)\n", msg, ncclGetErrorString(res));
        exit(1);
    }
}

int main(int argc, char** argv) {
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices < 2) {
        printf("Ce programme nécessite au moins 2 GPU.\n");
        return 1;
    }
    printf("Nombre de GPU disponibles : %d\n", num_devices);

    // On force l'utilisation de 2 GPU (0 et 1)
    num_devices = 2;

    double *d_A[2], *d_B[2];
    double *h_A[2], *h_B[2];
    size_t size = N * N * sizeof(double);

    // Allocation CPU/GPU pour 2 GPUs
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&d_A[i], size);
        cudaMalloc(&d_B[i], size);
        h_A[i] = (double*)malloc(size);
        h_B[i] = (double*)malloc(size);
        for (int j = 0; j < N*N; ++j) h_A[i][j] = h_B[i][j] = 0.0;
        cudaMemcpy(d_A[i], h_A[i], size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B[i], h_B[i], size, cudaMemcpyHostToDevice);
    }

    // Initialisation NCCL pour 2 GPU (0 et 1)
    ncclComm_t comms[2];
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclCommInitAll(comms, 2, (const int[]){0,1});

    dim3 block(16, 16);
    dim3 grid((N-2+block.x-1)/block.x, (N-2+block.y-1)/block.y);

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < num_devices; ++i) {
            cudaSetDevice(i);
            jacobi_kernel<<<grid, block>>>(d_A[i], d_B[i], N);
        }
        // Synchronisation
        for (int i = 0; i < num_devices; ++i) cudaSetDevice(i), cudaDeviceSynchronize();

        // Échange de la ligne d'halo entre GPU 0 et 1 (exemple : dernière ligne de 0 <-> première ligne de 1)
        // Supposons découpage horizontal : GPU0 = lignes 0..N/2-1, GPU1 = lignes N/2..N-1
        check_nccl(ncclSend(d_B[0]+(N/2-1)*N, N, ncclDouble, 1, comms[0], 0), "send halo 0->1");
        check_nccl(ncclRecv(d_B[1], N, ncclDouble, 0, comms[1], 0), "recv halo 0->1");
        check_nccl(ncclSend(d_B[1]+N, N, ncclDouble, 0, comms[1], 0), "send halo 1->0");
        check_nccl(ncclRecv(d_B[0]+N/2*N, N, ncclDouble, 1, comms[0], 0), "recv halo 1->0");

        // Swap buffers
        for (int i = 0; i < num_devices; ++i) {
            double* tmp = d_A[i]; d_A[i] = d_B[i]; d_B[i] = tmp;
        }
    }

    // Récupération des résultats
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(i);
        cudaMemcpy(h_A[i], d_A[i], size, cudaMemcpyDeviceToHost);
    }

    printf("Terminé %s\n", "etape5_nccl (2 GPU + NCCL)");
    for (int i = 0; i < num_devices; ++i) {
        cudaFree(d_A[i]); cudaFree(d_B[i]);
        free(h_A[i]); free(h_B[i]);
        ncclCommDestroy(comms[i]);
    }
    return 0;
}

