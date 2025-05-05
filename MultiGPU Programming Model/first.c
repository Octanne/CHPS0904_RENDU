#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    int rank,size;

    /* We initialize MPI */
    MPI_Init(&argc, &argv);
    // We get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // We get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* We get the local rank */
    int local_rank = -1;
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    printf("Rank %d, local rank %d\n", rank, local_rank);
    /* We set the device */
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(local_rank % num_devices);

    /* We get l2_norm and iter_max */
    double l2_norm = 1.0;
    double tol = 1.0e-6;
    int iter = 0;
    int iter_max = 1000;

    /* We create a stream */
    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    /* We create compute_done event */
    cudaEvent_t compute_done;
    cudaEventCreate(&compute_done);

    /* We create a and a_new */
    int nx = 1000;
    int ny = 1000;
    int iy_start = 1;
    int iy_end = ny - 1;
    double *a, *a_new;
    cudaMallocManaged(&a, nx * ny * sizeof(double), cudaMemAttachGlobal);
    cudaMallocManaged(&a_new, nx * ny * sizeof(double), cudaMemAttachGlobal);

    /* We initialize a and a_new */
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            a[iy * nx + ix] = 0.0;
            a_new[iy * nx + ix] = 0.0;
        }
    }

    /* Kernel definition jacobi kernel */
    __global__ void jacobi_kernel(double* a_new, const double* a, double* l2_norm, int iy_start, int iy_end, int nx, cudaStream_t stream) {
        int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
        int ix = blockIdx.x * blockDim.x + threadIdx.x;

        if (iy < iy_end && ix < nx) {
            a_new[iy * nx + ix] = 0.25 * (a[iy * nx + (ix + 1)] + a[iy * nx + (ix - 1)] +
                                          a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);

            double diff = a_new[iy * nx + ix] - a[iy * nx + ix];
            atomicAdd(l2_norm, diff * diff);
        }
    }

    /* Call MPI routines */
    while (l2_norm > tol && iter < iter_max) {
        cudaMemsetAsync(l2_norm, 0, sizeof(double), stream);
        jacobi_kernel(a_new, a, l2_norm, iy_start, iy_end, nx, compute_stream);
        cudaEventRecord(compute_done, compute_stream);
        cudaMemcpyAsync(l2_norm, l2_norm, sizeof(double), cudaMemcpyDeviceToHost, compute_stream);

        cudaEventSynchronize(compute_done);
        const int top = rank > 0 ? rank - 1 : size - 1;
        const int bottom = rank < size - 1 ? rank + 1 : 0;
        // Top/Bottom halo exchange -> TODO

        cudaStreamSynchronize(compute_stream);
        MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, &l2_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    }

    // I am at 12min52s

    /* We finalize MPI */
    MPI_Finalize();
    return 0;
}