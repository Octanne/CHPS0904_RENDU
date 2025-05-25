#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

using real = float;
const real tol = 1e-8;

extern "C" void launch_jacobi(float* a, float* a_new, int nx, int ny_local);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 5) {
        if (!rank) std::cerr << "Usage: "<< argv[0]<<" <iter_max> <nx> <ny> <nccheck>\n";
        MPI_Finalize(); return 1;
    }
    int iter_max = atoi(argv[1]);
    int nx       = atoi(argv[2]);
    int ny       = atoi(argv[3]);
    int nccheck  = atoi(argv[4]);

    // distribution des lignes (excluant bord haut/bas)
    int rows = ny - 2;
    int base = rows / size;
    int rem  = rows % size;
    int ny_local = base + (rank < rem);
    int offset   = rank*base + std::min(rank, rem) + 1;

    // assignation GPU
    int devCount; cudaGetDeviceCount(&devCount);
    cudaSetDevice(rank % devCount);

    // buffers sur GPU (2 halos)
    real* d_a;
    real* d_a_new;
    size_t bytes = nx * (ny_local+2) * sizeof(real);
    cudaMalloc(&d_a,     bytes);
    cudaMalloc(&d_a_new, bytes);
    cudaMemset(d_a,     0, bytes);
    cudaMemset(d_a_new, 0, bytes);

    // init conditions de Dirichlet (bord gauche/droit)
    std::vector<real> h_border((ny_local+2)*nx);
    const real PI = 3.141592653589793;
    for (int i=0; i<ny_local+2; ++i) {
        real y = sin(2*PI*(offset + i -1)/(ny-1));
        h_border[i*nx + 0]       = y;
        h_border[i*nx + (nx-1)]  = y;
    }
    cudaMemcpy(d_a,     h_border.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_new, h_border.data(), bytes, cudaMemcpyHostToDevice);

    int top = (rank==0) ? MPI_PROC_NULL : rank-1;
    int bot = (rank==size-1) ? MPI_PROC_NULL : rank+1;

    real l2 = 1;
    int iter=0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // buffers d’échange halos sur CPU
    std::vector<real> h_halo(nx);

    while (l2>tol && iter<iter_max) {
        // copie halo haut
        cudaMemcpy(h_halo.data(), d_a_new + nx, nx*sizeof(real), cudaMemcpyDeviceToHost);
        MPI_Sendrecv(h_halo.data(), nx, MPI_FLOAT, top, 0,
                     h_halo.data(), nx, MPI_FLOAT, bot, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaMemcpy(d_a_new + (ny_local+1)*nx, h_halo.data(), nx*sizeof(real), cudaMemcpyHostToDevice);

        // copie halo bas
        cudaMemcpy(h_halo.data(), d_a_new + ny_local*nx, nx*sizeof(real), cudaMemcpyDeviceToHost);
        MPI_Sendrecv(h_halo.data(), nx, MPI_FLOAT, bot, 1,
                     h_halo.data(), nx, MPI_FLOAT, top, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaMemcpy(d_a_new, h_halo.data(), nx*sizeof(real), cudaMemcpyHostToDevice);

        // lancement CUDA
        launch_jacobi(d_a, d_a_new, nx, ny_local);

        // calcul norme tous les nccheck iters
        if (iter % nccheck == 0) {
            // on récupère localement sur device
            size_t tot = nx*(ny_local+2);
            std::vector<real> h_all(tot);
            cudaMemcpy(h_all.data(), d_a_new, tot*sizeof(real), cudaMemcpyDeviceToHost);
            real local_sum=0;
            for (int i=1;i<=ny_local;i++) for(int j=1;j<nx-1;j++) {
                real diff = h_all[i*nx+j] - h_all[(i-1)*nx+j];
                local_sum += diff*diff;
            }
            MPI_Allreduce(&local_sum, &l2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            l2 = sqrt(l2);
        }

        std::swap(d_a, d_a_new);
        iter++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (!rank)
        std::cout<<"Done: "<<iter<<" iters in "<<(t1-t0)<<"s, norm="<<l2<<"\n";

    cudaFree(d_a);
    cudaFree(d_a_new);
    MPI_Finalize();
    return 0;
}