#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

using real = float;
const real tol = 1e-8;
extern "C" void launch_jacobi(real* a, real* a_new, int nx, int ny_local);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (argc<5) {
        if (!rank) std::cerr<<"Usage: "<<argv[0]<<" <iter_max> <nx> <ny> <nccheck>\n";
        MPI_Finalize(); return 1;
    }
    int iter_max = atoi(argv[1]);
    int nx       = atoi(argv[2]);
    int ny       = atoi(argv[3]);
    int nccheck  = atoi(argv[4]);

    // Sécurité : nx et ny doivent être >= 2
    if (nx < 2 || ny < 2) {
        if (!rank) std::cerr << "Erreur: nx et ny doivent être >= 2\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // partition des lignes (excluant bords)
    int rows = ny-2;
    int base = rows/size;
    int rem  = rows%size;
    int ny_local = base + (rank<rem);
    int offset   = rank*base + std::min(rank,rem) + 1;

    // Debug
    if (!rank) std::cout << "nx=" << nx << " ny=" << ny << " size=" << size << std::endl;
    std::cout << "Rang " << rank << ": ny_local=" << ny_local << " offset=" << offset << std::endl;

    // GPU select
    int devCount = 0;
    cudaError_t err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess || devCount < 1) {
        std::cerr << "Erreur CUDA: aucun GPU détecté (rank " << rank << ")\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaSetDevice(rank%devCount);

    // allocations device (incluant halos)
    real *d_a = nullptr, *d_a_new = nullptr;
    size_t bytes = nx*(ny_local+2)*sizeof(real);
    if (cudaMalloc(&d_a, bytes) != cudaSuccess ||
        cudaMalloc(&d_a_new, bytes) != cudaSuccess) {
        std::cerr << "Erreur cudaMalloc (rank " << rank << ")\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    cudaMemset(d_a,     0, bytes);
    cudaMemset(d_a_new, 0, bytes);

    // init bordures gauche/droite
    std::vector<real> h_border((ny_local+2)*nx);
    const real PI = 3.141592653589793f;
    for(int i=0;i<ny_local+2;i++){
        real y = sinf(2*PI*(offset+i-1)/(ny-1));
        h_border[i*nx+0]      = y;
        h_border[i*nx+nx-1]   = y;
    }
    if (cudaMemcpy(d_a,     h_border.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_a_new, h_border.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Erreur cudaMemcpy (rank " << rank << ")\n";
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    // halos device
    real *top_send = d_a_new + nx;
    real *bot_send = d_a_new + ny_local*nx;
    real *top_recv = d_a_new + (ny_local+1)*nx;
    real *bot_recv = d_a_new;

    MPI_Request reqs[4];
    MPI_Status  stats[4];
    int top = (rank==0? MPI_PROC_NULL: rank-1);
    int bot = (rank==size-1? MPI_PROC_NULL: rank+1);

    real l2=1.0f; int iter=0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    while(l2>tol && iter<iter_max){
        // reset norme
        real local_sum=0;

        // échange non bloquant Device<->Device grâce CUDA-aware MPI
        MPI_Irecv(top_recv, nx, MPI_FLOAT, top, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(bot_recv, nx, MPI_FLOAT, bot, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(top_send, nx, MPI_FLOAT, top, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(bot_send, nx, MPI_FLOAT, bot, 0, MPI_COMM_WORLD, &reqs[3]);

        // calcul Jacobi intérieur (sans halos)
        launch_jacobi(d_a, d_a_new, nx, ny_local);

        // attends fin comm avant d’utiliser nouvelles lignes
        MPI_Waitall(4, reqs, stats);

        // calcul norme si requis
        if(iter % nccheck ==0){
            // récupérer sur hôtes uniquement l'intérieur
            size_t interior = nx*ny_local;
            std::vector<real> h_chunk(interior);
            cudaMemcpy(h_chunk.data(), d_a_new+nx, interior*sizeof(real), cudaMemcpyDeviceToHost);
            for(int i=1;i<ny_local-1;i++){
                for(int j=1;j<nx-1;j++){
                    real diff = h_chunk[i*nx+j] - h_chunk[(i-1)*nx+j];
                    local_sum += diff*diff;
                }
            }
            MPI_Allreduce(&local_sum, &l2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            l2 = sqrtf(l2);
        }

        std::swap(d_a, d_a_new);
        iter++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if(!rank) std::cout<<"Overlap: "<<iter<<" iters en "<<(t1-t0)<<" s, norm="<<l2<<"\n";

    cudaFree(d_a);
    cudaFree(d_a_new);
    MPI_Finalize();
    return 0;
}