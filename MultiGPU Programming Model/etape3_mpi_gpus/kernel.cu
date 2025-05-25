#include <cmath>
#include <stdio.h>

#ifdef USE_DOUBLE
using real = double;
#else
using real = float;
#endif

#define CUDA_CHECK(call)                           \
  do {                                               \
    cudaError_t e = (call);                          \
    if (e != cudaSuccess) {                          \
      fprintf(stderr,                                \
        "CUDA error %s:%d: '%s'\n",               \
         __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                       \
    }                                                 \
  } while (0)

// kernel de Jacobi
__global__ void jacobi_kernel(
    const real* __restrict__ a,
    real* __restrict__ a_new,
    int nx, int ny_local)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (ix < nx-1 && iy < ny_local+1) {
        int idx = iy * nx + ix;
        real up    = a[(iy-1)*nx + ix];
        real down  = a[(iy+1)*nx + ix];
        real left  = a[iy*nx + ix-1];
        real right = a[iy*nx + ix+1];
        a_new[idx] = 0.25f*(up + down + left + right);
    }
}

extern "C" {
void launch_jacobi(real* a, real* a_new, int nx, int ny_local) {
    dim3 block(32, 32);
    dim3 grid((nx + block.x -1)/block.x, ((ny_local+2) + block.y -1)/block.y);
    jacobi_kernel<<<grid, block>>>(a, a_new, nx, ny_local);
    CUDA_CHECK(cudaGetLastError());
}
}