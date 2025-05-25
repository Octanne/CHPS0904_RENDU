__global__ void jacobi_kernel(double* A, double* B, int N_SIZE, int i_start, int i_end) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + i_start;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < i_end && j < N_SIZE-1) {
        B[i*N_SIZE + j] = 0.25 * (A[(i-1)*N_SIZE + j] + A[(i+1)*N_SIZE + j]
                            + A[i*N_SIZE + j-1] + A[i*N_SIZE + j+1]);
    }
}
