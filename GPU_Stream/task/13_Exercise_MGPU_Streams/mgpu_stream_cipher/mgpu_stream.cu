#include <cstdint>
#include <iostream>
#include "helpers.cuh"
#include "encryption.cuh"

void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {

    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}

__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters) {

    const uint64_t thrdID = blockIdx.x*blockDim.x+threadIdx.x;
    const uint64_t stride = blockDim.x*gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}

bool check_result_cpu(uint64_t * data, uint64_t num_entries,
                      bool parallel=true) {

    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += data[entry] == entry;

    return counter == num_entries;
}

int main (int argc, char * argv[]) {
    const char * encrypted_file = "/dli/task/encrypted";
    Timer timer;
    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters = 1UL << 10;
    const bool openmp = true;

    // Paramétrage : nombre de streams par GPU (ajustez pour optimiser)
    const uint64_t num_streams = 8;

    // Obtenir le nombre de GPUs disponibles
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) {
        std::cerr << "Aucun GPU détecté !" << std::endl;
        return 1;
    }

    // Définir la taille des chunks pour chaque stream et chaque GPU
    const uint64_t stream_chunk_size = sdiv(sdiv(num_entries, num_gpus), num_streams);
    const uint64_t gpu_chunk_size = stream_chunk_size * num_streams;

    uint64_t * data_cpu;
    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
    check_last_error();

    if (!encrypted_file_exists(encrypted_file)) {
        encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
        write_encrypted_to_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    } else {
        read_encrypted_from_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    }

    // Allocation mémoire sur chaque GPU
    uint64_t * data_gpu[16]; // 16 = max GPUs supportés ici, ajustez si besoin
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);
        const uint64_t lower = gpu_chunk_size * gpu;
        const uint64_t upper = std::min(lower + gpu_chunk_size, num_entries);
        const uint64_t width = upper - lower;
        cudaMalloc(&data_gpu[gpu], sizeof(uint64_t)*width);
        check_last_error();
    }

    // Création des streams pour chaque GPU
    cudaStream_t streams[16][num_streams];
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);
        for (uint64_t s = 0; s < num_streams; ++s)
            cudaStreamCreate(&streams[gpu][s]);
    }

    timer.start();
    // Pour chaque GPU et chaque stream, overlap copy/compute
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);
        for (uint64_t s = 0; s < num_streams; ++s) {
            const uint64_t stream_offset = stream_chunk_size * s;
            const uint64_t lower = gpu_chunk_size * gpu + stream_offset;
            const uint64_t upper = std::min(lower + stream_chunk_size, num_entries);
            const uint64_t width = upper - lower;
            // Hôte -> Device
            cudaMemcpyAsync(data_gpu[gpu] + stream_offset, data_cpu + lower, sizeof(uint64_t) * width, cudaMemcpyHostToDevice, streams[gpu][s]);
            // Kernel
            decrypt_gpu<<<80*32, 64, 0, streams[gpu][s]>>>(data_gpu[gpu] + stream_offset, width, num_iters);
            // Device -> Hôte
            cudaMemcpyAsync(data_cpu + lower, data_gpu[gpu] + stream_offset, sizeof(uint64_t) * width, cudaMemcpyDeviceToHost, streams[gpu][s]);
        }
    }
    // Synchronisation de tous les streams de tous les GPUs
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);
        for (uint64_t s = 0; s < num_streams; ++s)
            cudaStreamSynchronize(streams[gpu][s]);
    }
    timer.stop("total time on GPUs (multi-GPU overlap streams)");
    check_last_error();

    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;

    cudaFreeHost(data_cpu);
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);
        cudaFree(data_gpu[gpu]);
        for (uint64_t s = 0; s < num_streams; ++s)
            cudaStreamDestroy(streams[gpu][s]);
    }
    check_last_error();
}