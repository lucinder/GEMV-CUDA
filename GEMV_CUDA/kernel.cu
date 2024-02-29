
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <fstream>
#include <time.h>
#include <chrono>

// CUDA kernel for performing naive GEMV: y = Ax + y
__global__ void gemv_naive(float* A, float* x, float* y, int n) {
    int row = threadIdx.x;
    if (row < n) {
        float dotProduct = 0.0;
        for (int col = 0; col < n; ++col) {
            dotProduct += A[row * n + col] * x[col];
        }
        y[row] = dotProduct;
    }
}

int main() {
    std::ofstream file("data.csv");
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }
    file << "Size, Blocks, Threads, TimeWithMem, TimeWithoutMem, " << std::endl;

    const int blockSizes[] = { 256,500,512,540,544,608,700,800,1000,1024};

    for (int n = 10000; n <= 20000; n += 1000) {
        for (int i = 0; i < 10; ++i) {
            int blockSize = blockSizes[i];
            int numBlocks = (n + blockSize - 1) / blockSize;

            std::cout << "Size: " << n << std::endl;
            std::cout << "Blocks: " << numBlocks << std::endl;
            std::cout << "Threads: " << blockSize << std::endl;
            file << n << ", ";
            file << numBlocks << ", ";
            file << blockSize << ", ";

            auto start = std::chrono::high_resolution_clock::now();
            // Allocate memory on host
            float* A = (float*)malloc(n * n * sizeof(float));
            float* x = (float*)malloc(n * sizeof(float));
            float* y = (float*)malloc(n * sizeof(float));

            // Initialize matrix A and vector x on host
            for (int i = 0; i < n * n; i++) {
                A[i] = 1.0; // Example initialization
            }
            for (int i = 0; i < n; i++) {
                x[i] = 1.0; // Example initialization
            }

            // Allocate memory on device
            float* d_A = 0;
            float* d_x = 0;
            float* d_y = 0;
            cudaMalloc((void**)&d_A, n * n * sizeof(float));
            cudaMalloc((void**)&d_x, n * sizeof(float));
            cudaMalloc((void**)&d_y, n * sizeof(float));

            // Copy data from host to device
            cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

            auto start2 = std::chrono::high_resolution_clock::now();
            // Launch the kernel

            gemv_naive << <numBlocks, blockSize >> > (d_A, d_x, d_y, n);

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start2);
            double time_used_without_mem = duration.count();
            time_used_without_mem /= 1000000000;


            // Copy result back to host
            cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            double time_used_with_mem = duration.count();
            time_used_with_mem /= 1000000000;

            file << time_used_with_mem << ", ";
            file << time_used_without_mem << ", ";
                            file << std::endl;
            printf("addition with mem: %.7f seconds\n", time_used_with_mem);
            printf("addition without mem: %.7f seconds\n", time_used_without_mem);

            // Free device memory
            cudaFree(d_A);
            cudaFree(d_x);
            cudaFree(d_y);

            // Free host memory
            free(A);
            free(x);
            free(y);
        }
    }
    file.close();
    return 0;
}