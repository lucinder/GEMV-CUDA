
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

// CUDA kernel for performing naive GEMV - every thread computes a row
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

// Part 2 - CUDA kernel for performing GEMV optimized for shared memory
__global__ void gemv_shared(float* A, float* x, float* y, int n) {
    extern __shared__ float sharedTile[];
    int row = blockIdx.x;
    int col = threadIdx.x;
    sharedTile[col] = A[row * n + col];
    __syncthreads();
    y[row] = 0;
    for (int i = 0; i < blockDim.x; i++) {
        y[row] += sharedTile[i] * x[i];
    }
}

// Part 3 - CUDA kernel for performing GEMV optimied for registers
__global__ void gemv_unrolled(float* A, float* x, float* y, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int col = 0; col < N; col += 8) { // Increment by 8 for double unrolling
            sum += A[row * N + col] * x[col];
            sum += A[row * N + col + 1] * x[col + 1];
            sum += A[row * N + col + 2] * x[col + 2];
            sum += A[row * N + col + 3] * x[col + 3];
            // Additional unrolling
            sum += A[row * N + col + 4] * x[col + 4];
            sum += A[row * N + col + 5] * x[col + 5];
            sum += A[row * N + col + 6] * x[col + 6];
            sum += A[row * N + col + 7] * x[col + 7];
        }
        y[row] = sum;
    }
}

void initialize(float* A, float* x, int N) {
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        x[i/N] = static_cast<float>(rand()) / RAND_MAX; // moved this into the loop to shave off a bit of exec time
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    const int blockSizes[] = { 256, 512, 500 };
    std::ofstream outFile("gemv_timing_results.csv"); // File stream for writing results to CSV

    // Check if the file was successfully opened.
    if (!outFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return -1;
    }

    // Write CSV headers
    outFile << "Matrix Size,Block Size,Time (ms)\n";

    for (int N = 10000; N <= 20000; N += 1000) {
        for (int blockSizeIndex = 0; blockSizeIndex < 3; ++blockSizeIndex) {
            int blockSize = blockSizes[blockSizeIndex];
            printf("Testing with %d elements, block size %d\n", N, blockSize);
            float* d_A, * d_x, * d_y;
            cudaMalloc(&d_A, N * N * sizeof(float));
            cudaMalloc(&d_x, N * sizeof(float));
            cudaMalloc(&d_y, N * sizeof(float));

            float* h_A = new float[N * N];
            float* h_x = new float[N];

            initialize(h_A, h_x, N);

            cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

            int numBlocks = (N + blockSize - 1) / blockSize;

            // Start timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            // Uncomment whichever function you want to test!
            // gemv_naive << <numBlocks, blockSize >> > (d_A, d_x, d_y, N);
            gemv_shared << <numBlocks, blockSize >> > (d_A, d_x, d_y, N);
            // gemv_unrolled << <numBlocks, blockSize >> > (d_A, d_x, d_y, N);

            // Stop timing
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);

            // Write results to the CSV file
            outFile << N << "," << blockSize << "," << elapsedTime << "\n";

            // Cleanup
            cudaFree(d_A);
            cudaFree(d_x);
            cudaFree(d_y);
            delete[] h_A;
            delete[] h_x;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    outFile.close(); // Close the CSV file
    return 0;
}