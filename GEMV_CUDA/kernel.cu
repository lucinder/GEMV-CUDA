
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

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
    int n = 1024; // Matrix/vector length

    // Allocate memory on host
    float* A = (float*)malloc(n*n*sizeof(float));
    float* x = (float*)malloc(n*sizeof(float));
    float* y = (float*)malloc(n*sizeof(float));

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
    cudaMalloc((void**) & d_A, n*n*sizeof(float));
    cudaMalloc((void**) & d_x, n*sizeof(float));
    cudaMalloc((void**) & d_y, n*sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    gemv_naive <<<numBlocks, blockSize>>> (d_A, d_x, d_y, n);

    // Copy result back to host
    cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    free(A);
    free(x);
    free(y);

    return 0;
}