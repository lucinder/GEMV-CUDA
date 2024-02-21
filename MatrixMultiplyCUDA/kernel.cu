
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void mmKernel(float *c, const float *a, const float *b)
{
    // perform matrix multiply here!
}

int main()
{
    const int n = 5;
    float* A = (float*)malloc(n*n*sizeof(float));
    float* B = (float*)malloc(n * n * sizeof(float));
    float* C = (float*)malloc(n * n * sizeof(float));
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    cudaMalloc((void**)&dev_a, n * n * sizeof(float));
    cudaMalloc((void**)&dev_b, n * n * sizeof(float));
    cudaMalloc((void**)&dev_c, n * n * sizeof(float));

    cudaMemcpy(dev_a, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    mmKernel <<<n, n>>> (dev_c, dev_a, dev_b); // n blocks and n threads per block
    cudaDeviceSynchronize();

    cudaMemcpy(C, dev_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}