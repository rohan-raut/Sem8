#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int n = 1000000;
    size_t bytes = n * sizeof(float);

    // Allocate memory on the host
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Initialize the vectors
    for (int i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = i + 1;
    }

    // Allocate memory on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel on the device
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print first 10 elements of both vectors
    printf("First 10 elements of vector a:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.2f ", h_a[i]);
    }
    printf("\n");
    printf("Size of vector a: %d\n", n);
    printf("\n");

    printf("First 10 elements of vector b:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.2f ", h_b[i]);
    }
    printf("\n");
    printf("Size of vector b: %d\n", n);
    printf("\n");

    // Print first 10 elements of resultant vector
    printf("First 10 elements of resultant vector:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.2f ", h_c[i]);
    }
    printf("\n");

    // Print size of resultant vector
    printf("Size of resultant vector: %d\n", n);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
