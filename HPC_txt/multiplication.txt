#include <stdio.h>

#define TILE_WIDTH 32

__global__ void matrixMul(float *a, float *b, float *c, int m, int n, int p)
{
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    for (int k = 0; k < n / TILE_WIDTH; k++)
    {
        As[ty][tx] = a[row * n + k * TILE_WIDTH + tx];
        Bs[ty][tx] = b[(k * TILE_WIDTH + ty) * p + col];

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
        {
            Cvalue += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    c[row * p + col] = Cvalue;
}

int main()
{
    int m = 1024;
    int n = 1024;
    int p = 1024;
    size_t bytesA = m * n * sizeof(float);
    size_t bytesB = n * p * sizeof(float);
    size_t bytesC = m * p * sizeof(float);

    // Allocate memory on the host
    float *h_a = (float *)malloc(bytesA);
    float *h_b = (float *)malloc(bytesB);
    float *h_c = (float *)malloc(bytesC);

    // Initialize matrices
    for (int i = 0; i < m * n; i++)
    {
        h_a[i] = 1.0;
    }
    for (int i = 0; i < n * p; i++)
    {
        h_b[i] = 2.0;
    }

    // Allocate memory on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytesA);
    cudaMalloc(&d_b, bytesB);
    cudaMalloc(&d_c, bytesC);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytesB, cudaMemcpyHostToDevice);

    // Launch kernel on the device
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((p + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, p);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, bytesC, cudaMemcpyDeviceToHost);

    // Print 3x3 parts of both matrices
    printf("Matrix A (3x3 part):\n");
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("%.2f ", h_a[i * n + j]);
        }
        printf("\n");
    }
    printf("Size of Matrix A: %dx%d\n", m, n);
    printf("\n");

    printf("Matrix B (3x3 part):\n");
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("%.2f ", h_b[i * p + j]);
        }
        printf("\n");
    }
    printf("Size of Matrix B: %dx%d\n", n, p);
    printf("\n");

    // Print 3x3 part of resultant matrix
    printf("Resultant Matrix (3x3 part):\n");
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("%.2f ", h_c[i * p + j]);
        }
        printf("\n");
    }

    // Print size of resultant matrix
    printf("Size of Resultant Matrix: %dx%d\n", m, p);

    // Free memory on the host and device
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}