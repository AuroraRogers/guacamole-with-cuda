#include <cuda_runtime.h>
#include <stdio.h>
#include "kalman_filter.h"

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CUDA kernel for matrix addition
__global__ void matrixAdd(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA Kalman Filter implementation
void cuda_kalman_filter_predict(KalmanFilter* kf) {
    int n = kf->state_dim;
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    // x = F * x
    matrixMultiply<<<grid, block>>>(kf->d_F, kf->d_x, kf->d_x_temp, n);
    cudaMemcpy(kf->d_x, kf->d_x_temp, n * sizeof(float), cudaMemcpyDeviceToDevice);

    // P = F * P * F^T + Q
    matrixMultiply<<<grid, block>>>(kf->d_F, kf->d_P, kf->d_temp1, n);
    matrixMultiply<<<grid, block>>>(kf->d_temp1, kf->d_F_transpose, kf->d_temp2, n);
    matrixAdd<<<(n * n + 255) / 256, 256>>>(kf->d_temp2, kf->d_Q, kf->d_P, n * n);
}

void cuda_kalman_filter_update(KalmanFilter* kf, float* measurement) {
    int n = kf->state_dim;
    int m = kf->measurement_dim;
    
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    // Copy measurement to device
    cudaMemcpy(kf->d_measurement, measurement, m * sizeof(float), cudaMemcpyHostToDevice);

    // S = H * P * H^T + R
    matrixMultiply<<<grid, block>>>(kf->d_H, kf->d_P, kf->d_temp1, n);
    matrixMultiply<<<grid, block>>>(kf->d_temp1, kf->d_H_transpose, kf->d_S, n);
    matrixAdd<<<(n * n + 255) / 256, 256>>>(kf->d_S, kf->d_R, kf->d_S, n * n);

    // K = P * H^T * S^(-1)
    // Note: For simplicity, we're not implementing matrix inversion on GPU
    // In practice, you might want to use cuBLAS or cuSOLVER for this
    float* h_S = (float*)malloc(n * n * sizeof(float));
    cudaMemcpy(h_S, kf->d_S, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    // Implement matrix inversion here for h_S
    cudaMemcpy(kf->d_S_inv, h_S, n * n * sizeof(float), cudaMemcpyHostToDevice);
    free(h_S);

    matrixMultiply<<<grid, block>>>(kf->d_P, kf->d_H_transpose, kf->d_temp1, n);
    matrixMultiply<<<grid, block>>>(kf->d_temp1, kf->d_S_inv, kf->d_K, n);

    // x = x + K * (z - H * x)
    matrixMultiply<<<grid, block>>>(kf->d_H, kf->d_x, kf->d_temp1, n);
    // Implement vector subtraction for (z - H * x)
    matrixMultiply<<<grid, block>>>(kf->d_K, kf->d_temp1, kf->d_temp2, n);
    matrixAdd<<<(n * n + 255) / 256, 256>>>(kf->d_x, kf->d_temp2, kf->d_x, n * n);

    // P = (I - K * H) * P
    matrixMultiply<<<grid, block>>>(kf->d_K, kf->d_H, kf->d_temp1, n);
    matrixAdd<<<(n * n + 255) / 256, 256>>>(kf->d_I, kf->d_temp1, kf->d_temp2, n * n);
    matrixMultiply<<<grid, block>>>(kf->d_temp2, kf->d_P, kf->d_P, n);
}

// Initialize Kalman Filter
KalmanFilter* init_cuda_kalman_filter(int state_dim, int measurement_dim) {
    KalmanFilter* kf = (KalmanFilter*)malloc(sizeof(KalmanFilter));
    kf->state_dim = state_dim;
    kf->measurement_dim = measurement_dim;

    // Allocate device memory
    cudaMalloc(&kf->d_x, state_dim * sizeof(float));
    cudaMalloc(&kf->d_P, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_F, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_Q, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_H, measurement_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_R, measurement_dim * measurement_dim * sizeof(float));
    
    // Allocate temporary buffers
    cudaMalloc(&kf->d_temp1, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_temp2, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_K, state_dim * measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_S, measurement_dim * measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_S_inv, measurement_dim * measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_measurement, measurement_dim * sizeof(float));
    
    // Initialize identity matrix
    float* h_I = (float*)malloc(state_dim * state_dim * sizeof(float));
    for(int i = 0; i < state_dim; i++) {
        for(int j = 0; j < state_dim; j++) {
            h_I[i * state_dim + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    cudaMalloc(&kf->d_I, state_dim * state_dim * sizeof(float));
    cudaMemcpy(kf->d_I, h_I, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice);
    free(h_I);

    return kf;
}

// Free Kalman Filter resources
void free_cuda_kalman_filter(KalmanFilter* kf) {
    cudaFree(kf->d_x);
    cudaFree(kf->d_P);
    cudaFree(kf->d_F);
    cudaFree(kf->d_Q);
    cudaFree(kf->d_H);
    cudaFree(kf->d_R);
    cudaFree(kf->d_temp1);
    cudaFree(kf->d_temp2);
    cudaFree(kf->d_K);
    cudaFree(kf->d_S);
    cudaFree(kf->d_S_inv);
    cudaFree(kf->d_I);
    cudaFree(kf->d_measurement);
    free(kf);
}
