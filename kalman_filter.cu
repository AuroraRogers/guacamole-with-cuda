#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "kalman_filter.h"

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// CUDA kernel for matrix addition
__global__ void matrix_add(float* A, float* B, float* C, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

// CUDA kernel for matrix subtraction
__global__ void matrix_subtract(float* A, float* B, float* C, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        C[row * n + col] = A[row * n + col] - B[row * n + col];
    }
}

// CUDA kernel for matrix transpose
__global__ void matrix_transpose(float* A, float* AT, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        AT[col * m + row] = A[row * n + col];
    }
}

// CUDA kernel for matrix inverse (simplified for 2x2 matrices)
__global__ void matrix_inverse_2x2(float* A, float* Ainv) {
    float det = A[0] * A[3] - A[1] * A[2];
    if (det != 0) {
        float inv_det = 1.0f / det;
        Ainv[0] = A[3] * inv_det;
        Ainv[1] = -A[1] * inv_det;
        Ainv[2] = -A[2] * inv_det;
        Ainv[3] = A[0] * inv_det;
    } else {
        // Handle singular matrix
        Ainv[0] = Ainv[3] = 1.0f;
        Ainv[1] = Ainv[2] = 0.0f;
    }
}

// CUDA Kalman Filter implementation
void cuda_kalman_filter_predict(KalmanFilter* kf) {
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((kf->state_dim + blockDim.x - 1) / blockDim.x, 
                 (kf->state_dim + blockDim.y - 1) / blockDim.y);
    
    // x = F * x
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_F, kf->d_x, kf->d_x_temp, 
                                          kf->state_dim, kf->state_dim, 1);
    cudaMemcpy(kf->d_x, kf->d_x_temp, kf->state_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // P = F * P * F^T + Q
    // First compute F * P
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_F, kf->d_P, kf->d_temp1, 
                                          kf->state_dim, kf->state_dim, kf->state_dim);
    
    // Compute F^T
    matrix_transpose<<<gridDim, blockDim>>>(kf->d_F, kf->d_F_T, 
                                           kf->state_dim, kf->state_dim);
    
    // Compute (F * P) * F^T
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_temp1, kf->d_F_T, kf->d_temp2, 
                                          kf->state_dim, kf->state_dim, kf->state_dim);
    
    // Add Q to get P = F * P * F^T + Q
    matrix_add<<<gridDim, blockDim>>>(kf->d_temp2, kf->d_Q, kf->d_P, 
                                     kf->state_dim, kf->state_dim);
}

void cuda_kalman_filter_update(KalmanFilter* kf, float* measurement) {
    // Copy measurement to device
    cudaMemcpy(kf->d_z, measurement, kf->measurement_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((kf->state_dim + blockDim.x - 1) / blockDim.x, 
                 (kf->state_dim + blockDim.y - 1) / blockDim.y);
    
    // y = z - H * x
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_H, kf->d_x, kf->d_Hx, 
                                          kf->measurement_dim, kf->state_dim, 1);
    matrix_subtract<<<gridDim, blockDim>>>(kf->d_z, kf->d_Hx, kf->d_y, 
                                          kf->measurement_dim, 1);
    
    // S = H * P * H^T + R
    // First compute H * P
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_H, kf->d_P, kf->d_temp1, 
                                          kf->measurement_dim, kf->state_dim, kf->state_dim);
    
    // Compute H^T
    matrix_transpose<<<gridDim, blockDim>>>(kf->d_H, kf->d_H_T, 
                                           kf->measurement_dim, kf->state_dim);
    
    // Compute (H * P) * H^T
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_temp1, kf->d_H_T, kf->d_temp2, 
                                          kf->measurement_dim, kf->state_dim, kf->measurement_dim);
    
    // Add R to get S = H * P * H^T + R
    matrix_add<<<gridDim, blockDim>>>(kf->d_temp2, kf->d_R, kf->d_S, 
                                     kf->measurement_dim, kf->measurement_dim);
    
    // K = P * H^T * S^-1
    // For simplicity, we'll assume measurement_dim = 2 and use a specialized kernel for 2x2 inversion
    if (kf->measurement_dim == 2) {
        matrix_inverse_2x2<<<1, 1>>>(kf->d_S, kf->d_S_inv);
    } else {
        // For other dimensions, we would need a more general matrix inversion algorithm
        // This is a simplified implementation
        cudaMemcpy(kf->d_S_inv, kf->d_S, kf->measurement_dim * kf->measurement_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // Compute P * H^T
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_P, kf->d_H_T, kf->d_temp1, 
                                          kf->state_dim, kf->state_dim, kf->measurement_dim);
    
    // Compute (P * H^T) * S^-1 to get K
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_temp1, kf->d_S_inv, kf->d_K, 
                                          kf->state_dim, kf->measurement_dim, kf->measurement_dim);
    
    // x = x + K * y
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_K, kf->d_y, kf->d_Ky, 
                                          kf->state_dim, kf->measurement_dim, 1);
    matrix_add<<<gridDim, blockDim>>>(kf->d_x, kf->d_Ky, kf->d_x, 
                                     kf->state_dim, 1);
    
    // P = (I - K * H) * P
    // First compute K * H
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_K, kf->d_H, kf->d_KH, 
                                          kf->state_dim, kf->measurement_dim, kf->state_dim);
    
    // Compute I - K * H
    // First set I (identity matrix)
    float* h_I = (float*)malloc(kf->state_dim * kf->state_dim * sizeof(float));
    for (int i = 0; i < kf->state_dim; i++) {
        for (int j = 0; j < kf->state_dim; j++) {
            h_I[i * kf->state_dim + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    cudaMemcpy(kf->d_I, h_I, kf->state_dim * kf->state_dim * sizeof(float), cudaMemcpyHostToDevice);
    free(h_I);
    
    // Compute I - K * H
    matrix_subtract<<<gridDim, blockDim>>>(kf->d_I, kf->d_KH, kf->d_IKH, 
                                          kf->state_dim, kf->state_dim);
    
    // Compute (I - K * H) * P
    matrix_multiply<<<gridDim, blockDim>>>(kf->d_IKH, kf->d_P, kf->d_temp1, 
                                          kf->state_dim, kf->state_dim, kf->state_dim);
    cudaMemcpy(kf->d_P, kf->d_temp1, kf->state_dim * kf->state_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Copy the state back to host for the caller to use
    cudaMemcpy(measurement, kf->d_x, kf->measurement_dim * sizeof(float), cudaMemcpyDeviceToHost);
}

// Initialize Kalman Filter
KalmanFilter* init_cuda_kalman_filter(int state_dim, int measurement_dim) {
    KalmanFilter* kf = (KalmanFilter*)malloc(sizeof(KalmanFilter));
    if (!kf) {
        fprintf(stderr, "Failed to allocate memory for KalmanFilter\n");
        return NULL;
    }
    
    kf->state_dim = state_dim;
    kf->measurement_dim = measurement_dim;
    
    // Allocate device memory for matrices
    cudaMalloc(&kf->d_x, state_dim * sizeof(float));
    cudaMalloc(&kf->d_P, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_F, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_Q, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_H, measurement_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_R, measurement_dim * measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_z, measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_y, measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_S, measurement_dim * measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_K, state_dim * measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_I, state_dim * state_dim * sizeof(float));
    
    // Allocate temporary matrices for calculations
    cudaMalloc(&kf->d_x_temp, state_dim * sizeof(float));
    cudaMalloc(&kf->d_F_T, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_H_T, state_dim * measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_S_inv, measurement_dim * measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_temp1, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_temp2, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_Hx, measurement_dim * sizeof(float));
    cudaMalloc(&kf->d_Ky, state_dim * sizeof(float));
    cudaMalloc(&kf->d_KH, state_dim * state_dim * sizeof(float));
    cudaMalloc(&kf->d_IKH, state_dim * state_dim * sizeof(float));
    
    // Initialize matrices with default values
    float* h_x = (float*)calloc(state_dim, sizeof(float));
    float* h_P = (float*)calloc(state_dim * state_dim, sizeof(float));
    float* h_F = (float*)calloc(state_dim * state_dim, sizeof(float));
    float* h_Q = (float*)calloc(state_dim * state_dim, sizeof(float));
    float* h_H = (float*)calloc(measurement_dim * state_dim, sizeof(float));
    float* h_R = (float*)calloc(measurement_dim * measurement_dim, sizeof(float));
    
    // Initialize state vector to zeros
    for (int i = 0; i < state_dim; i++) {
        h_x[i] = 0.0f;
    }
    
    // Initialize state covariance matrix P to identity
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            h_P[i * state_dim + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Initialize state transition matrix F to identity
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            h_F[i * state_dim + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Initialize process noise covariance matrix Q
    float process_noise = 0.01f;  // Small process noise
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            h_Q[i * state_dim + j] = (i == j) ? process_noise : 0.0f;
        }
    }
    
    // Initialize measurement matrix H
    // For simplicity, we'll assume the measurement directly observes the state
    for (int i = 0; i < measurement_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            h_H[i * state_dim + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Initialize measurement noise covariance matrix R
    float measurement_noise = 0.1f;  // Measurement noise
    for (int i = 0; i < measurement_dim; i++) {
        for (int j = 0; j < measurement_dim; j++) {
            h_R[i * measurement_dim + j] = (i == j) ? measurement_noise : 0.0f;
        }
    }
    
    // Copy initialized matrices to device
    cudaMemcpy(kf->d_x, h_x, state_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kf->d_P, h_P, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kf->d_F, h_F, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kf->d_Q, h_Q, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kf->d_H, h_H, measurement_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kf->d_R, h_R, measurement_dim * measurement_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Free host memory
    free(h_x);
    free(h_P);
    free(h_F);
    free(h_Q);
    free(h_H);
    free(h_R);
    
    return kf;
}

// Free Kalman Filter resources
void free_cuda_kalman_filter(KalmanFilter* kf) {
    if (kf) {
        // Free device memory
        cudaFree(kf->d_x);
        cudaFree(kf->d_P);
        cudaFree(kf->d_F);
        cudaFree(kf->d_Q);
        cudaFree(kf->d_H);
        cudaFree(kf->d_R);
        cudaFree(kf->d_z);
        cudaFree(kf->d_y);
        cudaFree(kf->d_S);
        cudaFree(kf->d_K);
        cudaFree(kf->d_I);
        cudaFree(kf->d_x_temp);
        cudaFree(kf->d_F_T);
        cudaFree(kf->d_H_T);
        cudaFree(kf->d_S_inv);
        cudaFree(kf->d_temp1);
        cudaFree(kf->d_temp2);
        cudaFree(kf->d_Hx);
        cudaFree(kf->d_Ky);
        cudaFree(kf->d_KH);
        cudaFree(kf->d_IKH);
        
        // Free the KalmanFilter struct
        free(kf);
    }
}