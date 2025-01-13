#ifndef _GUAC_KALMAN_FILTER_H
#define _GUAC_KALMAN_FILTER_H

#include <cuda_runtime.h>

typedef struct {
    int state_dim;          // Dimension of state vector
    int measurement_dim;    // Dimension of measurement vector
    
    // Device pointers for Kalman filter matrices
    float* d_x;            // State vector
    float* d_P;            // State covariance matrix
    float* d_F;            // State transition matrix
    float* d_Q;            // Process noise covariance
    float* d_H;            // Measurement matrix
    float* d_R;            // Measurement noise covariance
    
    // Temporary device memory for computations
    float* d_temp1;
    float* d_temp2;
    float* d_K;            // Kalman gain
    float* d_S;            // Innovation covariance
    float* d_S_inv;        // Inverse of innovation covariance
    float* d_I;            // Identity matrix
    float* d_measurement;  // Current measurement
    float* d_x_temp;      // Temporary state vector
    float* d_F_transpose; // Transpose of state transition matrix
    float* d_H_transpose; // Transpose of measurement matrix
} KalmanFilter;

// Function declarations
KalmanFilter* init_cuda_kalman_filter(int state_dim, int measurement_dim);
void cuda_kalman_filter_predict(KalmanFilter* kf);
void cuda_kalman_filter_update(KalmanFilter* kf, float* measurement);
void free_cuda_kalman_filter(KalmanFilter* kf);

#endif /* _GUAC_KALMAN_FILTER_H */
