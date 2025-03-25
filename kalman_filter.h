#ifndef _GUAC_KALMAN_FILTER_H
#define _GUAC_KALMAN_FILTER_H

// Kalman Filter structure
typedef struct {
    int state_dim;          // Dimension of state vector
    int measurement_dim;    // Dimension of measurement vector
    
    // Device pointers for Kalman filter matrices
    float* d_x;             // State vector
    float* d_P;             // State covariance matrix
    float* d_F;             // State transition matrix
    float* d_Q;             // Process noise covariance matrix
    float* d_H;             // Measurement matrix
    float* d_R;             // Measurement noise covariance matrix
    float* d_z;             // Measurement vector
    float* d_y;             // Innovation vector
    float* d_S;             // Innovation covariance matrix
    float* d_K;             // Kalman gain
    float* d_I;             // Identity matrix
    
    // Temporary matrices for calculations
    float* d_x_temp;        // Temporary state vector
    float* d_F_T;           // Transpose of F
    float* d_H_T;           // Transpose of H
    float* d_S_inv;         // Inverse of S
    float* d_temp1;         // Temporary matrix 1
    float* d_temp2;         // Temporary matrix 2
    float* d_Hx;            // H * x
    float* d_Ky;            // K * y
    float* d_KH;            // K * H
    float* d_IKH;           // I - K * H
} KalmanFilter;

// Function prototypes
KalmanFilter* init_cuda_kalman_filter(int state_dim, int measurement_dim);
void cuda_kalman_filter_predict(KalmanFilter* kf);
void cuda_kalman_filter_update(KalmanFilter* kf, float* measurement);
void free_cuda_kalman_filter(KalmanFilter* kf);

#endif /* _GUAC_KALMAN_FILTER_H */