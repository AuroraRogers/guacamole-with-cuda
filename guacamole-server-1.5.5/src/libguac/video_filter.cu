#include <cuda_runtime.h>
#include "video_filter.h"
#include <stdio.h>

// CUDA kernel for converting frame to float
__global__ void frameToFloat(const unsigned char* input, float* output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = (float)input[idx] / 255.0f;
    }
}

// CUDA kernel for converting float back to frame format
__global__ void floatToFrame(float* input, unsigned char* output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = (unsigned char)(input[idx] * 255.0f);
    }
}

VideoFilter* init_video_filter(int width, int height, int channels) {
    VideoFilter* filter = (VideoFilter*)malloc(sizeof(VideoFilter));
    filter->width = width;
    filter->height = height;
    filter->channels = channels;
    
    // Initialize Kalman filter for each pixel (simplified for demonstration)
    filter->kf = init_cuda_kalman_filter(channels, channels);
    
    int total_size = width * height * channels;
    
    // Allocate device memory
    cudaMalloc(&filter->d_frame_buffer, total_size * sizeof(float));
    cudaMalloc(&filter->d_filtered_buffer, total_size * sizeof(float));
    
    // Allocate host memory
    filter->h_frame_buffer = (float*)malloc(total_size * sizeof(float));
    
    return filter;
}

void process_frame(VideoFilter* filter, const unsigned char* frame_data) {
    int total_size = filter->width * filter->height * filter->channels;
    
    // Convert frame data to float and copy to device
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    
    unsigned char* d_input;
    cudaMalloc(&d_input, total_size);
    cudaMemcpy(d_input, frame_data, total_size, cudaMemcpyHostToDevice);
    
    frameToFloat<<<num_blocks, block_size>>>(d_input, filter->d_frame_buffer, total_size);
    
    cudaFree(d_input);
}

void apply_filter_to_frame(VideoFilter* filter) {
    int total_pixels = filter->width * filter->height;
    
    // Process each pixel with Kalman filter
    for(int i = 0; i < total_pixels; i++) {
        float* pixel = &filter->d_frame_buffer[i * filter->channels];
        
        // Predict step
        cuda_kalman_filter_predict(filter->kf);
        
        // Update step
        cuda_kalman_filter_update(filter->kf, pixel);
        
        // Copy filtered result back
        cudaMemcpy(&filter->d_filtered_buffer[i * filter->channels], 
                   filter->kf->d_x, 
                   filter->channels * sizeof(float), 
                   cudaMemcpyDeviceToDevice);
    }
}

void free_video_filter(VideoFilter* filter) {
    free_cuda_kalman_filter(filter->kf);
    cudaFree(filter->d_frame_buffer);
    cudaFree(filter->d_filtered_buffer);
    free(filter->h_frame_buffer);
    free(filter);
}
