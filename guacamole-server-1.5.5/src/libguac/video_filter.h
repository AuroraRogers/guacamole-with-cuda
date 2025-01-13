#ifndef _GUAC_VIDEO_FILTER_H
#define _GUAC_VIDEO_FILTER_H

#include "kalman_filter.h"
#include <guacamole/client.h>
#include <guacamole/layer.h>

typedef struct {
    KalmanFilter* kf;
    int width;
    int height;
    int channels;
    float* d_frame_buffer;     // Device memory for frame buffer
    float* d_filtered_buffer;  // Device memory for filtered frame
    float* h_frame_buffer;     // Host memory for frame buffer
} VideoFilter;

// Function declarations
VideoFilter* init_video_filter(int width, int height, int channels);
void process_frame(VideoFilter* filter, const unsigned char* frame_data);
void apply_filter_to_frame(VideoFilter* filter);
void free_video_filter(VideoFilter* filter);

#endif /* _GUAC_VIDEO_FILTER_H */
