#ifndef GUACENC_VIDEO_FILTER_H
#define GUACENC_VIDEO_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct VideoFilter VideoFilter;

// Function declarations
VideoFilter* init_video_filter(int width, int height, int channels);
void process_frame(VideoFilter* filter, const unsigned char* frame_data);
void apply_filter_to_frame(VideoFilter* filter);
void free_video_filter(VideoFilter* filter);

#ifdef __cplusplus
}
#endif

#endif /* GUACENC_VIDEO_FILTER_H */
