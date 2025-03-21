# Lanczos Audio Resampling for Guacamole PulseAudio

This implementation adds high-quality Lanczos interpolation for audio resampling in the PulseAudio component of Apache Guacamole. Lanczos interpolation is a sophisticated technique that provides superior audio quality compared to simpler resampling methods.

## Overview

The Lanczos resampling algorithm uses a windowed sinc function to interpolate between audio samples. It provides several advantages over simpler resampling methods:

1. **Better frequency response**: Lanczos preserves more high-frequency content while minimizing aliasing
2. **Reduced artifacts**: Less distortion and ringing compared to linear interpolation
3. **Improved transient response**: Better preservation of sharp audio transitions

## Implementation Details

The implementation consists of three main components:

1. **Core Lanczos Algorithm** (`lanczos_audio_resampler.c/h`):
   - Implements the Lanczos kernel function
   - Provides optimized resampling with lookup tables
   - Supports configurable Lanczos parameter (a) for quality control

2. **PulseAudio Integration** (`pulse_lanczos_integration.c/h`):
   - Extends the PulseAudio stream with Lanczos resampling
   - Manages audio buffers and state for continuous resampling
   - Provides a drop-in replacement for the standard PulseAudio stream

3. **Usage Example**:
   - Replace standard PulseAudio stream allocation with Lanczos-enabled version
   - Configure target sample rate for optimal audio quality

## How to Use

To use Lanczos resampling in your Guacamole protocol implementation:

1. Include the necessary headers:
   ```c
   #include "pulse_lanczos_integration.h"
   ```

2. Replace standard PulseAudio stream allocation with Lanczos-enabled version:
   ```c
   // Instead of:
   // guac_pa_stream* stream = guac_pa_stream_alloc(client, server_name);
   
   // Use:
   int target_rate = 48000; // High-quality output rate
   guac_pa_stream* stream = guac_pa_stream_alloc_lanczos(client, server_name, target_rate);
   ```

3. Free the stream when done:
   ```c
   guac_pa_stream_free_lanczos(stream);
   ```

## Performance Considerations

Lanczos resampling is more computationally intensive than simpler methods. The implementation includes several optimizations:

1. **Precomputed kernel table**: Reduces runtime calculations
2. **Optimized buffer management**: Minimizes memory operations
3. **Configurable quality parameter**: The Lanczos parameter (a) can be adjusted to balance quality and performance

For most modern systems, the performance impact is minimal compared to the quality improvement.

## Future Improvements

Potential enhancements for future versions:

1. **CUDA acceleration**: Implement GPU-accelerated version for even better performance
2. **Adaptive quality**: Dynamically adjust resampling quality based on CPU load
3. **Multi-rate support**: Allow different resampling rates for different audio streams
4. **Audio effects**: Add additional processing like noise reduction or echo cancellation

## References

- Lanczos, C. (1938). "Trigonometric Interpolation of Empirical and Analytical Functions"
- Smith, J.O. (2011). "Spectral Audio Signal Processing"
- Duchon, C.E. (1979). "Lanczos Filtering in One and Two Dimensions"