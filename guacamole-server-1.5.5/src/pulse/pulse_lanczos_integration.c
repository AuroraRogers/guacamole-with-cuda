/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "config.h"
#include "pulse/pulse.h"
#include "pulse/lanczos_audio_resampler.h"
#include "pulse_lanczos_integration.h"

#include <guacamole/audio.h>
#include <guacamole/mem.h>
#include <guacamole/client.h>
#include <guacamole/user.h>
#include <pulse/pulseaudio.h>

#include <stdlib.h>
#include <string.h>

/**
 * The default Lanczos parameter to use for audio resampling.
 * Higher values provide better quality but require more computation.
 * Typical values are 2 or 3.
 */
#define GUAC_LANCZOS_PARAM 3

/**
 * The size of the Lanczos kernel lookup table.
 * Larger tables provide more accurate interpolation but use more memory.
 */
#define GUAC_LANCZOS_TABLE_SIZE 1024

/**
 * Structure for tracking Lanczos resampling state.
 */
typedef struct guac_lanczos_state {
    
    /**
     * The Lanczos parameter (a).
     */
    int a;
    
    /**
     * Precomputed Lanczos kernel lookup table.
     */
    double* kernel_table;
    
    /**
     * Size of the kernel lookup table.
     */
    int table_size;
    
    /**
     * Target sample rate for resampling.
     */
    int target_rate;
    
    /**
     * Source sample rate from PulseAudio.
     */
    int source_rate;
    
    /**
     * Number of audio channels.
     */
    int channels;
    
    /**
     * Bits per sample.
     */
    int bps;
    
    /**
     * Temporary buffer for input samples.
     */
    short* input_buffer;
    
    /**
     * Current number of samples in the input buffer.
     */
    int input_buffer_size;
    
    /**
     * Maximum capacity of the input buffer.
     */
    int input_buffer_capacity;
    
    /**
     * Temporary buffer for resampled output.
     */
    short* output_buffer;
    
    /**
     * Maximum capacity of the output buffer.
     */
    int output_buffer_capacity;
    
} guac_lanczos_state;

/**
 * Creates a new Lanczos resampling state object.
 *
 * @param source_rate
 *     The source sample rate from PulseAudio.
 * @param target_rate
 *     The target sample rate for output.
 * @param channels
 *     The number of audio channels.
 * @param bps
 *     The bits per sample.
 * @param a
 *     The Lanczos parameter (typically 2 or 3).
 * @return
 *     A newly allocated Lanczos state object.
 */
guac_lanczos_state* guac_lanczos_state_alloc(int source_rate, int target_rate,
                                            int channels, int bps, int a) {
    
    guac_lanczos_state* state = guac_mem_alloc(sizeof(guac_lanczos_state));
    
    state->a = a;
    state->source_rate = source_rate;
    state->target_rate = target_rate;
    state->channels = channels;
    state->bps = bps;
    
    // Allocate and precompute the Lanczos kernel lookup table
    state->table_size = GUAC_LANCZOS_TABLE_SIZE;
    state->kernel_table = guac_mem_alloc(state->table_size * sizeof(double));
    precompute_lanczos_kernel(a, state->table_size, state->kernel_table);
    
    // Allocate input and output buffers
    // The input buffer should be large enough to hold at least 2*a samples
    // plus the maximum expected input chunk size
    state->input_buffer_capacity = GUAC_PULSE_AUDIO_FRAGMENT_SIZE / (bps/8) / channels + 2*a;
    state->input_buffer = guac_mem_alloc(state->input_buffer_capacity * channels * sizeof(short));
    state->input_buffer_size = 0;
    
    // The output buffer should be sized based on the resampling ratio
    double ratio = (double)target_rate / source_rate;
    state->output_buffer_capacity = (int)(state->input_buffer_capacity * ratio) + 1;
    state->output_buffer = guac_mem_alloc(state->output_buffer_capacity * channels * sizeof(short));
    
    return state;
}

/**
 * Frees a Lanczos resampling state object.
 *
 * @param state
 *     The Lanczos state object to free.
 */
void guac_lanczos_state_free(guac_lanczos_state* state) {
    if (state) {
        guac_mem_free(state->kernel_table);
        guac_mem_free(state->input_buffer);
        guac_mem_free(state->output_buffer);
        guac_mem_free(state);
    }
}

/**
 * Processes audio data through the Lanczos resampler.
 *
 * @param state
 *     The Lanczos state object.
 * @param input
 *     The input PCM data buffer.
 * @param length
 *     The length of the input buffer in bytes.
 * @param audio
 *     The audio stream to write the resampled data to.
 */
void guac_lanczos_process(guac_lanczos_state* state, const void* input,
                         int length, guac_audio_stream* audio) {
    
    // Convert input bytes to samples
    int bytes_per_sample = state->bps / 8;
    int input_samples = length / bytes_per_sample / state->channels;
    
    // Convert input to 16-bit samples if needed
    const short* input_pcm;
    short* temp_buffer = NULL;
    
    if (state->bps == 16) {
        // Input is already 16-bit
        input_pcm = (const short*)input;
    } else if (state->bps == 8) {
        // Convert 8-bit to 16-bit
        temp_buffer = guac_mem_alloc(input_samples * state->channels * sizeof(short));
        const unsigned char* input_8bit = (const unsigned char*)input;
        
        for (int i = 0; i < input_samples * state->channels; i++) {
            // Convert 8-bit unsigned to 16-bit signed
            temp_buffer[i] = ((int)input_8bit[i] - 128) * 256;
        }
        
        input_pcm = temp_buffer;
    } else {
        // Unsupported bit depth
        if (temp_buffer)
            guac_mem_free(temp_buffer);
        return;
    }
    
    // Check if we need to resample
    if (state->source_rate == state->target_rate) {
        // No resampling needed, just pass through
        guac_audio_stream_write_pcm(audio, (unsigned char*)input_pcm, 
                                   input_samples * state->channels * sizeof(short));
    } else {
        // Add new samples to the input buffer
        if (state->input_buffer_size + input_samples > state->input_buffer_capacity) {
            // Buffer overflow, discard oldest samples
            int discard = state->input_buffer_size + input_samples - state->input_buffer_capacity;
            
            if (discard < state->input_buffer_size) {
                // Shift buffer to make room
                memmove(state->input_buffer, 
                       state->input_buffer + discard * state->channels,
                       (state->input_buffer_size - discard) * state->channels * sizeof(short));
                
                state->input_buffer_size -= discard;
            } else {
                // Discard all old samples
                state->input_buffer_size = 0;
            }
        }
        
        // Copy new samples to input buffer
        memcpy(state->input_buffer + state->input_buffer_size * state->channels,
              input_pcm, input_samples * state->channels * sizeof(short));
        
        state->input_buffer_size += input_samples;
        
        // Calculate output size based on resampling ratio
        double ratio = (double)state->target_rate / state->source_rate;
        int output_samples = (int)(state->input_buffer_size * ratio);
        
        if (output_samples > 0) {
            // Ensure output buffer is large enough
            if (output_samples > state->output_buffer_capacity) {
                state->output_buffer_capacity = output_samples;
                state->output_buffer = guac_mem_realloc(state->output_buffer, 
                                                      state->output_buffer_capacity * 
                                                      state->channels * sizeof(short));
            }
            
            // Perform Lanczos resampling
            lanczos_resample_optimized(state->input_buffer, state->input_buffer_size,
                                     state->output_buffer, output_samples,
                                     state->channels, state->kernel_table,
                                     state->table_size, state->a);
            
            // Write resampled data to audio stream
            guac_audio_stream_write_pcm(audio, (unsigned char*)state->output_buffer,
                                       output_samples * state->channels * sizeof(short));
            
            // Reset input buffer
            state->input_buffer_size = 0;
        }
    }
    
    // Free temporary buffer if allocated
    if (temp_buffer)
        guac_mem_free(temp_buffer);
}

void guac_pa_stream_read_callback_lanczos(pa_stream* stream, size_t length, void* data) {
    
    guac_pa_stream* guac_stream = (guac_pa_stream*)data;
    guac_audio_stream* audio = guac_stream->audio;
    guac_lanczos_state* lanczos_state = (guac_lanczos_state*)guac_stream->lanczos_state;
    
    const void* buffer;
    
    // Read data
    pa_stream_peek(stream, &buffer, &length);
    
    // Process audio data if not silence
    if (!guac_pa_is_silence(buffer, length) && lanczos_state != NULL) {
        guac_lanczos_process(lanczos_state, buffer, length, audio);
    } else if (!guac_pa_is_silence(buffer, length)) {
        // Fallback to direct PCM writing if Lanczos state is not available
        guac_audio_stream_write_pcm(audio, buffer, length);
    } else {
        // Flush on silence
        guac_audio_stream_flush(audio);
    }
    
    // Advance buffer
    pa_stream_drop(stream);
}

guac_pa_stream* guac_pa_stream_alloc_lanczos(guac_client* client,
                                           const char* server_name,
                                           int target_rate) {
    
    guac_audio_stream* audio = guac_audio_stream_alloc(client, NULL,
                                                     target_rate,
                                                     GUAC_PULSE_AUDIO_CHANNELS,
                                                     GUAC_PULSE_AUDIO_BPS);
    
    // Abort if audio stream cannot be created
    if (audio == NULL)
        return NULL;
    
    // Init main loop
    guac_pa_stream* stream = guac_mem_alloc(sizeof(guac_pa_stream));
    stream->client = client;
    stream->audio = audio;
    stream->pa_mainloop = pa_threaded_mainloop_new();
    
    // Create Lanczos resampling state
    guac_lanczos_state* lanczos_state = guac_lanczos_state_alloc(
        GUAC_PULSE_AUDIO_RATE,  // Source rate from PulseAudio
        target_rate,            // Target rate for output
        GUAC_PULSE_AUDIO_CHANNELS,
        GUAC_PULSE_AUDIO_BPS,
        GUAC_LANCZOS_PARAM);
    
    stream->lanczos_state = lanczos_state;
    
    // Create context
    pa_context* context = pa_context_new(
            pa_threaded_mainloop_get_api(stream->pa_mainloop),
            "Guacamole Audio");
    
    // Set up context
    pa_context_set_state_callback(context, __context_state_callback, stream);
    pa_context_connect(context, server_name, PA_CONTEXT_NOAUTOSPAWN, NULL);
    
    // Start loop
    pa_threaded_mainloop_start(stream->pa_mainloop);
    
    return stream;
}

void guac_pa_stream_free_lanczos(guac_pa_stream* stream) {
    
    // Stop loop
    pa_threaded_mainloop_stop(stream->pa_mainloop);
    
    // Free Lanczos state
    guac_lanczos_state_free((guac_lanczos_state*)stream->lanczos_state);
    
    // Free underlying audio stream
    guac_audio_stream_free(stream->audio);
    
    // Stream now ended
    guac_client_log(stream->client, GUAC_LOG_INFO, "Audio stream finished");
    guac_mem_free(stream);
}