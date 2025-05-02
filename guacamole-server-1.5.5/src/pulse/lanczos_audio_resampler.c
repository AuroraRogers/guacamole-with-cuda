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
#include "pulse/lanczos_audio_resampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * Lanczos kernel function with parameter a.
 * This is the sinc function windowed by the Lanczos window.
 *
 * @param x
 *     The input value.
 * @param a
 *     The Lanczos parameter (typically 2 or 3).
 * @return
 *     The Lanczos kernel value at x.
 */
static double lanczos_kernel(double x, int a) {
    // Handle the special case at x=0
    if (x == 0.0) {
        return 1.0;
    }
    
    // Outside the window, the kernel is zero
    if (x <= -a || x >= a) {
        return 0.0;
    }
    
    // Compute the Lanczos kernel: sinc(x) * sinc(x/a)
    double pi_x = M_PI * x;
    return a * sin(pi_x) * sin(pi_x / a) / (pi_x * pi_x);
}

void precompute_lanczos_kernel(int a, int table_size, double* kernel_table) {
    double step = (double)a / ((table_size - 1) / 2);
    int half_size = table_size / 2;
    
    for (int i = 0; i < table_size; i++) {
        double x = (i - half_size) * step;
        kernel_table[i] = lanczos_kernel(x, a);
    }
}

void lanczos_resample(const short* input, int input_size, 
                      short* output, int output_size, 
                      int channels, int a) {
    
    // Calculate the resampling ratio
    double ratio = (double)input_size / output_size;
    
    // For each output sample
    for (int i = 0; i < output_size; i++) {
        // Calculate the corresponding position in the input
        double input_pos = i * ratio;
        int input_index = (int)input_pos;
        double frac = input_pos - input_index;
        
        // For each channel
        for (int c = 0; c < channels; c++) {
            double sum = 0.0;
            double norm = 0.0;
            
            // Apply the Lanczos kernel
            for (int j = -a + 1; j < a; j++) {
                int idx = input_index + j;
                
                // Handle boundary conditions (zero padding)
                if (idx >= 0 && idx < input_size) {
                    double weight = lanczos_kernel(j - frac, a);
                    sum += input[idx * channels + c] * weight;
                    norm += weight;
                }
            }
            
            // Normalize and store the result
            if (norm != 0.0) {
                sum /= norm;
            }
            
            // Clamp to short range and convert back
            if (sum > 32767.0) sum = 32767.0;
            if (sum < -32768.0) sum = -32768.0;
            
            output[i * channels + c] = (short)sum;
        }
    }
}

void lanczos_resample_optimized(const short* input, int input_size, 
                               short* output, int output_size, 
                               int channels, const double* kernel_table, 
                               int table_size, int a) {
    
    // Calculate the resampling ratio
    double ratio = (double)input_size / output_size;
    int half_table = table_size / 2;
    double table_scale = half_table / (double)a;
    
    // For each output sample
    for (int i = 0; i < output_size; i++) {
        // Calculate the corresponding position in the input
        double input_pos = i * ratio;
        int input_index = (int)input_pos;
        double frac = input_pos - input_index;
        
        // For each channel
        for (int c = 0; c < channels; c++) {
            double sum = 0.0;
            double norm = 0.0;
            
            // Apply the Lanczos kernel using the lookup table
            for (int j = -a + 1; j < a; j++) {
                int idx = input_index + j;
                
                // Handle boundary conditions (zero padding)
                if (idx >= 0 && idx < input_size) {
                    // Map the kernel position to the table index
                    double kernel_pos = (j - frac) * table_scale + half_table;
                    int kernel_idx = (int)kernel_pos;
                    
                    // Linear interpolation between table entries
                    double kernel_frac = kernel_pos - kernel_idx;
                    double weight;
                    
                    if (kernel_idx >= 0 && kernel_idx < table_size - 1) {
                        weight = kernel_table[kernel_idx] * (1.0 - kernel_frac) + 
                                 kernel_table[kernel_idx + 1] * kernel_frac;
                    } else if (kernel_idx == table_size - 1) {
                        weight = kernel_table[kernel_idx];
                    } else {
                        weight = 0.0;
                    }
                    
                    sum += input[idx * channels + c] * weight;
                    norm += weight;
                }
            }
            
            // Normalize and store the result
            if (norm != 0.0) {
                sum /= norm;
            }
            
            // Clamp to short range and convert back
            if (sum > 32767.0) sum = 32767.0;
            if (sum < -32768.0) sum = -32768.0;
            
            output[i * channels + c] = (short)sum;
        }
    }
}