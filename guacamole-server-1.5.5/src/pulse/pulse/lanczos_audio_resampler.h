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

#ifndef LANCZOS_AUDIO_RESAMPLER_H
#define LANCZOS_AUDIO_RESAMPLER_H

/**
 * Precompute the Lanczos kernel for faster resampling.
 *
 * @param a
 *     The Lanczos parameter (typically 2 or 3).
 * @param table_size
 *     The size of the lookup table.
 * @param kernel_table
 *     The output table to store the precomputed kernel values.
 */
void precompute_lanczos_kernel(int a, int table_size, double* kernel_table);

/**
 * Resample audio data using Lanczos interpolation.
 *
 * @param input
 *     The input audio buffer.
 * @param input_size
 *     The number of samples in the input buffer.
 * @param output
 *     The output audio buffer.
 * @param output_size
 *     The number of samples to generate in the output buffer.
 * @param channels
 *     The number of audio channels.
 * @param a
 *     The Lanczos parameter (typically 2 or 3).
 */
void lanczos_resample(const short* input, int input_size, 
                      short* output, int output_size, 
                      int channels, int a);

/**
 * Optimized version of Lanczos resampling using a precomputed kernel table.
 *
 * @param input
 *     The input audio buffer.
 * @param input_size
 *     The number of samples in the input buffer.
 * @param output
 *     The output audio buffer.
 * @param output_size
 *     The number of samples to generate in the output buffer.
 * @param channels
 *     The number of audio channels.
 * @param kernel_table
 *     The precomputed Lanczos kernel table.
 * @param table_size
 *     The size of the kernel table.
 * @param a
 *     The Lanczos parameter (typically 2 or 3).
 */
void lanczos_resample_optimized(const short* input, int input_size, 
                               short* output, int output_size, 
                               int channels, const double* kernel_table, 
                               int table_size, int a);

#endif /* LANCZOS_AUDIO_RESAMPLER_H */