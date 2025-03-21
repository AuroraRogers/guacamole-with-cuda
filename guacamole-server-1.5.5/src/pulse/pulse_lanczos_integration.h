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

#ifndef PULSE_LANCZOS_INTEGRATION_H
#define PULSE_LANCZOS_INTEGRATION_H

#include "config.h"
#include "pulse/pulse.h"

#include <guacamole/client.h>
#include <pulse/pulseaudio.h>

/**
 * Enhanced version of the PulseAudio stream read callback that applies
 * Lanczos resampling to the audio data.
 *
 * @param stream
 *     The PulseAudio stream which has PCM data available.
 * @param length
 *     The number of bytes of PCM data available on the given stream.
 * @param data
 *     A pointer to the guac_pa_stream structure associated with the Guacamole
 *     stream receiving audio data from PulseAudio.
 */
void guac_pa_stream_read_callback_lanczos(pa_stream* stream, size_t length, void* data);

/**
 * Allocates a new PulseAudio stream with Lanczos resampling support.
 *
 * @param client
 *     The client to stream audio to.
 * @param server_name
 *     The hostname of the PulseAudio server to connect to, or NULL to connect
 *     to the default (local) server.
 * @param target_rate
 *     The target sample rate for output (after resampling).
 * @return
 *     A newly-allocated PulseAudio stream, or NULL if audio cannot be
 *     streamed.
 */
guac_pa_stream* guac_pa_stream_alloc_lanczos(guac_client* client,
                                           const char* server_name,
                                           int target_rate);

/**
 * Frees a PulseAudio stream with Lanczos resampling support.
 *
 * @param stream
 *     The PulseAudio stream to free.
 */
void guac_pa_stream_free_lanczos(guac_pa_stream* stream);

#endif /* PULSE_LANCZOS_INTEGRATION_H */