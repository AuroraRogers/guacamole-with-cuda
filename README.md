# Guacamole Kalman Filter Middleware

This middleware acts as a proxy between Guacamole clients and the Guacamole server (guacd), applying a Kalman filter to image data to smooth cursor movements and optimize video quality.

## How It Works

The middleware intercepts the Guacamole protocol instructions, specifically looking for `img` instructions that contain image data (PNG, JPEG). When it finds these instructions, it applies a Kalman filter to the image data before forwarding it to the client.

## Requirements

- CUDA-capable GPU
- CUDA Toolkit (10.0 or later)
- Apache Guacamole (1.4.0 or later)
- GCC/G++ compiler
- Make

## Building

1. Clone this repository:
   ```
   git clone https://github.com/AuroraRogers/guac-kalman-filter-middleware.git
   cd guac-kalman-filter-middleware
   ```

2. Build the middleware:
   ```
   make
   ```

3. Install the middleware:
   ```
   sudo make install
   ```

## Configuration

The middleware is configured using a configuration file located at `/usr/local/etc/guacamole/kalman-proxy.conf`. The default configuration file is installed during the installation process, but you can modify it to suit your needs.

Here's an example configuration:

```
# Kalman Proxy Configuration

# Port to listen on
listen_port = 4822

# Guacamole daemon host
guacd_host = 127.0.0.1

# Guacamole daemon port
guacd_port = 4823

# Debug level (0-3)
debug_level = 1

# Enable Kalman filter for image processing
enable_kalman_filter = true

# Kalman filter parameters
kalman_process_noise = 0.01
kalman_measurement_noise = 0.1
```

## Running

To start the middleware, run:

```
sudo /usr/local/bin/kalman-proxy /usr/local/etc/guacamole/kalman-proxy.conf
```

Or use the provided start script:

```
sudo ./kalman-proxy-start.sh
```

## Troubleshooting

### Address already in use

If you see the error "Failed to bind socket: Address already in use", it means that another process is already using the port specified in the configuration file. You can use the `kalman-proxy-start.sh` script to automatically kill any existing processes using the port before starting the middleware.

### Debugging

If you're having issues with the middleware, you can increase the debug level in the configuration file to get more detailed logs:

```
debug_level = 3
```

## Architecture

The middleware works by:

1. Listening for connections from Guacamole clients
2. For each client connection, creating a connection to the Guacamole server
3. Intercepting and parsing Guacamole protocol instructions
4. Applying the Kalman filter to image data in `img` instructions
5. Forwarding the modified instructions to the appropriate destination

## License

This project is licensed under the Apache License 2.0.