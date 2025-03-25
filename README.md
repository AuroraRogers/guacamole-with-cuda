# Guacamole Kalman Filter Middleware

This middleware sits between the Guacamole client and the guacd server, intercepting image instructions and applying a Kalman filter to smooth cursor movements and optimize video quality using CUDA acceleration.

## How It Works

The middleware acts as a proxy that:

1. Listens for connections from Guacamole clients
2. Connects to the guacd server
3. Forwards messages between them
4. Intercepts image instructions (`img` instructions)
5. Applies a Kalman filter to the image data using CUDA
6. Forwards the filtered data

## Requirements

- CUDA-capable GPU
- CUDA Toolkit (10.0 or later)
- Guacamole Server (with development headers)
- GCC compiler
- Make

## Building

```bash
make
```

## Installation

```bash
sudo make install
```

This will:
- Install the `kalman-proxy` binary to `/usr/local/bin/`
- Install the default configuration file to `/usr/local/etc/guacamole/`

## Configuration

Edit the configuration file at `/usr/local/etc/guacamole/kalman-proxy.conf`:

```
# Port to listen on for client connections
listen_port = 4822

# guacd host and port
guacd_host = 127.0.0.1
guacd_port = 4823

# Debug level (0=none, 1=basic, 2=verbose)
debug_level = 1
```

## Running

```bash
sudo kalman-proxy /usr/local/etc/guacamole/kalman-proxy.conf
```

## Configuring Guacamole

Update your Guacamole client configuration to connect to the middleware instead of directly to guacd:

1. In your `guacamole.properties` file, set:
   ```
   guacd-hostname: 127.0.0.1
   guacd-port: 4822
   ```

2. Make sure the middleware is configured to connect to the actual guacd server (default port 4823).

## Troubleshooting

### "Address already in use" error

If you see this error:
```
Failed to bind socket: Address already in use
Failed to create server socket
```

It means another process is using the port. You can:

1. Use the provided script to fix this:
   ```
   sudo ./kalman-proxy-fix.sh /usr/local/etc/guacamole/kalman-proxy.conf
   ```

2. Or manually find and kill the process:
   ```
   sudo lsof -i :4822
   sudo kill <PID>
   ```

### No Kalman filter being applied

If the middleware is running but not applying the Kalman filter:

1. Check the debug output (set `debug_level = 2` in the config)
2. Verify that image instructions are being intercepted
3. Make sure your CUDA environment is properly set up

## License

This project is licensed under the Apache License 2.0.