# Kalman Filter Middleware Debugging Tools

This repository contains tools to help debug and fix issues with the Kalman filter middleware for Apache Guacamole.

## Problem Description

The Kalman filter middleware is supposed to intercept image instructions (img, png, jpeg) from the Guacamole protocol and apply a CUDA-accelerated Kalman filter to smooth cursor movements and optimize video quality. However, the middleware is not currently invoking the Kalman filter when these instructions are passed through.

## Included Tools

### 1. `check-kalman-middleware.sh`

This script checks if the Kalman filter middleware is properly configured to intercept and process image instructions.

**Usage:**
```bash
sudo ./check-kalman-middleware.sh /usr/local/etc/guacamole/kalman-proxy.conf
```

The script will:
- Check if the Kalman filter is enabled in the configuration
- Verify if image instructions are configured to be filtered
- Check if the middleware binary exists and is executable
- Verify CUDA environment
- Check if the middleware is running and listening on the correct port

### 2. `kalman-middleware-debug.sh`

This script provides more detailed debugging by monitoring network traffic and checking if the Kalman filter is being invoked.

**Usage:**
```bash
sudo ./kalman-middleware-debug.sh /usr/local/etc/guacamole/kalman-proxy.conf [debug_level]
```

Where `debug_level` can be:
- 1: Basic debugging
- 2: Detailed debugging (default)
- 3: Verbose debugging

The script will:
- Monitor network traffic between client, middleware, and server
- Log when "img" instructions are detected
- Check if CUDA functions are being called
- Monitor the middleware log for errors or important messages

### 3. `kalman-proxy-sample.conf`

A sample configuration file with the recommended settings to enable the Kalman filter for image instructions.

## Common Issues and Solutions

1. **Kalman filter not enabled in configuration**
   - Make sure `enable_kalman_filter = true` is set in your configuration file

2. **Image instructions not configured to be filtered**
   - Add or update `filter_instructions = img,png,jpeg` in your configuration file

3. **CUDA environment issues**
   - Ensure NVIDIA drivers and CUDA are properly installed
   - Check if the middleware has access to the GPU

4. **Middleware not running or not listening on the correct port**
   - Start the middleware with `sudo /usr/local/bin/kalman-proxy /path/to/config.conf`
   - Check for any error messages in the logs

5. **Middleware not intercepting instructions**
   - Use the debug script to monitor network traffic and verify if instructions are being passed through

## How to Use These Tools

1. First, run the check script to identify any configuration issues:
   ```bash
   sudo ./check-kalman-middleware.sh /usr/local/etc/guacamole/kalman-proxy.conf
   ```

2. If needed, update your configuration based on the recommendations or use the sample configuration as a reference.

3. For more detailed debugging, run the debug script:
   ```bash
   sudo ./kalman-middleware-debug.sh /usr/local/etc/guacamole/kalman-proxy.conf
   ```

4. Check the logs in `/tmp/kalman-middleware-debug/` for any issues or errors.

## Next Steps

If the tools identify that the middleware is properly configured but still not invoking the Kalman filter, you may need to:

1. Check the middleware source code to ensure it's correctly parsing and handling image instructions
2. Verify that the CUDA Kalman filter implementation is being properly linked and called
3. Check for any runtime errors in the middleware logs

## Support

If you continue to experience issues, please provide:
1. The output from the check script
2. The debug logs from the debug script
3. Your current configuration file (with any sensitive information removed)

This will help diagnose and fix the issue more effectively.