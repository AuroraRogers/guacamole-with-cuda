# Kalman Filter Activation Fix

This repository contains tools to fix issues with the Kalman filter not being invoked in the guac-kalman-filter-middleware project.

## Problem Description

The issue you're experiencing is that the kalman-proxy is running, but it's not actually invoking the Kalman filter when processing image or cursor movement instructions. This could be due to several reasons:

1. The Kalman filter is not enabled in the configuration
2. Missing configuration parameters for image processing or cursor smoothing
3. Missing CUDA libraries or GPU support
4. Incorrect proxy configuration

## Solution

This repository provides the following tools to fix the issue:

1. **kalman-filter-activation-fix.sh** - A comprehensive script that:
   - Checks if kalman-proxy is installed and running
   - Verifies the configuration file and ensures Kalman filter is enabled
   - Adds necessary configuration parameters for image processing and cursor smoothing
   - Checks for CUDA availability and libraries
   - Creates a wrapper script to monitor filter activation
   - Restarts the proxy with proper settings

2. **kalman-proxy.conf.sample** - A sample configuration file with:
   - Properly configured Kalman filter settings
   - Image processing and cursor smoothing enabled
   - Debug logging enabled
   - Comments explaining each setting

## How to Use

1. Make the fix script executable:
   ```bash
   chmod +x kalman-filter-activation-fix.sh
   ```

2. Run the fix script with your configuration file:
   ```bash
   sudo ./kalman-filter-activation-fix.sh /usr/local/etc/guacamole/kalman-proxy.conf
   ```

3. The script will:
   - Analyze your configuration
   - Enable the Kalman filter if it's not enabled
   - Add necessary configuration parameters
   - Check for CUDA support
   - Create a wrapper script to monitor filter activation
   - Restart the proxy with proper settings

4. If you need to create a new configuration file, use the sample:
   ```bash
   cp kalman-proxy.conf.sample /usr/local/etc/guacamole/kalman-proxy.conf
   ```

## Troubleshooting

If the Kalman filter is still not being invoked after running the fix script, check the following:

1. **Check the logs**: Look at `/var/log/kalman-proxy.log` for any error messages related to the Kalman filter.

2. **Verify CUDA installation**: Make sure CUDA is properly installed and a compatible GPU is available:
   ```bash
   nvidia-smi
   ```

3. **Check for CUDA libraries**: Verify that the CUDA libraries are installed and accessible:
   ```bash
   ldconfig -p | grep libcuda
   ```

4. **Test with a simple connection**: Connect to a Guacamole session through the proxy and try moving the cursor or viewing images to trigger the Kalman filter.

5. **Check guacd configuration**: Make sure guacd is properly configured and running:
   ```bash
   ps aux | grep guacd
   ```

## Understanding the Kalman Filter in Guacamole

The Kalman filter in this project is designed to:

1. **Smooth cursor movements**: Reduce jitter and provide a more fluid cursor experience
2. **Optimize image quality**: Process images to improve quality and reduce bandwidth

For the filter to be invoked, the following conditions must be met:

1. `kalman_enabled = true` must be set in the configuration
2. For cursor smoothing: `smooth_cursor = true` must be set
3. For image processing: `process_images = true` must be set
4. CUDA libraries must be available (the filter uses GPU acceleration)
5. The proxy must be correctly forwarding the relevant Guacamole instructions (cursor movement, image data)

The fix script ensures all these conditions are met and provides monitoring to verify that the filter is being invoked.