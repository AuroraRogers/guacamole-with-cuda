#!/bin/bash

# Script to check if the Kalman filter middleware is properly configured
# to intercept and process "img" instructions

# Check if we have root privileges
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit 1
fi

# Get the configuration file path from command line argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/kalman-proxy.conf"
  exit 1
fi

CONFIG_FILE="$1"

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Configuration file not found: $CONFIG_FILE"
  exit 1
fi

# Extract configuration values
LISTEN_PORT=$(grep -oP 'listen_port\s*=\s*\K[0-9]+' "$CONFIG_FILE" 2>/dev/null)
DEST_PORT=$(grep -oP 'dest_port\s*=\s*\K[0-9]+' "$CONFIG_FILE" 2>/dev/null)
DEST_HOST=$(grep -oP 'dest_host\s*=\s*\K[^[:space:]]+' "$CONFIG_FILE" 2>/dev/null)
FILTER_ENABLED=$(grep -oP 'enable_kalman_filter\s*=\s*\K(true|false|1|0)' "$CONFIG_FILE" 2>/dev/null)
FILTER_INSTRUCTIONS=$(grep -oP 'filter_instructions\s*=\s*\K[^[:space:]]+' "$CONFIG_FILE" 2>/dev/null)

echo "=== Kalman Middleware Configuration Check ==="
echo ""

# Check basic configuration
echo "Basic Configuration:"
echo "  Listen Port: ${LISTEN_PORT:-NOT FOUND}"
echo "  Destination: ${DEST_HOST:-NOT FOUND}:${DEST_PORT:-NOT FOUND}"
echo ""

# Check if Kalman filter is enabled
echo "Kalman Filter Configuration:"
if [ -z "$FILTER_ENABLED" ]; then
  echo "  [WARNING] 'enable_kalman_filter' setting not found in config file"
  echo "  The Kalman filter may be disabled by default"
  FILTER_ENABLED="unknown"
else
  if [ "$FILTER_ENABLED" = "true" ] || [ "$FILTER_ENABLED" = "1" ]; then
    echo "  Kalman Filter: ENABLED"
  else
    echo "  [WARNING] Kalman Filter: DISABLED"
    echo "  Set 'enable_kalman_filter = true' in your config file"
  fi
fi

# Check if img instructions are being filtered
echo "  Filter Instructions: ${FILTER_INSTRUCTIONS:-NOT FOUND}"
if [ -z "$FILTER_INSTRUCTIONS" ]; then
  echo "  [WARNING] 'filter_instructions' setting not found in config file"
  echo "  The middleware may not be configured to intercept 'img' instructions"
else
  if echo "$FILTER_INSTRUCTIONS" | grep -q "img"; then
    echo "  'img' instructions will be filtered: YES"
  else
    echo "  [WARNING] 'img' instructions will be filtered: NO"
    echo "  Make sure 'filter_instructions' includes 'img' in your config file"
  fi
fi

echo ""

# Check if kalman-proxy binary exists
echo "Binary Check:"
if [ -f "/usr/local/bin/kalman-proxy" ]; then
  echo "  kalman-proxy binary: FOUND"
  
  # Check if it's executable
  if [ -x "/usr/local/bin/kalman-proxy" ]; then
    echo "  kalman-proxy is executable: YES"
  else
    echo "  [WARNING] kalman-proxy is executable: NO"
    echo "  Run: chmod +x /usr/local/bin/kalman-proxy"
  fi
  
  # Check if it's a valid ELF binary
  if file "/usr/local/bin/kalman-proxy" | grep -q "ELF"; then
    echo "  kalman-proxy is a valid binary: YES"
  else
    echo "  [WARNING] kalman-proxy may not be a valid binary"
    echo "  Run: file /usr/local/bin/kalman-proxy"
  fi
else
  echo "  [ERROR] kalman-proxy binary not found at /usr/local/bin/kalman-proxy"
  echo "  Make sure the middleware is properly installed"
fi

echo ""

# Check if CUDA is available
echo "CUDA Environment Check:"
if command -v nvidia-smi &> /dev/null; then
  echo "  NVIDIA driver: INSTALLED"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
  echo "  [WARNING] NVIDIA driver: NOT FOUND"
  echo "  The Kalman filter requires CUDA to function properly"
fi

if [ -d "/usr/local/cuda" ]; then
  echo "  CUDA installation: FOUND"
  if [ -f "/usr/local/cuda/version.txt" ]; then
    echo "  CUDA version: $(cat /usr/local/cuda/version.txt)"
  fi
else
  echo "  [WARNING] CUDA installation: NOT FOUND at /usr/local/cuda"
  echo "  The Kalman filter requires CUDA to function properly"
fi

echo ""

# Check if the middleware is currently running
echo "Process Check:"
MIDDLEWARE_PID=$(pgrep -f "kalman-proxy $CONFIG_FILE" || echo "")
if [ -n "$MIDDLEWARE_PID" ]; then
  echo "  kalman-proxy is running: YES (PID: $MIDDLEWARE_PID)"
  
  # Check if it's using the correct ports
  if netstat -tuln | grep -q ":$LISTEN_PORT "; then
    echo "  Listening on port $LISTEN_PORT: YES"
  else
    echo "  [WARNING] Not listening on port $LISTEN_PORT"
    echo "  The middleware may not be properly binding to the configured port"
  fi
  
  # Check if it's connecting to the destination
  if netstat -tun | grep -q "$DEST_HOST:$DEST_PORT"; then
    echo "  Connected to $DEST_HOST:$DEST_PORT: YES"
  else
    echo "  [INFO] No active connection to $DEST_HOST:$DEST_PORT detected"
    echo "  This is normal if no client is currently connected"
  fi
else
  echo "  [WARNING] kalman-proxy is not running"
  echo "  Start it with: sudo /usr/local/bin/kalman-proxy $CONFIG_FILE"
fi

echo ""
echo "=== Recommendations ==="
echo ""

# Provide recommendations based on findings
if [ "$FILTER_ENABLED" != "true" ] && [ "$FILTER_ENABLED" != "1" ]; then
  echo "1. Make sure 'enable_kalman_filter = true' is set in your config file"
fi

if [ -z "$FILTER_INSTRUCTIONS" ] || ! echo "$FILTER_INSTRUCTIONS" | grep -q "img"; then
  echo "2. Add or update 'filter_instructions = img,png,jpeg' in your config file"
fi

if [ -z "$MIDDLEWARE_PID" ]; then
  echo "3. Start the middleware with: sudo /usr/local/bin/kalman-proxy $CONFIG_FILE"
fi

echo "4. Use the kalman-middleware-debug.sh script to monitor traffic and debug issues"
echo "5. Check the middleware logs for any error messages"

echo ""
echo "For more detailed debugging, run: sudo ./kalman-middleware-debug.sh $CONFIG_FILE"