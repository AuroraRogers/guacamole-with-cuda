#!/bin/bash

# Script to debug why the Kalman filter middleware isn't processing image instructions
# This script will:
# 1. Monitor network traffic between client, middleware, and server
# 2. Log when "img" instructions are detected
# 3. Check if the Kalman filter is being invoked

# Check if we have root privileges
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit 1
fi

# Get the configuration file path from command line argument
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/kalman-proxy.conf [debug_level]"
  echo "  debug_level: 1=basic, 2=detailed, 3=verbose (default: 2)"
  exit 1
fi

CONFIG_FILE="$1"
DEBUG_LEVEL="${2:-2}"

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Configuration file not found: $CONFIG_FILE"
  exit 1
fi

# Extract ports from the configuration file
LISTEN_PORT=$(grep -oP 'listen_port\s*=\s*\K[0-9]+' "$CONFIG_FILE" 2>/dev/null)
DEST_PORT=$(grep -oP 'dest_port\s*=\s*\K[0-9]+' "$CONFIG_FILE" 2>/dev/null)
DEST_HOST=$(grep -oP 'dest_host\s*=\s*\K[^[:space:]]+' "$CONFIG_FILE" 2>/dev/null)

if [ -z "$LISTEN_PORT" ] || [ -z "$DEST_PORT" ] || [ -z "$DEST_HOST" ]; then
  echo "Could not extract all required configuration values:"
  echo "  listen_port = $LISTEN_PORT"
  echo "  dest_port = $DEST_PORT"
  echo "  dest_host = $DEST_HOST"
  echo "Please check your configuration file."
  exit 1
fi

echo "Middleware Configuration:"
echo "  Listen Port: $LISTEN_PORT"
echo "  Destination: $DEST_HOST:$DEST_PORT"

# Create a log directory
LOG_DIR="/tmp/kalman-middleware-debug"
mkdir -p "$LOG_DIR"
TCPDUMP_LOG="$LOG_DIR/tcpdump.log"
MIDDLEWARE_LOG="$LOG_DIR/middleware.log"
STRACE_LOG="$LOG_DIR/strace.log"

# Function to clean up background processes
cleanup() {
  echo "Cleaning up..."
  [ -n "$TCPDUMP_PID" ] && kill $TCPDUMP_PID 2>/dev/null
  [ -n "$STRACE_PID" ] && kill $STRACE_PID 2>/dev/null
  [ -n "$MIDDLEWARE_PID" ] && kill $MIDDLEWARE_PID 2>/dev/null
  exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Start tcpdump to capture network traffic if debug level is high enough
if [ "$DEBUG_LEVEL" -ge 2 ]; then
  echo "Starting network capture..."
  tcpdump -i any -nn "port $LISTEN_PORT or port $DEST_PORT" -A -s 0 > "$TCPDUMP_LOG" 2>&1 &
  TCPDUMP_PID=$!
  echo "Network capture started (PID: $TCPDUMP_PID, Log: $TCPDUMP_LOG)"
fi

# Function to monitor for "img" instructions in the network traffic
monitor_img_instructions() {
  echo "Monitoring for 'img' instructions..."
  tail -f "$TCPDUMP_LOG" | grep --line-buffered -a "img" | while read line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] IMG instruction detected: $line" >> "$LOG_DIR/img_instructions.log"
    echo "IMG instruction detected!"
  done
}

# Start the monitoring in background if debug level is high enough
if [ "$DEBUG_LEVEL" -ge 2 ]; then
  monitor_img_instructions &
  MONITOR_PID=$!
fi

# Start the middleware with strace to track system calls if debug level is high enough
if [ "$DEBUG_LEVEL" -ge 3 ]; then
  echo "Starting middleware with strace..."
  strace -f -e trace=network,process -o "$STRACE_LOG" /usr/local/bin/kalman-proxy "$CONFIG_FILE" > "$MIDDLEWARE_LOG" 2>&1 &
  MIDDLEWARE_PID=$!
else
  echo "Starting middleware..."
  /usr/local/bin/kalman-proxy "$CONFIG_FILE" > "$MIDDLEWARE_LOG" 2>&1 &
  MIDDLEWARE_PID=$!
fi

echo "Middleware started (PID: $MIDDLEWARE_PID, Log: $MIDDLEWARE_LOG)"

# Function to check if CUDA functions are being called
check_cuda_calls() {
  echo "Checking for CUDA function calls..."
  while true; do
    if grep -q "cuda" "$STRACE_LOG" 2>/dev/null; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA functions are being called" >> "$LOG_DIR/cuda_calls.log"
      echo "CUDA functions detected!"
    fi
    sleep 5
  done
}

# Start CUDA call checking if debug level is high enough
if [ "$DEBUG_LEVEL" -ge 3 ]; then
  check_cuda_calls &
  CUDA_CHECK_PID=$!
fi

# Monitor the middleware log for any errors or important messages
tail -f "$MIDDLEWARE_LOG" | grep --line-buffered -E "ERROR|WARN|kalman|filter|cuda" &
TAIL_PID=$!

echo "Debug environment set up. Press Ctrl+C to stop."
echo "Log files are in $LOG_DIR"

# Keep the script running
wait $MIDDLEWARE_PID
echo "Middleware process has exited."
cleanup