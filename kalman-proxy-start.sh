#!/bin/bash

# Script to start the kalman-proxy after checking for existing instances

CONFIG_FILE="/usr/local/etc/guacamole/kalman-proxy.conf"
PROXY_BIN="/usr/local/bin/kalman-proxy"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found."
    exit 1
fi

# Get the port from the config file
PORT=$(grep "listen_port" "$CONFIG_FILE" | cut -d'=' -f2 | tr -d ' ')
if [ -z "$PORT" ]; then
    echo "Warning: Could not find listen_port in config file, using default 4822"
    PORT=4822
fi

# Check if the port is already in use
PID=$(lsof -t -i:$PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "Port $PORT is already in use by process $PID"
    echo "Killing existing process..."
    kill -9 $PID
    sleep 1
fi

# Start the kalman-proxy
echo "Starting kalman-proxy on port $PORT..."
$PROXY_BIN $CONFIG_FILE

# Check if it started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start kalman-proxy"
    exit 1
fi

echo "kalman-proxy started successfully"