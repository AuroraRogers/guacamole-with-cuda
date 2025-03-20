#!/bin/bash

# Script to fix "Address already in use" error for kalman-proxy
# This script will:
# 1. Find any processes using the port configured in kalman-proxy.conf
# 2. Kill those processes
# 3. Start kalman-proxy with the configuration file

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

# Extract the port from the configuration file
# This assumes the port is specified in the format "listen_port = XXXX"
PORT=$(grep -oP 'listen_port\s*=\s*\K[0-9]+' "$CONFIG_FILE" 2>/dev/null)

# If we couldn't find the port in the config, use a default (8080 is common)
if [ -z "$PORT" ]; then
  echo "Could not determine port from config file, checking common ports..."
  # Check for processes using common proxy ports
  for COMMON_PORT in 8080 4822 4823 8443; do
    PROCS=$(lsof -i :$COMMON_PORT -t 2>/dev/null)
    if [ ! -z "$PROCS" ]; then
      echo "Found processes using port $COMMON_PORT: $PROCS"
      PORT=$COMMON_PORT
      break
    fi
  done
  
  # If still no port found, ask user
  if [ -z "$PORT" ]; then
    echo "Could not automatically determine port. Please enter the port number:"
    read PORT
  fi
fi

echo "Checking for processes using port $PORT..."

# Find processes using the port
PROCS=$(lsof -i :$PORT -t 2>/dev/null)

if [ ! -z "$PROCS" ]; then
  echo "Found processes using port $PORT: $PROCS"
  echo "Killing these processes..."
  
  # Kill each process
  for PID in $PROCS; do
    echo "Killing process $PID"
    kill -9 $PID
  done
  
  # Wait a moment for processes to terminate
  sleep 2
  
  # Check if any processes are still using the port
  REMAINING=$(lsof -i :$PORT -t 2>/dev/null)
  if [ ! -z "$REMAINING" ]; then
    echo "Warning: Some processes are still using port $PORT: $REMAINING"
    echo "You may need to kill these manually or restart your system."
    exit 1
  fi
  
  echo "All processes using port $PORT have been terminated."
else
  echo "No processes found using port $PORT."
fi

# Start kalman-proxy with the configuration file
echo "Starting kalman-proxy with configuration file: $CONFIG_FILE"
/usr/local/bin/kalman-proxy "$CONFIG_FILE"

# Exit with the exit code from kalman-proxy
exit $?