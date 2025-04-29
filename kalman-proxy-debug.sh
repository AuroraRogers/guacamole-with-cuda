#!/bin/bash

# Script to debug and fix kalman-proxy integration issues
# This script will:
# 1. Check if kalman-proxy is installed and running
# 2. Verify the configuration file
# 3. Test if the kalman filter is being called
# 4. Fix common issues

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we have root privileges
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}Please run as root for full functionality${NC}"
  echo -e "${YELLOW}Running with limited capabilities...${NC}"
fi

# Get the configuration file path from command line argument
if [ "$#" -ne 1 ]; then
  echo -e "${YELLOW}Usage: $0 /path/to/kalman-proxy.conf${NC}"
  echo -e "${YELLOW}Trying to find configuration file...${NC}"
  
  # Try to find the configuration file in common locations
  POSSIBLE_CONFIGS=(
    "/usr/local/etc/guacamole/kalman-proxy.conf"
    "/etc/guacamole/kalman-proxy.conf"
    "./kalman-proxy.conf"
  )
  
  for CONFIG in "${POSSIBLE_CONFIGS[@]}"; do
    if [ -f "$CONFIG" ]; then
      echo -e "${GREEN}Found configuration file: $CONFIG${NC}"
      CONFIG_FILE="$CONFIG"
      break
    fi
  done
  
  if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}Could not find configuration file. Please specify the path.${NC}"
    exit 1
  fi
else
  CONFIG_FILE="$1"
  # Check if the configuration file exists
  if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
  fi
fi

echo -e "${GREEN}Using configuration file: $CONFIG_FILE${NC}"

# Extract configuration parameters
echo -e "${YELLOW}Analyzing configuration file...${NC}"
LISTEN_PORT=$(grep -oP 'listen_port\s*=\s*\K[0-9]+' "$CONFIG_FILE" 2>/dev/null)
GUACD_HOST=$(grep -oP 'guacd_host\s*=\s*\K[^[:space:]]+' "$CONFIG_FILE" 2>/dev/null)
GUACD_PORT=$(grep -oP 'guacd_port\s*=\s*\K[0-9]+' "$CONFIG_FILE" 2>/dev/null)
KALMAN_ENABLED=$(grep -oP 'kalman_enabled\s*=\s*\K(true|false|1|0)' "$CONFIG_FILE" 2>/dev/null)

echo -e "Configuration parameters:"
echo -e "  Listen Port: ${LISTEN_PORT:-"Not found"}"
echo -e "  Guacd Host: ${GUACD_HOST:-"Not found"}"
echo -e "  Guacd Port: ${GUACD_PORT:-"Not found"}"
echo -e "  Kalman Enabled: ${KALMAN_ENABLED:-"Not found"}"

# Check if kalman_enabled is set to true
if [ -z "$KALMAN_ENABLED" ] || [ "$KALMAN_ENABLED" = "false" ] || [ "$KALMAN_ENABLED" = "0" ]; then
  echo -e "${RED}ERROR: Kalman filter is not enabled in the configuration file!${NC}"
  echo -e "${YELLOW}Adding kalman_enabled=true to the configuration...${NC}"
  
  # Add kalman_enabled=true to the configuration file if it doesn't exist
  if ! grep -q "kalman_enabled" "$CONFIG_FILE"; then
    echo "kalman_enabled = true" >> "$CONFIG_FILE"
    echo -e "${GREEN}Added kalman_enabled=true to the configuration file.${NC}"
  else
    # Replace the existing kalman_enabled line
    sed -i 's/kalman_enabled\s*=\s*.*/kalman_enabled = true/' "$CONFIG_FILE"
    echo -e "${GREEN}Updated kalman_enabled to true in the configuration file.${NC}"
  fi
fi

# Check if kalman-proxy binary exists
KALMAN_PROXY_PATH="/usr/local/bin/kalman-proxy"
if [ ! -f "$KALMAN_PROXY_PATH" ]; then
  echo -e "${RED}ERROR: kalman-proxy binary not found at $KALMAN_PROXY_PATH${NC}"
  echo -e "${YELLOW}Checking for kalman-proxy in other locations...${NC}"
  
  KALMAN_PROXY_PATH=$(which kalman-proxy 2>/dev/null)
  if [ -z "$KALMAN_PROXY_PATH" ]; then
    echo -e "${RED}ERROR: kalman-proxy binary not found in PATH${NC}"
    echo -e "${YELLOW}Please make sure kalman-proxy is installed correctly.${NC}"
    exit 1
  else
    echo -e "${GREEN}Found kalman-proxy at: $KALMAN_PROXY_PATH${NC}"
  fi
fi

# Check if kalman-proxy is already running
if [ ! -z "$LISTEN_PORT" ]; then
  RUNNING_PID=$(lsof -i :$LISTEN_PORT -t 2>/dev/null)
  if [ ! -z "$RUNNING_PID" ]; then
    echo -e "${YELLOW}kalman-proxy is already running with PID: $RUNNING_PID${NC}"
    
    # Check if it's actually kalman-proxy
    PROCESS_NAME=$(ps -p $RUNNING_PID -o comm= 2>/dev/null)
    if [[ "$PROCESS_NAME" == *"kalman"* ]]; then
      echo -e "${GREEN}Confirmed it's a kalman-proxy process.${NC}"
    else
      echo -e "${RED}WARNING: Port $LISTEN_PORT is being used by another process: $PROCESS_NAME${NC}"
      echo -e "${YELLOW}Would you like to kill this process and start kalman-proxy? (y/n)${NC}"
      read -r KILL_PROCESS
      
      if [[ "$KILL_PROCESS" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Killing process $RUNNING_PID...${NC}"
        kill -9 $RUNNING_PID
        sleep 2
      else
        echo -e "${YELLOW}Please choose a different port in your configuration file.${NC}"
        exit 1
      fi
    fi
  else
    echo -e "${YELLOW}No process is currently using port $LISTEN_PORT.${NC}"
  fi
fi

# Check for CUDA availability
echo -e "${YELLOW}Checking for CUDA availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
  echo -e "${GREEN}CUDA is available. GPU information:${NC}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo -e "${RED}WARNING: CUDA/GPU not detected. The Kalman filter may not work properly.${NC}"
  echo -e "${YELLOW}Make sure CUDA drivers are installed and a compatible GPU is available.${NC}"
fi

# Check for libcuda.so
echo -e "${YELLOW}Checking for CUDA libraries...${NC}"
if ldconfig -p | grep -q libcuda.so; then
  echo -e "${GREEN}CUDA libraries found.${NC}"
else
  echo -e "${RED}WARNING: CUDA libraries not found in the system.${NC}"
  echo -e "${YELLOW}The Kalman filter may not work without proper CUDA libraries.${NC}"
fi

# Add debug logging to the configuration
echo -e "${YELLOW}Adding debug logging to the configuration...${NC}"
if ! grep -q "log_level" "$CONFIG_FILE"; then
  echo "log_level = debug" >> "$CONFIG_FILE"
  echo -e "${GREEN}Added log_level=debug to the configuration file.${NC}"
else
  # Replace the existing log_level line
  sed -i 's/log_level\s*=\s*.*/log_level = debug/' "$CONFIG_FILE"
  echo -e "${GREEN}Updated log_level to debug in the configuration file.${NC}"
fi

# Create a test script to verify kalman filter functionality
echo -e "${YELLOW}Creating a test script to verify kalman filter functionality...${NC}"
TEST_SCRIPT="/tmp/test-kalman-filter.sh"
cat > "$TEST_SCRIPT" << 'EOF'
#!/bin/bash

# Test script to verify kalman filter functionality
echo "Starting kalman-proxy with debug logging..."
KALMAN_PROXY_PATH="$1"
CONFIG_FILE="$2"

# Start kalman-proxy with the configuration file
"$KALMAN_PROXY_PATH" "$CONFIG_FILE" > /tmp/kalman-proxy.log 2>&1 &
PROXY_PID=$!

echo "kalman-proxy started with PID: $PROXY_PID"
echo "Waiting for 5 seconds to allow startup..."
sleep 5

# Check if the process is still running
if kill -0 $PROXY_PID 2>/dev/null; then
  echo "kalman-proxy is running."
else
  echo "ERROR: kalman-proxy failed to start or crashed."
  echo "Check the log file at /tmp/kalman-proxy.log for details."
  exit 1
fi

# Check the log file for kalman filter initialization
echo "Checking logs for kalman filter initialization..."
if grep -q "Kalman filter initialized" /tmp/kalman-proxy.log; then
  echo "SUCCESS: Kalman filter was initialized!"
elif grep -q "kalman" /tmp/kalman-proxy.log; then
  echo "PARTIAL SUCCESS: Found references to kalman in the logs."
  echo "Log entries containing 'kalman':"
  grep -i "kalman" /tmp/kalman-proxy.log
else
  echo "WARNING: No evidence of Kalman filter initialization in the logs."
  echo "This may indicate that the Kalman filter is not being called."
fi

# Clean up
echo "Stopping kalman-proxy..."
kill $PROXY_PID
wait $PROXY_PID 2>/dev/null

echo "Test completed. Log file is available at /tmp/kalman-proxy.log"
EOF

chmod +x "$TEST_SCRIPT"

echo -e "${GREEN}Test script created at $TEST_SCRIPT${NC}"
echo -e "${YELLOW}Would you like to run the test now? (y/n)${NC}"
read -r RUN_TEST

if [[ "$RUN_TEST" =~ ^[Yy]$ ]]; then
  echo -e "${YELLOW}Running test script...${NC}"
  "$TEST_SCRIPT" "$KALMAN_PROXY_PATH" "$CONFIG_FILE"
  
  echo -e "${YELLOW}Would you like to view the full log? (y/n)${NC}"
  read -r VIEW_LOG
  
  if [[ "$VIEW_LOG" =~ ^[Yy]$ ]]; then
    less /tmp/kalman-proxy.log
  fi
fi

echo -e "${GREEN}Debug script completed.${NC}"
echo -e "${YELLOW}To start kalman-proxy with the updated configuration, run:${NC}"
echo -e "${GREEN}$KALMAN_PROXY_PATH $CONFIG_FILE${NC}"