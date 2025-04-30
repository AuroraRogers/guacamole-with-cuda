#!/bin/bash

# Script to fix the issue with kalman-filter not being invoked
# This script will:
# 1. Check if the kalman filter is properly enabled in the configuration
# 2. Verify that the proxy is correctly configured to use the filter
# 3. Add instrumentation to verify filter activation
# 4. Restart the proxy with proper settings

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we have root privileges
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}Please run as root for full functionality${NC}"
  exit 1
fi

# Get the configuration file path from command line argument
if [ "$#" -ne 1 ]; then
  echo -e "${YELLOW}Usage: $0 /path/to/kalman-proxy.conf${NC}"
  
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

# Check for image processing settings
echo -e "${YELLOW}Checking for image processing settings...${NC}"
if ! grep -q "process_images" "$CONFIG_FILE"; then
  echo "process_images = true" >> "$CONFIG_FILE"
  echo -e "${GREEN}Added process_images=true to the configuration file.${NC}"
else
  # Replace the existing process_images line
  sed -i 's/process_images\s*=\s*.*/process_images = true/' "$CONFIG_FILE"
  echo -e "${GREEN}Updated process_images to true in the configuration file.${NC}"
fi

# Check for cursor movement settings
echo -e "${YELLOW}Checking for cursor movement settings...${NC}"
if ! grep -q "smooth_cursor" "$CONFIG_FILE"; then
  echo "smooth_cursor = true" >> "$CONFIG_FILE"
  echo -e "${GREEN}Added smooth_cursor=true to the configuration file.${NC}"
else
  # Replace the existing smooth_cursor line
  sed -i 's/smooth_cursor\s*=\s*.*/smooth_cursor = true/' "$CONFIG_FILE"
  echo -e "${GREEN}Updated smooth_cursor to true in the configuration file.${NC}"
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

# Check if kalman-proxy is already running
if [ ! -z "$LISTEN_PORT" ]; then
  RUNNING_PID=$(lsof -i :$LISTEN_PORT -t 2>/dev/null)
  if [ ! -z "$RUNNING_PID" ]; then
    echo -e "${YELLOW}kalman-proxy is already running with PID: $RUNNING_PID${NC}"
    
    # Check if it's actually kalman-proxy
    PROCESS_NAME=$(ps -p $RUNNING_PID -o comm= 2>/dev/null)
    if [[ "$PROCESS_NAME" == *"kalman"* ]]; then
      echo -e "${GREEN}Confirmed it's a kalman-proxy process.${NC}"
      echo -e "${YELLOW}Stopping the current process to apply new settings...${NC}"
      kill -9 $RUNNING_PID
      sleep 2
    else
      echo -e "${RED}WARNING: Port $LISTEN_PORT is being used by another process: $PROCESS_NAME${NC}"
      echo -e "${YELLOW}Killing this process to free the port...${NC}"
      kill -9 $RUNNING_PID
      sleep 2
    fi
  else
    echo -e "${YELLOW}No process is currently using port $LISTEN_PORT.${NC}"
  fi
fi

# Create a wrapper script to monitor filter activation
echo -e "${YELLOW}Creating a wrapper script to monitor filter activation...${NC}"
WRAPPER_SCRIPT="/usr/local/bin/kalman-proxy-wrapper.sh"
cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash

# Wrapper script to monitor kalman filter activation
CONFIG_FILE="$1"
LOG_FILE="/var/log/kalman-proxy.log"

# Start kalman-proxy with the configuration file
echo "Starting kalman-proxy with configuration file: $CONFIG_FILE"
echo "Logging to: $LOG_FILE"

# Run kalman-proxy in the background
/usr/local/bin/kalman-proxy "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
PROXY_PID=$!

echo "kalman-proxy started with PID: $PROXY_PID"
echo "Monitoring logs for filter activation..."

# Wait for 10 seconds to allow startup
sleep 10

# Check if the process is still running
if kill -0 $PROXY_PID 2>/dev/null; then
  echo "kalman-proxy is running."
else
  echo "ERROR: kalman-proxy failed to start or crashed."
  echo "Check the log file at $LOG_FILE for details."
  exit 1
fi

# Monitor the log file for filter activation
tail -f "$LOG_FILE" | grep --line-buffered -i "kalman" &
TAIL_PID=$!

# Keep the script running
echo "Press Ctrl+C to stop monitoring and exit"
trap "kill $TAIL_PID; echo 'Stopping monitoring...'; exit 0" INT

# Wait for the proxy process to end
wait $PROXY_PID
EOF

chmod +x "$WRAPPER_SCRIPT"
echo -e "${GREEN}Wrapper script created at $WRAPPER_SCRIPT${NC}"

# Start the proxy with the wrapper script
echo -e "${YELLOW}Starting kalman-proxy with the wrapper script...${NC}"
echo -e "${GREEN}$WRAPPER_SCRIPT $CONFIG_FILE${NC}"
echo -e "${YELLOW}This will start the proxy and monitor for filter activation.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop monitoring.${NC}"

# Execute the wrapper script
"$WRAPPER_SCRIPT" "$CONFIG_FILE"