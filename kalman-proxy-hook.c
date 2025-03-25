#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>

#include "kalman_filter.h"

// Debug flag - set to 1 to enable debug logging
#define DEBUG 1

// Log function for debugging
#define LOG_DEBUG(fmt, ...) \
    do { if (DEBUG) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__); } while (0)

// Structure to hold instruction data
typedef struct {
    char* opcode;
    char** args;
    int arg_count;
} guac_instruction;

// Parse a Guacamole instruction from a string
guac_instruction* parse_instruction(const char* data) {
    guac_instruction* instruction = malloc(sizeof(guac_instruction));
    if (!instruction) {
        perror("Failed to allocate memory for instruction");
        return NULL;
    }
    
    // Initialize instruction
    instruction->opcode = NULL;
    instruction->args = NULL;
    instruction->arg_count = 0;
    
    // Make a copy of the data to tokenize
    char* data_copy = strdup(data);
    if (!data_copy) {
        perror("Failed to duplicate data string");
        free(instruction);
        return NULL;
    }
    
    // Count the number of dots (separators)
    int dot_count = 0;
    for (const char* c = data; *c != '\0'; c++) {
        if (*c == '.') dot_count++;
    }
    
    // Allocate memory for arguments (dot_count + 1 for opcode)
    instruction->args = malloc((dot_count + 1) * sizeof(char*));
    if (!instruction->args) {
        perror("Failed to allocate memory for args");
        free(data_copy);
        free(instruction);
        return NULL;
    }
    
    // Parse the opcode and arguments
    char* token = strtok(data_copy, ".");
    int i = 0;
    
    while (token != NULL) {
        // Get the length of the token
        size_t token_len = strlen(token);
        
        // Parse the length prefix
        char* endptr;
        long length = strtol(token, &endptr, 10);
        
        // Check if the token starts with a valid length
        if (endptr != token && *endptr == '.') {
            // Extract the actual value
            char* value = endptr + 1;
            
            // Allocate memory for the value
            instruction->args[i] = malloc(length + 1);
            if (!instruction->args[i]) {
                perror("Failed to allocate memory for arg");
                // Clean up
                for (int j = 0; j < i; j++) {
                    free(instruction->args[j]);
                }
                free(instruction->args);
                free(data_copy);
                free(instruction);
                return NULL;
            }
            
            // Copy the value
            strncpy(instruction->args[i], value, length);
            instruction->args[i][length] = '\0';
            
            // If this is the first token, it's the opcode
            if (i == 0) {
                instruction->opcode = instruction->args[i];
            }
            
            i++;
        }
        
        // Get the next token
        token = strtok(NULL, ".");
    }
    
    instruction->arg_count = i;
    
    free(data_copy);
    return instruction;
}

// Free the memory used by an instruction
void free_instruction(guac_instruction* instruction) {
    if (instruction) {
        for (int i = 0; i < instruction->arg_count; i++) {
            free(instruction->args[i]);
        }
        free(instruction->args);
        free(instruction);
    }
}

// Apply Kalman filter to image data
void apply_kalman_filter_to_image(unsigned char* data, int width, int height, int channels) {
    LOG_DEBUG("Applying Kalman filter to image: %dx%d with %d channels", width, height, channels);
    
    // Initialize Kalman filter
    KalmanFilter* kf = init_cuda_kalman_filter(channels, channels);
    if (!kf) {
        LOG_DEBUG("Failed to initialize Kalman filter");
        return;
    }
    
    // Process each pixel with Kalman filter
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Get pixel data
            float pixel[channels];
            for (int c = 0; c < channels; c++) {
                pixel[c] = (float)data[(y * width + x) * channels + c];
            }
            
            // Apply Kalman filter
            cuda_kalman_filter_predict(kf);
            cuda_kalman_filter_update(kf, pixel);
            
            // Update pixel data
            for (int c = 0; c < channels; c++) {
                data[(y * width + x) * channels + c] = (unsigned char)pixel[c];
            }
        }
    }
    
    // Free Kalman filter
    free_cuda_kalman_filter(kf);
    
    LOG_DEBUG("Kalman filter applied successfully");
}

// Process an img instruction
void process_img_instruction(guac_instruction* instruction) {
    LOG_DEBUG("Processing img instruction with %d args", instruction->arg_count);
    
    // Check if this is an img instruction
    if (strcmp(instruction->opcode, "img") != 0) {
        LOG_DEBUG("Not an img instruction: %s", instruction->opcode);
        return;
    }
    
    // Check if we have enough arguments
    if (instruction->arg_count < 7) {
        LOG_DEBUG("Not enough arguments for img instruction: %d", instruction->arg_count);
        return;
    }
    
    // Extract image data and dimensions
    int width = atoi(instruction->args[4]);
    int height = atoi(instruction->args[5]);
    
    // The image data is in the last argument
    char* image_data = instruction->args[6];
    int data_length = strlen(image_data);
    
    LOG_DEBUG("Image dimensions: %dx%d, data length: %d", width, height, data_length);
    
    // Apply Kalman filter to image data
    // Note: In a real implementation, you would need to decode the image data
    // (which might be base64 encoded) and then apply the filter
    
    // For demonstration purposes, we'll just log that we would apply the filter
    LOG_DEBUG("Would apply Kalman filter to image data here");
    
    // In a real implementation:
    // 1. Decode the image data
    // 2. Apply the Kalman filter
    // 3. Re-encode the image data
    // 4. Update the instruction with the new image data
}

// Hook function to intercept and process instructions
void* process_instructions(void* arg) {
    int client_socket = *((int*)arg);
    free(arg);
    
    char buffer[4096];
    ssize_t bytes_read;
    
    LOG_DEBUG("Started processing instructions for client socket %d", client_socket);
    
    while ((bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0)) > 0) {
        // Null-terminate the received data
        buffer[bytes_read] = '\0';
        
        LOG_DEBUG("Received %zd bytes: %s", bytes_read, buffer);
        
        // Parse the instruction
        guac_instruction* instruction = parse_instruction(buffer);
        if (instruction) {
            LOG_DEBUG("Parsed instruction: %s with %d args", 
                     instruction->opcode, instruction->arg_count);
            
            // Check if this is an img instruction
            if (strcmp(instruction->opcode, "img") == 0) {
                LOG_DEBUG("Found img instruction!");
                process_img_instruction(instruction);
            }
            
            free_instruction(instruction);
        }
        
        // Forward the data to the Guacamole server
        // In a real implementation, you would modify the data if needed
        send(client_socket, buffer, bytes_read, 0);
    }
    
    LOG_DEBUG("Client socket %d closed", client_socket);
    close(client_socket);
    return NULL;
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }
    
    // Read configuration file
    FILE* config_file = fopen(argv[1], "r");
    if (!config_file) {
        perror("Failed to open config file");
        return 1;
    }
    
    // Parse configuration
    char line[256];
    int listen_port = 4822;  // Default port
    char guacd_host[256] = "127.0.0.1";
    int guacd_port = 4822;
    
    while (fgets(line, sizeof(line), config_file)) {
        // Remove newline
        line[strcspn(line, "\n")] = '\0';
        
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\0') {
            continue;
        }
        
        // Parse key-value pairs
        char* key = strtok(line, "=");
        char* value = strtok(NULL, "=");
        
        if (key && value) {
            // Trim whitespace
            while (*key && isspace(*key)) key++;
            while (*value && isspace(*value)) value++;
            
            if (strcmp(key, "listen_port") == 0) {
                listen_port = atoi(value);
            } else if (strcmp(key, "guacd_host") == 0) {
                strncpy(guacd_host, value, sizeof(guacd_host) - 1);
            } else if (strcmp(key, "guacd_port") == 0) {
                guacd_port = atoi(value);
            }
        }
    }
    
    fclose(config_file);
    
    // Create socket
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("Failed to create socket");
        return 1;
    }
    
    // Set socket options
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("Failed to set socket options");
        return 1;
    }
    
    // Bind socket
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(listen_port);
    
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Failed to bind socket");
        return 1;
    }
    
    // Listen for connections
    if (listen(server_socket, 5) < 0) {
        perror("Failed to listen on socket");
        return 1;
    }
    
    printf("Kalman proxy listening on port %d\n", listen_port);
    printf("Forwarding to guacd at %s:%d\n", guacd_host, guacd_port);
    
    // Accept connections
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) {
            perror("Failed to accept connection");
            continue;
        }
        
        printf("Accepted connection from %s:%d\n", 
               inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
        
        // Create a thread to handle the connection
        pthread_t thread;
        int* client_socket_ptr = malloc(sizeof(int));
        if (!client_socket_ptr) {
            perror("Failed to allocate memory for client socket");
            close(client_socket);
            continue;
        }
        
        *client_socket_ptr = client_socket;
        
        if (pthread_create(&thread, NULL, process_instructions, client_socket_ptr) != 0) {
            perror("Failed to create thread");
            free(client_socket_ptr);
            close(client_socket);
            continue;
        }
        
        // Detach the thread
        pthread_detach(thread);
    }
    
    return 0;
}