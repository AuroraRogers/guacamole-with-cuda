#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <libguac/kalman_filter.h>
#include <libguac/video_filter.h>

#define MAX_BUFFER_SIZE 16384
#define DEFAULT_PORT 4822
#define DEFAULT_GUACD_HOST "127.0.0.1"
#define DEFAULT_GUACD_PORT 4823

// Configuration structure
typedef struct {
    int listen_port;
    char guacd_host[256];
    int guacd_port;
    int debug_level;
} KalmanProxyConfig;

// Client connection state
typedef struct {
    int client_fd;
    int guacd_fd;
    pthread_t thread;
    int active;
    VideoFilter* video_filter;
} ClientConnection;

// Global variables
KalmanProxyConfig config;
int server_fd = -1;
int running = 1;
ClientConnection* connections = NULL;
int max_connections = 0;

// Function prototypes
void signal_handler(int sig);
void cleanup();
int load_config(const char* config_file);
int setup_server();
int accept_client();
int connect_to_guacd();
void* handle_client(void* arg);
int process_guacd_message(ClientConnection* conn, char* buffer, int length);
int process_client_message(ClientConnection* conn, char* buffer, int length);
int apply_kalman_filter(unsigned char* image_data, int width, int height, int channels);

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    printf("Received signal %d, shutting down...\n", sig);
    running = 0;
}

// Cleanup resources
void cleanup() {
    if (server_fd >= 0) {
        close(server_fd);
    }
    
    // Cleanup client connections
    if (connections) {
        for (int i = 0; i < max_connections; i++) {
            if (connections[i].active) {
                connections[i].active = 0;
                pthread_join(connections[i].thread, NULL);
                close(connections[i].client_fd);
                close(connections[i].guacd_fd);
                if (connections[i].video_filter) {
                    free_video_filter(connections[i].video_filter);
                }
            }
        }
        free(connections);
    }
}

// Load configuration from file
int load_config(const char* config_file) {
    FILE* file = fopen(config_file, "r");
    if (!file) {
        fprintf(stderr, "Failed to open config file: %s\n", config_file);
        return -1;
    }
    
    // Set defaults
    config.listen_port = DEFAULT_PORT;
    strcpy(config.guacd_host, DEFAULT_GUACD_HOST);
    config.guacd_port = DEFAULT_GUACD_PORT;
    config.debug_level = 0;
    
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        // Remove comments and trailing whitespace
        char* comment = strchr(line, '#');
        if (comment) *comment = '\0';
        
        // Trim trailing whitespace
        int len = strlen(line);
        while (len > 0 && (line[len-1] == ' ' || line[len-1] == '\t' || 
               line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        
        if (len == 0) continue;
        
        // Parse key-value pairs
        char key[128], value[128];
        if (sscanf(line, "%127[^=]=%127s", key, value) == 2) {
            // Trim leading/trailing whitespace from key
            char* k = key;
            while (*k == ' ' || *k == '\t') k++;
            char* end = k + strlen(k) - 1;
            while (end > k && (*end == ' ' || *end == '\t')) *end-- = '\0';
            
            if (strcmp(k, "listen_port") == 0) {
                config.listen_port = atoi(value);
            } else if (strcmp(k, "guacd_host") == 0) {
                strncpy(config.guacd_host, value, sizeof(config.guacd_host) - 1);
            } else if (strcmp(k, "guacd_port") == 0) {
                config.guacd_port = atoi(value);
            } else if (strcmp(k, "debug_level") == 0) {
                config.debug_level = atoi(value);
            }
        }
    }
    
    fclose(file);
    
    printf("Configuration loaded:\n");
    printf("  listen_port: %d\n", config.listen_port);
    printf("  guacd_host: %s\n", config.guacd_host);
    printf("  guacd_port: %d\n", config.guacd_port);
    printf("  debug_level: %d\n", config.debug_level);
    
    return 0;
}

// Set up the server socket
int setup_server() {
    struct sockaddr_in server_addr;
    
    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Failed to create socket");
        return -1;
    }
    
    // Set socket options
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("Failed to set socket options");
        close(server_fd);
        return -1;
    }
    
    // Bind socket
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(config.listen_port);
    
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Failed to bind socket");
        close(server_fd);
        return -1;
    }
    
    // Listen for connections
    if (listen(server_fd, 10) < 0) {
        perror("Failed to listen on socket");
        close(server_fd);
        return -1;
    }
    
    printf("Server listening on port %d\n", config.listen_port);
    
    // Initialize client connections array
    max_connections = 100;  // Maximum number of simultaneous connections
    connections = calloc(max_connections, sizeof(ClientConnection));
    if (!connections) {
        perror("Failed to allocate memory for connections");
        close(server_fd);
        return -1;
    }
    
    return 0;
}

// Accept a client connection
int accept_client() {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) {
        if (errno == EINTR) {
            // Interrupted by signal, not an error
            return -1;
        }
        perror("Failed to accept connection");
        return -1;
    }
    
    printf("Client connected from %s:%d\n", 
           inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
    
    return client_fd;
}

// Connect to guacd
int connect_to_guacd() {
    struct sockaddr_in guacd_addr;
    int guacd_fd;
    
    // Create socket
    guacd_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (guacd_fd < 0) {
        perror("Failed to create socket for guacd connection");
        return -1;
    }
    
    // Connect to guacd
    memset(&guacd_addr, 0, sizeof(guacd_addr));
    guacd_addr.sin_family = AF_INET;
    guacd_addr.sin_port = htons(config.guacd_port);
    
    if (inet_pton(AF_INET, config.guacd_host, &guacd_addr.sin_addr) <= 0) {
        perror("Invalid guacd address");
        close(guacd_fd);
        return -1;
    }
    
    if (connect(guacd_fd, (struct sockaddr*)&guacd_addr, sizeof(guacd_addr)) < 0) {
        perror("Failed to connect to guacd");
        close(guacd_fd);
        return -1;
    }
    
    printf("Connected to guacd at %s:%d\n", config.guacd_host, config.guacd_port);
    
    return guacd_fd;
}

// Parse Guacamole protocol elements
int parse_guacamole_instruction(const char* buffer, char** elements, int max_elements) {
    int count = 0;
    const char* start = buffer;
    
    while (*start && count < max_elements) {
        // Parse element length
        char* end;
        long length = strtol(start, &end, 10);
        
        if (end == start || *end != '.') {
            // Invalid format
            return -1;
        }
        
        // Skip the dot
        start = end + 1;
        
        // Store the element
        elements[count++] = (char*)start;
        
        // Move to the next element
        start += length;
        
        // Check for separator or end
        if (*start == ',') {
            // Null-terminate the current element
            *(char*)start = '\0';
            start++;
        } else if (*start == ';') {
            // Null-terminate the current element
            *(char*)start = '\0';
            break;
        } else {
            // Invalid format
            return -1;
        }
    }
    
    return count;
}

// Process messages from guacd
int process_guacd_message(ClientConnection* conn, char* buffer, int length) {
    // Just forward the message to the client for now
    if (config.debug_level > 1) {
        printf("From guacd: %.*s\n", length, buffer);
    }
    
    return send(conn->client_fd, buffer, length, 0);
}

// Apply Kalman filter to image data
int apply_kalman_filter(unsigned char* image_data, int width, int height, int channels) {
    // Initialize video filter if not already done
    static VideoFilter* filter = NULL;
    if (!filter) {
        filter = init_video_filter(width, height, channels);
        if (!filter) {
            fprintf(stderr, "Failed to initialize video filter\n");
            return -1;
        }
    }
    
    // Process the frame with the Kalman filter
    process_frame(filter, image_data);
    apply_filter_to_frame(filter);
    
    // Copy filtered data back to the original buffer
    // (In a real implementation, you would need to handle this properly)
    
    return 0;
}

// Process messages from the client
int process_client_message(ClientConnection* conn, char* buffer, int length) {
    if (config.debug_level > 1) {
        printf("From client: %.*s\n", length, buffer);
    }
    
    // Parse the Guacamole instruction
    char* elements[16];
    int count = parse_guacamole_instruction(buffer, elements, 16);
    
    if (count > 0 && strcmp(elements[0], "img") == 0) {
        // This is an img instruction, which we want to intercept and apply the Kalman filter
        if (config.debug_level > 0) {
            printf("Intercepted img instruction\n");
        }
        
        // In a real implementation, you would:
        // 1. Parse the image data
        // 2. Apply the Kalman filter
        // 3. Replace the image data in the instruction
        
        // For now, we'll just log that we found an img instruction
        printf("Found img instruction: %s\n", buffer);
    }
    
    // Forward the message to guacd
    return send(conn->guacd_fd, buffer, length, 0);
}

// Handle client connection in a separate thread
void* handle_client(void* arg) {
    ClientConnection* conn = (ClientConnection*)arg;
    char buffer[MAX_BUFFER_SIZE];
    fd_set read_fds;
    int max_fd = (conn->client_fd > conn->guacd_fd) ? conn->client_fd : conn->guacd_fd;
    
    while (conn->active) {
        FD_ZERO(&read_fds);
        FD_SET(conn->client_fd, &read_fds);
        FD_SET(conn->guacd_fd, &read_fds);
        
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        
        int activity = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);
        
        if (activity < 0) {
            if (errno == EINTR) continue;
            perror("Select error");
            break;
        }
        
        // Check for data from client
        if (FD_ISSET(conn->client_fd, &read_fds)) {
            int bytes_read = recv(conn->client_fd, buffer, sizeof(buffer) - 1, 0);
            if (bytes_read <= 0) {
                if (bytes_read < 0) perror("Client read error");
                else printf("Client disconnected\n");
                break;
            }
            
            buffer[bytes_read] = '\0';
            if (process_client_message(conn, buffer, bytes_read) < 0) {
                perror("Failed to forward client message");
                break;
            }
        }
        
        // Check for data from guacd
        if (FD_ISSET(conn->guacd_fd, &read_fds)) {
            int bytes_read = recv(conn->guacd_fd, buffer, sizeof(buffer) - 1, 0);
            if (bytes_read <= 0) {
                if (bytes_read < 0) perror("guacd read error");
                else printf("guacd disconnected\n");
                break;
            }
            
            buffer[bytes_read] = '\0';
            if (process_guacd_message(conn, buffer, bytes_read) < 0) {
                perror("Failed to forward guacd message");
                break;
            }
        }
    }
    
    // Cleanup
    conn->active = 0;
    close(conn->client_fd);
    close(conn->guacd_fd);
    if (conn->video_filter) {
        free_video_filter(conn->video_filter);
        conn->video_filter = NULL;
    }
    
    printf("Client handler thread exiting\n");
    return NULL;
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Load configuration
    if (load_config(argv[1]) < 0) {
        return 1;
    }
    
    // Set up server
    if (setup_server() < 0) {
        return 1;
    }
    
    printf("Kalman filter middleware started\n");
    
    // Main loop
    while (running) {
        // Accept new client
        int client_fd = accept_client();
        if (client_fd < 0) {
            if (!running) break;
            continue;
        }
        
        // Connect to guacd
        int guacd_fd = connect_to_guacd();
        if (guacd_fd < 0) {
            close(client_fd);
            continue;
        }
        
        // Find an available connection slot
        int slot = -1;
        for (int i = 0; i < max_connections; i++) {
            if (!connections[i].active) {
                slot = i;
                break;
            }
        }
        
        if (slot < 0) {
            fprintf(stderr, "Maximum connections reached\n");
            close(client_fd);
            close(guacd_fd);
            continue;
        }
        
        // Initialize connection
        connections[slot].client_fd = client_fd;
        connections[slot].guacd_fd = guacd_fd;
        connections[slot].active = 1;
        connections[slot].video_filter = NULL;
        
        // Create thread to handle client
        if (pthread_create(&connections[slot].thread, NULL, handle_client, &connections[slot]) != 0) {
            perror("Failed to create thread");
            connections[slot].active = 0;
            close(client_fd);
            close(guacd_fd);
            continue;
        }
        
        // Detach thread
        pthread_detach(connections[slot].thread);
    }
    
    // Cleanup
    cleanup();
    
    printf("Kalman filter middleware stopped\n");
    
    return 0;
}