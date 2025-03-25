CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -g -O2
LDFLAGS = -lpthread

# Path to Guacamole includes and libraries
GUAC_INCDIR = /usr/local/include/guacamole
GUAC_LIBDIR = /usr/local/lib

# Path to CUDA includes and libraries
CUDA_INCDIR = /usr/local/cuda/include
CUDA_LIBDIR = /usr/local/cuda/lib64

# Include directories
INCLUDES = -I$(GUAC_INCDIR) -I$(CUDA_INCDIR) -I.

# Libraries
LIBS = -L$(GUAC_LIBDIR) -lguac -L$(CUDA_LIBDIR) -lcudart

# Source files
SRCS = kalman-proxy-hook.c
CUDA_SRCS = kalman_filter.cu

# Object files
OBJS = $(SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Target executable
TARGET = kalman-proxy

all: $(TARGET)

$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f $(OBJS) $(CUDA_OBJS) $(TARGET)

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/
	mkdir -p /usr/local/etc/guacamole
	[ -f /usr/local/etc/guacamole/kalman-proxy.conf ] || \
		install -m 644 kalman-proxy.conf /usr/local/etc/guacamole/

.PHONY: all clean install