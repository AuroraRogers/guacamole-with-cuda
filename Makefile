CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -g -I/usr/local/include -I/usr/include/guacamole
LDFLAGS = -L/usr/local/lib -lguac -lpthread -lcuda -lcudart

# CUDA flags
CUDA_INCLUDE = -I/usr/local/cuda/include
CUDA_LDFLAGS = -L/usr/local/cuda/lib64

all: kalman-proxy

kalman-proxy: kalman-middleware.o
	$(CC) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)

kalman-middleware.o: kalman-middleware.c
	$(CC) $(CFLAGS) $(CUDA_INCLUDE) -c $< -o $@

install: kalman-proxy
	install -m 755 kalman-proxy /usr/local/bin/
	install -m 644 kalman-proxy.conf /usr/local/etc/guacamole/

clean:
	rm -f kalman-proxy *.o

.PHONY: all install clean