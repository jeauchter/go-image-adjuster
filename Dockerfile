# Use NVIDIA's CUDA base image with Go
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \# Base image with CUDA support
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ make \
    golang \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy CUDA kernel source
COPY cuda/resize_kernel.cu /app/cuda/

# Compile CUDA kernel to PTX
RUN nvcc -ptx /app/cuda/resize_kernel.cu -o /app/cuda/resize_kernel.ptx

# Copy Go source code
COPY . .

# Build Go application
RUN go build -o gpu-image-resizer main.go

# Final runtime image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy compiled binary and PTX file from builder
COPY --from=builder /app/gpu-image-resizer .
COPY --from=builder /app/cuda/resize_kernel.ptx ./cuda/

# Expose gRPC server port
EXPOSE 50051

# Run the application
CMD ["./gpu-image-resizer"]

    protobuf-compiler \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Go
RUN curl -OL https://golang.org/dl/go1.21.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz && \
    rm go1.21.0.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
WORKDIR /app

# Install Go gRPC and protobuf plugins
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest \
    && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Copy project files
COPY . .

# Generate gRPC code from .proto files
RUN protoc --go_out=. --go-grpc_out=. proto/image_resizer.proto

# Build the Go application
RUN /usr/local/go/bin/go build -o gpu-image-resizer main.go

# Use NVIDIA runtime base image for execution
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /app/gpu-image-resizer .

# Expose gRPC port
EXPOSE 50051

# Set NVIDIA runtime for Docker
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run the application
CMD ["./gpu-image-resizer"]
