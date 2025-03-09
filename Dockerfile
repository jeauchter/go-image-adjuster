# Base image with CUDA support
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ make \
    golang \
    protobuf-compiler \
    wget \
    && rm -rf /var/lib/apt/lists/*
# Install Go 1.24.1
RUN wget https://go.dev/dl/go1.24.1.linux-amd64.tar.gz && \
    rm -rf /usr/local/go && \
    tar -C /usr/local -xzf go1.24.1.linux-amd64.tar.gz && \
    rm go1.24.1.linux-amd64.tar.gz

# Set up Go environment
ENV PATH="/usr/local/go/bin:$PATH"
ENV GOPATH="/root/go"
ENV PATH="$GOPATH/bin:$PATH"

# Verify Go version (ensures correct installation)
RUN go version

# Install protoc Go plugins
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Set PATH to include Go binaries
ENV PATH="/root/go/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy Go modules
COPY go.mod go.sum ./
RUN go mod download

# Copy the whole project
COPY . .

# Compile protobuf files
RUN protoc --go_out=. --go-grpc_out=. proto/*.proto

# Compile CUDA kernel to PTX
RUN nvcc -ptx cuda/resize_kernel.cu -o cuda/resize_kernel.ptx

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
