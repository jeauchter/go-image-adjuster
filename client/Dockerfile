FROM golang:1.24

# Set working directory
WORKDIR /app

# Copy Go modules
COPY go.mod go.sum ./
RUN go mod download

# Copy the whole project
COPY . .

# Build the client application
RUN go build -o image-resizer-client main.go

# Run the client application
CMD ["./image-resizer-client"]