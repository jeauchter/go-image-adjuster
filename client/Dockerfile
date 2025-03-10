FROM golang:1.24

WORKDIR /app

# Copy module files from root of project
COPY go.mod go.sum ./
RUN go mod download

# Copy the proto folder from root of project so the client can import it
COPY proto ./proto

# Copy the client code
COPY client ./client

# Build the client app
RUN go build -o image-resizer-client ./client/main.go

CMD ["./image-resizer-client"]