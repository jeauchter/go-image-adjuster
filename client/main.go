package main

import (
	"context"
	"log"
	"os"
	"time"

	pb "github.com/jeauchter/go-image-adjuster/proto"
	"google.golang.org/grpc"
)

func main() {
	// Get the gRPC server address from the environment variable
	serverAddress := os.Getenv("GRPC_SERVER_ADDRESS")
	if serverAddress == "" {
		serverAddress = "localhost:50051"
	}
	// Read image from a file inside the container
	imagePath := "/client/test_image/Test-image.jpeg"
	// check that image exists
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		log.Fatalf("image file does not exist: %v", err)
	}
	// Read image data
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		log.Fatalf("failed to read input image: %v", err)
	}

	// check size of image
	log.Printf("Image size: %d bytes", len(imageData))

	// Create a context with a timeout for the connection
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	// Connect to the gRPC server
	conn, err := grpc.DialContext(ctx, serverAddress, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	client := pb.NewImageResizerClient(conn)

	// Prepare the request
	req := &pb.ResizeImageRequest{
		ImageData: imageData, // Add your image data here
		Width:     800,
		Height:    600,
		Quality:   90,
	}

	// Send the request
	ctx, cancel = context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	res, err := client.ResizeImage(ctx, req)
	if err != nil {
		log.Fatalf("could not resize image: %v", err)
	}

	// Handle the response
	log.Printf("Resized image size: %d bytes", len(res.ResizedImage))
	log.Printf("Used GPU: %v", res.UsedGpu)
	if res.ErrorMessage != "" {
		log.Printf("Error: %s", res.ErrorMessage)
	}
}
