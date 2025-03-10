package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	_ "image/jpeg"
	"log"
	"net"
	"os"
	"unsafe"

	"github.com/barnex/cuda5/cu"
	"github.com/nfnt/resize"
	"google.golang.org/grpc"

	pb "github.com/jeauchter/go-image-adjuster/proto"
)

// checkGPUAvailability checks if an NVIDIA GPU is available
func checkGPUAvailability() bool {
	// Initialize CUDA
	cu.Init(0)
	deviceCount := cu.DeviceGetCount()
	return deviceCount > 0
}

// getGPUDevices lists available GPU devices
func getGPUDevices() []string {
	// Initialize CUDA
	cu.Init(0)
	deviceCount := cu.DeviceGetCount()

	// Get device properties
	var devices []string
	for i := 0; i < deviceCount; i++ {
		device := cu.Device(i)
		name := device.Name()
		devices = append(devices, name)
	}

	return devices
}

// resizeImageCPU resizes an image using a CPU-based method
func resizeImageCPU(imageData []byte, width, height uint, quality int) ([]byte, error) {
	// Decode image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// Resize using CPU
	resizedImg := resize.Resize(width, height, img, resize.Lanczos3)

	// Encode resized image as JPEG
	var output bytes.Buffer
	err = jpeg.Encode(&output, resizedImg, &jpeg.Options{Quality: quality})
	if err != nil {
		return nil, fmt.Errorf("failed to encode resized image: %w", err)
	}

	return output.Bytes(), nil
}

// Decode the image on the CPU, converting it to NRGBA
func decodeToNRGBA(imageData []byte) (*image.NRGBA, error) {
	// check the size of imageData
	if len(imageData) == 0 {
		return nil, fmt.Errorf("image data is empty")
	}

	// First, check the format with DecodeConfig (optional but helpful)
	_, format, err := image.DecodeConfig(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("failed to read image config: %w", err)
	}
	if format != "jpeg" {
		return nil, fmt.Errorf("unsupported image format: %s", format)
	}

	// Then actually decode the full image bytes
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image on CPU: %w", err)
	}

	// Convert to NRGBA
	bounds := img.Bounds()
	nrgbaImg := image.NewNRGBA(bounds)
	draw.Draw(nrgbaImg, bounds, img, bounds.Min, draw.Src)
	return nrgbaImg, nil
}

// resizeImageGPU resizes the image using the GPU
func resizeImageGPU(imageData []byte, newWidth, newHeight, quality int) ([]byte, error) {
	// 1. Decode the image on the CPU
	cpuImg, err := decodeToNRGBA(imageData)
	if err != nil {
		return nil, err
	}

	// 2. Get pixel data in RGBA format
	oldWidth := cpuImg.Bounds().Dx()
	oldHeight := cpuImg.Bounds().Dy()
	rgbaBytes := cpuImg.Pix
	// Initialize CUDA
	cu.Init(0)

	// Create a device and context
	dev := cu.Device(0)
	ctx := cu.CtxCreate(0, dev)
	defer ctx.Destroy()

	// Allocate GPU memory for input and output images
	deviceInputImage := cu.MemAlloc(int64(len(rgbaBytes)))
	defer cu.MemFree(deviceInputImage)

	deviceOutputImage := cu.MemAlloc(int64(newWidth * newHeight * 4)) // Assuming 4 bytes per pixel (RGBA)
	defer cu.MemFree(deviceOutputImage)

	// Copy input image data to GPU
	cu.MemcpyHtoD(deviceInputImage, unsafe.Pointer(&rgbaBytes[0]), int64(len(rgbaBytes)))

	err = launchResizeKernel(deviceInputImage, oldWidth, oldHeight, newWidth, newHeight)
	if err != nil {
		return nil, fmt.Errorf("failed to launch resize kernel: %w", err)
	}

	// Copy resized image data back to host
	resizedImageData := make([]byte, newWidth*newHeight*4)
	cu.MemcpyDtoH(unsafe.Pointer(&resizedImageData[0]), deviceOutputImage, int64(len(resizedImageData)))

	// Convert raw image data to an image.NRGBA to satisfy jpeg.Encode
	img := &image.NRGBA{
		Pix:    resizedImageData,
		Stride: newWidth * 4,
		Rect:   image.Rect(0, 0, newWidth, newHeight),
	}

	// Encode resized image as JPEG
	var output bytes.Buffer
	err = jpeg.Encode(&output, img, &jpeg.Options{Quality: quality})
	if err != nil {
		return nil, fmt.Errorf("failed to encode resized image: %w", err)
	}

	return output.Bytes(), nil
}

// gRPC server implementation
type server struct {
	pb.UnimplementedImageResizerServer
}

func (s *server) ResizeImage(ctx context.Context, req *pb.ResizeImageRequest) (*pb.ResizeImageResponse, error) {
	// Check if the context is canceled before doing expensive work
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		// proceed
	}
	log.Println("Received resize request")

	// Check for GPU availability
	if checkGPUAvailability() {
		log.Println("Using GPU for resizing")
		resizedData, err := resizeImageGPU(req.GetImageData(), int(req.GetWidth()), int(req.GetHeight()), int(req.GetQuality()))
		if err == nil {
			log.Println("GPU resizing successful")
			return &pb.ResizeImageResponse{ResizedImage: resizedData, UsedGpu: true}, nil
		}
		log.Printf("GPU resizing failed, falling back to CPU: %v", err)
	}

	// Fallback to CPU if GPU is unavailable or fails
	resizedData, err := resizeImageCPU(req.GetImageData(), uint(req.GetWidth()), uint(req.GetHeight()), int(req.GetQuality()))
	if err != nil {
		log.Printf("CPU resize failed: %v", err)
		return nil, fmt.Errorf("CPU resize failed: %w", err)
	}
	log.Println("CPU resizing successful")
	return &pb.ResizeImageResponse{ResizedImage: resizedData, UsedGpu: false}, nil
}

// loadPTX loads the precompiled CUDA kernel
func loadPTX() ([]byte, error) {
	ptxFile := "./cuda/resize_kernel.ptx"
	return os.ReadFile(ptxFile)
}

func launchResizeKernel(deviceImage cu.DevicePtr, oldWidth, oldHeight, newWidth, newHeight int) error {
	// Load PTX file
	ptx, err := loadPTX()
	if err != nil {
		return fmt.Errorf("failed to load PTX file: %w", err)
	}

	// Load the CUDA module
	module := cu.ModuleLoadData(string(ptx))

	// Get the kernel function from the module
	kernel := module.GetFunction("resizeKernel")

	// Set up kernel parameters
	params := []unsafe.Pointer{
		unsafe.Pointer(&deviceImage),
		unsafe.Pointer(&oldWidth),
		unsafe.Pointer(&oldHeight),
		unsafe.Pointer(&newWidth),
		unsafe.Pointer(&newHeight),
	}

	// Define grid and block dimensions
	blockDimX, blockDimY := 16, 16
	gridDimX := (newWidth + blockDimX - 1) / blockDimX
	gridDimY := (newHeight + blockDimY - 1) / blockDimY

	// Launch the kernel
	cu.LaunchKernel(
		kernel,
		gridDimX, gridDimY, 1, // Grid dimensions
		blockDimX, blockDimY, 1, // Block dimensions
		0, cu.Stream(0), // Shared memory and stream
		params, // Kernel parameters
	)

	return nil

}

func main() {
	// Check if a GPU is available
	gpuAvailable := checkGPUAvailability()
	fmt.Println("GPU Available:", gpuAvailable)

	if gpuAvailable {
		gpus := getGPUDevices()
		fmt.Println("Available GPUs:", gpus)
	}

	// Start gRPC server
	listener, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterImageResizerServer(s, &server{})
	fmt.Println("gRPC server is running on port 50051")
	if err := s.Serve(listener); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
