package main

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"net"
	"os"
	"unsafe"

	"github.com/mumax/3/cuda"
	"github.com/nfnt/resize"
	"google.golang.org/grpc"

	pb "./proto"
)

// checkGPUAvailability checks if an NVIDIA GPU is available
func checkGPUAvailability() bool {
	// Initialize CUDA
	err := cuda.Init()
	if err != nil {
		log.Printf("CUDA initialization failed: %v", err)
		return false
	}

	// Get the number of CUDA-capable devices
	deviceCount, err := cuda.GetDeviceCount()
	if err != nil {
		log.Printf("Failed to get CUDA device count: %v", err)
		return false
	}

	// Return true if at least one CUDA-capable device is found
	return deviceCount > 0
}

// getGPUDevices lists available GPU devices

func getGPUDevices() []string {
	// Initialize CUDA
	err := cuda.Init()
	if err != nil {
		log.Printf("CUDA initialization failed: %v", err)
		return nil
	}

	// Get the number of CUDA-capable devices
	deviceCount, err := cuda.GetDeviceCount()
	if err != nil {
		log.Printf("Failed to get CUDA device count: %v", err)
		return nil
	}

	// Get device properties
	var devices []string
	for i := 0; i < deviceCount; i++ {
		device, err := cuda.GetDevice(i)
		if err != nil {
			log.Printf("Failed to get CUDA device %d: %v", i, err)
			continue
		}
		props, err := device.GetProperties()
		if err != nil {
			log.Printf("Failed to get properties for CUDA device %d: %v", i, err)
			continue
		}
		devices = append(devices, props.Name)
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

// resizeImageGPU resizes the image using the GPU
func resizeImageGPU(imageData []byte, newWidth, newHeight, quality int) ([]byte, error) {
	// Initialize CUDA
	err := cuda.Init()
	if err != nil {
		return nil, fmt.Errorf("CUDA initialization failed: %w", err)
	}

	// Allocate GPU memory for input and output images
	deviceInputImage, err := cuda.Malloc(len(imageData))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate GPU memory for input image: %w", err)
	}
	defer cuda.Free(deviceInputImage)

	deviceOutputImage, err := cuda.Malloc(newWidth * newHeight * 4) // Assuming 4 bytes per pixel (RGBA)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate GPU memory for output image: %w", err)
	}
	defer cuda.Free(deviceOutputImage)

	// Copy input image data to GPU
	err = cuda.MemcpyHtoD(deviceInputImage, imageData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input image data to GPU: %w", err)
	}

	// Launch the CUDA kernel to resize the image
	oldWidth, oldHeight := getImageDimensions(imageData) // Assuming a function to get image dimensions
	err = launchResizeKernel(deviceInputImage, oldWidth, oldHeight, newWidth, newHeight)
	if err != nil {
		return nil, fmt.Errorf("failed to launch resize kernel: %w", err)
	}

	// Copy resized image data back to host
	resizedImageData := make([]byte, newWidth*newHeight*4)
	err = cuda.MemcpyDtoH(resizedImageData, deviceOutputImage)
	if err != nil {
		return nil, fmt.Errorf("failed to copy resized image data from GPU: %w", err)
	}

	// Encode resized image as JPEG
	var output bytes.Buffer
	err = jpeg.Encode(&output, resizedImageData, &jpeg.Options{Quality: quality})
	if err != nil {
		return nil, fmt.Errorf("failed to encode resized image: %w", err)
	}

	return output.Bytes(), nil
}

// gRPC server implementation
type server struct {
	pb.UnimplementedImageResizerServer
}

func (s *server) ResizeImage(req *pb.ResizeImageRequest, stream pb.ImageResizer_ResizeImageServer) error {
	log.Println("Received resize request")

	// Check for GPU availability
	if checkGPUAvailability() {
		log.Println("Using GPU for resizing")
		resizedData, err := resizeImageGPU(req.GetImageData(), req.GetWidth(), req.GetHeight(), int(req.GetQuality()))
		if err == nil {
			log.Println("GPU resizing successful")
			return stream.Send(&pb.ResizeImageResponse{ResizedImage: resizedData, UsedGpu: true})
		}
		log.Printf("GPU resizing failed, falling back to CPU: %v", err)
	}

	// Fallback to CPU if GPU is unavailable or fails
	resizedData, err := resizeImageCPU(req.GetImageData(), req.GetWidth(), req.GetHeight(), int(req.GetQuality()))
	if err != nil {
		log.Printf("CPU resize failed: %v", err)
		return fmt.Errorf("CPU resize failed: %w", err)
	}
	log.Println("CPU resizing successful")
	return stream.Send(&pb.ResizeImageResponse{ResizedImage: resizedData, UsedGpu: false})
}

// loadPTX loads the precompiled CUDA kernel
func loadPTX() ([]byte, error) {
	ptxFile := "./cuda/resize_kernel.ptx"
	return os.ReadFile(ptxFile)
}

func launchResizeKernel(deviceImage cuda.DevicePtr, oldWidth, oldHeight, newWidth, newHeight int) error {
	// Load PTX file
	ptx, err := loadPTX()
	if err != nil {
		return fmt.Errorf("failed to load PTX file: %w", err)
	}

	// Load the CUDA module
	module, err := cuda.LoadModuleData(ptx)
	if err != nil {
		return fmt.Errorf("failed to load CUDA module: %w", err)
	}
	defer module.Unload()

	// Get the kernel function from the module
	kernel, err := module.GetFunction("resizeKernel")
	if err != nil {
		return fmt.Errorf("failed to get kernel function: %w", err)
	}

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
	err = kernel.Launch(
		gridDimX, gridDimY, 1, // Grid dimensions
		blockDimX, blockDimY, 1, // Block dimensions
		0, nil, // Shared memory and stream
		params, nil, // Kernel parameters
	)
	if err != nil {
		return fmt.Errorf("failed to launch kernel: %w", err)
	}

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
