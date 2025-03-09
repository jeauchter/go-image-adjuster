extern "C" __global__
void resizeKernel(unsigned char* input, int inWidth, int inHeight, unsigned char* output, int outWidth, int outHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outWidth && y < outHeight) {
        int srcX = (x * inWidth) / outWidth;
        int srcY = (y * inHeight) / outHeight;
        int srcIdx = (srcY * inWidth + srcX) * 4; // Assuming RGBA
        int dstIdx = (y * outWidth + x) * 4;

        output[dstIdx] = input[srcIdx];      // R
        output[dstIdx + 1] = input[srcIdx + 1];  // G
        output[dstIdx + 2] = input[srcIdx + 2];  // B
        output[dstIdx + 3] = input[srcIdx + 3];  // A
    }
}
