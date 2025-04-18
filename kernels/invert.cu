extern "C" __global__ void invert(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        // Invert RGB channels
        output[idx] = 255 - input[idx];
        output[idx + 1] = 255 - input[idx + 1];
        output[idx + 2] = 255 - input[idx + 2];
        output[idx + 3] = input[idx + 3];  // Preserve alpha
    }
}
