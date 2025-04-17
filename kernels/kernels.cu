/* 
 * Sample CUDA kernels for use with the GStreamer cudakernel plugin
 *
 * This file contains several example kernels that can be compiled to PTX/CUBIN
 * and loaded by the cudakernel plugin at runtime.
 *
 * Compile with:
 *   nvcc -ptx -arch=sm_52 -o kernels.ptx kernels.cu
 */

/**
 * Basic image processing kernels
 * Each follows the same parameter pattern expected by the cudakernel plugin:
 * 
 * Parameters:
 * - input: Input image (RGBA)
 * - output: Output image (RGBA)
 * - width: Image width
 * - height: Image height
 * - [additional parameters specific to the kernel]
 */

/**
 * Simple grayscale conversion
 */
extern "C" __global__ void grayscale(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        // Read RGB components
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        unsigned char a = input[idx + 3];
        
        // Convert to grayscale using luminance formula
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        
        // Write grayscale value to all RGB channels
        output[idx] = gray;
        output[idx + 1] = gray;
        output[idx + 2] = gray;
        output[idx + 3] = a;  // Preserve alpha
    }
}

/**
 * Brightness adjustment
 * Parameters:
 * - factor: Brightness multiplier (1.0 = unchanged, >1.0 = brighter, <1.0 = darker)
 */
extern "C" __global__ void brightness(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    float factor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        // Apply brightness factor with clamping
        output[idx] = min(255, max(0, (int)(input[idx] * factor)));
        output[idx + 1] = min(255, max(0, (int)(input[idx + 1] * factor)));
        output[idx + 2] = min(255, max(0, (int)(input[idx + 2] * factor)));
        output[idx + 3] = input[idx + 3];  // Preserve alpha
    }
}

/**
 * Color inversion
 */
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

/**
 * Gaussian blur (simple version without shared memory optimization)
 * Parameters:
 * - radius: Blur radius
 */
extern "C" __global__ void blur(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float r_sum = 0, g_sum = 0, b_sum = 0;
        int count = 0;
        
        // Simple box blur
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int idx = (ny * width + nx) * 4;
                    r_sum += input[idx];
                    g_sum += input[idx + 1];
                    b_sum += input[idx + 2];
                    count++;
                }
            }
        }
        
        // Write averaged values
        int idx = (y * width + x) * 4;
        output[idx] = (unsigned char)(r_sum / count);
        output[idx + 1] = (unsigned char)(g_sum / count);
        output[idx + 2] = (unsigned char)(b_sum / count);
        output[idx + 3] = input[idx + 3];  // Preserve alpha
    }
}

/**
 * Edge detection using Sobel operator
 * Parameters:
 * - threshold: Edge detection threshold (0-255)
 */
extern "C" __global__ void edge_detect(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        // Convert neighborhood to grayscale
        float gray[3][3];
        
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int idx = ((y+j) * width + (x+i)) * 4;
                float r = input[idx];
                float g = input[idx + 1];
                float b = input[idx + 2];
                gray[j+1][i+1] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }
        
        // Sobel operators
        float gx = gray[0][0] - gray[0][2] + 2*gray[1][0] - 2*gray[1][2] + gray[2][0] - gray[2][2];
        float gy = gray[0][0] + 2*gray[0][1] + gray[0][2] - gray[2][0] - 2*gray[2][1] - gray[2][2];
        
        // Gradient magnitude
        float mag = sqrtf(gx*gx + gy*gy);
        
        // Thresholding
        int idx = (y * width + x) * 4;
        unsigned char edge = (mag > threshold) ? 255 : 0;
        
        output[idx] = edge;
        output[idx + 1] = edge;
        output[idx + 2] = edge;
        output[idx + 3] = input[idx + 3];  // Preserve alpha
    }
    else if (x < width && y < height) {
        // Border pixels are set to black
        int idx = (y * width + x) * 4;
        output[idx] = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 0;
        output[idx + 3] = input[idx + 3];  // Preserve alpha
    }
}

/**
 * Chroma key (green screen)
 * Parameters:
 * - threshold: Color difference threshold (0.0-1.0)
 * - smoothness: Edge smoothness (0.0-1.0)
 */
extern "C" __global__ void chroma_key(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    float threshold,
    float smoothness)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        // Get RGB values
        float r = input[idx] / 255.0f;
        float g = input[idx + 1] / 255.0f;
        float b = input[idx + 2] / 255.0f;
        
        // Green screen keying - distance from pure green (0,1,0)
        float diff = sqrtf((r*r) + ((g-1)*(g-1)) + (b*b));
        
        // Apply smooth transition
        float alpha = 1.0f;
        if (diff < threshold) {
            alpha = 0.0f;
        } else if (diff < threshold + smoothness) {
            alpha = (diff - threshold) / smoothness;
        }
        
        // Write result with modified alpha
        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
        output[idx + 3] = (unsigned char)(alpha * input[idx + 3]);
    }
}

/**
 * Default process function - pass through with no modification
 * This is useful as a template or for testing
 */
extern "C" __global__ void process(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        // Simply copy input to output
        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
        output[idx + 3] = input[idx + 3];
    }
}