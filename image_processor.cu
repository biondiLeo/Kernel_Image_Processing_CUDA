// image_processor.cu
#include "image_processor.h"
#include <png++/png.hpp>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define MAX_KERNEL_SIZE 25

__device__ __constant__ float d_filterKernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// Helper function per calcolare la dimensione dei blocchi CUDA
unsigned int calcolaBlocchi(unsigned int total, unsigned int blockSize) {
    return (total + blockSize - 1) / blockSize;
}

// Kernel CUDA per elaborazione con memoria globale
__global__ void processaImmagineGlobale(
    float* d_input, float* d_output, float* d_kernel,
    int width, int height, int paddedWidth, int paddedHeight,
    int kernelSize)
{
    const int radius = kernelSize / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int px = x + kx + radius;
            int py = y + ky + radius;
            float pixelValue = d_input[py * paddedWidth + px];
            float kernelValue = d_kernel[(ky + radius) * kernelSize + (kx + radius)];
            sum += pixelValue * kernelValue;
        }
    }

    sum = fmaxf(0.0f, fminf(sum, 255.0f));
    d_output[y * width + x] = sum;
}

// Kernel CUDA per elaborazione con memoria costante
__global__ void processaImmagineConstante(
    float* d_input, float* d_output,
    int width, int height, int paddedWidth, int paddedHeight,
    int kernelSize)
{
    const int radius = kernelSize / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int px = x + kx + radius;
            int py = y + ky + radius;
            float pixelValue = d_input[py * paddedWidth + px];
            float kernelValue = d_filterKernel[(ky + radius) * kernelSize + (kx + radius)];
            sum += pixelValue * kernelValue;
        }
    }

    sum = fmaxf(0.0f, fminf(sum, 255.0f));
    d_output[y * width + x] = sum;
}

template<int BLOCK_SIZE>
__global__ void processaImmagineShared(
    float* d_input, float* d_output,
    int width, int height, int paddedWidth, int paddedHeight,
    int kernelSize)
{
    __shared__ float sharedMem[BLOCK_SIZE + 24][BLOCK_SIZE + 24]; // Assumendo kernel max 5x5

    const int radius = kernelSize / 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_SIZE + tx;
    const int y = blockIdx.y * BLOCK_SIZE + ty;

    // Carica i dati nella memoria condivisa, includendo il padding
    for (int dy = 0; dy < (BLOCK_SIZE + 2 * radius + 1) / BLOCK_SIZE; dy++) {
        for (int dx = 0; dx < (BLOCK_SIZE + 2 * radius + 1) / BLOCK_SIZE; dx++) {
            int srcY = y - radius + dy * BLOCK_SIZE;
            int srcX = x - radius + dx * BLOCK_SIZE;

            if (srcY >= 0 && srcY < paddedHeight && srcX >= 0 && srcX < paddedWidth) {
                sharedMem[ty + dy * BLOCK_SIZE][tx + dx * BLOCK_SIZE] =
                    d_input[srcY * paddedWidth + srcX];
            }
        }
    }

    __syncthreads();

    // Applica il filtro usando la memoria condivisa
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                float pixelValue = sharedMem[ty + radius + ky][tx + radius + kx];
                float kernelValue = d_filterKernel[(ky + radius) * kernelSize + (kx + radius)];
                sum += pixelValue * kernelValue;
            }
        }

        sum = fmaxf(0.0f, fminf(sum, 255.0f));
        d_output[y * width + x] = sum;
    }
}

// Implementazione dei metodi della classe
ImageProcessor::ImageProcessor() : width(0), height(0) {}

ImageProcessor::~ImageProcessor() {
    imageData.clear();
    std::vector<float>().swap(imageData);
}

bool ImageProcessor::loadImageFromFile(const char* filepath) {
    try {
        png::image<png::gray_pixel> image(filepath);
        width = image.get_width();
        height = image.get_height();

        imageData.resize(width * height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                imageData[y * width + x] = static_cast<float>(image[y][x]);
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore nel caricamento dell'immagine: " << e.what() << std::endl;
        return false;
    }
}

bool ImageProcessor::saveImageToFile(const char* filepath) const {
    try {
        png::image<png::gray_pixel> image(width, height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                image[y][x] = static_cast<png::gray_pixel>(imageData[y * width + x]);
            }
        }
        image.write(filepath);
        std::cout << "Immagine salvata in: " << filepath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore nel salvataggio dell'immagine: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> ImageProcessor::createPaddedImage(int paddingY, int paddingX) const {
    int paddedWidth = width + 2 * paddingX;
    int paddedHeight = height + 2 * paddingY;
    std::vector<float> padded(paddedWidth * paddedHeight);

    // Copia l'immagine centrale
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            padded[(y + paddingY) * paddedWidth + (x + paddingX)] =
                imageData[y * width + x];
        }
    }

    // Replica i bordi
    // Bordi orizzontali
    for (int y = 0; y < paddingY; ++y) {
        std::copy_n(&padded[paddingY * paddedWidth], width,
            &padded[y * paddedWidth + paddingX]);
        std::copy_n(&padded[(height + paddingY - 1) * paddedWidth], width,
            &padded[(paddedHeight - y - 1) * paddedWidth + paddingX]);
    }

    // Bordi verticali
    for (int y = 0; y < paddedHeight; ++y) {
        for (int x = 0; x < paddingX; ++x) {
            padded[y * paddedWidth + x] = padded[y * paddedWidth + paddingX];
            padded[y * paddedWidth + paddedWidth - 1 - x] =
                padded[y * paddedWidth + paddedWidth - paddingX - 1];
        }
    }

    return padded;
}

bool ImageProcessor::applyFilterParallel(
    ImageProcessor& output, const FilterKernel& filter, const CudaMemoryType memType)
{
    int kernelSize = filter.getSize();
    if (kernelSize > MAX_KERNEL_SIZE) {
        std::cerr << "Dimensione kernel troppo grande" << std::endl;
        return false;
    }

    int radius = kernelSize / 2;
    auto padded = createPaddedImage(radius, radius);
    int paddedWidth = width + 2 * radius;
    int paddedHeight = height + 2 * radius;

    // Alloca memoria su GPU
    float* d_input, * d_output, * d_kernel = nullptr;
    cudaMalloc(&d_input, paddedWidth * paddedHeight * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    // Copia i dati sulla GPU
    cudaMemcpy(d_input, padded.data(),
        paddedWidth * paddedHeight * sizeof(float),
        cudaMemcpyHostToDevice);

    // Prepara i parametri per il kernel CUDA
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize(
        calcolaBlocchi(width, BLOCK_DIM_X),
        calcolaBlocchi(height, BLOCK_DIM_Y)
    );

    // Esegui il kernel appropriato
    if (memType == CudaMemoryType::GLOBAL_MEM) {
        cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
        cudaMemcpy(d_kernel, filter.getKernelData().data(),
            kernelSize * kernelSize * sizeof(float),
            cudaMemcpyHostToDevice);

        processaImmagineGlobale << <gridSize, blockSize >> > (
            d_input, d_output, d_kernel,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }
    else if (memType == CudaMemoryType::SHARED_MEM) {
        cudaMemcpyToSymbol(d_filterKernel, filter.getKernelData().data(),
            kernelSize * kernelSize * sizeof(float));

        processaImmagineShared<BLOCK_DIM_X> << <gridSize, blockSize >> > (
            d_input, d_output,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }
    else {  // CONSTANT_MEM
        cudaMemcpyToSymbol(d_filterKernel, filter.getKernelData().data(),
            kernelSize * kernelSize * sizeof(float));

        processaImmagineConstante << <gridSize, blockSize >> > (
            d_input, d_output,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }

    // Verifica errori
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Errore CUDA: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    // Copia il risultato indietro
    std::vector<float> result(width * height);
    cudaMemcpy(result.data(), d_output,
        width * height * sizeof(float),
        cudaMemcpyDeviceToHost);

    // Pulisci la memoria
    cudaFree(d_input);
    cudaFree(d_output);
    if (d_kernel) cudaFree(d_kernel);

    output.setImageData(result, width, height);
    return true;
}

bool ImageProcessor::applyFilterParallelWithTimings(
    ImageProcessor& output,
    const FilterKernel& filter,
    const CudaMemoryType memType,
    double& computationTime,
    double& transferTime)
{
    int kernelSize = filter.getSize();
    int radius = kernelSize / 2;
    auto padded = createPaddedImage(radius, radius);
    int paddedWidth = width + 2 * radius;
    int paddedHeight = height + 2 * radius;

    // Alloca memoria su GPU
    float* d_input, * d_output, * d_kernel = nullptr;

    auto transferStart = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_input, paddedWidth * paddedHeight * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    // Copia i dati sulla GPU
    cudaMemcpy(d_input, padded.data(),
        paddedWidth * paddedHeight * sizeof(float),
        cudaMemcpyHostToDevice);

    auto computeStart = std::chrono::high_resolution_clock::now();

    // Prepara i parametri per il kernel CUDA
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize(
        calcolaBlocchi(width, BLOCK_DIM_X),
        calcolaBlocchi(height, BLOCK_DIM_Y)
    );

    // Esegui il kernel appropriato
    if (memType == CudaMemoryType::GLOBAL_MEM) {
        cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
        cudaMemcpy(d_kernel, filter.getKernelData().data(),
            kernelSize * kernelSize * sizeof(float),
            cudaMemcpyHostToDevice);

        processaImmagineGlobale << <gridSize, blockSize >> > (
            d_input, d_output, d_kernel,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }
    else if (memType == CudaMemoryType::SHARED_MEM) {
        cudaMemcpyToSymbol(d_filterKernel, filter.getKernelData().data(),
            kernelSize * kernelSize * sizeof(float));

        processaImmagineShared<BLOCK_DIM_X> << <gridSize, blockSize >> > (
            d_input, d_output,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }
    else {  // CONSTANT_MEM
        cudaMemcpyToSymbol(d_filterKernel, filter.getKernelData().data(),
            kernelSize * kernelSize * sizeof(float));

        processaImmagineConstante << <gridSize, blockSize >> > (
            d_input, d_output,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }

    cudaDeviceSynchronize();
    auto computeEnd = std::chrono::high_resolution_clock::now();

    // Copia il risultato indietro
    std::vector<float> result(width * height);
    cudaMemcpy(result.data(), d_output,
        width * height * sizeof(float),
        cudaMemcpyDeviceToHost);

    auto transferEnd = std::chrono::high_resolution_clock::now();

    // Calcola i tempi
    computationTime = std::chrono::duration_cast<std::chrono::microseconds>(
        computeEnd - computeStart).count();
    transferTime = std::chrono::duration_cast<std::chrono::microseconds>(
        transferEnd - transferStart).count() - computationTime;

    // Pulisci la memoria
    cudaFree(d_input);
    cudaFree(d_output);
    if (d_kernel) cudaFree(d_kernel);

    output.setImageData(result, width, height);
    return true;
}

// Implementazione dei metodi getter/setter
int ImageProcessor::getWidth() const { return width; }
int ImageProcessor::getHeight() const { return height; }
int ImageProcessor::getChannels() const { return 1; }  // Immagini in scala di grigi

bool ImageProcessor::setImageData(const std::vector<float>& data, int w, int h) {
    if (data.size() != w * h) {
        std::cerr << "Dimensioni dati non valide" << std::endl;
        return false;
    }
    width = w;
    height = h;
    imageData = data;
    return true;
}

std::vector<float> ImageProcessor::getImageData() const {
    return imageData;
}