// image_processor.cu
#include "image_processor.h"
#include <png++/png.hpp>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Dimensioni del blocco CUDA per l'elaborazione parallela
 *
 * Definisce la dimensione dei blocchi di thread CUDA.
 * 16x16 = 256 thread per blocco, un buon compromesso tra parallelismo e risorse
 */
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

 /**
  * @brief Dimensione massima supportata per il kernel del filtro
  *
  * Limita la dimensione del kernel per la memoria costante.
  * 25x25 è sufficiente per la maggior parte dei filtri di convoluzione
  */
#define MAX_KERNEL_SIZE 25

  /**
   * @brief Memoria costante per il kernel del filtro
   *
   * Allocazione in memoria costante GPU per il kernel del filtro.
   * Accessibile da tutti i thread in sola lettura e cacheable
   */
__device__ __constant__ float d_filterKernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

/**
 * @brief Calcola il numero di blocchi necessari per coprire una dimensione
 *
 * @param total Dimensione totale da coprire
 * @param blockSize Dimensione del blocco
 * @return Numero di blocchi necessari
 */
unsigned int calcolaBlocchi(unsigned int total, unsigned int blockSize) {
    return (total + blockSize - 1) / blockSize;
}

/**
 * @brief Kernel CUDA che utilizza la memoria globale
 *
 * Implementa la convoluzione utilizzando la memoria globale della GPU.
 * Ogni thread elabora un pixel dell'immagine di output.
 */
__global__ void processaImmagineGlobale(
    float* d_input, float* d_output, float* d_kernel,
    int width, int height, int paddedWidth, int paddedHeight,
    int kernelSize)
{
    const int radius = kernelSize / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // Coordinata x del pixel
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // Coordinata y del pixel

    // Verifica che il thread elabori un pixel valido
    if (x >= width || y >= height) return;

    // Applica la convoluzione
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

    // Limita il risultato nell'intervallo [0, 255]
    sum = fmaxf(0.0f, fminf(sum, 255.0f));
    d_output[y * width + x] = sum;
}

/**
 * @brief Kernel CUDA che utilizza la memoria costante
 *
 * Simile a processaImmagineGlobale ma utilizza la memoria costante
 * per il kernel del filtro, offrendo migliori performance per
 * accessi in sola lettura e cacheable
 */
__global__ void processaImmagineConstante(
    float* d_input, float* d_output,
    int width, int height, int paddedWidth, int paddedHeight,
    int kernelSize)
{
    // Implementazione simile a processaImmagineGlobale ma usa d_filterKernel
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

/**
 * @brief Kernel CUDA che utilizza la memoria condivisa
 *
 * Implementa la convoluzione utilizzando la memoria condivisa per
 * ottimizzare gli accessi ai dati dell'immagine. Carica un blocco
 * di dati in memoria condivisa prima dell'elaborazione.
 *
 * @tparam BLOCK_SIZE Dimensione del blocco di thread
 */
template<int BLOCK_SIZE>
__global__ void processaImmagineShared(
    float* d_input, float* d_output,
    int width, int height, int paddedWidth, int paddedHeight,
    int kernelSize)
{
    // Alloca memoria condivisa con padding per il blocco di immagine
    __shared__ float sharedMem[BLOCK_SIZE + 24][BLOCK_SIZE + 24]; // Per kernel fino a 5x5

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

    // Sincronizza tutti i thread del blocco prima di procedere
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

/**
 * @brief Costruttore della classe ImageProcessor
 * Inizializza un'immagine vuota
 */
ImageProcessor::ImageProcessor() : width(0), height(0) {}

/**
 * @brief Distruttore della classe ImageProcessor
 * Libera la memoria dell'immagine
 */
ImageProcessor::~ImageProcessor() {
    imageData.clear();
    std::vector<float>().swap(imageData);
}

/**
 * @brief Carica un'immagine PNG in scala di grigi
 *
 * Utilizza la libreria png++ per caricare l'immagine e
 * convertirla in un array di float
 */
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

/**
 * @brief Salva l'immagine in formato PNG
 *
 * Converte l'array di float in un'immagine PNG in scala di grigi
 */
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

/**
 * @brief Crea una versione dell'immagine con padding
 *
 * Aggiunge bordi all'immagine replicando i pixel dei bordi.
 * Necessario per l'applicazione corretta dei filtri di convoluzione.
 */
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

    // Replica i bordi per il padding
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

/**
 * @brief Applica un filtro usando l'elaborazione CUDA
 *
 * Implementa l'elaborazione parallela su GPU usando diversi tipi di memoria:
 * - Memoria globale: più lenta ma senza limitazioni di dimensione
 * - Memoria costante: più veloce per dati in sola lettura
 * - Memoria condivisa: molto veloce ma limitata per blocco
 */
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

/**
 * @brief Applica un filtro con misurazione dei tempi
 *
 * Versione strumentata di applyFilterParallel che misura separatamente
 * i tempi di trasferimento dati e di computazione
 */
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

// Implementazione dei metodi getter/setter base
int ImageProcessor::getWidth() const { return width; }
int ImageProcessor::getHeight() const { return height; }
int ImageProcessor::getChannels() const { return 1; }  // Immagini in scala di grigi

/**
 * @brief Imposta i dati dell'immagine
 *
 * Verifica la validità dei dati prima di impostarli
 */
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

/**
 * @brief Ottiene i dati dell'immagine
 */
std::vector<float> ImageProcessor::getImageData() const {
    return imageData;
}