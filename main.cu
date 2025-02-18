#include <iostream>
#include <chrono>
#include <string>
#include <omp.h>
#include "image_processor.h"
#include "filter_kernel.h"
#include "performance_test.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

/**
 * Definizione delle costanti per i comandi dell'applicazione
 */
 // Tipi di filtro disponibili
#define CMD_GAUSSIAN     "gaussian"    // Filtro di sfocatura gaussiana
#define CMD_SHARPEN      "sharpen"     // Filtro di nitidezza
#define CMD_EDGE         "edge"        // Rilevamento bordi
#define CMD_LAPLACIAN    "laplacian"   // Filtro laplaciano
#define CMD_DOG          "dog"         // Difference of Gaussian

// Tipi di memoria CUDA disponibili
#define CMD_CUDA_GLOBAL  "global"      // Memoria globale GPU
#define CMD_CUDA_CONST   "constant"    // Memoria costante GPU
#define CMD_CUDA_SHARED  "shared"      // Memoria condivisa GPU

// Costanti per la gestione dei file
#define OUTPUT_DIR       "output/"     // Directory per i file di output
#define IMG_EXT         ".png"         // Estensione dei file immagine

/**
 * @brief Programma principale per l'elaborazione delle immagini
 *
 * Implementa un'applicazione di confronto tra diverse tecniche
 * di elaborazione immagini: sequenziale (CPU), parallela (CUDA)
 * e multi-thread (OpenMP).
 *
 * Utilizzo: program tipo_filtro percorso_immagine [tipo_memoria_cuda]
 */
int main(int argc, char** argv) {
    // Verifica i parametri di input
    if (argc < 3) {
        std::cerr << "Utilizzo: " << argv[0] << " tipo_filtro percorso_immagine [tipo_memoria_cuda]" << std::endl;
        std::cerr << "tipo_filtro: <gaussian | sharpen | edge | laplacian | dog>" << std::endl;
        std::cerr << "percorso_immagine: percorso del file immagine" << std::endl;
        std::cerr << "(opzionale) tipo_memoria_cuda: <global | constant | shared> (default: constant)" << std::endl;
        return 1;
    }

    // Parsing dei parametri da linea di comando
    std::string cmdFilter = std::string(argv[1]);
    std::string imagePath = std::string(argv[2]);
    std::string cudaMemType = (argc > 3) ? std::string(argv[3]) : CMD_CUDA_CONST;

    /**
     * Configurazione del filtro
     * Crea il filtro specificato con i parametri appropriati
     */
    FilterKernel filter;
    if (cmdFilter == CMD_GAUSSIAN) {
        filter.createGaussianFilter(7, 1.0f);  // Kernel 7x7, sigma=1.0
    }
    else if (cmdFilter == CMD_SHARPEN) {
        filter.createSharpeningFilter();
    }
    else if (cmdFilter == CMD_EDGE) {
        filter.createEdgeDetectionFilter();
    }
    else if (cmdFilter == CMD_LAPLACIAN) {
        filter.createLaplacianFilter();
    }
    else if (cmdFilter == CMD_DOG) {
        filter.createDoGFilter();
    }
    else {
        std::cerr << "Tipo di filtro non valido: " << cmdFilter << std::endl;
        return 1;
    }

    // Visualizza il kernel del filtro selezionato
    std::cout << "\nFiltro selezionato: " << cmdFilter << std::endl;
    filter.displayKernel();

    /**
     * Caricamento dell'immagine di input
     */
    ImageProcessor inputImage;
    if (!inputImage.loadImageFromFile(imagePath.c_str())) {
        std::cerr << "Errore nel caricamento dell'immagine: " << imagePath << std::endl;
        return 1;
    }

    // Preparazione degli oggetti per i risultati
    ImageProcessor outputCUDA;
    ImageProcessor outputCPU;
    std::vector<ImageProcessor> outputsOpenMP;

    /**
     * Test Elaborazione Sequenziale (CPU)
     * Misura il tempo di esecuzione dell'elaborazione sequenziale
     */
    std::cout << "\n=== Versione Sequenziale ===" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    bool cpuResult = inputImage.applyFilterSequential(outputCPU, filter);
    auto t2 = std::chrono::high_resolution_clock::now();

    if (cpuResult) {
        auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "Tempo di esecuzione CPU: " << cpuDuration << " microsec" << std::endl;
        std::string outputCpuPath = OUTPUT_DIR + std::string("cpu_") + cmdFilter + IMG_EXT;
        outputCPU.saveImageToFile(outputCpuPath.c_str());
    }

    /**
     * Test Elaborazione CUDA (GPU)
     * Configura ed esegue l'elaborazione parallela su GPU
     */
    std::cout << "\n=== Versione Parallela CUDA ===" << std::endl;
    // Determina il tipo di memoria CUDA da utilizzare
    CudaMemoryType memType = CudaMemoryType::CONSTANT_MEM;
    if (cudaMemType == CMD_CUDA_GLOBAL) {
        memType = CudaMemoryType::GLOBAL_MEM;
    }
    else if (cudaMemType == CMD_CUDA_SHARED) {
        memType = CudaMemoryType::SHARED_MEM;
    }

    // Inizializza il runtime CUDA
    cudaFree(0);  // Forza l'inizializzazione del context CUDA

    // Esegue e misura l'elaborazione CUDA
    auto t3 = std::chrono::high_resolution_clock::now();
    bool cudaResult = inputImage.applyFilterParallel(outputCUDA, filter, memType);
    auto t4 = std::chrono::high_resolution_clock::now();

    if (cudaResult) {
        auto cudaDuration = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        std::cout << "Tempo di esecuzione CUDA: " << cudaDuration << " microsec" << std::endl;
        std::string outputCudaPath = OUTPUT_DIR + std::string("cuda_") + cmdFilter + IMG_EXT;
        outputCUDA.saveImageToFile(outputCudaPath.c_str());
    }

    /**
     * Test Elaborazione OpenMP (CPU Multi-thread)
     * Esegue test con numero crescente di thread
     */
    std::cout << "\n=== Versione Parallela OpenMP ===" << std::endl;
    int maxThreads = omp_get_max_threads();
    std::cout << "Numero massimo di thread disponibili: " << maxThreads << std::endl;

    // Test con diversi numeri di thread (potenze di 2)
    for (int numThreads = 2; numThreads <= maxThreads; numThreads *= 2) {
        ImageProcessor outputOMP;
        auto t5 = std::chrono::high_resolution_clock::now();
        bool ompResult = inputImage.applyFilterOpenMP(outputOMP, filter, numThreads);
        auto t6 = std::chrono::high_resolution_clock::now();

        if (ompResult) {
            auto ompDuration = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
            std::cout << "Tempo di esecuzione OpenMP (" << numThreads << " threads): "
                << ompDuration << " microsec" << std::endl;

            // Salva il risultato
            std::string outputOmpPath = OUTPUT_DIR + std::string("omp") +
                std::to_string(numThreads) + "_" +
                cmdFilter + IMG_EXT;
            outputOMP.saveImageToFile(outputOmpPath.c_str());
            outputsOpenMP.push_back(outputOMP);
        }
    }

    /**
     * Esecuzione dei test di performance
     * Avvia una serie di test di performance più dettagliati
     */
    std::cout << "\n=== Test di Performance ===" << std::endl;
    runPerformanceTests("", filter);  // I parametri non vengono più usati ma manteniamo la compatibilità

    return 0;
}