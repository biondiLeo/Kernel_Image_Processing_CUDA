#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <iomanip>
#include <direct.h>
#include <map>
#include <string>
#include <omp.h>
#include "image_processor.h"
#include "filter_kernel.h"
#include <sys/stat.h>

struct TestResult {
    int imageWidth;
    int imageHeight;
    int kernelSize;
    int numThreads;     // Per i test OpenMP
    std::string testType;  // "Sequential", "CUDA_Global", "CUDA_Constant", "OpenMP"
    double totalTime;          // Tempo totale incluso trasferimento (execution time)
    double computationTime;    // Solo tempo di computazione
    double transferTime;       // Solo tempo di trasferimento
    double speedup;     // Rispetto alla versione sequenziale
};

class PerformanceTester {
private:
    std::vector<std::pair<std::string, std::pair<int, int>>> imageFiles;
    std::vector<int> kernelSizes;
    std::vector<int> threadCounts;
    std::vector<TestResult> allResults;

    double runSingleTest(ImageProcessor& input, const FilterKernel& filter, 
        const std::string& testType, double& computationTime, 
        double& transferTime, int numThreads = 0)
    {
        ImageProcessor output;
        computationTime = 0;
        transferTime = 0;

        if (testType == "Sequential") {
            auto start = std::chrono::high_resolution_clock::now();
            input.applyFilterSequential(output, filter);
            auto end = std::chrono::high_resolution_clock::now();
            computationTime = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
        }
        else if (testType.find("CUDA_") != std::string::npos) {
            CudaMemoryType memType;
            if (testType == "CUDA_Global") memType = CudaMemoryType::GLOBAL_MEM;
            else if (testType == "CUDA_Constant") memType = CudaMemoryType::CONSTANT_MEM;
            else memType = CudaMemoryType::SHARED_MEM;

            input.applyFilterParallelWithTimings(output, filter, memType,
                computationTime, transferTime);
        }
        else if (testType == "OpenMP") {
            auto start = std::chrono::high_resolution_clock::now();
            input.applyFilterOpenMP(output, filter, numThreads);
            auto end = std::chrono::high_resolution_clock::now();
            computationTime = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
        }

        return computationTime + transferTime;
    }

    void saveResults() {
        char currentPath[256];
        std::string outputPath;

        if (_getcwd(currentPath, sizeof(currentPath)) != NULL) {
            outputPath = std::string(currentPath) + "\\output\\performance_results.csv";
        }
        else {
            std::cerr << "Errore nel recupero del percorso corrente" << std::endl;
            return;
        }

        std::ofstream out(outputPath);
        if (!out.is_open()) {
            std::cerr << "Errore nell'apertura del file dei risultati: " << outputPath << std::endl;
            return;
        }

        // Header del CSV aggiornato
        out << "Image_Resolution,Kernel_Size,Test_Type,Num_Threads,"
            << "Total_Time_ms,Computation_Time_ms,Transfer_Time_ms,Speedup\n";

        // Scrivi i risultati
        for (const auto& result : allResults) {
            out << result.imageWidth << "x" << result.imageHeight << ","
                << result.kernelSize << "x" << result.kernelSize << ","
                << result.testType << ","
                << result.numThreads << ","
                << std::fixed << std::setprecision(3)
                << result.totalTime / 1000.0 << ","           // Converte in millisecondi
                << result.computationTime / 1000.0 << ","     // Converte in millisecondi
                << result.transferTime / 1000.0 << ","        // Converte in millisecondi
                << std::fixed << std::setprecision(3)
                << result.speedup << "\n";
        }

        out.close();
        std::cout << "\nRisultati salvati in: " << outputPath << std::endl;
    }

public:
    PerformanceTester() {
        // Definisci le immagini da testare
        imageFiles = {
            {"input/lena_512.png", {512, 512}},
            {"input/lena_768.png", {768, 768}},
            {"input/lena_1024.png", {1024, 1024}},
            {"input/lena_1536.png", {1536, 1536}},
            {"input/lena_2048.png", {2048, 2048}},
            {"input/lena_8192.png", {8192, 8192}}
        };

        // Dimensioni kernel da testare per il filtro gaussiano
        kernelSizes = { 3, 5, 7 };

        // Calcola il numero di thread da testare (potenze di 2 fino al massimo disponibile)
        int maxThreads = omp_get_max_threads();
        for (int threads = 2; threads <= maxThreads; threads *= 2) {
            threadCounts.push_back(threads);
        }

        // Crea la directory output se non esiste
        char currentPath[256];
        if (_getcwd(currentPath, sizeof(currentPath)) != NULL) {
            std::string outputPath = std::string(currentPath) + "\\output";
            _mkdir(outputPath.c_str());
        }
    }

    void runTests() {
        std::cout << "\n=== Test di Performance del Filtro Gaussiano ===\n";
        std::cout << "Risoluzioni da testare: " << imageFiles.size() << "\n";
        std::cout << "Dimensioni kernel: 3x3, 5x5, 7x7\n";
        std::cout << "Implementazioni CUDA: Global Memory, Constant Memory, Shared Memory\n";
        std::cout << "Configurazioni OpenMP: ";
        for (int threads : threadCounts) {
            std::cout << threads << " threads, ";
        }
        std::cout << "\n\n";

        for (const auto& imageInfo : imageFiles) {
            std::cout << "\nProcessing " << imageInfo.first << " ("
                << imageInfo.second.first << "x" << imageInfo.second.second << ")...\n";

            ImageProcessor inputImage;
            if (!inputImage.loadImageFromFile(imageInfo.first.c_str())) {
                std::cout << "Errore nel caricamento dell'immagine, skip...\n";
                continue;
            }

            for (int kernelSize : kernelSizes) {
                std::cout << "  Kernel " << kernelSize << "x" << kernelSize << "...\n";

                FilterKernel filter;
                filter.createGaussianFilter(kernelSize, 1.0f);

                // Variabili per i tempi
                double computationTime, transferTime;

                // Test sequenziale
                double seqTime = runSingleTest(inputImage, filter, "Sequential", computationTime, transferTime);
                TestResult seqResult = {
                    imageInfo.second.first,
                    imageInfo.second.second,
                    kernelSize,
                    0,
                    "Sequential",
                    seqTime,
                    computationTime,
                    transferTime,
                    1.0  // Speedup base
                };
                allResults.push_back(seqResult);

                // Test CUDA Global Memory
                double cudaGlobalTime = runSingleTest(inputImage, filter, "CUDA_Global", computationTime, transferTime);
                TestResult cudaGlobalResult = {
                    imageInfo.second.first,
                    imageInfo.second.second,
                    kernelSize,
                    0,
                    "CUDA_Global",
                    cudaGlobalTime,
                    computationTime,
                    transferTime,
                    seqTime / cudaGlobalTime
                };
                allResults.push_back(cudaGlobalResult);

                // Test CUDA Constant Memory
                double cudaConstTime = runSingleTest(inputImage, filter, "CUDA_Constant", computationTime, transferTime);
                TestResult cudaConstResult = {
                    imageInfo.second.first,
                    imageInfo.second.second,
                    kernelSize,
                    0,
                    "CUDA_Constant",
                    cudaConstTime,
                    computationTime,
                    transferTime,
                    seqTime / cudaConstTime
                };
                allResults.push_back(cudaConstResult);

                // Test CUDA Shared Memory
                double cudaSharedTime = runSingleTest(inputImage, filter, "CUDA_Shared", computationTime, transferTime);
                TestResult cudaSharedResult = {
                    imageInfo.second.first,
                    imageInfo.second.second,
                    kernelSize,
                    0,
                    "CUDA_Shared",
                    cudaSharedTime,
                    computationTime,
                    transferTime,
                    seqTime / cudaSharedTime
                };
                allResults.push_back(cudaSharedResult);

                // Test OpenMP con diversi numeri di thread
                for (int threads : threadCounts) {
                    double ompTime = runSingleTest(inputImage, filter, "OpenMP", computationTime, transferTime, threads);
                    TestResult ompResult = {
                        imageInfo.second.first,
                        imageInfo.second.second,
                        kernelSize,
                        threads,
                        "OpenMP",
                        ompTime,
                        computationTime,
                        transferTime,
                        seqTime / ompTime
                    };
                    allResults.push_back(ompResult);
                }

                std::cout << "    Completato\n";
            }
        }

        // Salva tutti i risultati
        saveResults();
    }
};

int runPerformanceTests(const std::string&, const FilterKernel&) {
    try {
        std::cout << "Avvio test delle prestazioni...\n";
        PerformanceTester tester;
        tester.runTests();
        std::cout << "Test completati con successo.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Errore durante i test: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}