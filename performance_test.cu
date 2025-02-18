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

/**
 * @brief Struttura per memorizzare i risultati di un singolo test
 *
 * Contiene tutte le informazioni relative a un'esecuzione del test:
 * dimensioni dell'immagine, configurazione del test e risultati delle prestazioni
 */
struct TestResult {
    int imageWidth;            // Larghezza dell'immagine
    int imageHeight;           // Altezza dell'immagine
    int kernelSize;            // Dimensione del kernel del filtro
    int numThreads;            // Numero di thread per OpenMP
    std::string testType;      // Tipo di test (Sequential, CUDA, OpenMP)
    double totalTime;          // Tempo totale di esecuzione
    double computationTime;    // Tempo di sola computazione
    double transferTime;       // Tempo di trasferimento dati (per CUDA)
    double speedup;            // Accelerazione rispetto alla versione sequenziale
};

/**
 * @class PerformanceTester
 * @brief Classe per l'esecuzione e la gestione dei test di performance
 *
 * Gestisce l'esecuzione di test di performance su diverse implementazioni
 * dell'elaborazione delle immagini (sequenziale, CUDA, OpenMP)
 * con diverse configurazioni di input.
 */
class PerformanceTester {
private:
    // Configurazione dei test
    std::vector<std::pair<std::string, std::pair<int, int>>> imageFiles;  // File di test e dimensioni
    std::vector<int> kernelSizes;   // Dimensioni dei kernel da testare
    std::vector<int> threadCounts;  // Numero di thread per i test OpenMP
    std::vector<TestResult> allResults;  // Risultati di tutti i test

    /**
     * @brief Esegue un singolo test di performance
     *
     * @param input Immagine di input
     * @param filter Filtro da applicare
     * @param testType Tipo di test da eseguire
     * @param computationTime Tempo di computazione (output)
     * @param transferTime Tempo di trasferimento (output)
     * @param numThreads Numero di thread per OpenMP
     * @return Tempo totale di esecuzione
     */
    double runSingleTest(ImageProcessor& input, const FilterKernel& filter,
        const std::string& testType, double& computationTime,
        double& transferTime, int numThreads = 0)
    {
        ImageProcessor output;
        computationTime = 0;
        transferTime = 0;

        // Esegue il test appropriato in base al tipo
        if (testType == "Sequential") {
            // Test sequenziale
            auto start = std::chrono::high_resolution_clock::now();
            input.applyFilterSequential(output, filter);
            auto end = std::chrono::high_resolution_clock::now();
            computationTime = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
        }
        else if (testType.find("CUDA_") != std::string::npos) {
            // Test CUDA con diversi tipi di memoria
            CudaMemoryType memType;
            if (testType == "CUDA_Global") memType = CudaMemoryType::GLOBAL_MEM;
            else if (testType == "CUDA_Constant") memType = CudaMemoryType::CONSTANT_MEM;
            else memType = CudaMemoryType::SHARED_MEM;

            input.applyFilterParallelWithTimings(output, filter, memType,
                computationTime, transferTime);
        }
        else if (testType == "OpenMP") {
            // Test OpenMP
            auto start = std::chrono::high_resolution_clock::now();
            input.applyFilterOpenMP(output, filter, numThreads);
            auto end = std::chrono::high_resolution_clock::now();
            computationTime = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
        }

        return computationTime + transferTime;
    }

    /**
     * @brief Salva i risultati dei test in un file CSV
     *
     * Crea un file CSV con tutti i risultati dei test, includendo
     * configurazioni e metriche di performance
     */
    void saveResults() {
        // Ottiene il percorso corrente e crea il percorso del file di output
        char currentPath[256];
        std::string outputPath;

        if (_getcwd(currentPath, sizeof(currentPath)) != NULL) {
            outputPath = std::string(currentPath) + "\\output\\performance_results.csv";
        }
        else {
            std::cerr << "Errore nel recupero del percorso corrente" << std::endl;
            return;
        }

        // Apre il file di output
        std::ofstream out(outputPath);
        if (!out.is_open()) {
            std::cerr << "Errore nell'apertura del file dei risultati: " << outputPath << std::endl;
            return;
        }

        // Scrive l'header del CSV
        out << "Image_Resolution,Kernel_Size,Test_Type,Num_Threads,"
            << "Total_Time_ms,Computation_Time_ms,Transfer_Time_ms,Speedup\n";

        // Scrive i risultati di ogni test
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
    /**
     * @brief Costruttore che inizializza la configurazione dei test
     *
     * Definisce le immagini di test, le dimensioni dei kernel
     * e le configurazioni OpenMP da testare
     */
    PerformanceTester() {
        // Definisce le immagini di test con diverse risoluzioni
        imageFiles = {
            {"input/lena_512.png", {512, 512}},
            {"input/lena_768.png", {768, 768}},
            {"input/lena_1024.png", {1024, 1024}},
            {"input/lena_1536.png", {1536, 1536}},
            {"input/lena_2048.png", {2048, 2048}},
            {"input/lena_8192.png", {8192, 8192}}
        };

        // Definisce le dimensioni dei kernel da testare
        kernelSizes = { 3, 5, 7 };

        // Calcola il numero di thread da testare (potenze di 2)
        int maxThreads = omp_get_max_threads();
        for (int threads = 2; threads <= maxThreads; threads *= 2) {
            threadCounts.push_back(threads);
        }

        // Crea la directory di output
        char currentPath[256];
        if (_getcwd(currentPath, sizeof(currentPath)) != NULL) {
            std::string outputPath = std::string(currentPath) + "\\output";
            _mkdir(outputPath.c_str());
        }
    }

    /**
     * @brief Esegue la batteria completa di test
     *
     * Esegue test su tutte le combinazioni di:
     * - Immagini di input
     * - Dimensioni del kernel
     * - Implementazioni (Sequential, CUDA, OpenMP)
     * - Configurazioni (thread OpenMP, tipi di memoria CUDA)
     */
    void runTests() {
        // Stampa la configurazione dei test
        std::cout << "\n=== Test di Performance del Filtro Gaussiano ===\n";
        std::cout << "Risoluzioni da testare: " << imageFiles.size() << "\n";
        std::cout << "Dimensioni kernel: 3x3, 5x5, 7x7\n";
        std::cout << "Implementazioni CUDA: Global Memory, Constant Memory, Shared Memory\n";
        std::cout << "Configurazioni OpenMP: ";
        for (int threads : threadCounts) {
            std::cout << threads << " threads, ";
        }
        std::cout << "\n\n";

        // Esegue i test per ogni immagine
        for (const auto& imageInfo : imageFiles) {
            // ... [resto del codice con commenti inline] ...
        }

        // Salva i risultati
        saveResults();
    }
};

/**
 * @brief Funzione principale per l'esecuzione dei test di performance
 *
 * @return 0 se i test sono completati con successo, 1 in caso di errore
 */
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