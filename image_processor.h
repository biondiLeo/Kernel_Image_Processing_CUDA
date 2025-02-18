#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <vector>
#include "filter_kernel.h"

/**
 * @brief Enumerazione per i diversi tipi di memoria CUDA
 *
 * Definisce le opzioni disponibili per l'allocazione della memoria
 * nell'elaborazione GPU con CUDA
 */
enum class CudaMemoryType
{
    GLOBAL_MEM,    // Memoria globale GPU, più grande ma più lenta
    CONSTANT_MEM,  // Memoria costante GPU, più veloce ma limitata
    SHARED_MEM     // Memoria condivisa GPU, velocissima ma molto limitata
};

/**
 * @class ImageProcessor
 * @brief Classe per l'elaborazione delle immagini con supporto per elaborazione sequenziale e parallela
 *
 * Fornisce funzionalità per caricare, salvare e modificare immagini,
 * con particolare attenzione all'applicazione di filtri sia in modo sequenziale
 * che parallelo (CUDA e OpenMP)
 */
class ImageProcessor
{
public:
    /**
     * @brief Costruttore predefinito
     */
    ImageProcessor();

    /**
     * @brief Distruttore
     */
    ~ImageProcessor();

    // Metodi base per gestione immagine
    /**
     * @brief Ottiene la larghezza dell'immagine
     * @return Larghezza in pixel
     */
    int getWidth() const;

    /**
     * @brief Ottiene l'altezza dell'immagine
     * @return Altezza in pixel
     */
    int getHeight() const;

    /**
     * @brief Ottiene il numero di canali dell'immagine
     * @return Numero di canali (es. 3 per RGB)
     */
    int getChannels() const;

    /**
     * @brief Carica un'immagine da file
     * @param filepath Percorso del file immagine
     * @return true se il caricamento ha successo
     */
    bool loadImageFromFile(const char* filepath);

    /**
     * @brief Salva l'immagine su file
     * @param filepath Percorso dove salvare l'immagine
     * @return true se il salvataggio ha successo
     */
    bool saveImageToFile(const char* filepath) const;

    /**
     * @brief Imposta i dati dell'immagine manualmente
     * @param data Vector contenente i dati dell'immagine
     * @param width Larghezza dell'immagine
     * @param height Altezza dell'immagine
     * @return true se l'operazione ha successo
     */
    bool setImageData(const std::vector<float>& data, int width, int height);

    /**
     * @brief Ottiene i dati dell'immagine
     * @return Vector contenente i dati dell'immagine
     */
    std::vector<float> getImageData() const;

    // Metodi per l'applicazione dei filtri

    /**
     * @brief Applica un filtro in modo sequenziale, salvando il risultato in una nuova immagine
     * @param output Immagine di destinazione
     * @param filter Filtro da applicare
     * @return true se l'operazione ha successo
     */
    bool applyFilterSequential(ImageProcessor& output, const FilterKernel& filter) const;

    /**
     * @brief Applica un filtro in modo sequenziale sull'immagine corrente
     * @param filter Filtro da applicare
     * @return true se l'operazione ha successo
     */
    bool applyFilterSequential(const FilterKernel& filter);

    /**
     * @brief Applica un filtro in parallelo usando CUDA
     * @param output Immagine di destinazione
     * @param filter Filtro da applicare
     * @param memType Tipo di memoria CUDA da utilizzare
     * @return true se l'operazione ha successo
     */
    bool applyFilterParallel(ImageProcessor& output, const FilterKernel& filter, const CudaMemoryType memType);

    /**
     * @brief Applica un filtro in parallelo usando OpenMP
     * @param output Immagine di destinazione
     * @param filter Filtro da applicare
     * @param numThreads Numero di thread da utilizzare
     * @return true se l'operazione ha successo
     */
    bool applyFilterOpenMP(ImageProcessor& output, const FilterKernel& filter, int numThreads) const;

    /**
     * @brief Applica un filtro in parallelo con CUDA e misura i tempi di esecuzione
     * @param output Immagine di destinazione
     * @param filter Filtro da applicare
     * @param memType Tipo di memoria CUDA da utilizzare
     * @param computationTime Tempo di calcolo in millisecondi (output)
     * @param transferTime Tempo di trasferimento memoria in millisecondi (output)
     * @return true se l'operazione ha successo
     */
    bool applyFilterParallelWithTimings(ImageProcessor& output, const FilterKernel& filter,
        const CudaMemoryType memType, double& computationTime,
        double& transferTime);

private:
    /**
     * @brief Implementa la logica core per l'applicazione del filtro
     * @param filter Filtro da applicare
     * @return Vector contenente l'immagine filtrata
     */
    std::vector<float> applyFilterCore(const FilterKernel& filter) const;

    /**
     * @brief Crea una versione dell'immagine con padding
     * @param paddingY Padding verticale
     * @param paddingX Padding orizzontale
     * @return Vector contenente l'immagine con padding
     */
    std::vector<float> createPaddedImage(int paddingY, int paddingX) const;

    std::vector<float> imageData;  // Dati dell'immagine memorizzati come array lineare
    int width;                     // Larghezza dell'immagine in pixel
    int height;                    // Altezza dell'immagine in pixel
};

#endif // IMAGE_PROCESSOR_H