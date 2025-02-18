#ifndef FILTER_KERNEL_H
#define FILTER_KERNEL_H

#include <vector>

// Definizione delle costanti per i diversi tipi di filtro
// Costanti per il filtro di nitidezza: il pixel centrale viene amplificato mentre i pixel circostanti vengono sottratti
constexpr float SHARPEN_CENTER = 9.0f;
constexpr float SHARPEN_SURROUND = -1.0f;

// Costanti per il filtro di rilevamento dei bordi: simile al filtro di nitidezza ma con pesi diversi
constexpr float EDGE_CENTER = 8.0f;
constexpr float EDGE_SURROUND = -1.0f;

// Costanti per il filtro Laplaciano: utilizzato per rilevare bordi e cambiamenti rapidi di intensità
constexpr float LAPLACE_CENTER = 4.0f;
constexpr float LAPLACE_SURROUND = -1.0f;

/**
 * @class FilterKernel
 * @brief Una classe per la creazione e gestione di vari kernel/filtri per l'elaborazione delle immagini
 *
 * Questa classe fornisce funzionalità per creare diversi tipi di filtri per l'elaborazione
 * delle immagini come sfocatura gaussiana, nitidezza, rilevamento dei bordi e filtri Laplaciani.
 * Ogni filtro è rappresentato come una matrice quadrata (kernel) memorizzata in un vettore lineare.
 */
class FilterKernel {
public:
    // Costruttore predefinito
    FilterKernel();

    // Distruttore che pulisce correttamente la memoria del vettore
    ~FilterKernel() {
        kernelData.clear();
        std::vector<float>().swap(kernelData);
    }

    /**
     * @brief Visualizza i valori attuali del kernel per debug o visualizzazione
     */
    void displayKernel() const;

    /**
     * @brief Crea un filtro di sfocatura gaussiana
     * @param size La dimensione del kernel (deve essere dispari)
     * @param sigma La deviazione standard della distribuzione gaussiana
     * @return true se la creazione ha successo, false altrimenti
     */
    bool createGaussianFilter(int size, float sigma);

    /**
     * @brief Crea un filtro di nitidezza che migliora i bordi
     * @return true se la creazione ha successo, false altrimenti
     */
    bool createSharpeningFilter();

    /**
     * @brief Crea un filtro per il rilevamento dei bordi
     * @return true se la creazione ha successo, false altrimenti
     */
    bool createEdgeDetectionFilter();

    /**
     * @brief Crea un filtro Laplaciano per il rilevamento e il miglioramento dei bordi
     * @return true se la creazione ha successo, false altrimenti
     */
    bool createLaplacianFilter();

    /**
     * @brief Crea un filtro Difference of Gaussian (DoG)
     * @return true se la creazione ha successo, false altrimenti
     */
    bool createDoGFilter();

    /**
     * @brief Ottiene la dimensione del kernel
     * @return La dimensione di un lato del kernel quadrato
     */
    int getSize() const;

    /**
     * @brief Ottiene i dati del kernel
     * @return Vettore contenente la matrice del kernel linearizzata
     */
    std::vector<float> getKernelData() const;

private:
    /**
     * @brief Metodo di supporto per inizializzare il kernel con valori centrali e circostanti
     * @param kernel Riferimento al vettore del kernel da inizializzare
     * @param centerValue Valore per il pixel centrale
     * @param surroundValue Valore per i pixel circostanti
     * @param size Dimensione del kernel
     * @return true se l'inizializzazione ha successo, false altrimenti
     */
    bool initializeKernel(std::vector<float>& kernel, float centerValue, float surroundValue, int size);

    std::vector<float> kernelData;  // Matrice del filtro linearizzata
    int size;                       // Dimensione di un lato del kernel quadrato
};

#endif // FILTER_KERNEL_H