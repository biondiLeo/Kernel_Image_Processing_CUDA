#include "filter_kernel.h"
#include <iostream>
#include <cmath>
#include <iomanip>

// Definizione della costante matematica PI se non è già definita nel sistema
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Costruttore della classe FilterKernel
 * Inizializza un kernel vuoto con dimensione zero
 */
FilterKernel::FilterKernel() {
    kernelData.clear();
    size = 0;
}

/**
 * @brief Visualizza il kernel come matrice per scopi di debug
 * Mostra i valori del kernel formattati con 4 decimali
 */
void FilterKernel::displayKernel() const {
    if (size == 0 || kernelData.empty()) {
        std::cout << "Il kernel non è stato ancora inizializzato" << std::endl;
        return;
    }

    std::cout << "\n=== Matrice del Filtro ===\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << std::fixed << std::setprecision(4)
                << kernelData[j + i * size] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "=====================\n\n";
}

/**
 * @brief Crea un filtro gaussiano per la sfocatura delle immagini
 *
 * Implementa una distribuzione gaussiana 2D normalizzata.
 * La formula utilizzata è: exp(-(x² + y²)/(2σ²)) / (2πσ²)
 *
 * @param kernelSize Dimensione del kernel (deve essere dispari e >= 3)
 * @param sigma Deviazione standard della distribuzione gaussiana
 * @return true se il filtro viene creato con successo
 */
bool FilterKernel::createGaussianFilter(int kernelSize, float sigma) {
    std::cout << "Creazione filtro gaussiano..." << std::endl;

    // Verifica parametri di input
    if (kernelSize % 2 == 0 || kernelSize < 3) {
        std::cerr << "Errore: la dimensione del kernel deve essere dispari e >= 3" << std::endl;
        return false;
    }

    if (sigma <= 0) {
        std::cerr << "Errore: sigma deve essere positivo" << std::endl;
        return false;
    }

    size = kernelSize;
    kernelData.resize(size * size);
    float sum = 0.0f;

    // Calcolo centro del kernel
    int center = size / 2;

    // Creazione del filtro gaussiano
    for (int y = -center; y <= center; y++) {
        for (int x = -center; x <= center; x++) {
            // Calcolo del valore gaussiano per ogni posizione
            float value = exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            value /= 2.0f * M_PI * sigma * sigma;

            kernelData[(y + center) * size + (x + center)] = value;
            sum += value;
        }
    }

    // Normalizzazione per garantire che la somma dei valori sia 1
    for (int i = 0; i < size * size; i++) {
        kernelData[i] /= sum;
    }

    return true;
}

/**
 * @brief Crea un filtro per aumentare la nitidezza dell'immagine
 *
 * Utilizza una matrice 3x3 con un valore centrale alto positivo
 * e valori negativi intorno per enfatizzare i bordi
 */
bool FilterKernel::createSharpeningFilter() {
    size = 3;
    kernelData.resize(size * size);

    return initializeKernel(kernelData, SHARPEN_CENTER, SHARPEN_SURROUND, size);
}

/**
 * @brief Crea un filtro per il rilevamento dei bordi
 *
 * Implementa una matrice 3x3 che evidenzia le differenze
 * tra il pixel centrale e i suoi vicini
 */
bool FilterKernel::createEdgeDetectionFilter() {
    size = 3;
    kernelData.resize(size * size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == 1 && j == 1) {
                kernelData[j + i * size] = EDGE_CENTER;
            }
            else {
                kernelData[j + i * size] = EDGE_SURROUND;
            }
        }
    }

    return true;
}

/**
 * @brief Crea un filtro Laplaciano per il rilevamento dei bordi
 *
 * Implementa un operatore Laplaciano discreto che è sensibile
 * ai cambiamenti rapidi di intensità nell'immagine
 */
bool FilterKernel::createLaplacianFilter() {
    size = 3;
    kernelData.resize(size * size);

    // Inizializza tutti gli elementi a -1
    for (int i = 0; i < size * size; i++) {
        kernelData[i] = LAPLACE_SURROUND;
    }

    // Imposta il valore centrale
    kernelData[4] = LAPLACE_CENTER;  // indice centrale in una matrice 3x3

    // Imposta gli angoli a 0 per ridurre la sensibilità al rumore
    kernelData[0] = 0.0f;  // alto-sinistra
    kernelData[2] = 0.0f;  // alto-destra
    kernelData[6] = 0.0f;  // basso-sinistra
    kernelData[8] = 0.0f;  // basso-destra

    return true;
}

/**
 * @brief Crea un filtro Difference of Gaussian (DoG)
 *
 * Implementa un filtro 5x5 che approssima la differenza
 * di due filtri gaussiani con diverse deviazioni standard.
 * Utile per il rilevamento dei bordi e l'enfasi dei dettagli
 */
bool FilterKernel::createDoGFilter() {
    size = 5;
    kernelData.resize(size * size, 0.0f);  // Inizializza tutto a 0

    // Valori per i diversi "anelli" del filtro
    float centerValue = 16.0f;
    float primarySurround = -2.0f;    // Primi vicini
    float secondarySurround = -1.0f;  // Secondi vicini

    // Imposta il valore centrale
    kernelData[12] = centerValue;  // Centro della matrice 5x5

    // Imposta i valori primari circostanti (croce)
    kernelData[7] = primarySurround;   // Sopra-centro
    kernelData[11] = primarySurround;  // Sinistra-centro
    kernelData[13] = primarySurround;  // Destra-centro
    kernelData[17] = primarySurround;  // Sotto-centro

    // Imposta i valori secondari (diagonali e esterni)
    kernelData[2] = secondarySurround;   // Superiore
    kernelData[6] = secondarySurround;   // Sinistra-superiore
    kernelData[8] = secondarySurround;   // Destra-superiore
    kernelData[10] = secondarySurround;  // Sinistra
    kernelData[14] = secondarySurround;  // Destra
    kernelData[16] = secondarySurround;  // Sinistra-inferiore
    kernelData[18] = secondarySurround;  // Destra-inferiore
    kernelData[22] = secondarySurround;  // Inferiore

    return true;
}

/**
 * @brief Inizializza un kernel con un valore centrale e valori circostanti
 *
 * Metodo di utilità per creare kernel 3x3 con un valore centrale
 * e valori uniformi intorno
 *
 * @param kernel Vettore del kernel da inizializzare
 * @param centerValue Valore per il pixel centrale
 * @param surroundValue Valore per i pixel circostanti
 * @param kernelSize Dimensione del kernel
 * @return true se l'inizializzazione ha successo
 */
bool FilterKernel::initializeKernel(std::vector<float>& kernel, float centerValue, float surroundValue, int kernelSize) {
    if (kernelSize % 2 == 0 || kernelSize < 3) {
        std::cerr << "Errore: dimensione kernel non valida" << std::endl;
        return false;
    }

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            if (i == kernelSize / 2 && j == kernelSize / 2) {
                kernel[j + i * kernelSize] = centerValue;
            }
            else {
                kernel[j + i * kernelSize] = surroundValue;
            }
        }
    }

    return true;
}

/**
 * @brief Restituisce la dimensione del kernel
 */
int FilterKernel::getSize() const {
    return size;
}

/**
 * @brief Restituisce i dati del kernel come vettore
 */
std::vector<float> FilterKernel::getKernelData() const {
    return kernelData;
}