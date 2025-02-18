#include "filter_kernel.h"
#include <iostream>
#include <cmath>
#include <iomanip>

// Definizione di PI se non presente
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FilterKernel::FilterKernel() {
    kernelData.clear();
    size = 0;
}

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
            float value = exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            value /= 2.0f * M_PI * sigma * sigma;

            kernelData[(y + center) * size + (x + center)] = value;
            sum += value;
        }
    }

    // Normalizzazione
    for (int i = 0; i < size * size; i++) {
        kernelData[i] /= sum;
    }

    return true;
}

bool FilterKernel::createSharpeningFilter() {
    size = 3;
    kernelData.resize(size * size);

    return initializeKernel(kernelData, SHARPEN_CENTER, SHARPEN_SURROUND, size);
}

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

bool FilterKernel::createLaplacianFilter() {
    size = 3;
    kernelData.resize(size * size);

    // Inizializza tutti gli elementi a -1
    for (int i = 0; i < size * size; i++) {
        kernelData[i] = LAPLACE_SURROUND;
    }

    // Imposta il valore centrale
    kernelData[4] = LAPLACE_CENTER;  // indice centrale in una matrice 3x3

    // Imposta gli angoli a 0
    kernelData[0] = 0.0f;  // alto-sinistra
    kernelData[2] = 0.0f;  // alto-destra
    kernelData[6] = 0.0f;  // basso-sinistra
    kernelData[8] = 0.0f;  // basso-destra

    return true;
}

bool FilterKernel::createDoGFilter() {
    size = 5;
    kernelData.resize(size * size, 0.0f);  // Inizializza tutto a 0

    // Valori per il filtro Difference of Gaussian
    float centerValue = 16.0f;
    float primarySurround = -2.0f;
    float secondarySurround = -1.0f;

    // Imposta il valore centrale
    kernelData[12] = centerValue;  // Centro della matrice 5x5

    // Imposta i valori primari circostanti
    kernelData[7] = primarySurround;   // Sopra-centro
    kernelData[11] = primarySurround;  // Sinistra-centro
    kernelData[13] = primarySurround;  // Destra-centro
    kernelData[17] = primarySurround;  // Sotto-centro

    // Imposta i valori secondari
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

int FilterKernel::getSize() const {
    return size;
}

std::vector<float> FilterKernel::getKernelData() const {
    return kernelData;
}