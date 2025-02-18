#ifndef FILTER_KERNEL_H
#define FILTER_KERNEL_H

#include <vector>

// Definiamo le costanti come constexpr
constexpr float SHARPEN_CENTER = 9.0f;
constexpr float SHARPEN_SURROUND = -1.0f;
constexpr float EDGE_CENTER = 8.0f;
constexpr float EDGE_SURROUND = -1.0f;
constexpr float LAPLACE_CENTER = 4.0f;
constexpr float LAPLACE_SURROUND = -1.0f;

class FilterKernel {
public:
    FilterKernel();
    ~FilterKernel() {
        kernelData.clear();
        std::vector<float>().swap(kernelData);
    }

    // Metodi di visualizzazione
    void displayKernel() const;

    // Creazione filtri predefiniti
    bool createGaussianFilter(int size, float sigma);
    bool createSharpeningFilter();
    bool createEdgeDetectionFilter();
    bool createLaplacianFilter();
    bool createDoGFilter(); // Difference of Gaussian

    // Getter per le proprietà del kernel
    int getSize() const;
    std::vector<float> getKernelData() const;

private:
    // Metodo helper per la costruzione dei kernel
    bool initializeKernel(std::vector<float>& kernel, float centerValue, float surroundValue, int size);

    std::vector<float> kernelData;  // Matrice del filtro linearizzata
    int size;                       // Dimensione del kernel (assumiamo kernel quadrato)
};

#endif // FILTER_KERNEL_H