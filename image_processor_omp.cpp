#include "image_processor.h"
#include <omp.h>
#include <algorithm>
#include <iostream>

/**
 * @brief Applica un filtro all'immagine utilizzando parallelizzazione OpenMP
 *
 * Questa implementazione utilizza OpenMP per parallelizzare l'applicazione
 * del filtro dividendo il carico di lavoro tra più thread. L'elaborazione
 * viene eseguita su una versione dell'immagine con padding per gestire
 * correttamente i bordi.
 *
 * @param output Immagine di destinazione dove verrà salvato il risultato
 * @param filter Kernel del filtro da applicare
 * @param numThreads Numero di thread da utilizzare per la parallelizzazione
 * @return true se l'operazione viene completata con successo
 */
bool ImageProcessor::applyFilterOpenMP(
    ImageProcessor& output, const FilterKernel& filter, int numThreads) const
{
    // Ottiene le dimensioni del kernel e calcola il raggio per il padding
    int kernelSize = filter.getSize();
    int radius = kernelSize / 2;
    std::vector<float> kernel = filter.getKernelData();

    // Crea una versione dell'immagine con padding per gestire i bordi
    // Il padding è necessario per applicare il filtro ai pixel di bordo
    auto padded = createPaddedImage(radius, radius);

    // Vettore per memorizzare il risultato dell'elaborazione
    std::vector<float> result(width * height);

    // Configura OpenMP con il numero di thread specificato
    omp_set_num_threads(numThreads);

    /**
     * Direttive OpenMP per la parallelizzazione:
     * - parallel for: divide il lavoro tra i thread
     * - collapse(2): unisce i due cicli for per una migliore distribuzione del carico
     * - schedule(dynamic): assegna dinamicamente i blocchi di lavoro ai thread
     *   per bilanciare meglio il carico
     */
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Accumula il risultato della convoluzione per il pixel corrente
            float sum = 0.0f;

            // Applica il kernel di convoluzione
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    // Calcola le coordinate nel buffer con padding
                    int px = x + kx + radius;
                    int py = y + ky + radius;

                    // Recupera il valore del pixel dall'immagine con padding
                    float pixelValue = padded[py * (width + 2 * radius) + px];

                    // Recupera il valore corrispondente dal kernel
                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];

                    // Accumula il prodotto
                    sum += pixelValue * kernelValue;
                }
            }

            // Limita il risultato all'intervallo [0, 255] e salvalo
            result[y * width + x] = std::min(255.0f, std::max(0.0f, sum));
        }
    }

    // Aggiorna l'immagine di output con il risultato
    output.setImageData(result, width, height);
    return true;
}