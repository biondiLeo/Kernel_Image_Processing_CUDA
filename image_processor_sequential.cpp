#include "image_processor.h"
#include <algorithm>
#include <iostream>

/**
 * @brief Applica un filtro all'immagine in modo sequenziale
 *
 * Implementa l'algoritmo di convoluzione in modo sequenziale, elaborando
 * un pixel alla volta. Utilizza un'immagine con padding per gestire
 * correttamente i bordi dell'immagine.
 *
 * @param output Immagine di destinazione dove verrà salvato il risultato
 * @param filter Kernel del filtro da applicare
 * @return true se l'operazione viene completata con successo
 */
bool ImageProcessor::applyFilterSequential(
    ImageProcessor& output, const FilterKernel& filter) const
{
    // Ottiene le dimensioni del kernel e calcola il raggio per il padding
    int kernelSize = filter.getSize();
    int radius = kernelSize / 2;
    std::vector<float> kernel = filter.getKernelData();

    // Crea una versione dell'immagine con padding per gestire i bordi
    // Il padding evita artefatti ai bordi dell'immagine durante la convoluzione
    auto padded = createPaddedImage(radius, radius);

    // Vettore per memorizzare il risultato dell'elaborazione
    std::vector<float> result(width * height);

    // Elabora sequenzialmente ogni pixel dell'immagine
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Accumula il risultato della convoluzione per il pixel corrente
            float sum = 0.0f;

            // Applica il kernel di convoluzione al pixel corrente
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    // Calcola le coordinate nel buffer con padding
                    int px = x + kx + radius;
                    int py = y + ky + radius;

                    // Recupera il valore del pixel dall'immagine con padding
                    float pixelValue = padded[py * (width + 2 * radius) + px];

                    // Recupera il valore corrispondente dal kernel
                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];

                    // Accumula il prodotto per la convoluzione
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

/**
 * @brief Versione semplificata che applica il filtro direttamente sull'immagine corrente
 *
 * Questa è una versione di convenienza che crea un'immagine temporanea
 * per l'elaborazione e poi aggiorna l'immagine corrente con il risultato.
 * Utilizza la versione const del metodo applyFilterSequential internamente.
 *
 * @param filter Kernel del filtro da applicare
 * @return true se l'operazione viene completata con successo
 */
bool ImageProcessor::applyFilterSequential(const FilterKernel& filter) {
    // Crea un'immagine temporanea per il risultato
    ImageProcessor output;

    // Applica il filtro usando la versione const del metodo
    if (applyFilterSequential(output, filter)) {
        // Aggiorna l'immagine corrente con il risultato
        setImageData(output.getImageData(), output.getWidth(), output.getHeight());
        return true;
    }
    return false;
}