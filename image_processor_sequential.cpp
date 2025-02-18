#include "image_processor.h"
#include <algorithm>
#include <iostream>

bool ImageProcessor::applyFilterSequential(
    ImageProcessor& output, const FilterKernel& filter) const
{
    int kernelSize = filter.getSize();
    int radius = kernelSize / 2;
    std::vector<float> kernel = filter.getKernelData();

    // Crea l'immagine con padding
    auto padded = createPaddedImage(radius, radius);
    std::vector<float> result(width * height);

    // Applica il filtro sequenzialmente
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            // Applicazione kernel
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    int px = x + kx + radius;
                    int py = y + ky + radius;
                    float pixelValue = padded[py * (width + 2 * radius) + px];
                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    sum += pixelValue * kernelValue;
                }
            }
            // Normalizzazione risultato
            result[y * width + x] = std::min(255.0f, std::max(0.0f, sum));
        }
    }

    output.setImageData(result, width, height);
    return true;
}

bool ImageProcessor::applyFilterSequential(const FilterKernel& filter) {
    ImageProcessor output;
    if (applyFilterSequential(output, filter)) {
        setImageData(output.getImageData(), output.getWidth(), output.getHeight());
        return true;
    }
    return false;
}