#include "image_processor.h"
#include <omp.h>
#include <algorithm>
#include <iostream>

bool ImageProcessor::applyFilterOpenMP(
    ImageProcessor& output, const FilterKernel& filter, int numThreads) const
{
    int kernelSize = filter.getSize();
    int radius = kernelSize / 2;
    std::vector<float> kernel = filter.getKernelData();

    // Crea l'immagine con padding
    auto padded = createPaddedImage(radius, radius);
    std::vector<float> result(width * height);

    // Imposta il numero di thread
    omp_set_num_threads(numThreads);

    // Applica il filtro in parallelo usando OpenMP
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    int px = x + kx + radius;
                    int py = y + ky + radius;
                    float pixelValue = padded[py * (width + 2 * radius) + px];
                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    sum += pixelValue * kernelValue;
                }
            }
            result[y * width + x] = std::min(255.0f, std::max(0.0f, sum));
        }
    }

    output.setImageData(result, width, height);
    return true;
}