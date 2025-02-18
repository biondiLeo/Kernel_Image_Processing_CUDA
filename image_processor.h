#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <vector>
#include "filter_kernel.h"

enum class CudaMemoryType
{
    GLOBAL_MEM,
    CONSTANT_MEM,
    SHARED_MEM
};

class ImageProcessor
{
public:
    ImageProcessor();
    ~ImageProcessor();

    // Metodi base per gestione immagine
    int getWidth() const;
    int getHeight() const;
    int getChannels() const;

    // Caricamento e salvataggio
    bool loadImageFromFile(const char* filepath);
    bool saveImageToFile(const char* filepath) const;

    // Gestione dati immagine
    bool setImageData(const std::vector<float>& data, int width, int height);
    std::vector<float> getImageData() const;

    // Applicazione filtri
    bool applyFilterSequential(ImageProcessor& output, const FilterKernel& filter) const;
    bool applyFilterSequential(const FilterKernel& filter);
    bool applyFilterParallel(ImageProcessor& output, const FilterKernel& filter, const CudaMemoryType memType);
    bool applyFilterOpenMP(ImageProcessor& output, const FilterKernel& filter, int numThreads) const;
    bool applyFilterParallelWithTimings(ImageProcessor& output, const FilterKernel& filter, const CudaMemoryType memType, double& computationTime, double& transferTime);


private:
    // Metodi helper
    std::vector<float> applyFilterCore(const FilterKernel& filter) const;
    std::vector<float> createPaddedImage(int paddingY, int paddingX) const;

    std::vector<float> imageData;
    int width;
    int height;
};

#endif
