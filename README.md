# Kernel Image Processing in C++/OpenMP/CUDA

This project implements and analyzes three different approaches to kernel image processing: sequential C++, parallel OpenMP for CPU, and CUDA for GPU acceleration.

## Overview

The project focuses on applying various kernel filters to images using different implementation strategies to compare performance and efficiency. It includes comprehensive testing and analysis of each approach under different conditions.

### Key Features

- Multiple implementation approaches:
  - Sequential C++ implementation
  - OpenMP parallel CPU implementation 
  - CUDA GPU implementation with three memory strategies
- Support for various kernel filters:
  - Gaussian Blur
  - Sharpening Filter
  - Edge Detection
  - Laplacian Filter
  - Difference of Gaussian (DoG)
- Performance analysis across different:
  - Image resolutions (512×512 to 8192×8192)
  - Kernel sizes (3×3, 5×5, 7×7)
  - Thread configurations
  - Memory optimization strategies

## Technical Details

### System Requirements

- CPU: Intel i7-7700K (4 cores/8 threads)
- GPU: NVIDIA GeForce RTX 3060
- RAM: 16GB
- Operating System: Windows 10
- CUDA Version: 12.7

### Implementation Details

#### Sequential Implementation
- Core convolution operation using four-nested loop structure
- Row-major memory access pattern
- Independent pixel processing with separate output buffer

#### OpenMP Implementation
- Parallel processing using dynamic scheduling
- collapse(2) clause for increased parallel workload granularity
- Thread-safe design with separate output buffer regions
- Configurable thread count

#### CUDA Implementation
Three distinct memory optimization approaches:

1. Global Memory Version:
   - Baseline implementation using global memory
   - Direct memory access for image and kernel data

2. Constant Memory Version:
   - Kernel coefficients stored in constant memory
   - Optimized for simultaneous kernel value access

3. Shared Memory Version:
   - Uses shared memory for image data caching
   - Two-phase approach with cooperative data loading
   - Block size: 16×16 threads

## Performance Results

### OpenMP Performance

Performance with 3×3 kernel (execution time in ms):

| Resolution | Sequential | 2 threads | 4 threads | 8 threads |
|------------|------------|-----------|-----------|-----------|
| 512²       | 23.86      | 23.42     | 13.77     | 11.91     |
| 8192²      | 6705.15    | 4254.24   | 2828.40   | 2444.93   |

### CUDA Performance

7×7 Kernel on 8192×8192 image processing time (ms):

| Implementation    | Computation | Transfer | Total    |
|------------------|-------------|----------|----------|
| Global Memory    | 151.854     | 247.046  | 398.900  |
| Constant Memory  | 69.371      | 126.100  | 195.471  |
| Shared Memory    | 110.776     | 138.196  | 248.972  |

## Key Findings

### Implementation Performance
- CUDA with constant memory achieved up to 398.90× speedup over sequential processing
- OpenMP showed moderate speedup (up to 3.84× with 8 threads)
- Memory transfer significantly impacts CUDA performance (60-90% reduction)

### Scaling Characteristics
- Better efficiency with larger image sizes
- Larger kernels (5×5, 7×7) show better parallelization benefits
- OpenMP shows diminishing returns beyond 4 threads

### Memory Optimization Impact
- Constant memory: ~54% faster than global memory
- Shared memory: Balanced performance between constant and global
- Block size configuration (16×16) optimizes hardware utilization

## Usage Guidelines

### When to Use Each Implementation

1. Sequential Implementation:
   - Small images (<512×512)
   - Simple kernel operations
   - Systems without parallel processing capabilities

2. OpenMP Implementation:
   - Medium-sized images
   - Systems without GPU
   - Moderate workloads requiring balanced performance

3. CUDA Implementation:
   - Large images (>2048×2048)
   - Complex kernel operations
   - Systems with compatible NVIDIA GPUs
   - Batch processing scenarios

## References

- [Kernel Image Processing Documentation](https://en.wikipedia.org/wiki/Kernel_(image_processing))
- [OpenMP Programming Guide](https://www.openmp.org/resources/refguides/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

