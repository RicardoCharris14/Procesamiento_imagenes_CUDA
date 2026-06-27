#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cimg_library;

__global__ void 

int main() {
    const int N_STREAMS = 4;
    int num_images = 100;
    const int chunk = num_images / N_STREAMS;

    int width = 100, height = 100, channels = 3;
    int img_size = width * height * channels;

    // Asignar memoria paginada bloqueada (Pinned Memory) para Streams
    float* h_dataset;
    float* d_dataset;

    // Asignar memoria paginada para las imagenes
    cudaMallocHost(&h_dataset, img_size * sizeof(float) * num_images);

    // Asignar memoria en GPU para las imagenes
    cudaMalloc(&d_dataset, img_size * sizeof(float) * num_images);

    // Bucle para cargar y aplanar las imagenes
    for (int k = 0; k < num_images; k++) {
        std::string filename = "data/DIV2K_valid_LR_bicubic/080" + std::to_string(k+1) + "x4.png";
        CImg<unsigned char> img(filename.c_str());
        img.resize(width, height, 1, channels);

        // Puntero al inicio del bloque de la imagen actual
        float* current_img_ptr = h_dataset + (k * img_size);

        // Pasar de unsigned char a float
        std::copy(img.data(), img.data() + img_size, current_img_ptr);
    }

    // AQUI VA SU CODIGO CUDA:
    // (Cálculo del promedio , centrado y Matriz de Covarianza)
    cudaStream_t streams[N_STREAMS];
    for (int s = 0; s < N_STREAMS ; s++){
        cudaStreamCreate(&streams[s]);
    }
    int offset;
    for (int s = 0 ; s < N_STREAMS ; s++){
        offset = s * chunk * img_size;
        cudaMemcpyAsync(d_dataset + offset, h_dataset + offset, chunk * img_size * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
    }

    // cudaFreeHost(h_dataset);
    return 0;
}