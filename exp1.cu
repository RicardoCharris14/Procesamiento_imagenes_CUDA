// nvcc exp1.cu -o exp1 -lpng -O2

#define cimg_display 0
#define cimg_use_png
#include "CImg.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
using namespace cimg_library;

// Configuracion
static const int NUM_IMAGES = 100;   // m
static const int SIZE_IMAGES = 512;  // tamaño de imagen (para redimensionar)
static const int CHANNELS = 3;       // RGB
static const int SIZE_TILES = 64;    // tamaño de tile para el kernel de covarianza (memoria compartida)
static const std::string DATASET_DIR = "dataset/";


static std::string nombreArchivoDIV2K(int indice) {
    int numero = 801 + indice;
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%04dx4.png", numero);
    return DATASET_DIR + buffer;
}

static void cargarYAplanarImagen(int k, int n, float *h_dataset) {
    std::string filename = nombreArchivoDIV2K(k);
    CImg<unsigned char> img(filename.c_str());

    // Resize al tamaño objetivo 
    img.resize(SIZE_IMAGES, SIZE_IMAGES, 1, CHANNELS, 3);

    // Aplanado 
    long idx = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int y = 0; y < SIZE_IMAGES; y++) {
            for (int x = 0; x < SIZE_IMAGES; x++) {
                unsigned char valor = img(x, y, 0, c);
                h_dataset[idx + (long)k * n] = static_cast<float>(valor);
                idx++;
            }
        }
    }
}

// Kernel 1: Vector promedio.
// Cada thread calcula mu_j para una fila j de V (componente de pixel j), recorriendo las m columnas (imagenes).
__global__ void kernelPromedio(const float *V, float *mu, int n, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    float suma = 0.0f;
    for (int k = 0; k < m; k++) {
        suma += V[j + (long)k * n];
    }
    mu[j] = suma / (float)m;
}

// Kernel 2: Centrado de datos
// Vbar[j + k*n] = V[j + k*n] - mu[j]
// Un thread por elemento de la matriz n x m.
__global__ void kernelCentrado(const float *V, const float *mu,
                                float *Vbar, int n, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // pixel
    int k = blockIdx.y * blockDim.y + threadIdx.y; // indice de imagen

    if (j >= n || k >= m) return;

    long idx = j + (long)k * n;
    Vbar[idx] = V[idx] - mu[j];
}

// Kernel 3: Matriz de covarianza
// C = (1/m) * Vbar * Vbar^T
// Cada bloque de threads calcula un tile de SIZE_TILES x SIZE_TILES de C, 
// Cargando a memoria compartida los tiles correspondientes de A y B por cada 
// "fase" a lo largo de m.
__global__ void kernelCovarianza(const float *Vbar, float *C,
                                       int n, int m) {
    __shared__ float tileA[SIZE_TILES][SIZE_TILES]; // tile de A = Vbar (n x m)
    __shared__ float tileB[SIZE_TILES][SIZE_TILES]; // tile de B = Vbar^T (m x n)

    int fila    = blockIdx.y * SIZE_TILES + threadIdx.y; // j
    int columna = blockIdx.x * SIZE_TILES + threadIdx.x; // j'

    float acumulado = 0.0f;

    int numFases = (m + SIZE_TILES - 1) / SIZE_TILES;

    for (int fase = 0; fase < numFases; fase++) {
        int kBase = fase * SIZE_TILES;

        // Cargar tile de A 
        int kA = kBase + threadIdx.x;
        if (fila < n && kA < m) {
            tileA[threadIdx.y][threadIdx.x] = Vbar[fila + (long)kA * n];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f; // zero padding
        }

        // Cargar tile de B
        int kB = kBase + threadIdx.y;
        if (columna < n && kB < m) {
            tileB[threadIdx.y][threadIdx.x] = Vbar[columna + (long)kB * n];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f; // zero padding
        }

        __syncthreads();

        // Producto parcial dentro del tile
        #pragma unroll
        for (int t = 0; t < SIZE_TILES; t++) {
            acumulado += tileA[threadIdx.y][t] * tileB[t][threadIdx.x];
        }

        __syncthreads();
    }

    if (fila < n && columna < n) {
        C[fila + (long)columna * n] = acumulado / (float)m;
    }
}

int main() {

    const int m = NUM_IMAGES;
    const int n = SIZE_IMAGES * SIZE_IMAGES * CHANNELS;

    std::cout << "m = " << m << ", n = " << n << std::endl;

    size_t bytesDataset = n * m * sizeof(float);
    size_t bytesMean    = n * sizeof(float);
    size_t bytesCov     = n * n * sizeof(float);

    // Reserva en host
    float *h_dataset = new float[n * m];
    float *h_cov     = new float[n * n];

    // Cargar y aplanar imagenes en h_dataset
    for (int k = 0; k < m; k++) {
        cargarYAplanarImagen(k, n, h_dataset);
    }
    std::cout << "Preprocesamiento completo" << std::endl;

    // Reserva en device
    float *d_dataset, *d_mean, *d_centered, *d_cov;
    cudaMalloc((void**)&d_dataset,  bytesDataset);
    cudaMalloc((void**)&d_mean,     bytesMean);
    cudaMalloc((void**)&d_centered, bytesDataset);
    cudaMalloc((void**)&d_cov,      bytesCov);
    std::cout << "Memoria reservada en device" << std::endl;

    // Eventos para medir tiempos
    cudaEvent_t inicioCopiaH2D, finCopiaH2D;
    cudaEvent_t inicioComputo,  finComputo;
    cudaEvent_t inicioCopiaD2H, finCopiaD2H;
    cudaEventCreate(&inicioCopiaH2D);
    cudaEventCreate(&finCopiaH2D);

    cudaEventCreate(&inicioComputo);
    cudaEventCreate(&finComputo);

    cudaEventCreate(&inicioCopiaD2H);
    cudaEventCreate(&finCopiaD2H);

    // Copia del host a device
    cudaEventRecord(inicioCopiaH2D, 0);
    cudaMemcpy(d_dataset, h_dataset, bytesDataset, cudaMemcpyHostToDevice);
    cudaEventRecord(finCopiaH2D, 0);
    cudaEventSynchronize(finCopiaH2D);

    float tiempoCopiaH2D = 0.0f;
    cudaEventElapsedTime(&tiempoCopiaH2D, inicioCopiaH2D, finCopiaH2D);
    std::cout << "Copia del host a device completa en " << tiempoCopiaH2D << " ms" << std::endl;

    // Kernels
    cudaEventRecord(inicioComputo, 0);
    // Kernel 1
    {
        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        kernelPromedio<<<blocks, threadsPerBlock>>>(d_dataset, d_mean, n, m);
        cudaGetLastError();
    }

    // Kernel 2
    {
        dim3 threadsPerBlock(16, 16);
        dim3 blocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
        kernelCentrado<<<blocks, threadsPerBlock>>>(d_dataset, d_mean, d_centered, n, m);
        cudaGetLastError();
    }

    // Kernel 3
    {
        dim3 threadsPerBlock(SIZE_TILES, SIZE_TILES);
        dim3 blocks((n + SIZE_TILES - 1) / SIZE_TILES,
                    (n + SIZE_TILES - 1) / SIZE_TILES);
        kernelCovarianza<<<blocks, threadsPerBlock>>>(d_centered, d_cov, n, m);
        cudaGetLastError();
    }
    cudaEventRecord(finComputo, 0);
    cudaEventSynchronize(finComputo);
    float tiempoComputo = 0.0f;
    cudaEventElapsedTime(&tiempoComputo, inicioComputo, finComputo);
    std::cout << "Computo completo en " << tiempoComputo << " ms" << std::endl;

    // Copia del device a host
    cudaEventRecord(inicioCopiaD2H, 0);
    cudaMemcpy(h_cov, d_cov, bytesCov, cudaMemcpyDeviceToHost);
    cudaEventRecord(finCopiaD2H, 0);
    cudaEventSynchronize(finCopiaD2H);

    float tiempoCopiaD2H = 0.0f;
    cudaEventElapsedTime(&tiempoCopiaD2H, inicioCopiaD2H, finCopiaD2H);
    std::cout << "Copia del device a host completa en " << tiempoCopiaD2H << " ms" << std::endl;

    
    // Liberacion de recursos
    cudaFree(d_dataset);
    cudaFree(d_mean);
    cudaFree(d_centered);
    cudaFree(d_cov);
    cudaEventDestroy(inicioCopiaH2D);
    cudaEventDestroy(finCopiaH2D);
    cudaEventDestroy(inicioComputo);
    cudaEventDestroy(finComputo);
    cudaEventDestroy(inicioCopiaD2H);
    cudaEventDestroy(finCopiaD2H);

    delete[] h_dataset;
    delete[] h_cov;

    return 0;
}
