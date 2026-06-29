#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cimg_library;

const int NUM_IMAGES = 100; 
const int WIDTH = 100, HEIGHT = 100, CHANNELS = 3;
const int IMG_SIZE = WIDTH * HEIGHT * CHANNELS;
const int DATASET_PIXELS_AMOUNT = IMG_SIZE * NUM_IMAGES;
const int CVM_SIZE = IMG_SIZE * IMG_SIZE;

__global__ void accumulate_pixel_value(float* images, float* mean_vector, int width, int height, int CHANNELS, int n_images, int images_limit){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pos_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (pos_x >= width || pos_y >= height || pos_z >= CHANNELS) return;

    int IMG_SIZE = width * height;
    int img_pixels_amount = IMG_SIZE * CHANNELS;
    int pixel_pos = (IMG_SIZE * pos_z) + (pos_y * width + pos_x);

    float sum = 0.0f;
    for (int i=0 ; i<n_images ; i++){
        int pixel = i * img_pixels_amount + (pixel_pos);

        if (pixel >= images_limit) break;

        sum += images[pixel];
    }

    atomicAdd(&mean_vector[pixel_pos], sum);
}

__global__ void divide_mean_vector(float* mean_vector, int vector_size, int n_images){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos_x >= vector_size) return;

    mean_vector[pos_x] /= n_images;
}

__global__ void center_images(float* images, float* mean_vector, int width, int height, int CHANNELS, int n_images, int images_limit){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pos_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (pos_x >= width || pos_y >= height || pos_z >= CHANNELS) return;

    int IMG_SIZE = width * height;
    int img_pixels_amount = IMG_SIZE * CHANNELS;
    int pixel_pos = (pos_z * IMG_SIZE) + (pos_y * width + pos_x);

    float mean = mean_vector[pixel_pos];

    for (int i=0 ; i<n_images ; i++){
        int pixel = (i * img_pixels_amount) + pixel_pos;

        if (pixel >= images_limit) return;

        images[pixel] -= mean;
    }
}

__global__ void covariance_matrix(float* images, float* matrix, int img_size, int n_pixels_j, int s, int n_images, int cvm_limit){
    int pos_j_prime = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_j = blockIdx.y * blockDim.y + threadIdx.y;

    int real_pos_j = s * n_pixels_j + pos_j;

    if (pos_j_prime >= img_size || pos_j >= n_pixels_j) return;

    if ((real_pos_j * img_size + pos_j_prime) >= cvm_limit) return;

    float sum = 0.0f;
    for (int i=0 ; i<n_images ; i++){
        sum += images[i * img_size + real_pos_j] * images[i * img_size + pos_j_prime];
    }

    matrix[img_size * pos_j + pos_j_prime] = sum / n_images;
}

float* load_images(float* h_dataset, int width, int height, int CHANNELS, int NUM_IMAGES, int IMG_SIZE){
    // Bucle para cargar y aplanar las imagenes
    for (int k = 0; k < NUM_IMAGES; k++) {
        std::string filename = "dataset/080" + std::to_string(k+1) + "x4.png";
        CImg<unsigned char> img(filename.c_str());
        img.resize(width, height, 1, CHANNELS);

        // Puntero al inicio del bloque de la imagen actual
        float* current_img_ptr = h_dataset + (k * IMG_SIZE);

        // Pasar de unsigned char a float
        std::copy(img.data(), img.data() + IMG_SIZE, current_img_ptr);
    }
}

void calculate_mean_vector(float* h_dataset, float* d_dataset, float* d_mean_vector, cudaStream_t* streams, int img_chunk, int n_streams){
    int x_t = 8, y_t = 8, z_t = 3;
    dim3 block(x_t, y_t, z_t), grid((WIDTH + x_t - 1) / x_t, (HEIGHT + y_t - 1) / y_t, (CHANNELS + z_t - 1) / z_t);

    // Acumulación de los valores de los pixeles usando streams
    for (int s = 0 ; s < n_streams ; s++){
        int offset = s * img_chunk * IMG_SIZE;
        
        cudaMemcpyAsync(d_dataset + offset, h_dataset + offset, img_chunk * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        accumulate_pixel_value<<<grid, block, 0, streams[s]>>>(d_dataset + offset, d_mean_vector, WIDTH, HEIGHT, CHANNELS, img_chunk, DATASET_PIXELS_AMOUNT);
    }

    // Esperar a que todos los streams terminen de acumular los valores de los pixeles
    for (int s = 0 ; s < n_streams ; s++){
        cudaStreamSynchronize(streams[s]);
    }

    int n_threads = 256; 
    int blocks_1d = (IMG_SIZE + n_threads - 1) / n_threads;
    
    // Calculo del vector promedio
    divide_mean_vector<<<blocks_1d, n_threads, 0, streams[s]>>>(d_mean_vector, IMG_SIZE, NUM_IMAGES);
    
    cudaDeviceSynchronize();
}

void center_images(float* h_mean_vector, float* d_dataset, float* d_mean_vector, cudaStream_t* streams, int img_chunk, int n_streams){
    int x_t = 8, y_t = 8, z_t = 3;
    dim3 block(x_t, y_t, z_t), grid((WIDTH + x_t - 1) / x_t, (HEIGHT + y_t - 1) / y_t, (CHANNELS + z_t - 1) / z_t);

    // Obtención de las imagenes centradas
    for (int s = 0 ; s < n_streams ; s++){
        int offset = s * img_chunk * IMG_SIZE;

        center_images<<<grid, block, 0, streams[s]>>>(d_dataset + offset, d_mean_vector, WIDTH, HEIGHT, CHANNELS, img_chunk, DATASET_PIXELS_AMOUNT);
    }

    for (int s = 0 ; s < n_streams ; s++){
        cudaStreamSynchronize(streams[s]);
    }

    // Liberación de la memoria que ya no se usará
    cudaFreeHost(h_mean_vector);
    cudaFree(d_mean_vector);
}

void calculate_cvmatrix(float* h_cv_matrix, float* d_dataset, float* d_cv_matrix, cudaStream_t* streams, int cvm_chunk, int n_streams){
    // Asignación de memoria para la matriz de covarianza
    cudaMallocHost(&h_cv_matrix, CVM_SIZE * sizeof(float));
    cudaMalloc(&d_cv_matrix, CVM_SIZE * sizeof(float));
    
    int n_threads = 16, n_blocks = (IMG_SIZE + n_threads - 1) / n_threads;
    dim3 block(n_threads, n_threads), grid(n_blocks, n_blocks);

    for (int s=0 ; s<n_streams ; s++){
        int offset = s * cvm_chunk * IMG_SIZE;

        covariance_matrix<<<grid, block, 0, streams[s]>>>(d_dataset, d_cv_matrix + offset, IMG_SIZE, cvm_chunk, s, NUM_IMAGES);
        cudaMemcpyAsync(h_cv_matrix + offset, d_cv_matrix + offset, cvm_chunk * IMG_SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
    }
    
    for (int s = 0 ; s < n_streams ; s++){
        cudaStreamSynchronize(streams[s]);
    }

    cudaFree(d_dataset);
    cudaFree(d_cv_matrix);
}

int main(int argc, char* argv[]) {
    if (argc != 3){
        std::cerr << "Correct usage: ./exec -s <N_STREAMS>" << std::endl;
        std::exit(1);
    }

    const int N_STREAMS = std::stoi(argv[2]);
    const int IMG_CHUNK = ((NUM_IMAGES - 1) / N_STREAMS) + 1;
    const int CVM_CHUNK = ((IMG_SIZE - 1) / N_STREAMS) + 1;

    // Asignar memoria paginada bloqueada (Pinned Memory) para Streams
    float *h_dataset, *h_mean_vector, *h_cv_matrix;
    float *d_dataset, *d_mean_vector, *d_cv_matrix;

    // Asignar memoria paginada para las imagenes
    cudaMallocHost(&h_dataset, DATASET_PIXELS_AMOUNT * sizeof(float));
    cudaMallocHost(&h_mean_vector, IMG_SIZE * sizeof(float));

    // Asignar memoria en GPU para las imagenes
    cudaMalloc(&d_dataset, DATASET_PIXELS_AMOUNT * sizeof(float));
    cudaMalloc(&d_mean_vector, IMG_SIZE * sizeof(float));

    // Inicializamos el vector promedio con ceros
    cudaMemset(d_mean_vector, 0, IMG_SIZE * sizeof(float));

    load_images(h_dataset, WIDTH, HEIGHT, CHANNELS, NUM_IMAGES, IMG_SIZE);

    // AQUI VA SU CODIGO CUDA:
    // (Cálculo del promedio , centrado y Matriz de Covarianza)

    // Creación de los streams de ejecución
    cudaStream_t streams[N_STREAMS];
    for (int s = 0; s < N_STREAMS ; s++){
        cudaStreamCreate(&streams[s]);
    }

    calculate_mean_vector(h_dataset, d_dataset, d_mean_vector, streams, IMG_CHUNK, N_STREAMS);
    center_images(h_mean_vector, d_dataset, d_mean_vector, streams, IMG_CHUNK, N_STREAMS);
    calculate_cvmatrix(h_cv_matrix, d_dataset, d_cv_matrix, streams, CVM_CHUNK, N_STREAMS);

    cudaFreeHost(h_dataset);

    return 0;
}