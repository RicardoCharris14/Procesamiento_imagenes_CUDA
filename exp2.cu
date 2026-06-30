#define cimg_display 0
#define cimg_use_jpeg
#define cimg_use_png

#include "CImg.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>

using namespace cimg_library;

constexpr int NUM_IMAGES = 100; 
constexpr int WIDTH = 100, HEIGHT = 100, CHANNELS = 3;
constexpr int IMG_RES = WIDTH * HEIGHT;
constexpr int IMG_SIZE = WIDTH * HEIGHT * CHANNELS;
constexpr int DATASET_PIXELS_AMOUNT = IMG_SIZE * NUM_IMAGES;
constexpr int CVM_SIZE = IMG_SIZE * IMG_SIZE;

__global__ void accumulate_pixel_value(float* images, float* mean_vector, int n_images){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pos_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (pos_x >= WIDTH || pos_y >= HEIGHT || pos_z >= CHANNELS) return;

    int pixel_pos = (IMG_RES * pos_z) + (pos_y * WIDTH + pos_x);

    float sum = 0.0f;
    for (int i=0 ; i<n_images ; i++){
        int pixel = i * IMG_SIZE + (pixel_pos);

        if (pixel >= DATASET_PIXELS_AMOUNT) break;

        sum += images[pixel];
    }

    atomicAdd(&mean_vector[pixel_pos], sum);
}

__global__ void divide_mean_vector(float* mean_vector){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos_x >= IMG_SIZE) return;

    mean_vector[pos_x] /= NUM_IMAGES;
}

__global__ void center_images(float* images, float* mean_vector, int n_images){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pos_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (pos_x >= WIDTH || pos_y >= HEIGHT || pos_z >= CHANNELS) return;

    int pixel_pos = (pos_z * IMG_RES) + (pos_y * WIDTH + pos_x);

    float mean = mean_vector[pixel_pos];

    for (int i=0 ; i<n_images ; i++){
        int pixel = (i * IMG_SIZE) + pixel_pos;

        if (pixel >= DATASET_PIXELS_AMOUNT) return;

        images[pixel] -= mean;
    }
}

__global__ void covariance_matrix(float* images, float* matrix, int n_pixels_j, int s){
    int pos_j_prime = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_j = blockIdx.y * blockDim.y + threadIdx.y;

    int real_pos_j = s * n_pixels_j + pos_j;

    if (pos_j_prime >= IMG_SIZE || pos_j >= n_pixels_j) return;

    if ((real_pos_j * IMG_SIZE + pos_j_prime) >= CVM_SIZE) return;

    float sum = 0.0f;
    for (int i=0 ; i<NUM_IMAGES ; i++){
        sum += images[i * IMG_SIZE + real_pos_j] * images[i * IMG_SIZE + pos_j_prime];
    }

    matrix[IMG_SIZE * pos_j + pos_j_prime] = sum / NUM_IMAGES;
}

void load_images(float* h_dataset){
    // Bucle para cargar y aplanar las imagenes
    for (int k = 0; k < NUM_IMAGES; k++) {
        char buffer[100];
        std::sprintf(buffer, "dataset/%04dx4.png", 800 + k + 1);
        std::string filename(buffer);
        CImg<unsigned char> img(filename.c_str());
        img.resize(WIDTH, HEIGHT, 1, CHANNELS);

        // Puntero al inicio del bloque de la imagen actual
        float* current_img_ptr = h_dataset + (k * IMG_SIZE);

        // Pasar de unsigned char a float
        std::copy(img.data(), img.data() + IMG_SIZE, current_img_ptr);
    }
}

void calculate_mean_vector(float* h_dataset, float* d_dataset, float* d_mean_vector, cudaStream_t* streams, int img_chunk, int n_streams){
    int x_t = 8, y_threads = 8, z_threads = 3;
    dim3 block(x_t, y_threads, z_threads), grid((WIDTH + x_t - 1) / x_t, (HEIGHT + y_threads - 1) / y_threads, (CHANNELS + z_threads - 1) / z_threads);

    // Acumulación de los valores de los pixeles usando streams
    for (int s = 0 ; s < n_streams ; s++){
        int offset = s * img_chunk * IMG_SIZE;
        
        cudaMemcpyAsync(d_dataset + offset, h_dataset + offset, img_chunk * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        accumulate_pixel_value<<<grid, block, 0, streams[s]>>>(d_dataset + offset, d_mean_vector, img_chunk);
    }

    // Esperar a que todos los streams terminen de acumular los valores de los pixeles
    for (int s = 0 ; s < n_streams ; s++){
        cudaStreamSynchronize(streams[s]);
    }

    int n_threads = 256; 
    int blocks_1d = (IMG_SIZE + n_threads - 1) / n_threads;
    
    // Calculo del vector promedio
    divide_mean_vector<<<blocks_1d, n_threads>>>(d_mean_vector);
    
    cudaDeviceSynchronize();
}

void center_images(float* h_mean_vector, float* d_dataset, float* d_mean_vector, cudaStream_t* streams, int img_chunk, int n_streams){
    int x_threads = 8, y_threads = 8, z_threads = 3;
    dim3 block(x_threads, y_threads, z_threads);
    dim3 grid((WIDTH + x_threads - 1) / x_threads, (HEIGHT + y_threads - 1) / y_threads, (CHANNELS + z_threads - 1) / z_threads);

    // Obtención de las imagenes centradas
    for (int s = 0 ; s < n_streams ; s++){
        int offset = s * img_chunk * IMG_SIZE;

        center_images<<<grid, block, 0, streams[s]>>>(d_dataset + offset, d_mean_vector, img_chunk);
    }

    for (int s = 0 ; s < n_streams ; s++){
        cudaStreamSynchronize(streams[s]);
    }
}

void calculate_cvmatrix(float* h_cv_matrix, float* d_dataset, float* d_cv_matrix, cudaStream_t* streams, int cvm_chunk, int n_streams){
    int x_threads = 16, y_threads = 16;
    int x_blocks = (IMG_SIZE + x_threads - 1) / x_threads, y_blocks = (cvm_chunk + y_threads - 1) / y_threads;
    dim3 block(x_threads, y_threads), grid(x_blocks, y_blocks);

    for (int s=0 ; s<n_streams ; s++){
        int offset = s * cvm_chunk * IMG_SIZE;

        covariance_matrix<<<grid, block, 0, streams[s]>>>(d_dataset, d_cv_matrix + offset, cvm_chunk, s);
        cudaMemcpyAsync(h_cv_matrix + offset, d_cv_matrix + offset, cvm_chunk * IMG_SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
    }
    
    for (int s = 0 ; s < n_streams ; s++){
        cudaStreamSynchronize(streams[s]);
    }
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

    // Eventos para medir tiempo
    cudaEvent_t start_calculation, end_calculation;

    // Asignar memoria paginada en la CPU
    cudaMallocHost(&h_dataset, DATASET_PIXELS_AMOUNT * sizeof(float));
    cudaMallocHost(&h_mean_vector, IMG_SIZE * sizeof(float));
    cudaError_t err = cudaMallocHost(&h_cv_matrix, CVM_SIZE * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocHost h_cv_matrix falló: " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }

    // Asignar memoria en GPU para las imagenes
    cudaMalloc(&d_dataset, DATASET_PIXELS_AMOUNT * sizeof(float));
    cudaMalloc(&d_mean_vector, IMG_SIZE * sizeof(float));
    err = cudaMalloc(&d_cv_matrix, CVM_SIZE * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_cv_matrix falló: " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }

    // Inicializamos el vector promedio con ceros
    cudaMemset(d_mean_vector, 0, IMG_SIZE * sizeof(float));

    cudaEventCreate(&start_calculation);
    cudaEventCreate(&end_calculation);

    // Cargamos las imagenes en el host
    load_images(h_dataset);

    // AQUI VA SU CODIGO CUDA:
    // (Cálculo del promedio , centrado y Matriz de Covarianza)

    // Creación de los streams de ejecución
    cudaStream_t streams[N_STREAMS];
    for (int s = 0; s < N_STREAMS ; s++){
        cudaStreamCreate(&streams[s]);
    }

    cudaEventRecord(start_calculation, 0);

    calculate_mean_vector(h_dataset, d_dataset, d_mean_vector, streams, IMG_CHUNK, N_STREAMS);
    center_images(h_mean_vector, d_dataset, d_mean_vector, streams, IMG_CHUNK, N_STREAMS);
    calculate_cvmatrix(h_cv_matrix, d_dataset, d_cv_matrix, streams, CVM_CHUNK, N_STREAMS);

    cudaEventRecord(end_calculation, 0);
    cudaEventSynchronize(end_calculation);

    float calculation_time = 0.0f;
    cudaEventElapsedTime(&calculation_time, start_calculation, end_calculation);

    // Liberación de la memoria que ya no se usará
    cudaFreeHost(h_mean_vector);
    cudaFreeHost(h_dataset);
    cudaFreeHost(h_cv_matrix);

    cudaFree(d_mean_vector);
    cudaFree(d_dataset);
    cudaFree(d_cv_matrix);
    
    cudaEventDestroy(start_calculation);
    cudaEventDestroy(end_calculation);

    std::cout << "Tiempo total de procesamiento: " << calculation_time << std::endl;

    // Nombre de la carpeta y archivo de salida
    std::string folder_name = "data";
    std::string file_path = folder_name + "/resultados_rendimiento.csv";

    // Crear la carpeta "data" si no existe
    #if defined(_WIN32)
        _mkdir(folder_name.c_str());
    #else
        mkdir(folder_name.c_str(), 0777);
    #endif

    // Comprobar si el archivo ya existe para saber si escribir la cabecera
    std::ifstream check_file(file_path);
    bool file_exists = check_file.good();
    check_file.close();

    // Abrir el archivo CSV en modo "append" (añadir al final)
    std::ofstream csv_file(file_path, std::ios::app);

    if (csv_file.is_open()) {
        // Si el archivo es nuevo, escribimos los nombres de las columnas
        if (!file_exists) {
            csv_file << "Streams,ancho,alto,canales,Tiempo_ms\n";
        }

        // Guardar las variables del experimento actual (ejemplo usando tus constantes de exp2.cu)
        csv_file << N_STREAMS << ","
                << WIDTH << ","
                << HEIGHT << ","
                << CHANNELS << ","
                << calculation_time << "\n";

        csv_file.close();
        std::cout << "✔ Datos guardados exitosamente en: " << file_path << std::endl;
    } else {
        std::cerr << "❌ Error: No se pudo abrir o crear el archivo CSV." << std::endl;
    }
    
    return 0;
}