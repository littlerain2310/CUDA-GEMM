#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h> // 1. Thêm thư viện cuBLAS

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error tại dòng " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error tại dòng " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define TILE_WIDTH 16
#define BLOCK_COMPUTE 8

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float ds_A[TILE_WIDTH*BLOCK_COMPUTE][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH*BLOCK_COMPUTE];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH * BLOCK_COMPUTE + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH * BLOCK_COMPUTE + threadIdx.x;

    float sum[BLOCK_COMPUTE][BLOCK_COMPUTE] = {0.0f};

    for (int ph = 0; ph < (N + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        for (int b =0; b < BLOCK_COMPUTE; b++){
            int r_load = row + b * TILE_WIDTH;
            int c_load = col + b * TILE_WIDTH;

            if (r_load < N && (ph * TILE_WIDTH + tx) < N) {
                ds_A[ty + b * TILE_WIDTH][tx] = A[r_load * N + ph * TILE_WIDTH + tx];
            } else {
                ds_A[ty + b * TILE_WIDTH][tx] = 0.0f;
            }

            if (c_load < N && (ph * TILE_WIDTH + ty) < N) {
                ds_B[ty][tx + b * TILE_WIDTH] = B[(ph * TILE_WIDTH + ty) * N + c_load];
            } else {
                ds_B[ty][tx + b * TILE_WIDTH] = 0.0f;
            }

        }
        

        __syncthreads();
        float reg_A[BLOCK_COMPUTE];
        float reg_B[BLOCK_COMPUTE];
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int bi = 0; bi < BLOCK_COMPUTE; bi++) {
                reg_A[bi] = ds_A[ty + bi * TILE_WIDTH][k];
            }
            for (int bj = 0; bj < BLOCK_COMPUTE; bj++) {
                reg_B[bj] = ds_B[k][tx + TILE_WIDTH * bj];
            }
            for (int bi = 0; bi < BLOCK_COMPUTE;bi ++ ){
                for (int bj = 0; bj < BLOCK_COMPUTE;bj ++ ){
                    sum[bi][bj] += reg_A[bi] * reg_B[bj];
                }
            } 
        }
        

        __syncthreads();
    }
    
    for (int bi = 0; bi < BLOCK_COMPUTE;bi ++ ){
        int r =  row + bi* TILE_WIDTH;
        for (int bj = 0; bj < BLOCK_COMPUTE;bj ++ ){
            int c = col + bj *TILE_WIDTH;
            if (r < N && c < N) {
                C[r * N + c] = sum[bi][bj];
            }
        }
    } 
}



// Hàm kiểm tra kết quả của Custom Kernel dựa trên kết quả chuẩn của cuBLAS (nhanh hơn dùng CPU nhiều lần)
void verifyWithCublas(const float* h_C_custom, const float* h_C_cublas, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(h_C_custom[i] - h_C_cublas[i]) > 1e-2f) {
            std::cerr << "SAI LECH tai vi tri " << i << "! Custom: " << h_C_custom[i]
                      << ", cuBLAS: " << h_C_cublas[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "XAC THUC THANH CONG: Ket qua khop voi cuBLAS!" << std::endl;
}

int main() {
    // Tăng N lên 2048 để GPU chạy đủ tải và đo đạc chính xác
    const int N = 2048;
    size_t bytes = N * N * sizeof(float);
    double total_flops = 2.0 * N * N * N;

    std::cout << "Khoi tao ma tran kich thuoc: " << N << "x" << N << std::endl;

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_custom = (float*)malloc(bytes);
    float *h_C_cublas = (float*)malloc(bytes);

    std::srand(std::time(nullptr));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C_custom, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C_custom, bytes));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Khởi tạo các biến Event để bấm giờ trên GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int num_repeats = 20; // Chạy lặp lại 20 lần để lấy thời gian trung bình

    // =========================================================================
    // PHẦN 1: BENCHMARK CUSTOM KERNEL
    // =========================================================================
    int step = TILE_WIDTH * BLOCK_COMPUTE;

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridSize((N + step - 1) / step, (N + step - 1) / step, 1);

    

    // Chạy nháp 1 lần (Warm-up) để đánh thức GPU khỏi trạng thái tiết kiệm điện
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C_custom, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_repeats; ++i) {
        matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C_custom, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float custom_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&custom_ms, start, stop));
    custom_ms /= num_repeats; // Thời gian trung bình 1 lần chạy
    double custom_gflops = total_flops / (custom_ms * 1e6);

    // =========================================================================
    // PHẦN 2: BENCHMARK CUBLAS (TIÊU CHUẨN CỦA NVIDIA)
    // =========================================================================
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Lưu ý quan trọng: C/C++ lưu ma trận theo hàng (Row-major), còn cuBLAS mặc định
    // đọc theo cột (Column-major). Theo tính chất đại số: (A * B)^T = B^T * A^T.
    // Do đó ta truyền d_B trước, d_A sau để cuBLAS trả về đúng ma trận Row-major!
    
    // Warm-up cuBLAS
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_cublas, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_repeats; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_cublas, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    cublas_ms /= num_repeats;
    double cublas_gflops = total_flops / (cublas_ms * 1e6);

    // =========================================================================
    // PHẦN 3: IN BÁO CÁO & KIỂM TRA ĐỘ CHÍNH XÁC
    // =========================================================================
    CUDA_CHECK(cudaMemcpy(h_C_custom, d_C_custom, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C_cublas, bytes, cudaMemcpyDeviceToHost));

    verifyWithCublas(h_C_custom, h_C_cublas, N);

    std::cout << "\n--- KET QUA HIEU NANG (N = " << N << ") ---" << std::endl;
    std::cout << "1. Custom Shared Mem : " << custom_ms << " ms | " << custom_gflops << " GFLOPS" << std::endl;
    std::cout << "2. NVIDIA cuBLAS     : " << cublas_ms << " ms | " << cublas_gflops << " GFLOPS" << std::endl;
    std::cout << "=> Hieu suat dat     : " << (custom_gflops / cublas_gflops) * 100.0 << "% so voi cuBLAS" << std::endl;

    // Dọn dẹp tài nguyên
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_custom));
    CUDA_CHECK(cudaFree(d_C_cublas));
    free(h_A); free(h_B); free(h_C_custom); free(h_C_cublas);

    return 0;
}
