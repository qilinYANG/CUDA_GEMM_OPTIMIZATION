
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include <cmath>

 #define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "      \
                << cudaGetErrorString(status_) << '\n';                         \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t status_ = (call);                                             \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::cerr << "cuBLAS error " << static_cast<int>(status_) << " at "        \
                << __FILE__ << ':' << __LINE__ << '\n';                         \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

#define THREAD_PER_BLOCK_X 32
#define THREAD_PER_BLOCK_Y 32


constexpr int WARM_UP = 1;
__global__ void naive_gemm_kernel(const float* __restrict__ A, 
                                  const float* __restrict__ B, 
                                  float* __restrict__ C,
                                  int M, int N, int K)
{

    int myCol = threadIdx.x + blockDim.x * blockIdx.x;
    int myRow = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (myRow >= M || myCol >= N)
        return;

    float val = 0.0;
    for(int idx = 0; idx < K; idx++){
        val += A[myRow * K + idx] * B[idx * N + myCol];
    }

    
    C[myRow * N +myCol] = val;

}


int round_up(int x, int multiple){
    return ((x + multiple - 1) / multiple) * multiple;
}

float benchmark_naive_gemm(const float* A, const float* B, float* C,
                        int M, int N, int K, int iterations){

    dim3 threads( THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, 1);
    dim3 blocks( ( N  + THREAD_PER_BLOCK_X - 1) / THREAD_PER_BLOCK_X ,
                ( M  + THREAD_PER_BLOCK_Y - 1) / THREAD_PER_BLOCK_Y , 1 );

    for(int i = 0; i < WARM_UP; i++){
        naive_gemm_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < iterations; i++){
        naive_gemm_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iterations;

}

float benchmark_cublas(cublasHandle_t handle, const float* A, const float* B,
    float* C, int M, int N, int K, int iterations) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major. Row-major C=A*B is evaluated as C^T=B^T*A^T.
    auto launch = [&]() {
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        B, N,   // swapped: B first
        A, K,
        &beta, C, N));
    };
    

    for (int i = 0; i < WARM_UP; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / iterations;
}

bool compare(const std::vector<float>& got, const std::vector<float>& ref,
    int M, int N, int leading_dim) {
    
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t failures = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const size_t idx = static_cast<size_t>(i) * leading_dim + j;
            double actual = static_cast<double>(got[idx]);
            double expected = static_cast<double>(ref[idx]);
            const double atol = 2e-5;
            const double rtol = 1e-3;

            if (!std::isfinite(actual) || !std::isfinite(expected)) {
                if (failures < 10) {
                    std::cout << "Non-finite at (" << i << ", " << j << ")"
                              << ": actual=" << actual
                              << ", expected=" << expected << '\n';
                }
                ++failures;
                continue;
            }
            

            const double abs_err = std::abs(actual - expected);
            const double allowed = atol + rtol * std::abs(expected);
            const double rel_err =
            abs_err / std::max(1e-7, std::abs(expected));

            max_abs = std::max(max_abs, abs_err);
            max_rel = std::max(max_rel, rel_err);

            if (abs_err > allowed) {
                if (failures < 10) {
                    std::cout << std::scientific << std::setprecision(9)
                              << "Mismatch at (" << i << ", " << j << ")"
                              << ": actual=" << actual
                              << ", expected=" << expected
                              << ", abs_error=" << abs_err
                              << ", allowed=" << allowed << '\n';
                }
                ++failures;
                        
            }
        }
    }
    std::cout << "Correctness vs cuBLAS: " << (failures == 0 ? "PASS" : "FAIL")
    << " (failed=" << failures << ", max_abs=" << max_abs
    << ", max_rel=" << max_rel << ")\n";
    return failures == 0;
    }


int main( int argc, char *argv[] ){

    int M = argc > 1 ? std::atoi(argv[1]) : 512;
    int N = argc > 2 ? std::atoi(argv[2]) : 512;
    int K = argc > 3 ? std::atoi(argv[3]) : 512;
    int iterations = argc > 4 ? std::atoi(argv[4]) : 10;

    if (M <= 0 || N <= 0 || K <= 0 || iterations <= 0) {
        std::cerr << "Usage: " << argv[0] << " [M N K iterations], all positive\n";
        return EXIT_FAILURE;
    }
 
    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const int Mp = round_up(M, 32);
    const int Np = round_up(N, 32);
    const int Kp = round_up(K, 32);

    const size_t a_count = static_cast<size_t>(Mp) * Kp;
    const size_t b_count = static_cast<size_t>(Kp) * Np;
    const size_t c_count = static_cast<size_t>(Mp) * Np;

    std::vector<float> h_A(a_count, 0.0f);
    std::vector<float> h_B(b_count, 0.0f);
    

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M; ++i)
     for (int k = 0; k < K; ++k)
        h_A[static_cast<size_t>(i) * Kp + k] = dist(rng);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
        h_B[static_cast<size_t>(k) * Np + j] = dist(rng);

    float *d_A, *d_B, *d_C, *d_ref;

    CUDA_CHECK(cudaMalloc(&d_A, a_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, c_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));


    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    const float naive_gemm_ms = benchmark_naive_gemm(d_A, d_B, d_C, Mp, Np, Kp, iterations);
    const float cublas_ms = benchmark_cublas(handle, d_A, d_B, d_ref,
                                            Mp, Np, Kp, iterations);

    std::vector<float> h_C(c_count, 0.0f), h_ref(c_count, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, c_count * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "GPU: " << prop.name << " (SM " << prop.major << '.' << prop.minor << ")\n"
                << "Logical shape: M=" << M << ", N=" << N << ", K=" << K << '\n'
                << "Executed shape: M=" << Mp << ", N=" << Np << ", K=" << Kp << '\n';
    const bool correct = compare(h_C, h_ref, M, N, Np);

    const double logical_flops = 2.0 * M * static_cast<double>(N) * K;
    const double executed_flops = 2.0 * Mp * static_cast<double>(Np) * Kp;
    auto tflops = [](double flops, float ms) { return flops / (ms * 1.0e9); };
    std::cout << std::fixed << std::setprecision(3)
                << " naive gemm: " << naive_gemm_ms << " ms, " << tflops(logical_flops, naive_gemm_ms)
                << " effective TFLOP/s, " << tflops(executed_flops, naive_gemm_ms)
                << " executed TFLOP/s\n"
                << "cuBLAS: " << cublas_ms << " ms, " << tflops(logical_flops, cublas_ms)
                << " effective TFLOP/s\n";

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;


}