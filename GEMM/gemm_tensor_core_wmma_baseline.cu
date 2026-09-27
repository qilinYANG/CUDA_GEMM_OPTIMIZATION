// Tensor Core GEMM using CUDA WMMA.
// Computes C[M,N] = A[M,K] * B[K,N] with FP16 inputs and FP32 accumulation.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 GEMM/gemm_tensor_core.cu -lcublas -o tensor_core_gemm
// Run:
//   ./tensor_core_gemm [M N K iterations]

// This asynchronous-copy implementation requires Ampere (SM 8.0) or newer.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace wmma = nvcuda::wmma;

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

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 4;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

constexpr int BM = 2 * WMMA_M * WARPS_M;
constexpr int BN = 2* WMMA_N * WARPS_N;
constexpr int K_MULTIPLE = 3;
constexpr int BK = K_MULTIPLE * WMMA_K;
constexpr int A_STRIDE = BK + 8;
constexpr int B_STRIDE = BN + 8;


constexpr int WARM_UP = 1;


constexpr int A_STAGE_ELEMENTS = BM * A_STRIDE;
constexpr int B_STAGE_ELEMENTS = BK * B_STRIDE;
constexpr int STAGE_ELEMENTS = A_STAGE_ELEMENTS + B_STAGE_ELEMENTS;
constexpr size_t SHARED_BYTES = 2 * STAGE_ELEMENTS * sizeof(half);
static_assert(THREADS_PER_BLOCK <= 1024, "Too many warps per block");
static_assert(BK > 0 && BK % WMMA_K == 0, "BK must contain whole WMMA steps");
static_assert(A_STRIDE % 8 == 0 && B_STRIDE % 8 == 0,
              "16-byte copies and half WMMA loads require aligned strides");
static_assert(STAGE_ELEMENTS % 16 == 0 && A_STAGE_ELEMENTS % 16 == 0,
              "Each WMMA buffer must start on a 32-byte boundary");

// Copy eight half values directly from global to shared memory. Invalid chunks
// are zero-filled. The host pads dimensions to 16, so a chunk is fully in/out.
__device__ __forceinline__ void copy_async_16(half* dst, const half* src,
                                             bool valid) {
#if __CUDA_ARCH__ >= 800
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int source_bytes = valid ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
               :: "r"(address), "l"(src), "r"(source_bytes) : "memory");
#endif
}

// All threads participate, including warps outside the output matrix boundary.
__device__ __forceinline__ void prefetch_tile(
    half* stage, const half* A, const half* B,
    int block_row, int block_col, int k, int M, int N, int K) {
#if __CUDA_ARCH__ >= 800
  for (int chunk = threadIdx.x; chunk < BM * BK / 8; chunk += blockDim.x) {
    const int r = chunk / (BK / 8);
    const int c = (chunk % (BK / 8)) * 8;
    const bool valid = block_row + r < M && k + c < K;
    // Use a valid base pointer even when source_bytes is zero.
    const half* src = valid ? A + static_cast<size_t>(block_row + r) * K + k + c : A;
    copy_async_16(stage + r * A_STRIDE + c, src, valid);
  }
  for (int chunk = threadIdx.x; chunk < BK * BN / 8; chunk += blockDim.x) {
    const int r = chunk / (BN / 8);
    const int c = (chunk % (BN / 8)) * 8;
    const bool valid = k + r < K && block_col + c < N;
    const half* src = valid ? B + static_cast<size_t>(k + r) * N + block_col + c : B;
    copy_async_16(stage + A_STAGE_ELEMENTS + r * B_STRIDE + c, src, valid);
  }
  // Groups and waits are per thread; even threads with no copies commit a group.
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// Each warp owns one 16x16 output tile; the block covers BM x BN (64 x 128).
// A and B are row-major. M/N/K must be positive multiples of 16 (host-padded).
__global__ void tensor_core_gemm_kernel(const half* __restrict__ A,
                                        const half* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "Build this cp.async kernel with -arch=sm_80 or newer"
#endif
#if __CUDA_ARCH__ >= 800
  const int warp_id = threadIdx.x / 32;
  const int warp_m = warp_id / WARPS_N;
  const int warp_n = warp_id % WARPS_N;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int row = block_row + 2 * warp_m * WMMA_M;
  const int col = block_col + 2 * warp_n * WMMA_N;
  const bool valid_warp = row < M && col < N;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                 half, wmma::row_major> a_frag[2][2];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                 half, wmma::row_major> b_frag[2][2];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
        wmma::fill_fragment(c_frag[i][j], 0.0f);

  // Layout: [A0 | B0 | A1 | B1], including the bank-conflict padding.
  extern __shared__ __align__(128) half shared[];
  int current = 0;
  prefetch_tile(shared, A, B, block_row, block_col, 0, M, N, K);
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads(); // Every producer's first tile must be ready for all warps.

  for (int k = 0; k < K; k += BK) {
    const int next = current ^ 1;
    if (k + BK < K) {
      prefetch_tile(shared + next * STAGE_ELEMENTS, A, B,
                    block_row, block_col, k + BK, M, N, K);
    }

    // The next global tile is in flight while Tensor Cores consume this tile.
    const half* shared_A = shared + current * STAGE_ELEMENTS;
    const half* shared_B = shared_A + A_STAGE_ELEMENTS;
    if (valid_warp) {
      
      #pragma unroll
      for (int i = 0; i < 2; ++i){
        wmma::load_matrix_sync(
          a_frag[0][i],
          shared_A + (2 * warp_m + i) * WMMA_M * A_STRIDE,
          A_STRIDE
        );
      }

      #pragma unroll
      for (int j = 0; j < 2; ++j){
        wmma::load_matrix_sync(
          b_frag[0][j],
          shared_B + (2 * warp_n + j) * WMMA_N,
          B_STRIDE);
      }

      #pragma unroll
      for (int rep = 0; rep < K_MULTIPLE; ++rep) {
        const int cur = rep & 1;
        const int next = cur ^ 1;


        if (rep + 1 < K_MULTIPLE){

          const int next_kk = (rep + 1) * WMMA_K;

          #pragma unroll
          for(int i = 0; i < 2; ++i)
          {
            wmma::load_matrix_sync(a_frag[next][i],
              shared_A + (2 * warp_m + i) * WMMA_M * A_STRIDE + next_kk, A_STRIDE);
          
          }

          #pragma unroll
          for(int j = 0; j < 2; ++j)
          {
            wmma::load_matrix_sync(b_frag[next][j],
              shared_B + next_kk * B_STRIDE + (2 * warp_n + j) * WMMA_N, B_STRIDE);

          }
        
        }
        
        #pragma unroll
        for(int i = 0; i < 2; ++ i)
        {
          for (int j = 0; j < 2; ++ j)
          {
            wmma::mma_sync(c_frag[i][j], a_frag[cur][i], b_frag[cur][j], c_frag[i][j]);
          }
        }
        
      }
    }

    asm volatile("cp.async.wait_group 0;" ::: "memory");
    // Publish every thread's copies AND finish all readers before buffer reuse.
    // A cp.async wait alone does not synchronize different threads.
    __syncthreads();
    current = next;
  }
  
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      const int out_row = row + i * WMMA_M;
      const int out_col = col + j * WMMA_N;
      if (out_row < M && out_col < N) {
        wmma::store_matrix_sync(C + static_cast<size_t>(out_row) * N + out_col,
                                c_frag[i][j], N, wmma::mem_row_major);
      }
    }
  }
#endif
}

int round_up(int x, int multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

float benchmark_wmma(const half* A, const half* B, float* C,
                     int M, int N, int K, int iterations) {
  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid((N + BN - 1) / BN,
                  (M + BM - 1) / BM);

  // Opt in to larger allocations when BK is increased. Prefer enough shared
  // memory for both resident blocks; the driver treats carveout as a hint.
  CUDA_CHECK(cudaFuncSetAttribute(tensor_core_gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SHARED_BYTES)));
  CUDA_CHECK(cudaFuncSetAttribute(tensor_core_gemm_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));

  for (int i = 0; i < WARM_UP; ++i)
    tensor_core_gemm_kernel<<<grid, block, SHARED_BYTES>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i)
    tensor_core_gemm_kernel<<<grid, block, SHARED_BYTES>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms / iterations;
}

float benchmark_cublas(cublasHandle_t handle, const half* A, const half* B,
                       float* C, int M, int N, int K, int iterations) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // cuBLAS is column-major. Row-major C=A*B is evaluated as C^T=B^T*A^T.
  auto launch = [&]() {
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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

void compare(const std::vector<float>& got, const std::vector<float>& ref,
             int M, int N, int leading_dim) {
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t failures = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      const size_t idx = static_cast<size_t>(i) * leading_dim + j;
      const double abs_err = std::abs(static_cast<double>(got[idx]) - ref[idx]);
      const double rel_err = abs_err / std::max(1.0e-6, std::abs(static_cast<double>(ref[idx])));
      max_abs = std::max(max_abs, abs_err);
      max_rel = std::max(max_rel, rel_err);
      if (abs_err > 0.05 + 0.005 * std::abs(ref[idx])) ++failures;
    }
  }
  std::cout << "Correctness vs cuBLAS: " << (failures == 0 ? "PASS" : "FAIL")
            << " (failed=" << failures << ", max_abs=" << max_abs
            << ", max_rel=" << max_rel << ")\n";
}

int main(int argc, char** argv) {
  int M = argc > 1 ? std::atoi(argv[1]) : 512;
  int N = argc > 2 ? std::atoi(argv[2]) : 512;
  int K = argc > 3 ? std::atoi(argv[3]) : 512;
  int iterations = argc > 4 ? std::atoi(argv[4]) : 100;
  if (M <= 0 || N <= 0 || K <= 0 || iterations <= 0) {
    std::cerr << "Usage: " << argv[0] << " [M N K iterations], all positive\n";
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 8) {
    std::cerr << "This kernel requires asynchronous copies (compute capability >= 8.0).\n";
    return EXIT_FAILURE;
  }

  // Padding makes every WMMA load/store legal and handles arbitrary M/N/K.
  const int Mp = round_up(M, WMMA_M);
  const int Np = round_up(N, WMMA_N);
  const int Kp = round_up(K, WMMA_K);
  const size_t a_count = static_cast<size_t>(Mp) * Kp;
  const size_t b_count = static_cast<size_t>(Kp) * Np;
  const size_t c_count = static_cast<size_t>(Mp) * Np;

  std::vector<half> h_A(a_count, __float2half(0.0f));
  std::vector<half> h_B(b_count, __float2half(0.0f));
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      h_A[static_cast<size_t>(i) * Kp + k] = __float2half(dist(rng));
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      h_B[static_cast<size_t>(k) * Np + j] = __float2half(dist(rng));

  half *d_A = nullptr, *d_B = nullptr;
  float *d_C = nullptr, *d_ref = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, a_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_B, b_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_C, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ref, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_count * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_count * sizeof(half), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  const float wmma_ms = benchmark_wmma(d_A, d_B, d_C, Mp, Np, Kp, iterations);
  const float cublas_ms = benchmark_cublas(handle, d_A, d_B, d_ref,
                                           Mp, Np, Kp, iterations);

  std::vector<float> h_C(c_count), h_ref(c_count);
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, c_count * sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "GPU: " << prop.name << " (SM " << prop.major << '.' << prop.minor << ")\n"
            << "Logical shape: M=" << M << ", N=" << N << ", K=" << K << '\n'
            << "Executed shape: M=" << Mp << ", N=" << Np << ", K=" << Kp << '\n';
  compare(h_C, h_ref, M, N, Np);

  const double logical_flops = 2.0 * M * static_cast<double>(N) * K;
  const double executed_flops = 2.0 * Mp * static_cast<double>(Np) * Kp;
  auto tflops = [](double flops, float ms) { return flops / (ms * 1.0e9); };
  std::cout << std::fixed << std::setprecision(3)
            << "WMMA:   " << wmma_ms << " ms, " << tflops(logical_flops, wmma_ms)
            << " effective TFLOP/s, " << tflops(executed_flops, wmma_ms)
            << " executed TFLOP/s\n"
            << "cuBLAS: " << cublas_ms << " ms, " << tflops(logical_flops, cublas_ms)
            << " effective TFLOP/s\n";

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_ref));
  return EXIT_SUCCESS;
}
