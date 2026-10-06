// Tensor Core GEMM using XOR-swizzled shared memory and explicit PTX MMA.
// Computes C[M,N] = A[M,K] * B[K,N] with FP16 inputs and FP32 accumulation.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 GEMM/gemm_tensor_core.cu -lcublas -o tensor_core_gemm
// Run:
//   ./tensor_core_gemm [M N K iterations]

// This asynchronous-copy implementation requires Ampere (SM 8.0) or newer.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


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

// Explicit m16n8k16 instructions: each warp computes a 32x32 output tile.
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 4;
constexpr int THREADS_PER_BLOCK = WARPS_M * WARPS_N * 32;
constexpr int BM = WARPS_M * 32;
constexpr int BN = WARPS_N * 32;
// 64 gives eight contiguous 16-byte chunks per A row. Override with
// -DGEMM_BK=48 for a comparison at the baseline's K-stage size.
#ifndef GEMM_BK
#define GEMM_BK 64
#endif
constexpr int BK = GEMM_BK;
constexpr int K_MULTIPLE = BK / MMA_K;
constexpr int A_STRIDE = ((BK + 63) / 64) * 64;
constexpr int B_STRIDE = BN;
constexpr int A_STAGE_ELEMENTS = BM * A_STRIDE;
constexpr int B_STAGE_ELEMENTS = BK * B_STRIDE;
constexpr int STAGE_ELEMENTS = A_STAGE_ELEMENTS + B_STAGE_ELEMENTS;
constexpr size_t SHARED_BYTES = 2 * STAGE_ELEMENTS * sizeof(half);
constexpr int WARM_UP = 1;
static_assert(THREADS_PER_BLOCK <= 1024, "Too many threads");
static_assert(BK > 0 && BK % MMA_K == 0, "BK must contain whole MMA steps");
static_assert(A_STRIDE % 64 == 0 && B_STRIDE % 64 == 0,
              "Swizzle requires complete 64-half panels");
static_assert(A_STAGE_ELEMENTS % 64 == 0 && STAGE_ELEMENTS % 64 == 0,
              "All shared buffer bases must be 128-byte aligned");

// XOR permutes eight 16-byte chunks in each 128-byte row panel.
// Both the copy producer and ldmatrix consumer use this exact mapping.
// Elements within a chunk remain contiguous. Eight matrix rows at a fixed
// logical chunk occupy eight disjoint four-bank groups.
__host__ __device__ constexpr int shared_offset(int r, int c, int stride) {
  return r * stride + (((c / 8) ^ (r & 7)) * 8) + c % 8;
}

__device__ __forceinline__ void copy_async_16(half* dst, const half* src,
                                            bool valid) {
#if __CUDA_ARCH__ >= 800
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int source_bytes = valid ? 16 : 0;
  // Preserve the baseline's .ca policy so cache policy is not another change.
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
               :: "r"(address), "l"(src), "r"(source_bytes) : "memory");
#endif
}

__device__ __forceinline__ void prefetch_tile(
    half* stage, const half* A, const half* B,
    int block_row, int block_col, int k, int M, int N, int K) {
#if __CUDA_ARCH__ >= 800
  for (int chunk = threadIdx.x; chunk < BM * BK / 8; chunk += blockDim.x) {
    const int r = chunk / (BK / 8);
    const int c = (chunk % (BK / 8)) * 8;
    const bool valid = block_row + r < M && k + c < K;
    // Host padding to 16 makes each eight-half chunk entirely valid/invalid.
    const half* src = valid ? A + static_cast<size_t>(block_row + r) * K + k + c : A;
    copy_async_16(stage + shared_offset(r, c, A_STRIDE), src, valid);
  }
  for (int chunk = threadIdx.x; chunk < BK * BN / 8; chunk += blockDim.x) {
    const int r = chunk / (BN / 8);
    const int c = (chunk % (BN / 8)) * 8;
    const bool valid = k + r < K && block_col + c < N;
    const half* src = valid ? B + static_cast<size_t>(k + r) * N + block_col + c : B;
    copy_async_16(stage + A_STAGE_ELEMENTS + shared_offset(r, c, B_STRIDE),
                  src, valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// Each lane holds four packed half2 registers for a logical 16x16 operand.
struct MatrixRegisters { unsigned x[4]; };

__device__ __forceinline__ void load_operands(
    MatrixRegisters (&a)[2], MatrixRegisters (&b)[2],
    const half* sa, const half* sb, int warp_m, int warp_n, int kk, int lane) {
#if __CUDA_ARCH__ >= 800
  // x4 address groups: top-left, bottom-left, top-right, bottom-right 8x8.
  // All 32 lanes execute each collective load, including on boundary tiles.
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int r = warp_m * 32 + i * 16 + (lane & 15);
    int c = kk + (lane / 16) * 8;
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(
        sa + shared_offset(r, c, A_STRIDE)));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a[i].x[0]), "=r"(a[i].x[1]), "=r"(a[i].x[2]), "=r"(a[i].x[3])
        : "r"(addr) : "memory");
  }
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    int r = kk + (lane & 15);
    int c = warp_n * 32 + j * 16 + (lane / 16) * 8;
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(
        sb + shared_offset(r, c, B_STRIDE)));
    // Transpose each 8x8 while loading row-major B into the MMA col operands.
    // Registers [0,1] cover the first N=8 half; [2,3] cover the second.
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(b[j].x[0]), "=r"(b[j].x[1]), "=r"(b[j].x[2]), "=r"(b[j].x[3])
        : "r"(addr) : "memory");
  }
#endif
}

__device__ __forceinline__ void mma_16x8(float* d, const unsigned* a,
                                       const unsigned* b) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// 128x128 block tile, 32x32 per warp; padded FP16 inputs, FP32 accumulators.
// Shared ping-pong overlaps global copies; register ping-pong overlaps LDSM/HMMA.
__global__ void tensor_core_gemm_kernel(const half* __restrict__ A,
                                       const half* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "Build this kernel with -arch=sm_80 or newer"
#endif
#if __CUDA_ARCH__ >= 800
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int row = block_row + warp_m * 32;
  const int col = block_col + warp_n * 32;
  const bool valid_warp = row < M && col < N;
  MatrixRegisters a[2], b[2]; // [pipeline buffer][16x16 sub-tile]
  float acc[2][4][4] = {}; // [M=16 sub-tile][N=8 sub-tile][lane register]
  extern __shared__ __align__(128) half shared[];
  int current = 0;
  prefetch_tile(shared, A, B, block_row, block_col, 0, M, N, K);
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  for (int k = 0; k < K; k += BK) {
    const int next_stage = current ^ 1;
    if (k + BK < K)
      prefetch_tile(shared + next_stage * STAGE_ELEMENTS, A, B,
                    block_row, block_col, k + BK, M, N, K);
    const half* sa = shared + current * STAGE_ELEMENTS;
    const half* sb = sa + A_STAGE_ELEMENTS;
    if (valid_warp) {
      
      #pragma unroll 1
      for (int rep = 0; rep < K_MULTIPLE; ++rep) {
        load_operands(a, b, sa, sb,
                        warp_m, warp_n, rep * MMA_K, lane);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          #pragma unroll
          for (int j = 0; j < 4; ++j){
            mma_16x8(acc[i][j], a[i].x, &b[j / 2].x[(j & 1) * 2]);
          }
            
        }
      }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    // Complete every producer and every current-stage reader before reuse.
    __syncthreads();
    current = next_stage;
  }
  // PTX m16n8k16 accumulator mapping: lane/4 selects a row in each 8-row
  // half; lane%4 selects a pair of adjacent columns. Each output has one owner.
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = row + i * 16 + lane / 4 + h * 8;
        const int c = col + j * 8 + (lane & 3) * 2;
        if (r < M && c + 1 < N) {
          // N is padded to 16; both elements and the float2 pointer are valid.
          *reinterpret_cast<float2*>(C + static_cast<size_t>(r) * N + c) =
              make_float2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
        }
      }
    }
  }
#endif
}

int round_up(int x, int multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

float benchmark_tensor_core(const half* A, const half* B, float* C,
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

bool compare(const std::vector<float>& got, const std::vector<float>& ref,
             int M, int N, int leading_dim) {
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t failures = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      const size_t idx = static_cast<size_t>(i) * leading_dim + j;
      const double abs_err = std::abs(static_cast<double>(got[idx]) - ref[idx]);
      const double rel_err = abs_err / std::max(1.0e-7, std::abs(static_cast<double>(ref[idx])));
      max_abs = std::max(max_abs, abs_err);
      max_rel = std::max(max_rel, rel_err);
      if (!std::isfinite(got[idx]) || !std::isfinite(ref[idx]) ||
          abs_err > 0.05 + 0.005 * std::abs(ref[idx])) ++failures;
    }
  }
  std::cout << "Correctness vs cuBLAS: " << (failures == 0 ? "PASS" : "FAIL")
            << " (failed=" << failures << ", max_abs=" << max_abs
            << ", max_rel=" << max_rel << ")\n";
  return failures == 0;
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

  // Pad to 16 so vector copies are entirely in/out of bounds, including tails.
  const int Mp = round_up(M, MMA_M);
  const int Np = round_up(N, MMA_N); // Preserve 16-half vector-copy boundaries.
  const int Kp = round_up(K, MMA_K);
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

  const float tensor_ms = benchmark_tensor_core(d_A, d_B, d_C, Mp, Np, Kp, iterations);
  const float cublas_ms = benchmark_cublas(handle, d_A, d_B, d_ref,
                                           Mp, Np, Kp, iterations);

  std::vector<float> h_C(c_count), h_ref(c_count);
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
            << "Swizzled MMA: " << tensor_ms << " ms, " << tflops(logical_flops, tensor_ms)
            << " effective TFLOP/s, " << tflops(executed_flops, tensor_ms)
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
