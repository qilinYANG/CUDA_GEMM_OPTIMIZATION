#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>

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



// Reduction depth loaded into shared memory per iteration.
constexpr int K_TILES = 32;

// Thread-block dimensions; the launch must match these values.
constexpr int TDS_PER_BLOCK_X = 16;
constexpr int TDS_PER_BLOCK_Y = 16;
constexpr int WARM_UP = 1;

// Each thread owns this many output rows and consecutive output columns.
#define TILES_PER_TDS_M 4
#define TILES_PER_TDS_N 4



// One block computes a BM x BN output tile (currently 128 x 128).
constexpr int BM = TDS_PER_BLOCK_Y * TILES_PER_TDS_M;
constexpr int BN = TDS_PER_BLOCK_X * TILES_PER_TDS_N;

// orchestrate each warp into 4 * 8 logical microtile 
constexpr int WARP_TILES_M = 4;
constexpr int WARP_TILES_N = 8;
    
constexpr int WAPRS_PER_ROW = TDS_PER_BLOCK_X / WARP_TILES_N;

// Row-major GEMM: C[M,N] = A[M,K] * B[K,N].
// Threads cooperatively stage A/B tiles, then reuse them to accumulate private
// output tiles. With the current settings, each thread computes 4 x 4 outputs.
__global__ void gemm_register_tiling(const float* __restrict__ A, 
                                     const float* __restrict__ B, 
                                     float* __restrict__ C,
                                     int M, int N, int K)
{

    // Round up so a partial final K tile is included and zero-padded.
    int num_tiles = (K + K_TILES - 1) / K_TILES;

    // Stride for distributing shared-tile elements across the block.
    int tds_per_blk = TDS_PER_BLOCK_X * TDS_PER_BLOCK_Y;

    // Top-left output coordinate owned by this block.
    int g_m_start = blockIdx.y * BM;
    int g_n_start = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;
    int warp = tid / 32;


    int lane = tid % 32;

    int tile_m = (warp / WAPRS_PER_ROW) * WARP_TILES_M + lane / WARP_TILES_N;
    int tile_n = (warp % WAPRS_PER_ROW) * WARP_TILES_N + lane % WARP_TILES_N;

    // Initialize once: partial sums persist across every K tile.
    float acc[TILES_PER_TDS_M][TILES_PER_TDS_N] = {0.0f};

    for(int tileIdx = 0; tileIdx < num_tiles; tileIdx++ )
    {
        // Reused each iteration; total storage is (BM + BN) * K_TILES floats.


        extern __shared__ float smem[];

        float (*smemA)[K_TILES + 1] =
            reinterpret_cast<float (*)[K_TILES + 1]>(smem);

        float (*smemB)[BN] =
            reinterpret_cast<float (*)[BN]>(smem + BM * (K_TILES + 1));        
        
        

        // The strided loop covers A exactly once, even with uneven work per thread.
        // Its bound protects shared memory; the predicate zero-pads global edges.
        for(int idx = tid; idx < BM * K_TILES; idx += tds_per_blk)
        {
            int a_r = idx / K_TILES;
            int a_c = idx % K_TILES;

            int global_m = g_m_start + a_r;
            int global_k = K_TILES * tileIdx + a_c;

            smemA[a_r][a_c] = (global_m < M && global_k < K)
                ? A[global_m * K + global_k]
                : 0.0f;
        }

        // Stage the corresponding K_TILES x BN tile of B in row-major order.
        for(int idx = tid; idx < K_TILES * BN; idx += tds_per_blk)
        {
            int b_r = idx / BN;
            int b_c = idx % BN;

            int global_k = K_TILES * tileIdx + b_r;
            int global_n = g_n_start + b_c;

            smemB[b_r][b_c] = (global_k < K && global_n <N)
                ? B[global_k * N + global_n]
                : 0.0f;

        }
        
        // All tile writes must finish before any thread consumes shared data.
        __syncthreads();

        // Per-k operands reused across the thread's entire output tile.
        float reg_a[TILES_PER_TDS_M]; float reg_b[TILES_PER_TDS_N];
        
        // resolve bank-conflict with warp re-orchestration
        for(int k = 0; k < K_TILES; k++)
        {
            for(int td_r = 0; td_r < TILES_PER_TDS_M; td_r++)
            {
                reg_a[td_r] = smemA[tile_m * TILES_PER_TDS_M + td_r][k];
            }

            for(int td_c = 0; td_c < TILES_PER_TDS_N; td_c++)
            {
                reg_b[td_c] = smemB[k][tile_n * TILES_PER_TDS_N + td_c];
            }

            // Outer product: each A operand is reused across columns and each
            // B operand across rows (16 multiply-adds from 8 operands here).
            for(int r = 0; r < TILES_PER_TDS_M; r++)
            {
                for(int c = 0; c < TILES_PER_TDS_N; c++)
                {
                    acc[r][c] += reg_a[r] * reg_b[c];
                }
            }
        }

        // Finish all reads before faster threads overwrite the next shared tile.
        __syncthreads();

    }

    for (int r = 0; r < TILES_PER_TDS_M; r++)
    {
        int row = g_m_start + tile_m * TILES_PER_TDS_M + r;
        int col = g_n_start + tile_n * TILES_PER_TDS_N;

    
        if (row < M && col < N)
        {
            
            float* dst = &C[static_cast<size_t>(row) * N + col];

            bool full_vector = col + 3 < N;
            bool aligned =
                (reinterpret_cast<std::uintptr_t>(dst) & (alignof(float4) - 1)) == 0;

            if (full_vector && aligned)
            {
                *reinterpret_cast<float4*>(dst) =
                    make_float4(acc[r][0], acc[r][1],
                                acc[r][2], acc[r][3]);
            }
            else
            {
                #pragma unroll
                for (int cc = 0; cc < TILES_PER_TDS_N; ++cc)
                {
                    if (col + cc < N)
                        dst[cc] = acc[r][cc];
                }
            }
        }
    }


    
}

int round_up(int x, int multiple){
    return ((x + multiple - 1) / multiple) * multiple;
}

float benchmark_reg_tile_gemm(const float* A, const float* B, float* C,
                        int M, int N, int K, int iterations){

                            
    dim3 threads( TDS_PER_BLOCK_X, TDS_PER_BLOCK_Y, 1);
    dim3 blocks( ( N  + BN - 1) / BN ,
                ( M  + BM - 1) / BM , 1 );

    const size_t shared_bytes =
    (static_cast<size_t>(BM) * (K_TILES + 1)
        + static_cast<size_t>(K_TILES) * BN) * sizeof(float);

    for(int i = 0; i < WARM_UP; i++){
        gemm_register_tiling<<<blocks, threads, shared_bytes>>>(A, B, C, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < iterations; i++){
        gemm_register_tiling<<<blocks, threads, shared_bytes>>>(A, B, C, M, N, K);
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

    const float reg_tile_gemm_ms = benchmark_reg_tile_gemm(d_A, d_B, d_C, Mp, Np, Kp, iterations);
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
                << " register-tile gemm: " << reg_tile_gemm_ms << " ms, " << tflops(logical_flops, reg_tile_gemm_ms)
                << " effective TFLOP/s, " << tflops(executed_flops, reg_tile_gemm_ms)
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

