#include <stdio.h>
#include <math.h>
#include <stdlib.h>  
#include <string.h>  
#include <cublas_v2.h>

#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK() 
#endif


// Reduction depth loaded into shared memory per iteration.
#define K_TILES 32

// Thread-block dimensions; the launch must match these values.
#define TDS_PER_BLOCK_X 32
#define TDS_PER_BLOCK_Y 16

// Each thread owns this many output rows and consecutive output columns.
#define TILES_PER_TDS_M 8
#define TILES_PER_TDS_N 4

#define M 1024
#define N 1024
#define K 1024

// constexpr int WARMUP_ITERS = 10;
// constexpr int BENCH_ITERS = 100;

// One block computes a BM x BN output tile (currently 128 x 128).
constexpr int BM = TDS_PER_BLOCK_Y * TILES_PER_TDS_M;
constexpr int BN = TDS_PER_BLOCK_X * TILES_PER_TDS_N;

constexpr int WARP_TILES_M = 4;
constexpr int WARP_TILES_N = 8;
    
constexpr int WAPRS_PER_ROW = TDS_PER_BLOCK_X / WARP_TILES_N;

// Row-major GEMM: C[M,N] = A[M,K] * B[K,N].
// Threads cooperatively stage A/B tiles, then reuse them to accumulate private
// output tiles. With the current settings, each thread computes 4 x 4 outputs.
__global__ void gemm_register_tiling(float *a, float *b, float *c)
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
                ? a[global_m * K + global_k]
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
                ? b[global_k * N + global_n]
                : 0.0f;

        }
        
        // All tile writes must finish before any thread consumes shared data.
        __syncthreads();

        // Per-k operands reused across the thread's entire output tile.
        float reg_a[TILES_PER_TDS_M]; float reg_b[TILES_PER_TDS_N];
        
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

        if (row < M)
        {
            float* dst = &c[row * N + col];

            bool full_vector = col + 3 < N;
            bool aligned =
                (reinterpret_cast<uintptr_t>(dst) & (alignof(float4) - 1)) == 0;

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

int main( int argc, char *argv[] ){

    /* Declaing Pointer for Array*/
    float *h_a, *h_b, *h_c, *h_c_ref;
    float *d_a, *d_b, *d_c, *d_c_ref;

    size_t numbytes_a = M * K * sizeof(float);
    size_t numbytes_b = K * N * sizeof(float);
    size_t numbytes_c = M * N * sizeof(float);


    /* Allocating Host Memory */

    h_a = (float *) malloc( numbytes_a );
    if( h_a == NULL )
    {
        fprintf(stderr,"Error in host malloc h_a\n");
        return 911;
    }

    h_b = (float *) malloc( numbytes_b );

    if( h_b == NULL)
    {
        fprintf(stderr, "Error in host malloc h_b\n");
        return 911;
    }
    
    h_c = (float *)malloc( numbytes_c );

    if ( h_c == NULL)
    {

        fprintf(stderr, "Error in host malloc h_b\n");
        return 911;

    }

    h_c_ref = (float *)malloc( numbytes_c );

    if ( h_c_ref == NULL)
    {

        fprintf(stderr, "Error in host malloc h_b\n");
        return 911;

    }

    CUDA_CALL(cudaMalloc( (void**) &d_a, numbytes_a ) );
    CUDA_CALL(cudaMalloc( (void**) &d_b, numbytes_b ) );
    CUDA_CALL(cudaMalloc( (void**) &d_c, numbytes_c ) );

    CUDA_CALL(cudaMalloc( (void**) &d_c_ref, numbytes_c ) );

    memset( h_c, 0, numbytes_c );
    CUDA_CALL( cudaMemset( d_c, 0, numbytes_c ) );

    memset( h_c_ref, 0, numbytes_c );
    CUDA_CALL( cudaMemset( d_c_ref, 0, numbytes_c ) );


    for( int i = 0; i < M * K; i++ )
    {
        h_a[i] = float( rand() ) / ( float(RAND_MAX) + 1.0 );
    } 

    for( int i = 0; i < K * N; i++)
    {
        h_b[i] = float( rand() ) / ( float(RAND_MAX) + 1.0);
    }

    CUDA_CALL( cudaMemcpy( d_a, h_a, numbytes_a, cudaMemcpyHostToDevice ) );
    CUDA_CALL( cudaMemcpy( d_b, h_b, numbytes_b, cudaMemcpyHostToDevice ) );
 
 
    
    dim3 threads( TDS_PER_BLOCK_X, TDS_PER_BLOCK_Y, 1);
    dim3 blocks( ( N + BN - 1) / BN ,
                 ( M + BM - 1) / BM , 1 );



    size_t shared_bytes =
        (size_t(BM) * (K_TILES + 1) + size_t(K_TILES) * BN) * sizeof(float);

    
    int device;
    int max_shared_bytes;
    
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaDeviceGetAttribute(
        &max_shared_bytes,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device));
    printf("max shared bytes: %d KiB\n", max_shared_bytes/1024);

    if (shared_bytes > static_cast<size_t>(max_shared_bytes))
    {
        fprintf(stderr,
                "Requested %zu shared bytes, device supports %d\n",
                shared_bytes, max_shared_bytes);
        return 1;
    }
    
    CUDA_CALL(cudaFuncSetAttribute(
        gemm_register_tiling,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));

    // for (int i = 0; i < WARMUP_ITERS; i++)
    // {
    //     gemm_register_tiling<<<blocks, threads, shared_bytes>>>(
    //         d_a, d_b, d_c);;
    // }
    CUDA_CALL( cudaDeviceSynchronize() );

    cudaEvent_t start, stop;
    CUDA_CALL( cudaEventCreate( &start) );
    CUDA_CALL( cudaEventCreate( &stop) );

    CUDA_CALL( cudaEventRecord( start, 0) );

    // for(int i = 0; i < BENCH_ITERS; i++)
    // {
    //     gemm_register_tiling<<<blocks, threads, shared_bytes>>>(
    //         d_a, d_b, d_c);
        
    // }
    gemm_register_tiling<<<blocks, threads, shared_bytes>>>(
        d_a, d_b, d_c);
    
    CUDA_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_CALL( cudaEventSynchronize( stop ) );


    // float total_ms;
    // CUDA_CALL( cudaEventElapsedTime( &total_ms, start, stop ) );

    // float elapsedTime = total_ms / BENCH_ITERS;
    // fprintf(stdout, "Average Kernel duration is %f sec\n", elapsedTime / 1000.0f );

    // float flops = 2.0f * float(M) * float(N) * float(K) * BENCH_ITERS;

    // float tflops = flops / (total_ms * 1.0e9);
    // printf("computing speed : %f TFLOPS\n", tflops);

    

    CUDA_CALL( cudaMemcpy( h_c, d_c, numbytes_c, cudaMemcpyDeviceToHost ) );

    // use cuBLAS for verification 
    cublasHandle_t handle;
    cublasCreate( &handle );

    float alpha = 1.0, beta = 0.0;

    /* run reference kernel*/
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        d_b, N,   // swapped: B first
        d_a, K,
        &beta, d_c_ref, N);

    cudaDeviceSynchronize();
    cublasDestroy( handle );

    CUDA_CALL( cudaMemcpy(h_c_ref, d_c_ref, numbytes_c, cudaMemcpyDeviceToHost ) );

    /* result check*/
    for( int col = 0; col < M; col++ )
    {
     for( int row = 0; row < N; row++ )
     {
        float actual = h_c[col*N+row];
        float expected = h_c_ref[col*N+row];

        float diff = fabsf(actual - expected);
        float relErr = diff / (fabsf(expected) + 1e-6f);
        if( !isfinite(actual) || !isfinite(expected) ||relErr > 1e-3f )   // reasonable tolerance for fp32 GEMM with K~1000
        {
            printf("Mismatch in position rowIdx: %d,colIdx: %d\n", col,row );
            printf("gemm_register_tiling %f, reference %f\n",actual, expected);
            printf("FAIL\n");
            goto end;
        }
    
     } 
    } 
 /* free the memory */
   printf("PASS\n");

   end:
   free( h_a );
   free( h_b );
   free( h_c );
   free( h_c_ref );
   CUDA_CALL( cudaFree( d_a ) );
   CUDA_CALL( cudaFree( d_b ) );
   CUDA_CALL( cudaFree( d_c ) );
   CUDA_CALL( cudaFree( d_c_ref ) );
 
   CUDA_CALL( cudaDeviceReset() );
 
   return 0;
 

}

