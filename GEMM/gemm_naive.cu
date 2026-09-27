
 #include <stdio.h>
 #include <math.h>
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

#define THREAD_PER_BLOCK_X 32
#define THREAD_PER_BLOCK_Y 32

#define M 1024
#define N 1024
#define K 1024

__global__ void naive_gemm_kernel(float *a, float *b, float *c)
{


    int myRow = threadIdx.x + blockDim.x * blockIdx.x;
    int myCol = threadIdx.y + blockDim.y * blockIdx.y;
    

    float val = 0.0;
    for(int idx = 0; idx < K; idx++){
        val += a[myCol * K + idx] * b[idx * M + myRow];
    }

    c[myCol * N +myRow] = val;


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
 
 
    cudaEvent_t start, stop;
    CUDA_CALL( cudaEventCreate( &start) );
    CUDA_CALL( cudaEventCreate( &stop) );

    dim3 threads( THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, 1);
    dim3 blocks( ( N  + THREAD_PER_BLOCK_X - 1) / THREAD_PER_BLOCK_X ,
                 ( M  + THREAD_PER_BLOCK_Y - 1) / THREAD_PER_BLOCK_Y , 1 );

    CUDA_CALL( cudaEventRecord( start, 0) );

    naive_gemm_kernel<<<blocks, threads>>>(d_a, d_b, d_c);
    CUDA_CHECK();
    CUDA_CALL( cudaDeviceSynchronize() );

    CUDA_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_CALL( cudaEventSynchronize( stop ) );
    float elapsedTime;
    CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );

    CUDA_CALL( cudaMemcpy( h_c, d_c, numbytes_c, cudaMemcpyDeviceToHost ) );

    // use cuBLAS for verification 
    cublasHandle_t handle;
    cublasCreate( &handle );

    float alpha = 1.0, beta = 0.0;


    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        d_b, N,   // swapped: B first
        d_a, K,
        &beta, d_c_ref, N);

    cudaDeviceSynchronize();
    cublasDestroy( handle );

    CUDA_CALL( cudaMemcpy(h_c_ref, d_c_ref, numbytes_c, cudaMemcpyDeviceToHost ) );

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