
#include<iostream>
//#include <cstdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            printf("CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            printf("CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

inline cublasStatus_t cublasGgetrfBatched(cublasHandle_t handle,
                                   int n,
                                   float * Aarray[],
                                   int lda,
                                   int *PivotArray,
                                   int *infoArray,
                                   int batchSize)
{
    return cublasSgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

inline cublasStatus_t cublasGgetrfBatched(cublasHandle_t handle,
                                   int n,
                                   double * Aarray[],
                                   int lda,
                                   int *PivotArray,
                                   int *infoArray,
                                   int batchSize)
{
    return cublasDgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

template<class T>
void cublas_lu(int m, int n, T* a, int *h_pivot)
{
    int rowsA = m;
    int colsA = n;

    // allocate the pivoting vector and the info array
    int *d_pivot_array;

    cudacall(cudaMalloc(&d_pivot_array, n * sizeof(int)));
    int *d_info_array;
    cudacall(cudaMalloc(&d_info_array, sizeof(int)));

    cublasHandle_t handle;
    cublascall(cublasCreate(&handle));

    int matrixSizeA;
    matrixSizeA = rowsA * colsA;

    // Create copy of host A matrix
    T **devPtrA = 0;
    devPtrA =(T **)malloc(1 * sizeof(*devPtrA));
    if (devPtrA == NULL){ 
     perror("malloc"); 
     exit(EXIT_FAILURE); 
    }
	
    cudacall(cudaMalloc(devPtrA, matrixSizeA * sizeof(double)));

    // Create device A mastrix
    T **devPtrA_dev = NULL;
    cudacall(cudaMalloc(&devPtrA_dev, 1 * sizeof(*devPtrA)));

    // Copy address of devPtrA to devPtrA_dev on device
    cudacall(cudaMemcpy(devPtrA_dev, devPtrA, 1 * sizeof(*devPtrA), cudaMemcpyHostToDevice));
   
    // Copy host A matrix data into devPtrA 
    cublascall(cublasSetMatrix(rowsA, colsA, sizeof(a[0]), a, rowsA, devPtrA[0], rowsA));

    // Perform LU decomposition
    cublascall(cublasGgetrfBatched(handle, m, devPtrA_dev, m, d_pivot_array, d_info_array, 1));

    // Copy device LU matrix into host a matrix
    cublascall(cublasGetMatrix(m, n, sizeof(T), devPtrA[0], m, a, m));

    // Copy device pivot matrix into host pivot matrix
    cublascall(cublasGetVector(m, sizeof(int), d_pivot_array, 1, h_pivot, 1));

    return;
}

template<class T>
T determinant(T* A, int n){

    int *h_pivot = (int *)malloc(n * sizeof(int));

    cublas_lu(n, n, A, h_pivot);
#if 1
    printf("Combined LU matrix (pivot applied):\n");
    for(int j=0; j<n; j++)
    {
        for(int i=0; i<n; i++)
            printf("%f\t",A[i*n+j]);
        printf("\n");
    }					
#endif
   T det = A[0];
   for(int i = 1; i < n; ++i)
     det *= A[i*(n+1)];

   for(int i = 0; i < n; ++i){
     printf(" P[%d]: %d \n", i, h_pivot[i]-1); // Subtract 1 to get to C++ 0-base vector values
     if(h_pivot[i]-1 != i)
       det *= -1.;
   }
   return det;
}

int main()
{
#if 1
	const int n = 4;
        // A in column major form.  Det = 295.
	double A[n * n] = { 1.,  0., -1.,  4.,
	  	            2.,  2.,  3.,  1.,
			    6.,  1.,  2.,  0.,
			   -3., -4.,  1., -1. };
#endif
#if 0
	const int n = 3;
        // A in column major form.
	double A[n * n] = { 4.0, 3.0, 8.0,
	  	            9.0, 5.0, 1.0,
			    2.0, 7.0, 6.0 };
#endif
#if 0
	const int n = 2;
        // A in column major form.
	double A[n * n] = { 4.0, 3.0,
			    1.0, 5.0 };
#endif
#if 0
	const int n = 2;
        // A in column major form.
	double A[n * n] = { 3.0, 4.0,
			    5.0, 1.0 };
#endif
#if 0
	const int n = 5;
        // A in column major form.
	double A[n * n] = { 1.0,  6.0, 11.0, 16.0, 21.0,
	  	            2.0,  7.0, 12.0, 17.0, 22.0,
			    3.0,  8.0, 13.0, 18.0, 23.0,
			    4.0,  9.0, 14.0, 19.0, 24.0,
                            5.0, 10.0, 15.0, 20.0, 25.0 };
#endif
    printf("Initial A matrix: \n");
    for(int j=0; j<n; j++)
    {
        for(int i=0; i<n; i++)
            printf("%f\t",(float)A[i*n+j]);
        printf("\n");
    }					
   float det = (float)determinant(A, n);

   printf(" det: %f \n", det);
}
