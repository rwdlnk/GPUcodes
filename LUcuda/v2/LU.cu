
#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
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
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

void cublas_lu(int m, int n, double* a, int *h_pivot)
{
    cublasHandle_t handle;
    double **devPtrA = 0;
    double **devPtrA_dev = NULL;
    int *d_pivot_array;
    int *d_info_array;
    int rowsA = m;
    int colsA = n;
    int matrixSizeA;

    // allocate the pivoting vector and the info array
    cudacall(cudaMalloc(&d_pivot_array, n * sizeof(int)));
    cudacall(cudaMalloc(&d_info_array, sizeof(int)));

    cublascall(cublasCreate(&handle));
    matrixSizeA = rowsA * colsA;

    devPtrA =(double **)malloc(1 * sizeof(*devPtrA));
    if (devPtrA == NULL){ 
     perror("malloc"); 
     exit(EXIT_FAILURE); 
    }
	
    cudacall(cudaMalloc(devPtrA, matrixSizeA * sizeof(double)));
    cudacall(cudaMalloc(&devPtrA_dev, 1 * sizeof(*devPtrA)));

    cudacall(cudaMemcpy(devPtrA_dev, devPtrA, 1 * sizeof(*devPtrA), cudaMemcpyHostToDevice));
    
    cublascall(cublasSetMatrix(rowsA, colsA, sizeof(a[0]), a, rowsA, devPtrA[0], rowsA));

    // Perform LU decomposition
    cublascall(cublasDgetrfBatched(handle, m, devPtrA_dev, m, d_pivot_array, d_info_array, 1));

    cublascall(cublasGetMatrix(m, n, sizeof(double), devPtrA[0], m, a, m));
    cublascall(cublasGetVector(m, sizeof(int), d_pivot_array, 1, h_pivot, 1));

    return;
}


int main()
{
#if 0
	const int n = 4;
        // A in column major form.
	double A[n * n] = { 1.0, 1.0,  3.0, -2.0,
	  	            1.0, 2.0, -1.0,  3.0,
			    2.0, 1.0,  3.0, -1.0,
			    1.0, 2.0, -2.0,  1.0 };
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
#if 1
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
            fprintf(stdout,"%f\t",A[i*n+j]);
        fprintf(stdout,"\n");
    }					

    int *h_pivot = (int *)malloc(n * sizeof(int));

    cublas_lu(n, n, A, h_pivot);

    fprintf(stdout, "Combined LU matrix (pivot applied):\n");
    for(int j=0; j<n; j++)
    {
        for(int i=0; i<n; i++)
            fprintf(stdout,"%f\t",A[i*n+j]);
        fprintf(stdout,"\n");
    }					

   double det = A[0];
   for(int i = 1; i < n; ++i)
     det *= A[i*(n+1)];

   for(int i = 0; i < n; ++i){
     printf(" P[%d]: %d \n", i, h_pivot[i]-1);
     if(h_pivot[i]-1 != i)
       det *= -1.;
   }

   fprintf(stdout," det: %f \n", det);
}
