
#include <stdio.h>
#include <fstream>
#include <iomanip>
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

/***********************************/
/* SAVE REAL ARRAY FROM GPU TO FILE */
/************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const char *filename, const int M) {

    T *h_in = (T *)malloc(M * sizeof(T));

    cudacall(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));

    std::ofstream outfile;
    outfile.open(filename);
    for (int i = 0; i < M; i++) outfile << std::setprecision(10) << h_in[i] << "\n";
    outfile.close();

}

int* cublas_lu(int m, int n, double* a)
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

   saveGPUrealtxt(d_pivot_array, "pivot.txt", m);

   return d_pivot_array;
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
#if 1
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
    printf("Initial A matrix: \n");
    for(int j=0; j<n; j++)
    {
        for(int i=0; i<n; i++)
            fprintf(stdout,"%f\t",A[i*n+j]);
        fprintf(stdout,"\n");
    }					

    int * d_pivot;
    cudacall(cudaMalloc(&d_pivot, n * sizeof(int)));
    int *h_pivot = (int *)malloc(n * sizeof(int));

    d_pivot = cublas_lu(n, n, A);

    cudacall(cudaMemcpy(h_pivot, d_pivot, n*sizeof(int), cudaMemcpyDeviceToHost));

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
