
/*
 Thanks to
 https://stackoverflow.com/questions/22814040/lu-decomposition-in-cuda
*/

#include <stdio.h>

#include "cuda_runtime.h" 
#include "device_launch_parameters.h"

#include "cublas_v2.h"

/**********************/ 
/* HANDLE CUDA ERRORS */  
/**********************/
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
    
/************************/
/* HANDLE cublas ERRORS */
/************************/

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

void rearrange(double *vec, int *pivotArray, int N){
    for (int i = 0; i < N; i++) {
        double temp = vec[i];
        vec[i] = vec[pivotArray[i] - 1];
        vec[pivotArray[i] - 1] = temp;
    }
}

int main() {

    const unsigned int N = 4; 

    const unsigned int Nmatrices = 1;

    cublasHandle_t handle;
    cublascall(cublasCreate(&handle));

    // --- Matrices to be inverted (only one in this example)
    float *h_A = new float[N*N*Nmatrices];

    h_A[0] = 4.f;  
    h_A[1] = 3.f;
    h_A[2] = 8.f;
    h_A[3] = 9.f;
    h_A[4] = 5.f; 
    h_A[5] = 1.f; 
    h_A[6] = 2.f; 
    h_A[7] = 7.f;
    h_A[8] = 6.f;
    h_A[9] = -4.f;  
    h_A[10] = 3.f;
    h_A[11] = 2.f;
    h_A[12] = 7.f;
    h_A[13] = -5.f; 
    h_A[14] = 2.f; 
    h_A[15] = 4.f; 

    // --- Allocate device matrices 
    float *d_A;
    cudacall(cudaMalloc((void**)&d_A, N*N*Nmatrices*sizeof(float)));

    // --- Move the matrix to be inverted from host to device
    cudacall(cudaMemcpy(d_A, h_A, N*N*Nmatrices*sizeof(float), cudaMemcpyHostToDevice));

    // --- Creating the array of pointers needed as input to the batched getrf
    float **h_inout_pointers = (float **)malloc(Nmatrices*sizeof(float *));
    for (int i=0; i<Nmatrices; i++)
      h_inout_pointers[i]=(float *)(d_A+i*N*N);

    float **d_inout_pointers;
    cudacall(cudaMalloc((void**)&d_inout_pointers, Nmatrices*sizeof(float *)));
    cudacall(cudaMemcpy(d_inout_pointers, h_inout_pointers, Nmatrices*sizeof(float *), cudaMemcpyHostToDevice));
    free(h_inout_pointers);

    int *d_PivotArray; 
    cudacall(cudaMalloc((void**)&d_PivotArray, N*Nmatrices*sizeof(int)));

    int *d_InfoArray;  
    cudacall(cudaMalloc((void**)&d_InfoArray,  Nmatrices*sizeof(int)));

    int *h_PivotArray = (int *)malloc(N*Nmatrices*sizeof(int));
    int *h_InfoArray  = (int *)malloc(  Nmatrices*sizeof(int));

    //cublascall(cublasSgetrfBatched(handle, N, d_inout_pointers, N, d_PivotArray, d_InfoArray, Nmatrices));
    cublascall(cublasSgetrfBatched(handle, N, d_inout_pointers, N, NULL, d_InfoArray, Nmatrices));

    cudacall(cudaMemcpy(h_InfoArray,d_InfoArray,Nmatrices*sizeof(int),cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nmatrices; i++)
        if (h_InfoArray[i]  != 0) {
            fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }

    cudacall(cudaMemcpy(h_A,d_A,N*N*sizeof(float),cudaMemcpyDeviceToHost));
    //cudacall(cudaMemcpy(h_PivotArray,d_PivotArray,N*Nmatrices*sizeof(int),cudaMemcpyDeviceToHost));

    for (int i=0; i<N*N; i++)
      printf("A[%i]=%f\n", i, h_A[i]);

    printf("\n\n");
    //for (int i=0; i<N; i++)
    //  printf("P[%d]=%d\n", i, h_PivotArray[i]);

    // Compute the determinant of h_A 
    float det = h_A[0];
    for(int i = 1; i < N; ++i)
      det *= h_A[i*(N+1)];

     printf("det: %f ", det);

    return 0;
}

