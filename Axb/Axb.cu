
#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <assert.h>

#include "cuda_runtime.h" 
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#define prec_save 10

#define BLOCKSIZE 256

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

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

/************************************/
/* SAVE REAL ARRAY FROM CPU TO FILE */
/************************************/
template <class T>
void saveCPUrealtxt(const T * h_in, const char *filename, const int M) {

    std::ofstream outfile;
    outfile.open(filename);
    for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
    outfile.close();

}
/************************************/
/* SAVE REAL ARRAY FROM GPU TO FILE */
/************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const char *filename, const int M) {

    T *h_in = (T *)malloc(M * sizeof(T));

    cudacall(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));

    std::ofstream outfile;
    outfile.open(filename);
    for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
    outfile.close();

}

/***************************************************/
/* FUNCTION TO SET THE VALUES OF THE HANKEL MATRIX */
/***************************************************/
// --- https://en.wikipedia.org/wiki/Hankel_matrix
void setHankelMatrix(double * __restrict h_A, const int N) {

    double *h_atemp = (double *)malloc((2 * N - 1) * sizeof(double));

    // --- Initialize random seed
    srand(time(NULL));

    // --- Generate random numbers
    for (int k = 0; k < 2 * N - 1; k++) h_atemp[k] = rand();

    // --- Fill the Hankel matrix. The Hankel matrix is symmetric, so filling by row or column is equivalent.
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_A[i * N + j] = h_atemp[(i + 1) + (j + 1) - 2];

    free(h_atemp);

}

/***********************************************/
/* FUNCTION TO COMPUTE THE COEFFICIENTS VECTOR */
/***********************************************/
void computeCoefficientsVector(const double * __restrict h_A, const double * __restrict h_xref, 
                               double * __restrict h_y, const int N) {

    for (int k = 0; k < N; k++) h_y[k] = 0.f;

    for (int m = 0; m < N; m++)
        for (int n = 0; n < N; n++)
            h_y[m] = h_y[m] + h_A[n * N + m] * h_xref[n];

}


/************************************/
/* COEFFICIENT REARRANGING FUNCTION */
/************************************/
void rearrange(double *vec, int *pivotArray, int N){
    for (int i = 0; i < N; i++) {
        double temp = vec[i];
        vec[i] = vec[pivotArray[i] - 1];
        vec[pivotArray[i] - 1] = temp;
    }   
}

/********/
/* MAIN */
/********/
int main() {

    const unsigned int N = 1500;

    const unsigned int Nmatrices = 1;

    // --- CUBLAS initialization
    cublasHandle_t cublas_handle;
    cublascall(cublasCreate(&cublas_handle));

    cudaEvent_t startLU, startApr1, startApr2;
    cudaEvent_t stopLU, stopApr1, stopApr2;
    cudaEventCreate(&startLU);
    cudaEventCreate(&startApr1);
    cudaEventCreate(&startApr2);
    cudaEventCreate(&stopLU);
    cudaEventCreate(&stopApr1);
    cudaEventCreate(&stopApr2);

    float timingLU=0;
    float timingApr1=0;
    float timingApr2=0;

    /***********************/
    /* SETTING THE PROBLEM */
    /***********************/
    // --- Matrices to be inverted (only one in this example)
    double *h_A = (double *)malloc(N * N * Nmatrices * sizeof(double));

    // --- Setting the Hankel matrix
    setHankelMatrix(h_A, N);

    // --- Defining the solution
    double *h_xref = (double *)malloc(N * sizeof(double));
    for (int k = 0; k < N; k++) h_xref[k] = 1.f;

    // --- Coefficient vectors (only one in this example)
    double *h_y = (double *)malloc(N * sizeof(double));

    computeCoefficientsVector(h_A, h_xref, h_y, N);

    // --- Result (only one in this example)
    double *h_x = (double *)malloc(N * sizeof(double));

    // --- Allocate device space for the input matrices 
    double *d_A; cudacall(cudaMalloc(&d_A, N * N * Nmatrices * sizeof(double)));
    double *d_y; cudacall(cudaMalloc(&d_y, N *                 sizeof(double)));
    double *d_x; cudacall(cudaMalloc(&d_x, N *                 sizeof(double)));

    // --- Move the relevant matrices from host to device
    cudacall(cudaMemcpy(d_A, h_A, N * N * Nmatrices * sizeof(double), cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_y, h_y, N *                 sizeof(double), cudaMemcpyHostToDevice));

    /**********************************/
    /* COMPUTING THE LU DECOMPOSITION */
    /**********************************/
    cudaEventRecord(startLU, 0);

    // --- Creating the array of pointers needed as input/output to the batched getrf
    double **h_inout_pointers = (double **)malloc(Nmatrices * sizeof(double *));
    for (int i = 0; i < Nmatrices; i++) h_inout_pointers[i] = d_A + i * N * N;

    double **d_inout_pointers;
    cudacall(cudaMalloc(&d_inout_pointers, Nmatrices * sizeof(double *)));
    cudacall(cudaMemcpy(d_inout_pointers, h_inout_pointers, Nmatrices * sizeof(double *), cudaMemcpyHostToDevice));
    free(h_inout_pointers);

    int *d_pivotArray; cudacall(cudaMalloc(&d_pivotArray, N * Nmatrices * sizeof(int)));
    int *d_InfoArray;  cudacall(cudaMalloc(&d_InfoArray,      Nmatrices * sizeof(int)));

    int *h_InfoArray  = (int *)malloc(Nmatrices * sizeof(int));

    cublascall(cublasDgetrfBatched(cublas_handle, N, d_inout_pointers, N, d_pivotArray, d_InfoArray, Nmatrices));
    //cublascall(cublasDgetrfBatched(cublas_handle, N, d_inout_pointers, N, NULL, d_InfoArray, Nmatrices));

    cudacall(cudaMemcpy(h_InfoArray, d_InfoArray, Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nmatrices; i++)
        if (h_InfoArray[i] != 0) {
            fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }

    cudaEventRecord(stopLU, 0);
    cudaEventSynchronize(stopLU);

    cudaEventElapsedTime(&timingLU, startLU, stopLU);

    cudaEventDestroy(startLU);
    cudaEventDestroy(stopLU);

    printf("The elapsed time for LU decomp in gpu was %.2f ms\n", timingLU);

    /*********************************/
    /* CHECKING THE LU DECOMPOSITION */
    /*********************************/
    saveCPUrealtxt(h_A,          "output/A.txt", N * N);
    saveCPUrealtxt(h_y,          "output/y.txt", N);
    saveGPUrealtxt(d_A,          "output/Adecomposed.txt", N * N);
    saveGPUrealtxt(d_pivotArray, "output/pivotArray.txt", N);

    /******************************************************************************/
    /* APPROACH NR.1: COMPUTE THE INVERSE OF A STARTING FROM ITS LU DECOMPOSITION */
    /******************************************************************************/
    cudaEventRecord(startApr1, 0);

    // --- Allocate device space for the inverted matrices 
    double *d_Ainv; cudacall(cudaMalloc(&d_Ainv, N * N * Nmatrices * sizeof(double)));

    // --- Creating the array of pointers needed as output to the batched getri
    double **h_out_pointers = (double **)malloc(Nmatrices * sizeof(double *));
    for (int i = 0; i < Nmatrices; i++) h_out_pointers[i] = (double *)((char*)d_Ainv + i * ((size_t)N * N) * sizeof(double));

    double **d_out_pointers;
    cudacall(cudaMalloc(&d_out_pointers, Nmatrices*sizeof(double *)));
    cudacall(cudaMemcpy(d_out_pointers, h_out_pointers, Nmatrices*sizeof(double *), cudaMemcpyHostToDevice));
    free(h_out_pointers);

    cublascall(cublasDgetriBatched(cublas_handle, N, (const double **)d_inout_pointers, N, d_pivotArray, d_out_pointers, N, d_InfoArray, Nmatrices));

    cudacall(cudaMemcpy(h_InfoArray, d_InfoArray, Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nmatrices; i++)
        if (h_InfoArray[i] != 0) {
        fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
        }

    double alpha1 = 1.f;
    double beta1 = 0.f;

    cublascall(cublasDgemv(cublas_handle, CUBLAS_OP_N, N, N, &alpha1, d_Ainv, N, d_y, 1, &beta1, d_x, 1));

    cudaEventRecord(stopApr1, 0);
    cudaEventSynchronize(stopApr1);

    cudaEventElapsedTime(&timingApr1, startApr1, stopApr1);

    cudaEventDestroy(startApr1);
    cudaEventDestroy(stopApr1);

    printf("The elapsed time for Approach 1 in gpu was %.2f ms\n", timingApr1);

    /**************************/
    /* CHECKING APPROACH NR.1 */
    /**************************/
    saveGPUrealtxt(d_x, "output/xApproach1.txt", N);

    /*************************************************************/
    /* APPROACH NR.2: INVERT UPPER AND LOWER TRIANGULAR MATRICES */
    /*************************************************************/
    cudaEventRecord(startApr2, 0);

    double *d_P; cudacall(cudaMalloc(&d_P, N * N * sizeof(double)));

    cudacall(cudaMemcpy(h_y, d_y, N * Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));
    int *h_pivotArray = (int *)malloc(N * Nmatrices*sizeof(int));
    cudacall(cudaMemcpy(h_pivotArray, d_pivotArray, N * Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));

    rearrange(h_y, h_pivotArray, N);
    cudacall(cudaMemcpy(d_y, h_y, N * Nmatrices * sizeof(double), cudaMemcpyHostToDevice));

    // --- Now P*A=L*U
    //     Linear system A*x=y => P.'*L*U*x=y => L*U*x=P*y

    // --- 1st phase - solve Ly = b 
    const double alpha = 1.f;

    // --- Function solves the triangular linear system with multiple right hand sides, function overrides b as a result 

    // --- Lower triangular part
    cublascall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, 1, &alpha, d_A, N, d_y, N));

    // --- Upper triangular part
    cublascall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, 1, &alpha, d_A, N, d_y, N));

    cudaEventRecord(stopApr2, 0);
    cudaEventSynchronize(stopApr2);

    cudaEventElapsedTime(&timingApr2, startApr2, stopApr2);

    cudaEventDestroy(startApr2);
    cudaEventDestroy(stopApr2);

    printf("The elapsed time for Approach 2 in gpu was %.2f ms\n", timingLU + timingApr2);
    
    /**************************/
    /* CHECKING APPROACH NR.2 */
    /**************************/
    saveGPUrealtxt(d_y, "output/xApproach2.txt", N);

    return 0;
}
