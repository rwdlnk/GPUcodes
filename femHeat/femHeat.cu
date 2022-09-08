
#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <assert.h>
#include <math.h>

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

/************************************/
/* SAVE REAL ARRAY FROM GPU TO FILE */
/************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const T * h_x, const char *filename, const int M) {

    T *h_in = (T *)malloc(M * sizeof(T));

    cudacall(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));

    std::ofstream outfile;
    outfile.open(filename);
    outfile << std::setprecision(prec_save) << h_x[0] << " " << 0. << "\n";
    for (int i = 1; i < M-1; i++) outfile << std::setprecision(prec_save) << h_x[i] << " " << h_in[i] << "\n";
    outfile << std::setprecision(prec_save) << h_x[M-1] << " " << 0. << "\n";
    outfile.close();

}

/************************************/
/* SAVE REAL ARRAY FROM GPU TO FILE */
/************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const T * h_exact, const T * h_coord, const char *filename, const int M) {

    T *h_in = (T *)malloc(M * sizeof(T));

    cudacall(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));

    std::ofstream outfile;
    outfile.open(filename);
    outfile << "x " << "Tfem " << "Texact" << "\n";
    outfile << std::setprecision(prec_save) << h_coord[0] << " " << 0.  << " " << h_exact[0] << "\n";
    for (int i = 1; i < M-1; i++) outfile << std::setprecision(prec_save) << h_coord[i] << " " << h_in[i] << " " << h_exact[i] << "\n";
    outfile << std::setprecision(prec_save) << h_coord[M-1] << " " << 0. << " " << h_exact[M-1] << "\n";
    outfile.close();

}

/*********************************/
/*********************************/
/* EXACT SOLUTION                */
/*********************************/
void exactSolution(double * __restrict h_exact, const double * h_x, const int E, const double K, const double M, const double S0, const double L) {

  double alpha = M*L/K;
  double beta = S0*L*L*L*L/K;
  double ba4 = beta/(alpha*alpha*alpha*alpha);
  double ax;

  for(int i = 0; i < E+1; ++i) {
    ax = alpha*h_x[i]/L;
    h_exact[i] = ba4*(ax*(2. + ax + ax*ax/3.) - alpha*(2. + alpha + alpha*alpha/3.)*(1.-exp(ax))/(1.-exp(alpha)));
  }
}

/**********************************************/
/* FUNCTION TO SET THE VALUES OF THE A MATRIX */
/**********************************************/
void setAMatrix(double * __restrict h_A, const int E, const double K, const double M, const double Le) {

    const int N = E + 1;

    double Ae[4] = { K/Le - M*0.5, -K/Le - M*0.5,
                    -K/Le + M*0.5,  K/Le + M*0.5};

    // --- Initialize the h_A matrix
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_A[i * N + j] = 0.;

    // --- Assemble the h_A matrix
    int g_i, g_j;
    for(int e = 0; e < E; ++e){
      for(int i = 0; i < 2; ++i){
        g_i = i + e;
        for(int j = 0; j < 2; ++j){
          g_j = j + e;
          h_A[g_i * N + g_j] += Ae[i * 2 + j];
        }
     }
   }

  // Impose fixed BCs for first and last nodes
  for(int j = 1; j < N; ++j)
     h_A[j] = 0.;
  h_A[0] = 1.;
  for(int j = 0; j < N-1; ++j)
     h_A[(N-1)*N + j] = 0.;
  h_A[N*N-1] = 1.;
}

/************************************/
/* FUNCTION TO COMPUTE THE B VECTOR */
/************************************/
void computeBVector(double * __restrict h_y, double* h_x, const int N, const double S0, const double Le) {

    // Compute Nodal Coordinates

    for (int k = 0; k < N; k++) h_y[k] = 0.f;

    double L3 = S0*Le*Le*Le/12.;
    double aL;
    int g_i = 0;
    for(int e = 0; e < N-1; ++e){
       aL = h_x[e]/Le;
       double Be[2] = {L3*(6.*aL*aL + 4.*aL + 1.),
                       L3*(6.*aL*aL + 8.*aL + 3.)};
       for(int i = 0; i < 2; ++i){
         g_i = e + i;
         h_y[g_i] += Be[i];
       }
    }

  // Impose fixed BCs for first and last nodes
     h_y[0] = 0.;
     h_y[N-1] = 0.;
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

/********************************************************************/
/*                                                                  */
/* Solve:      _      _                                             */
/*          d | d T(x) |     d T(x)       2                         */
/*        K --| ------ | -Mx ------ + S0 x = 0, 0 < x < L           */
/*          dx|   dx   |       dx                                   */
/*            --      --                                            */
/* with T(0) = T(L) = 0; K, Mx, S0 are constants                    */
/* using the finite element method with E 1-dimensional simplex     */
/* elements and N = E + 1 nodes in the global problem.              */
/* For each element, x_i <= x <= x_(i+1) , i = 0,...,E,             */
/* L = x_(i+1) - x_i:                                               */
/*                                                                  */
/*           | 1  -1|            |-1 -1 |                           */
/* [K] = K/L |      |, [M] = M/2 |      |,                          */
/*           |-1   1|            | 1  1 |                           */
/*                                                                  */
/*                     2                                            */
/*           3    | 6(x_i/L) + 4 (x_i/L) + 1 |                      */
/* {S} = S0 L /12 |        2                 |                      */
/*                | 6(x_i/L) + 8 (x_i/L) + 3 |                      */
/*                                                                  */
/* so that the element approximation to T(x) becomes                */
/*   (e)                                                            */
/*  T(x) = (x_(1+1) - x)/L * T_i + (x - x_1)/L T_(i+1)              */
/*                                                                  */
/* and the element equations are                                    */
/*                     | T_i     |                                  */
/*         ([K] + [M]) |         | = {S}                            */
/*                     | T_(i+1) |                                  */
/*                                                                  */
/* The exact solution is:                                           */
/*   4                   2    3                                     */
/* (a /b) T(y) = -(2a + a  + a /3)(1-exp(a*y))/(1-exp(a))           */
/*                              2      3                            */
/*                  + 2ay + (ay) + (ay)/3                           */
/*                                      4                           */
/* where y = x/L, a = M L / K, b = S0 L / K                         */
/*                                                                  */
/********************************************************************/

    // Define problem constants
    const unsigned int E = 2000;    // Number of equal length elements
    const unsigned int N = E+1;   // Number of nodes: assuming 1D simplex elements
    const double K = 1.;          // Thermal conductivity of material
    const double M = 10.;         // X forced convection: M = rho*C_p*U_x
    const double S0 = 4.;         // Thermal source strength
    const double L = 1.;          // Length of domain

    const unsigned int Nmatrices = 1;

    // --- CUBLAS initialization
    cublasHandle_t cublas_handle;
    cublascall(cublasCreate(&cublas_handle));

    cudaEvent_t startLU, startSoln;
    cudaEvent_t stopLU, stopSoln;
    cudaEventCreate(&startLU);
    cudaEventCreate(&startSoln);
    cudaEventCreate(&stopLU);
    cudaEventCreate(&stopSoln);

    float timingLU=0;
    float timingSoln=0;

    /***********************/
    /* SETTING THE PROBLEM */
    /***********************/
    // --- Matrices to be inverted (only one in this example)
    double *h_A = (double *)malloc(N * N * Nmatrices * sizeof(double));

    // --- Setting the Element A matrix
    double Le = L/double(E);

    setAMatrix(h_A, E, K, M, Le);

    // --- Coefficient vectors (only one in this example)
    double *h_y = (double *)malloc(N * sizeof(double));

    double *h_xcoord = (double *)malloc(N * sizeof(double));
    h_xcoord[0] = 0.;
    for(int i = 1; i < N; ++i)
       h_xcoord[i] = h_xcoord[i-1] + Le;

    double *h_exact = (double *)malloc(N * sizeof(double));

    // Compute exact solution
    exactSolution(h_exact, h_xcoord, E, K, M, S0, L);

    computeBVector(h_y, h_xcoord, N, S0, Le);

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
    // For non-pivot solution, comment above line and uncomment next line
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

    /**********************************************/
    /* INVERT UPPER AND LOWER TRIANGULAR MATRICES */
    /**********************************************/
    cudaEventRecord(startSoln, 0);

    double *d_P; cudacall(cudaMalloc(&d_P, N * N * sizeof(double)));

    cudacall(cudaMemcpy(h_y, d_y, N * Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));
    int *h_pivotArray = (int *)malloc(N * Nmatrices*sizeof(int));
    cudacall(cudaMemcpy(h_pivotArray, d_pivotArray, N * Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));

    rearrange(h_y, h_pivotArray, N);
    cudacall(cudaMemcpy(d_y, h_y, N * Nmatrices * sizeof(double), cudaMemcpyHostToDevice));

    // --- Now P*A=L*U
    //     Linear system A*x=y => P*A*x=P*y => L*U*x = P*y
    // Let z = U*x so that L*z = P*y solves for z.  Then, solve U*x = z for x.

    // --- 1st phase - solve Ly = b 
    const double alpha = 1.f;

    // --- Function solves the triangular linear system with multiple right hand sides, function overrides b as a result 

    // --- Lower triangular part
    cublascall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, 1, &alpha, d_A, N, d_y, N));

    // --- Upper triangular part
    cublascall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, 1, &alpha, d_A, N, d_y, N));

    cudaEventRecord(stopSoln, 0);
    cudaEventSynchronize(stopSoln);

    cudaEventElapsedTime(&timingSoln, startSoln, stopSoln);
    printf("The elapsed time for matrix inversion and calculation of solution on GPU was %.2f ms\n", timingLU + timingSoln);

    //cudaEventElapsedTime(&timingLU, startLU, stopSoln);
    //printf("The elapsed time for solve in GPU was %.2f ms\n", timingLU );

    cudaEventDestroy(startSoln);
    cudaEventDestroy(stopSoln);

    
    /**************************/
    /* CHECKING APPROACH NR.2 */
    /**************************/
    saveGPUrealtxt(d_y, h_exact, h_xcoord, "output/solution.txt", N);

    free(h_xcoord);

    return 0;
}
