
#include <iostream>
#include <fstream>
#include <algorithm>
#include <assert.h>

using namespace std;

//#define TILE_WIDTH 1
//#define TILE_WIDTH 2
//#define TILE_WIDTH 3
#define TILE_WIDTH 4
//#define TILE_WIDTH 16

__global__ void MatrixMulKernel(const float* d_A, const float* d_B, float* d_C, const int rowsA, const int columnsA, const int rowsB, const int columnsB) {
//
// Assistance from
// https://github.com/debowin/cuda-tiled-matrix-multiplication
// is greatly appreciated.
//
// Calculating A[i][k] * B[k][j] = C[k][k]
//
   // Statically sized arrays of GPU shared memory
   __shared__ float tileAs[TILE_WIDTH][TILE_WIDTH];
   __shared__ float tileBs[TILE_WIDTH][TILE_WIDTH];

   //printf("Hello from block-x: %u, thread: %u\n", blockIdx.x, threadIdx.x);
   //printf("Hello from block-y: %u, thread: %u\n", blockIdx.y, threadIdx.y);

   int bx = blockIdx.x;
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   //printf("bx: %u, by: %u\n", bx, by);
   //printf("tx: %u, ty: %u\n", tx, ty);

   // Compute GLOBAL row/column indices
   int Row = by * TILE_WIDTH + ty;
   int Col = bx * TILE_WIDTH + tx;
 
   float Cvalue = 0;

   // Sweep tiles over entire matrix:
   int spanA = ceilf(columnsA/(float)TILE_WIDTH);

   for(int i = 0; i < spanA; i++){
     // move the tiles and update shared memory value for new tile positions
     if(Row < rowsA && (i*TILE_WIDTH + tx) < columnsA)
	tileAs[ty][tx] = d_A[Row*columnsA + i*TILE_WIDTH + tx];
     else
	tileAs[ty][tx] = 0;

     if(Col < columnsB && (i*TILE_WIDTH + ty) < rowsB)
	tileBs[ty][tx] = d_B[(i*TILE_WIDTH + ty)*columnsB + Col];
     else
	tileBs[ty][tx] = 0;

     // after the entire tile's values are available, proceed
     // Do not want to overwrite the shared memory
     __syncthreads();

     // Compute this tile's value of C
     for(int k = 0; k < TILE_WIDTH; k++)
	Cvalue += tileAs[ty][k] * tileBs[k][tx];

     // after the entire tile's values have been used, proceed
     __syncthreads();
  }

  // boundary check and set global C value
  if(Row < rowsA && Col < columnsB)
     d_C[Row*columnsB+Col] = Cvalue;
}	

void computeGold(float* C, const float* A, const float* B, const int hA, const int wA, const int wB)
{
  for(int i = 0; i < hA; ++i){
    for(int j = 0; j < wB; ++j){
      float sum = 0.;
      for(int k = 0; k < wA; ++k){
        float a = A[i*wA + k];
        float b = B[k*wB + j];
        sum += a*b;
      }
      C[i*wB + j] = sum;
    }
  }
} 

// returns true iff A and B have same elements in same order
bool CompareMatrices(float* A, float* B) {
    int size = sizeof(A);

    for (int i = 0; i < size; i++)
        if (abs(A[i] - B[i]) > 0.0001f){
            cout << "Fails at i = " << i << "   A: " << A[i] << " B: " << B[i]  << "   A-B: " << A[i]-B[i] << endl;
            return false;
        }
    return true;
}

void writeMatrixFile(const float* A, const int rA, const int cA, char* name){ 

  ofstream outfile;
  outfile.open(name, ios::out );

  for(int i = 0; i < rA; ++i){
    for(int j = 0; j < cA; ++j){
      outfile << A[i*cA+j] << " ";
    }
  }
  outfile.close();
}

int main(void) {

//  Compute A[i][k]*B[k][j] = C[k][k]


// Options for different size and shaped matrices
#if 0
  const int ROW_A = 3;
  const int COL_A = 3;
  const int ROW_B = 3;
  const int COL_B = 3;
#endif
#if 0
  const int ROW_A = 9;
  const int COL_A = 9;
  const int ROW_B = 9;
  const int COL_B = 9;
#endif
#if 0
  const int ROW_A = 9;
  const int COL_A = 2;
  const int ROW_B = 2;
  const int COL_B = 4;
#endif
#if 0
  const int ROW_A = 4;
  const int COL_A = 3;
  const int ROW_B = 3;
  const int COL_B = 9;
#endif
#if 0
  const int ROW_A = 8;
  const int COL_A = 8;
  const int ROW_B = 8;
  const int COL_B = 6;
#endif
#if 0
  const int ROW_A = 11;
  const int COL_A = 25;
  const int ROW_B = 25;
  const int COL_B = 51;
#endif
#if 1
  const int ROW_A = 51;
  const int COL_A = 25;
  const int ROW_B = 25;
  const int COL_B = 11;
#endif
#if 0
  const int ROW_A = 512;
  const int COL_A = 250;
  const int ROW_B = 250;
  const int COL_B = 901;
#endif

  assert(COL_A == ROW_B && "Number of columns in A must equal number of rows in B." );

  const int ROW_C = ROW_A;
  const int COL_C = COL_B;

  cout << endl << "General Matrix Shared Memory Multiply on GPU: A[i][k] B[k][j] = C[k][k]" << endl; 
  cout << "A matrix: " << ROW_A << " x " << COL_A << endl;
  cout << "B matrix: " << ROW_B << " x " << COL_B << endl;
  cout << "C matrix: " << ROW_C << " x " << COL_C << endl <<endl;

  const int N_A = ROW_A*COL_A;
  const int N_B = ROW_B*COL_B;
  const int N_C = ROW_C*COL_C;

  int size_A = N_A * sizeof(float);
  int size_B = N_B * sizeof(float);
  int size_C = N_C * sizeof(float);

  // Define host and device matrices
  float *A, *B, *C, *gC; // host copies of A, B, C, gold-C
  float *d_A, *d_B, *d_C; // device copies of A, B, C

  // Alloc space for host copies and setup values
  A = (float *)malloc(size_A); 
  for(int i = 0; i < N_A; ++i)
    A[i] = (float)i + 1.0f;

  char Aname[] = "Amatrix.dat";
  writeMatrixFile(A, ROW_A, COL_A, Aname);

  B = (float *)malloc(size_B); 
  for(int i = 0; i < N_B; ++i)
    B[i] = (float)(i+N_A) + 1.0f;

  char Bname[] = "Bmatrix.dat";
  writeMatrixFile(B, ROW_B, COL_B, Bname);

  C = (float *)malloc(size_C); 
  fill(C + 0, C + N_C, 0.);

  gC = (float *)malloc(size_C); 
  fill(gC + 0, gC + N_C, 0.);

#if 0
  for(int i = 0; i < N_A; ++i){
    cout << "filled A[" << i << "]: " << *(A+i) << endl;
  }
  for(int i = 0; i < N_B; ++i){
    cout << "filled B[" << i << "]: " << *(B+i) << endl;
  }
  for(int i = 0; i < N_C; ++i){
    cout << "filled C[" << i << "]: " << *(C+i) << endl;
  }
#endif

  // Alloc space for device copies
  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_B, size_B);
  cudaMalloc((void **)&d_C, size_C);

  // Copy to device
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice);

  // Setup the execution configuration
  dim3 gridSize, blockSize;

  blockSize.x = blockSize.y = TILE_WIDTH;
  blockSize.z = 1;
    
  gridSize.x = ceil(COL_C/(float)blockSize.x);
  gridSize.y = ceil(ROW_C/(float)blockSize.y);
  gridSize.z = 1;

  // Launch the device computation threads!
  // Multiply A and B returning C
  MatrixMulKernel<<<gridSize, blockSize >>>(d_A, d_B, d_C, ROW_A, COL_A, ROW_B, COL_B);

  // Copy result back to host
  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

#if 0
  for(int i = 0; i < N_A; ++i){
    cout << "Pre Mult A[" << i << "]: " << A[i] << endl;
  }
  for(int i = 0; i < N_B; ++i){
    cout << "Pre Mult: B[" << i << "]: " << B[i] << endl;
  }
  for(int i = 0; i < N_C; ++i){
    cout << "Post Mult: C[" << i << "]: " << C[i] << endl;
  }

#endif

  computeGold(gC, A, B, ROW_A, COL_A, COL_B);

  //for(int i = 0; i < N_C; ++i){
  //// cout << " gC[" << i << "]: " << gC[i] << "    C[" << i << "]: " << C[i] << endl;
  //}  

  // check if the device result is equivalent to the expected solution
  bool res = CompareMatrices(gC, C);
  printf("Test %s\n", res ? "PASSED" : "FAILED");
 
  // Cleanup
  free(A); free(B), free(C), free(gC);
  cudaFree(d_A); cudaFree(d_B), cudaFree(d_C);

  return 0;
}
