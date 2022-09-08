
#include <iostream>
#include <algorithm>

using namespace std;

#define TILE_WIDTH 4

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {

   __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
   __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

   //printf("Hello from block-x: %u, thread: %u\n", blockIdx.x, threadIdx.x);
   //printf("Hello from block-y: %u, thread: %u\n", blockIdx.y, threadIdx.y);

   int bx = blockIdx.x;
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   int Row = by * TILE_WIDTH + ty;
   int Col = bx * TILE_WIDTH + tx;
 
   //printf("Row: %u, Col: %u\n", Row, Col);

   float Pvalue = 0;
	
   for (int m = 0; m < Width/TILE_WIDTH;m++) {
      Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
      Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
      __syncthreads(); //Wait till all the threads have finished loading elements into the tiles.
		
      for (int k = 0; k < TILE_WIDTH;k++) {
         Pvalue += Mds[ty][k] * Nds[k][tx];
         __syncthreads();
         //printf("Pvalue: %f\n", Pvalue);
      }
		
      d_P[Row*Width + Col] = Pvalue;
   }
}	

int main(void) {

  const int WIDTH = 1024;
  const int N = WIDTH*WIDTH;

  cout << endl << "Square Matrix Shared Memory Multiply on GPU" << endl; 
  cout << "Square matrix: " << WIDTH << " x " << WIDTH << endl;
  cout << "TILE_Width: " << TILE_WIDTH << endl;
  cout << "Grid: " << WIDTH/TILE_WIDTH << " x " << WIDTH/TILE_WIDTH << endl;
  cout << "Block: " << TILE_WIDTH  << " x " << TILE_WIDTH << endl << endl;

  float *in, *mid, *out; // host copies of a, b, c
  float *d_in, *d_mid, *d_out; // device copies of a, b, c
  int size = N * sizeof(float);

  // Alloc space for host copies and setup values
  in = (float *)malloc(size); 
  fill(in+0, in+N, 4.0);

  mid = (float *)malloc(size); 
  fill(mid+0, mid+N, 2.0);

  out = (float *)malloc(size); 
  fill(out+0, out+N,0.);

#if 0
  for(int i = 0; i < N; ++i){
    cout << "filled in[" << i << "]: " << *(in+i) << ", mid: " << *(mid+i) << ", out: " << *(out+i) << endl;
  }
#endif
  // Alloc space for device copies
  cudaMalloc((void **)&d_in, size);
  cudaMalloc((void **)&d_mid, size);
  cudaMalloc((void **)&d_out, size);

  // Copy to device
  cudaMemcpy(d_in,   in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mid, mid, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

  // define grid and blocks
  dim3 dimGrid(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
 
  // Multiply
  MatrixMulKernel<<<dimGrid, dimBlock >>>(d_in, d_mid, d_out, WIDTH);

  // Copy result back to host
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  for(int i = 0; i < 4; ++i){
    cout << "Pre Mult in[" << i << "]: " << in[i] << endl;
  }
  for(int i = 0; i < 4; ++i){
    cout << "Pre Mult: mid[" << i << "]: " << mid[i] << endl;
  }
  for(int i = 0; i < 4; ++i){
    cout << "After Mult out[" << i << "]: " << out[i] << endl;
  }  
  
  // Cleanup
  free(in); free(mid), free(out);
  cudaFree(d_in); cudaFree(d_mid), cudaFree(d_out);

  return 0;
}
