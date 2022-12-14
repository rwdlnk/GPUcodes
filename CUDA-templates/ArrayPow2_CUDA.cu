 //ArrayPow2_CUDA.cu
 
 #include "Array2D_CUDA.h"
 #include "Array2D.h"
 #include "cuda.h"
 #include "cuda_runtime_api.h"

 #define BLOCK_SIZE 1024
 
template <class T>
__global__ void pow2(T* in, T* out, size_t N){
     int idx = threadIdx.x + blockIdx.x*blockDim.x;
     if (idx < N)
       out[idx] = in[idx] * in[idx];
}
  
template <class T>
void ArrayPow2_CUDA(Array2D<T>& in, Array2D<T>& result) {

  std::cout << "Using the GPU code...\n";

  Array2D< Cutype<T> > in_d(in);
  Array2D< Cutype<T> > result_d(result);

  size_t N = in.size();
  pow2 <<< (N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>> (in_d.begin(), result_d.begin(), in.size());
  cudacall(cudaMemcpy(result.begin(), result_d.begin(), sizeof(T) * N, cudaMemcpyDeviceToHost));
}
 
 template void ArrayPow2_CUDA(Array2D<float>&, Array2D<float>&);
 template __global__ void pow2(float*, float*, size_t);
