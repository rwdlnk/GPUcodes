// main.cpp

#include <iostream>
#include <cstring>
#include "Array2D.h"
#include "ArrayPow2_CPU.h"

#ifdef ENABLE_GPU
#include "Array2D_CUDA.h"
#include "ArrayPow2_CUDA.cuh"

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

#endif //ENABLE_GPU

// Define a function pointer
template <class T>
using ArrayPow2_F = void(*)(Array2D<T>&, Array2D<T>&);

ArrayPow2_F<float> ArrayPow2;

using namespace std;

int main(int argc, char** argv) {
#ifdef ENABLE_GPU
    // Assign funtion pointer to match desired execution; cpu or gpu
    cout << "argc: " << argc << endl;
    // If cpu is requested with gpu build, 
    //   this is equivalent to execution on the host only
    if (argc>1 && !strcmp(argv[1],"cpu")){
       cout << "argv[0]: " << argv[0] << "  argv[1]: " << argv[1] << endl;
       ArrayPow2 = ArrayPow2_CPU;
    } else
    {
        // Use GPUs to process data
        cout << "argv[0]: " << argv[0] << endl;
        ArrayPow2 = ArrayPow2_CUDA;
    }
#else
    ArrayPow2 = ArrayPow2_CPU;
#endif //ENABLE_GPU

    Array2D<float> arr(new float[120], 60, 2);

    // Initialize arr
    int a = 2.;
    for (auto& i:arr)
       i = ++a;

    cout << "Initial arr[0]" << *arr.begin()<< endl;
    cout << "        arr[1]" << *(arr.begin()+1) << endl;

    // create result and copy arr into it.
    Array2D<float> result(arr);
    cout << "Initial result[0]" << *result.begin()<< endl;
    cout << "        result[1]" << *(result.begin()+1) << endl;

    ArrayPow2(arr, result);

    cout << "arr[0]   = " << *arr.begin() << endl;
    cout << "result[0] = arr[0]^2 = " << *result.begin() << endl;
    cout << "arr[1] = " << *(arr.begin()+1) << endl;
    cout << "result[1] = " << *(result.begin()+1) << endl;

    return 0;
}
