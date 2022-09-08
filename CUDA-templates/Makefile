
all: cpu
clean:
	-rm demo
cpu:
	g++ -std=c++11 main.cpp -o demoCPU
gpu:
	nvcc -std=c++11 main.cpp -D ENABLE_GPU ArrayPow2_CUDA.cu -o demoGPU
