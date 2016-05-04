
#include "even_odd_sort.h"
 
 
 
 
__global__ void even_sort(int* ary, int size) {
	int tid = (blockIdx.z * gridDim.x * gridDim.y
                + blockIdx.y * gridDim.x
                + blockIdx.x) * blockDim.x + threadIdx.x;
    int index = tid * 2 + 1;
    if (index >= size) return;
	 
	if(ary[index  - 1] > ary[index] ) {
			int tp = ary[index - 1];
			ary[index - 1] = ary[index];
			ary[index] = tp;
 
	}
 
	
 
}
__global__ void odd_sort(int* ary, int size) {
	int tid = (blockIdx.z * gridDim.x * gridDim.y
                + blockIdx.y * gridDim.x
                + blockIdx.x) * blockDim.x + threadIdx.x;
    int index = tid * 2 + 1;
    if (index + 1 >= size) return;
	 
	if(ary[index] > ary[index + 1] ) {
			int tp = ary[index + 1];
			ary[index + 1] = ary[index];
			ary[index] = tp;
 
	}
    
}
 
void even_odd_sort(Data *data)
{
	const int SIZE = data->length;
  
    int threads;
    Grid grid;
    int size = 2;
	cal_grid(&grid, &threads, data->length, size);	 
    dim3 blocks(grid.blockx, grid.blocky, grid.blockz);
    int* gary;
	cudaMalloc((void**) &gary, SIZE * sizeof(int));
	cudaMemcpy(gary, data->intarray, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	for(int i = 0; i < SIZE; ++ i) {
	    if (i % 2 == 0) {
		    even_sort<<<blocks, threads>>>(gary, SIZE);
		} else {
		    odd_sort<<<blocks, threads>>>(gary, SIZE);
		}
 
	}
	 
	cudaMemcpy(data->intarray, gary, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	 
	cudaFree(gary);
	 
	 
}
