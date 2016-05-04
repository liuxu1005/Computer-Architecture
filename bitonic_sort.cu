#include "bitonic_sort.h"


__global__ void create_bitonic(int *input,  int stride, int length, int reference) {
    int tid = (blockIdx.z * gridDim.x * gridDim.y
                + blockIdx.y * gridDim.x
                + blockIdx.x) * blockDim.x + threadIdx.x; 
 
    int groupnumber = stride / 2;
    int group = tid / groupnumber;
    int order = tid % groupnumber;
    bool odd = (tid/ (reference/2)) % 2;
  
    
    int index = group * stride  + order;
    int index2 = index + (stride/2);
 
    bool less = input[index] < input[index2] ? 1 : 0;
    bool greater = input[index] > input[index2] ? 1 : 0;
        if ((!odd && greater)||(odd && less)) {
            int  tmp = input[index];
            input[index] = input[index2];
            input[index2] = tmp;
        }                      
}


__global__ void bitonic_sort(int *input,  int stride, int length) {
    int tid = (blockIdx.z * gridDim.x * gridDim.y
                + blockIdx.y * gridDim.x
                + blockIdx.x) * blockDim.x + threadIdx.x;   
    int groupnumber = stride / 2;
    int group = tid / groupnumber;
    int order = tid % groupnumber;
    
  
    
    int index = group * stride  + order;
    int index2 = index + (stride/2);
 
    if (input[index] > input[index2]) {
        int  tmp = input[index];
        input[index] = input[index2];
        input[index2] = tmp;        
        
    }       
        
}

__global__ void initial(int * input, int start, int m) {
    int tid = (blockIdx.z * gridDim.x * gridDim.y
                + blockIdx.y * gridDim.x
                + blockIdx.x) * blockDim.x + threadIdx.x;  
    input[start + tid] = m;
}
void bitonic_sort(Data *data) {
 

    int powerlength = pow(2, ceil(log2((double)(data->length))));
    int length = powerlength * sizeof(int);

    clock_t begin, end;
    double time_spent;
        
    
    int *input;
    cudaMalloc((void**)&input, length);
    cudaMemcpy(input, data->intarray, data->length * sizeof(int), cudaMemcpyHostToDevice);

    int threads;
    Grid grid; 
    //add fake data
    cal_grid(&grid, &threads, powerlength - data->length, 1);
    dim3 blocks(grid.blockx, grid.blocky, grid.blockz);
    initial<<<blocks, threads>>>(input, data->length, INT_MAX);    
 
    cal_grid(&grid, &threads, powerlength, 2);
 
    begin = clock();
   
    for (int i = 2; i < powerlength; i *= 2) {
        for (int j = i; j > 1; j /= 2) {
            create_bitonic<<<blocks, threads>>>(input, j, powerlength, i);
        }
    }
    
    //sort
    for (int k = powerlength; k > 1; k /= 2)  {
        bitonic_sort<<<blocks, threads>>>(input, k, powerlength); 
    }
        
    cudaMemcpy(data->intarray, input, data->length * sizeof(int), cudaMemcpyDeviceToHost);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
    fprintf(stdout, "Parallel Bitonic sort time: %f\n", time_spent);
    cudaFree(input);

}
