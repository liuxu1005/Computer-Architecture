#include "parallel_merge_sort.h"

void parallel_merge_sort(Data *data) {
    if (data->array_used == INT) {
        parallel_merge_sort_int(data);
    } else {
        parallel_merge_sort_float(data);
    }
}



__global__ void merge_int(int *input, int *output, int length, int size) {

    int tid = (blockIdx.z * gridDim.x * gridDim.y
                + blockIdx.y * gridDim.x
                + blockIdx.x) * blockDim.x + threadIdx.x;
    int index1 = tid * size;
    int index2 = index1 + size/2;
    int end1 = index2 > length ? length : index2;
    int end2 = index2 +  size/2 > length ? length : index2 + size/2;
    int tmpIndex = index1;

    while (index1 < end1 && index2 < end2) {
        if (input[index1] <= input[index2]) {
            output[tmpIndex++] = input[index1++];
        } else {
            output[tmpIndex++] = input[index2++];
        }
    }

    while (index1 < end1) {
        output[tmpIndex++] = input[index1++];
    }
    while (index2 < end2) {
        output[tmpIndex++] = input[index2++];
    }

}

__global__ void merge_float(float *input, float *output, int length, int size) {

    int tid = (blockIdx.z * gridDim.x * gridDim.y
                + blockIdx.y * gridDim.x
                + blockIdx.x) * blockDim.x + threadIdx.x;

    int index1 = tid * size;
    int index2 = index1 + size/2;
    int end1 = index2 > length ? length : index2;
    int end2 = index2 +  size/2 > length ? length : index2 + size/2;
    int tmpIndex = index1;


    while (index1 < end1 && index2 < end2) {
        if (input[index1] <= input[index2]) {
            output[tmpIndex++] = input[index1++];
        } else {
            output[tmpIndex++] = input[index2++];
        }
    }

    while (index1 < end1) {
        output[tmpIndex++] = input[index1++];
    }
    while (index2 < end2) {
        output[tmpIndex++] = input[index2++];
    }

}


 
void cal_grid(Grid *grid, int *threads, int length, int size) {
 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
 
    long long unsigned tmp = (length + size - 1)/size ;
 
    if (tmp <= prop.maxThreadsDim[0]) {
        *threads = tmp;
        grid->blockx = 1;
        grid->blocky = 1;
        grid->blockz = 1;
    } else {
	//if one block is not enough but 1 dimension grid is enough
        *threads = prop.maxThreadsDim[0];
        tmp =  (tmp + prop.maxThreadsDim[0] - 1)/prop.maxThreadsDim[0];
        if (tmp <= prop.maxGridSize[0]) {
            grid->blockx =  tmp;
            grid->blocky = 1;
            grid->blockz = 1;
        } else {
	    //if 1 dimension grid is not enough but 2 dimension grid is enough 
            tmp = (tmp + prop.maxGridSize[0] - 1)/prop.maxGridSize[0];
            if (tmp <= prop.maxGridSize[1]) {
                grid->blockx =  prop.maxGridSize[0];
                grid->blocky = tmp;
                grid->blockz = 1;
            } else {
                tmp = (tmp + prop.maxGridSize[1] - 1)/prop.maxGridSize[1];
                if (tmp <= prop.maxGridSize[2]) {
                    grid->blockx =  prop.maxGridSize[0];
                    grid->blocky = prop.maxGridSize[1];
                    grid->blockz = tmp;
                } else
                    fprintf(stderr, "The arrary is too large.\n");
            }

        }

    }

}
void parallel_merge_sort_float(Data *data) {

    int threads;
    Grid grid;
    int size = 2;
    float *input, *tmp, *output;
    int length = sizeof(float) * data->length;

    clock_t begin, end;
    double time_spent;
    begin = clock();    
    cudaMalloc((void**)&input, length);
    cudaMalloc((void**)&output, length);
    cudaMemcpy(input, data->floatarray, length, cudaMemcpyHostToDevice);


    while (size/2 < data->length) {

	    cal_grid(&grid, &threads, data->length, size);

        dim3 blocks(grid.blockx, grid.blocky, grid.blockz);
        merge_float<<<blocks, threads>>>(input, output, data->length, size);

        size = size * 2;
        tmp = input;
        input = output;
        output = tmp;

    }

    cudaMemcpy(data->floatarray, input, length, cudaMemcpyDeviceToHost);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    fprintf(stdout, "Parallel Merge sort time: %f\n", time_spent);
}


void parallel_merge_sort_int(Data *data) {

    int threads;
    Grid grid;
    int size = 2;

    clock_t begin, end;
    double time_spent;
        
    int *input, *tmp, *output;
    int length = sizeof(int) * data->length;
    begin = clock();
    cudaMalloc((void**)&input, length);
    cudaMalloc((void**)&output, length);
    cudaMemcpy(input, data->intarray, length, cudaMemcpyHostToDevice);


    while (size/2 < data->length) {

	    cal_grid(&grid, &threads, data->length, size);

        dim3 blocks(grid.blockx, grid.blocky, grid.blockz);
        merge_int<<<blocks, threads>>>(input, output, data->length, size);

        size = size * 2;

        tmp = input;
        input = output;
        output = tmp;

    }

    cudaMemcpy(data->intarray, input, length, cudaMemcpyDeviceToHost);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    fprintf(stdout, "Parallel Merge sort time: %f\n", time_spent);
}
