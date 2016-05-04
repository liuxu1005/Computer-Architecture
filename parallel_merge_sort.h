#ifndef PARALLEL_MERGESORT
#define PARALLEL_MERGESORT


#include <stdio.h>
#include <time.h>
#include "structs.h"
__global__ void merge_int(int *input, int *output, int length, int size);
__global__ void merge_float(float *input, float *output, int length, int size);
void cal_grid(Grid *grid, int *threads, int length, int size);
void parallel_merge_sort_int(Data *data);
void parallel_merge_sort_float(Data *data);
void parallel_merge_sort(Data *data);


#endif
