#ifndef EVEN_ODD
#define EVEN_ODD
 
#include <cuda.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "structs.h" 
#include "parallel_merge_sort.h"
__global__ void even_sort(int* ary, int size);
 
__global__ void odd_sort(int* ary, int size);
 
void even_odd_sort(Data *data);
 
#endif
