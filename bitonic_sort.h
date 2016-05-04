#ifndef  BITONIC_H
#define  BITONIC_H

#include <cmath>
#include <stdio.h>
#include <time.h>
#include "parallel_merge_sort.h"
#include <climits>
 


__global__ void create_bitonic(int *input,  int stride, int length, int reference);
__global__ void bitonic_sort(int *input,  int stride, int length);

void bitonic_sort(Data *data);

#endif
