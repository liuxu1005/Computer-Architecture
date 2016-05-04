#ifndef MERGE_H
#define MERGE_H

#include "structs.h"
#include <time.h>
#include <stdio.h>

void merge_sort_int(int *arr, int leftIndex, int rightIndex);
void merge_sort_float(float *arr, int leftIndex, int rightIndex);
void merge_sort(Data *data);

#endif
