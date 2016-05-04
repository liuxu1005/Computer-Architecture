#include "merge_sort.h"

void merge_sort(Data *data) {
    clock_t begin, end;
    double time_spent;

    if (data->array_used == INT) {
        begin = clock();       
        merge_sort_int(data->intarray, 0, data->length - 1);
        end = clock();

    } else if (data->array_used == FLOAT) {
        begin = clock();
        merge_sort_float(data->floatarray, 0, data->length - 1);
        end = clock();
 
    }
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        fprintf(stdout, "Serial Merge Sort time: %f\n", time_spent);
};

void merge_sort_int(int *arr, int leftIndex, int rightIndex) {

 

    if (leftIndex >= rightIndex)
        return;

    int middle = (leftIndex + rightIndex) / 2;
    //divide
    merge_sort_int(arr, leftIndex, middle);
    merge_sort_int(arr, middle + 1, rightIndex);

    //merge
    int length = (rightIndex - leftIndex + 1);
    int *tmp = (int *)malloc(length * sizeof(int));

    int lhalf = leftIndex;
    int rhalf = middle + 1;
    int tmpIndex = 0;

    while (lhalf <= middle && rhalf <= rightIndex) {
        tmp[tmpIndex++] = arr[lhalf] < arr[rhalf] ? arr[lhalf++] : arr[rhalf++];
    }

    while (lhalf <= middle) {
        tmp[tmpIndex++] = arr[lhalf++];
    }

    while (rhalf <= rightIndex) {

        tmp[tmpIndex++] = arr[rhalf++];
    }

    tmpIndex = 0;
    lhalf = leftIndex;
    //copy back
    while (lhalf <= rightIndex) {
        arr[lhalf++] = tmp[tmpIndex++];
    }





}

void merge_sort_float(float *arr, int leftIndex, int rightIndex) {

    if (leftIndex >= rightIndex)
        return;

    int middle = (leftIndex + rightIndex) / 2;
    //divide
    merge_sort_float(arr, leftIndex, middle);
    merge_sort_float(arr, middle + 1, rightIndex);

    //merge
    int length = (rightIndex - leftIndex + 1);
    float *tmp = (float *)malloc(length * sizeof(float));

    int lhalf = leftIndex;
    int rhalf = middle + 1;
    int tmpIndex = 0;

    while (lhalf <= middle && rhalf <= rightIndex) {
        tmp[tmpIndex++] = arr[lhalf] < arr[rhalf] ? arr[lhalf++] : arr[rhalf++];
    }

    while (lhalf <= middle) {
        tmp[tmpIndex++] = arr[lhalf++];
    }

    while (rhalf <= rightIndex) {

        tmp[tmpIndex++] = arr[rhalf++];
    }

    tmpIndex = 0;
    lhalf = leftIndex;
    //copy back
    while (lhalf <= rightIndex) {
        arr[lhalf++] = tmp[tmpIndex++];
    }

    return;

}
