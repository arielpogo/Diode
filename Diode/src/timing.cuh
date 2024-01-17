//Written by Dr. Yicheng Tu
#include "core.h"

cudaEvent_t TIMING_START_EVENT, TIMING_STOP_EVENT;
float TIMING_ELAPSED_TIME;

void TIMING_START() {
    cudaEventCreate(&TIMING_START_EVENT);
    cudaEventCreate(&TIMING_STOP_EVENT);
    cudaEventRecord(TIMING_START_EVENT, 0);
}


void TIMING_STOP() {
    cudaEventRecord(TIMING_STOP_EVENT, 0);
    cudaEventSynchronize(TIMING_STOP_EVENT);
    cudaEventElapsedTime(&TIMING_ELAPSED_TIME, TIMING_START_EVENT, TIMING_STOP_EVENT);
    cudaEventDestroy(TIMING_START_EVENT);
    cudaEventDestroy(TIMING_STOP_EVENT);
}

void TIMING_PRINT() {
    printf("******* Total Running Time of Kernel = %f ms *******\n", TIMING_ELAPSED_TIME);
}
