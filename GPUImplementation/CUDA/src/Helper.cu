#ifndef _IMPORTS
#define _IMPORTS

#include "../headers/imports.h"

#endif

using namespace std;
cudaEvent_t start, stop;

void handleCudaError(int error) {
    switch (error) {
        case cudaSuccess:
            return;
        case cudaErrorInvalidValue:
            printf("Cuda Error: Invalide Value, have you created all events?\n");
            break;
        case cudaErrorInitializationError:
            printf("Cuda Error: Initialization Error\n");
            break;
        case cudaErrorPriorLaunchFailure:
            printf("Cuda Error: Prior Launch Failure, An error has appeared somewhere :/\n");
            break;
        case cudaErrorInvalidResourceHandle:
            printf("Cuda Error: Invalide Resource Handle\n");
            break;
        default:
            printf("ERROR NOT RECOGNIZED:%s\n", cudaGetErrorString((cudaError_t) error));
    }
    exit(-1);
}

void startStopWatch () {
	handleCudaError(cudaEventCreate(&start));
	handleCudaError(cudaEventCreate(&stop));
	handleCudaError(cudaEventRecord(start));
}

float stopStopWatch () {
	handleCudaError(cudaEventRecord(stop));
	handleCudaError(cudaEventSynchronize(stop));
	float time = 0;
    handleCudaError(cudaEventElapsedTime(&time, start, stop));
    return time;
}

void printMatrix(real* matrix,int x,int y){
	for(int i=0;i<y;i++){
		for(int j=0;j<x;j++){
			if(matrix[i*x+j]<0.001) printf("0 ");
			else printf("%f ",matrix[i*x+j]);
		}
		printf("\n");
	}
}
 
