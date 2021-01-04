#ifndef _IMPORTS
#define _IMPORTS

#include "../headers/imports.h"

#endif

__global__
void checkDifference(real*g_idata,real*outData){
    __shared__ double sdata[1024];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = (g_idata[i]<g_idata[i+blockDim.x])?g_idata[i+blockDim.x]:g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] = sdata[tid]<sdata[tid + s]?sdata[tid+s]:sdata[tid];
        }
        __syncthreads();
    }
        // write result for this block to global mem
    if (tid == 0) outData[blockIdx.x] = sdata[0];
}
__device__
void thomas(real*a,real*b,real*c,real*x,real*z,int sizeSmallerSystem,int stride){
    int systemSize = stride*sizeSmallerSystem;
    int i = threadIdx.x;
    c[i] = c[i] / b[i];
    z[i] = z[i] / b[i];
    int startLocationSystem = stride + i;
    for (int i = startLocationSystem;i<systemSize;i += stride){
        real tmp = b[i]-a[i]*c[i-stride];
        c[i]  = c[i] / tmp;
        z[i]  = (z[i]-z[i-stride]*a[i]) / tmp;
    }
    int endLocationSystem = systemSize-stride + i;
    x[endLocationSystem] = z[endLocationSystem];
    for (int i = endLocationSystem-stride;i>= 0;i-= stride) x[i] = z[i]-c[i]*x[i + stride];
}

__device__
void thomas(real*a,real*b,real*c,real*x,real*z,real* sol, real*err,int sizeSmallerSystem,int stride){
    int systemSize = stride*sizeSmallerSystem;
    int i = threadIdx.x;
    c[i] = c[i] / b[i];
    z[i] = z[i] / b[i];
    int startLocationSystem = stride + i;
    for (int i = startLocationSystem;i<systemSize;i += stride){
        real tmp = b[i]-a[i]*c[i-stride];
        c[i]  = c[i] / tmp;
        z[i]  = (z[i]-z[i-stride]*a[i]) / tmp;
    }
    int endLocationSystem = systemSize-stride + i;
    x[endLocationSystem] = z[endLocationSystem];
    for (int i = endLocationSystem-stride;i>= 0;i-= stride) {
        x[i] = z[i]-c[i]*x[i + stride];
        err[i] = abs(sol[i]-x[i]);
    }
}

__device__ 
void PCRTHOMAS2048global(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem,int sizeSmallerSystem) {
    int delta = 1;
    int i = threadIdx.x;
    int i1 = i+1024;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i+delta;
        iRight = (iRight>=sizeSystem)?sizeSystem-1:iRight;
        int iLeft = i-delta;
        iLeft = (iLeft <0)?0:iLeft;

        real tmp1 = a[i] / b[iLeft];
        real tmp2 = c[i] / b[iRight];
        real bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        real zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        real aNew = -a[iLeft] * tmp1;
        real cNew = -c[iRight] * tmp2;
        
        iRight = i1+delta;
        iRight = (iRight>=sizeSystem)?sizeSystem-1:iRight;
        iLeft = i1-delta;
        iLeft = (iLeft <0)?0:iLeft;
        tmp1 = a[i1] / b[iLeft];
        tmp2 = c[i1] / b[iRight];
        real bNew1 = b[i1] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        real zNew1 = z[i1] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        real aNew1 = -a[iLeft] * tmp1;
        real cNew1 = -c[iRight] * tmp2;

        __syncthreads();
        b[i] = bNew;
        z[i] = zNew;
        a[i] = aNew;
        c[i] = cNew;
        b[i1] = bNew1;
        z[i1] = zNew1;
        a[i1] = aNew1;
        c[i1] = cNew1;	
        __syncthreads();
        delta *= 2;
    }
    int thomasstride = sizeSystem/sizeSmallerSystem;
    if (i < thomasstride) {
        thomas(a,b,c,x,z,sizeSmallerSystem,thomasstride);
    }
}
__device__ 
void PCRTHOMAS2048global(real * a, real * b, real * c, real * x, real * z, real* sol, real* err, int numSteps, int sizeSystem,int sizeSmallerSystem) {
    int delta = 1;
    int i = threadIdx.x;
    int i1 = i+1024;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i+delta;
        iRight = (iRight>=sizeSystem)?sizeSystem-1:iRight;
        int iLeft = i-delta;
        iLeft = (iLeft <0)?0:iLeft;

        real tmp1 = a[i] / b[iLeft];
        real tmp2 = c[i] / b[iRight];
        real bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        real zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        real aNew = -a[iLeft] * tmp1;
        real cNew = -c[iRight] * tmp2;
        
        iRight = i1+delta;
        iRight = (iRight>=sizeSystem)?sizeSystem-1:iRight;
        iLeft = i1-delta;
        iLeft = (iLeft <0)?0:iLeft;
        tmp1 = a[i1] / b[iLeft];
        tmp2 = c[i1] / b[iRight];
        real bNew1 = b[i1] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        real zNew1 = z[i1] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        real aNew1 = -a[iLeft] * tmp1;
        real cNew1 = -c[iRight] * tmp2;

        __syncthreads();
        b[i] = bNew;
        z[i] = zNew;
        a[i] = aNew;
        c[i] = cNew;
        b[i1] = bNew1;
        z[i1] = zNew1;
        a[i1] = aNew1;
        c[i1] = cNew1;	
        __syncthreads();
        delta *= 2;
    }
    int thomasstride = sizeSystem/sizeSmallerSystem;
    if (i < thomasstride) {
        thomas(a,b,c,x,z,sol,err,sizeSmallerSystem,thomasstride);
    }
}

__global__
void fillTridiagonals_Row(int nx,int ny,real alpha,real dyodx,real dxdy,real *devA,real *devB,real *devC,real *devZ,real *sol_rows,real *sol_cols,real* sol,real* errArray,real *fArray, int *code,int elements){
	real alp2dyodx = alpha+2*dyodx;
	real dxody = ((real)1)/dyodx;
	dyodx *=-1;
	real __shared__ bb[2048];
	real __shared__ cc[2048];
	real __shared__ zz[2048];
	int idx = blockIdx.x+2;
	int index = idx*nx;
	for(int e = 0; e < elements; e++){	
		int j = 2+elements*threadIdx.x+e;
		int sharedJ = elements*threadIdx.x+e;
		int indJ = index+j;
		int ind = idx+j*ny;
		bb[sharedJ] = (code[indJ]==2) ? alp2dyodx : 1;
		devA[indJ] = (code[indJ]==2) ? dyodx : 0;
		cc[sharedJ] = (code[indJ]==2) ? dyodx : 0;
		zz[sharedJ]= (code[indJ]==2) ? 
			((sol_cols[ind-1]-sol_cols[ind]-sol_cols[ind]+sol_cols[ind+1])*dxody + fArray[indJ]*dxdy + alpha*sol_cols[ind]) :
			code[indJ]*sol_rows[index + j];
	}
	__syncthreads();
	PCRTHOMAS2048global(&devA[idx*nx+2],bb,cc,&sol_rows[idx*nx+2],zz,&sol[idx*nx+2],&errArray[idx*nx+2],7,2048,8);
	//CRPCR(&devA[idx*nx+2],bb,cc,&sol_rows[idx*nx+2],zz,10,1024,2048,2);
}

__global__
void fillTridiagonals_Col(int nx,int ny,real alpha,real dxody,real dxdy,real *devA,real *devB,real *devC,real *devZ,real *sol_rows,real *sol_cols,real *fArray, int *code,int elements){
	real dyodx = ((real)1)/dxody;
	real alp2dxody = alpha+2*dxody;
	dxody*=-1;
	int idx = blockIdx.x+2;
	real __shared__ bb[2048];
	real __shared__ cc[2048];
	real __shared__ zz[2048];

	for(int e = 0; e < elements; e++){	
		int j = 2+elements*threadIdx.x+e;
		int sharedJ = elements*threadIdx.x+e;
		int index2 = idx*ny+j;
		int ind = idx+j*nx;
		bb[sharedJ] = (code[ind]==2) ? alp2dxody : 1;
		devA[index2] = (code[ind]==2) ? dxody : 0;
		cc[sharedJ] = (code[ind]==2) ? dxody : 0;
		zz[sharedJ]= (code[ind]==2) ? 
			((sol_rows[ind - 1] - sol_rows[ind] - sol_rows[ind] + sol_rows[ind + 1])*dyodx + fArray[ind]*dxdy + alpha*sol_rows[ind]) :
			code[ind]*sol_cols[index2];
	} 
	__syncthreads();
	PCRTHOMAS2048global(&devA[idx*ny+2],bb,cc,&sol_cols[idx*ny+2],zz,7,2048,8);
}
 

void initGPUVariables(Cells *c,real *domSol,int nn,real *sol_cols,real *sol_rows,real *sol,real *fArray, int *code){
	cudaMemcpy(sol_cols,c->sol_cols,nn*sizeof(real),cudaMemcpyHostToDevice);
	cudaMemcpy(sol_rows,c->sol_rows,nn*sizeof(real),cudaMemcpyHostToDevice);
	cudaMemcpy(sol,domSol,nn*sizeof(real),cudaMemcpyHostToDevice);	
	cudaMemcpy(fArray,c->fArray,nn*sizeof(real),cudaMemcpyHostToDevice);	
	cudaMemcpy(code,c->code,nn*sizeof(int),cudaMemcpyHostToDevice);	
}
void freeGPUVariables(real *devA,real *devB,real *devC,real *devZ,real *sol_cols,real *sol_rows,real *sol,real *fArray, int *code){
	cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
	cudaFree(devZ);
    cudaFree(sol_cols);
    cudaFree(sol_rows);
    cudaFree(sol);
    cudaFree(fArray);
	cudaFree(code);
}

void solveADI(Domain *d, Grid *g, Cells *c){
	cudaStream_t stream1;
	cudaError_t cudaErr;
	cudaErr = cudaStreamCreate(&stream1);
	real *errArray,*devA,*devB,*devC,*devZ,*result; //Tridiagonal vectors 
	real *sol_cols,*sol_rows,*sol,*fArray,*tempErr;//Cell variables on GPU
	int *code,elements;
	real *calcErr;
	real err = 1e3, err_ant = 1e2, epsi = 3e-13;
	int nx = g->size_x,ny = g->size_y, it=0,nn = nx*ny;
	
	real dx = g->max_x / (real)nx;
	real dy = g->max_y / (real)ny;
	real dxdy = dx*dy;
	real dxody = dx/dy;
	real dyodx = dy/dx; 
	real alpha = d->alpha;

	cudaMalloc((void**) &tempErr, 2048* sizeof(real));
	cudaMalloc((void**) &result, sizeof(real));
	cudaMalloc((void**) &errArray, nn * sizeof(real));
	cudaMalloc((void**) &devA, nn * sizeof(real));
    cudaMalloc((void**) &devB, nn * sizeof(real));
    cudaMalloc((void**) &devC, nn * sizeof(real));
	cudaMalloc((void**) &devZ, nn * sizeof(real));

	cudaMalloc((void**) &calcErr, sizeof(real));
	
    cudaMalloc((void**) &sol_cols, nn * sizeof(real));
    cudaMalloc((void**) &sol_rows, nn * sizeof(real));
    cudaMalloc((void**) &sol, nn * sizeof(real));
    cudaMalloc((void**) &fArray, nn * sizeof(real));
	cudaMalloc((void**) &code, nn * sizeof(int));

	initGPUVariables(c,d->sol,nn,sol_cols,sol_rows,sol,fArray,code);

	startStopWatch();
	while(it<10000 && (fabs(err-err_ant) > epsi)){

		err_ant = err;
		//err--;
		
		fillTridiagonals_Row<<<2048,1024>>>(nx,ny,alpha,dyodx,dxdy,devA,devB,devC,devZ,sol_rows,sol_cols,sol,errArray,fArray,code,2);
        checkDifference<<<2048,1024>>>(errArray,tempErr);cudaDeviceSynchronize();checkDifference<<<1,1024>>>(tempErr,result); cudaDeviceSynchronize();
        cudaMemcpyAsync(&err,result,sizeof(real),cudaMemcpyDeviceToHost);
		
		fillTridiagonals_Col<<<2048,1024>>>(nx,ny,alpha,dxody,dxdy,devA,devB,devC,devZ,sol_rows,sol_cols,fArray,code,2);
		
		it++;
		//if(it%100==0)*/ printf("Iteration %d - %.3e\n",it,err);
	}
	cudaMemcpy(c->sol_cols,sol_cols,nn*sizeof(real),cudaMemcpyDeviceToHost);/* 
	printMatrix(c->sol_cols,nx,ny); */
	printf("Iterations:%d\nError:%f - %f\n",it,err,err_ant);
	cudaErr = cudaStreamDestroy(stream1);
	cudaErr = cudaGetLastError();
	if (!cudaSuccess==cudaErr){
		printf("ERROR:%d\n",cudaErr);
        exit(-1);
    }
	printf("Time:%f s\n",stopStopWatch()/1000);

    freeGPUVariables(devA,devB,devC,devZ,sol_cols,sol_rows,sol,fArray,code);
}
