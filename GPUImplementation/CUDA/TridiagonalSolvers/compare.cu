/**
   * Code based on the book "GPU Computing Gems Jade Edition"
   */

#include "tridiagonalSolvers.h"
cudaEvent_t start, stop;
/**
   * Function to check errors in CUDA
   * @param error Error to return message.
   */
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
//Start Timer
void startStopWatch() {
    handleCudaError(cudaEventCreate( & start));
    handleCudaError(cudaEventCreate( & stop));
    handleCudaError(cudaEventRecord(start, 0));
}

//Stop Timer, returns the time in milliseconds
float stopStopWatch() {
    handleCudaError(cudaEventRecord(stop, 0));
    handleCudaError(cudaEventSynchronize(stop));
    float time = 0;
    handleCudaError(cudaEventElapsedTime( & time, start, stop));
    return time;
}
/**
   * Solves Ax=z using Thomas algorithm
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param N Size of the whole System
   */
void solve_thomas(real *a,real *b,real *c,real *x,real *z, int N){
	for(int k = 1; k <= N; k++){
		real m = -a[k] / b[k - 1];
        if(isnan(m)){ printf("NaN Found in m[%d]! Exiting! - a[%d]=%f - b[%d]=%f\n",k,k,a[k],k-1,b[k-1]);exit(-1);}
        b[k] = b[k] + m * c[k - 1];
        if(isnan(b[k])){ printf("NaN Found in b[%d]! Exiting!\n",k);exit(-1);}
		z[k] = z[k] + m * z[k - 1];
        if(isnan(z[k])){ printf("NaN Found in z[%d]! Exiting!\n",k);exit(-1);}
	}
	x[N-1] = z[N-1] / b[N-1];
	for(int k = N-2; k >= 0; k--){
		x[k] = (z[k] - c[k]*x[k + 1]) / b[k];
        if(isnan(x[k])){ printf("NaN Found in x[%d]! Exiting!\n",k);exit(-1);}
	}
}

/**
   * Initializes random tridiagonal systems of linear equations
   * @param a Lower Diagonals
   * @param b Main Diagonals
   * @param c Upper Diagonals
   * @param x Solutions
   * @param z Vectors Z
   * @param size Size of each system
   * @param systems Number of systems
   */
void initVectorsLocal(real** a,real** b,real** c,real** x, real** z,int size,int systems){
    *a = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    *b = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    *c = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    *x = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    *z = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    for(int i=0;i<systems;i++){
        (*z)[i*size] = 1;
        (*a)[i*size] = 0;
        (*b)[i*size] = 1;
        (*c)[i*size] = 0;
        (*z)[(i+1)*size-1] = 1;
        (*a)[(i+1)*size-1] = 0;
        (*b)[(i+1)*size-1] = 1;
        (*c)[(i+1)*size-1] = 0;
    
        for (int j = 1; j < size-1; j++){
            (*z)[i*size+j] = ((real)rand()) / ((real)RAND_MAX);;
            (*a)[i*size+j] = ((real)rand()) / ((real)RAND_MAX);;
            (*b)[i*size+j] = ((real)rand()) / ((real)RAND_MAX);;
            (*c)[i*size+j] = ((real)rand()) / ((real)RAND_MAX);;
        }
    }
}
/**
   * Initializes Identity matrices
   * @param a Lower Diagonals
   * @param b Main Diagonals
   * @param c Upper Diagonals
   * @param x Solutions
   * @param z Vectors Z
   * @param size Size of each system
   * @param systems Number of systems
   */
void initVectorsLocalId(real** a,real** b,real** c,real** x, real** z,int size,int systems){
    *a = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    *b = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    *c = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    *x = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    *z = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));

    for (int i = 0; i < systems*size; i++){
        ( * z)[i] = 1;
        ( * a)[i] = 0;
        ( * b)[i] = 1;
        ( * c)[i] = 0;
    }
}

//Function to compare tridiagonal solver algorithms.
//Uses Thomas algorithm as baseline.
void compareAll(int argc,char** argv){
    srand(0);
    int N = atoi(argv[1]);
    int TOTALSYSTEMS = 1;
    int smallerSystemSize = 2;
    if(argc>=4) TOTALSYSTEMS = atoi(argv[3]);
    if(argc>=5) smallerSystemSize = atoi(argv[4]);
    int bigN = N*TOTALSYSTEMS;
    int alg = atoi(argv[2]);
    real *a,*b,*c,*algAn,*z,*baseline;
    real *aa,*bb,*cc,*xx,*zz;
    cudaMalloc((void**)&aa,sizeof(real)*bigN);
    cudaMalloc((void**)&bb,sizeof(real)*bigN);
    cudaMalloc((void**)&cc,sizeof(real)*bigN);
    cudaMalloc((void**)&xx,sizeof(real)*bigN);
    cudaMalloc((void**)&zz,sizeof(real)*bigN);
    //INIT LINE
    initVectorsLocal(&a,&b,&c,&baseline,&z,N,TOTALSYSTEMS);
    algAn = (real*) malloc(sizeof(real)*bigN);
    cudaMemcpy(aa,a,bigN*sizeof(real),cudaMemcpyHostToDevice);
	cudaMemcpy(bb,b,bigN*sizeof(real),cudaMemcpyHostToDevice);
	cudaMemcpy(cc,c,bigN*sizeof(real),cudaMemcpyHostToDevice);	
    cudaMemcpy(zz,z,bigN*sizeof(real),cudaMemcpyHostToDevice);	
    
    for(int i=0;i<TOTALSYSTEMS;i++) solve_thomas(&a[i*N],&b[i*N],&c[i*N],&baseline[i*N],&z[i*N],N);
 
    //printf("\n\nSTARTING TESTS:\n");
    startStopWatch();
    if(alg==1)crKernel(aa,bb,cc,xx,zz,N,TOTALSYSTEMS);  
    if(alg==2)pcrKernel(aa,bb,cc,xx,zz,N,TOTALSYSTEMS);
    if(alg==3)crpcrKernel(aa,bb,cc,xx,zz,N,TOTALSYSTEMS,smallerSystemSize);
    if(alg==4)pcrThomasKernel(aa,bb,cc,xx,zz,N,TOTALSYSTEMS,smallerSystemSize);
    if(alg==5)crpcrThomasKernel(aa,bb,cc,xx,zz,N,TOTALSYSTEMS,smallerSystemSize);
    float time = stopStopWatch();
    printf("Time:%f\n",time);
    //printf("%f\n",time);

    cudaMemcpy(algAn,xx,bigN*sizeof(real),cudaMemcpyDeviceToHost);

    //COMPARE
    double totalDif=0,totalReal=0;
    int nan = 0;
    for(int i=0;i<bigN;i++){
        totalReal += abs(algAn[i]);
		if(!isnan(algAn[i])){
            totalDif += abs(baseline[i]-algAn[i]);
            //Use this line while working with identity matrices, will spam messages if turned on with random matrices.
            //if(baseline[i]-algAn[i]!=0) printf("Error in %d, should be %f and is %f, diff is %.3e\n",i,baseline[i],algAn[i],abs(baseline[i]-algAn[i]));
        }
        else nan++;
    }
    if(nan==0) printf("FINAL RESULTS:\nAlg Sum: %f\nAlg Error: %.3e\n",totalReal,totalDif);
    else printf("NaN found %d times!\n",nan);
}

int main(int argc, char** argv){
    if(argc<3) {
        printf("Choose a size and an alg\n");
        exit(-1);
    }
    compareAll(argc,argv);
    return 0;
}
