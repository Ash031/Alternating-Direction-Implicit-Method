#include"../headers/imports.h"

#define real double
#define SIZE 512

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
            (*z)[i*size+j] = ((real)rand()) / ((real)RAND_MAX);
            (*a)[i*size+j] = ((real)rand()) / ((real)RAND_MAX);
            (*b)[i*size+j] = ((real)rand()) / ((real)RAND_MAX);
            (*c)[i*size+j] = ((real)rand()) / ((real)RAND_MAX);
        }
    }
}
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
void testTridiagonal(char** argv, int argc){
    OpenCLVars *vars = (OpenCLVars*)malloc(sizeof(OpenCLVars));
    initOpenCL(vars);

    srand(0);
    int ret = 0;
    //Choose one of the algorithms in the kernels.cl file.
    //REMEMBER TO DOUBLE CHECK LOCAL AND GLOBAL MEMORY
    cl_kernel kernel = clCreateKernel(vars->program, "CRPCR",&ret); 
    if(ret!=0)printf("clCreateKernel Error:%d\n",ret);
    int size = 2048;
    size = (argc>=2) ? atoi(argv[1]) : 2048;
    int elements = 1;
    int smallSystems = 32;
    smallSystems = (argc>=3) ? atoi(argv[2]) : 32;
    int numStepsSmall = (int)log2(smallSystems)-1;
    //int numSteps = (int)log2(size)-1;//Use in CR/PCR
    int numSteps = (int)log2(size) -1-numStepsSmall;//Use in Hybrid Solvers
    int systems = 1;
    size_t local_item_size = size/2;//Use if first algorithm is either CR or PCR2048
    //size_t local_item_size = size;//Use else
    size_t global_item_size = local_item_size;
    double *a,*b,*c,*x,*z,*xx;      
    xx = (real*)_mm_malloc(size*systems*sizeof(real),sizeof(real));
    //Uncomment for extra debug info
    //printf("DEBUG INFO:\nSIZE: %d\n NUM STEPS: %d\nSMALL SYSTEM SIZE: %d\nSMALL SYSTEM STEPS: %d\n",size,numSteps,smallSystems,numStepsSmall);
    initVectorsLocal(&a,&b,&c,&x,&z,size,systems);

    cl_mem a_mem = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, NULL);
    clEnqueueWriteBuffer(vars->command_queue, a_mem, CL_TRUE, 0, sizeof(double) * size, a, 0, NULL, NULL);
    cl_mem b_mem = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, NULL);
    clEnqueueWriteBuffer(vars->command_queue, b_mem, CL_TRUE, 0, sizeof(double) * size, b, 0, NULL, NULL);
    cl_mem c_mem = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, NULL);
    clEnqueueWriteBuffer(vars->command_queue, c_mem, CL_TRUE, 0, sizeof(double) * size, c, 0, NULL, NULL);
    cl_mem z_mem = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, NULL);
    clEnqueueWriteBuffer(vars->command_queue, z_mem, CL_TRUE, 0, sizeof(double) * size, z, 0, NULL, NULL);
    cl_mem x_mem = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, NULL);
    
/* 
    // Use with CR/PCR
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem);if(ret!=0){printf("Error Setting args 0: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem);if(ret!=0){printf("Error Setting args 1: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem);if(ret!=0){printf("Error Setting args 2: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&z_mem);if(ret!=0){printf("Error Setting args 3: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&x_mem);if(ret!=0){printf("Error Setting args 4: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&size);if(ret!=0){printf("Error Setting args 5: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&numSteps);if(ret!=0){printf("Error Setting args 7: %d\n",ret);exit(-1);} 
    //Use if CR
    ret = clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&elements);if(ret!=0){printf("Error Setting args 7: %d\n",ret);exit(-1);}
*/
  // Use With CRPCR  
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem);if(ret!=0){printf("Error Setting args 0: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem);if(ret!=0){printf("Error Setting args 1: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem);if(ret!=0){printf("Error Setting args 2: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&z_mem);if(ret!=0){printf("Error Setting args 3: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&x_mem);if(ret!=0){printf("Error Setting args 4: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&size);if(ret!=0){printf("Error Setting args 5: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&smallSystems);if(ret!=0){printf("Error Setting args 6: %d\n",ret);exit(-1);} 
    ret = clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&numSteps);if(ret!=0){printf("Error Setting args 7: %d\n",ret);exit(-1);} 
    ret = clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&numStepsSmall);if(ret!=0){printf("Error Setting args 8: %d\n",ret);exit(-1);} 
/*   
    // Use with PCR Thomas
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem);if(ret!=0){printf("Error Setting args 0: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem);if(ret!=0){printf("Error Setting args 1: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem);if(ret!=0){printf("Error Setting args 2: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&x_mem);if(ret!=0){printf("Error Setting args 4: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&z_mem);if(ret!=0){printf("Error Setting args 3: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&size);if(ret!=0){printf("Error Setting args 5: %d\n",ret);exit(-1);}
    ret = clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&smallSystems);if(ret!=0){printf("Error Setting args 6: %d\n",ret);exit(-1);} 
    ret = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&numSteps);if(ret!=0){printf("Error Setting args 7: %d\n",ret);exit(-1);}
// */
    solve_thomas(a,b,c,x,z,size);
    cl_event kernRun;
    auto tmp = std::chrono::high_resolution_clock::now();

    ret = clEnqueueNDRangeKernel(vars->command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &kernRun);
    if(ret!=0){
        printf("Error running kernel: %d\n",ret);
        exit(-1);
    }
    clFlush(vars->command_queue);
    clWaitForEvents(1,&kernRun);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-tmp).count();

    ret = clEnqueueReadBuffer(vars->command_queue, x_mem, CL_TRUE, 0,size*sizeof(double),xx, 0, NULL, NULL);
    if(ret!=0){
        printf("Error copying to host: %d\n",ret);
        exit(-1);
    }
    double totalDif=0,totalReal=0;
    int nan = 0;
    for(int i=0;i<size;i++){
        totalReal += abs(xx[i]); 
		if(!isnan(xx[i])){
            totalDif += abs(x[i]-xx[i]);
            //Use only with identity matrices, random will spam messages
            //if(baseline[i]-algAn[i]!=0) printf("Error in %d, should be %f and is %f, diff is %.3e\n",i,baseline[i],algAn[i],abs(baseline[i]-algAn[i]));
        }
        else {
            nan++;
            //printf("NaN in %d\n",i);
        }
    }
    if(nan==0) 
        printf("FINAL RESULTS:\nTime: %f ms\nAlg Sum: %f\nAlg Error: %.3e\n",ms,totalReal,totalDif);
    else 
        printf("NaN found %d times!\n",nan);

}

// This file is used for tridiagonal only, ADI function here is to make sure that the only needed change from the main 
// executable is the input file while compiling, will always call the testTridiagonal function
void ADI(char** argv, int argc){
    testTridiagonal(argv, argc);
}