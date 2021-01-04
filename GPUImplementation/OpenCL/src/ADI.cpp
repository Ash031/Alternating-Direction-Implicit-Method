#include"../headers/imports.h"

#define real double
#define SIZE 512


void ADI(char** argv, int argc){
    OpenCLVars *vars = (OpenCLVars*)malloc(sizeof(OpenCLVars));
    initOpenCL(vars);

    Cells* cells = (Cells*)malloc(sizeof(Cells));
    Domain* d = (Domain*)malloc(sizeof(Domain));
    Grid *g = (Grid*)malloc(sizeof(Grid));
    init(g,d,cells);
    int nx = g->size_x,ny = g->size_y,ret,nn=nx*ny;
    double dx = g->max_x / (double)nx;
	double dy = g->max_y / (double)ny;
	double dxdy = dx*dy;
	double dxody = dx/dy;
	double dyodx = dy/dx; 
	double alpha = d->alpha;

    cl_kernel ADIRowkernel = clCreateKernel(vars->program, "ADIROW", &ret);
    if(ret!=0)printf("clCreateKernel Error:%d\n",ret);
    cl_kernel calcError1 = clCreateKernel(vars->program, "getMax", &ret);
    if(ret!=0)printf("clCreateKernel Error:%d\n",ret);
    cl_kernel calcError2 = clCreateKernel(vars->program, "getMax", &ret);
    if(ret!=0)printf("clCreateKernel Error:%d\n",ret);
    cl_kernel ADIColKernel = clCreateKernel(vars->program, "ADICOL", &ret);
    if(ret!=0)printf("clCreateKernel Error:%d\n",ret);

    size_t local_item_size = 1024;
    size_t global_item_size = local_item_size*2048; 
    int elementsperThread=nx/local_item_size,numSteps=log2(nx)-1;
    double err=1000,errAnt=0,eps=3e-13;

    cl_mem tmpErr = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, 2048 * sizeof(double), NULL, &ret);
    cl_mem itErr= clCreateBuffer(vars->context, CL_MEM_READ_WRITE, sizeof(double), NULL, &ret);
    int shared = 1024*sizeof(double), workSizeError=2048; 
    size_t errGlobalSize = local_item_size*2048; 
     
    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, nn * sizeof(double), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, nn * sizeof(double), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, nn * sizeof(double), NULL, &ret);
    cl_mem z_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, nn * sizeof(double), NULL, &ret);
    cl_mem row_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, nn * sizeof(double), NULL, &ret);
    cl_mem col_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, nn * sizeof(double), NULL, &ret);
    cl_mem sol_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_ONLY, nn * sizeof(double), NULL, &ret);
    cl_mem err_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_WRITE, nn * sizeof(double), NULL, &ret);
    cl_mem fon_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_ONLY, nn * sizeof(double), NULL, &ret);
    cl_mem code_mem_obj = clCreateBuffer(vars->context, CL_MEM_READ_ONLY, nn * sizeof(int), NULL, &ret);
    
    ret += clEnqueueWriteBuffer(vars->command_queue, row_mem_obj, CL_TRUE, 0, sizeof(double) * nn, cells->sol_rows, 0, NULL, NULL);
    ret += clEnqueueWriteBuffer(vars->command_queue, col_mem_obj, CL_TRUE, 0, sizeof(double) * nn, cells->sol_cols, 0, NULL, NULL);
    ret += clEnqueueWriteBuffer(vars->command_queue, sol_mem_obj, CL_TRUE, 0, sizeof(double) * nn, d->sol, 0, NULL, NULL);
    ret += clEnqueueWriteBuffer(vars->command_queue, fon_mem_obj, CL_TRUE, 0, sizeof(double) * nn, cells->fArray, 0, NULL, NULL);
    ret += clEnqueueWriteBuffer(vars->command_queue, code_mem_obj, CL_TRUE, 0, sizeof(int) * nn, cells->code, 0, NULL, NULL);

    // Set the arguments of the kernel
    ret = clSetKernelArg(ADIRowkernel, 0, sizeof(int), (void *)&nx);
    ret = clSetKernelArg(ADIRowkernel, 1, sizeof(int), (void *)&ny);
    ret = clSetKernelArg(ADIRowkernel, 2, sizeof(double), (void *)&alpha);
    ret = clSetKernelArg(ADIRowkernel, 3, sizeof(double), (void *)&dyodx);
    ret = clSetKernelArg(ADIRowkernel, 4, sizeof(double), (void *)&dxdy);
    ret = clSetKernelArg(ADIRowkernel, 5, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 6, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 7, sizeof(cl_mem), (void *)&c_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 8, sizeof(cl_mem), (void *)&z_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 9, sizeof(cl_mem), (void *)&row_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 10, sizeof(cl_mem), (void *)&col_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 11, sizeof(cl_mem), (void *)&sol_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 12, sizeof(cl_mem), (void *)&err_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 13, sizeof(cl_mem), (void *)&fon_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 14, sizeof(cl_mem), (void *)&code_mem_obj);
    ret = clSetKernelArg(ADIRowkernel, 15, sizeof(cl_int), (void *)&elementsperThread);
    ret = clSetKernelArg(ADIRowkernel, 16, sizeof(cl_int), (void *)&numSteps);

    ret = clSetKernelArg(ADIColKernel, 0, sizeof(int), (void *)&nx);
    ret = clSetKernelArg(ADIColKernel, 1, sizeof(int), (void *)&ny);
    ret = clSetKernelArg(ADIColKernel, 2, sizeof(double), (void *)&alpha);
    ret = clSetKernelArg(ADIColKernel, 3, sizeof(double), (void *)&dxody);
    ret = clSetKernelArg(ADIColKernel, 4, sizeof(double), (void *)&dxdy);
    ret = clSetKernelArg(ADIColKernel, 5, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(ADIColKernel, 6, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(ADIColKernel, 7, sizeof(cl_mem), (void *)&c_mem_obj);
    ret = clSetKernelArg(ADIColKernel, 8, sizeof(cl_mem), (void *)&z_mem_obj);
    ret = clSetKernelArg(ADIColKernel, 9, sizeof(cl_mem), (void *)&row_mem_obj);
    ret = clSetKernelArg(ADIColKernel, 10, sizeof(cl_mem), (void *)&col_mem_obj);
    ret = clSetKernelArg(ADIColKernel, 11, sizeof(cl_mem), (void *)&sol_mem_obj);
    ret = clSetKernelArg(ADIColKernel, 12, sizeof(cl_mem), (void *)&fon_mem_obj);
    ret = clSetKernelArg(ADIColKernel, 13, sizeof(cl_mem), (void *)&code_mem_obj);
    ret = clSetKernelArg(ADIColKernel, 14, sizeof(cl_int), (void *)&elementsperThread);
    ret = clSetKernelArg(ADIColKernel, 15, sizeof(cl_int), (void *)&numSteps);

    ret = clSetKernelArg(calcError1, 0, sizeof(cl_mem), (void *)&err_mem_obj);
    ret = clSetKernelArg(calcError1, 1, sizeof(cl_mem), (void *)&tmpErr);
    ret = clSetKernelArg(calcError1, 2, sizeof(cl_int), &errGlobalSize);
    ret = clSetKernelArg(calcError1, 3, shared, NULL);

    ret = clSetKernelArg(calcError2, 0, sizeof(cl_mem), (void *)&tmpErr);
    ret = clSetKernelArg(calcError2, 1, sizeof(cl_mem), (void *)&itErr);
    ret = clSetKernelArg(calcError2, 2, sizeof(cl_int), &workSizeError);
    ret = clSetKernelArg(calcError2, 3, shared, NULL);
 
 
    // Execute the OpenCL kernel on the list
    cl_event kernRun;
    int it = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    for(;/* fabs(err-errAnt)>eps && */ it <1000; it++){
        errAnt = err;

        auto ret = clEnqueueNDRangeKernel(vars->command_queue, ADIRowkernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &kernRun);
        clFlush(vars->command_queue);
        clWaitForEvents(1,&kernRun);
        
        ret = clEnqueueNDRangeKernel(vars->command_queue, calcError1, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &kernRun);
        clFlush(vars->command_queue);
        clWaitForEvents(1,&kernRun);
        ret = clEnqueueNDRangeKernel(vars->command_queue, calcError2, 1, NULL, &local_item_size, &local_item_size, 0, NULL, &kernRun);
        clFlush(vars->command_queue);
        clWaitForEvents(1,&kernRun);
        ret = clEnqueueReadBuffer(vars->command_queue, itErr, CL_FALSE, 0,sizeof(double),&err, 0, NULL, &kernRun);
        if(ret!=0){printf("clEnqueueReadBuffer Error:%d\n",ret);exit(-1);}
        
        ret = clEnqueueNDRangeKernel(vars->command_queue, ADIColKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &kernRun);
        clFlush(vars->command_queue);
        clWaitForEvents(1,&kernRun);
        

    }
    auto t_end = std::chrono::high_resolution_clock::now();
    printf("CR Time: %f ms in %d iterations\n", std::chrono::duration<double, std::milli>(t_end - t_start).count(),it);

    ret = clEnqueueReadBuffer(vars->command_queue, row_mem_obj, CL_TRUE, 0,2052*2052*sizeof(double),cells->sol_rows, 0, NULL, &kernRun);
    if(ret!=0)printf("clEnqueueReadBuffer Error:%d\n",ret);

    printf("tmpErr[%d]=%f\n",it,err);

    
    
    // Clean up
    ret = clFlush(vars->command_queue);
    ret = clFinish(vars->command_queue);
    ret = clReleaseKernel(ADIRowkernel);
    
    clReleaseMemObject(a_mem_obj);
    clReleaseMemObject(b_mem_obj);
    clReleaseMemObject(c_mem_obj);
    clReleaseMemObject(z_mem_obj);
}