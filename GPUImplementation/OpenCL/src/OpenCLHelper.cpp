#include"../headers/imports.h"


void initOpenCL(OpenCLVars * vars){
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("kernels.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_device_type dType = CL_DEVICE_TYPE_GPU;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if(ret!=0)printf("clGetPlatformIDs Error:%d\n",ret);
    ret = clGetDeviceIDs( platform_id, dType, 1,&device_id, &ret_num_devices);
    if(ret!=0)printf("clGetDevicesIDs Error:%d\n",ret);
    

    // Create an OpenCL context
    vars->context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    if(ret!=0)printf("clCreateContext Error:%d\n",ret);
 
    // Create a command queue
    vars->command_queue = clCreateCommandQueue(vars->context, device_id, 0, &ret);
    if(ret!=0)printf("clCreateCommandQueue Error:%d\n",ret);
    // Create a program from the kernel source
    vars->program = clCreateProgramWithSource(vars->context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    if(ret!=0)printf("clCreateProgramWithSource Error:%d\n",ret);
    ret = clBuildProgram(vars->program, 1, &device_id, NULL, NULL, NULL);  
    if (ret != 0) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(vars->program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(vars->program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
}