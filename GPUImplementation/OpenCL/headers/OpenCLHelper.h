#ifndef HELPER
#define HELPER

#define MAX_SOURCE_SIZE (0x100000)

struct OpenCLVars{
    cl_program program;
    cl_context context;
    cl_command_queue command_queue;
    cl_kernel adiRow,adiCol,maxError;
};
void initOpenCL(OpenCLVars * vars);

#endif