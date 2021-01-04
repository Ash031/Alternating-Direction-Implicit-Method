/**
   *  Error checker implementation in OpenCL, based on sum reduction of nvidia https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
   * @param g_idata input array, looking for max in this one
   * @param g_odata output element, g_odata[0] will have max
   * @param n Size of the g_idata array
   * @param sdata local memory to be used to speedup the search
   */
__kernel void getMax(__global double *g_idata, __global double *g_odata, unsigned int n, __local double* sdata)
{
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    if (i + get_local_size(0) < n) 
        sdata[tid] = (sdata[tid] < g_idata[i+get_local_size(0)]) ? g_idata[i+get_local_size(0)] : sdata[tid];

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(unsigned int s=get_local_size(0)/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] = (sdata[tid] < sdata[tid + s]) ? sdata[tid+s] : sdata[tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem 
    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
}

/**
   * Solving Ax = d using CR 
   * This function scales for sizes bigger than 2048.
   * This fuction uses local memory on vector b and d
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param d Vector D
   * @param x Solution
   * @param system_size Size of the whole System
   * @param iterations Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param elementsPerThread Number of elements to work with in each thread
   */
__kernel void cyclic_branch_free_kernel(__global double *a, __local double *b, __global double *c, __local double *d, __global double *x,int system_size, int iterations, int elementsPerThread){
    int thid = get_local_id(0);

	int stride = 1;
    int half_size = system_size >> 1;
	int thid_num = half_size;
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
	// forward elimination
	for (int j = 0; j < iterations; j++){
		barrier(CLK_LOCAL_MEM_FENCE);

		stride <<= 1;
		int delta = stride >> 1;
        for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * thid;
            if (element < thid_num){ 
			    int i = stride * element + stride - 1;
			    int iRight = i+delta;
			    iRight = iRight & (system_size-1);
			    double tmp1 = a[i] / b[i-delta];
			    double tmp2 = c[i] / b[iRight];
			    b[i] = b[i] - c[i-delta] * tmp1 - a[iRight] * tmp2;
			    d[i] = d[i] - d[i-delta] * tmp1 - d[iRight] * tmp2;
			    a[i] = -a[i-delta] * tmp1;
			    c[i] = -c[iRight]  * tmp2;
		    }
        }

        thid_num >>= 1;
	}

    if (thid < 2){
		int addr1 = stride - 1;
		int addr2 = (stride << 1) - 1;
		double tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
    }
    
    // backward substitution
    thid_num = 2;
    for (int j = 0; j < iterations; j++){
		int delta = stride >> 1;
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * thid;
            if (element < thid_num) {
                int i = stride * element + (stride >> 1) - 1;
                if (i == delta - 1)
                    x[i] = (d[i] - c[i] * x[i+delta]) / b[i];
		        else
		            x[i] = (d[i] - a[i] * x[i-delta] - c[i] * x[i+delta]) / b[i];
            }
         }
		 stride >>= 1;
         thid_num <<= 1;
	}
}

/**
   * Solving Ax = d using CR 
   * This function scales for sizes bigger than 2048.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param d Vector D
   * @param x Solution
   * @param system_size Size of the whole System
   * @param iterations Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param elementsPerThread Number of elements to work with in each thread
   */
__kernel void CRGLOBAL(
	__global double *a, __global double *b, __global double *c, __global double *d, __global double *x,
	int system_size, int iterations, int elementsPerThread
	){
    
	int thid = get_local_id(0);

	int stride = 1;
    int half_size = system_size >> 1;
	int thid_num = half_size;
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
	// forward elimination
	for (int j = 0; j < iterations; j++){
		barrier(CLK_LOCAL_MEM_FENCE);

		stride <<= 1;
		int delta = stride >> 1;
        for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * thid;
            if (element < thid_num){ 
			    int i = stride * element + stride - 1;
			    int iRight = i+delta;
			    iRight = iRight & (system_size-1);
			    double tmp1 = a[i] / b[i-delta];
			    double tmp2 = c[i] / b[iRight];
			    b[i] = b[i] - c[i-delta] * tmp1 - a[iRight] * tmp2;
			    d[i] = d[i] - d[i-delta] * tmp1 - d[iRight] * tmp2;
			    a[i] = -a[i-delta] * tmp1;
			    c[i] = -c[iRight]  * tmp2;
		    }
        }

        thid_num >>= 1;
	}

    if (thid < 2){
		int addr1 = stride - 1;
		int addr2 = (stride << 1) - 1;
		double tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
    }
    
    // backward substitution
    thid_num = 2;
    for (int j = 0; j < iterations; j++){
		int delta = stride >> 1;
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * thid;
            if (element < thid_num) {
                int i = stride * element + (stride >> 1) - 1;
                if (i == delta - 1)
                    x[i] = (d[i] - c[i] * x[i+delta]) / b[i];
		        else
		            x[i] = (d[i] - a[i] * x[i-delta] - c[i] * x[i+delta]) / b[i];
            }
         }
		 stride >>= 1;
         thid_num <<= 1;
	}
}

/**
   * Solving Ax = d using CR 
   * This function scales for sizes bigger than 2048.
   * This fuction uses local memory on vector b and d
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param d Vector D
   * @param x Solution
   * @param err Array to store the error for each cell
   * @param sol Solution array 
   * @param system_size Size of the whole System
   * @param iterations Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param elementsPerThread Number of elements to work with in each thread
   */
__kernel void cyclic_branch_free_kernel_err(__global double *a, __local double *b, __global double *c, __local double *d, __global double *x, __global double *err,__global double *sol,int system_size, int iterations, int elementsPerThread){
    int thid = get_local_id(0);

	int stride = 1;
    int half_size = system_size >> 1;
	int thid_num = half_size;
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
	// forward elimination
	for (int j = 0; j < iterations; j++){
		barrier(CLK_LOCAL_MEM_FENCE);

		stride <<= 1;
		int delta = stride >> 1;
        for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * thid;
            if (element < thid_num){ 
			    int i = stride * element + stride - 1;
			    int iRight = i+delta;
			    iRight = iRight & (system_size-1);
			    double tmp1 = a[i] / b[i-delta];
			    double tmp2 = c[i] / b[iRight];
			    b[i] = b[i] - c[i-delta] * tmp1 - a[iRight] * tmp2;
			    d[i] = d[i] - d[i-delta] * tmp1 - d[iRight] * tmp2;
			    a[i] = -a[i-delta] * tmp1;
			    c[i] = -c[iRight]  * tmp2;
		    }
        }

        thid_num >>= 1;
	}

    if (thid < 2){
		int addr1 = stride - 1;
		int addr2 = (stride << 1) - 1;
		double tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
		err[addr1] = (x[addr1]<sol[addr1])?sol[addr1]-x[addr1]:x[addr1]-sol[addr1];
		err[addr2] = (x[addr2]<sol[addr2])?sol[addr2]-x[addr2]:x[addr2]-sol[addr2];
    }
    
    // backward substitution
    thid_num = 2;
    for (int j = 0; j < iterations; j++){
		int delta = stride >> 1;
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * thid;
            if (element < thid_num) {
                int i = stride * element + (stride >> 1) - 1;
                if (i == delta - 1)
                    x[i] = (d[i] - c[i] * x[i+delta]) / b[i];
		        else
		            x[i] = (d[i] - a[i] * x[i-delta] - c[i] * x[i+delta]) / b[i];
				err[i] = (x[i]<sol[i])?sol[i]-x[i]:x[i]-sol[i];
            }
         }
		 stride >>= 1;
         thid_num <<= 1;
	}
}

/**
   * Solving Ax = d using PCR 
   * This function scales for sizes up to 1024.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param d Vector D
   * @param x Solution
   * @param system_size Size of the whole System
   * @param iterations Number of Steps for the algorithm (log2(sizeSystem)-1)
   */
__kernel void pcr_branch_free_kernel(__global double *a, __global double *b, __global double *c, __global double *d, __global double *x, 
									 int system_size, int iterations)
{
	int thid = get_local_id(0);
    int blid = get_group_id(0);

	int delta = 1;
  
	float aNew, bNew, cNew, dNew;
  
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// parallel cyclic reduction
	for (int j = 0; j < iterations; j++){
		int i = thid;

		int iRight = i+delta;
		iRight = iRight & (system_size-1);

		int iLeft = i-delta;
		iLeft = iLeft & (system_size-1);

		float tmp1 = a[i] / b[iLeft];
		float tmp2 = c[i] / b[iRight];

		bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
		dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
		aNew = -a[iLeft] * tmp1;
		cNew = -c[iRight] * tmp2;

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
		b[i] = bNew;
 		d[i] = dNew;
		a[i] = aNew;
		c[i] = cNew;	
    
	    delta *= 2;
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}

	if (thid < delta)
	{
		int addr1 = thid;
		int addr2 = thid + delta;
		float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
		
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
	}
    
}
/**
   * Solving Ax = d using PCR 
   * This function is a size of 2048.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param d Vector D
   * @param x Solution
   * @param system_size Size of the whole System
   * @param iterations Number of Steps for the algorithm (log2(sizeSystem)-1)
   */
__kernel void pcr_branch_free_kernel2048(__global double *a, __local double *b, __global double *c, __local double *d, __global double *x, 
									 int system_size, int iterations)
{
	int thid = get_local_id(0);
    int blid = get_group_id(0);

	int delta = 1;
  
	float aNew, bNew, cNew, dNew;
	float aNew1, bNew1, cNew1, dNew1;
  
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// parallel cyclic reduction
	for (int j = 0; j < iterations; j++){
		int i = thid;

		int iRight = i+delta;
		iRight = iRight & (system_size-1);

		int iLeft = i-delta;
		iLeft = iLeft & (system_size-1);

		float tmp1 = a[i] / b[iLeft];
		float tmp2 = c[i] / b[iRight];

		bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
		dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
		aNew = -a[iLeft] * tmp1;
		cNew = -c[iRight] * tmp2;

		int i1=i+1024;
		iRight = i1+delta;
		iRight = iRight & (system_size-1);

		iLeft = i1-delta;
		iLeft = iLeft & (system_size-1);

		tmp1 = a[i1] / b[iLeft];
		tmp2 = c[i1] / b[iRight];

		bNew1 = b[i1] - c[iLeft] * tmp1 - a[iRight] * tmp2;
		dNew1 = d[i1] - d[iLeft] * tmp1 - d[iRight] * tmp2;
		aNew1 = -a[iLeft] * tmp1;
		cNew1 = -c[iRight] * tmp2;

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
		b[i] = bNew;
 		d[i] = dNew;
		a[i] = aNew;
		c[i] = cNew;
		b[i1] = bNew1;
 		d[i1] = dNew1;
		a[i1] = aNew1;
		c[i1] = cNew1;	
    
	    delta *= 2;
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}

	if (thid < delta)
	{
		int addr1 = thid;
		int addr2 = thid + delta;
		float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
		
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
	}
    
}
/**
   * Solving Ax = d using PCR and computes the error of the iteration
   * This function is a size of 2048.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param d Vector D
   * @param x Solution
   * @param system_size Size of the whole System
   * @param iterations Number of Steps for the algorithm (log2(sizeSystem)-1)
   */
__kernel void pcr_branch_free_kernel2048_err(__global double *a, __local double *b, __global double *c, __local double *d, __global double *x,
											 __global double *err, __global double *sol,
									 int system_size, int iterations)
{
	int thid = get_local_id(0);
    int blid = get_group_id(0);

	int delta = 1;
  
	float aNew, bNew, cNew, dNew;
	float aNew1, bNew1, cNew1, dNew1;
  
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// parallel cyclic reduction
	for (int j = 0; j < iterations; j++){
		int i = thid;

		int iRight = i+delta;
		iRight = iRight & (system_size-1);

		int iLeft = i-delta;
		iLeft = iLeft & (system_size-1);

		float tmp1 = a[i] / b[iLeft];
		float tmp2 = c[i] / b[iRight];

		bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
		dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
		aNew = -a[iLeft] * tmp1;
		cNew = -c[iRight] * tmp2;

		int i1=i+1024;
		iRight = i1+delta;
		iRight = iRight & (system_size-1);

		iLeft = i1-delta;
		iLeft = iLeft & (system_size-1);

		tmp1 = a[i1] / b[iLeft];
		tmp2 = c[i1] / b[iRight];

		bNew1 = b[i1] - c[iLeft] * tmp1 - a[iRight] * tmp2;
		dNew1 = d[i1] - d[iLeft] * tmp1 - d[iRight] * tmp2;
		aNew1 = -a[iLeft] * tmp1;
		cNew1 = -c[iRight] * tmp2;

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
		b[i] = bNew;
 		d[i] = dNew;
		a[i] = aNew;
		c[i] = cNew;
		b[i1] = bNew1;
 		d[i1] = dNew1;
		a[i1] = aNew1;
		c[i1] = cNew1;	
    
	    delta *= 2;
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}

	if (thid < delta)
	{
		int addr1 = thid;
		int addr2 = thid + delta;
		float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
		
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
		err[addr1] = (x[addr1]<sol[addr1])?sol[addr1]-x[addr1]:x[addr1]-sol[addr1];
		err[addr2] = (x[addr2]<sol[addr2])?sol[addr2]-x[addr2]:x[addr2]-sol[addr2];
	}
    
}

/**
   * Solving Ax = d using CRPCR
   * This function already uses local memory
   * This function works for sizes up to 2048
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param d Vector D
   * @param x Solution
   * @param system_size Size of the whole System
   * @param PCRSize Size of the PCR system
   * @param numSteps Number of iterations for CR
   * @param smallSteps Number of iterations for PCR
   */
__kernel void CRPCR(
	__global double *a, __local double *b, __global double *c, __local double *d, __global double *x,
	int system_size, int PCRSize, int numSteps, int smallSteps
	){
    
	int idx = get_local_id(0);

	int pcrStride= system_size/PCRSize;
	int stride = 1;
    int half_size = system_size >> 1;
	int thid_num = half_size;
	int i;
    for (int j = 0; j < numSteps; j++) {
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        stride *= 2;
        int delta = stride / 2;
        if (idx < thid_num) {
            int i = stride * idx + stride - 1;
            int iLeft = i - delta;
            if (iLeft < 0) iLeft = 0;
            int iRight = i + delta;
            if (iRight >= system_size) iRight = system_size - 1;

            double tmp1 = a[i] / b[iLeft];
    	    double tmp2 = c[i] / b[iRight];
            b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
            d[i] = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
            a[i] = -a[iLeft] * tmp1;
            c[i] = -c[iRight] * tmp2;
        }
        thid_num /= 2;
    }
	int deltaPCR = pcrStride;
    i = idx * pcrStride + pcrStride - 1;
	bool workingThread = idx < PCRSize;
    double aNew,bNew,cNew,dNew;
    for (int j = 0; j < smallSteps; j++) {
        if(workingThread){
            int iRight = i + deltaPCR;
               if (iRight >= system_size) iRight = system_size - 1;
               int iLeft = i - deltaPCR;
               if (iLeft < pcrStride-1) iLeft = pcrStride-1;
       
               double tmp1 = a[i] / b[iLeft];
               double tmp2 = c[i] / b[iRight];
               bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
               dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
               aNew = -a[iLeft] * tmp1;
               cNew = -c[iRight] * tmp2;
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if(workingThread){
            b[i] = bNew;
            d[i] = dNew;	
            a[i] = aNew;
            c[i] = cNew;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        deltaPCR *= 2;
	}
    if(workingThread){
           int addr1 = i;
           int addr2 = i  + deltaPCR ;
           double tmp3 = (b[addr2] * b[addr1] - c[addr1] * a[addr2]);
           x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
           x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
    }
	for (int j = 0; j < numSteps; j++) {
        int delta = stride / 2;
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
            if (idx < PCRSize) {
                int i = stride * idx + stride / 2 - 1;
                if (i == delta - 1) x[i] = (d[i] - c[i] * x[i + delta]) / b[i];
                else x[i] = (d[i] - a[i] * x[i - delta] - c[i] * x[i + delta]) / b[i];
            }
        
        stride /= 2;
        PCRSize *= 2;
    }
	
}

/**
   * Solving Ax = d using CRPCR and computes the error of the iteration
   * This function already uses local memory
   * This function works for sizes up to 2048
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param d Vector D
   * @param x Solution
   * @param system_size Size of the whole System
   * @param PCRSize Size of the PCR system
   * @param numSteps Number of iterations for CR
   * @param smallSteps Number of iterations for PCR
   */
__kernel void CRPCR_err(
	__global double *a, __local double *b, __global double *c, __local double *d, __global double *x, __global double *err, __global double *sol,
	int system_size, int PCRSize, int numSteps, int smallSteps
	){
    
	int idx = get_local_id(0);

	int pcrStride= 64;
	int stride = 1;
    int half_size = system_size >> 1;
	int thid_num = half_size;
	int i;
    for (int j = 0; j < numSteps; j++) {
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        stride *= 2;
        int delta = stride / 2;
        if (idx < thid_num) {
            int i = stride * idx + stride - 1;
            int iLeft = i - delta;
            if (iLeft < 0) iLeft = 0;
            int iRight = i + delta;
            if (iRight >= system_size) iRight = system_size - 1;

            double tmp1 = a[i] / b[iLeft];
    	    double tmp2 = c[i] / b[iRight];
            b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
            d[i] = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
            a[i] = -a[iLeft] * tmp1;
            c[i] = -c[iRight] * tmp2;
        }
        thid_num /= 2;
    }
	int deltaPCR = pcrStride;
    i = idx * pcrStride + pcrStride - 1;
	bool workingThread = idx < PCRSize;
    double aNew,bNew,cNew,dNew;
    for (int j = 0; j < smallSteps; j++) {
        if(workingThread){
            int iRight = i + deltaPCR;
               if (iRight >= system_size) iRight = system_size - 1;
               int iLeft = i - deltaPCR;
               if (iLeft < pcrStride-1) iLeft = pcrStride-1;
       
               double tmp1 = a[i] / b[iLeft];
               double tmp2 = c[i] / b[iRight];
               bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
               dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
               aNew = -a[iLeft] * tmp1;
               cNew = -c[iRight] * tmp2;
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if(workingThread){
            b[i] = bNew;
            d[i] = dNew;	
            a[i] = aNew;
            c[i] = cNew;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        deltaPCR *= 2;
	}
    if(workingThread){
           int addr1 = i;
           int addr2 = i + deltaPCR;
           double tmp3 = (b[addr2] * b[addr1] - c[addr1] * a[addr2]);
           x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
           x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
		   err[addr1] = (x[addr1]<sol[addr1])?sol[addr1]-x[addr1]:x[addr1]-sol[addr1];
		   err[addr2] = (x[addr2]<sol[addr2])?sol[addr2]-x[addr2]:x[addr2]-sol[addr2];
    }
	for (int j = 0; j < numSteps; j++) {
        int delta = stride / 2;
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
            if (idx < PCRSize) {
                int i = stride * idx + stride / 2 - 1;
                if (i == delta - 1) x[i] = (d[i] - c[i] * x[i + delta]) / b[i];
                else x[i] = (d[i] - a[i] * x[i - delta] - c[i] * x[i + delta]) / b[i];
				err[i] = (x[i]<sol[i])?sol[i]-x[i]:x[i]-sol[i];
            }
        
        stride /= 2;
        PCRSize *= 2;
    } 
	
}

/**
   * Solving Ax = z using PCR-Thomas, this is the Thomas algorithm phase.
   * This function already uses local memory
   * This function works for sizes up to 2048
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param z Vector Z
   * @param x Solution
   * @param initialPos Position of the first element
   * @param sizeSmallerSystem Number of elements to process
   * @param stride stride between each element
   */
__kernel 
void thomas(
	__global double*a,__local double*b,__global double*c,__local double*x,__global double*z,
	int initialPos,int sizeSmallerSystem,int stride
	){

    int systemSize = stride*sizeSmallerSystem;
    int i = initialPos;
    c[i] = c[i] / b[i];
    z[i] = z[i] / b[i];
    int startLocationSystem = stride + i;
    for (int i = startLocationSystem;i<systemSize;i += stride){
        double tmp = b[i]-a[i]*c[i-stride];
        c[i]  = c[i] / tmp;
        z[i]  = (z[i]-z[i-stride]*a[i]) / tmp;
    }
    int endLocationSystem = systemSize-stride + i;
    x[endLocationSystem] = z[endLocationSystem];
    for (int i = endLocationSystem-stride;i>= 0;i-= stride) x[i] = z[i]-c[i]*x[i + stride];
}

/**
   * Solving Ax = z using PCR-Thomas, this is the Thomas algorithm phase and computes the error of the iteration.
   * This function already uses local memory
   * This function works for sizes up to 2048
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param z Vector Z
   * @param x Solution
   * @param initialPos Position of the first element
   * @param sizeSmallerSystem Number of elements to process
   * @param stride stride between each element
   */
__kernel 
void thomas_err(
	__global double*a,__local double*b,__global double*c,__local double*x,__global double*z,__global double*err,__global double*sol,
	int initialPos,int sizeSmallerSystem,int stride
	){

    int systemSize = stride*sizeSmallerSystem;
    int i = initialPos;
    c[i] = c[i] / b[i];
    z[i] = z[i] / b[i];
    int startLocationSystem = stride + i;
    for (int i = startLocationSystem;i<systemSize;i += stride){
        double tmp = b[i]-a[i]*c[i-stride];
        c[i]  = c[i] / tmp;
        z[i]  = (z[i]-z[i-stride]*a[i]) / tmp;
    }
    int endLocationSystem = systemSize-stride + i;
    x[endLocationSystem] = z[endLocationSystem];
    for (int i = endLocationSystem-stride;i>= 0;i-= stride) {
		x[i] = z[i]-c[i]*x[i + stride];
		err[i] = (x[i]<sol[i])?sol[i]-x[i]:x[i]-sol[i];
	}
}

/**
   * Solving Ax = z using PCR-Thomas, this is the PCR phase.
   * This function already uses local memory
   * This function works for sizes up to 1024
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param sizeSystem Size of the whole system to solve
   * @param numSteps Number of iteration of the PCR
   * @param sizeSmallerSystem Number of elements for the Thomas algorithm
   */
__kernel 
void PCRTHOMAS(
	__global double * a, __local double * b, __global double * c, __local double * x, __global double * z, 
	int sizeSystem, int numSteps, int sizeSmallerSystem
	){
    int delta = 1;
    int i = get_local_id(0);
    double aNew, bNew, cNew, zNew;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        int iLeft = i - delta;
        if (iLeft < 0) iLeft = 0;
        double tmp1 = a[i] / b[iLeft];
        double tmp2 = c[i] / b[iRight];
        bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        aNew = -a[iLeft] * tmp1;
        cNew = -c[iRight] * tmp2;
        __syncthreads();
        b[i] = bNew;
        z[i] = zNew;
        a[i] = aNew;
        c[i] = cNew;
        __syncthreads();
        
        delta *= 2;
    }
    int stride = sizeSystem/sizeSmallerSystem;
    if (i < stride) {
        thomas(a,b,c,x,z,i,sizeSmallerSystem,stride);
    }
}
/**
   * Solving Ax = z using PCR-Thomas, this is the PCR phase and computes the error of the iteration.
   * This function already uses local memory
   * This function works for a size of 2048
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param sizeSystem Size of the whole system to solve
   * @param numSteps Number of iteration of the PCR
   * @param sizeSmallerSystem Number of elements for the Thomas algorithm
   */
__kernel 
void PCRTHOMAS2048_err(
	__global double * a, __local double * b, __global double * c, __local double * x, __global double * z,__global double * err,__global double *sol, 
	int sizeSystem, int numSteps, int sizeSmallerSystem
	){
    int delta = 1;
    int i = get_local_id(0);
    double aNew, bNew, cNew, zNew,aNew1, bNew1, cNew1, zNew1;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        int iLeft = i - delta;
        if (iLeft < 0) iLeft = 0;
        double tmp1 = a[i] / b[iLeft];
        double tmp2 = c[i] / b[iRight];
        bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        aNew = -a[iLeft] * tmp1;
        cNew = -c[iRight] * tmp2;
		int i1 = i+1024;
        iRight = i1 + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        iLeft = i1 - delta;
        if (iLeft < 0) iLeft = 0;
        tmp1 = a[i1] / b[iLeft];
        tmp2 = c[i1] / b[iRight];
        bNew1 = b[i1] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        zNew1 = z[i1] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        aNew1 = -a[iLeft] * tmp1;
        cNew1 = -c[iRight] * tmp2;
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
    int stride = sizeSystem/sizeSmallerSystem;
    if (i < stride) {
        thomas_err(a,b,c,x,z,err,sol,i,sizeSmallerSystem,stride);
    }
}
/**
   * Solving Ax = z using PCR-Thomas, this is the PCR phase.
   * This function already uses local memory
   * This function works for a size of 2048
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param sizeSystem Size of the whole system to solve
   * @param numSteps Number of iteration of the PCR
   * @param sizeSmallerSystem Number of elements for the Thomas algorithm
   */
__kernel 
void PCRTHOMAS2048(
	__global double * a, __local double * b, __global double * c, __local double * x, __global double * z, 
	int sizeSystem, int numSteps, int sizeSmallerSystem
	){
    int delta = 1;
    int i = get_local_id(0);
    double aNew, bNew, cNew, zNew,aNew1, bNew1, cNew1, zNew1;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        int iLeft = i - delta;
        if (iLeft < 0) iLeft = 0;
        double tmp1 = a[i] / b[iLeft];
        double tmp2 = c[i] / b[iRight];
        bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        aNew = -a[iLeft] * tmp1;
        cNew = -c[iRight] * tmp2;
		int i1 = i+1024;
        iRight = i1 + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        iLeft = i1 - delta;
        if (iLeft < 0) iLeft = 0;
        tmp1 = a[i1] / b[iLeft];
        tmp2 = c[i1] / b[iRight];
        bNew1 = b[i1] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        zNew1 = z[i1] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        aNew1 = -a[iLeft] * tmp1;
        cNew1 = -c[iRight] * tmp2;
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
    int stride = sizeSystem/sizeSmallerSystem;
    if (i < stride) {
        thomas(a,b,c,x,z,i,sizeSmallerSystem,stride);
    }
}

/**
   * Uses ADI to solve the problem.
   * This function solves the rows 
   */
__kernel void ADIROW(
    int nx,int ny,double alpha,double dyodx,double dxdy //Initial Variables
    ,__global  double *a,__global  double *devB,__global  double *c,__global  double *devD,__global  double *sol_rows,__global  double *sol_cols //Tridiagonal Arrays
    ,__global  double* sol,__global  double* errArray,__global  double *fArray, __global  int *code,int elementsPerThread, int iterations){ //Other info
	
    double alp2dyodx = alpha+2*dyodx, dxody = ((double)1)/dyodx;
	dyodx *=-1;
	double __local b[2048];
	double __local d[2048];
	int threadIdx =  get_local_id(0);
	int idx = get_group_id(0)+2; //idx
	int index = idx*nx;
    for(int e = 0; e < elementsPerThread; e++){	
		int j = 2+elementsPerThread*threadIdx+e;
		int sharedJ = elementsPerThread*threadIdx+e;
		int indJ = index+j;
		int ind = idx+j*ny;
		b[sharedJ] = (code[indJ]==2) ? alp2dyodx : 1;
		a[indJ] = (code[indJ]==2) ? dyodx : 0;
		c[indJ] = (code[indJ]==2) ? dyodx : 0;
		d[sharedJ]= (code[indJ]==2) ? 
			((sol_cols[ind-1]-sol_cols[ind]-sol_cols[ind]+sol_cols[ind+1])*dxody + fArray[indJ]*dxdy + alpha*sol_cols[ind]) :
			code[indJ]*sol_rows[index + j];
    }
	cyclic_branch_free_kernel_err(&a[index+2],b,&c[index+2],d,&sol_rows[index+2] ,&errArray[index+2],&sol[index+2],2048,10,1);
}

/**
   * Uses ADI to solve the problem.
   * This function solves the columns 
   */
__kernel void ADICOL(
    int nx,int ny,double alpha,double dxody,double dxdy //Initial Variables
    ,__global  double *a,__global  double *devB,__global  double *c,__global  double *devD,__global  double *sol_rows,__global  double *sol_cols //Tridiagonal Arrays
    ,__global  double* sol,__global  double *fArray, __global  int *code,int elementsPerThread, int iterations){ //Other info
	
	double dyodx = 1.0/dxody;
    double alp2dxody = alpha+2*dxody;
	dxody *=-1;
	double __local b[2048];
	double __local d[2048];
	int threadIdx =  get_local_id(0);
	int idx = get_group_id(0)+2; //idx
	int index = idx*ny;

    for(int e = 0; e < elementsPerThread; e++){	
		int j = 2+elementsPerThread*threadIdx+e;
		int sharedJ = elementsPerThread*threadIdx+e;
		int index2 = idx*ny+j;
		int ind = idx+j*nx;
		b[sharedJ] = (code[ind]==2) ? alp2dxody : 1;
		a[index2] = (code[ind]==2) ? dxody : 0;
		c[index2] = (code[ind]==2) ? dxody : 0;
		d[sharedJ]= (code[ind]==2) ? 
			((sol_rows[ind-1]-sol_rows[ind]-sol_rows[ind]+sol_rows[ind+1])*dyodx + fArray[ind]*dxdy + alpha*sol_rows[ind]) :
			code[ind]*sol_cols[index2];
    }
	cyclic_branch_free_kernel(&a[index+2],b,&c[index+2],d,&sol_cols[index+2],2048,10,1);
}
