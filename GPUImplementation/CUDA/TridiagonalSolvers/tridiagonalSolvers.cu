#include "tridiagonalSolvers.h"


//Most function can auto-select the usable options for each algorithm depending the number of Max threads per block.
//NOTE: SOME FUNCTIONS ARE CODED TO ONLY WORK FOR GPU WITH MAXTHREADS OF 1024. WHEN CHANGING THIS VALUE THREAD CAREFULLY AND MAKE THE NECESSARY CHANGES (SPECIALLY FOR FUNCTIONS WITH NUMBER IN THEIR NAME, THAT USUALLY ARE HARDCODED FOR THIS VALUE). 
#define MAXTHREADSPERBLOCK 1024

//CR Functions

/**
   * First step of Cyclic Reduction (Forward Sweep) for Ax = z 
   * This function scales for sizes bigger than 2048. If the size is smaller
   * call the function without the parameter elementsPerThread
   * This function is a global function and should be called from the Host.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   * @param elementsPerThread Number of elements to work with in each thread
   */
__global__
void cyclicReductionFRDEV(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem, int elementsPerThread) {
    int stride = 1, idx = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        __syncthreads();
        stride *= 2;
        int delta = stride / 2;
        for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * idx;
            if (element < numThreads) {
                int i = stride * element + stride - 1;
                int iLeft = i - delta;
                if (iLeft < 0) iLeft = 0;
                int iRight = i + delta;
                if (iRight >= sizeSystem) iRight = sizeSystem - 1;

                real tmp1 = a[i] / b[iLeft];
                real tmp2 = c[i] / b[iRight];
                b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
                z[i] = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
                a[i] = -a[iLeft] * tmp1;
                c[i] = -c[iRight] * tmp2;
            }
        }
        numThreads /= 2;
    }
}

/**
   * Second step of Cyclic Reduction (Backwards substitution) for Ax = z 
   * This function scales for sizes bigger than 2048. If the size is smaller
   * call the function without the parameter elementsPerThread
   * This function is a global function and should be called from the Host.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   * @param elementsPerThread Number of elements to work with in each thread
   * @param stride The initial stride of the elements
   */
__global__
void cyclicReductionBSDEV(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem, int elementsPerThread, int stride) {
    int idx = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        int delta = stride / 2;
        __syncthreads();
        for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * idx;
            if (element < numThreads) {
                int element = e + elementsPerThread * idx;
                int i = stride * element + stride / 2 - 1;
                if (i == delta - 1) x[i] = (z[i] - c[i] * x[i + delta]) / b[i];
                else x[i] = (z[i] - a[i] * x[i - delta] - c[i] * x[i + delta]) / b[i];
            }
        }
        stride /= 2;
        numThreads *= 2;
    }
}

/**
   * First step of Cyclic Reduction (Forward Sweep) for Ax = z 
   * This function is a device function and should be called from another device function.
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   * @param elementsPerThread Number of elements to work with in each thread
   */
__device__
int cyclicReductionFR(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem, int elementsPerThread) {
    int stride = 1, idx = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        __syncthreads();
        stride *= 2;
        int delta = stride / 2;
        for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * idx;
            if (element < numThreads) {
                int i = stride * element + stride - 1;
                int iLeft = i - delta;
                if (iLeft < 0) iLeft = 0;
                int iRight = i + delta;
                if (iRight >= sizeSystem) iRight = sizeSystem - 1;

                real tmp1 = a[i] / b[iLeft];
                real tmp2 = c[i] / b[iRight];
                b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
                z[i] = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
                a[i] = -a[iLeft] * tmp1;
                c[i] = -c[iRight] * tmp2;
            }
        }
        numThreads /= 2;
    }
    return stride;
}

/**
   * Second step of Cyclic Reduction (Backwards substitution) for Ax = z 
   * This function is a device function and should be called from another device function.
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   * @param elementsPerThread Number of elements to work with in each thread
   * @param stride The initial stride of the elements
   */
__device__
void cyclicReductionBS(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem, int elementsPerThread, int stride) {
    int idx = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        int delta = stride / 2;
        __syncthreads();
        for (int e = 0; e < elementsPerThread; e++) {
            int element = e + elementsPerThread * idx;
            if (element < numThreads) {
                int i = stride * element + stride / 2 - 1;
                if (i == delta - 1) x[i] = (z[i] - c[i] * x[i + delta]) / b[i];
                else x[i] = (z[i] - a[i] * x[i - delta] - c[i] * x[i + delta]) / b[i];
            }
        }
        stride /= 2;
        numThreads *= 2;
    }
}
/**
   * First step of Cyclic Reduction (Forward Sweep) for Ax = z 
   * This function does not scale for sizes higher than 2048
   * This function is a device function and should be called from another device function.
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   */
__device__
int cyclicReductionFR(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem) {
    int stride = 1, idx = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        __syncthreads();
        stride *= 2;
        int delta = stride / 2;
        int element = idx;
        if (element < numThreads) {
            int i = stride * element + stride - 1;
            int iLeft = i - delta;
            if (iLeft < 0) iLeft = 0;
            int iRight = i + delta;
            if (iRight >= sizeSystem) iRight = sizeSystem - 1;

            real tmp1 = a[i] / b[iLeft];
            real tmp2 = c[i] / b[iRight];
            b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
            z[i] = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
            a[i] = -a[iLeft] * tmp1;
            c[i] = -c[iRight] * tmp2;
        }
        numThreads /= 2;
    }
    return stride;
}
/**
   * Second step of Cyclic Reduction (Backwards substitution) for Ax = z 
   * This function does not scale for sizes higher than 2048
   * This function is a device function and should be called from another device function.
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   * @param stride The initial stride of the elements
   */
__device__
void cyclicReductionBS(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem, int stride) {
    int idx = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        int delta = stride / 2;
        __syncthreads();
        if (idx < numThreads) {
            int i = stride * idx + stride / 2 - 1;
            if (i == delta - 1) x[i] = (z[i] - c[i] * x[i + delta]) / b[i];
            else x[i] = (z[i] - a[i] * x[i - delta] - c[i] * x[i + delta]) / b[i];
        }
        stride /= 2;
        numThreads *= 2;
    }
}

/**
   * Calculates vector x in Ax=z using the Cyclic Reduction 
   * This function does not scale for sizes higher than 2048, for it to scale add the parameter elementsPerThread
   * This function is a global function and should be called from the Host.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   */
__global__
void cyclicReduction(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem) {
    int stride = cyclicReductionFR(a, b, c, x, z, numSteps, numThreads, sizeSystem);
    if (threadIdx.x < 1) {
        int addr1 = stride - 1;
        int addr2 = 2 * stride - 1;
        real tmp = b[addr2] * b[addr1] - c[addr1] * a[addr2];
        x[addr1] = (b[addr2] * z[addr1] - c[addr1] * z[addr2]) / tmp;
        x[addr2] = (z[addr2] * b[addr1] - z[addr1] * a[addr2]) / tmp;
    }
    cyclicReductionBS(a, b, c, x, z, numSteps, 2, sizeSystem, stride);
}

/**
   * Calculates vector x in Ax=z using the Cyclic Reduction 
   * This function scales for sizes higher than 2048
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   * @param elementsPerThread Number of elements to work with in each thread
   */
__global__
void cyclicReduction(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem, int elementsPerThread) {
    int stride = cyclicReductionFR(a, b, c, x, z, numSteps, numThreads, sizeSystem, elementsPerThread);
    if (threadIdx.x < 1) {
        int addr1 = stride - 1;
        int addr2 = 2 * stride - 1;
        real tmp = b[addr2] * b[addr1] - c[addr1] * a[addr2];
        x[addr1] = (b[addr2] * z[addr1] - c[addr1] * z[addr2]) / tmp;
        x[addr2] = (z[addr2] * b[addr1] - z[addr1] * a[addr2]) / tmp;
    }
    cyclicReductionBS(a, b, c, x, z, numSteps, 2, sizeSystem, elementsPerThread, stride);
}

/**
   * Calculates vector x in Ax=z using the Cyclic Reduction 
   * This function scales for sizes higher than 2048, this function 
   * automatically chooses the best match for your size and runs 
   * the Cyclic Reduction for every system
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param N Size of each System
   * @param TOTALSYSTEMS Number of INDEPENDANT systems
   */
void crKernel(real * a, real * b, real * c, real * x, real * z, int N, int TOTALSYSTEMS) {
    int numSteps = (int) log2((float) N) - 1;
    if (N <= 2048) {
        for (int i = 0; i < TOTALSYSTEMS; i++)
            cyclicReduction << < 1, N / 2 >>> ( &
                a[N * i], & b[N * i], & c[N * i], & x[N * i], & z[N * i],
                numSteps, N / 2, N
            );
        return;
    }
    int elementsPerThread = ceil(N / (2 * MAXTHREADSPERBLOCK));

    for (int i = 0; i < TOTALSYSTEMS; i++)
        cyclicReduction << < 1, 1024 >>> ( &
            a[N * i], & b[N * i], & c[N * i], & x[N * i], & z[N * i],
            numSteps, N / 2, N, elementsPerThread
        );
}


//PCR Functions
/**
   * Calculates vector x in Ax=z using the Parallel Cyclic Reduction 
   * This function does not scale for sizes higher than 1024 
   * This function is a global function and should be called from the Host.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param sizeSystem Size of the whole System
   */
__global__
void parallelCyclicReduction(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem) {
    int i = threadIdx.x;
    int delta = 1;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        int iLeft = i - delta;
        if (iLeft < 0) iLeft = 0;

        real tmp1 = a[i] / b[iLeft];
        real tmp2 = c[i] / b[iRight];
        real bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        real zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        real aNew = -a[iLeft] * tmp1;
        real cNew = -c[iRight] * tmp2;

        __syncthreads();
        b[i] = bNew;
        z[i] = zNew;
        a[i] = aNew;
        c[i] = cNew;
        __syncthreads();
        delta *= 2;
    }
    if (i < delta) {
        int addr1 = i;
        int addr2 = i + delta;
        real tmp3 = (b[addr2] * b[addr1] - c[addr1] * a[addr2]);
        x[addr1] = (b[addr2] * z[addr1] - c[addr1] * z[addr2]) / tmp3;
        x[addr2] = (z[addr2] * b[addr1] - z[addr1] * a[addr2]) / tmp3;
    }
}

/**
   * Calculates vector x in Ax=z using the Parallel Cyclic Reduction 
   * This function does not work for sizes different than 2048 
   * This function is a global function and should be called from the Host.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param sizeSystem Size of the whole System
   */
__global__ 
void parallelCyclicReduction2048( real *a,  real *b,  real *c, real *x,real *z, int numSteps, int sizeSystem) {
    int i = threadIdx.x;
    int delta = 1;
    int i1=i+1024;

    __syncthreads();

    // parallel cyclic reduction
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
    real tmp3 = 1/(b[i1] * b[i] - c[i] * a[i1]);
    x[i] = (b[i1] * z[i] - c[i] * z[i1]) * tmp3;
    x[i1] = (z[i1] * b[i] - z[i] * a[i1]) * tmp3;
}

/**
   * Calculates vector x in Ax=z using the Parallel Cyclic Reduction 
   * This function does not scale for sizes higher than 2048 
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param sizeSystem Size of the whole System
   */
__device__
void parCycRed(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem) {
    int delta = 1;
    int i = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        int iLeft = i - delta;
        if (iLeft < 0) iLeft = 0;

        real tmp1 = a[i] / b[iLeft];
        real tmp2 = c[i] / b[iRight];
        real bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        real zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        real aNew = -a[iLeft] * tmp1;
        real cNew = -c[iRight] * tmp2;

        __syncthreads();
        b[i] = bNew;
        z[i] = zNew;
        a[i] = aNew;
        c[i] = cNew;
        __syncthreads();
        delta *= 2;
    }
    if (i < delta) {
        int addr1 = i;
        int addr2 = i + delta;
        real tmp3 = (b[addr2] * b[addr1] - c[addr1] * a[addr2]);
        x[addr1] = (b[addr2] * z[addr1] - c[addr1] * z[addr2]) / tmp3;
        x[addr2] = (z[addr2] * b[addr1] - z[addr1] * a[addr2]) / tmp3;
    }
}
/**
   * Calculates vector x in Ax=z using the Parallel Cyclic Reduction 
   * This function scales for sizes higher than 2048, but this function needs to 
   * allocate memory. Not very effective so should avoid usage.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param sizeSystem Size of the whole System
   * @param elementsPerThread Number of elements to work with in each thread
   */
__global__
void parallelCyclicReduction(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem, int elementsPerThread) {
    int delta = 1;
    int idx = threadIdx.x, i;
    real* bNew = (real*)malloc(sizeof(real)*sizeSystem);
    real* zNew = (real*)malloc(sizeof(real)*sizeSystem);
    real* aNew = (real*)malloc(sizeof(real)*sizeSystem);
    real* cNew = (real*)malloc(sizeof(real)*sizeSystem);
    for (int j = 0; j < numSteps; j++) {
        for (int e = 0; e < elementsPerThread; e++) {
            i = idx * elementsPerThread + e;
            int iRight = i + delta;
            if (iRight >= sizeSystem) iRight = sizeSystem - 1;
            int iLeft = i - delta;
            if (iLeft < 0) iLeft = 0;

            real tmp1 = a[i] / b[iLeft];
            real tmp2 = c[i] / b[iRight];
            bNew[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
            zNew[i] = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
            aNew[i] = -a[iLeft] * tmp1;
            cNew[i] = -c[iRight] * tmp2;

            __syncthreads();
        }
        for (int e = 0; e < elementsPerThread; e++) {
            i = idx * elementsPerThread + e;
            b[i] = bNew[i];
            z[i] = zNew[i];
            a[i] = aNew[i];
            c[i] = cNew[i];
            __syncthreads();
        }
        delta *= 2;
    }
    for (int e = 0; e < elementsPerThread; e++) {
        i = idx * elementsPerThread + e;
        if (i < delta) {
            int addr1 = i;
            int addr2 = i + delta;
            real tmp3 = (b[addr2] * b[addr1] - c[addr1] * a[addr2]);
            x[addr1] = (b[addr2] * z[addr1] - c[addr1] * z[addr2]) / tmp3;
            x[addr2] = (z[addr2] * b[addr1] - z[addr1] * a[addr2]) / tmp3;
        }
    }
}


//CRPCR Functions


/**
   * This function does not scale for sizes higher than 2048 , and is used for CR-PCR, 
   * this function should be used when the smaller system size is the same as the number of threads allocated.
   * This function is a global function and should be called from the Host.
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param sizeSystem Size of the whole System
   * @param stride The initial stride of the elements
   */
   __global__
   void parCycRedDEV(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem, int stride) {
       int delta = stride;
       int i = threadIdx.x * stride + stride - 1;
       for (int j = 0; j < numSteps; j++) {
           int iRight = i + delta;
           if (iRight >= sizeSystem) iRight = sizeSystem - 1;
           int iLeft = i - delta;
           if (iLeft < stride-1) iLeft = stride-1;
   
           real tmp1 = a[i] / b[iLeft];
           real tmp2 = c[i] / b[iRight];
           real bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
           real zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
           real aNew = -a[iLeft] * tmp1;
           real cNew = -c[iRight] * tmp2;
   
           __syncthreads();
           b[i] = bNew;
           z[i] = zNew;
           a[i] = aNew;
           c[i] = cNew;
           __syncthreads();
           delta *= 2;
       }
       int addr1 = i;
       int addr2 = i + delta;
       real tmp3 = (b[addr2] * b[addr1] - c[addr1] * a[addr2]);
       x[addr1] = (b[addr2] * z[addr1] - c[addr1] * z[addr2]) / tmp3;
       x[addr2] = (z[addr2] * b[addr1] - z[addr1] * a[addr2]) / tmp3;
   }
   /** 
   * This function does not scale for sizes higher than 2048 , and is used for CR-PCR, 
   * this function should be used when the smaller system size is the same as the number of threads.
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param sizeSystem Size of the whole System
   * @param stride The initial stride of the elements
   * @param workingThread This variables is assigned to each thread to make sure if they can work or not
   */
   __device__
   void parCycRed(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem, int stride,bool workingThread) {
       int delta = stride;
       int i = threadIdx.x * stride + stride - 1;
       real aNew,bNew,cNew,zNew;
       for (int j = 0; j < numSteps; j++) {
           if(workingThread){
               int iRight = i + delta;
               if (iRight >= sizeSystem) iRight = sizeSystem - 1;
               int iLeft = i - delta;
               if (iLeft < stride-1) iLeft = stride-1;
       
               real tmp1 = a[i] / b[iLeft];
               real tmp2 = c[i] / b[iRight];
               bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
               zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
               aNew = -a[iLeft] * tmp1;
               cNew = -c[iRight] * tmp2;
           }
   
           __syncthreads();
           if(workingThread){
               b[i] = bNew;
               z[i] = zNew;
               a[i] = aNew;
               c[i] = cNew;
           }
           __syncthreads();
           delta *= 2;
       }
       if(workingThread){
           int addr1 = i;
           int addr2 = i + delta;
           real tmp3 = (b[addr2] * b[addr1] - c[addr1] * a[addr2]);
           x[addr1] = (b[addr2] * z[addr1] - c[addr1] * z[addr2]) / tmp3;
           x[addr2] = (z[addr2] * b[addr1] - z[addr1] * a[addr2]) / tmp3;
       }
   }

/**
   * Calculates vector x in Ax=z using CR-PCR
   * The number of PCR steps is not dynamically changed.
   * This function does not scale for sizes higher than 2048
   * All memory should be allocated on the device!
   * Currently only works with PCR sizes up to 32
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   * @param elementsPerThread Number of elements to work with in each thread
   * @param PCRSize Number of elements to work with in PCR
   */
__global__
void CRPCRGlobal(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem, int elementsPerThread,int PCRsize) {
    int pcrSteps = (int) log2((float) PCRsize) - 1, crSteps = numSteps - pcrSteps;
    int pcrStride = sizeSystem / PCRsize;
    int stride = cyclicReductionFR(a, b, c, x, z, crSteps, numThreads, sizeSystem, elementsPerThread);
    parCycRed(a, b, c, x, z, pcrSteps, sizeSystem, pcrStride,threadIdx.x<PCRsize);
    cyclicReductionBS(a, b, c, x, z, crSteps, PCRsize, sizeSystem, elementsPerThread, stride);
}

/**
   * Calculates vector x in Ax=z using CR-PCR
   * The number of PCR steps can be dynamically changed.
   * This function does not scale for sizes higher than 2048
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param sizeSystem Size of the whole System
   * @param elementsPerThread Number of elements to work with in each thread
   */
void CRPCR(real * a, real * b, real * c, real * x, real * z, int numSteps, int numThreads, int sizeSystem, int elementsPerThread, int PCRsize){ 
    int pcrSteps = (int) log2((float) PCRsize) - 1, crSteps = numSteps - pcrSteps;
    int pcrStride = sizeSystem / PCRsize;
    cyclicReductionFRDEV<<<1,MAXTHREADSPERBLOCK>>>(a, b, c, x, z, crSteps, numThreads, sizeSystem, elementsPerThread);
    parCycRedDEV<<<1,PCRsize>>>(a, b, c, x, z, pcrSteps, sizeSystem, pcrStride);
    cyclicReductionBSDEV<<<1,MAXTHREADSPERBLOCK>>>(a, b, c, x, z, crSteps, PCRsize, sizeSystem, elementsPerThread, pcrStride);
}

/**
   * Calculates vector x in Ax=z using CR-PCR
   * This function does not scale for sizes higher than 2048
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param N Size of each System
   * @param TOTALSYSTEMS Number of INDEPENDANT systems
   */
void crpcrKernel(real * a, real * b, real * c, real * x, real * z, int N, int TOTALSYSTEMS) {
    int elementsPerThread = ceil(N / (2 * MAXTHREADSPERBLOCK));
    int numSteps = (int) log2((float) N) - 1;
    for (int i = 0; i < TOTALSYSTEMS; i++)
    CRPCRGlobal<< < 1, 1024 >>> ( &
            a[N * i], & b[N * i], & c[N * i], & x[N * i], & z[N * i],
            numSteps, N / 2, N, elementsPerThread,512
        );
}

//This function MUST take the values already copied to device.
void crpcrKernel(real * a, real * b, real * c, real * x, real * z, int N, int TOTALSYSTEMS,int smallerSystemSize) {
    int elementsPerThread = ceil(N / (2 * MAXTHREADSPERBLOCK));
    int numSteps = (int) log2((float) N) - 1;
    for (int i = 0; i < TOTALSYSTEMS; i++)
    CRPCRGlobal<<<1,N/2>>>( 
            & a[N * i], & b[N * i], & c[N * i], & x[N * i], & z[N * i],
            numSteps, N/2 , N , 2,smallerSystemSize
        );
}
//PCRTHOMAS functions
/**
   * Calculates vector x in Ax=z using Thomas algorithm in conjunction with PCR
   * This function only works for a single system inside PCR various systems
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param sizeSmallerSystem Size of each System
   * @param stride stride of each element
   */
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

/**
   * Calculates vector x in Ax=z using PCR-Thomas
   * Usable with power of 2 sizes up to 1024
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of PCR iterations
   * @param sizeSystem Size of the whole system
   * @param sizeSmallerSystem Size of Thomas algorithm System
   */
__global__
void PCRTHOMAS(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem,int sizeSmallerSystem) {
    int delta = 1;
    int i = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        int iLeft = i - delta;
        if (iLeft < 0) iLeft = 0;

        real tmp1 = a[i] / b[iLeft];
        real tmp2 = c[i] / b[iRight];
        real bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        real zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        real aNew = -a[iLeft] * tmp1;
        real cNew = -c[iRight] * tmp2;

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
        thomas(a,b,c,x,z,sizeSmallerSystem,stride);
    }
}
/**
   * Calculates vector x in Ax=z using PCR-Thomas
   * Usable with size of 2048 with a few tweaks to avoid allocating memory (using 2 variables per array)
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of PCR iterations
   * @param sizeSystem Size of the whole system
   * @param sizeSmallerSystem Size of Thomas algorithm System
   */
__global__
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
/**
   * Calculates vector x in Ax=z using PCR-Thomas
   * Usable with size of 1024, callable from the host only
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of PCR iterations
   * @param sizeSystem Size of the whole system
   * @param sizeSmallerSystem Size of Thomas algorithm System
   */
__global__
void PCRTHOMAS1024global(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem,int sizeSmallerSystem) {
    int delta = 1;
    int i = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        int iLeft = i - delta;
        if (iLeft < 0) iLeft = 0;

        real tmp1 = a[i] / b[iLeft];
        real tmp2 = c[i] / b[iRight];
        real bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        real zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        real aNew = -a[iLeft] * tmp1;
        real cNew = -c[iRight] * tmp2;

        __syncthreads();
        b[i] = bNew;
        z[i] = zNew;
        a[i] = aNew;
        c[i] = cNew;
        __syncthreads();
        delta *= 2;
    }
    int thomasstride = sizeSystem/sizeSmallerSystem;
    if (i < thomasstride) {
        thomas(a,b,c,x,z,sizeSmallerSystem,thomasstride);
    }
}
/**
   * Calculates vector x in Ax=z using PCR-Thomas
   * Usable with size of 1024, callable from the device only
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of PCR iterations
   * @param sizeSystem Size of the whole system
   * @param sizeSmallerSystem Size of Thomas algorithm System
   */
__device__
void PCRTHOMAS1024(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem,int sizeSmallerSystem) {
    int delta = 1;
    int i = threadIdx.x;
    for (int j = 0; j < numSteps; j++) {
        int iRight = i + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
        int iLeft = i - delta;
        if (iLeft < 0) iLeft = 0;

        real tmp1 = a[i] / b[iLeft];
        real tmp2 = c[i] / b[iRight];
        real bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        real zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
        real aNew = -a[iLeft] * tmp1;
        real cNew = -c[iRight] * tmp2;

        __syncthreads();
        b[i] = bNew;
        z[i] = zNew;
        a[i] = aNew;
        c[i] = cNew;
        __syncthreads();
        delta *= 2;
    }
    int thomasstride = sizeSystem/sizeSmallerSystem;
    if (i < thomasstride) {
        thomas(a,b,c,x,z,sizeSmallerSystem,thomasstride);
    }
}

/**
   * Calculates vector x in Ax=z using CR-PCR-Thomas
   * This is the PCR-Thomas function coded for the PCR size of 512 and Thomas size of 4 
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   */
__global__
void PCRTHOMAS(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem, int elementsPerThread,int sizeSmallerSystem) {
    int delta = 1;
    int idx = threadIdx.x, i;
    real bNew[2048];
    real zNew[2048];
    real aNew[2048];
    real cNew[2048];
    for (int j = 0; j < numSteps; j++) {
        for (int e = 0; e < elementsPerThread; e++) {
            i = idx * elementsPerThread + e;
            int iRight = i + delta;
            if (iRight >= sizeSystem) iRight = sizeSystem - 1;
            int iLeft = i - delta;
            if (iLeft < 0) iLeft = 0;

            real tmp1 = a[i] / b[iLeft];
            real tmp2 = c[i] / b[iRight];
            bNew[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
            zNew[i] = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
            aNew[i] = -a[iLeft] * tmp1;
            cNew[i] = -c[iRight] * tmp2;

            __syncthreads();
        }
        for (int e = 0; e < elementsPerThread; e++) {
            i = idx * elementsPerThread + e;
            b[i] = bNew[i];
            z[i] = zNew[i];
            a[i] = aNew[i];
            c[i] = cNew[i];
            __syncthreads();
        }
        delta *= 2;
    }
    int stride = sizeSystem/sizeSmallerSystem;
    if (threadIdx.x < stride) {
        thomas(a,b,c,x,z,sizeSmallerSystem,stride);
    }
}

//CRPCRTHOMAS functions

/**
   * Calculates vector x in Ax=z using Thomas algorithm in conjunction with CRPCR
   * This function only works for a single system inside CRPCR various systems
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param initialPos Position of designated initial element
   * @param sizeSmallerSystem Size of each System
   * @param stride stride of each element
   */
__device__
void thomas(real*a,real*b,real*c,real*x,real*z,int initialPos,int sizeSmallerSystem,int stride){
    int systemSize = stride*sizeSmallerSystem;
    int i = initialPos;
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
/**
   * Calculates vector x in Ax=z using CR-PCR-Thomas
   * This is the PCR-Thomas function coded for the PCR size of 512 and Thomas size of 4 
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param initialPos Position of designated initial element
   * @param sizeSmallerSystem Size of PCR System
   * @param stride stride of each element
   * @param workingThread boolean that assigns if each thread is working or not
   */
__device__
void parCycRedThomas(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem,int sizeSmallerSystem, int stride,bool workingThread) {
    int delta = stride;
    int thomasSteps = 1;
    int PCRSteps = numSteps-1-thomasSteps;
    int i = threadIdx.x * stride + stride - 1;
    real aNew,bNew,cNew,zNew;
    for (int j = 0; j < PCRSteps; j++) {
        if(workingThread){
            int iRight = i + delta;
            if (iRight >= sizeSystem) iRight = sizeSystem - 1;
            int iLeft = i - delta;
            if (iLeft < stride-1) iLeft = stride-1;
    
            real tmp1 = a[i] / b[iLeft];
            real tmp2 = c[i] / b[iRight];
            bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
            zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
            aNew = -a[iLeft] * tmp1;
            cNew = -c[iRight] * tmp2;
        }

        __syncthreads();
        if(workingThread){
            b[i] = bNew;
            z[i] = zNew;
            a[i] = aNew;
            c[i] = cNew;
        }
        __syncthreads();
        delta *= 2;
    }
    if (threadIdx.x < 128) {
        thomas(a,b,c,x,z,i,4,512);
    }
}
/**
   * Calculates vector x in Ax=z using CR-PCR-Thomas
   * This is the main algorithm function 
   * All memory should be allocated on the device!
   * @param a Lower Diagonal
   * @param b Main Diagonal
   * @param c Upper Diagonal
   * @param x Solution
   * @param z Vector Z
   * @param numSteps Number of Steps for the algorithm (log2(sizeSystem)-1)
   * @param sizeSystem Size of the whole System
   * @param numThreads Number of Threads to work with (sizeSystem/2)
   * @param elementsPerThread Number of elements to work with in each thread
   */
__global__
void CRPCRTHOMAS(real * a, real * b, real * c, real * x, real * z, int numSteps, int sizeSystem,int numThreads, int elementsPerThread,int sizeSmallerSystem) {
    int PCRsize = 512;
    int pcrSteps = (int) log2((float) PCRsize) - 1, crSteps = numSteps - pcrSteps;
    int pcrStride = sizeSystem / PCRsize;
    int stride = cyclicReductionFR(a, b, c, x, z, crSteps, numThreads, sizeSystem, elementsPerThread);
    parCycRedThomas(a, b, c, x, z, pcrSteps, sizeSystem, sizeSmallerSystem,pcrStride,threadIdx.x<PCRsize);
    cyclicReductionBS(a, b, c, x, z, crSteps, PCRsize, sizeSystem, elementsPerThread, stride);
}

//API Functions (WRAPPERS)
//This function MUST take the values already copied to device.
void pcrKernel(real * a, real * b, real * c, real * x, real * z, int N, int TOTALSYSTEMS) {
    int numSteps = (int) log2((float) N) - 1;
    if(N==2*MAXTHREADSPERBLOCK){
        for (int i = 0; i < TOTALSYSTEMS; i++)
        parallelCyclicReduction2048 << < 1, MAXTHREADSPERBLOCK >>> ( &
            a[i * N], & b[i * N], & c[i * N], & x[i * N], & z[i * N], numSteps, N
        );
        return;
    }
    if (N <= MAXTHREADSPERBLOCK) {
        for (int i = 0; i < TOTALSYSTEMS; i++)
            parallelCyclicReduction << < 1, N >>> ( &
                a[i * N], & b[i * N], & c[i * N], & x[i * N], & z[i * N], numSteps, N
            );
        return;
    }
    int elementsPerThread = ceil(N / MAXTHREADSPERBLOCK);
    for (int i = 0; i < TOTALSYSTEMS; i++)
        parallelCyclicReduction << < 1, 1024 >>> ( &
            a[i * N], & b[i * N], & c[i * N], & x[i * N], & z[i * N], numSteps, N, elementsPerThread
        );
}


//This function MUST take the values already copied to device.
void pcrThomasKernel(real * a, real * b, real * c, real * x, real * z, int N, int TOTALSYSTEMS,int smallerSystemSize) {
    int numSteps = (int) log2((float) N/smallerSystemSize)-1;
    if (N == 2*MAXTHREADSPERBLOCK) {
        for (int i = 0; i < TOTALSYSTEMS; i++)
            PCRTHOMAS2048global<<<1,MAXTHREADSPERBLOCK>>>(&a[i * N], & b[i * N], & c[i * N], & x[i * N], & z[i * N],numSteps,N,smallerSystemSize);
        return;
    }if (N <= MAXTHREADSPERBLOCK) {
        for (int i = 0; i < TOTALSYSTEMS; i++)
            PCRTHOMAS1024global<<<1,N>>>(&a[i * N], & b[i * N], & c[i * N], & x[i * N], & z[i * N],numSteps,N,smallerSystemSize);
        return;
    }
    int elementsPerThread = ceil(N / MAXTHREADSPERBLOCK);
    for (int i = 0; i < TOTALSYSTEMS; i++)
        PCRTHOMAS << < 1, 1024>>> ( &
            a[i * N], & b[i * N], & c[i * N], & x[i * N], & z[i * N], numSteps, N, elementsPerThread, smallerSystemSize
        );
}
//This function MUST take the values already copied to device.
void crpcrThomasKernel(real * a, real * b, real * c, real * x, real * z, int N, int TOTALSYSTEMS,int smallerSystemSize) {
    int numSteps = (int) log2((float) N)-1;
    int elementsPerThread = ceil(N / (2*MAXTHREADSPERBLOCK));
    for (int i = 0; i < TOTALSYSTEMS; i++)
        CRPCRTHOMAS << < 1, 1024 >>> ( &
            a[i * N], & b[i * N], & c[i * N], & x[i * N], & z[i * N], numSteps, N, N/2,elementsPerThread, smallerSystemSize//int numSteps, int sizeSystem,int numThreads, int elementsPerThread,int sizeSmallerSystem
        );
}