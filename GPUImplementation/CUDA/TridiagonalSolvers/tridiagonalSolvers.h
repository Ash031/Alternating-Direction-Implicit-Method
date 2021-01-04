#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <mm_malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define real double
using namespace std;

void crKernel(real* a,real* b,real* c,real* x,real* z,int N,int TOTALSYSTEMS);
void pcrKernel(real* a,real* b,real* c, real* x, real* z,int N,int TOTALSYSTEMS);
void crpcrKernel(real* a,real* b,real* c, real* x, real* z,int N,int TOTALSYSTEMS);
void crpcrKernel(real* a,real* b,real* c, real* x, real* z,int N,int TOTALSYSTEMS,int sizeSmallerSystem);
void pcrThomasKernel(real * a, real * b, real * c, real * x, real * z, int N, int TOTALSYSTEMS,int smallerSystemSize); 
void crpcrThomasKernel(real * a, real * b, real * c, real * x, real * z, int N, int TOTALSYSTEMS,int smallerSystemSize);
