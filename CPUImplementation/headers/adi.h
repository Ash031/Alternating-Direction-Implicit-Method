#ifndef _IMPORTS
#define _IMPORTS

#include "imports.h"

#endif

#ifndef _IMPORTS
#define _IMPORTS

#include "../headers/imports.h"

#endif

inline void solve_thomasR(Grid *g, Cells *c, real * __restrict__ aa,real * __restrict__ ab,real * __restrict__ ac,real * __restrict__ aknown, Domain *d, int i);
inline void solve_thomasC(Grid *g, Cells *c, real * __restrict__ aa,real * __restrict__ ab,real * __restrict__ ac,real * __restrict__ aknown, Domain *d, int i);
void calcSol_rows(Grid *g, Domain *d, Cells *c, real * __restrict__ aa,real * __restrict__ ab,real * __restrict__ ac,real * __restrict__ aknown);
void calcSol_cols(Grid *g, Domain *d, Cells *c, real *__restrict__ aa,real *__restrict__ ab,real *__restrict__ ac,real *__restrict__ aknown);