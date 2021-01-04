#ifndef _IMPORTS
#define _IMPORTS

#include "../headers/imports.h"

#endif

void print_info(Grid *g, Domain *d, Cells *c);
real calcError(Grid *g, Domain *d, Cells *c);
real calcError_mean(Grid *g, Domain *d, Cells *c);