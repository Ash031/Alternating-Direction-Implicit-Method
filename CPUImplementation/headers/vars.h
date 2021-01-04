#ifndef _IMPORTS
#define _IMPORTS

#include "imports.h"

#endif

//Problem's Domain information
struct Domain
{
	real radius;
	real cx, cy;
	real *sol;
	real alpha;
	int min_x, min_y, max_x, max_y;	
};

//Auxiliary Struct to store temporary data.
struct AuxSolver
{	
	real *known;
	real *a __attribute__((aligned(64))), *b __attribute__((aligned(64))), *c __attribute__((aligned(64)));
};

//Problem's Grid information
struct Grid
{
	real min_x, max_x;
	real min_y, max_y;
	int size_x, size_y;
};

//Problem's Cells information (Each cell code, value,...) and some extra information (Padding information)
struct Cells
{
	real *x, *y;
	real *xc, *yc;
	real *fArray,*fArrayc;
	real *sol_rows, *sol_cols; //exact value at the baricenter for rows and columns
	int *code;
	int *west, *east, *north, *south; //limits
};

inline int test_Domain(Domain *d, real x, real y);
inline real get_ExactValue(real x, real y);
inline real fonte(real x, real y);
void initGrid(Grid *g, int minx, int miny, int maxx, int maxy, int sizex, int sizey);
void initDomain(Domain *d, Grid *g, real r, real a);
void initAux(Domain *d, Grid *g, AuxSolver *a);
void initCells(Grid *g, Domain *d, Cells *c);
void cleanGrid(Grid *g);
void cleanDomain(Domain *d);
void cleanAux(AuxSolver *a);
void cleanCells(Cells *c);
void zeroAux(Grid *g, AuxSolver *a);
void initGridAndDomain(Grid* g, Domain* d);
void init(Grid *g, Domain *d,AuxSolver *a,Cells*c);
void cleanAll(Grid *g, Domain *d,AuxSolver *a,Cells*c);