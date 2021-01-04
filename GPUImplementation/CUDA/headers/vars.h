#ifndef _IMPORTS
#define _IMPORTS

#include "imports.h"

#endif

struct Domain
{
	real radius;
	real cx, cy;
	real *sol;
	real alpha;
	int min_x, min_y, max_x, max_y;	
};


struct Grid
{
	real min_x, max_x;
	real min_y, max_y;
	int size_x, size_y;
};

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
void initCells(Grid *g, Domain *d, Cells *c);
void cleanGrid(Grid *g);
void cleanDomain(Domain *d);
void cleanCells(Cells *c);
void initGridAndDomain(Grid* g, Domain* d);
void init(Grid *g, Domain *d,Cells*c);
void cleanAll(Grid *g, Domain *d,Cells*c);