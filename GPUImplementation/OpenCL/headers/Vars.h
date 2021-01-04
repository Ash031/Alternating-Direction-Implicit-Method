#ifndef VARS__IMP
#define VARS__IMP



struct Domain
{
	double radius;
	double cx, cy;
	double* sol;
	double alpha;
	int min_x, min_y, max_x, max_y;
};


struct Grid
{
	double min_x, max_x;
	double min_y, max_y;
	int size_x, size_y;
};

struct Cells
{
	double* x, * y;
	double* xc, * yc;
	double* fArray, * fArrayc;
	double* sol_rows, * sol_cols; //exact value at the baricenter for rows and columns
	int* code;
	int* west, * east, * north, * south; //limits
};

inline int test_Domain(Domain* d, double x, double y);
inline double get_ExactValue(double x, double y);
inline double fonte(double x, double y);
void initGrid(Grid* g, int minx, int miny, int maxx, int maxy, int sizex, int sizey);
void initDomain(Domain* d, Grid* g, double r, double a);
void initCells(Grid* g, Domain* d, Cells* c);
void cleanGrid(Grid* g);
void cleanDomain(Domain* d);
void cleanCells(Cells* c);
void initGridAndDomain(Grid* g, Domain* d);
void init(Grid* g, Domain* d, Cells* c);
void cleanAll(Grid* g, Domain* d, Cells* c);


#endif