#ifndef _IMPORTS
#define _IMPORTS

#include "../headers/imports.h"

#endif

int main(int argc, char** argv)
{
	clock_t t;

	t = clock();
	//init structures
	printf("\nInit\n");
	Grid *g = (Grid*)malloc(sizeof(Grid));
	Domain *d = (Domain*)malloc(sizeof(Domain));
	AuxSolver *a = (AuxSolver*)malloc(sizeof(AuxSolver));
	Cells *c = (Cells*)malloc(sizeof(Cells));
	init(g,d,a,c);

	

	//solve cycle
	int it = 0;
	real err = 1e3, err_ant = 1e2, epsi = 3e-13;
	real mean = calcError_mean(g, d, c);
	printf("\nMean Error Start: %e\n\nStart Solving\n", mean);
	while((fabs(err-err_ant) > epsi) && (it<10000))
	{
		err_ant = err;
		calcSol_rows(g, d, c, a->a,a->b,a->c,a->known);
		calcSol_cols(g, d, c, a->a,a->b,a->c,a->known);
		err = calcError(g, d, c);
		it++;
		if(it%100==0) printf("\nIteration %d with error:%e with variation:%e\n", it, err, fabs(err-err_ant));
	}
	//printf("\nIteration %d with error:%e with variation:%e\n", it, err, fabs(err-err_ant));
	mean = calcError_mean(g, d, c);
	printf("\nEnd results:\n iterations:%d\n error:%e mean_error:%e\n", it, err, mean);
	printf("\nTime spent: %f seconds\n", (real)(clock()-t)/CLOCKS_PER_SEC);

	
	
	cleanAll(g,d,a,c);
	return 0;
}