#ifndef _IMPORTS
#define _IMPORTS

#include "../headers/imports.h"

#endif


int main(int argc, char** argv)
{
	clock_t t;

	printf("\nInit\n");
	Grid *g = (Grid*)malloc(sizeof(Grid));
	Domain *d = (Domain*)malloc(sizeof(Domain));
	Cells *c = (Cells*)malloc(sizeof(Cells));

	init(g,d,c);


	solveADI(d,g,c);


	cleanAll(g,d,c);
	return 0;
}