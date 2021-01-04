#ifndef _IMPORTS
#define _IMPORTS

#include "../headers/imports.h"

#endif
//print info
void print_info(Grid *g, Domain *d, Cells *c)
{
	int nx = g->size_x, ny = g->size_y;
	//print info
	printf("\n");
	for(int j=0; j<ny; j++)
	{
		printf("\n");
		for(int i=0; i<nx; i++)
		{
			printf("%d,", c->code[i + j*nx]);
		}
	}
	printf("\n");
	printf("West and East Limits\n");
	for(int i=0; i<nx; i++)
	{
		printf("w:%d,e:%d\n", c->west[i], c->east[i]);				
	}
	printf("North and South Limits\n");
	for(int j=0; j<ny; j++)
	{
		printf("n:%d,s:%d\n", c->north[j], c->south[j]);	
	}
	printf("\n");
	printf("\nDomain limits: w:%d;e:%d;n:%d,s:%d\n", d->min_x, d->max_x, d->min_y, d->max_y);
}

//Evaluation of the error of the system
//Error is calculated for the rows and columns.
//
real calcError(Grid *g, Domain *d, Cells *c)
{
	int nx = g->size_x, ny = g->size_y;
	real res = 1e-6;
	real aux_r, aux_c;
	real min = 1e-6;
		
	int id, id2;
	int nThreads = 32;
	int linesPerThread = ny/nThreads;
	#pragma omp parallel for private(res)
	for(int t=0;t<nThreads;t++){
		for(int j=0; j<linesPerThread; j++){	
			for(int i=0; i<nx; i++){
				
				id = i + (linesPerThread*t+j)*nx;
				id2 = i*ny + (linesPerThread*t+j);
				if(c->code[id] == 2)
				{
					aux_r = fabs(c->sol_rows[id] - d->sol[id]);
					aux_c = fabs(c->sol_cols[id2] - d->sol[id]);
					if(res < aux_r) res = aux_r; 
					if(res < aux_c) res = aux_c; 
				}
			}
		}
		#pragma omp critical
		min = (min<res ? res : min);
	}
	return min;
}

//evaluation of the error of the system
//error is calculated for the rows and columns
real calcError_mean(Grid *g, Domain *d, Cells *c)
{
	int nx = g->size_x, ny = g->size_y;
	real res_rows = 1.0, res_cols = 1.0;
	real aux_r = 0.0, aux_c = 0.0;
 	int elements = 0, id, id2;
	for(int j=0; j<ny; j++)
	{	
		for(int i=0; i<nx; i++)
		{
			id = i + j*nx;
			id2 = i*ny + j;
			if(c->code[id] == 2)
			{
				aux_r += fabs(c->sol_rows[id] - d->sol[id]);
				aux_c += fabs(c->sol_cols[id2] - d->sol[id]);
				elements++;
			}
		}
	}
	//printf("\naux_r:%f :: aux_c:%f -> elems:%d\n", aux_r, aux_c, elements);
	res_rows = aux_r/(real)elements;
	res_cols = aux_c/(real)elements;
	return (res_rows > res_cols) ? res_rows : res_cols;
}