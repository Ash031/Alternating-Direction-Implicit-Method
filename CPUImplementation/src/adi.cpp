#ifndef _IMPORTS
#define _IMPORTS

#include "../headers/imports.h"

#endif

//solve tridiagonal system with thomas method for rows
//receives information about the cells
//the triagonal matrix [b,a,c] -> (upper, mid, lower) diagonals
// i - index of the row
inline void solve_thomasR(Grid *g, Cells *c, real * __restrict__ aa,real * __restrict__ ab,real * __restrict__ ac,real * __restrict__ aknown, Domain *d, int i){
	int nx = g->size_x;	
	int index = i*nx;
	int begin = c->west[i], end = c->east[i];


	for(int k = begin + 1; k <= end; k++){
		aa[k] = aa[k]*aa[k-1] - ac[k]*ab[k-1];
		ab[k] *= aa[k-1];
		aknown[k] = aknown[k]*aa[k-1] -ac[k]*aknown[k - 1];
		ac[k] = 0.0;
	}
	c->sol_rows[index + end] = aknown[end] / aa[end];
	int k = end-1;
	real *p = (real*)malloc(sizeof(real)*nx);
	real *q = (real*)malloc(sizeof(real)*nx);
	for(int i = 0,k = end-1; k > begin+2;i+=4,k-=4){
		p[i] = aa[k];
		p[i+1] = p[i]*aa[k-1];
		p[i+2] = p[i+1]*aa[k-2];
		p[i+3] = p[i+2]*aa[k-3];
		q[i] = aknown[k] - ab[k]*c->sol_rows[index + k + 1];
		q[i+1] = p[i]*aknown[k-1] - ab[k-1]*q[i];
		q[i+2] = p[i+1]*aknown[k-2] - ab[k-2]*q[i+1];
		q[i+3] = p[i+2]*aknown[k-3] - ab[k-3]*q[i+2];
	}	
	for(int i = 0,k = end-1; k > begin+2;i++,k--){
		c->sol_rows[index + k] = q[i]/p[i];
	}
	for(int kk = k; kk >= begin; kk--)
	{
		c->sol_rows[index + kk] = (aknown[kk] - ab[kk]*c->sol_rows[index + kk + 1]) / aa[kk];
	}
	free(p);
	free(q);
}

//solve tridiagonal system with thomas method for columns
//receives information about the cells
//the triagonal matrix [b,a,c] -> (upper, mid, lower) diagonals
// i - index of the column
inline void solve_thomasC(Grid *g, Cells *c, real * __restrict__ aa,real * __restrict__ ab,real * __restrict__ ac,real * __restrict__ aknown, Domain *d, int i)
{
	int ny = g->size_y;
	int index = i*ny;
	int begin = c->north[i], end = c->south[i];

	for(int k = begin + 1; k <= end; k++)
	{ // c -> a || a -> b || b -> c
		aa[k] = aa[k]*aa[k-1] - ac[k]*ab[k - 1];
		ab[k] *= aa[k-1];
		aknown[k] = aknown[k]*aa[k-1] -ac[k]*aknown[k - 1];
		ac[k] = 0.0;
	}
	c->sol_cols[index + end] = aknown[end] / aa[end];
	int k = end-1;
	real *p = (real*)malloc(sizeof(real)*ny);
	real *q = (real*)malloc(sizeof(real)*ny);
	
	for(int i = 0,k = end-1; k > begin+2;i+=4,k-=4){
		p[i] = aa[k];
		p[i+1] = p[i]*aa[k-1];
		p[i+2] = p[i+1]*aa[k-2];
		p[i+3] = p[i+2]*aa[k-3];
		q[i] = aknown[k] - ab[k]*c->sol_cols[index + k + 1];
		q[i+1] = p[i]*aknown[k-1] - ab[k-1]*q[i];
		q[i+2] = p[i+1]*aknown[k-2] - ab[k-2]*q[i+1];
		q[i+3] = p[i+2]*aknown[k-3] - ab[k-3]*q[i+2];
	}	
	for(int i = 0,k = end-1; k > begin+2;i++,k--){
		c->sol_cols[index + k] = q[i]/p[i];
	}
	for(int kk = k; kk >= begin; kk--)
	{
		c->sol_cols[index + kk] = (aknown[kk] - ab[kk]*c->sol_cols[index + kk + 1]) / aa[kk];
	}
	free(p);free(q);
}

//caculate the values for the rows
void calcSol_rows(Grid *g, Domain *d, Cells *c, real *__restrict__ aa,real *__restrict__ ab,real *__restrict__ ac,real *__restrict__ aknown)
{
	int nx = g->size_x, ny = g->size_y;
	int begin = d->min_y, end = d->max_y;
	real dx = g->max_x / (real)nx;
	real dy = g->max_y / (real)ny;
	real dxdy = dx*dy;
	real dxody = dx/dy;
	real dyodx = dy/dx; 
	real alpha = d->alpha;
	real alp2dyodx = alpha+2*dyodx;
	dyodx*=-1;
	//initialize auxiliar variables
	#pragma omp parallel for schedule(guided)
	for(int i = begin; i <= end; i++)
	{
		real * __restrict__ aa =(real*)_mm_malloc(sizeof(real)*ny,sizeof(real));
		real * __restrict__ ab=(real*)_mm_malloc(sizeof(real)*ny,sizeof(real));
		real * __restrict__ ac=(real*)_mm_malloc(sizeof(real)*ny,sizeof(real));
		real * __restrict__ aknown=(real*)_mm_malloc(sizeof(real)*ny,sizeof(real));
		//zeroAux(g, a);
		int index = i*nx;
		int it_i = c->west[i];
		int it_f = c->east[i];
		//initial element
		
		
		for(int j = it_i; j <= it_f; j++)
		{	
			if(c->code[index+j]==2){
				int ind = i+j*ny;
				aa[j] = alp2dyodx;
				ab[j] = dyodx;
				ac[j] = dyodx;
				aknown[j]=(c->sol_cols[ind-1]-c->sol_cols[ind]-c->sol_cols[ind]+c->sol_cols[ind+1])*dxody 
						+ c->fArray[index+j]*dxdy 
						+ alpha*c->sol_cols[ind];
			}
			else{
				aa[j] = 1;
				ab[j] = 0;
				ac[j] = 0;
				aknown[j] = c->code[index+j]*c->sol_rows[index + j];
			}	
		} 
		

		//solve with thomas
		solve_thomasR(g, c, aa,ab,ac,aknown, d, i);
		_mm_free(aa);
		_mm_free(ab);
		_mm_free(ac);
		_mm_free(aknown);
	}
}

//caculate the values for the cols
void calcSol_cols(Grid *g, Domain *d, Cells *c, real * __restrict__ aa,real * __restrict__ ab,real * __restrict__ ac,real * __restrict__ aknown)
{
	int nx = g->size_x, ny = g->size_y;
	int begin = d->min_x, end = d->max_x;
	real dx = g->max_x / (real)nx;
	real dy = g->max_y / (real)ny;
	real dxdy = dx*dy;
	real dxody = dx/dy;
	real dyodx = dy/dx; 
	real alpha = d->alpha;
	real alp2dxody = alpha+2*dxody;
	//initialize auxiliar variables
	dxody*=-1;
	#pragma omp parallel for schedule(guided)
	for(int i = begin; i <= end; i++)
	{
		
		real* aa =(real*)_mm_malloc(sizeof(real)*nx,sizeof(real));
		real* ab=(real*)_mm_malloc(sizeof(real)*nx,sizeof(real));
		real* ac=(real*)_mm_malloc(sizeof(real)*nx,sizeof(real));
		real* aknown=(real*)_mm_malloc(sizeof(real)*nx,sizeof(real));
		//zeroAux(g, a);
		int index = i;
		int index2 = i*ny;
		int it_i = c->north[i];
		int it_f = c->south[i];
		//printf("\ninicio:%d :: fim:%d\n", it_i, it_f);
		//initial element
		
		for(int j = it_i; j <= it_f; j++)
		{
			int jr = j*nx;
			if(c->code[index2+j]==2)
			{
				int ind = index + jr;
				aa[j] = alp2dxody;
				ab[j] = dxody;
				ac[j] = dxody;
			    aknown[j] =   (c->sol_rows[ind - 1] - c->sol_rows[ind] - c->sol_rows[ind] + c->sol_rows[ind + 1])*dyodx + 
					c->fArrayc[index2+j]*dxdy + 
					alpha*c->sol_rows[ind];
			}	
			else{
				aa[j] = 1;
				ab[j] = 0;
				ac[j] = 0;
				aknown[j] = c->code[index2+j]*c->sol_cols[index2 + j];
			}
		} 
		//solve with thomas
		solve_thomasC(g, c, aa,ab,ac,aknown, d, i);
		_mm_free(aa);
		_mm_free(ab);
		_mm_free(ac);
		_mm_free(aknown);
	}
}
