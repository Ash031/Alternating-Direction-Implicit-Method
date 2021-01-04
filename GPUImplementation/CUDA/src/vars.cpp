#ifndef _IMPORTS
#define _IMPORTS

#include "../headers/imports.h"

#endif

//test if a point is inside the domain
inline int test_Domain(Domain *d, real x, real y)
{
	return (((d->cx-x)*(d->cx-x) + (d->cy-y)*(d->cy-y)) < (d->radius*d->radius)) ? 1 : 0;
}

//get the expected result at point (x,y)
inline real get_ExactValue(real x, real y)
{
	return exp(x + 2.0*y);
	//return (1.0 + x + y);
}

//get the expected result at point (x,y)
inline real fonte(real x, real y)
{
	return -5.0*exp(x + 2.0*y);
	//return 0.0;
}

//initialize the grid with the domain values
void initGrid(Grid *g, int minx, int miny, int maxx, int maxy, int sizex, int sizey)
{
	g->min_x = minx;
	g->min_y = miny;
	g->max_x = maxx;
	g->max_y = maxy;
	g->size_x = sizex;
	g->size_y = sizey;
}

//initialize the domain with the values for the problem
void initDomain(Domain *d, Grid *g, real r, real a)
{
	d->sol = (real*)malloc(sizeof(real)*g->size_x*g->size_y);
	d->radius = r;
	d->alpha = a;
}

//initialize the values of the cells
void initCells(Grid *g, Domain *d, Cells *c)
{
	int nx = g->size_x, ny = g->size_y, nn = nx*ny;
	int counter;
	//allocate the values in memory
	c->x = (real*)malloc(sizeof(real)*nn);
	c->y = (real*)malloc(sizeof(real)*nn);
	c->xc = (real*)malloc(sizeof(real)*nn);
	c->yc = (real*)malloc(sizeof(real)*nn);
	c->sol_rows = (real*)malloc(sizeof(real)*nn);
	c->sol_cols = (real*)malloc(sizeof(real)*nn);
	c->code = (int*)malloc(sizeof(int)*nn);
	c->fArray = (real*)malloc(sizeof(real)*nn);
	c->fArrayc = (real*)malloc(sizeof(real)*nn);
	c->west = (int*)malloc(sizeof(int)*nx);
	c->east = (int*)malloc(sizeof(int)*nx);
	c->north = (int*)malloc(sizeof(int)*ny);
	c->south = (int*)malloc(sizeof(int)*ny);


	//get the cells baricenters
	real inv_x = g->max_x / (real)nx;
	real inv_y = g->max_y / (real)ny;
	real inv_x2 = inv_x * 0.5;
	real inv_y2 = inv_y * 0.5;
	for(int j=0; j<ny; j++)
	{
		for(int i=0; i<nx; i++)
		{	
			real x=i * inv_x + (inv_x2),y=j * inv_y + (inv_y2);
			c->x[i + j*nx] = x;
			c->y[i + j*nx] = y;
			c->fArray[i + j*nx] = fonte(x,y);
			c->xc[i*ny + j] = x;
			c->yc[i*ny + j] = y;
			c->fArrayc[i*ny + j] = fonte(x,y);
			
		}
	}
	
	//calculates the center of the problem domain
	d->cx = c->x[(int)ceil(nn*0.5 + nx*0.5)];
	d->cy = c->y[(int)ceil(nn*0.5 + nx*0.5)];
	
	//attribute the codes for every cell
	//depending on the places related to the problem domain
	// 0 -> out
	// 1 -> boundary
	// 2 -> in
	for(int i=0; i<nn; i++)
	{
		counter = 0;
		counter += test_Domain(d, c->x[i] - inv_x2, c->y[i] - inv_y2);
		counter += test_Domain(d, c->x[i] + inv_x2, c->y[i] - inv_y2);
		counter += test_Domain(d, c->x[i] - inv_x2, c->y[i] + inv_y2);
		counter += test_Domain(d, c->x[i] + inv_x2, c->y[i] + inv_y2);	
		if(counter == 0) //cell outside the domain
		{
			c->code[i] = 0;
		}
		else if(counter == 4) //cell inside the domain
		{
			c->code[i] = 2;
		}
		else //cell in the boundary
		{
			c->code[i] = 1;
		}
	}

	//compute the solution value for each cell
	// 0 for outer cells
	// 1 for inner cells
	// exact solution for boundary cells
	int id, id2;
	for(int j=0; j<ny; j++)
	{	
		for(int i=0; i<nx; i++)
		{
			id = i + j*nx;
			id2 = i*ny + j;
			real val = get_ExactValue(c->x[id], c->y[id]);
			if(c->code[id] == 0)
			{
				c->sol_rows[id] = 0.0;
				c->sol_cols[id2] = 0.0;
				d->sol[id] = 0.0;
			}
			else if(c->code[id] == 1)
			{
				c->sol_rows[id] = val;
				c->sol_cols[id2] = val;
				d->sol[id] = val;
			}
			else
			{
				c->sol_rows[id] = 0.5;
				c->sol_cols[id2] = 0.5;
				d->sol[id] = val;
			}
		}
	}

	//initiate limit values
	for(int i=0; i<ny; i++)
	{
		c->west[i] = -1;
		c->east[i] = -1;				
	}
	for(int j=0; j<nx; j++)
	{
		c->north[j] = -1; 
		c->south[j] = -1;	
	}
	//evaluate inner cells limits
	//west, east, north and south limits
	
	for(int j=1; j<(ny-1); j++)
	{	
		for(int i=1; i<(nx-1); i++)
		{
			c->west[j] = 2;
			c->east[j] = nx-2;
			c->north[i] = 2;
			c->south[i] = ny-2;
		}
	}
	
	d->min_x = 2;
	d->max_x = nx-2;
	d->min_y = 2;
	d->max_y = ny-2;
}


//clean grid from memory
void cleanGrid(Grid *g)
{
	free(g);
}

//clean domain from memory
void cleanDomain(Domain *d)
{
	free(d->sol);
	free(d);
}


//clean cells from memory
void cleanCells(Cells *c)
{
	free(c->x); free(c->y);
	free(c->sol_rows); free(c->sol_cols); free(c->code);
	free(c->west); free(c->east);
	free(c->north); free(c->south);
	free(c->xc);free(c->yc);free(c->fArray);free(c->fArrayc);
	free(c);

}

void initGridAndDomain(Grid* g, Domain* d){
	//All Tested Sizes
	/*initGrid(g, 0.0, 0.0, 2.0, 2.0, 80, 80); // initGrid(grid,min,y,max,y,size,y)
	initDomain(d, g, 0.5, 0.150); //initDomain(domain,grid,radius,alpha)*/
	/*initGrid(g, 0.0, 0.0, 2.0, 2.0, 160, 160);
	initDomain(d, g, 0.5, 0.0650);*/
	/*initGrid(g, 0.0, 0.0, 2.0, 2.0, 320, 320);
	initDomain(d, g, 0.5, 0.0375);*/
	/*initGrid(g, 0.0, 0.0, 2.0, 2.0, 640, 640);
	initDomain(d, g, 0.5, 0.0140);*/
	initGrid(g, 0.0, 0.0, 2.0, 2.0, 2052, 2052);
	initDomain(d, g, 0.5, 0.00875);
}

void init(Grid *g, Domain *d,Cells*c){
	initGridAndDomain(g,d);
	initCells(g, d, c);
}

void cleanAll(Grid *g, Domain *d,Cells*c){
	cleanGrid(g);
	cleanDomain(d);
	cleanCells(c);
}
