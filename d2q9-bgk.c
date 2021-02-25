/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <immintrin.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    n;             // nx*ny number of all cells
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
/* AOS
typedef struct
{
  float speeds[NSPEEDS];
} t_speed; 
*/

// SOA
typedef struct
{
  float* c0;
  float* c1;
  float* c2;
  float* c3;
  float* c4;
  float* c5;
  float* c6;
  float* c7;
  float* c8;
} t_speed_SOA;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, int** obstacles_ptr, float** av_vels_ptr, 
               t_speed_SOA** cells_ptr_SOA, t_speed_SOA** tmp_cells_ptr_SOA);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed_SOA* cells, t_speed_SOA* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed_SOA* cells, int* obstacles);
int collision(const t_param params, const t_speed_SOA*  cells, t_speed_SOA*  tmp_cells, int*  obstacles);
int write_values(const t_param params, t_speed_SOA* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_SOA** cells_ptr, t_speed_SOA** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_SOA* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_SOA*  cells, int*  obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_SOA* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed_SOA *cells = NULL;    /* grid containing fluid densities */
  t_speed_SOA *tmp_cells = NULL;/* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;      /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
    {
      usage(argv[0]);
    }
  else
    {
      paramfile = argv[1];
      obstaclefile = argv[2];
    }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &obstacles, &av_vels, &cells, &tmp_cells);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  for (int tt = 0; tt < params.maxIters; tt++)
    {
      timestep(params, cells, tmp_cells, obstacles);

      //pointer switch
      t_speed_SOA *tempptr = cells;
      cells = tmp_cells;
      tmp_cells = tempptr;

      av_vels[tt] = av_velocity(params, cells, obstacles);

#ifdef DEBUG
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed_SOA* cells, t_speed_SOA* tmp_cells, int* obstacles)
{ 
  accelerate_flow(params, cells, obstacles);
  collision(params, cells, tmp_cells, obstacles);
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed_SOA* cells, int* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
	    && (cells->c3[ii + jj*params.nx] - w1) > 0.f
	    && (cells->c6[ii + jj*params.nx] - w2) > 0.f
	    && (cells->c7[ii + jj*params.nx] - w2) > 0.f)
	  {
	    /* increase 'east-side' densities */
	    cells->c1[ii + jj*params.nx] += w1;
	    cells->c5[ii + jj*params.nx] += w2;
	    cells->c8[ii + jj*params.nx] += w2;
	    /* decrease 'west-side' densities */
	    cells->c3[ii + jj*params.nx] -= w1;
	    cells->c6[ii + jj*params.nx] -= w2;
	    cells->c7[ii + jj*params.nx] -= w2;
	  }
  }
  return EXIT_SUCCESS;
}

int collision(const t_param params, const t_speed_SOA*  cells, t_speed_SOA*  tmp_cells, int*  obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
	  { 
	    /* determine indices of axis-direction neighbours
	    ** respecting periodic boundary conditions (wrap around) */
	    int y_n = (jj + 1) % params.ny;
	    int x_e = (ii + 1) % params.nx;
	    int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
	    int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

	    /* don't consider occupied cells */
	    if (!obstacles[ii + jj*params.nx])
	    { 
	      /* propagate densities from neighbouring cells, following
	      ** appropriate directions of travel and writing into
	      ** scratch space grid */
	      float tmp_0 = cells->c0[ii  +  jj*params.nx]; /* central cell, no movement */
	      float tmp_1 = cells->c1[x_w +  jj*params.nx]; /* east */
	      float tmp_2 = cells->c2[ii  + y_s*params.nx]; /* north */
	      float tmp_3 = cells->c3[x_e +  jj*params.nx]; /* west */
	      float tmp_4 = cells->c4[ii  + y_n*params.nx]; /* south */
	      float tmp_5 = cells->c5[x_w + y_s*params.nx]; /* north-east */
	      float tmp_6 = cells->c6[x_e + y_s*params.nx]; /* north-west */
	      float tmp_7 = cells->c7[x_e + y_n*params.nx]; /* south-west */
	      float tmp_8 = cells->c8[x_w + y_n*params.nx]; /* south-east */

	      /* compute local density total */
	      float local_density = 0.f;

	      float tmp_158 = tmp_1 + tmp_5 + tmp_8;
	      float tmp_367 = tmp_3 + tmp_6 + tmp_7;
	      float tmp_256 = tmp_2 + tmp_5 + tmp_6;
	      float tmp_478 = tmp_4 + tmp_7 + tmp_8;

	      local_density = tmp_0 + tmp_256 + tmp_478 + tmp_3 + tmp_1;

	      /* compute x velocity component */
	      float u_x = (tmp_158 - tmp_367) / local_density;
	      /* compute y velocity component */
	      float u_y = (tmp_256 - tmp_478) / local_density;

	      /* velocity squared */
	      float u_sq = u_x * u_x + u_y * u_y;

	      /* directional velocity components */
	      float u[NSPEEDS];
	      u[1] =   u_x;        /* east */
	      u[2] =         u_y;  /* north */
	      u[3] = - u_x;        /* west */
	      u[4] =       - u_y;  /* south */
	      u[5] =   u_x + u_y;  /* north-east */
	      u[6] = - u_x + u_y;  /* north-west */
	      u[7] = - u_x - u_y;  /* south-west */
	      u[8] =   u_x - u_y;  /* south-east */

	      /* equilibrium densities */
	      float d_equ[NSPEEDS];

	      /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density * (1.f -u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq + (u[1] * u[1]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq + (u[2] * u[2]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq + (u[3] * u[3]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq + (u[4] * u[4]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq + (u[5] * u[5]) / (2.f * c_sq * c_sq) -u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq + (u[6] * u[6]) / (2.f * c_sq * c_sq) -u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq + (u[7] * u[7]) / (2.f * c_sq * c_sq) -u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq + (u[8] * u[8]) / (2.f * c_sq * c_sq) -u_sq / (2.f * c_sq));
          
        tmp_cells->c0[ii + jj*params.nx] = tmp_0 + params.omega * (d_equ[0] - tmp_0);
        tmp_cells->c1[ii + jj*params.nx] = tmp_1 + params.omega * (d_equ[1] - tmp_1);
        tmp_cells->c2[ii + jj*params.nx] = tmp_2 + params.omega * (d_equ[2] - tmp_2);
        tmp_cells->c3[ii + jj*params.nx] = tmp_3 + params.omega * (d_equ[3] - tmp_3);
        tmp_cells->c4[ii + jj*params.nx] = tmp_4 + params.omega * (d_equ[4] - tmp_4);
        tmp_cells->c5[ii + jj*params.nx] = tmp_5 + params.omega * (d_equ[5] - tmp_5);
        tmp_cells->c6[ii + jj*params.nx] = tmp_6 + params.omega * (d_equ[6] - tmp_6);
        tmp_cells->c7[ii + jj*params.nx] = tmp_7 + params.omega * (d_equ[7] - tmp_7);
        tmp_cells->c8[ii + jj*params.nx] = tmp_8 + params.omega * (d_equ[8] - tmp_8);
	    }
	    else
	    { 
	      tmp_cells->c0[ii + jj*params.nx] = cells->c0[ii  +  jj*params.nx];
	      tmp_cells->c1[ii + jj*params.nx] = cells->c3[x_e +  jj*params.nx];
	      tmp_cells->c2[ii + jj*params.nx] = cells->c4[ii  + y_n*params.nx];
	      tmp_cells->c3[ii + jj*params.nx] = cells->c1[x_w +  jj*params.nx];
	      tmp_cells->c4[ii + jj*params.nx] = cells->c2[ii  + y_s*params.nx];
	      tmp_cells->c5[ii + jj*params.nx] = cells->c7[x_e + y_n*params.nx];
	      tmp_cells->c6[ii + jj*params.nx] = cells->c8[x_w + y_n*params.nx];
	      tmp_cells->c7[ii + jj*params.nx] = cells->c5[x_w + y_s*params.nx];
	      tmp_cells->c8[ii + jj*params.nx] = cells->c6[x_e + y_s*params.nx];
	    }
	  }
  }
  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed_SOA*  cells, int*  obstacles)
{
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
	  {
	    /* ignore occupied cells */
	    if (!obstacles[ii + jj*params.nx])
	    {
	      /* local density total */
	      float local_density = 0.f;

	      float tmp_0 = cells->c0[ii + jj*params.nx]; /* central cell, no movement */
	      float tmp_1 = cells->c1[ii + jj*params.nx]; /* east */
	      float tmp_2 = cells->c2[ii + jj*params.nx]; /* north */
	      float tmp_3 = cells->c3[ii + jj*params.nx]; /* west */
	      float tmp_4 = cells->c4[ii + jj*params.nx]; /* south */
	      float tmp_5 = cells->c5[ii + jj*params.nx]; /* north-east */
	      float tmp_6 = cells->c6[ii + jj*params.nx]; /* north-west */
	      float tmp_7 = cells->c7[ii + jj*params.nx]; /* south-west */
	      float tmp_8 = cells->c8[ii + jj*params.nx]; /* south-east */


	      float tmp_158 = tmp_1 + tmp_5 + tmp_8;
	      float tmp_367 = tmp_3 + tmp_6 + tmp_7;
	      float tmp_256 = tmp_2 + tmp_5 + tmp_6;
	      float tmp_478 = tmp_4 + tmp_7 + tmp_8;

	      local_density = tmp_0 + tmp_256 + tmp_478 + tmp_3 + tmp_1;

	      /* compute x velocity component */
	      float u_x = (tmp_158 - tmp_367) / local_density;
	      /* compute y velocity component */
	      float u_y = (tmp_256 - tmp_478) / local_density;

	      /* accumulate the norm of x- and y- velocity components */
	      tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
	      /* increase counter of inspected cells */
	      ++tot_cells;
	    }
	  }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, int** obstacles_ptr, float** av_vels_ptr, 
               t_speed_SOA** cells_ptr_SOA, t_speed_SOA** tmp_cells_ptr_SOA)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
    {
      sprintf(message, "could not open input parameter file: %s", paramfile);
      die(message, __LINE__, __FILE__);
    }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  params->n = params->nx * params->ny;

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  //*cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  //if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  // allocating memory for SOA
  *cells_ptr_SOA = malloc(sizeof(t_speed_SOA));
  if (*cells_ptr_SOA == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  (*cells_ptr_SOA)->c0 = NULL;
  (*cells_ptr_SOA)->c1 = NULL;
  (*cells_ptr_SOA)->c2 = NULL;
  (*cells_ptr_SOA)->c3 = NULL;
  (*cells_ptr_SOA)->c4 = NULL;
  (*cells_ptr_SOA)->c5 = NULL;
  (*cells_ptr_SOA)->c6 = NULL;
  (*cells_ptr_SOA)->c7 = NULL;
  (*cells_ptr_SOA)->c8 = NULL;  
  
  (*cells_ptr_SOA)->c0 = _mm_malloc(sizeof(float)*params->n,64);
  (*cells_ptr_SOA)->c1 = _mm_malloc(sizeof(float)*params->n,64);
  (*cells_ptr_SOA)->c2 = _mm_malloc(sizeof(float)*params->n,64);
  (*cells_ptr_SOA)->c3 = _mm_malloc(sizeof(float)*params->n,64);
  (*cells_ptr_SOA)->c4 = _mm_malloc(sizeof(float)*params->n,64);
  (*cells_ptr_SOA)->c5 = _mm_malloc(sizeof(float)*params->n,64);
  (*cells_ptr_SOA)->c6 = _mm_malloc(sizeof(float)*params->n,64);
  (*cells_ptr_SOA)->c7 = _mm_malloc(sizeof(float)*params->n,64);
  (*cells_ptr_SOA)->c8 = _mm_malloc(sizeof(float)*params->n,64);

  *tmp_cells_ptr_SOA = malloc(sizeof(t_speed_SOA));
  if (*tmp_cells_ptr_SOA == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  (*tmp_cells_ptr_SOA)->c0 = NULL;
  (*tmp_cells_ptr_SOA)->c1 = NULL;
  (*tmp_cells_ptr_SOA)->c2 = NULL;
  (*tmp_cells_ptr_SOA)->c3 = NULL;
  (*tmp_cells_ptr_SOA)->c4 = NULL;
  (*tmp_cells_ptr_SOA)->c5 = NULL;
  (*tmp_cells_ptr_SOA)->c6 = NULL;
  (*tmp_cells_ptr_SOA)->c7 = NULL;
  (*tmp_cells_ptr_SOA)->c8 = NULL;

  (*tmp_cells_ptr_SOA)->c0 = _mm_malloc(sizeof(float)*params->n,64);
  (*tmp_cells_ptr_SOA)->c1 = _mm_malloc(sizeof(float)*params->n,64);
  (*tmp_cells_ptr_SOA)->c2 = _mm_malloc(sizeof(float)*params->n,64);
  (*tmp_cells_ptr_SOA)->c3 = _mm_malloc(sizeof(float)*params->n,64);
  (*tmp_cells_ptr_SOA)->c4 = _mm_malloc(sizeof(float)*params->n,64);
  (*tmp_cells_ptr_SOA)->c5 = _mm_malloc(sizeof(float)*params->n,64);
  (*tmp_cells_ptr_SOA)->c6 = _mm_malloc(sizeof(float)*params->n,64);
  (*tmp_cells_ptr_SOA)->c7 = _mm_malloc(sizeof(float)*params->n,64);
  (*tmp_cells_ptr_SOA)->c8 = _mm_malloc(sizeof(float)*params->n,64);
  

  /* 'helper' grid, used as scratch space */
  //*tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  //if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
	  {
	  /* centre */
	  (*cells_ptr_SOA)->c0[ii + jj*params->nx] = w0;
	  /* axis directions */
	  (*cells_ptr_SOA)->c1[ii + jj*params->nx] = w1;
	  (*cells_ptr_SOA)->c2[ii + jj*params->nx] = w1;
	  (*cells_ptr_SOA)->c3[ii + jj*params->nx] = w1;
	  (*cells_ptr_SOA)->c4[ii + jj*params->nx] = w1;
	  /* diagonals */
	  (*cells_ptr_SOA)->c5[ii + jj*params->nx] = w2;
	  (*cells_ptr_SOA)->c6[ii + jj*params->nx] = w2;
	  (*cells_ptr_SOA)->c7[ii + jj*params->nx] = w2;
	  (*cells_ptr_SOA)->c8[ii + jj*params->nx] = w2;
	  }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
	  {
	  (*obstacles_ptr)[ii + jj*params->nx] = 0;
	  }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
    {
      sprintf(message, "could not open input obstacles file: %s", obstaclefile);
      die(message, __LINE__, __FILE__);
    }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
      /* some checks */
      if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

      if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

      if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

      if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

      /* assign to array */
      (*obstacles_ptr)[xx + yy*params->nx] = blocked;
    }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed_SOA** cells_ptr, t_speed_SOA** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  _mm_free((*cells_ptr)->c0);
  _mm_free((*cells_ptr)->c1);
  _mm_free((*cells_ptr)->c2);
  _mm_free((*cells_ptr)->c3);
  _mm_free((*cells_ptr)->c4);
  _mm_free((*cells_ptr)->c5);
  _mm_free((*cells_ptr)->c6);
  _mm_free((*cells_ptr)->c7);
  _mm_free((*cells_ptr)->c8);
  free(*cells_ptr);

  (*cells_ptr)->c0 = NULL;
  (*cells_ptr)->c1 = NULL;
  (*cells_ptr)->c2 = NULL;
  (*cells_ptr)->c3 = NULL;
  (*cells_ptr)->c4 = NULL;
  (*cells_ptr)->c5 = NULL;
  (*cells_ptr)->c6 = NULL;
  (*cells_ptr)->c7 = NULL;
  (*cells_ptr)->c8 = NULL;
  *cells_ptr = NULL;

  _mm_free((*tmp_cells_ptr)->c0);
  _mm_free((*tmp_cells_ptr)->c1);
  _mm_free((*tmp_cells_ptr)->c2);
  _mm_free((*tmp_cells_ptr)->c3);
  _mm_free((*tmp_cells_ptr)->c4);
  _mm_free((*tmp_cells_ptr)->c5);
  _mm_free((*tmp_cells_ptr)->c6);
  _mm_free((*tmp_cells_ptr)->c7);
  _mm_free((*tmp_cells_ptr)->c8);
  free(*tmp_cells_ptr);

  (*tmp_cells_ptr)->c0 = NULL;
  (*tmp_cells_ptr)->c1 = NULL;
  (*tmp_cells_ptr)->c2 = NULL;
  (*tmp_cells_ptr)->c3 = NULL;
  (*tmp_cells_ptr)->c4 = NULL;
  (*tmp_cells_ptr)->c5 = NULL;
  (*tmp_cells_ptr)->c6 = NULL;
  (*tmp_cells_ptr)->c7 = NULL;
  (*tmp_cells_ptr)->c8 = NULL;
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed_SOA* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed_SOA* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
	  {
	  total += cells->c0[ii + jj*params.nx];
	  total += cells->c1[ii + jj*params.nx];
	  total += cells->c2[ii + jj*params.nx];
	  total += cells->c3[ii + jj*params.nx];
	  total += cells->c4[ii + jj*params.nx];
	  total += cells->c5[ii + jj*params.nx];
	  total += cells->c6[ii + jj*params.nx];
	  total += cells->c7[ii + jj*params.nx];
	  total += cells->c8[ii + jj*params.nx];
	  }
  }
  return total;
}

int write_values(const t_param params, t_speed_SOA* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
    {
      die("could not open file output file", __LINE__, __FILE__);
    }

  for (int jj = 0; jj < params.ny; jj++)
    {
      for (int ii = 0; ii < params.nx; ii++)
	{
	  /* an occupied cell */
	  if (obstacles[ii + jj*params.nx])
	    {
	      u_x = u_y = u = 0.f;
	      pressure = params.density * c_sq;
	    }
	  /* no obstacle */
	  else
	    {
	      local_density = 0.f;

	     	local_density += cells->c0[ii + jj*params.nx];
	      local_density += cells->c1[ii + jj*params.nx];
	      local_density += cells->c2[ii + jj*params.nx];
	      local_density += cells->c3[ii + jj*params.nx];
	      local_density += cells->c4[ii + jj*params.nx];
	      local_density += cells->c5[ii + jj*params.nx];
	      local_density += cells->c6[ii + jj*params.nx];
	      local_density += cells->c7[ii + jj*params.nx];
	      local_density += cells->c8[ii + jj*params.nx];

	      /* compute x velocity component */
	      u_x = (cells->c0[ii + jj*params.nx]
               + cells->c5[ii + jj*params.nx]
               + cells->c8[ii + jj*params.nx]
		     - (cells->c3[ii + jj*params.nx]
                  + cells->c6[ii + jj*params.nx]
			+ cells->c7[ii + jj*params.nx]))
		/ local_density;
	      /* compute y velocity component */
	      u_y = (cells->c2[ii + jj*params.nx]
               + cells->c5[ii + jj*params.nx]
               + cells->c6[ii + jj*params.nx]
		     - (cells->c4[ii + jj*params.nx]
                  + cells->c7[ii + jj*params.nx]
			+ cells->c8[ii + jj*params.nx]))
		/ local_density;
	      /* compute norm of velocity */
	      u = sqrtf((u_x * u_x) + (u_y * u_y));
	      /* compute pressure */
	      pressure = local_density * c_sq;
	    }

	  /* write to file */
	  fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
	}
    }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
    {
      die("could not open file output file", __LINE__, __FILE__);
    }

  for (int ii = 0; ii < params.maxIters; ii++)
    {
      fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}