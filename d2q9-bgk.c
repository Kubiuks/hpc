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
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define MASTER 0

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    non_obst;       // number of non obstacle cells
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int nprocs;
  int rank;
  int rank_row_above;
  int rank_row_below;
  int start_ny;
  int end_ny;
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
               t_param* params, float** obstacles_ptr, float** av_vels_ptr, 
               t_speed_SOA** cells_ptr_SOA, t_speed_SOA** tmp_cells_ptr_SOA);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed_SOA* cells, t_speed_SOA* tmp_cells, float* obstacles);
int accelerate_flow(const t_param params, t_speed_SOA* cells, float* obstacles);
float collision(const t_param params, const t_speed_SOA*  cells, t_speed_SOA*  tmp_cells, float* obstacles);
int write_values(const t_param params, t_speed_SOA* cells, float* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_SOA** cells_ptr, t_speed_SOA** tmp_cells_ptr,
             float** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_SOA* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_SOA*  cells, float*  obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_SOA* cells, float* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed_SOA *cells = NULL;    /* grid containing fluid densities */
  t_speed_SOA *tmp_cells = NULL;/* scratch space */
  float*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;      /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

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

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &obstacles, &av_vels, &cells, &tmp_cells);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
    {
      av_vels[tt] = timestep(params, cells, tmp_cells, obstacles);
      
      //pointer switch
      t_speed_SOA *tempptr = cells;
      cells = tmp_cells;
      tmp_cells = tempptr;

#ifdef DEBUG
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;



  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));

  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  MPI_Finalize();

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed_SOA* cells, t_speed_SOA* tmp_cells, float* obstacles)
{ 
  accelerate_flow(params, cells, obstacles);
  return collision(params, cells, tmp_cells, obstacles);
}

int accelerate_flow(const t_param params, t_speed_SOA* cells, float* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;
  
  for (int ii = 8; ii < (params.nx-8); ii++)
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

float collision(const t_param params, const t_speed_SOA*  cells, t_speed_SOA*  tmp_cells, float* obstacles)
{
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  // initialise to zero
  __m256 tot_u = _mm256_set1_ps(0.f);
  // initialise some constants
  __m256 zeros       = _mm256_set1_ps(0.f);
  __m256 ones        = _mm256_set1_ps(1.f);
  __m256 w0_vector   = _mm256_set1_ps(w0);
  __m256 w1_vector   = _mm256_set1_ps(w1);
  __m256 w2_vector   = _mm256_set1_ps(w2);
  

  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 8; ii < (params.nx-8); ii += 8)
	  { 
	    /* determine indices of axis-direction neighbours
	    ** respecting periodic boundary conditions (wrap around) */
	    int y_n = (jj + 1) % params.ny;
	    int x_e = ii + 1;
	    int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
	    int x_w = ii - 1;

	    __m256 tmp_0 = _mm256_load_ps(&cells->c0[ii  +  jj*params.nx]); /* central cell, no movement */
	    __m256 tmp_1 = _mm256_load_ps(&cells->c1[x_w +  jj*params.nx]); /* east */
	    __m256 tmp_2 = _mm256_load_ps(&cells->c2[ii  + y_s*params.nx]); /* north */
	    __m256 tmp_3 = _mm256_load_ps(&cells->c3[x_e +  jj*params.nx]); /* west */
	    __m256 tmp_4 = _mm256_load_ps(&cells->c4[ii  + y_n*params.nx]); /* south */
	    __m256 tmp_5 = _mm256_load_ps(&cells->c5[x_w + y_s*params.nx]); /* north-east */
	    __m256 tmp_6 = _mm256_load_ps(&cells->c6[x_e + y_s*params.nx]); /* north-west */
	    __m256 tmp_7 = _mm256_load_ps(&cells->c7[x_e + y_n*params.nx]); /* south-west */
	    __m256 tmp_8 = _mm256_load_ps(&cells->c8[x_w + y_n*params.nx]); /* south-east */      

      __m256 tmp_158 = _mm256_add_ps(_mm256_add_ps(tmp_1, tmp_5), tmp_8);
      __m256 tmp_367 = _mm256_add_ps(_mm256_add_ps(tmp_3, tmp_6), tmp_7);
      __m256 tmp_256 = _mm256_add_ps(_mm256_add_ps(tmp_2, tmp_5), tmp_6);
      __m256 tmp_478 = _mm256_add_ps(_mm256_add_ps(tmp_4, tmp_7), tmp_8);

      __m256 local_density = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(tmp_0, tmp_256), _mm256_add_ps(tmp_478, tmp_3)), tmp_1);

      __m256 u_x = _mm256_div_ps(_mm256_sub_ps(tmp_158, tmp_367), local_density);
      __m256 u_y = _mm256_div_ps(_mm256_sub_ps(tmp_256, tmp_478), local_density);

      __m256 u_sq =_mm256_fmadd_ps(u_x, u_x, _mm256_mul_ps(u_y, u_y));

      __m256 u1 = u_x;
      __m256 u2 = u_y;
      __m256 u3 = _mm256_sub_ps(_mm256_set1_ps(0.f), u_x);
      __m256 u4 = _mm256_sub_ps(_mm256_set1_ps(0.f), u_y);
      __m256 u5 = _mm256_add_ps(u_x, u_y);
      __m256 u6 = _mm256_sub_ps(u_y, u_x);
      __m256 u7 = _mm256_sub_ps(_mm256_set1_ps(0.f), u5);
      __m256 u8 = _mm256_sub_ps(u_x, u_y);

      __m256 w0_local_d = _mm256_mul_ps(local_density, w0_vector);
      __m256 w1_local_d = _mm256_mul_ps(local_density, w1_vector);
      __m256 w2_local_d = _mm256_mul_ps(local_density, w2_vector);

      // 1-(u_sq/2c_sq), but c_sq is 1/3 so -> 1-(u_sq/(2/3)) -> 1-(u_sq*3/2) -> 1-(u_sq*1.5)) -> -u_sq*1.5+1
      // fnmadd(a, b, c) -> -(a*b)+c 
      __m256 constant = _mm256_fnmadd_ps(_mm256_set1_ps(1.5f), u_sq, _mm256_set1_ps(1.f));

      // u/c_sq + u*u/2*c_sq*c_sq --> 3u + 4.5u*u --> u(4.5*u+3)
      __m256 d_equ0 = _mm256_mul_ps(w0_local_d, constant);
      __m256 d_equ1 = _mm256_mul_ps(w1_local_d, _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(4.5f), u1, _mm256_set1_ps(3.f)), u1, constant));
      __m256 d_equ2 = _mm256_mul_ps(w1_local_d, _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(4.5f), u2, _mm256_set1_ps(3.f)), u2, constant));
      __m256 d_equ3 = _mm256_mul_ps(w1_local_d, _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(4.5f), u3, _mm256_set1_ps(3.f)), u3, constant));
      __m256 d_equ4 = _mm256_mul_ps(w1_local_d, _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(4.5f), u4, _mm256_set1_ps(3.f)), u4, constant));
      __m256 d_equ5 = _mm256_mul_ps(w2_local_d, _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(4.5f), u5, _mm256_set1_ps(3.f)), u5, constant));
      __m256 d_equ6 = _mm256_mul_ps(w2_local_d, _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(4.5f), u6, _mm256_set1_ps(3.f)), u6, constant));
      __m256 d_equ7 = _mm256_mul_ps(w2_local_d, _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(4.5f), u7, _mm256_set1_ps(3.f)), u7, constant));
      __m256 d_equ8 = _mm256_mul_ps(w2_local_d, _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(4.5f), u8, _mm256_set1_ps(3.f)), u8, constant));


      __m256 omega = _mm256_set1_ps(params.omega);

      __m256 tmp_0_new = _mm256_fmadd_ps(_mm256_sub_ps(d_equ0, tmp_0), omega, tmp_0);
      __m256 tmp_1_new = _mm256_fmadd_ps(_mm256_sub_ps(d_equ1, tmp_1), omega, tmp_1);
      __m256 tmp_2_new = _mm256_fmadd_ps(_mm256_sub_ps(d_equ2, tmp_2), omega, tmp_2);
      __m256 tmp_3_new = _mm256_fmadd_ps(_mm256_sub_ps(d_equ3, tmp_3), omega, tmp_3);
      __m256 tmp_4_new = _mm256_fmadd_ps(_mm256_sub_ps(d_equ4, tmp_4), omega, tmp_4);
      __m256 tmp_5_new = _mm256_fmadd_ps(_mm256_sub_ps(d_equ5, tmp_5), omega, tmp_5);
      __m256 tmp_6_new = _mm256_fmadd_ps(_mm256_sub_ps(d_equ6, tmp_6), omega, tmp_6);
      __m256 tmp_7_new = _mm256_fmadd_ps(_mm256_sub_ps(d_equ7, tmp_7), omega, tmp_7);
      __m256 tmp_8_new = _mm256_fmadd_ps(_mm256_sub_ps(d_equ8, tmp_8), omega, tmp_8);


      __m256 obst = _mm256_load_ps(&obstacles[ii + jj*params.nx]);


      // mask for the obstacles
      // if (!obstacles[ii + jj*params.nx])
      // if obstacles[0-7] = |0.00000|0.00000|1.00000|0.00000|0.00000|0.00000|0.00000|1.00000| (0.f is free, 1.f is blocked)
      // zeros             = |0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|
      // then mask         = |1111111|1111111|0000000|1111111|1111111|1111111|1111111|0000000|
      __m256 mask_obst = _mm256_cmp_ps(obst, zeros, _CMP_EQ_OQ);

      // similarly for cells
      __m256 mask_cell = _mm256_cmp_ps(obst, ones, _CMP_EQ_OQ);

      // now I need to store tmp_0_new if the cell is not obstacle and the original tmp_0 if it is
      // for tmp_cells->c1 i need to store tmp_1_new if it is not obst and tmp_3 if it is, etc to mimic rebound
      // so in each of the tmp_x_new vectors I need to replace the values with the original tmp_x if that
      // value in the vector is an obstacle

      // performing an AND on tmp_x_new and mask_obst will zero out all the obstacle entries
      // tmp_x_new: |01010|01110|01011|01010|
      // mask_obst: |11111|11111|00000|11111|
      // give:      |01010|01110|00000|01010|

      // performing an AND on tmp_x and mask_cell will zero out all the cell entries

      // adding those two together will give the desired vector
      _mm256_store_ps(&tmp_cells->c0[ii + jj*params.nx], _mm256_add_ps(_mm256_and_ps(tmp_0_new, mask_obst), _mm256_and_ps(tmp_0, mask_cell)));
      _mm256_store_ps(&tmp_cells->c1[ii + jj*params.nx], _mm256_add_ps(_mm256_and_ps(tmp_1_new, mask_obst), _mm256_and_ps(tmp_3, mask_cell)));
      _mm256_store_ps(&tmp_cells->c2[ii + jj*params.nx], _mm256_add_ps(_mm256_and_ps(tmp_2_new, mask_obst), _mm256_and_ps(tmp_4, mask_cell)));
      _mm256_store_ps(&tmp_cells->c3[ii + jj*params.nx], _mm256_add_ps(_mm256_and_ps(tmp_3_new, mask_obst), _mm256_and_ps(tmp_1, mask_cell)));
      _mm256_store_ps(&tmp_cells->c4[ii + jj*params.nx], _mm256_add_ps(_mm256_and_ps(tmp_4_new, mask_obst), _mm256_and_ps(tmp_2, mask_cell)));
      _mm256_store_ps(&tmp_cells->c5[ii + jj*params.nx], _mm256_add_ps(_mm256_and_ps(tmp_5_new, mask_obst), _mm256_and_ps(tmp_7, mask_cell)));
      _mm256_store_ps(&tmp_cells->c6[ii + jj*params.nx], _mm256_add_ps(_mm256_and_ps(tmp_6_new, mask_obst), _mm256_and_ps(tmp_8, mask_cell)));
      _mm256_store_ps(&tmp_cells->c7[ii + jj*params.nx], _mm256_add_ps(_mm256_and_ps(tmp_7_new, mask_obst), _mm256_and_ps(tmp_5, mask_cell)));
      _mm256_store_ps(&tmp_cells->c8[ii + jj*params.nx], _mm256_add_ps(_mm256_and_ps(tmp_8_new, mask_obst), _mm256_and_ps(tmp_6, mask_cell)));

      // calcuale av_vel
      // can't use no_obst because we would divide by 0
      __m256 tmp_158_new = _mm256_add_ps(_mm256_add_ps(tmp_1_new, tmp_5_new), tmp_8_new);
      __m256 tmp_367_new = _mm256_add_ps(_mm256_add_ps(tmp_3_new, tmp_6_new), tmp_7_new);
      __m256 tmp_256_new = _mm256_add_ps(_mm256_add_ps(tmp_2_new, tmp_5_new), tmp_6_new);
      __m256 tmp_478_new = _mm256_add_ps(_mm256_add_ps(tmp_4_new, tmp_7_new), tmp_8_new);

      __m256 local_density_new = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(tmp_0_new, tmp_256_new), _mm256_add_ps(tmp_478_new, tmp_3_new)), tmp_1_new);

      __m256 u_x_new = _mm256_div_ps(_mm256_sub_ps(tmp_158_new, tmp_367_new), local_density_new);
      __m256 u_y_new = _mm256_div_ps(_mm256_sub_ps(tmp_256_new, tmp_478_new), local_density_new);

      __m256 temp_u = _mm256_sqrt_ps(_mm256_fmadd_ps(u_x_new, u_x_new, _mm256_mul_ps(u_y_new, u_y_new)));

      // now zero out all entries in temp_u that are 0000000 in mask
      // so that we add 0 to tot_u for those entires
      // we can just perform AND on temp_u and mask such that
      // |0101010|011011| - some floats
      // |1111111|000000| - mask
      // |0101010|000000| - ANDing the two

      tot_u = _mm256_add_ps(tot_u, _mm256_and_ps(temp_u, mask_obst));

	  }
  }

  float res_tot_u = 0.f;

  for(int i=0;i<8;i++)
  {
    res_tot_u += (float)tot_u[i];
  }
  
  return res_tot_u / (float)params.non_obst;
}

float av_velocity(const t_param params, t_speed_SOA*  cells, float*  obstacles)
{
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 8; ii < (params.nx-8); ii++)
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
               t_param* params, float** obstacles_ptr, float** av_vels_ptr, 
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

  /* and close up the file */
  fclose(fp);

  // calculate what rows the rank will be allocated and other constants
  params->non_obst = params->nx*params->ny;
  params->nx = params->nx+16;

  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  params->nprocs = nprocs;
  params->rank = rank;

  params->rank_row_above = (rank + 1) % nprocs;
  params->rank_row_below = (rank == MASTER) ? (rank + nprocs - 1) : (rank - 1);
  
  // to spread the work as evenly as possible I split the number of rows like:
  // (⌊ny/nprocs⌋ + 1) rows in (ny (mod nprocs)) ranks
  // ⌊ny/nprocs⌋ rows in the rest of the ranks (nprocs - (ny (mod nprocs)))
  int num_rows_per_rank = floor(params->ny/nprocs);
  int num_heavier_ranks = params->ny % nprocs;
  if(rank < num_heavier_ranks){
    // these get 1 more row than rest
    params->start_ny = rank * (num_rows_per_rank+1);
    params->end_ny = params->start_ny + (num_rows_per_rank+1);
  } else {
    params->start_ny = rank * num_rows_per_rank + num_heavier_ranks;
    params->end_ny = params->start_ny + num_rows_per_rank;
  }
  int my_num_rows = params->end_ny - params->start_ny;

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
  
  (*cells_ptr_SOA)->c0 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*cells_ptr_SOA)->c1 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*cells_ptr_SOA)->c2 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*cells_ptr_SOA)->c3 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*cells_ptr_SOA)->c4 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*cells_ptr_SOA)->c5 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*cells_ptr_SOA)->c6 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*cells_ptr_SOA)->c7 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*cells_ptr_SOA)->c8 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);

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

  (*tmp_cells_ptr_SOA)->c0 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*tmp_cells_ptr_SOA)->c1 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*tmp_cells_ptr_SOA)->c2 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*tmp_cells_ptr_SOA)->c3 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*tmp_cells_ptr_SOA)->c4 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*tmp_cells_ptr_SOA)->c5 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*tmp_cells_ptr_SOA)->c6 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*tmp_cells_ptr_SOA)->c7 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*tmp_cells_ptr_SOA)->c8 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(float) * params->ny * params->nx);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;
  
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 8; ii < (params->nx-8); ii++)
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
    for (int ii = 8; ii < (params->nx-8); ii++)
	  {
	  (*obstacles_ptr)[ii + jj*params->nx] = 0.f;
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
      (*obstacles_ptr)[xx + 8 + yy*params->nx] = (float)blocked;
      --params->non_obst;
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
             float** obstacles_ptr, float** av_vels_ptr)
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


float calc_reynolds(const t_param params, t_speed_SOA* cells, float* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed_SOA* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 8; ii < (params.nx-8); ii++)
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

int write_values(const t_param params, t_speed_SOA* cells, float* obstacles, float* av_vels)
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
      for (int ii = 8; ii < (params.nx-8); ii++)
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
	  fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", (ii-8), jj, u_x, u_y, u, pressure, (int)obstacles[ii * params.nx + jj]);
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
