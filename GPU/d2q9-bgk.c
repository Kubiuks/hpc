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
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <immintrin.h>
#include <CL/opencl.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCL_KERNELS_FILE "kernels.cl"
#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

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
} t_param;

/* struct to hold OpenCL objects */
typedef struct {
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;

  cl_program program;

  size_t local_nx;
  size_t local_ny;

  cl_mem d_cells_c0;
  cl_mem d_cells_c1;
  cl_mem d_cells_c2;
  cl_mem d_cells_c3;
  cl_mem d_cells_c4;
  cl_mem d_cells_c5;
  cl_mem d_cells_c6;
  cl_mem d_cells_c7;
  cl_mem d_cells_c8;
  cl_mem d_tmp_cells_c0;
  cl_mem d_tmp_cells_c1;
  cl_mem d_tmp_cells_c2;
  cl_mem d_tmp_cells_c3;
  cl_mem d_tmp_cells_c4;
  cl_mem d_tmp_cells_c5;
  cl_mem d_tmp_cells_c6;
  cl_mem d_tmp_cells_c7;
  cl_mem d_tmp_cells_c8;
  cl_mem d_obstacles;
  cl_mem d_av_vels;

  cl_kernel  accelerate_flow_even_iter;
  cl_kernel  collision_even_iter;

  cl_kernel  accelerate_flow_odd_iter;
  cl_kernel  collision_odd_iter;

} t_ocl;

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
void checkError(cl_int err, const char *op, const int line);
cl_device_id selectOpenCLDevice();

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** obstacles_ptr, float** av_vels_ptr, 
               t_speed_SOA** h_cells_ptr_SOA, t_speed_SOA** h_tmp_cells_ptr_SOA, t_ocl *ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed_SOA* h_cells, t_speed_SOA* h_tmp_cells, float* obstacles, t_ocl ocl, int even_odd);
int accelerate_flow(const t_param params, t_speed_SOA* h_cells, float* obstacles, t_ocl ocl, int even_odd);
int collision(const t_param params, const t_speed_SOA*  h_cells, t_speed_SOA*  h_tmp_cells, float* obstacles, t_ocl ocl, int even_odd);
int write_values(const t_param params, t_speed_SOA* h_cells, float* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_SOA** h_cells_ptr, t_speed_SOA** h_tmp_cells_ptr,
             float** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_SOA* h_cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_SOA*  h_cells, float*  obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_SOA* h_cells, float* obstacles);

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
  t_speed_SOA *h_cells = NULL;    /* grid containing fluid densities */
  t_speed_SOA *h_tmp_cells = NULL;/* scratch space */
  float*     h_obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;      /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  t_ocl ocl;
  cl_int err;

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

  // /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &h_obstacles, &av_vels, &h_cells, &h_tmp_cells, &ocl);
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells_c0, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c0, 0, NULL, NULL);
  checkError(err, "writing  h_cells.c0 data", __LINE__);
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells_c1, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c1, 0, NULL, NULL);
  checkError(err, "writing  h_cells.c1 data", __LINE__);
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells_c2, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c2, 0, NULL, NULL);
  checkError(err, "writing  h_cells.c2 data", __LINE__);
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells_c3, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c3, 0, NULL, NULL);
  checkError(err, "writing  h_cells.c3 data", __LINE__);
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells_c4, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c4, 0, NULL, NULL);
  checkError(err, "writing  h_cells.c4 data", __LINE__);
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells_c5, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c5, 0, NULL, NULL);
  checkError(err, "writing  h_cells.c5 data", __LINE__);
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells_c6, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c6, 0, NULL, NULL);
  checkError(err, "writing  h_cells.c6 data", __LINE__);
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells_c7, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c7, 0, NULL, NULL);
  checkError(err, "writing  h_cells.c7 data", __LINE__);
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells_c8, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c8, 0, NULL, NULL);
  checkError(err, "writing  h_cells.c8 data", __LINE__);
	err = clEnqueueWriteBuffer(ocl.queue, ocl.d_obstacles, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_obstacles, 0, NULL, NULL);
  checkError(err, "writing  h_obstacles data", __LINE__);

  // Set kernel arguments
  int second_row_from_top = params.ny - 2;
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 0, sizeof(cl_mem), &ocl.d_cells_c1);
  checkError(err, "setting accelerate_flow_even_iter arg 0", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 1, sizeof(cl_mem), &ocl.d_cells_c3);
  checkError(err, "setting accelerate_flow_even_iter arg 1", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 2, sizeof(cl_mem), &ocl.d_cells_c5);
  checkError(err, "setting accelerate_flow_even_iter arg 2", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 3, sizeof(cl_mem), &ocl.d_cells_c6);
  checkError(err, "setting accelerate_flow_even_iter arg 3", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 4, sizeof(cl_mem), &ocl.d_cells_c7);
  checkError(err, "setting accelerate_flow_even_iter arg 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 5, sizeof(cl_mem), &ocl.d_cells_c8);
  checkError(err, "setting accelerate_flow_even_iter arg 5", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 6, sizeof(cl_mem), &ocl.d_obstacles);
  checkError(err, "setting accelerate_flow_even_iter arg 6", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 7, sizeof(cl_int), &second_row_from_top);
  checkError(err, "setting accelerate_flow_even_iter arg 7", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 8, sizeof(cl_int), &params.nx);
  checkError(err, "setting accelerate_flow_even_iter arg 8", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 9, sizeof(cl_float), &params.density);
  checkError(err, "setting accelerate_flow_even_iter arg 9", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_even_iter, 10, sizeof(cl_float), &params.accel);
  checkError(err, "setting accelerate_flow_even_iter arg 10", __LINE__);

  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 0, sizeof(cl_mem), &ocl.d_tmp_cells_c1);
  checkError(err, "setting accelerate_flow_odd_iter arg 0", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 1, sizeof(cl_mem), &ocl.d_tmp_cells_c3);
  checkError(err, "setting accelerate_flow_odd_iter arg 1", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 2, sizeof(cl_mem), &ocl.d_tmp_cells_c5);
  checkError(err, "setting accelerate_flow_odd_iter arg 2", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 3, sizeof(cl_mem), &ocl.d_tmp_cells_c6);
  checkError(err, "setting accelerate_flow_odd_iter arg 3", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 4, sizeof(cl_mem), &ocl.d_tmp_cells_c7);
  checkError(err, "setting accelerate_flow_odd_iter arg 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 5, sizeof(cl_mem), &ocl.d_tmp_cells_c8);
  checkError(err, "setting accelerate_flow_odd_iter arg 5", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 6, sizeof(cl_mem), &ocl.d_obstacles);
  checkError(err, "setting accelerate_flow_odd_iter arg 6", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 7, sizeof(cl_int), &second_row_from_top);
  checkError(err, "setting accelerate_flow_odd_iter arg 7", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 8, sizeof(cl_int), &params.nx);
  checkError(err, "setting accelerate_flow_odd_iter arg 8", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 9, sizeof(cl_float), &params.density);
  checkError(err, "setting accelerate_flow_odd_iter arg 9", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow_odd_iter, 10, sizeof(cl_float), &params.accel);
  checkError(err, "setting accelerate_flow_odd_iter arg 10", __LINE__);
  
  err = clSetKernelArg(ocl.collision_even_iter, 0, sizeof(cl_mem), &ocl.d_cells_c0);
  checkError(err, "setting collision_even_iter arg 0", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 1, sizeof(cl_mem), &ocl.d_cells_c1);
  checkError(err, "setting collision_even_iter arg 1", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 2, sizeof(cl_mem), &ocl.d_cells_c2);
  checkError(err, "setting collision_even_iter arg 2", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 3, sizeof(cl_mem), &ocl.d_cells_c3);
  checkError(err, "setting collision_even_iter arg 3", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 4, sizeof(cl_mem), &ocl.d_cells_c4);
  checkError(err, "setting collision_even_iter arg 4", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 5, sizeof(cl_mem), &ocl.d_cells_c5);
  checkError(err, "setting collision_even_iter arg 5", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 6, sizeof(cl_mem), &ocl.d_cells_c6);
  checkError(err, "setting collision_even_iter arg 6", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 7, sizeof(cl_mem), &ocl.d_cells_c7);
  checkError(err, "setting collision_even_iter arg 7", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 8, sizeof(cl_mem), &ocl.d_cells_c8);
  checkError(err, "setting collision_even_iter arg 8", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 9, sizeof(cl_mem), &ocl.d_tmp_cells_c0);
  checkError(err, "setting collision_even_iter arg 9", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 10, sizeof(cl_mem), &ocl.d_tmp_cells_c1);
  checkError(err, "setting collision_even_iter arg 10", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 11, sizeof(cl_mem), &ocl.d_tmp_cells_c2);
  checkError(err, "setting collision_even_iter arg 11", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 12, sizeof(cl_mem), &ocl.d_tmp_cells_c3);
  checkError(err, "setting collision_even_iter arg 12", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 13, sizeof(cl_mem), &ocl.d_tmp_cells_c4);
  checkError(err, "setting collision_even_iter arg 13", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 14, sizeof(cl_mem), &ocl.d_tmp_cells_c5);
  checkError(err, "setting collision_even_iter arg 14", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 15, sizeof(cl_mem), &ocl.d_tmp_cells_c6);
  checkError(err, "setting collision_even_iter arg 15", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 16, sizeof(cl_mem), &ocl.d_tmp_cells_c7);
  checkError(err, "setting collision_even_iter arg 15", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 17, sizeof(cl_mem), &ocl.d_tmp_cells_c8);
  checkError(err, "setting collision_even_iter arg 17", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 18, sizeof(cl_mem), &ocl.d_obstacles);
  checkError(err, "setting collision_even_iter arg 18", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 19, sizeof(cl_mem), &ocl.d_av_vels);
  checkError(err, "setting collision_even_iter arg 19", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 20, sizeof(cl_int), &params.ny);
  checkError(err, "setting collision_even_iter arg 20", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 21, sizeof(cl_int), &params.nx);
  checkError(err, "setting collision_even_iter arg 21", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 22, sizeof(cl_float), &params.omega);
  checkError(err, "setting collision_even_iter arg 22", __LINE__);
  err = clSetKernelArg(ocl.collision_even_iter, 23, sizeof(cl_float) * ocl.local_nx * ocl.local_ny, NULL);
  checkError(err, "setting collision_even_iter arg 23", __LINE__);

  err = clSetKernelArg(ocl.collision_odd_iter, 0, sizeof(cl_mem), &ocl.d_tmp_cells_c0);
  checkError(err, "setting collision_odd_iter arg 0", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 1, sizeof(cl_mem), &ocl.d_tmp_cells_c1);
  checkError(err, "setting collision_odd_iter arg 1", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 2, sizeof(cl_mem), &ocl.d_tmp_cells_c2);
  checkError(err, "setting collision_odd_iter arg 2", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 3, sizeof(cl_mem), &ocl.d_tmp_cells_c3);
  checkError(err, "setting collision_odd_iter arg 3", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 4, sizeof(cl_mem), &ocl.d_tmp_cells_c4);
  checkError(err, "setting collision_odd_iter arg 4", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 5, sizeof(cl_mem), &ocl.d_tmp_cells_c5);
  checkError(err, "setting collision_odd_iter arg 5", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 6, sizeof(cl_mem), &ocl.d_tmp_cells_c6);
  checkError(err, "setting collision_odd_iter arg 6", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 7, sizeof(cl_mem), &ocl.d_tmp_cells_c7);
  checkError(err, "setting collision_odd_iter arg 7", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 8, sizeof(cl_mem), &ocl.d_tmp_cells_c8);
  checkError(err, "setting collision_odd_iter arg 8", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 9, sizeof(cl_mem), &ocl.d_cells_c0);
  checkError(err, "setting collision_odd_iter arg 9", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 10, sizeof(cl_mem), &ocl.d_cells_c1);
  checkError(err, "setting collision_odd_iter arg 10", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 11, sizeof(cl_mem), &ocl.d_cells_c2);
  checkError(err, "setting collision_odd_iter arg 11", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 12, sizeof(cl_mem), &ocl.d_cells_c3);
  checkError(err, "setting collision_odd_iter arg 12", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 13, sizeof(cl_mem), &ocl.d_cells_c4);
  checkError(err, "setting collision_odd_iter arg 13", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 14, sizeof(cl_mem), &ocl.d_cells_c5);
  checkError(err, "setting collision_odd_iter arg 14", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 15, sizeof(cl_mem), &ocl.d_cells_c6);
  checkError(err, "setting collision_odd_iter arg 15", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 16, sizeof(cl_mem), &ocl.d_cells_c7);
  checkError(err, "setting collision_odd_iter arg 15", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 17, sizeof(cl_mem), &ocl.d_cells_c8);
  checkError(err, "setting collision_odd_iter arg 17", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 18, sizeof(cl_mem), &ocl.d_obstacles);
  checkError(err, "setting collision_odd_iter arg 18", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 19, sizeof(cl_mem), &ocl.d_av_vels);
  checkError(err, "setting collision_odd_iter arg 19", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 20, sizeof(cl_int), &params.ny);
  checkError(err, "setting collision_odd_iter arg 20", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 21, sizeof(cl_int), &params.nx);
  checkError(err, "setting collision_odd_iter arg 21", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 22, sizeof(cl_float), &params.omega);
  checkError(err, "setting collision_odd_iter arg 22", __LINE__);
  err = clSetKernelArg(ocl.collision_odd_iter, 23, sizeof(cl_float) * ocl.local_nx * ocl.local_ny, NULL);
  checkError(err, "setting collision_odd_iter arg 23", __LINE__);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {   
      err = clSetKernelArg(ocl.collision_even_iter, 24, sizeof(cl_int), &tt);
      checkError(err, "setting collision_even_iter arg 24", __LINE__);
      err = clSetKernelArg(ocl.collision_odd_iter, 24, sizeof(cl_int), &tt);
      checkError(err, "setting collision_odd_iter arg 24", __LINE__);
      int even_odd = tt % 2;
      timestep(params, h_cells, h_tmp_cells, h_obstacles, ocl, even_odd);
  }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  err = clEnqueueReadBuffer(ocl.queue, ocl.d_tmp_cells_c0, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c0, 0, NULL, NULL);
  checkError(err, "reading h_cells->c0 data", __LINE__);
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_tmp_cells_c1, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c1, 0, NULL, NULL);
  checkError(err, "reading h_cells->c1 data", __LINE__);
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_tmp_cells_c2, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c2, 0, NULL, NULL);
  checkError(err, "reading h_cells->c2 data", __LINE__);
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_tmp_cells_c3, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c3, 0, NULL, NULL);
  checkError(err, "reading h_cells->c3 data", __LINE__);
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_tmp_cells_c4, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c4, 0, NULL, NULL);
  checkError(err, "reading h_cells->c4 data", __LINE__);
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_tmp_cells_c5, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c5, 0, NULL, NULL);
  checkError(err, "reading h_cells->c5 data", __LINE__);
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_tmp_cells_c6, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c6, 0, NULL, NULL);
  checkError(err, "reading h_cells->c6 data", __LINE__);
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_tmp_cells_c7, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c7, 0, NULL, NULL);
  checkError(err, "reading h_cells->c7 data", __LINE__);
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_tmp_cells_c8, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, h_cells->c8, 0, NULL, NULL);
  checkError(err, "reading h_cells->c8 data", __LINE__);

  // get average velocities from gpu

  int number_local_groups = (params.nx / ocl.local_nx) * (params.ny / ocl.local_ny);
  float h_tmp_av_vels[number_local_groups*params.maxIters];
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_av_vels, CL_TRUE, 0, sizeof(float) * number_local_groups * params.maxIters, h_tmp_av_vels, 0, NULL, NULL);
  checkError(err, "reading h_tmp_av_vels data", __LINE__); 

  for (int j = 0; j < params.maxIters; ++j) {
		float tot_u = 0.f;
		for (int i = 0; i < number_local_groups; ++i) {
			tot_u += h_tmp_av_vels[i + j*number_local_groups];
		}
		av_vels[j] = tot_u / (float)params.non_obst;
	}

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, h_cells, h_obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, h_cells, h_obstacles, av_vels);
  finalise(&params, &h_cells, &h_tmp_cells, &h_obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed_SOA* h_cells, t_speed_SOA* h_tmp_cells, float* obstacles, t_ocl ocl, int even_odd)
{ 
  accelerate_flow(params, h_cells, obstacles, ocl, even_odd);
  collision(params, h_cells, h_tmp_cells, obstacles, ocl, even_odd);
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed_SOA* h_cells, float* obstacles, t_ocl ocl, int even_odd)
{ 
  cl_int err;
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, even_odd ? ocl.accelerate_flow_odd_iter : ocl.accelerate_flow_even_iter, 1, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);
  err = clFinish(ocl.queue);
  checkError(err, "waiting for accelerate_flow kernel", __LINE__);
  return EXIT_SUCCESS;
}

int collision(const t_param params, const t_speed_SOA*  h_cells, t_speed_SOA*  h_tmp_cells, float* obstacles, t_ocl ocl, int even_odd)
{
  cl_int err;
  size_t global[2] = {params.nx, params.ny};
  size_t local[2] = {ocl.local_nx, ocl.local_ny};
  err = clEnqueueNDRangeKernel(ocl.queue, even_odd ? ocl.collision_odd_iter : ocl.collision_even_iter, 2, NULL, global, local, 0, NULL, NULL);
  checkError(err, "enqueueing collision kernel", __LINE__);
  err = clFinish(ocl.queue);
  checkError(err, "waiting for collision kernel", __LINE__);
  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed_SOA*  h_cells, float*  obstacles)
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

	      float tmp_0 = h_cells->c0[ii + jj*params.nx]; /* central cell, no movement */
	      float tmp_1 = h_cells->c1[ii + jj*params.nx]; /* east */
	      float tmp_2 = h_cells->c2[ii + jj*params.nx]; /* north */
	      float tmp_3 = h_cells->c3[ii + jj*params.nx]; /* west */
	      float tmp_4 = h_cells->c4[ii + jj*params.nx]; /* south */
	      float tmp_5 = h_cells->c5[ii + jj*params.nx]; /* north-east */
	      float tmp_6 = h_cells->c6[ii + jj*params.nx]; /* north-west */
	      float tmp_7 = h_cells->c7[ii + jj*params.nx]; /* south-west */
	      float tmp_8 = h_cells->c8[ii + jj*params.nx]; /* south-east */


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
               t_speed_SOA** h_cells_ptr_SOA, t_speed_SOA** h_tmp_cells_ptr_SOA, t_ocl *ocl)
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

  params->non_obst = params->nx*params->ny;

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
  *h_cells_ptr_SOA = malloc(sizeof(t_speed_SOA));
  if (*h_cells_ptr_SOA == NULL) die("cannot allocate memory for h_cells", __LINE__, __FILE__);
  (*h_cells_ptr_SOA)->c0 = NULL;
  (*h_cells_ptr_SOA)->c1 = NULL;
  (*h_cells_ptr_SOA)->c2 = NULL;
  (*h_cells_ptr_SOA)->c3 = NULL;
  (*h_cells_ptr_SOA)->c4 = NULL;
  (*h_cells_ptr_SOA)->c5 = NULL;
  (*h_cells_ptr_SOA)->c6 = NULL;
  (*h_cells_ptr_SOA)->c7 = NULL;
  (*h_cells_ptr_SOA)->c8 = NULL;  
  
  (*h_cells_ptr_SOA)->c0 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_cells_ptr_SOA)->c1 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_cells_ptr_SOA)->c2 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_cells_ptr_SOA)->c3 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_cells_ptr_SOA)->c4 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_cells_ptr_SOA)->c5 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_cells_ptr_SOA)->c6 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_cells_ptr_SOA)->c7 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_cells_ptr_SOA)->c8 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);

  *h_tmp_cells_ptr_SOA = malloc(sizeof(t_speed_SOA));
  if (*h_tmp_cells_ptr_SOA == NULL) die("cannot allocate memory for h_tmp_cells", __LINE__, __FILE__);
  (*h_tmp_cells_ptr_SOA)->c0 = NULL;
  (*h_tmp_cells_ptr_SOA)->c1 = NULL;
  (*h_tmp_cells_ptr_SOA)->c2 = NULL;
  (*h_tmp_cells_ptr_SOA)->c3 = NULL;
  (*h_tmp_cells_ptr_SOA)->c4 = NULL;
  (*h_tmp_cells_ptr_SOA)->c5 = NULL;
  (*h_tmp_cells_ptr_SOA)->c6 = NULL;
  (*h_tmp_cells_ptr_SOA)->c7 = NULL;
  (*h_tmp_cells_ptr_SOA)->c8 = NULL;

  (*h_tmp_cells_ptr_SOA)->c0 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_tmp_cells_ptr_SOA)->c1 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_tmp_cells_ptr_SOA)->c2 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_tmp_cells_ptr_SOA)->c3 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_tmp_cells_ptr_SOA)->c4 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_tmp_cells_ptr_SOA)->c5 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_tmp_cells_ptr_SOA)->c6 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_tmp_cells_ptr_SOA)->c7 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  (*h_tmp_cells_ptr_SOA)->c8 = _mm_malloc(sizeof(float)*params->nx*params->ny,64);
  

  /* 'helper' grid, used as scratch space */
  //*h_tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  //if (*h_tmp_cells_ptr == NULL) die("cannot allocate memory for h_tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(float) * params->ny * params->nx);

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
	  (*h_cells_ptr_SOA)->c0[ii + jj*params->nx] = w0;
	  /* axis directions */
	  (*h_cells_ptr_SOA)->c1[ii + jj*params->nx] = w1;
	  (*h_cells_ptr_SOA)->c2[ii + jj*params->nx] = w1;
	  (*h_cells_ptr_SOA)->c3[ii + jj*params->nx] = w1;
	  (*h_cells_ptr_SOA)->c4[ii + jj*params->nx] = w1;
	  /* diagonals */
	  (*h_cells_ptr_SOA)->c5[ii + jj*params->nx] = w2;
	  (*h_cells_ptr_SOA)->c6[ii + jj*params->nx] = w2;
	  (*h_cells_ptr_SOA)->c7[ii + jj*params->nx] = w2;
	  (*h_cells_ptr_SOA)->c8[ii + jj*params->nx] = w2;
	  }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
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

      --params->non_obst;
      /* assign to array */
      (*obstacles_ptr)[xx + yy*params->nx] = (float)blocked;
    }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  // Initialise OpenCL
  // Get an OpenCL device
  char *ocl_src;      /* OpenCL kernel source */
  long ocl_size;      /* size of OpenCL kernel source */
  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCL_KERNELS_FILE, "r");
  if (fp == NULL) {
    sprintf(message, "could not open OpenCL kernel file: %s", OCL_KERNELS_FILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char *)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(ocl->context, 1,
                                           (const char **)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t sz;
    clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0,
                          NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, sz,
                          buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow_even_iter = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accerelate_flow kernel", __LINE__);
  ocl->collision_even_iter = clCreateKernel(ocl->program, "collision", &err);
  checkError(err, "creating collision kernel", __LINE__);
  ocl->accelerate_flow_odd_iter = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accerelate_flow kernel", __LINE__);
  ocl->collision_odd_iter = clCreateKernel(ocl->program, "collision", &err);
  checkError(err, "creating collision kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->d_cells_c0 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_cells_c0", __LINE__);
  ocl->d_cells_c1 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_cells_c1", __LINE__);
  ocl->d_cells_c2 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_cells_c2", __LINE__);
  ocl->d_cells_c3 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_cells_c3", __LINE__);
  ocl->d_cells_c4 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_cells_c4", __LINE__);
  ocl->d_cells_c5 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_cells_c5", __LINE__);
  ocl->d_cells_c6 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_cells_c6", __LINE__);
  ocl->d_cells_c7 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_cells_c7", __LINE__);
  ocl->d_cells_c8 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_cells_c8", __LINE__);
  ocl->d_tmp_cells_c0 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_tmp_cells_c0", __LINE__);
  ocl->d_tmp_cells_c1 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_tmp_cells_c1", __LINE__);
  ocl->d_tmp_cells_c2 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_tmp_cells_c2", __LINE__);
  ocl->d_tmp_cells_c3 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_tmp_cells_c3", __LINE__);
  ocl->d_tmp_cells_c4 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_tmp_cells_c4", __LINE__);
  ocl->d_tmp_cells_c5 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_tmp_cells_c5", __LINE__);
  ocl->d_tmp_cells_c6 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_tmp_cells_c6", __LINE__);
  ocl->d_tmp_cells_c7 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_tmp_cells_c7", __LINE__);
  ocl->d_tmp_cells_c8 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_tmp_cells_c8", __LINE__);
  ocl->d_obstacles = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,  sizeof(float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating buffer d_obstacles", __LINE__);

  ocl->local_nx = 32;
  ocl->local_ny = 32;
  int number_local_groups = (params->nx / ocl->local_nx) * (params->ny / ocl->local_ny);
	ocl->d_av_vels = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(float) * number_local_groups * params->maxIters, NULL, &err);
  checkError(err, "creating buffer d_av_vels", __LINE__);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed_SOA** h_cells_ptr, t_speed_SOA** h_tmp_cells_ptr,
             float** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
{
  /*
  ** free up allocated memory
  */
  _mm_free((*h_cells_ptr)->c0);
  _mm_free((*h_cells_ptr)->c1);
  _mm_free((*h_cells_ptr)->c2);
  _mm_free((*h_cells_ptr)->c3);
  _mm_free((*h_cells_ptr)->c4);
  _mm_free((*h_cells_ptr)->c5);
  _mm_free((*h_cells_ptr)->c6);
  _mm_free((*h_cells_ptr)->c7);
  _mm_free((*h_cells_ptr)->c8);
  free(*h_cells_ptr);

  (*h_cells_ptr)->c0 = NULL;
  (*h_cells_ptr)->c1 = NULL;
  (*h_cells_ptr)->c2 = NULL;
  (*h_cells_ptr)->c3 = NULL;
  (*h_cells_ptr)->c4 = NULL;
  (*h_cells_ptr)->c5 = NULL;
  (*h_cells_ptr)->c6 = NULL;
  (*h_cells_ptr)->c7 = NULL;
  (*h_cells_ptr)->c8 = NULL;
  *h_cells_ptr = NULL;

  _mm_free((*h_tmp_cells_ptr)->c0);
  _mm_free((*h_tmp_cells_ptr)->c1);
  _mm_free((*h_tmp_cells_ptr)->c2);
  _mm_free((*h_tmp_cells_ptr)->c3);
  _mm_free((*h_tmp_cells_ptr)->c4);
  _mm_free((*h_tmp_cells_ptr)->c5);
  _mm_free((*h_tmp_cells_ptr)->c6);
  _mm_free((*h_tmp_cells_ptr)->c7);
  _mm_free((*h_tmp_cells_ptr)->c8);
  free(*h_tmp_cells_ptr);

  (*h_tmp_cells_ptr)->c0 = NULL;
  (*h_tmp_cells_ptr)->c1 = NULL;
  (*h_tmp_cells_ptr)->c2 = NULL;
  (*h_tmp_cells_ptr)->c3 = NULL;
  (*h_tmp_cells_ptr)->c4 = NULL;
  (*h_tmp_cells_ptr)->c5 = NULL;
  (*h_tmp_cells_ptr)->c6 = NULL;
  (*h_tmp_cells_ptr)->c7 = NULL;
  (*h_tmp_cells_ptr)->c8 = NULL;
  *h_tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.d_cells_c0);
  clReleaseMemObject(ocl.d_cells_c1);
  clReleaseMemObject(ocl.d_cells_c2);
  clReleaseMemObject(ocl.d_cells_c3);
  clReleaseMemObject(ocl.d_cells_c4);
  clReleaseMemObject(ocl.d_cells_c5);
  clReleaseMemObject(ocl.d_cells_c6);
  clReleaseMemObject(ocl.d_cells_c7);
  clReleaseMemObject(ocl.d_cells_c8);
  clReleaseMemObject(ocl.d_tmp_cells_c0);
  clReleaseMemObject(ocl.d_tmp_cells_c1);
  clReleaseMemObject(ocl.d_tmp_cells_c2);
  clReleaseMemObject(ocl.d_tmp_cells_c3);
  clReleaseMemObject(ocl.d_tmp_cells_c4);
  clReleaseMemObject(ocl.d_tmp_cells_c5);
  clReleaseMemObject(ocl.d_tmp_cells_c6);
  clReleaseMemObject(ocl.d_tmp_cells_c7);
  clReleaseMemObject(ocl.d_tmp_cells_c8);
  clReleaseKernel(ocl.accelerate_flow_even_iter);
  clReleaseKernel(ocl.collision_even_iter);
  clReleaseKernel(ocl.accelerate_flow_odd_iter);
  clReleaseKernel(ocl.collision_odd_iter);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed_SOA* h_cells, float* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, h_cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed_SOA* h_cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
	  {
	    total += h_cells->c0[ii + jj*params.nx];
	    total += h_cells->c1[ii + jj*params.nx];
	    total += h_cells->c2[ii + jj*params.nx];
	    total += h_cells->c3[ii + jj*params.nx];
	    total += h_cells->c4[ii + jj*params.nx];
	    total += h_cells->c5[ii + jj*params.nx];
	    total += h_cells->c6[ii + jj*params.nx];
	    total += h_cells->c7[ii + jj*params.nx];
	    total += h_cells->c8[ii + jj*params.nx];
	  }
  }
  return total;
}

int write_values(const t_param params, t_speed_SOA* h_cells, float* obstacles, float* av_vels)
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

	     	local_density += h_cells->c0[ii + jj*params.nx];
	      local_density += h_cells->c1[ii + jj*params.nx];
	      local_density += h_cells->c2[ii + jj*params.nx];
	      local_density += h_cells->c3[ii + jj*params.nx];
	      local_density += h_cells->c4[ii + jj*params.nx];
	      local_density += h_cells->c5[ii + jj*params.nx];
	      local_density += h_cells->c6[ii + jj*params.nx];
	      local_density += h_cells->c7[ii + jj*params.nx];
	      local_density += h_cells->c8[ii + jj*params.nx];

	      /* compute x velocity component */
	      u_x = (h_cells->c0[ii + jj*params.nx]
               + h_cells->c5[ii + jj*params.nx]
               + h_cells->c8[ii + jj*params.nx]
		     - (h_cells->c3[ii + jj*params.nx]
                  + h_cells->c6[ii + jj*params.nx]
			+ h_cells->c7[ii + jj*params.nx]))
		/ local_density;
	      /* compute y velocity component */
	      u_y = (h_cells->c2[ii + jj*params.nx]
               + h_cells->c5[ii + jj*params.nx]
               + h_cells->c6[ii + jj*params.nx]
		     - (h_cells->c4[ii + jj*params.nx]
                  + h_cells->c7[ii + jj*params.nx]
			+ h_cells->c8[ii + jj*params.nx]))
		/ local_density;
	      /* compute norm of velocity */
	      u = sqrtf((u_x * u_x) + (u_y * u_y));
	      /* compute pressure */
	      pressure = local_density * c_sq;
	    }

	  /* write to file */
	  fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, (int)obstacles[ii + params.nx * jj]);
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

void checkError(cl_int err, const char *op, const int line) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
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

cl_device_id selectOpenCLDevice() {
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++) {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES - total_devices, devices + total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++) {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env) {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices) {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME, MAX_DEVICE_NAME, name,
                  NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}