/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DIFFEQMODEL_H
#define DIFFEQMODEL_H

// these defines are needed by SignalAverager ..   TODO: I should move these to a separate file
typedef enum {
	PX_BOUND = 1,
	MX_BOUND = 2,
	PY_BOUND = 4,
	MY_BOUND = 8,
	PZ_BOUND = 16,
	MZ_BOUND = 32,
	WALL_ELEMENT = 64

} BoundaryClass;

#define ANY_BOUNDARY (PX_BOUND | MX_BOUND | PY_BOUND | MY_BOUND | PZ_BOUND | MZ_BOUND)

#include <stdlib.h>
#include "xtalk_sim.h"
#include "../Util/WorkerInfoQueue.h"
#include "SignalAverager.h"
#include "WellVoxelTracker.h"
#include "DelsqCUDA.h"

#define NUM_THREADS 16

// distribute processing across multiple threads for speedy goodness
typedef enum
{
    DiffEqStep,
    CmatrixUpdate,
    ExitThreads
} WorkType;

typedef struct {
	int kstart;
	int kend;
    WorkType task;
} ThreadControlStruct;

// control structure that defines size/dimensions of simulation
// and some necessary constants
class DiffEqModel
{
public:
	int ni;				// number of elements along x dimension
	int nj;				// number of elements along y dimension
	int nk;				// number of elements along z dimension
	DATA_TYPE dx;		// size of element along x dimension (microns)
	DATA_TYPE dy;		// size of element along y dimension (microns)
	DATA_TYPE dz;		// size of element along z dimension (microns)
	DATA_TYPE dt;		// time step size (in seconds)
	DATA_TYPE dcoeff;	// diffusion coefficient
	double   simTime;	// the current time point of the simulation, this needs to be double precision
						// in order to keep track of small time steps correctly
	unsigned char *boundary_class;	// pointer to 3-d matrix of boundary conditions per voxel
	DATA_TYPE *correction_factor;	// correction factor caused by special boundary conditions
	DATA_TYPE *buffer_effect;		// pointer to 3-d matrix of buffering information per voxel
	DATA_TYPE *layer_step_frac;     // pointer to convection constant for flow in x-direction (1 value per nk)
	DATA_TYPE *cmatrix;				// pointer to current matrix of concentration values
	SignalAverager *head_avg;		// pointer to list of signal averagers (used to capture the simulated signal in each well)
	DATA_TYPE *xvel;				// flow velocity in x-direction as a function of z
	WellVoxelTracker *pwell;	    // pointer to object that keeps track of the voxel we inject incorporation signal into

	// allocates publically-accessible memory and performs some basic initialization
	DiffEqModel(int ni,int nj,int nk);

	// starts processing threads, allocates internal memory, and indexes boundary conditions
	void initializeModel(void);

	// runs the model until the stopTime.  Make be called multiple times to sucessively simulate further time points
	int runModel(double endTime);

	// shuts down the processing threads.  deallocates internal memory.
	void shutdownModel();
	
	// frees up public memory
	~DiffEqModel();

	/**** these methods are in DiffEqPattern_Init and are used to build the boundary conditions ****/

	// build a hexagonal well pattern in the bottom wellz elements of the simulation space
	void BuildHexagonalWellPattern(int wellz,DATA_TYPE pitch,DATA_TYPE well_fraction);
	
	// configures the buffer effect matrix to account for surface buffer of well walls
	void SetBufferEffect(DATA_TYPE wash_buffer_cap);

	// sets boundary_class correctly for the edge pixels of the matrix
	void SetEdgeBoundaryClass(void);
	
	// configures convection (only supported as directly along the x-axis for now)
	void SetConvection(int wellz);

	// set the initial concentration in a specified well to a specific value
	void SetSignalInWell(int wellz,int row,int col,DATA_TYPE pitch,DATA_TYPE well_fraction,DATA_TYPE signal);

	// handles the propagation of [dNTP] to the point of simulation over time
	void DynamicDNTPBoundaryCondition(DATA_TYPE simTime);
	
	// calculates the average buffering within a single well
	DATA_TYPE AddUpWellBuffering(int wellz,int row,int col,DATA_TYPE pitch,DATA_TYPE well_fraction,DATA_TYPE wash_buffer_cap);
	
	// adds the specified amount of buffering to all voxels that aren't wall material in the bottom wellz of the simulation
	// which effectively adds bead-like buffering to all the wells in the simulation
	void AddBeadBufferingToAllWells(int wellz,DATA_TYPE bead_buffer_amount,DATA_TYPE wash_buffer_cap);
	
	// adds specified amount of buffering to all voxel in a single well
	void AddBeadBufferingToSingleWell(int wellz,int row,int col,DATA_TYPE pitch,DATA_TYPE well_fraction,DATA_TYPE bead_buffer_amount,DATA_TYPE wash_buffer_cap);
	
	// configures a single well for generation of incorporation signal during the simulation
	void SetupIncorpSignalInjection(int wellz,int row,int col,DATA_TYPE pitch,DATA_TYPE well_fraction);
	
	// build matricies for single well signal injection that are usable by the GPU for injection in GPU code
	void SetupIncorporationSignalInjectionOnGPU(void);

	// adds incorporation signal to the configured well
	void AddSignalToWell(WellVoxelTracker *pwell,DATA_TYPE signal);
	
	// configures the radious of the bead for signal generation to accurately reflect the
	// distribution of signal generation in the well
	void ConfigureBeadSize(DATA_TYPE bead_radius);

private:

	// static entry point to thread processing
	static void *ThreadProc_entry(void *arg)
	{
		DiffEqModel *myObj = (DiffEqModel *)arg;
		myObj->ThreadProc();
		pthread_exit(NULL);
	}

	// non-static method that performs specific processing tasks
	void ThreadProc(void);

	// main diff-eq computation
	void delsq(DATA_TYPE *src,DATA_TYPE *dst,int startk,int stopk);

	// handles boundary special computation for boundary conditions
	void delsq_special(DATA_TYPE *src,DATA_TYPE *dst);

	// compute the main diff-eq calculation using available threads
	void MultiThreadedDiffEqCalc(void);

	// build an index of elements that will require special processing
	void indexBoundaryConditions(void);

	// launch processing threads
	void createThreads(void);

	// for internal use by model routines
	WorkerInfoQueue *workQ;		// worker Q for threads
	DATA_TYPE *cmatrix_next;	// pointer to the next value of the cmatrix that is currently being computed
	DATA_TYPE *dptr_last;		// pointer to scratch-pad for diffeq calc
	int *boundaryIndex;			// pointer to array of matrix indicies where special boundary processing
								// will be necessary
	int nBoundary;				// number of elements in boundaryIndex
	DATA_TYPE *zeros_ptr;		// pointer to array of zeros needed when the diffeq calc hits an edge of the matrix
	DATA_TYPE *ccl_ptr;			// pointer to aligned scratch space for temporary data during computation
	DATA_TYPE *ccr_ptr;			// pointer to aligned scratch space for temporary data during computation
	pthread_t *threadInfo;		// pointer to thread identifiers
	
	DATA_TYPE *bead_volume_per_slice;	// normalized fraction of the bead volume in each vertical slice of the well
	int bead_z_slices;			// number of vertical slices that the bead occupies
	size_t    *index_array;		// indicies used by GPU for incorporation signal injection
	DATA_TYPE *weight_array;	// weights used by GPU for incorporation signal injection
	int incorp_inject_cnt;		// number of elements in index_array and weight_array

// CUDA execution
  DelsqCUDA * cudaModel;

};

#endif // DIFFEQMODEL_H
