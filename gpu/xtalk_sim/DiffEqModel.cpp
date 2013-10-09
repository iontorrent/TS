
#include <stdlib.h>
#include "DiffEqModel.h"
//#include "vectorclass.h"
#include "immintrin.h"
#include <assert.h>


#define USE_CUDA

typedef float v8f __attribute__ ((vector_size (32)));

extern DATA_TYPE GetIncorpFlux(double time);

// allocates publically-accessible memory and performs some basic initialization
DiffEqModel::DiffEqModel(int ni,int nj,int nk)
{
	this->ni = ni;
	this->nj = nj;
	this->nk = nk;

	boundary_class = new unsigned char[ni*nj*nk];
	assert(!posix_memalign((void **)&correction_factor,32,sizeof(DATA_TYPE[ni*nj*nk])));
	assert(!posix_memalign((void **)&buffer_effect,32,sizeof(DATA_TYPE[ni*nj*nk])));
	layer_step_frac = new DATA_TYPE[nk];
	assert(!posix_memalign((void **)&cmatrix,32,sizeof(DATA_TYPE[ni*nj*nk])));
	xvel = new DATA_TYPE[nk];

	memset(boundary_class,0,sizeof(unsigned char[ni*nj*nk]));
	memset(correction_factor,0,sizeof(DATA_TYPE[ni*nj*nk]));
	memset(buffer_effect,0,sizeof(DATA_TYPE[ni*nj*nk]));
	memset(layer_step_frac,0,sizeof(DATA_TYPE[nk]));
	memset(cmatrix,0,sizeof(DATA_TYPE[ni*nj*nk]));
	memset(xvel,0,sizeof(DATA_TYPE[nk]));
	
	simTime = 0.0;
	
	head_avg = NULL;
	pwell = NULL;
	
	bead_volume_per_slice = NULL;
	bead_z_slices = 0;
	index_array = NULL;
	weight_array = NULL;
	incorp_inject_cnt = 0;
}

// frees up public memory
DiffEqModel::~DiffEqModel()
{
	delete [] boundary_class;
	free(correction_factor);
	free(buffer_effect);
	delete [] layer_step_frac;
	free(cmatrix);
	delete [] xvel;
	
	while (head_avg != NULL)
	{
		SignalAverager *next = head_avg->GetNext();
		delete head_avg;
		head_avg = next;
	}
	
	if (pwell != NULL)
		delete pwell;
	
	if (bead_volume_per_slice != NULL)
		delete [] bead_volume_per_slice;
		
	if (index_array != NULL)
		delete [] index_array;
	if (weight_array != NULL)
		delete [] weight_array;
}

// calculates the right-hand-side of the differential equation
// this is basically the sum of the 2nd derivative along each dimension,
// with scaling to account for voxel dimensions and a convection term
// along the x-dimension
// edges are handled in every direction by assuming no flux across the edge
// These special cases (at the edges of the matrix, and at well wall boundaries
// are handled by applying a correction factor to the coefficients used in
// the main computation)
void DiffEqModel::delsq(DATA_TYPE *src,DATA_TYPE *dst,int startk,int stopk)
{
    DATA_TYPE dxsq = dx*dx;
    DATA_TYPE cx1 = dcoeff*dt/dxsq;
    DATA_TYPE cx2 = -2.0f*dcoeff*dt/dxsq;

    DATA_TYPE dysq = dy*dy;
    DATA_TYPE cy1 = dcoeff*dt/dysq;
    DATA_TYPE cy2 = -2.0f*dcoeff*dt/dysq;

    DATA_TYPE dzsq = dz*dz;
    DATA_TYPE cz1 = dcoeff*dt/dzsq;
    DATA_TYPE cz2 = -2.0f*dcoeff*dt/dzsq;

    for (int k=startk;(k < stopk) && (k < nk-1);k++)
    {
      DATA_TYPE layer_frac = layer_step_frac[k];
 //     DATA_TYPE cx2_l = cx2 - layer_frac;
      DATA_TYPE cx1_l = cx1 + layer_frac;
            
//      DATA_TYPE c_cc = cx2_l + cy2 + cz2;

		  int ndx;
		  DATA_TYPE *src_cc;
		  DATA_TYPE *dst_cc;

		  DATA_TYPE *src_y0;
		  DATA_TYPE *src_y2;
		  DATA_TYPE *src_z0;
		  DATA_TYPE *src_z2;

		  DATA_TYPE *cf;
		DATA_TYPE *be;

		DATA_TYPE new_dp;
		
		// skipping l=0 and l=nj-1 creates an 'infinite' sink boundary at the edges
        for (int l=1;l < nj-1;l++)
        {
			ndx = k*(ni*nj)+l*ni;
            src_cc = src+ndx;
            dst_cc = dst+ndx;
            src_y0 = src_cc-ni;
            src_y2 = src_cc+ni;
            src_z0 = src_cc-(ni*nj);
            src_z2 = src_cc+(ni*nj);
			be = buffer_effect+ndx;

			  cf = correction_factor+ndx;
			
            // handle y and z edges
        if ((l==0) || (l==(nj-1)))
        {
          if (l==0)
            src_y0 = zeros_ptr;
          else
            src_y2 = zeros_ptr;
        }
					
			  if ((k==0) || (k==(nk-1)))
			  {
				  if (k==0)
					  src_z0 = zeros_ptr;
				  else
					  src_z2 = zeros_ptr;
			  }

        for (int i=1;i < (ni-1);i++)
			  {
          new_dp = cx1_l*(*(src_cc+i-1))+cx1*(*(src_cc+i+1))
                           +cy1*(*(src_y2+i)+*(src_y0+i))
                           +cz1*(*(src_z2+i)+*(src_z0+i))
                           +cf[i]*(*(src_cc+i));
				dst_cc[i] = new_dp*be[i]+src_cc[i];
			}
				
        // handle extreme right and left x
        new_dp = cx1*(*(src_cc+1))
                       +cy1*(*(src_y2)+*(src_y0))
                       +cz1*(*(src_z2)+*(src_z0))
                       +cf[0]*(*src_cc);
			dst_cc[0] = new_dp*be[0]+src_cc[0];

        new_dp = cx1_l*(*(src_cc+ni-2))
                       +cy1*(*(src_y2+ni-1)+*(src_y0+ni-1))
                       +cz1*(*(src_z2+ni-1)+*(src_z0+ni-1))
                       +cf[ni-1]*(*(src_cc+ni-1));
			dst_cc[ni-1] = new_dp*be[ni-1]+src_cc[ni-1];
        }
    }
}

// this routine 'patches' the diff-eq calculation to fix it for pixels that
// are on the boundary of well wall material by adding back in the portion
// of the 2nd-derivative calculation that will be incorrectly subtracted
// by the dellsq routine
void DiffEqModel::delsq_special(DATA_TYPE *src,DATA_TYPE *dst)
{
    DATA_TYPE dxsq = dx*dx;
    DATA_TYPE cx1 = dcoeff*dt/dxsq;

    DATA_TYPE dysq = dy*dy;
    DATA_TYPE cy1 = dcoeff*dt/dysq;

    DATA_TYPE dzsq = dz*dz;
    DATA_TYPE cz1 = dcoeff*dt/dzsq;

    for (int i=0;i < nBoundary;i++)
    {
        unsigned int ndx = boundaryIndex[i];
        DATA_TYPE src_cc = src[ndx];
        DATA_TYPE *dst_cc = &dst[ndx];
        unsigned char bc = boundary_class[ndx];
		
        *dst_cc += correction_factor[ndx]*src_cc;
    }
}


void DiffEqModel::ThreadProc()
{
    int done = 0;
    
    while(!done)
    {
		WorkerInfoQueueItem qItem = workQ->GetItem();
		ThreadControlStruct *tctrl = (ThreadControlStruct *)(qItem.private_data);
		
        switch(tctrl->task)
        {
            case DiffEqStep:
                delsq(cmatrix,cmatrix_next,tctrl->kstart,tctrl->kend);
                break;
            case ExitThreads:
			default:
                done = 1;
                break;
        }
		workQ->DecrementDone();
    }
}

// do one step of the diff-equation calculation
void DiffEqModel::MultiThreadedDiffEqCalc()
{
	ThreadControlStruct tctrl[nk];
	WorkerInfoQueueItem qitem;
	
	// put all layers on the queue for processing
	int delk=2;
	for (int i=0,k=0;k < nk;i++,k+=delk)
	{
		tctrl[i].kstart = k;
		int kend = k+delk;
		if (kend > nk)
			kend = nk;
		tctrl[i].kend = kend;
		tctrl[i].task = DiffEqStep;
		
		qitem.private_data = &tctrl[i];
		workQ->PutItem(qitem);
	}
	
	// wait for them to finish
	workQ->WaitTillDone();
	
	// NOTE: delsq_special removed...this was handled much more efficiently by pre-computing the
	// correction factor and then adding it into the main diff-eq loop...
	//// now do the special boundary condition processing...which currently isn't multi-threaded
	//// (since it is relatively small compared to everything else anyway)
	////delsq_special(cmatrix,dptr);
}

void DiffEqModel::indexBoundaryConditions()
{
	int nSpecial = 0;
    DATA_TYPE dxsq = dx*dx;
    DATA_TYPE cx1 = dcoeff*dt/dxsq;
    DATA_TYPE cx2 = -2.0f*dcoeff*dt/dxsq;

    DATA_TYPE dysq = dy*dy;
    DATA_TYPE cy1 = dcoeff*dt/dysq;
    DATA_TYPE cy2 = -2.0f*dcoeff*dt/dysq;

    DATA_TYPE dzsq = dz*dz;
    DATA_TYPE cz1 = dcoeff*dt/dzsq;
    DATA_TYPE cz2 = -2.0f*dcoeff*dt/dzsq;
	
	// have to processing the boundary_class data twice...once to count how many need special
	// processing and once again to store the indicies of the special elements
	for (int i=0;i < (ni*nj*nk);i++)
		if (boundary_class[i] & ANY_BOUNDARY)
			nSpecial++;

	boundaryIndex = new int[nSpecial];
	nSpecial = 0;
	for (int k=0;k < nk;k++)
	{
		DATA_TYPE layer_frac = layer_step_frac[k];
		DATA_TYPE cx2_l = cx2 - layer_frac;
		DATA_TYPE c_cc = cx2_l + cy2 + cz2;

		for (int i=0;i < (ni*nj);i++)
		{
			int ndx = i+k*ni*nj;
			
			if (boundary_class[ndx] & ANY_BOUNDARY)
			{
				boundaryIndex[nSpecial++] = i;
			}

			unsigned char bc = boundary_class[ndx];
			DATA_TYPE cf = c_cc;
			if (bc&PX_BOUND)
				cf += cx1;
			if (bc&MX_BOUND)
				cf += cx1;
			if (bc&PY_BOUND)
				cf += cy1;
			if (bc&MY_BOUND)
				cf += cy1;
			if (bc&PZ_BOUND)
				cf += cz1;
			if (bc&MZ_BOUND)
				cf += cz1;
			
			correction_factor[ndx] = cf;
		}
	}
	
	nBoundary = nSpecial;
}

// starts up threads and gives them a pointer to the workQ
void DiffEqModel::createThreads()
{
	threadInfo = new pthread_t[NUM_THREADS];
	
	for (int i=0;i < NUM_THREADS;i++)
		pthread_create(&threadInfo[i],NULL,&(DiffEqModel::ThreadProc_entry),this);
}

// starts processing threads, allocates memory, and indexes boundary conditions
void DiffEqModel::initializeModel()
{
	// create a matrix that holds the right-hand-side of the diff equation computation
	assert(!posix_memalign((void **)&cmatrix_next,32,sizeof(DATA_TYPE[ni*nj*nk])));
	memset(cmatrix_next,0,sizeof(DATA_TYPE[ni*nj*nk]));

	// find the boundary elements that will require special processing and build a quick index of them
	indexBoundaryConditions();
	
	// make small array of zeros needed to properly handle the edges of the cmatrix
	assert(!posix_memalign((void **)&zeros_ptr,32,sizeof(DATA_TYPE[ni])));
	memset(zeros_ptr,0,sizeof(DATA_TYPE[ni]));

	// create a matrix that holds the previous right-hand-side of the diff equation computation
	assert(!posix_memalign((void **)&dptr_last,32,sizeof(DATA_TYPE[ni*nj*nk])));

	assert(!posix_memalign((void **)&ccl_ptr,32,sizeof(DATA_TYPE[ni])));
	memset(ccl_ptr,0,sizeof(DATA_TYPE[ni]));
	assert(!posix_memalign((void **)&ccr_ptr,32,sizeof(DATA_TYPE[ni])));
	memset(ccr_ptr,0,sizeof(DATA_TYPE[ni]));


#ifndef USE_CUDA
	// might as well make this big enough to have one entry for each layer of the simulation...
	// should result in a minimal amount of context-switching during processing
	workQ = new WorkerInfoQueue(nk);
	
	createThreads();
#else
  cudaModel = new DelsqCUDA(ni,nj,nk,incorp_inject_cnt);
#endif
}

// runs the model until the stopTime.  Make be called multiple times to sucessively simulate further time points
int DiffEqModel::runModel(double endTime)
{
  int ncalls = 0;
#ifndef USE_CUDA

	while(simTime < endTime)
	{
		// keep track of how many iterations we performed
		ncalls++;
		
		simTime += dt;
		
		// calculate the right-hand-side of the differential equation using available threads
		MultiThreadedDiffEqCalc();

		DATA_TYPE *tmp = cmatrix;
		cmatrix = cmatrix_next;
		cmatrix_next = tmp;

		if (pwell)
		{
			AddSignalToWell(pwell,dt*GetIncorpFlux(simTime)*16.9282f);
		}
			
		//DynamicDNTPBoundaryCondition(simTime);
	}
#else

  cudaModel->setParams(dx,dy,dz,dcoeff,dt);
  cudaModel->setInput(cmatrix,buffer_effect,correction_factor, layer_step_frac, index_array, weight_array);
  cudaModel->copyIn();

	while(simTime < endTime)
	{
		// keep track of how many iterations we performed
		ncalls++;
		simTime += dt;
		// calculate the right-hand-side of the differential equation using available threads
		cudaModel->DoWork();
		cudaModel->DoIncorp(dt*GetIncorpFlux(simTime)*16.9282f);
	}
  cudaModel->setOutput(cmatrix);
  cudaModel->copyOut();

#endif

	
	return(ncalls);
}

// shuts down the processing threads.  deallocates memory.
void DiffEqModel::shutdownModel()
{
	// shutdown all the processing threads
#ifndef USE_CUDA
	ThreadControlStruct tctrl;
	WorkerInfoQueueItem qitem;
	
	tctrl.task = ExitThreads;
	qitem.private_data = (void *)&tctrl;
	
	for (int i=0;i < NUM_THREADS;i++)
		workQ->PutItem(qitem);
		
	workQ->WaitTillDone();
	delete [] threadInfo;
	delete workQ;
#else
  delete cudaModel;
#endif

	delete [] boundaryIndex;
	free(cmatrix_next);
	free(dptr_last);

	free(zeros_ptr);
	free(ccl_ptr);
	free(ccr_ptr);
}


