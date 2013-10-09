#include <stdlib.h>
#include "DiffEqModel.h"
#include <assert.h>
#include "DelsqCUDA.h"


//constan param symbol definition
__constant__ ConstStruct CP;


// kernel prototype
__global__ 
void delsq_kernel(  DATA_TYPE *src, 
                    DATA_TYPE *dst, 
                    DATA_TYPE *layer_step_frac, 
                    DATA_TYPE *buffer_effect, 
                    DATA_TYPE *correction_factor 
                  );

__global__
void incorp_sig_kernel( DATA_TYPE *cmatrix,
						size_t *indicies,
						DATA_TYPE *weights,
						DATA_TYPE incorp_signal
					   );


DelsqCUDA::DelsqCUDA(size_t x, size_t y, size_t z, int inject_cnt, int deviceId)
{
  Hcmatrix = NULL;
  Hdst = NULL; 
  Hlayer_step_frac = NULL;
  Hbuffer_effect = NULL; 
  Hcorrection_factor = NULL;
  Hindex_array = NULL;
  Hweight_array = NULL;

  Dsrc = NULL;
  Ddst = NULL; 
  Dlayer_step_frac = NULL;
  Dbuffer_effect = NULL; 
  Dcorrection_factor = NULL;
  Dindex_array = NULL;
  Dweight_array = NULL;

  dOutput = NULL;

  cParams.x = x;
  cParams.y = y;
  cParams.z = z;

  devId = deviceId;
  my_inject_cnt = inject_cnt;
  cParams.inject_cnt = my_inject_cnt;

  cudaSetDevice(devId);

  createCudaBuffers();
} 

DelsqCUDA::~DelsqCUDA()
{
  destroyCudaBuffers(); 
}

void DelsqCUDA::createCudaBuffers()
{

  cudaMalloc( &Dsrc, size()); CUDA_ERROR_CHECK(); 
  cudaMalloc( &Ddst, size()); CUDA_ERROR_CHECK(); 
  cudaMalloc( &Dbuffer_effect, size()); CUDA_ERROR_CHECK(); 
  cudaMalloc( &Dcorrection_factor, size()); CUDA_ERROR_CHECK(); 
  cudaMalloc( &Dlayer_step_frac, sizeZ()); CUDA_ERROR_CHECK(); 
  cudaMalloc( &Dindex_array, my_inject_cnt*sizeof(size_t)); CUDA_ERROR_CHECK(); 
  cudaMalloc( &Dweight_array, my_inject_cnt*sizeof(DATA_TYPE)); CUDA_ERROR_CHECK(); 
}


void DelsqCUDA::destroyCudaBuffers()
{

  if(Dsrc != NULL) cudaFree(Dsrc); CUDA_ERROR_CHECK(); 
  if(Ddst != NULL) cudaFree(Ddst); CUDA_ERROR_CHECK(); 
  if(Dbuffer_effect != NULL) cudaFree(Dbuffer_effect); CUDA_ERROR_CHECK(); 
  if(Dcorrection_factor != NULL) cudaFree(Dcorrection_factor); CUDA_ERROR_CHECK(); 
  if(Dlayer_step_frac != NULL) cudaFree(Dlayer_step_frac); CUDA_ERROR_CHECK(); 
  if(Dindex_array != NULL) cudaFree(Dindex_array); CUDA_ERROR_CHECK(); 
  if(Dweight_array != NULL) cudaFree(Dweight_array); CUDA_ERROR_CHECK(); 
}





void DelsqCUDA::setParams( DATA_TYPE dx, DATA_TYPE dy,DATA_TYPE dz,DATA_TYPE dcoeff, DATA_TYPE dt)
{

    DATA_TYPE dsq = dx*dx;

    cParams.cx1 = dcoeff*dt/dsq;
    cParams.cx2 = -2.0f*dcoeff*dt/dsq;

    dsq = dy*dy;
    cParams.cy1 = dcoeff*dt/dsq;
    cParams.cy2 = -2.0f*dcoeff*dt/dsq;

    dsq = dz*dz;
    cParams.cz1 = dcoeff*dt/dsq;
    cParams.cz2 = -2.0f*dcoeff*dt/dsq;
  
    cudaMemcpyToSymbol( CP, &cParams, sizeof(ConstStruct)); CUDA_ERROR_CHECK();

}


void DelsqCUDA::setInput( DATA_TYPE * cmatrix, 
                          DATA_TYPE *buffer_effect, 
                          DATA_TYPE *correction_factor, 
                          DATA_TYPE *layer_step_frac,
						  size_t *index_array,
						  DATA_TYPE *weight_array)
{

  assert(cmatrix); Hcmatrix = cmatrix;
  assert(buffer_effect); Hbuffer_effect = buffer_effect;
  assert(correction_factor); Hcorrection_factor = correction_factor;

  assert(layer_step_frac); Hlayer_step_frac = layer_step_frac;
  Hindex_array = index_array;
  Hweight_array = weight_array;
}


void DelsqCUDA::copyIn()
{

  assert(Hcmatrix); assert(Dsrc);
  cudaMemcpy( Dsrc, Hcmatrix, size(), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
  assert(Hbuffer_effect); assert(Dbuffer_effect);
  cudaMemcpy( Dbuffer_effect, Hbuffer_effect, size(), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
  assert(Hcorrection_factor); assert(Dcorrection_factor);
  cudaMemcpy( Dcorrection_factor, Hcorrection_factor, size(), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
  assert( Hlayer_step_frac); assert(Dlayer_step_frac);
  cudaMemcpy( Dlayer_step_frac, Hlayer_step_frac, sizeZ(), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 

  if (Hindex_array != NULL)
  {
	assert( Dindex_array );
	cudaMemcpy( Dindex_array, Hindex_array, my_inject_cnt*sizeof(size_t), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
  }

  if (Hweight_array != NULL)
  {
	assert( Dweight_array );
	cudaMemcpy( Dweight_array, Hweight_array, my_inject_cnt*sizeof(DATA_TYPE), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
  }
}


void DelsqCUDA::setOutput( DATA_TYPE * dst )
{
  assert(dst); Hdst = dst;
}

void DelsqCUDA::copyOut()
{
  assert(Hdst);
  assert(dOutput);
  cudaMemcpy( Hdst, dOutput , size(), cudaMemcpyDeviceToHost); CUDA_ERROR_CHECK(); 
}



void DelsqCUDA::DoWork()
{

  dim3 block;
  dim3 grid;
  
  block.x = 64;
  block.y = 2;
  block.z = 1;

  grid.x = (getX() + block.x - 1)/block.x;
  grid.y = (getY() + block.y - 1)/block.y;
  grid.z = (getZ() + block.z - 1)/block.z;

 
  delsq_kernel<<< grid, block >>>(Dsrc, Ddst, Dlayer_step_frac, Dbuffer_effect, Dcorrection_factor); CUDA_ERROR_CHECK();

  //exchange device poionters for next iteration   
  dOutput = Ddst;
  Ddst = Dsrc; 
  Dsrc = dOutput;

}

// 
// cmatrix, dsptr, layer_step_frac, buffer_effect, correction_factor
__global__ void delsq_kernel( DATA_TYPE *src, DATA_TYPE *dst, DATA_TYPE *layer_step_frac, DATA_TYPE *buffer_effect, DATA_TYPE * correction_factor )
{

    int tidx = blockIdx.x * blockDim.x + threadIdx.x; if(tidx >= CP.x) return;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y; if(tidy >= CP.y) return;
    int tidz = blockIdx.z * blockDim.z + threadIdx.z; if(tidz >= CP.z) return;

    DATA_TYPE layer_frac = layer_step_frac[tidz]; // layer_step_frac [nk]
    //DATA_TYPE cx2_l = CP.cx2 - layer_frac;
    DATA_TYPE cx1_l = CP.cx1 + layer_frac;
            
    //DATA_TYPE c_cc = cx2_l + CP.cy2 + CP.cz2;

	  int ndx;

    DATA_TYPE *src_cc;
		DATA_TYPE *dst_cc;

		DATA_TYPE src_y0;
		DATA_TYPE src_y2;
		DATA_TYPE src_z0;
		DATA_TYPE src_z2;

		DATA_TYPE cf;
		DATA_TYPE be;

		DATA_TYPE new_dp;
		
		ndx = tidz*(CP.x*CP.y)+tidy*CP.x+tidx;

    src_cc = src+ndx; // cmatrix [ni*nj*nk]
    dst_cc = dst+ndx; // dsptr  [ni*nj*nk]

    src_y0 = (tidy == 0)?(0):(*(src_cc-CP.x));
    src_y2 = (tidy== CP.y-1)?(0):(*(src_cc+CP.x));

    src_z0 = (tidz==0)?(0):(*(src_cc-(CP.x*CP.y)));
    src_z2 = (tidz == CP.z-1)?(0):(*(src_cc+(CP.x*CP.y)));

		be = *(buffer_effect+ndx);  // buffer_effect [ni*nj*nk]
		cf = *(correction_factor+ndx); // correction_factor  [ni*nj*nk]
			
		
    new_dp = (tidx == 0)?(0):( cx1_l  * (*(src_cc-1))); // handle left hand boundary
    new_dp += (tidx == CP.x-1)?(0):(   CP.cx1 * (*(src_cc+1))); // right hand boundary
    new_dp += CP.cy1*((src_y2)+(src_y0))
              +CP.cz1*((src_z2)+(src_z0))
              + cf*(*(src_cc));

    *dst_cc = new_dp*be+(*src_cc);
 }

void DelsqCUDA::DoIncorp( DATA_TYPE incorp_signal )
{
	int block = 64;
	int grid = (my_inject_cnt+block-1) / block;
	
	incorp_sig_kernel<<< grid, block >>>(Dsrc, Dindex_array, Dweight_array, incorp_signal); CUDA_ERROR_CHECK();
}

__global__ void incorp_sig_kernel( DATA_TYPE *cmatrix, size_t *indicies, DATA_TYPE *weights, DATA_TYPE incorp_signal )
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;  if (tidx >= CP.inject_cnt) return;
	
	size_t ndx = indicies[tidx];
	cmatrix[ndx] = cmatrix[ndx] + weights[tidx]*incorp_signal;
}




