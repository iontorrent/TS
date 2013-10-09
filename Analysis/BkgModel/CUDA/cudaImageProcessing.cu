
#include "cudaImageProcessing.h"
#include "cuda_runtime.h"
#include "cuda_error.h"
#include "dumper.h"
#include "Mask.h"

RawImage AllocAndCopyRawToDevice(const RawImage * raw, int fnum);
void FreeDeviceRaw(RawImage *draw);

__global__ void GenerateAllBeadTrace_k (FG_BUFFER_TYPE* fg_buffer, int * ptrx, int * ptry, RawImage raw, int* frames_per_point, float* t0_map, int numLBeads, int numFrames,  int regCol, int regRow, int regW);

__device__
void LoadImgWOffset( RawImage raw, 
                    FG_BUFFER_TYPE *fgptr, 
                    int * compFrms, 
                    int nfrms,
                    int numLBeads, 
                    int l_coord, 
                    float t0Shift);


template<class T> 
__global__ void transposeData_k(T *dest, T *source, int width, int height);



RawImage AllocAndCopyRawToDevice(const RawImage * raw, int fnum)
{

  (void) fnum;

  RawImage draw;
//  static int copyatflow = 0;
  draw = *raw;
//  if(draw.image == NULL){ //allocate
    cudaMalloc( &draw.image, (size_t)(sizeof(short) *draw.rows*draw.cols*draw.frames));
    cudaMalloc( &draw.interpolatedFrames, (size_t)(sizeof(int) * draw.uncompFrames));
    cudaMalloc( &draw.interpolatedMult, (size_t)(sizeof(float)*draw.uncompFrames));
    cudaMalloc( &draw.interpolatedDiv ,(size_t)(sizeof(float)*draw.uncompFrames));
    CUDA_ERROR_CHECK();
//    copyatflow = fnum;
//  }

//  if(copyatflow == fnum ){
    cudaMemcpy(draw.image, raw->image, sizeof(short) *draw.rows*draw.cols*draw.frames, cudaMemcpyHostToDevice);
    cudaMemcpy(draw.interpolatedFrames, raw->interpolatedFrames, sizeof(int) * draw.uncompFrames, cudaMemcpyHostToDevice);
    cudaMemcpy(draw.interpolatedMult, raw->interpolatedMult, sizeof(float)*draw.uncompFrames, cudaMemcpyHostToDevice);
    cudaMemcpy(draw.interpolatedDiv, raw->interpolatedDiv, sizeof(float)*draw.uncompFrames, cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK();
//    copyatflow = fnum + 1;
//  }

//  RawImage r;

//  r= *raw;
//  r.image=draw.image;
//  r.interpolatedFrames = draw.interpolatedFrames;
//  r.interpolatedMult = draw.interpolatedMult;
//  r.interpolatedDiv =  draw.interpolatedDiv;

  return draw;

}

void FreeDeviceRaw(RawImage *draw)
{

  if(draw->image != NULL) cudaFree(draw->image); draw->image=NULL;
  if(draw->interpolatedFrames != NULL) cudaFree(draw->interpolatedFrames); draw->interpolatedFrames=NULL;
  if(draw->interpolatedMult != NULL) cudaFree(draw->interpolatedMult); draw->interpolatedMult=NULL;
  if(draw->interpolatedDiv != NULL) cudaFree(draw->interpolatedDiv); draw->interpolatedDiv=NULL;

}


void GenerateAllBeadTrace_GPU (BkgTrace * Bkgtr, Region *region, BeadTracker &my_beads, Image *img,  int iFlowBuffer)
{
  // setup
  int numFrames=Bkgtr->time_cp->npts();
  int numLBeads = my_beads.numLBeads;

  int paddedN = ((numLBeads+31)/32)*32;

  const RawImage *raw = img->GetImage();

  //device pointers
  int * dFramesPerPoint;
  void * d_beadparams = NULL;
  float * d_beadparamsT = NULL;
  FG_BUFFER_TYPE *dFgBuffer;
  float * dT0Map = NULL;
  RawImage dRaw =  AllocAndCopyRawToDevice(raw,iFlowBuffer);

  // malloc
  cudaMalloc(&dFramesPerPoint,sizeof(int)*Bkgtr->time_cp->frames_per_point.size());
  cudaMalloc(&dT0Map,sizeof(float)*Bkgtr->t0_map.size());
  cudaMalloc(&d_beadparams,sizeof(bead_params)*paddedN);
  cudaMalloc(&d_beadparamsT,sizeof(bead_params)*paddedN);
  cudaMalloc(&dFgBuffer,sizeof(FG_BUFFER_TYPE)*paddedN*numFrames);
  CUDA_ERROR_CHECK();


  // copy to device
  cudaMemcpy (dFramesPerPoint, &Bkgtr->time_cp->frames_per_point[0], sizeof(int)*Bkgtr->time_cp->frames_per_point.size(), cudaMemcpyHostToDevice);
  cudaMemcpy (dT0Map, &Bkgtr->t0_map[0], sizeof(int)*Bkgtr->t0_map.size(), cudaMemcpyHostToDevice);
  
//  cout << "t0_map size: " << Bkgtr->t0_map.size()  << " numLBeads: " << numLBeads <<" " << region->w << "x" << region->h <<  endl;
  //copy in and transpose bead params //TODO replace by bead mask to do whole blockei
  cudaMemcpy (d_beadparams, &my_beads.params_nn[0], sizeof(bead_params)*numLBeads, cudaMemcpyHostToDevice);
 CUDA_ERROR_CHECK(); 

  dim3 block;
  dim3 grid;

  // transpose input data
  int StructLength = (sizeof(bead_params)/sizeof(float));
  block.x=32;
  block.y=32;
  grid.x = (StructLength + block.x-1)/block.x  ;
  grid.y = (paddedN+block.y-1)/block.y;
  transposeData_k<float><<< grid, block >>>( d_beadparamsT ,(float*)d_beadparams, StructLength, paddedN );
 CUDA_ERROR_CHECK(); 

  int * dPtrX = ((int*)d_beadparamsT) + (6 + 2*NUMFB + NUM_DM_PCA) * paddedN; // offset of first x in transposed beadParams
  int * dPtrY = dPtrX + paddedN;

//  int*check  = (int*)malloc(sizeof(int)*numLBeads);
//  cudaMemcpy(check, dPtrY, sizeof(int)*numLBeads, cudaMemcpyDeviceToHost);
//  for(int i=0; i < numLBeads; i++) if( my_beads.params_nn[i].y != check[i] ) cout << i << " " << my_beads.params_nn[i].y << " " << check[i] << endl;

 
  block.x=128;
  block.y=1;
  grid.x = (numLBeads+block.x-1)/block.x;
  grid.y = 1;
  int smem = block.x*block.y*sizeof(float);

 GenerateAllBeadTrace_k <<< grid, block, smem >>>(dFgBuffer, dPtrX, dPtrY, dRaw, dFramesPerPoint , dT0Map,  numLBeads, numFrames, region->col, region->row, region->w);
 CUDA_ERROR_CHECK(); 


 

  block.x=32;
  block.y=32;
  grid.x = (numLBeads + block.x-1)/block.x;
  grid.y = (numFrames+block.y-1)/block.y;

  transposeData_k<FG_BUFFER_TYPE><<< grid, block >>>((FG_BUFFER_TYPE*)dRaw.image, dFgBuffer, paddedN , numFrames);
 
  CUDA_ERROR_CHECK(); 

//  Bkgtr->KeepEmptyScale(region, my_beads,img, iFlowBuffer);  // can overlap with kernel eecution

 

 FG_BUFFER_TYPE* tmpfg = (FG_BUFFER_TYPE*)malloc(sizeof(FG_BUFFER_TYPE)*numLBeads*numFrames*NUMFB);

  //copy back fg_buffer
  cudaMemcpy2D(   (void*)&tmpfg[numFrames*iFlowBuffer], // (void*)&Bkgtr->fg_buffers[numFrames*iFlowBuffer], 
//  cudaMemcpy2D(   (void*)&Bkgtr->fg_buffers[numFrames*iFlowBuffer], 
                numFrames*NUMFB*sizeof(FG_BUFFER_TYPE), 
                (void*)dRaw.image, 
                sizeof(FG_BUFFER_TYPE)*numFrames, 
                sizeof(FG_BUFFER_TYPE)*numFrames, 
                numLBeads, 
                cudaMemcpyDeviceToHost); 
 CUDA_ERROR_CHECK(); 

 static DumpBuffer* dbGPU[50] = {NULL};   
 static DumpBuffer* dbCPU[50] = {NULL};   
 static int count =0;
 static DumpFile* dfGPU = NULL;
 static DumpFile* dfCPU =  NULL;

  if(region->col == 0 && region->row == 0 ){
//  if( iFlowBuffer == 1 ){
  cout << "BLAAAA " << __LINE__ << endl;
    int me = count++;
    char buffername[100]; 
    
    sprintf(buffername, "%s_c%d_r%d_f%d", "fgBuffer", region->col, region->row, iFlowBuffer);
    

    dbGPU[me] = new DumpWrapper<short>((size_t)(numLBeads*numFrames*sizeof(FG_BUFFER_TYPE)), buffername);
    dbCPU[me] = new DumpWrapper<short>((size_t)(numLBeads*numFrames*sizeof(FG_BUFFER_TYPE)), buffername);

    for(int n=0; n<numLBeads; n++){
      dbGPU[me]->addData(&tmpfg[numLBeads*numFrames*iFlowBuffer + n*NUMFB*numFrames], (numFrames*sizeof(FG_BUFFER_TYPE)));
      dbCPU[me]->addData(&Bkgtr->fg_buffers[numLBeads*numFrames*iFlowBuffer + n*NUMFB*numFrames], (numFrames*sizeof(FG_BUFFER_TYPE)));
    }
    
    if(dfGPU == NULL){
      dfGPU = new DumpFile();
    }
    dfGPU->addBuffer(dbGPU[me]);

    if(dfCPU == NULL){
      dfCPU = new DumpFile();
    }
    dfCPU->addBuffer(dbCPU[me]);
  
    if(region->col == 0 && region->row == 0 && iFlowBuffer == 10){
      dfCPU->writeToFile("fgflow1CPU.dat");
      dfGPU->writeToFile("fgflow1GPU.dat");
      cout << "BLAAAA " << __LINE__ << endl;

    }


  }


  cudaFree(dFramesPerPoint);
  cudaFree(dT0Map);
  cudaFree(d_beadparams);
  cudaFree(d_beadparamsT);
  cudaFree(dFgBuffer);
  FreeDeviceRaw(&dRaw);
 CUDA_ERROR_CHECK(); 

}


template<class T>
__global__ void transposeData_k(T *dest, T *source, int width, int height)
{
  __shared__ T tile[32][32+1];

  int xIndexIn = blockIdx.x * 32 + threadIdx.x;
  int yIndexIn = blockIdx.y * 32 + threadIdx.y;
  
    
  int Iindex = xIndexIn + (yIndexIn)*width;

  int xIndexOut = blockIdx.y * 32 + threadIdx.x;
  int yIndexOut = blockIdx.x * 32 + threadIdx.y;
  
  int Oindex = xIndexOut + (yIndexOut)*height;

  if(xIndexIn < width && yIndexIn < height) tile[threadIdx.y][threadIdx.x] = source[Iindex];

  
   __syncthreads();
  
  if(xIndexOut < height && yIndexOut < width) dest[Oindex] = tile[threadIdx.x][threadIdx.y];
}


/*
constant memory:
 CP = region_params
 CP.Region Region

transposed bead_params 
 x
 y

RawImage

output
fg_buffer

*/


// fgbuffer only for one flow
__global__
void GenerateAllBeadTrace_k (FG_BUFFER_TYPE* fg_buffer, int * ptrx, int * ptry, RawImage raw, int* frames_per_point, float* t0_map, int numLBeads, int numFrames,  int regCol, int regRow, int regW)
{
    // these are used by both the background and live-bead
  const int avgover = 8;  // 2^n < warp_size to remove the need for sync
	int l_coord;
  int rx,rxh,ry,ryh;
  FG_BUFFER_TYPE *fgPtr
  extern __shared__ float localT0[];  // one float per thread

  int ibd = threadIdx.x + blockIdx.x*blockDim.x;  
  int avg_offset = (threadIdx.x/avgover)*avgover;

 float* smlocalT0 = localT0 + threadIdx.x;
 float* avgT0 = localT0 + (threadIdx.x/avgover)*avgover;
  //CP+=sid; // update to point to const memory struct for this stream

  if(ibd >= numLBeads) ibd = numLBeads-1;  // 

  numLBeads = ((numLBeads+31)/32)*32;  //needs to be padded to 128bytes if working with transposed data 
 
  rx= ptrx[ibd];
  ry= ptry[ibd];
//  if(rx >= regW) printf("%d rx: %d >= %d\n", ibd, rx, regW);
//  if(ry >= regW) printf("%d ry: %d  >= %d\n", ibd, ry, regW);

  *smlocalT0 = t0_map[rx+ry*regW];
  
  rxh = rx + regCol; //const
  ryh = ry + regRow; //const

  l_coord = ryh*raw.cols+rxh;  // 

  //fgPtr = &fg_buffers[bead_flow_t*nbdx+npts*iFlowBuffer];
  //fgPtr = fg_buffers + (numLBeads*npts*iFlowBuffer+ibd);
  fgPtr = fg_buffer + ibd;


  if(threadIdx.x%avgover == 0){
    for(int o = 1; o < avgover; o++){
      *smlocalT0 += smlocalT0[o]; // add up in shared mem 
    }
    *smlocalT0 /= avgover; // calc shared mem avg for warp or block
  }

    	//LoadImgWOffset(raw, fgPtr, time_cp->frames_per_point, npts, l_coord, localT0);
  LoadImgWOffset(raw, fgPtr, frames_per_point, numFrames, numLBeads, l_coord, *avgT0);
      //KeepEmptyScale(region, my_beads,img, iFlowBuffer);
//      KeepEmptyScale(bead_scale_by_flow, rx, ry, numLBeads, regCol, regRow, regW, regH, rawCols, smooth_max_amplitude, iFlowBuffer);

}


/*
INPUTS:

raw:
  rows
  cols
  int uncompFrames
  int *interpolatedFrames  //sizeof(int) * uncompFrames
  int frameStride
  short* image   //sizeof(short) *rows*cols*frames
  float *interpolatedMult //sizeof(float)*uncompFrames
  float *interpolatedDiv  // "    "    "
*/

__device__
void LoadImgWOffset( RawImage raw, 
                    FG_BUFFER_TYPE *fgptr, 
                    int * compFrms, 
                    int nfrms,
                    int numLBeads, 
                    int l_coord, 
                    float t0Shift/*, int print*/)
{
	int t0ShiftWhole;
	float multT;
	float t0ShiftFrac;
	int my_frame = 0,compFrm,curFrms,curCompFrms;

	float prev;
	float next;
	float tmpAdder;

	int interf,lastInterf=-1;
	FG_BUFFER_TYPE lastVal;
	int f_coord;

	if(t0Shift < 0)
		t0Shift = 0;
	if(t0Shift > (raw.uncompFrames-2))
		t0Shift = (raw.uncompFrames-2);

	t0ShiftWhole=(int)t0Shift;
	t0ShiftFrac = t0Shift - (float)t0ShiftWhole;

	// first, skip t0ShiftWhole input frames
	my_frame = raw.interpolatedFrames[t0ShiftWhole]-1;
	compFrm = 0;
	tmpAdder=0.0f;
	curFrms=0;
	curCompFrms=compFrms[compFrm];


	while ((my_frame < raw.uncompFrames) && (compFrm < nfrms))
	{
	  interf= raw.interpolatedFrames[my_frame];

	  if(interf != lastInterf)
	  {
			  f_coord = l_coord+raw.frameStride*interf;
			  next = raw.image[f_coord];
		  if(interf > 0)
		  {
				  prev = raw.image[f_coord - raw.frameStride];
		  }
		  else
		  {
			  prev = next;
		  }
	  }

	  // interpolate
	  multT= raw.interpolatedMult[my_frame] - (t0ShiftFrac/raw.interpolatedDiv[my_frame]);
	  tmpAdder += ( (prev)-(next) ) * (multT) + (next);

	  if(++curFrms >= curCompFrms)
	  {
		  tmpAdder /= curCompFrms;
			fgptr[compFrm*numLBeads] = (FG_BUFFER_TYPE)(tmpAdder);
		  compFrm++;
		  curCompFrms = compFrms[compFrm];
		  curFrms=0;
      tmpAdder= 0.0f;
	  }
	  my_frame++;
	}
	if(compFrm > 0 && compFrm < nfrms)
	{
	  lastVal = fgptr[numLBeads*(compFrm-1)];  //TODO: keep last val in reg
		for(;compFrm < nfrms;compFrm++)
				fgptr[numLBeads*compFrm] =  lastVal;
	}
}

/*
INPUTS NEEDED:
region:
  col
  row
  w
  h
img: 
  smooth_max_amplitude //sizeof(float)*rawCols*rawRows 
raw:
  cols

numLBeads

beadParams.x
beadBarams.y

outputs:
 bead_scale_by_flow // sizeof(float)*numLBeads*NUMFB 

*/
/*
void KeepEmptyScale(float* bead_scale_by_flow, 
                    int x,
                    int y,
                    int numLBeads, 
                    int regCol, 
                    int regRow, 
                    int regW, 
                    int regH, 
                    int rawCols, 
                    float ewamp,  // calc outside
                    float* smooth_max_amplitude, 
                    int iFlowBuffer)
{
//  float ewamp =1.0f;
  bool ewscale_correct = (smooth_max_amplitude!=NULL);


  for (int nbd = 0;nbd < numLBeads;nbd++) // is this the right iterator here?
  {
    int rx = params[nbd].x;  // transpose
    int ry = params[nbd].y;  // transpose

    if (ewscale_correct)
    {
      bead_scale_by_flow[nbd*NUMFB+iFlowBuffer] = smooth_max_amplitude[(ry+regRow)*rawCols+(rx+regCol)]/ ewamp;
    }else{
      bead_scale_by_flow[nbd*NUMFB+iFlowBuffer] = 1.0f;  // shouldn't even allocate if we're not doing image rescaling
    }
  }
}
*/

/*
INPUTS NEEDED:
region:
  col
  row
  w
  h
img: 
  smooth_max_amplitude //sizeof(float)*rawCols*rawRows 
raw:
  cols
*/
/*
float GetEmptyWellAmplitudeRegionAverage (  int regCol, 
                                            int regRow, 
                                            int regW, 
                                            int regH, 
                                            int rawCols, 
                                            float * smooth_max_amplitude )
{
  float ew_avg = 0.0f;
  int ew_sum = 0;

  //assert ( smooth_max_amplitude != NULL ); //already done outside

  // calculate the average within the region
  for ( int ay=regRow;ay < ( regRow+regH );ay++ )
  {
    for ( int ax=regCol;ax < ( regCol+regH );ax++ )
    {
      if ( smooth_max_amplitude[ax+ay*rawCols] > 0.0f )
      {
        ew_avg += smooth_max_amplitude[ax+ay*rawCols];
        ew_sum++;
      }
    }
  }
  ew_avg /= ew_sum;

  return ( ew_avg );
}
*/

/*
__device__
bool Match(unsigned short * mask,  int n, MaskType type) 
{
		if ( n < 0 || n >= ( w*h ) )
			return false;

		return ( ( mask[n] & type ? true : false ) );
}


__device__ 
bool Match(unsigned short * mask,  int x, int y, int w, MaskType type) 
{
  int n = y*w+x;
  return Match(mask, n, type);
}



#define IMAGEBLOCK_X 1332
#define IMAGEBLOCK_Y 1288
#define NUMIMAGEBLOCKS_X 6
#define NUMIMAGEBLOCKS_Y 6
#define WARPSIZE 32


//TODO:
// fgbuffers are over allocated to the size of the raw image. 
// fg buffer offsets for each region are:  sizeof(FG_BUFFER_TYPE)*216*224*numframes

// t0_map is a array of arrays for each region
  

__global__
void GenerateAllBeadTraceWholeImage_k ( FG_BUFFER_TYPE* fg_buffer, 
                                        RawImage raw, 
                                        int* frames_per_point, 
                                        Region * reg, 
                                        float* t0_map[], 
                                        int  *numLBeadsPerRegion, 
                                        int *numFramesperRegion, 
                                        unsigned short * mask 
                                      )
{
    // these are used by both the background and live-bead
  const int avgover = 8;  // 2^n < warp_size to remove the need for sync
	int l_coord;
  int rx,rxh,ry,ryh;
  FG_BUFFER_TYPE *fgPtr;
  extern __shared__ float localT0[];  // one float per thread
  extern __shared__ int warpMask[];

  warpMask += blockDim.x + threadIdx.x/WARPSIZE;  // 1D block offset after localT0; 
  if(threadIdx.x%WARPSIZE == 0) *warpMask = 0; // mask shared between threads of one warp set to 0;
  
  int ibd = threadIdx.x + blockIdx.x*blockDim.x;
  
  int avg_offset = (threadIdx.x/avgover)*avgover;

 float* smlocalT0 = localT0 + threadIdx.x;
 float* avgT0 = localT0 + (threadIdx.x/avgover)*avgover;
 //CP+=sid; // update to point to const memory struct for this stream


  // TODO: so far hard coded  6x6 block with 36 
  int regY = blockIdx.x/NUMIMAGEBLOCKS_X;
  int regX = blockIdx.x%NUMIMAGEBLOCKS_X; 

  if(ibd >= numLBeads) ibd = numLBeads-1;  // 

  numLBeads = ((numLBeads+31)/32)*32;  //needs to be padded to 128bytes if working with transposed data 
 
  rx= ptrx[ibd];
  ry= ptry[ibd];
//  if(rx >= regW) printf("%d rx: %d >= %d\n", ibd, rx, regW);
//  if(ry >= regW) printf("%d ry: %d  >= %d\n", ibd, ry, regW);

  *smlocalT0 = t0_map[rx+ry*regW];
  
  rxh = rx + regCol; //const
  ryh = ry + regRow; //const

  l_coord = ryh*raw.cols+rxh;  // 

  //fgPtr = &fg_buffers[bead_flow_t*nbdx+npts*iFlowBuffer];
  //fgPtr = fg_buffers + (numLBeads*npts*iFlowBuffer+ibd);
  fgPtr = fg_buffer + ibd;


  if(threadIdx.x%avgover == 0){
    for(int o = 1; o < avgover; o++){
      *smlocalT0 += smlocalT0[o]; // add up in shared mem 
    }
    *smlocalT0 /= avgover; // calc shared mem avg for warp or block
  }

    	//LoadImgWOffset(raw, fgPtr, time_cp->frames_per_point, npts, l_coord, localT0);
  LoadImgWOffset(raw, fgPtr, frames_per_point, numFrames, numLBeads, l_coord, *avgT0);
      //KeepEmptyScale(region, my_beads,img, iFlowBuffer);
//      KeepEmptyScale(bead_scale_by_flow, rx, ry, numLBeads, regCol, regRow, regW, regH, rawCols, smooth_max_amplitude, iFlowBuffer);

}

*/



