/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>

#include "GenerateBeadTraceStream.h"
#include "GenerateBeadTraceKernels.h"
#include "SignalProcessingFitterQueue.h"
#include "SignalProcessingMasterFitter.h"
#include "cuda_error.h"
#include "Mask.h"
#include "Image.h"
#include "TimeCompression.h"
#include "Region.h"
#include "BeadTracker.h"
#include "BkgTrace.h"

using namespace std;

#define DEBUG_GENTRACE 1

map<int, BlockPersistentData> GenerateBeadTraceStream::_DevicePersistent;



BlockPersistentData::BlockPersistentData()
{
  _numOffsetsPerRow = 0;
  _imgW = 0;
  _imgH = 0;
  _regsX = 0;
  _regsY = 0;

  _hT0est = NULL;
  _hBfmask = NULL;
  _dT0est = NULL;
  _dBfmask = NULL;

  _dOffsetWarp = NULL;
  _dT0avgRegion = NULL;
  _dlBeadsRegion = NULL;

  _hlBeadsRegion = NULL;

  _init = false;
}

BlockPersistentData::~BlockPersistentData()
{
//  if(_dOffsetWarp != NULL) cudaFree(_dOffsetWarp); _dOffsetWarp = NULL; CUDA_ERROR_CHECK();
//  if(_dT0avgRegion != NULL) cudaFree(_dT0avgRegion); _dT0avgRegion = NULL; CUDA_ERROR_CHECK();
//  if(_dlBeadsRegion != NULL) cudaFree(_dlBeadsRegion); _dlBeadsRegion = NULL; CUDA_ERROR_CHECK();

//  if(_dBfmask != NULL) cudaFree(_dBfmask); _dBfmask=NULL;
//  if(_dT0est != NULL) cudaFree(_dT0est); _dT0est=NULL;
  _init = false;
}

void BlockPersistentData::Allocate( int OffsetsRow, int imgW, int imgH, int regsX, int regsY)
{


  if(AlreadyCreated()) return;


  if(_dOffsetWarp == NULL){
    _numOffsetsPerRow = OffsetsRow;
    _regsX = regsX;
    _regsY = regsY;
    _imgW = imgW;
    _imgH = imgH;

    cudaMalloc(&_dT0est,sizeof(float)*getImgSize());CUDA_ERROR_CHECK();
    cudaMalloc(&_dBfmask,sizeof(unsigned short)*getImgSize());CUDA_ERROR_CHECK();

    cudaMalloc(&_dOffsetWarp, sizeof(int)*getNumOffsets());CUDA_ERROR_CHECK();
    cudaMalloc(&_dlBeadsRegion, sizeof(int)*getNumRegions());CUDA_ERROR_CHECK();
    cudaMalloc(&_dT0avgRegion, sizeof(float)*getNumRegions());CUDA_ERROR_CHECK();

    _hlBeadsRegion = new int[getNumRegions()];
  }
}

//prepare inputs
void BlockPersistentData::PrepareInputs(BkgModelImgToTraceInfoGPU * info)
{
  if(AlreadyCreated()) return;

  //TODO prepare for async copy
  _hT0est = info->smooth_t0_est;
  _hBfmask = info->bfmask->GetMask();

}

void BlockPersistentData::CopyInputs()
{
  if(AlreadyCreated()) return;

  cudaMemcpy(_dT0est,_hT0est,(sizeof(float)*getImgSize()),cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();
  cudaMemcpy(_dBfmask, _hBfmask,(sizeof(unsigned short)*getImgSize()),cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();

}

void BlockPersistentData::CopyOutputs()
{
  if(AlreadyCreated()) return;

  cudaMemcpy(getHLBeadsRegion(), getDlBeadsRegion(), sizeof(int)*getNumRegions(), cudaMemcpyDeviceToHost);CUDA_ERROR_CHECK();


}


void BlockPersistentData::DebugPrint(int startX, int numX, int startY, int numY)
{


  int * hOffsetWarp = new int[getNumOffsets()]; //num beads per row, length == regHeight * (ceil(regMaxWidth/blockDim.x)+1)
  float * hT0avgRegion = new float[getNumRegions()];

  cudaMemcpy(hOffsetWarp, getDOffsetWarp(), sizeof(int)*getNumOffsets(), cudaMemcpyDeviceToHost);CUDA_ERROR_CHECK();

  cudaMemcpy(hT0avgRegion, getDT0avgRegion(), sizeof(int)*getNumRegions(), cudaMemcpyDeviceToHost);CUDA_ERROR_CHECK();


  numX = (numX == 0 || startX+numX > _regsX)?(_regsX):(numX);
  numY = (numY == 0 || startX+numY > _regsY)?(_regsY):(numY);

//  int offsetPerRegion = _imgH * getNumOffsetsPerRow();
  for(int ry = startY; ry < startY+numY ; ry ++){
  //  int offsetRegY =  ry*(_regsX* offsetPerRegion );
    for(int rx = startX; rx < startX+numX ; rx ++){
    //  int offsetRegXY = offsetRegY + rx*(offsetPerRegion);
      cout << "region " << rx << ":" << ry << " ";
      cout << "numLBeads: " <<  _hlBeadsRegion[ry*_regsX+rx] << " t0avg: " << hT0avgRegion[ry*_regsX+rx] << endl;
    //  for( int row = 0; row<_imgH; row++){
    //    int offsetRow = offsetRegXY + row * getNumOffsetsPerRow();
    //    cout << row << ": ";
    //    for(int o=0; o< getNumOffsetsPerRow(); o++){
    //      cout << hOffsetWarp[ offsetRow + o ] << " ";
    //    }
    //  cout << endl;
    //  }
    //  cout << endl;
    }
  }
  //cout << "done " << endl;

//  delete [] hOffsetWarp;
//  delete [] hT0avgRegion;

}



/*

void testmystuff( const unsigned short * bfmask,
              float* t0Est,
              RawImage * raw,
      // FG_BUFFER_TYPE *fg_buffers,  //int iFlowBuffer not needed since on device on one flow of fg_buffer present
              int * npts,
              int regMaxWidth,
              int regMaxHeight
)
{


  int imgWidth = raw->cols;
  int imgHeight = raw->rows;


  dim3 block(32,8);
  dim3 grid;
  int smSize = 0;

  grid.x = ((imgWidth + regMaxWidth -1)/ regMaxWidth);
  grid.y = ((imgHeight + regMaxHeight -1)/ regMaxHeight);


  int numRegs = grid.x*grid.y;


  float * dT0est = NULL;
  unsigned short * dBfmask = NULL;
  int * dOffsetWarp = NULL; //num beads per row, length == regHeight * (ceil(regMaxWidth/blockDim.x)+1)
  int * dlBeadsRegion = NULL;
  float * dT0avgRegion = NULL;
  // allocate device buffers
  int offsetsPerRow = ((regMaxWidth + block.x -1)/block.x) + 1;

  cout << "imgWidth: " << imgWidth <<" imgHeight: " << imgHeight <<" regMaxWidth: " << regMaxWidth <<" regMaxHeight: " << regMaxHeight << " offsetsPerRow: " <<  offsetsPerRow << endl;

  cudaMalloc(&dT0est,sizeof(float)*imgWidth*imgHeight);
  cudaMalloc(&dBfmask,sizeof(unsigned short)*imgHeight*imgWidth);
  cudaMalloc(&dOffsetWarp, sizeof(int)*offsetsPerRow*numRegs*regMaxHeight);
  cudaMalloc(&dlBeadsRegion, sizeof(int)*grid.x*grid.y);
  cudaMalloc(&dT0avgRegion, sizeof(float)*grid.x*grid.y);


  cudaMemcpy(dT0est,(void*)t0Est,(sizeof(float)*imgWidth*imgHeight),cudaMemcpyHostToDevice);
  cudaMemcpy(dBfmask,(void*)bfmask,(sizeof(unsigned short)*imgWidth*imgHeight),cudaMemcpyHostToDevice);
  cudaMemset(dOffsetWarp, 0,sizeof(int)*offsetsPerRow*numRegs*regMaxHeight);
  cudaMemset(dlBeadsRegion,0, sizeof(int)*grid.x*grid.y);
  cudaMemset(dT0avgRegion,0, sizeof(int)*grid.x*grid.y);

    smSize = block.x*block.y*sizeof(int);
    GenerateMetaPerWarpForBlock_k<<< gAlreadyCreatedrid, block, smSize >>>(
          dBfmask,
          dT0est,
          imgWidth,
          imgHeight,
          regMaxWidth,
          regMaxHeight,
          dOffsetWarp,
          dlBeadsRegion,
          dT0avgRegion
          );
     CUDA_ERROR_CHECK();

     int * hOffsetWarp = new int[offsetsPerRow*numRegs*regMaxHeight]; //num beads per row, length == regHeight * (ceil(regMaxWidth/blockDim.x)+1)
     int * hlBeadsRegion = new int[grid.x*grid.y];
     float * hT0avgRegion = new float[grid.x*grid.y];

     cudaMemcpy(hOffsetWarp, dOffsetWarp, sizeof(int)*offsetsPerRow*numRegs*regMaxHeight, cudaMemcpyDeviceToHost);
     cudaMemcpy(hlBeadsRegion, dlBeadsRegion, sizeof(int)*grid.x*grid.y, cudaMemcpyDeviceToHost);
     cudaMemcpy(hT0avgRegion, dT0avgRegion, sizeof(int)*grid.x*grid.y, cudaMemcpyDeviceToHost);


     for(int ry = 0; ry < grid.y ; ry ++){
       for(int rx = 0; rx < grid.x ; rx ++){
         cout << "region " << rx << " " << ry << endl;
         cout << "numLBeads: " << hlBeadsRegion[ry*grid.x+rx] << " t0avg: " << hT0avgRegion[ry*grid.x+rx] << endl;
         for( int row = 0; row<regMaxHeight; row++){
          // cout << row << ": ";
          // for(int o=0; o<offsetsPerRow; o++){
          //   cout << *hOffsetWarp << " ";
          //   hOffsetWarp++;
          // }
          // cout << endl;
          // for (int x= 0; x < imgWidth; x++)
          //         {
          //                 printf("GPU %d:%d: %f\n", row , x, t0Est[x+row*imgWidth]);
          //         }
         }

         cout << endl;

       }
     }
     cout << "done " << endl;



     int MaxNpts = 0;
     for(int i=0; i< numRegs; i++)
       MaxNpts = (npts[i]>MaxNpts)?(npts[i]):(MaxNpts);

     int fgRegSize = MaxNpts*regMaxWidth*regMaxHeight;


     //make temp fg buffer
     FG_BUFFER_TYPE ** hfgList = new FG_BUFFER_TYPE*[numRegs];
     FG_BUFFER_TYPE * hfgBase = new FG_BUFFER_TYPE[numRegs * fgRegSize];
     FG_BUFFER_TYPE ** dfgList = NULL;
     FG_BUFFER_TYPE * dfgBase = NULL;
     cudaMalloc(&dfgList,sizeof(FG_BUFFER_TYPE*)*numRegs);
     cudaMalloc(&dfgBase,sizeof(FG_BUFFER_TYPE)* numRegs * fgRegSize);


     RawImage dRaw;

     AllocAndCopyRawToDevice(&dRaw, raw);

     grid.y = imgHeight;
     smSize = block.x*block.y*sizeof(int);

     GenerateAllBeadTraceFromMeta_k<<< grid, block, smSize >>>(
        fg_buffer_list,
        dRaw,
        frames_per_point,
        regMaxWidth,
        regMaxHeight,
            dBfmask,
            dT0est,
            npts, // needs to be build outside
             //from meta data kernel:
            dOffsetWarp, //num beads per row, length == regHeight * (ceil(regMaxW/blockDim.x)+1)
            dlBeadsRegion, //numLbeads of whole region
            dT0avgRegion
             );



    FreeDeviceRaw(&dRaw);

    cudaFree(dT0est);
    cudaFree(dBfmask);
  cudaFree(dOffsetWarp);
  cudaFree(dlBeadsRegion);
  cudaFree(dT0avgRegion);

}

*/




//int GenerateBeadTraceStream::_bpb = 128;
//int GenerateBeadTraceStream::_l1type = -1;  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default


//TODO: test for best setting

/*
int GenerateBeadTraceStream::l1DefaultSetting()
{
  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default
  if(_computeVersion == 20 ) return 1;
  if(_computeVersion == 35 ) return 1;
  return 0;
}
*/

/////////////////////////////////////////////////
//GENERATE BEAD TRACES STREAM CLASS

GenerateBeadTraceStream::GenerateBeadTraceStream(streamResources * res, WorkerInfoQueueItem item ) : cudaSimpleStreamExecutionUnit(res, item)
{
  setName("GenerateBeadTraceStream");

  if(_verbose) cout << getLogHeader() << " created"  << endl;

  _info = static_cast<BkgModelImgToTraceInfoGPU *>(getJobData());

  const RawImage * raw = _info->img->GetImage();
  _draw = *raw;

  _imgWidth = _draw.cols;
  _imgHeight = _draw.rows;
  _regMaxWidth = _info->regionMaxX;
  _regMaxHeight = _info->regionMaxY;

  _regionsX = ((_imgWidth + _regMaxWidth -1)/ _regMaxWidth);
  _regionsY = ((_imgHeight + _regMaxHeight -1)/ _regMaxHeight);

  _threadBlockX = 32; // can NOT be larger than warp size (32) to guarantee sync free calculations
  _threadBlockY = 8;

  _hFgBufferRegion_Base =NULL;
  _dFgBufferRegion_Base = NULL;

  _draw.image = NULL;
  _draw.interpolatedFrames = NULL;
  _draw.interpolatedMult = NULL;
  _draw.interpolatedDiv = NULL;

  ///////////////
  //per item data

  //host pointer
  _hFgBufferRegion = NULL;
  _hFramesPerPointRegion = NULL;
  _hNptsRegion = NULL; // needs to be build outside

  _hFramesPerPointRegion_Base = NULL;

  //device pointer
  _dFgBufferRegion = NULL;
  _dFramesPerPointRegion = NULL;
  _dNptsRegion = NULL; // needs to be build outside

  _dFramesPerPointRegion_Base = NULL;
  ///////////




  _BlockPersistent = NULL;


}


GenerateBeadTraceStream::~GenerateBeadTraceStream()
{
  cleanUp();
}


void GenerateBeadTraceStream::cleanUp()
{

   if(_verbose) cout << getLogHeader() << " clean up"  << endl;


   DestroyRawDevice();

   CUDA_ERROR_CHECK();
}


// needs to be changed to use Stream Resources
void GenerateBeadTraceStream::AllocRawDevice()
{

  cudaMalloc( &_draw.image, (size_t)(sizeof(short) *_draw.rows*_draw.cols*_draw.frames));CUDA_ERROR_CHECK();
  cudaMalloc( &_draw.interpolatedFrames, (size_t)(sizeof(int) * _draw.uncompFrames));CUDA_ERROR_CHECK();
  cudaMalloc( &_draw.interpolatedMult, (size_t)(sizeof(float)*_draw.uncompFrames));CUDA_ERROR_CHECK();
  cudaMalloc( &_draw.interpolatedDiv ,(size_t)(sizeof(float)*_draw.uncompFrames));CUDA_ERROR_CHECK();


}

void GenerateBeadTraceStream::CopyRawDevice()
{

  const RawImage * raw = _info->img->GetImage();

  cudaMemcpy(_draw.image, raw->image, sizeof(short) *_draw.rows*_draw.cols*_draw.frames, cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();
  cudaMemcpy(_draw.interpolatedFrames, raw->interpolatedFrames, sizeof(int) * _draw.uncompFrames, cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();
  cudaMemcpy(_draw.interpolatedMult, raw->interpolatedMult, sizeof(float)*_draw.uncompFrames, cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();
  cudaMemcpy(_draw.interpolatedDiv, raw->interpolatedDiv, sizeof(float)*_draw.uncompFrames, cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();

}

void GenerateBeadTraceStream::DestroyRawDevice()
{

  if(_draw.image != NULL) cudaFree(_draw.image); _draw.image=NULL;CUDA_ERROR_CHECK();
  if(_draw.interpolatedFrames != NULL) cudaFree(_draw.interpolatedFrames); _draw.interpolatedFrames=NULL;CUDA_ERROR_CHECK();
  if(_draw.interpolatedMult != NULL) cudaFree(_draw.interpolatedMult); _draw.interpolatedMult=NULL;CUDA_ERROR_CHECK();
  if(_draw.interpolatedDiv != NULL) cudaFree(_draw.interpolatedDiv); _draw.interpolatedDiv=NULL;CUDA_ERROR_CHECK();

}


void GenerateBeadTraceStream::PreProcessAndAllocatePerRegionBuffers()
{

  int maxSizeFgRegion = MAX_COMPRESSED_FRAMES_GPU * _regMaxHeight * _regMaxWidth;

  _hNptsRegion = new int[_info->numfitters];


  //host side pointer arrays to hold device pointers
  _hFramesPerPointRegion = new int*[_info->numfitters];
  _hFgBufferRegion = new FG_BUFFER_TYPE*[_info->numfitters];



  //allocate device pointer arrays
  cudaMalloc( &_dFgBufferRegion, (size_t)(sizeof(FG_BUFFER_TYPE*) *_info->numfitters));CUDA_ERROR_CHECK();
  cudaMalloc( &_dFramesPerPointRegion, (size_t)(sizeof(int*) *_info->numfitters));CUDA_ERROR_CHECK();

  //allocate host data array
  _hFgBufferRegion_Base = new FG_BUFFER_TYPE[_info->numfitters*maxSizeFgRegion];

  cout << "All " << _info->numfitters << " tracebuffers with padding are: " << (sizeof(FG_BUFFER_TYPE)*maxSizeFgRegion*_info->numfitters)/(1024.0*1024.0) << " MB "<< endl;



  _hFramesPerPointRegion_Base = new int[MAX_COMPRESSED_FRAMES_GPU*_info->numfitters];
    int ** tmpHCopyFPPR = new int*[_info->numfitters];
  //allocate device data arrays
  cudaMalloc( &_dNptsRegion, (size_t)(sizeof(int) *_info->numfitters));CUDA_ERROR_CHECK();
  cudaMalloc( &_dFgBufferRegion_Base, (size_t)(sizeof(FG_BUFFER_TYPE)*maxSizeFgRegion*_info->numfitters));CUDA_ERROR_CHECK();
  cudaMalloc( &_dFramesPerPointRegion_Base, (size_t)(sizeof(int) *MAX_COMPRESSED_FRAMES_GPU));CUDA_ERROR_CHECK();



  for(int r=0; r<_info->numfitters; r++){
    SignalProcessingMasterFitter* BkgObj = _info->BkgObjList[r];
    BkgObj->InitProcessImageForGPU(_info->img, _info->flow);

    //calculate device offsets in host side pointer array
    _hFgBufferRegion[r] = _dFgBufferRegion_Base + maxSizeFgRegion * r;
    _hFramesPerPointRegion[r] = _dFramesPerPointRegion_Base + MAX_COMPRESSED_FRAMES_GPU * r;

    //store number of frames per region
    _hNptsRegion[r] = _info->BkgObjList[r]->get_time_c_npts();

    //store offset pointers
    tmpHCopyFPPR[r] = _hFramesPerPointRegion_Base+MAX_COMPRESSED_FRAMES_GPU * r;
    memcpy(tmpHCopyFPPR[r], &_info->BkgObjList[r]->region_data->time_c.frames_per_point[0],_info->BkgObjList[r]->region_data->time_c.frames_per_point.size() * sizeof(int));

  }






}

void GenerateBeadTraceStream::PostProcessAndDestroyPerRegionBuffers()
{

  for(int r=0; r<_info->numfitters; r++){

//   copy fg_buffers _info->BkgObjList[r]->region_data->my_trace.fg_buffers

    //_info->BkgObjList[r]->FinalizeProcessImageForGPU();
  // these get free'd by the thread that processes them
/*    _info->bkinfo[r].type = MULTI_FLOW_REGIONAL_FIT;
    _info->bkinfo[r].bkgObj = _info->BkgObjList[r];
    _info->bkinfo[r].flow = _info->flow;
    _info->bkinfo[r].sdat = NULL;
    _info->bkinfo[r].img = _info->img;
    _info->bkinfo[r].doingSdat = false;
    _info->bkinfo[r].last = _info->last;
    _info->bkinfo[r].pq = _info->pq;
    */
  }




  if(_dFgBufferRegion_Base!=NULL) cudaFree( _dFgBufferRegion_Base);CUDA_ERROR_CHECK();
  if(_dFramesPerPointRegion_Base !=NULL) cudaFree( _dFramesPerPointRegion_Base);CUDA_ERROR_CHECK();

  if(_dFramesPerPointRegion !=NULL) cudaFree( _dFramesPerPointRegion); _dFramesPerPointRegion = NULL; CUDA_ERROR_CHECK();
  if(_dFgBufferRegion !=NULL) cudaFree( _dFgBufferRegion); _dFgBufferRegion = NULL; CUDA_ERROR_CHECK();
  if(_dNptsRegion !=NULL) cudaFree( _dNptsRegion);_dNptsRegion = NULL; CUDA_ERROR_CHECK();


}



// This stuff lives per block and per device
void GenerateBeadTraceStream::AllocatePersistentData()
{
  int devID = _resources->getDevId();
  int offsetsPerRow = ((_regMaxWidth + _threadBlockX -1)/_threadBlockX) + 1;

  _DevicePersistent[devID].Allocate(offsetsPerRow, _imgWidth , _imgHeight, _regionsX ,_regionsY);
  _BlockPersistent = &_DevicePersistent[devID];
}

// static function that should be called when this process dies... probably from the streamManager
// This stuff lives per block and per device
void GenerateBeadTraceStream::DestroyPersistentData()
{
  _DevicePersistent.clear();
}


bool GenerateBeadTraceStream::ValidJob()
{

  if (_info == NULL )
    return false;

  if (_info->numfitters <= 0 )
    return false;

  if (_info->type != GENERATE_BEAD_TRACES )
    return false;

  return true;
}


//////////////////////////
// VIRTUAL MEMBER FUNCTIONS:
// INIT, ASYNC COPY FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING

void GenerateBeadTraceStream::printStatus()
{

  cout << getLogHeader()  << " status: " << endl
  << " +------------------------------" << endl
  << " | block size: " << _threadBlockX << "x" << _threadBlockY  << endl
  //<< " | l1 setting: " << getL1Setting() << endl
  << " | state: " << _state << endl;
  if(_resources->isSet())
    cout << " | streamResource acquired successfully"<< endl;
  else
    cout << " | streamResource not acquired"<< endl;
   // _myJob.printJobSummary();
    cout << " +------------------------------" << endl;
}


bool GenerateBeadTraceStream::InitJob()
{
  return ValidJob();
}



void GenerateBeadTraceStream::ExecuteJob()
{

  //any cuda calls that are not async have to happen here to keep things clean
  prepareInputs();
  //the following 3 calls can only contain async cuda calls!!!
  copyToDevice();
  executeKernel();
  copyToHost();

}


int GenerateBeadTraceStream::handleResults()
{

  if(_verbose) cout <<  getLogHeader() << " Handling Results " << endl;

    if(!_BlockPersistent->AlreadyCreated())
      _BlockPersistent->DebugPrint();

  _BlockPersistent->MarkAsCreated();


  PostProcessAndDestroyPerRegionBuffers();


  return 0; //signal Job complete
}

////////////////////////////////////////
// Execution step implementation

//allocation and  serialization of buffers in host page locked host memory for async copy
void GenerateBeadTraceStream::prepareInputs()
{

  AllocRawDevice();
  AllocatePersistentData();

  _BlockPersistent->PrepareInputs(_info);

  PreProcessAndAllocatePerRegionBuffers();

}


// trigger async copies, no more sync cuda calls from this [point on until handle results
void GenerateBeadTraceStream::copyToDevice()
{

  if(_verbose) cout << getLogHeader() << " Async Copy To Device" << endl;

  CopyRawDevice();

  _BlockPersistent->CopyInputs();


  //int maxSizeFgRegion = MAX_COMPRESSED_FRAMES_GPU * _regMaxHeight * _regMaxWidth;

  cudaMemcpy( _dNptsRegion, _hNptsRegion, (size_t)(sizeof(int) *_info->numfitters), cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();
  cudaMemcpy( _dFgBufferRegion, _hFgBufferRegion, (size_t)(sizeof(FG_BUFFER_TYPE*) *_info->numfitters),cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();
 cudaMemcpy( _dFramesPerPointRegion, _hFramesPerPointRegion, (size_t)(sizeof(int*) *_info->numfitters),cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();




  cudaMemcpy( _dFramesPerPointRegion_Base, _hFramesPerPointRegion_Base, (size_t)(sizeof(int) *MAX_COMPRESSED_FRAMES_GPU),cudaMemcpyHostToDevice);CUDA_ERROR_CHECK();



}


void GenerateBeadTraceStream::executeKernel()
{
  if(_verbose) cout << getLogHeader() << " Exec Async Kernels" << endl;


  ExecuteCreatePersistentMetaDataKernel();

  ExecuteGenerateBeadTraceKernel();

/*



*/
  //TODO: consider batching a number of kernels to e.g. handle the  Image one row of regions at a time

}


void GenerateBeadTraceStream::copyToHost()
{
   //cout << "Copy data to GPU" << endl;
    if(_verbose) cout << getLogHeader() << " Issue Async Copy To Host" << endl;


  _BlockPersistent->CopyOutputs();

  int maxSizeFgRegion = MAX_COMPRESSED_FRAMES_GPU * _regMaxHeight * _regMaxWidth;

  cudaMemcpy( _hFgBufferRegion_Base, _dFgBufferRegion_Base, (size_t)(sizeof(FG_BUFFER_TYPE)*maxSizeFgRegion*_info->numfitters),cudaMemcpyDeviceToHost);CUDA_ERROR_CHECK();


}

// end execution step implementation
/////////////////////////////////////////////////////

////////////////////////////////////////////
// Kernel invocation methods:


void GenerateBeadTraceStream::ExecuteCreatePersistentMetaDataKernel()
{

  if( _BlockPersistent->AlreadyCreated()) return;

  cout << getLogHeader() << " creating persistent meta data on device" << endl;

  dim3 grid(_regionsX,_regionsY);//one block per region
  dim3 block(_threadBlockX, _threadBlockY);

  int smSize = block.x*block.y*sizeof(int);

  GenerateMetaPerWarpForBlock_k<<< grid, block, smSize >>>(
      _BlockPersistent->getDBfMask(),
      _BlockPersistent->getDT0est(),
      _imgWidth,
      _imgHeight,
      _regMaxWidth,
      _regMaxHeight,
      _BlockPersistent->getDOffsetWarp(),
      _BlockPersistent->getDlBeadsRegion(),
      _BlockPersistent->getDT0avgRegion()
  );
  CUDA_ERROR_CHECK();



 }


void GenerateBeadTraceStream::ExecuteGenerateBeadTraceKernel()
{


  dim3 grid(_regionsX, _draw.rows);//one block per image row, per region
  dim3 block(_threadBlockX, _threadBlockY);

  int smSize = block.x*block.y*sizeof(int);

    GenerateAllBeadTraceFromMeta_k<<< grid, block, smSize >>>(
      _dFgBufferRegion,
      _dFramesPerPointRegion,
      _dNptsRegion,
        _draw,
      _BlockPersistent->getDBfMask(),
      _BlockPersistent->getDT0est(),
      _regMaxWidth,
      _regMaxHeight,
      //from meta data kernel:
      _BlockPersistent->getDOffsetWarp(),
      _BlockPersistent->getDlBeadsRegion(),
      _BlockPersistent->getDT0avgRegion()
            );
  CUDA_ERROR_CHECK();

 }

/*
int GenerateBeadTraceStream::getL1Setting()
{
  if(_l1type < 0 || _l1type > 2){
    return l1DefaultSettingMultiFit();
  }
  return _l1type;
}
*/







// Static member function


size_t GenerateBeadTraceStream::getMaxHostMem()
{

  size_t ret = 0;

  return ret;

}


size_t GenerateBeadTraceStream::getMaxDeviceMem()
{
  // if numFrames/numBeads are passed overwrite the predevined maxFrames/maxBeads
  // for the size calculation
  //Job.setMaxFrames(numFrames);

  //Job.setMaxBeads(numBeads);

  size_t ret = 0;

  return ret;
}






