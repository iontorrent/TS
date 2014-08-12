/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DELSQCUDA_H
#define DELSQCUDA_H

#include <stdlib.h>
#include "DiffEqModel.h"
#include "cuda_runtime.h"
#include "cuda_error.h"


#define DEVICE_ID 0


struct ConstStruct
{
  size_t x;
  size_t y;
  size_t z;
  DATA_TYPE cx1;
  DATA_TYPE cx2;
  DATA_TYPE cy1;
  DATA_TYPE cy2;
  DATA_TYPE cz1;
  DATA_TYPE cz2;
  int inject_cnt;
};


class DelsqCUDA
{

  int devId;

  ConstStruct cParams;

  DATA_TYPE *Hcmatrix;
  DATA_TYPE *Hdst; 
  DATA_TYPE *Hbuffer_effect; 
  DATA_TYPE *Hcorrection_factor;
  DATA_TYPE *Hlayer_step_frac;
  size_t *Hindex_array;
  DATA_TYPE *Hweight_array;

  DATA_TYPE *Dsrc;
  DATA_TYPE *Ddst; 
  DATA_TYPE *Dbuffer_effect; 
  DATA_TYPE *Dcorrection_factor;
  DATA_TYPE *Dlayer_step_frac;
  size_t *Dindex_array;
  DATA_TYPE *Dweight_array;

  int my_inject_cnt;
  
  DATA_TYPE *dOutput;  // pointer to current output buffer


public:

  DelsqCUDA(size_t x, size_t y, size_t z, int inject_cnt, int deviceId = DEVICE_ID);
  ~DelsqCUDA();

  void createCudaBuffers();
  void destroyCudaBuffers();


  void setParams(DATA_TYPE dx, DATA_TYPE dy,DATA_TYPE dz,DATA_TYPE dcoeff, DATA_TYPE dt);

  void setInput( DATA_TYPE * cmatrix, DATA_TYPE *buffer_effect, DATA_TYPE *correction_factor, DATA_TYPE *layer_step_frac, size_t *index_array, DATA_TYPE *weight_array);
  void copyIn();

  void setOutput( DATA_TYPE *dst );
  void copyOut();

  void DoWork();
  void DoIncorp( DATA_TYPE incorp_signal );


// little helper 
  size_t getX(){ return cParams.x; }
  size_t getY(){ return cParams.y; }
  size_t getZ(){ return cParams.z; }

  size_t size(){ return  sizeof(DATA_TYPE) * getX() * getY() * getZ();}
  size_t sizeLayer(){ return sizeof(DATA_TYPE) * getX() * getY();}
  size_t sizeX() { return sizeof(DATA_TYPE) * getX();}
  size_t sizeY() { return sizeof(DATA_TYPE) * getY();}
  size_t sizeZ() { return sizeof(DATA_TYPE) * getZ();}

};

#endif // DELSQCUDA_H

