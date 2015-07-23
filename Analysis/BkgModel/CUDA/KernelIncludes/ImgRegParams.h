/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * SingleFitKernel.h
 *
 *  Created on: Feb 6, 2014
 *      Author: jakob
 */

#ifndef IMGREGPARAMS_H_
#define IMGREGPARAMS_H_

#include <fstream>

/* ////////////////////////////////
 * class ImRegParams
 * dimensions of the image and the regions
 * comes with a bunch of simple getter functions to get all kinds of information that can be derived
 * from those values all member function are const, inlined and are available on host and device
 * object is meant to reside in device constant memory but can be used on the host side before
 * it has been copied to the symbol
 * WARNING:
 * class cannot have a non empty constructor to allow creation on constant memory symbol on device
 * therefore class members are uninitialized after creation and need to be initialized through
 * init() function!
 *
 */

class ImgRegParams
{
  size_t imgW;
  size_t imgH;
  size_t regW;
  size_t regH;

public:


  __host__ __device__
  void init(size_t imgW, size_t imgH, size_t regW,  size_t regH)
  {
    this->imgW = (imgW<1)?1:imgW;
    this->imgH = (imgH<1)?1:imgH;
    this->regW = (regW<1)?1:regW;
    this->regH = (regH<1)?1:regH;

  }

  __host__ __device__
  void print()
  {
    printf( "img %lu, %lu  reg %lu ,%lu numregs: %lu\n", imgW, imgH, regW, regH, getNumRegions());
  }

  __host__
  static ImgRegParams readImgRegParamsFromFile(const char * filename)
  {
    std::ifstream myFile (filename, std::ios::binary);

    printf("reading GPU data for new GPU fitting pipeline from file: %s\n", filename);

    if(!myFile){
      printf("file %s could not be opened!\n", filename);
      exit (-1);
    }

    ImgRegParams irtmp;
    printf("%s: reading image params.\n", filename);
    myFile.read((char*)&irtmp,sizeof(ImgRegParams));
    if (myFile){
      printf("%s: ImgRegParams read successfully.\n", filename);
      irtmp.print();
    }
    else {
      printf("%s: ImgregParams could not be read.\n", filename);
      exit(-1);
    }

    myFile.close();

    return irtmp;
  }

  // Dimensions/Strides
  __host__ __device__ inline
  size_t getImgH() const {
    return imgH;
  }

  __host__ __device__ inline
  size_t getImgW() const {
    return imgW;
  }
  //returns maximum region Width. to get width of actual region use getRegW(regId);
  __host__ __device__ inline
  size_t getRegW() const {
    return regW;
  }

  //returns maximum region Height. to get height of actual region use getRegH(regId);
  __host__ __device__ inline
  size_t getRegH() const {
    return regH;
  }


  //returns maximum number of Frames in this image
  __host__ __device__ inline
  size_t getImgSize() const { return imgW*imgH; }

  //return number of regions per row (in x direction)
  __host__ __device__ inline
  size_t getGridDimX() const { return ((imgW +regW -1)/regW); }

  //return number of regions per column (in y direction)
  __host__ __device__ inline
  size_t getGridDimY() const { return ((imgH +regH -1)/regH); }

  //returns a ImgReg Params object that represents an image of the grid size
  //with a reg size of imgRegP(gridX, gridY,1, 1)
  //optional a regW > 1 and a regH > 1 can be provided to create an image object with the same number
  //and order of regions as the original ImgRegParams but with new region dimensions
  //imgRegP(gridX*regW, gridY*regH, regW, regH)
  //can be used to quickly generate buffers layouts with fixed size per region buffers. e.g. background trace per region:
  // RawImagP.getGridParam(numCompressedFrames) creates a layout for numCompressedFrames per region.
  __host__ __device__ inline
  ImgRegParams getGridParam(size_t regW = 1, size_t regH = 1) const { ImgRegParams ip; ip.init(getGridDimX()*regW, getGridDimY()*regH, regW, regH); return ip; }

  //returns number of regions in grid GridDimx*GridDimY
  __host__ __device__ inline
  size_t getNumRegions() const { return getGridDimX() * getGridDimY(); }

  //returns the plane stride to move between x/y planes
  __host__ __device__ inline
  size_t getPlaneStride() const { return getImgSize(); }

  //Region Id to coordinates
  //return column in which the region with region Id regId is in
  __host__ __device__ inline
  size_t getRegCol(size_t regId) const {  return regId%getGridDimX();}

  //return row in which the region with region Id regId is in
  __host__ __device__ inline
  size_t getRegRow(size_t regId) const { return regId/getGridDimX(); }

  //returns the base idx of a region (can be used to determine base coords with get(X/Y)fromIdx()
  __host__ __device__ inline
  size_t getRegBaseIdx(size_t regId) const  { return  getRegRow(regId) * getRegH() * getImgW() + getRegCol(regId) * getRegW(); }

  __host__ __device__ inline
  size_t getRegIdFromGrid(size_t regCol, size_t regRow) const { return regRow*getGridDimX()+regCol; }

  //returns actual region width. execParam regW is the maximum. boarder regions might be smaller
  __host__ __device__ inline
  size_t getRegW(size_t regId) const {
    size_t b = imgW-(getRegCol(regId)*regW);
    return ( b > regW)?(regW):(b);
  }
  //returns actual region height. execParam regH is the maximum. boarder regions might be smaller
  __host__ __device__ inline
  size_t getRegH(size_t regId) const {
    size_t b = imgH - getRegRow(regId)*regH;
    return ( b > regH)?(regH):(b);
  }

  //returns number of elements in one plane of region regId
  __host__ __device__ inline
  size_t getRegSize(size_t regId = 0) const {
    return getRegW(regId)*getRegH(regId);
  }

  //return region id for absolute x and y coordinates
  __host__ __device__ inline
  size_t getRegId(size_t ix, size_t iy) const {
    return ((iy/regH) * getGridDimX() + (ix/regW));
  }

  //returns ImgRegParams for region regId.
  __host__ __device__ inline
    ImgRegParams getRegParam(size_t regId = 0) const {
      ImgRegParams ip;
      ip.init(getRegW(regId), getRegH(regId), getRegW(regId), getRegH(regId) );
      return ip;
  }


  ////////////////////
  //Well operation
  //returns image X coordinate from well index
  __host__ __device__ inline
  size_t getXFromIdx(size_t Idx) const {
    return Idx%imgW;
  }
  //returns image Y coordinate from well index
  __host__ __device__ inline
  size_t getYFromIdx(size_t Idx) const {
    return Idx/imgW;
  }

  //return region id for absolute index of pixel in image
  __host__ __device__ inline
  size_t getRegId(size_t idx) const {
    size_t iy = getYFromIdx(idx);
    size_t ix = getXFromIdx(idx);
    return getRegId(ix,iy);
  }

  //return idx of well in image for absolute x and y coordinates (no check if ix or iy are within image!)
  __host__ __device__ inline
  size_t getWellIdx(size_t ix, size_t iy) const { return iy*imgW + ix; }

  __host__ __device__ inline
  size_t getWellIdx(size_t regId, size_t rx, size_t ry) const {
    return (getRegRow(regId)*regH + ry)*imgW + getRegCol(regId)*regW + rx;
  }

  __host__ __device__ inline
  bool isValidIdx(size_t idx) const {
    if( idx < getImgSize()) return true;
    return false;
  }
  __host__ __device__ inline
  bool isValidCoord(size_t ix, size_t iy) const {
      if( ix < getImgW() && iy < getImgH()) return true;
      return false;
  }
  __host__ __device__ inline
  bool isValidRegId(size_t regId) const {
      if( regId < getNumRegions()) return true;
      return false;
  }

  /////
  // checks if a well with index idx is within a given region regId
  __host__ __device__ inline
  bool isInRegion(size_t regId, size_t idx) const {
    return (regId == getRegId(idx));
  }
  // checks if a well at image location ix,iy is within a give region regId
  __host__ __device__ inline
  bool isInRegion(size_t regId, size_t ix, size_t iy) const {
    return (regId == getRegId(ix,iy));
  }

  //////////////////
  //operator
  // == only compare image dimensions, but NOT the number of frames
  __host__ __device__ inline
  bool operator == (const ImgRegParams &that) const {
    return (this->imgW == that.imgW && this->imgH == that.imgH);
  }

};


#endif //IMGREGPARAMS_H_
