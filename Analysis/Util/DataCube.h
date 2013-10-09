/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DATACUBE_H
#define DATACUBE_H

#include <vector>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "IonErr.h"
/**
 * Container class for a cube of data which happens quite a bit at
 * IonTorrent as we have an x,y grid and lots of flows or frames. By
 * convention x is columns and y is rows which is slightly odd as
 * usually matrices are in row, col indexing order.
 */
template <class T, class C=size_t, class I=uint64_t>
class DataCube {

public:

  /** Basic constructor. */
  DataCube() { Init(0,0,0); }
  
  /** Constructor with size of each dimension. */
  DataCube(C numX, C numY, C numZ) { Init(numX, numY, numZ); }

  /** Initialize the internal state */
  void Init(C numX, C numY, C numZ) {
    mNumX = numX;
    mNumY = numY;
    mNumZ = numZ;
    SetRange( 0, mNumX, 0, mNumY, 0, mNumZ);
  }

  I ToIndex(C x, C y, C z) const { 
    
    // provide a useful error message and a place to trap
    if (!(x >= mXStart && x < mXEnd && 
              y >= mYStart && y < mYEnd && 
                z >= mZStart && z < mZEnd)){
	  printf("mXStart>=x<mXEnd : y>=mYEnd<mZStart : mZStart>=z<mZEnd \n");
      printf("%ld %ld %ld : %ld %ld %ld : %ld %ld %ld \n", mXStart,x,mXEnd,mYStart,y,mYEnd,mZStart,z,mZEnd);
    }
    // assert nearly useless as the error message doesn't give any hint as to the problem
    ION_ASSERT( x >= mXStart && x < mXEnd && 
              y >= mYStart && y < mYEnd && 
                z >= mZStart && z < mZEnd,
                "Index out of range.");
 
    //return ((y-mYStart) * (mXEnd - mXStart) + (x-mXStart)) * (mZEnd - mZStart) + (z-mZStart); 
    return ((x-mXStart) * (mYEnd - mYStart) + (y-mYStart)) * (mZEnd - mZStart) + (z-mZStart); 
  }

  /** Accessor. */
  T & At(C x, C y, C z) { return mData[ToIndex(x,y,z)]; }

  /** Const accessor. */
  const T & At(C x, C y, C z) const { return mData[ToIndex(x,y,z)]; }
  
  /** Allocate memory. */
  void AllocateBuffer() {
    mData.resize((mXEnd-mXStart) * (mYEnd-mYStart) * (mZEnd - mZStart));
    std::fill(mData.begin(), mData.end(), 0);
  }

  /** Set the current working range, or chunk of cube currently in memory */
  void SetRange(C xStart, C xEnd, C yStart, C yEnd, C zStart, C zEnd) {
    mXStart = xStart;
    mXEnd = xEnd;
    mYStart = yStart;
    mYEnd = yEnd;
    mZStart = zStart;
    mZEnd = zEnd;
  }

  /** Fill in the starts and ends of data range. */
  void SetStartsEnds(C starts[], C ends[]) {
    starts[0] = mXStart;
    starts[1] = mYStart;
    starts[2] = mZStart;
    ends[0] = mXEnd;
    ends[1] = mYEnd;
    ends[2] = mZEnd;
  }

  C GetChunkSize() { return mData.size(); }

  /** Dim accesors */
  C GetNumX() { return mNumX; }
  C GetNumY() { return mNumY; }
  C GetNumZ() { return mNumZ; }
  
  /** Get the current working range or chunk of cube in memory */
  void GetRange(C &xStart, C &xEnd, C &yStart, C &yEnd, C &zStart, C &zEnd) const {
    xStart = mXStart;
    xEnd = mXEnd;
    yStart = mYStart;
    yEnd = mYEnd;
    zStart = mZStart;
    zEnd = mZEnd;
  }

  T GetExampleType() { T t = 0; return t; }

  /** Utility to get all of the z values for a givent x,y position */
  void GetXY(C x, C y, C size, T *t) const { 
    I i = Index(x, y, 0);
    std::copy(mData.begin() + i, mData.end() + i + size, t);
  }

  /** Utility to set all of the z values for a givent x,y position */
  void SetXY(C x, C y, C size, T *t) const { 
    I i = Index(x, y, 0);
    std::copy(t, t + size, mData.begin() + i);
  }
  
  /** Access to underlying memory for I/O */
  T * GetMemPtr() { return &mData[0]; }
  
private:
  std::vector<T> mData; ///< Actual values stored in z,x,y order increasing
  /// Size of window that we're currently working in
  C mXStart, mXEnd, mYStart, mYEnd, mZStart, mZEnd; 
  /// Maximum value of dimensions possible
  C mNumX, mNumY, mNumZ;
};

//#ifndef __CUDACC__
//#ifndef __CUDA_ARCH__
// Boost serialization support:
// Note: restricting index types to size_t to get around complaint from cuda compiler.
template<class Archive, class T>
void serialize(Archive& ar, DataCube<T,std::size_t,std::size_t>& c, const unsigned int version)
{
  ar
  & c.mData
  & c.mXStart
  & c.mXEnd
  & c.mYStart
  & c.mYEnd
  & c.mZStart
  & c.mZEnd
  & c.mNumX
  & c.mNumY
  & c.mNumZ;
}
//#endif

#endif // DATACUBE_H
