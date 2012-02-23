/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DATATABLE_H
#define DATATABLE_H

#include <vector>
#include <algorithm>
#include <stdint.h>
#include "IonErr.h"
/**
 */
template <class T, class C=size_t, class I=uint64_t>
class DataTable {

public:

  /** Basic constructor. */
  DataTable() { Init(0,0); }
  
  /** Constructor with size of each dimension. */
  DataTable(C numX, C numY) { Init(numX, numY); }

  /** Initialize the internal state */
  void Init(C numX, C numY, C numZ) {
    mNumX = numX;
    mNumY = numY;
    SetRange( 0, mNumX, 0, mNumY);
  }

  I ToIndex(C x, C y) const { 
    ION_ASSERT( x >= mXStart && x < mXEnd && 
                y >= mYStart && y < mYEnd,
                "Index out of range.");
    return ((y-mYStart) * (mXEnd - mXStart) + (x-mXStart));
  }

  /** Accessor. */
  T & At(C x, C y) { return mData[ToIndex(x,y)]; }

  /** Const accessor. */
  const T & At(C x, C y) const { return mData[ToIndex(x,y)]; }
  
  /** Allocate memory. */
  void AllocateBuffer() {
    mData.resize((mXEnd-mXStart) * (mYEnd-mYStart));
    std::fill(mData.begin(), mData.end(), 0);
  }

  /** Set the current working range, or chunk of cube currently in memory */
  void SetRange(C xStart, C xEnd, C yStart, C yEnd) {
    mXStart = xStart;
    mXEnd = xEnd;
    mYStart = yStart;
    mYEnd = yEnd;
  }

  /** Dim accesors */
  C GetNumX() { return mNumX; }
  C GetNumY() { return mNumY; }
  
  /** Get the current working range or chunk of cube in memory */
  void GetRange(C &xStart, C &xEnd, C &yStart, C &yEnd) const {
    xStart = mXStart;
    xEnd = mXEnd;
    yStart = mYStart;
    yEnd = mYEnd;
  }

  T GetExampleType() { T t = 0; return t; }
  
  /** Access to underlying memory for I/O */
  T * GetMemPtr() { return &mData[0]; }
  
private:
  std::vector<T> mData; ///< Actual values stored in z,x,y order increasing
  /// Size of window that we're currently working in
  C mXStart, mXEnd, mYStart, mYEnd;
  /// Maximum value of dimensions possible
  C mNumX, mNumY;
};

#endif // DATATABLE_H
