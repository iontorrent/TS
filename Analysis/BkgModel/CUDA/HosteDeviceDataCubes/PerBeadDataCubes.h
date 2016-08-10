/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * BeadParamCubeClass.h
 *
 *  Created on: Sep 1, 2015
 *      Author: Jakob Siegel
 */

#ifndef PERBEADDATACUBES_H_
#define PERBEADDATACUBES_H_


#include "LayoutCubeRegionsTemplate.h"
#include "DeviceParamDefines.h"
#include "ImgRegParams.h"

struct BkgModelWorkInfo;

class perBeadParamCubeClass : public LayoutCubeWithRegions<float>
{
protected:
  void translateHostToCube(BkgModelWorkInfo* pBase_bkinfo); //base pointer to array with  iP.getNumRegions() regions.

public:
  perBeadParamCubeClass(ImgRegParams iP, MemoryType mtype, vector<size_t>& sizeBytes  ):LayoutCubeWithRegions<float>(iP, Bp_NUM_PARAMS, mtype, sizeBytes){};
  perBeadParamCubeClass(ImgRegParams iP, MemoryType mtype ):LayoutCubeWithRegions<float>(iP, Bp_NUM_PARAMS, mtype){};
  perBeadParamCubeClass( const perBeadParamCubeClass &that, MemoryType mtype ):LayoutCubeWithRegions<float>(that, mtype){};
  virtual ~perBeadParamCubeClass(){};
  void init(BkgModelWorkInfo* pBase_bkinfo);


};

class perBeadPolyClonalCubeClass : public LayoutCubeWithRegions<float>
{
public:
  perBeadPolyClonalCubeClass(ImgRegParams iP, MemoryType mtype, vector<size_t>& sizeBytes  ):LayoutCubeWithRegions<float>(iP, Poly_NUM_PARAMS, mtype, sizeBytes){};
  perBeadPolyClonalCubeClass(ImgRegParams iP, MemoryType mtype ):LayoutCubeWithRegions<float>(iP, Poly_NUM_PARAMS, mtype){};
  perBeadPolyClonalCubeClass(const perBeadPolyClonalCubeClass &that, MemoryType mtype ):LayoutCubeWithRegions<float>(that, mtype){};
  virtual ~perBeadPolyClonalCubeClass(){};

  void initHostRegion(BkgModelWorkInfo* pRegion_bkinfo, size_t regId);
  void initHost(BkgModelWorkInfo* pBase_bkinfo); //base pointer to array with  iP.getNumRegions() regions.
  void init(BkgModelWorkInfo* pBase_bkinfo);

  void reinjectHostStructures(BkgModelWorkInfo* pBase_bkinfo);

};




class perBeadT0CubeClass : public LayoutCubeWithRegions<float>
{
public:
  perBeadT0CubeClass(ImgRegParams iP, MemoryType mtype, vector<size_t>& sizeBytes  ):LayoutCubeWithRegions<float>(iP, 1, mtype, sizeBytes){};
  perBeadT0CubeClass(ImgRegParams iP, MemoryType mtype ):LayoutCubeWithRegions<float>(iP, 1, mtype){};
  perBeadT0CubeClass(const perBeadT0CubeClass & that, MemoryType mtype ):LayoutCubeWithRegions<float>(that, mtype){};
  virtual ~perBeadT0CubeClass(){};

  void init(BkgModelWorkInfo* pBkinfo);
};



class perBeadStateMaskClass : public LayoutCubeWithRegions<unsigned short>
{
public:
  perBeadStateMaskClass(ImgRegParams iP, MemoryType mtype, vector<size_t>& sizeBytes  ):LayoutCubeWithRegions<unsigned short>(iP, 1, mtype, sizeBytes){};
  perBeadStateMaskClass(ImgRegParams iP, MemoryType mtype ):LayoutCubeWithRegions<unsigned short>(iP, 1, mtype){};
  perBeadStateMaskClass(const perBeadStateMaskClass & that,MemoryType mtype ):LayoutCubeWithRegions<unsigned short>(that, mtype){};
  virtual ~perBeadStateMaskClass(){};

  void initHostRegion(BkgModelWorkInfo* pRegion_bkinfo, size_t regId);
  void initHost(BkgModelWorkInfo* pBase_bkinfo);
  void init(BkgModelWorkInfo* pBase_bkinfo);

  void reinjectHostStructures(BkgModelWorkInfo* pBase_bkinfo);

};



//for a host side object this only wraps the original host pointer since data does not need to be reorganized
class perBeadBfMaskClass : public LayoutCubeWithRegions<unsigned short>
{
public:

  perBeadBfMaskClass(ImgRegParams iP, MemoryType mtype, vector<size_t>& sizeBytes  ):LayoutCubeWithRegions<unsigned short>(iP, 1, mtype, sizeBytes){};
  perBeadBfMaskClass(ImgRegParams iP, MemoryType mtype ):LayoutCubeWithRegions<unsigned short>(iP, 1, mtype){};
  perBeadBfMaskClass(unsigned short * ptr, ImgRegParams iP, MemoryType mtype):LayoutCubeWithRegions<unsigned short>(ptr,iP, 1, mtype){};
  virtual ~perBeadBfMaskClass(){};

  void init(BkgModelWorkInfo* pBkinfo);
};


//for a host side object this only wraps the original host pointer since data does not need to be reorganized
class perBeadTraceCubeClass : public LayoutCubeWithRegions<short>
{
public:
  perBeadTraceCubeClass(ImgRegParams iP, int maxFrames,  MemoryType mtype, vector<size_t>& sizeBytes  ):LayoutCubeWithRegions<short>(iP, maxFrames, mtype, sizeBytes){};
  perBeadTraceCubeClass(ImgRegParams iP, int maxFrames, MemoryType mtype ):LayoutCubeWithRegions<short>(iP, maxFrames, mtype){};
  perBeadTraceCubeClass(short * ptrTraces, ImgRegParams iP, int maxFrames, MemoryType mtype ):LayoutCubeWithRegions<short>(ptrTraces, iP, maxFrames, mtype){};
  virtual ~perBeadTraceCubeClass(){};

  void init(BkgModelWorkInfo * bkinfo);

};



class perBeadResultCubeClass : public LayoutCubeWithRegions<short>
{
public:
  perBeadResultCubeClass(ImgRegParams iP,  MemoryType mtype, vector<size_t>& sizeBytes  ):LayoutCubeWithRegions<short>(iP, Result_NUM_PARAMS, mtype, sizeBytes){};
  perBeadResultCubeClass(ImgRegParams iP, MemoryType mtype ):LayoutCubeWithRegions<short>(iP, Result_NUM_PARAMS, mtype){};
  virtual ~perBeadResultCubeClass(){};
};


#endif /* PERBEADDATACUBES_H_ */
