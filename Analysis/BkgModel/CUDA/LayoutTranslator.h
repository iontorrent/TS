/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 *
 *  Created on: Jan 30, 2014
 *      Author: jakob
 */

#ifndef LAYOUTTRANSLATOR_H_
#define LAYOUTTRANSLATOR_H_


#include "LayoutCubeRegionsTemplate.h"

#include "RegionParams.h"
#include "BeadParams.h"
#include "ParamStructs.h"
#include "ImgRegParams.h"
#include "DeviceParamDefines.h"
#include "SingleFlowFitKernels.h"
#include "Image.h"
#include "NonBasicCompare.h"

//#define DUMP_PATH  "/results/Jakob/Dump"
//#define DUMP_PATH  "."
#define DUMP_PATH  "CubeDump"

//#define RESULTS_CHECK_PATH  "ResultsDump/"





//////////////////////////////////////////////////////////////////////////////
// Actual Translators


class BeadMask : public LayoutCubeWithRegions<unsigned short>
{

  //returns true if any bit set in type is also set in mask value
  bool Match(unsigned short maskvalue, unsigned short type){ return (maskvalue & type)? true:false;}
  //only returns true if all bits set in type are also set in mask value
  bool MatchAllBits(unsigned short maskvalue, unsigned short type){ return ((maskvalue & type) == type);}

public:
  BeadMask( size_t imgW, size_t imgH,
      size_t regW, size_t regH):
        LayoutCubeWithRegions<unsigned short>(imgW,imgH,regW,regH)
        { setRWStrideX();} //stride is set to X since mask is 2D and we might want to iterate over the mask


  //use
  //copyIn(unsigned short * mask)   copyOut(unsigned short * mask) to copy a hostside mask or to extract the mask
  //getAt(x,y) and putAt(value,x,y) to access block mask
  //getAtReg(regId,x,y) and putAtReg(value,regId,x,y) to access region regId mask
  //sets and advances RW pointer by stride (default = 1, next mask entry)
  bool Match( size_t x, size_t y, unsigned short type) { return Match(getAt(x,y),type); }
  bool MatchReg(size_t regId, size_t x, size_t y, unsigned short type){ return Match(getAtReg(regId,x,y),type); }
  //does not advance RW pointer
  bool Match(unsigned short type) { return Match(getCurrent(),type); }

  //bit or type with current mask value and returns new mask value, sets and advances RW pointer by stride (default = 1, next mask entry)
  unsigned short And(unsigned short type, size_t x, size_t y) { setRWPtr(x,y); unsigned short tmp = (getCurrent() & type); write(tmp); return tmp;};
  unsigned short AndReg(unsigned short type, size_t regId, size_t x, size_t y){setRWPtrRegion(regId,x,y); unsigned short tmp = (getCurrent() & type); write(tmp); return tmp;}
  unsigned short Or( unsigned short type, size_t x, size_t y) { setRWPtr(x,y); unsigned short tmp = (getCurrent() | type); write(tmp); return tmp;};
  unsigned short OrReg(unsigned short type, size_t regId, size_t x, size_t y){setRWPtrRegion(regId,x,y); unsigned short tmp = (getCurrent()| type); write(tmp); return tmp;}
  unsigned short XOr( unsigned short type, size_t x, size_t y) { setRWPtr(x,y); unsigned short tmp = (getCurrent() ^ type); write(tmp); return tmp;};
  unsigned short XOrReg(unsigned short type, size_t regId, size_t x, size_t y){setRWPtrRegion(regId,x,y); unsigned short tmp = (getCurrent() ^ type); write(tmp); return tmp;}

  //does not advance RW pointer
  unsigned short And( unsigned short type){ unsigned short tmp = ( getCurrent() & type); putCurrent(tmp); return tmp;}
  unsigned short Or( unsigned short type){ unsigned short tmp = ( getCurrent() | type); putCurrent(tmp); return tmp;}
  unsigned short XOr( unsigned short type){ unsigned short tmp = ( getCurrent() ^ type); putCurrent(tmp); return tmp;}

};


///////////////////////////////////////////////////////////////////////////////////
//



//more generic dump function

template <typename T>
class CubePerFlowDump{

  size_t flowBlockBase;
  size_t regionsDumped;
  size_t regionsDumpedLastBlock;
  size_t FlowBlockSize;
  string filePathPrefix;
  vector< LayoutCubeWithRegions<T>*> FlowCubes;
  ImgRegParams ImageParams;


  void destroy();
  void ReadInOneFlow(size_t realflowIdx);
  void WriteOneFlowToFile(LayoutCubeWithRegions<T> * dumpCube, size_t dumpflowIdx);
  void WriteAllFlowsToFile();
  void ClearFlowCubes();

public:
  //set up size of one cube
  //size
  CubePerFlowDump(size_t planeW, size_t planeH, size_t regW, size_t regH, size_t planes, size_t numFlowsinBlock);
  CubePerFlowDump( ImgRegParams iP, size_t planes, size_t numFlowsinBlock);

  ~CubePerFlowDump();

  void setFilePathPrefix(string filep);

  //dumps the data of the current region for the current flow block automatically dumps to file when moving on the next
  //block of numFlowsinBlock. keeps track of how many regions got dumped for current flow block and will dump automatically as soon as that number of regions is reached again
  //very crude function that only write nPerFlow values sequential into memroy from the start element of region regId in the designated plane
  //use with caution (OR DO NOT USE AT ALL)
  void DumpFlowBlockRegion(size_t regId, T* data, size_t flowBlockStartFlow, size_t nPerFlow  = 1 , size_t flowstride = 1, size_t plane = 0);

  //improved version from above. expects a Cube where the dimension of regId in the dump fit the dimension of iRegId in the
  //passed cube and will copy the input cube region into the dum region starting from plane to plane+numPlanes
  //only dumps one flow designated by absolute flow numnber = flowBlockStartFlow + flowInBlockIdx.
  //For the dump to work correctly the flows for one region have to be dumped  from 0 to numFlowsinBlock-1 before duimping the flows oif the next region
  void DumpOneFlowRegion(size_t regId, LayoutCubeWithRegions<T> & input, size_t iRegId, size_t flowBlockStartFlow, size_t flowInBlockIdx = 0, size_t plane = 0 ,  size_t numPlanes = 1) ;

  void DumpOneFlowBlock(LayoutCubeWithRegions<T> & input, size_t flowBlockStartFlow, size_t flowInBlockIdx = 0);

  //determine region by absolute coordinates in image. make sure the dimensions match
  void DumpFlowBlockRegion(size_t ax,size_t ay, T* data, size_t realflowIdx, size_t nPerFlow = 1, size_t flowstride = 1 );

  ImgRegParams getIP(){return ImageParams;}


  LayoutCubeWithRegions<T> & getFlowCube(size_t realflowIdx);

};

///////////////////////////////////
//translator functions
//translate function will translate the old layout into a Cube layout for the new gpu pipeline
//none of the functions check if the passed cube's dimensions fit the data, it will only assert
//during translation if elements would be placed outside a region

namespace TranslatorsFlowByFlow {

  void TranslateFgBuffer_RegionToCube(  LayoutCubeWithRegions<short> & ImageCube,
      size_t numLBeads,
      size_t numFrames,
      size_t flowsPerBLock,
      FG_BUFFER_TYPE *fgPtr, // has to point to first bead of flow to translate
      BeadParams * bP,
      size_t regId);
  void TranslateFgBuffer_CubeToRegion(  LayoutCubeWithRegions<short> & ImageCube,
      size_t numLBeads,
      size_t numFrames,
      size_t flowsPerBLock,
      FG_BUFFER_TYPE *fgPtr, // has to point to first bead of flow to translate
      BeadParams * bP,
      size_t regId);

  void TranslateBeadParams_RegionToCube( LayoutCubeWithRegions<float> & BeadParamCube,void * bkinfo, size_t regId);
  void TranslateBeadParams_CubeToRegion( LayoutCubeWithRegions<float> & BeadParamCube,
      size_t numLBeads,
      BeadParams * bP,
      size_t regId);

  void TranslatePolyClonal_RegionToCube( LayoutCubeWithRegions<float> & BeadParamCube,void * bkinfo, size_t regId);

  void TranslateBeadStateMask_RegionToCube(  LayoutCubeWithRegions<unsigned short> & BkgModelMask,void * bkinfo, size_t regId);
  void TranslateBeadStateMask_CubeToRegion(  LayoutCubeWithRegions<unsigned short> & BkgModelMask,void * bkinfo, size_t regId);

  void TranslateResults_RegionToCube( LayoutCubeWithRegions<float> & BeadParamCube,
      size_t numLBeads,
      size_t flowIdxInBlock,
      BeadParams * bP,
      size_t regId);
  void  TranslateResults_CubeToRegion(LayoutCubeWithRegions<float> & ResultCube, void * bkinfo, size_t flowIdxInBlock, size_t regId);

  void TranslateRegionParams_CubeToRegion( LayoutCubeWithRegions<reg_params> & RegionCube,
      reg_params * rP,  size_t regId = 0);

  void TranslateRegionParams_RegionToCube( LayoutCubeWithRegions<reg_params> & RegionCube, void* bkinfo,
      size_t regId);

  void TranslateRegionFrameCube_RegionToCube( LayoutCubeWithRegions<float> & RegionFrameCube, void * bkinfo, size_t regId);
  void TranslateRegionFramesPerPoint_RegionToCube( LayoutCubeWithRegions<int> & RegionFramesPerPoint, void * bkinfo, size_t regId);

  void TranslateEmphasis_RegionToCube(LayoutCubeWithRegions<float> & RegionEmphasis, void * bkinfo, size_t regId);
  void TranslateNonZeroEmphasisFrames_RegionToCube(LayoutCubeWithRegions<int> & RegionNonZeroEmphFrames, void * bkinfo, size_t regId);
  void TranslateNucRise_RegionToCube(LayoutCubeWithRegions<float> & NucRise, void *bkinfo, size_t flowIdx,  size_t regId);

  void TranslatePerFlowRegionParams_RegionToCube(LayoutCubeWithRegions<PerFlowParamsRegion> & PerFlowParamReg, void * bkinfo, size_t flowIdx, size_t regId );
  void TranslatePerFlowRegionParams_CubeToRegion(LayoutCubeWithRegions<PerFlowParamsRegion> &PerFlowParamReg, void *bkinfo, size_t regId);
  void UpdatePerFlowRegionParams_RegionToCube(LayoutCubeWithRegions<PerFlowParamsRegion> & PerFlowParamReg, reg_params * rP, size_t flowIdx, size_t regId );

  void TranslateConstantRegionParams_RegionToCube(LayoutCubeWithRegions<ConstantParamsRegion> & ConstParamReg, void * bkinfo, size_t regId);
  void TranslatePerNucRegionParams_RegionToCube(LayoutCubeWithRegions<PerNucParamsRegion> & PerNucCube, void * bkinfo, size_t regId);

}

//Populate and Copy Constant Memory Symbols

namespace ConstanSymbolCopier {
  void PopulateSymbolConstantImgageParams( ImgRegParams iP, ConstantFrameParams & CfP, void * bkinfoArray);
  void PopulateSymbolConstantGlobal( ConstantParamsGlobal & CpG, void * bkinfo);
  void PopulateSymbolConfigParams(ConfigParams & confP, void * bkinfo);
  void PopulateSymbolPerFlowGlobal(PerFlowParamsGlobal & pFpG, void * bkinfo);
  //void PopulateSymbolConstantRegParamBounds(ConstantRegParamBounds & CpB, void * bkinfo);
}








void BuildGenericSampleMask(
    bool * sampleMask, //global base pointer to mask Initialized with false
    const ImgRegParams &imgP,
    size_t regId);


void BuildMaskFromBeadParams_RegionToCube(   LayoutCubeWithRegions<unsigned short> & Mask,
    size_t numLBeads,
    BeadParams * bP,
    size_t regId);



/*

class PerNucParamsRegionHostCubeWrapper: public LayoutCubeWithRegions<float> {



public:
  PerNucParamsRegionHostCubeWrapper(size_t imgWidth, size_t imgHeight, size_t maxRegWidth, size_t maxRegHeight):LayoutCubeWithRegions<float>(imgWidth,imgHeight,maxRegWidth,maxRegHeight,NUMNUC*Nuc_NUM_PARAMS,HostMem)
    {};

  PerNucParamsRegionHostCubeWrapper(ImgRegParams iP):LayoutCubeWithRegions<float>(iP.getImgW(),iP.getImgH(),iP.getRegW(),iP.getRegH(),NUMNUC*Nuc_NUM_PARAMS,HostMem)
    {};

  __host__
   void setD(float d) {
     this->d = d;
   }
   __host__
   void setKmax(float kmax) {
     this->kmax = kmax;
   }
   __host__
   void setKrate(float krate) {
     this->krate = krate;
   }
   __host__
   void setNucModifyRatio(float nucModifyRatio) {
     NucModifyRatio = nucModifyRatio;
   }
   __host__
   void setTMidNucDelay(float tMidNucDelay) {
     t_mid_nuc_delay = tMidNucDelay;
   }
   __host__ inline
   float getD() const {
     return LDG_MEMBER(d);
   }
   __host__ inline
   float getKmax() const {
     return LDG_MEMBER(kmax);
   }
   __host__ inline
   float getKrate() const {
     return LDG_MEMBER(krate);
   }
   __host__ inline
   float getNucModifyRatio() const {
     return LDG_MEMBER(NucModifyRatio);
   }
   __host__ inline
   float getTMidNucDelay() const {
     return LDG_MEMBER(t_mid_nuc_delay);
   }

   __host__  inline
   void print() const {
     printf("PerNucParamsRegion d %f kmax %f krate %f t_mid_nuc_delay %f NucModifyRatio %f\n",
         getD(), getKmax(), getKrate(), getTMidNucDelay(), getNucModifyRatio());
   }


};


 */






#endif /* LAYOUTTRANSLATOR_H_ */
