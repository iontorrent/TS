/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * RegionParamsGPU.h
 *
 *  Created on: Jun 3, 2014
 *      Author: jakob
 */

#ifndef REGIONPARAMSGPU_H_
#define REGIONPARAMSGPU_H_

#include <iostream>
#include <cstring>
#include <sstream>
#include "cuda_runtime.h"
#include "cuda_error.h"
#include "Utils.h"
#include "CudaDefines.h"
#include "CudaDefines.h"
#include "BkgMagicDefines.h"
#include "EnumDefines.h"



//LDG LOADER
#if __CUDA_ARCH__ >= 350
#define LDG_MEMBER(var) \
    (__ldg(&(this->var)))
#else
#define LDG_MEMBER(var) \
    (var)
#endif

#if __CUDA_ARCH__ >= 350
#define LDG_LOAD(ptr) \
    (__ldg(ptr))
#else
#define LDG_LOAD(ptr) \
    (*(ptr))
#endif

#define MY_STD_DELIMITER ','


struct SampleCoordPair{
  unsigned short x;
  unsigned short y;

  __host__ __device__
  SampleCoordPair():x(0),y(0){};
  __host__ __device__
  SampleCoordPair(unsigned short _x, unsigned short _y ):x(_x),y(_y){};

};



//XTALK DEFINES
//#define XTALK_SPAN_X  2
//#define XTALK_SPAN_Y  2
//#define XTALK_WIDTH   ( 2 * XTALK_SPAN_X +1 )
//#define XTALK_HEIGHT  ( 2 * XTALK_SPAN_Y +1 )
//#define XTALK_MAP ( XTALK_WIDTH * XTALK_HEIGHT )

#define XTALK_DIM(a) (2*(a)+1)
#define XTALK_MAP(sx,sy) (XTALK_DIM(sx)*XTALK_DIM(sy))



////////////////////////////////////////////////////////////
//CONSTANT MEMORY OBJECTS

template<int MaxSpanX, int MaxSpanY>
class WellsLevelXTalkParamsConst{

protected:

  int spanX;
  int spanY;
  float evenPhaseMap[ XTALK_MAP(MaxSpanX,MaxSpanY)];
  float oddPhaseMap[ XTALK_MAP(MaxSpanX,MaxSpanY) ];

public:

  __host__ __device__
  int Hash(int lx, int ly) const{
    return(ly*XTALK_DIM(spanX)+lx);
  }

  __host__ __device__
  float even(int lx, int ly) const
  {
    return evenPhaseMap[Hash(lx,ly)];
  }
  __host__ __device__
  float even(int idx) const
  {
    return evenPhaseMap[idx];
  }
  __host__ __device__
  float odd(int lx, int ly) const
  {
    return oddPhaseMap[Hash(lx,ly)];
  }
  __host__ __device__
  float odd(int idx) const
  {
    return oddPhaseMap[idx];
  }
  __host__ __device__
  float coeff(int lx, int ly, int phase) const
  {
    //phase == 1_-> even since we count starting at 0
    return (phase)?(even(lx,ly)):(odd(lx,ly));
  }
  __host__ __device__
  int getPhase(int col) const
  {
    return (col & 1);
  }

  __host__ __device__
  int getSpanX() const {return spanX;}
  __host__ __device__
  int getSpanY() const {return spanY;}
  __host__ __device__
  int getMapWidth() const {return 2*spanX+1;}
  __host__ __device__
  int getMapHeight() const {return 2*spanY+1;}
  __host__ __device__
  int getMapSize() const {return getMapWidth()*getMapHeight();}


  __host__ __device__
  int getMaxSpanX() const {return MaxSpanX;}
  __host__ __device__
  int getMaxSpanY(){return MaxSpanY;}

  __host__ __device__
  void print(){
    printf("WellsLevelXTalkParamsConst\n spanX: %d spanY: %d\n", spanX, spanY);
    printf(" Map size: %d\n  n,  odd, even\n", getMapSize());
    for(int n=0; n<getMapSize(); n++) printf("%3d,%5f,%5f\n",n,oddPhaseMap[n],evenPhaseMap[n]);
  }


};




//this class is meant only for use as constant symbol on the device side.
//derived host side class with constructor and setter in: TraceLevelXTalk.h
template<size_t numNeighbours>
class XTalkNeighbourStatsConst
{

protected:

  int cx[numNeighbours];
  int cy[numNeighbours];
  float multiplier[numNeighbours];
  //float tauTop[numNeighbours]; // not used for simple XTalk
  //float tauFluid[numNeighbours]; // not used for simple XTalk

  int initialPhase;
  size_t numN;
  bool hexPacked;
  bool threeSeries;




  __host__ __device__
  void NeighborByGridPhase(int &ncx, int &ncy, const int x, const int y, const int nei, const int phase) const
  {
    if (phase==0)
    {
      ncx = x+cx[nei];
      ncy = y+cy[nei];
    } else
    {
      ncy = y+cy[nei];
      if (cy[nei]!=0)
        ncx = x+cx[nei]-phase; // up/down levels are offset alternately on rows
      else
        ncx = x+cx[nei];
    }
  }

  // bb operates the other direction
  __host__ __device__
  void NeighborByGridPhaseBB(int &ncx, int &ncy, const int x, const int y, const int nei, const int phase) const
  {
    ncx = x+cx[nei]; // neighbor columns are always correct
    ncy = y+cy[nei]; // map is correct
    if ((phase!=0) & (((cx[nei]+16) %2)!=0)) //neighbors may be more than one away!!!!
      ncy -= phase; // up/down levels are offset alternately on cols
  }


  //for block coordinated x and y are coordinates in block and region_x/y are both 0
  //for region coordinates x and y are coordinates withing region and region_x/y have to be the region dimensions.
  __host__ __device__
  void NeighborByChipType(int &ncx, int &ncy, const int x, const int y, const int nei_idx, const int region_x = 0, const int region_y = 0) const
  {
    // the logic has now become complex, so encapsulate it
    // phase for hex-packed
    if (!hexPacked)
      NeighborByGridPhase (ncx,ncy,x,y,nei_idx, 0);
    else
    {
      if (threeSeries)
        NeighborByGridPhase (ncx,ncy,x,y,nei_idx,(y+region_y+1)%2); // maybe????
      else
        NeighborByGridPhaseBB(ncx,ncy,x,y,nei_idx,(x+region_x+1+initialPhase) % 2); // maybe????
    }
  }



public:


  __host__ __device__
  size_t getNumMaxNeighbours() const { return numNeighbours; }
  __host__ __device__
  size_t getNumNeighbours() const { return numN; }


  //float getTauTop(int nid) const { return tauTop[nid]; } // not used for simple XTalk
  //float getTauFluid(int nid) const { return tauFluid[nid]; } // not used for simple XTalk
  __host__ __device__
  float getMultiplier( int nid) const { return multiplier[nid];}

  //x and y are bead coordinates in block
  __host__ __device__
  void getBlockCoord(int& nx, int& ny, const int nid, const int x,const int y) const { NeighborByChipType(nx,ny,x,y,nid); }
  //rx and ry are bead coordinates in region and region_x/y are region dimensions
  __host__ __device__
  void getRegionCoord(int& nx, int& ny, const int nid, const int rx,const int ry,const int region_x, const int region_y) const { NeighborByChipType(nx,ny,rx,ry,nid,region_x,region_y); }

  __host__ __device__
  void print(){
    printf("XTalkNeighbourStatsConst\n initial Phase: %d\n hex packed: %s\n three series: %s\n", initialPhase, hexPacked?("true"):("false"),threeSeries?("true"):("false"));
    printf(" num Neighbours: %zu\n  n, nx, ny, mult\n", numN);
    for(size_t n=0; n<numN; n++) printf("%3zu,%3d,%3d, %f\n",n,cx[n],cy[n],multiplier[n]);
  }



};


class ConfigParams {

  enum ConfigMask {
    ConfMaskNone = 0,
    ConfMaskFitTauE = (1 << 0),
    ConfMaskFitKmult = (1 << 1),
    ConfMaskUseDarkMatterPCA = (1 << 2),
    ConfMaskUseDynamicEmphasis = (1 << 3),
    ConfMaskUseAlternativeEtbRequation = (1 << 4),
    ConfMaskPerformExpTailFitting = (1 << 5),
    ConfMaskPerformBkgAdjInExpTailFit = ( 1 << 6),
    ConfMaskPerformRecompressTailRawTrace = (1 << 7),
    ConfMaskPerformWellsLevelXTalk = ( 1 << 8),
    ConfMaskPerformTraceLevelXTalk = ( 1 << 9),
    ConfMaskPerformPolyClonalFilter = ( 1 << 10),
    ConfMaskFitTmidNucShift = ( 1 << 11)
  };

  unsigned short maskvalue;

public:
  __host__ void clear() {
    maskvalue  =  ConfMaskNone;
  }

  __host__ void setFitTauE() {
    maskvalue = maskvalue | ConfMaskFitTauE;
  }
  __host__ void setFitKmult() {
    maskvalue = maskvalue | ConfMaskFitKmult;
  }
  __host__ void setUseDarkMatterPCA() {
    maskvalue = maskvalue | ConfMaskUseDarkMatterPCA;
  }
  __host__ void setUseDynamicEmphasis() {
    maskvalue = maskvalue | ConfMaskUseDynamicEmphasis;
  }
  __host__ void setUseAlternativeEtbRequation() {
    maskvalue = maskvalue | ConfMaskUseAlternativeEtbRequation;
  }
  __host__ void setPerformExpTailFitting() {
    maskvalue = maskvalue | ConfMaskPerformExpTailFitting;
  }
  __host__ void setPerformBkgAdjInExpTailFit() {
    maskvalue = maskvalue | ConfMaskPerformBkgAdjInExpTailFit;
  }
  __host__ void setPerformRecompressTailRawTrace() {
    maskvalue = maskvalue | ConfMaskPerformRecompressTailRawTrace;
  }
  __host__ void setPerformWellsLevelXTalk() {
    maskvalue = maskvalue | ConfMaskPerformWellsLevelXTalk;
  }
  __host__ void setPerformTraceLevelXTalk() {
    maskvalue = maskvalue | ConfMaskPerformTraceLevelXTalk;
  }
  __host__ void setPerformPolyClonalFilter() {
    maskvalue = maskvalue | ConfMaskPerformPolyClonalFilter;
  }
  __host__ void setFitTmidNucShift() {
    maskvalue = maskvalue | ConfMaskFitTmidNucShift;
  }



  __host__ __device__ inline
  bool FitTauE() const {
    return maskvalue & ConfMaskFitTauE;
  }
  __host__ __device__ inline
  bool FitKmult() const {
    return maskvalue & ConfMaskFitKmult;
  }
  __host__ __device__ inline
  bool UseDarkMatterPCA() const {
    return maskvalue & ConfMaskUseDarkMatterPCA;
  }
  __host__ __device__ inline
  bool UseDynamicEmphasis() const {
    return maskvalue & ConfMaskUseDynamicEmphasis;
  }
  __host__ __device__ inline
  bool UseAlternativeEtbRequation() const {
    return maskvalue & ConfMaskUseAlternativeEtbRequation;
  }
  __host__ __device__ inline
  bool PerformExpTailFitting() const {
    return maskvalue & ConfMaskPerformExpTailFitting;
  }
  __host__ __device__ inline
  bool PerformBkgAdjInExpTailFit() const {
    return maskvalue & ConfMaskPerformBkgAdjInExpTailFit;
  }
  __host__ __device__ inline
  bool PerformRecompressTailRawTrace() const {
    return maskvalue & ConfMaskPerformRecompressTailRawTrace;
  }
  __host__ __device__ inline
  bool PerformWellsLevelXTalk() const {
    return maskvalue & ConfMaskPerformWellsLevelXTalk;
  }
  __host__ __device__ inline
  bool PerformTraceLevelXTalk() const {
    return maskvalue & ConfMaskPerformTraceLevelXTalk;
  }
  __host__ __device__ inline
  bool PerformPolyClonalFilter() const {
    return maskvalue & ConfMaskPerformPolyClonalFilter;
  }
  __host__ __device__ inline
  bool FitTmidNucShift() const {
    return maskvalue & ConfMaskFitTmidNucShift;
  }

  __host__ __device__ inline
  void print(){
    printf("ConfigParams\n FitTauE %s\n FitKmult %s\n UseDarkMatterPCA %s\n UseDynamicEmphasis %s\n UseAlternativeEtbRequation %s\n PerformExpTailFitting %s\n PerformBkgAdjInExpTailFit %s\n PerformRecompressTailRawTrace %s\n PerformWellsLevelXTalk %s\n PerformTraceLevelXTalk %s\n PerfromPolyClonalFilter %s\n FitTmidNucShift %s\n",
        (FitTauE())?("true"):("false"),
            (FitKmult())?("true"):("false"),
                (UseDarkMatterPCA())?("true"):("false"),
                    (UseDynamicEmphasis())?("true"):("false"),
                        (UseAlternativeEtbRequation())?("true"):("false"),
                            (PerformExpTailFitting())?("true"):("false"),
                                (PerformBkgAdjInExpTailFit())?("true"):("false"),
                                    (PerformRecompressTailRawTrace())?("true"):("false"),
                                        (PerformWellsLevelXTalk())?("true"):("false"),
                                            (PerformTraceLevelXTalk())?("true"):("false"),
                                                (PerformPolyClonalFilter())?("true"):("false"),
                                                    (FitTmidNucShift())?("true"):("false"));

  }

  friend ostream& operator<<(ostream& os, const ConfigParams& obj);

};


class ConstantParamsGlobal {

  float valve_open; // timing parameter
  float nuc_flow_span; // timing of nuc flow
  float magic_divisor_for_timing; // timing parameter
  float minAmpl;
  float minKmult;
  float maxKmult;
  float adjKmult;
  float min_tauB;
  float max_tauB; // range of possible values
  float scaleLimit;
  float tailDClowerBound;
  float empWidth;
  float empAmpl;
  float empParams[NUMEMPHASISPARAMETERS];
  int clonalFilterFirstFLow;
  int clonalFilterLastFLow;


public:
  __host__
  void setMagicDivisorForTiming(float magicDivisorForTiming) {
    magic_divisor_for_timing = magicDivisorForTiming;
  }
  __host__
  void setNucFlowSpan(float nucFlowSpan) {
    nuc_flow_span = nucFlowSpan;
  }
  __host__
  void setValveOpen(float valveOpen) {
    valve_open = valveOpen;
  }
  __host__
  void setAdjKmult(float adjKmult) {
    this->adjKmult = adjKmult;
  }
  __host__
  void setMaxTauB(float maxTauB) {
    max_tauB = maxTauB;
  }
  __host__
  void setMaxKmult(float maxKmult) {
    this->maxKmult = maxKmult;
  }
  __host__
  void setMinTauB(float minTauB) {
    min_tauB = minTauB;
  }
  __host__
  void setScaleLimit(float scale_limit) {
    this->scaleLimit = scale_limit;
  }
  __host__
  void setTailDClowerBound(float tail_dc_lower_bound) {
    tailDClowerBound = tail_dc_lower_bound;
  }
  __host__
  void setMinAmpl(float minAmpl) {
    this->minAmpl = minAmpl;
  }
  __host__
  void setMinKmult(float minKmult) {
    this->minKmult = minKmult;
  }
  __host__
  void setEmphWidth(float empwidth) {
    this->empWidth = empwidth;
  }
  __host__
  void setEmphAmpl(float empampl) {
    this->empAmpl = empampl;
  }
  __host__
  void setEmphParams(const float *empparams) {
    memcpy(this->empParams, empparams, sizeof(float)*NUMEMPHASISPARAMETERS);
  }

  __host__
  void setClonalFilterFirstFlow(int clonalFilterFromFlow) {
    clonalFilterFirstFLow = clonalFilterFromFlow;
  }
  __host__
  void setClonalFilterLastFlow(int clonalFilterToFlow) {
    clonalFilterLastFLow = clonalFilterToFlow;
  }

  __host__ __device__ inline
  float getMagicDivisorForTiming() const {
    return magic_divisor_for_timing;
  }
  __host__ __device__ inline
  float getNucFlowSpan() const {
    return nuc_flow_span;
  }
  __host__ __device__ inline
  float getValveOpen() const {
    return valve_open;
  }
  __host__ __device__ inline
  float getAdjKmult() const {
    return adjKmult;
  }
  __host__ __device__ inline
  float getMaxTauB() const {
    return max_tauB;
  }
  __host__ __device__ inline
  float getMaxKmult() const {
    return maxKmult;
  }
  __host__ __device__ inline
  float getMinTauB() const {
    return min_tauB;
  }
  __host__ __device__ inline
  float getScaleLimit() const {
    return scaleLimit;
  }
  __host__ __device__ inline
  float getTailDClowerBound() const {
    return tailDClowerBound;
  }
  __host__ __device__ inline
  float getMinAmpl() const {
    return minAmpl;
  }
  __host__ __device__ inline
  float getMinKmult() const {
    return minKmult;
  }
  __host__ __device__ inline
  float getEmphWidth() const {
    return empWidth;
  }
  __host__ __device__ inline
  float getEmphAmpl() const {
    return empAmpl;
  }
  __host__ __device__ inline
  const float* getEmphParams() const {
    return empParams;
  }
  __host__ __device__ inline
  int getClonalFilterFirstFlow() const {
    return clonalFilterFirstFLow;
  } __host__ __device__ inline

  int getClonalFilterLastFlow() const {
    return clonalFilterLastFLow;
  }
  __host__ __device__ inline
  bool isClonalUpdateFlow( int flow) const {
    return ( flow >= getClonalFilterFirstFlow() &&  flow < getClonalFilterLastFlow() );
  }
  __host__ __device__ inline
  bool isLastClonalUpdateFlow( int flow) const {
    return ( flow == getClonalFilterLastFlow() - 1);
  }
  __host__ __device__ inline
  bool isApplyClonalFilterFlow( int flow) const {
    return ( flow == getClonalFilterLastFlow());
  }
  __host__ __device__ inline
  int getClonalFilterNumFlows() const {
    return getClonalFilterLastFlow() - getClonalFilterFirstFlow();
  }


  __host__ __device__ inline
  void print() const {
    printf("ConstantParamsGlobal\n valve_open %f magic_divisor_for_timing %f nuc_flow_span %f clonalFilterFirstFlow %d clonalFilterLastFlow %d\n",
        valve_open,  magic_divisor_for_timing, nuc_flow_span, clonalFilterFirstFLow, clonalFilterLastFLow);
    printf(" minAmpl %f minKmult %f maxKmult %f adjKmult %f min_tauB %f max_tauB %f \n",
        minAmpl, minKmult, maxKmult, adjKmult, min_tauB, max_tauB);
  }

  friend ostream& operator<<(ostream& os, const ConstantParamsGlobal& obj);

};


//persistent throughout execution
struct ConstantFrameParams
{

  int rawFrames; // frames from the image  aka vfc compressed frames
  int uncompFrames; // actually frames after trace is uncompressed...max we can get
  int maxCompFrames; // maximum compressed frames for signal processing across all regions

public:

  int   interpolatedFrames[MAX_UNCOMPRESSED_FRAMES_GPU];
  float interpolatedMult[MAX_UNCOMPRESSED_FRAMES_GPU];
  float interpolatedDiv[MAX_UNCOMPRESSED_FRAMES_GPU];

  __host__
  void setRawFrames(int rawFrames) {
    this->rawFrames = rawFrames;
  }
  __host__
  void setUncompFrames(int uncompFrames) {
    this->uncompFrames = uncompFrames;
  }
  __host__
  void setMaxCompFrames(int maxCompFrames) {
    this->maxCompFrames = maxCompFrames;
  }

  __host__ __device__ inline
  int getRawFrames() const {
    return rawFrames;
  }
  __host__ __device__ inline
  int getUncompFrames() const {
    return uncompFrames;
  }
  __host__ __device__ inline
  int getMaxCompFrames() const {
    return maxCompFrames;
  }

  __host__ __device__ inline
  int getImageAllocFrames() const {
    return max(maxCompFrames,rawFrames);
  }


  __host__ __device__ inline
  void print(){
    printf("ConstantFrameParams\n GPU raw Image frames: %d, uncomp frames: %d, maxBkgFrames: %d \n", rawFrames, uncompFrames, maxCompFrames);
    printf("interpolatedFrames:");
    for(int i=0; i < uncompFrames; i++)
      printf("%d,",interpolatedFrames[i]);
    printf("\n");
    printf("interpolatedMult:");
    for(int i=0; i < uncompFrames; i++)
      printf("%f,",interpolatedMult[i]);
    printf("\n");
    printf("interpolatedDiv:");
    for(int i=0; i < uncompFrames; i++)
      printf("%f,",interpolatedDiv[i]);
    printf("\n");
  }

  friend ostream& operator<<(ostream& os, const ConstantFrameParams& obj);

};

/*class ConstantRegParamBounds
{
  float minTmidNuc;
  float maxTmidNuc;
  float minRatioDrift;
  float minCopyDrift;
  float maxCopyDrift;

public:

  __host__ inline
  void setMinTmidNuc(float minTmidNuc) {
    this->minTmidNuc = minTmidNuc;
  }

  __host__ inline
  void setMaxTmidNuc(float maxTmidNuc) {
    this->maxTmidNuc = maxTmidNuc;
  }

  __host__ inline
  void setMinRatioDrift(float minRatioDrift) {
    this->minRatioDrift = minRatioDrift;
  }

  __host__ inline
  void setMinCopyDrift(float minCopyDrift) {
    this->minCopyDrift = minCopyDrift;
  }

  __host__ inline
  void setMaxCopyDrift(float maxCopyDrift) {
    this->maxCopyDrift = maxCopyDrift;
  }

  __host__ __device__ inline
  float getMinTmidNuc() const {
    return minTmidNuc;
  }

  __host__ __device__ inline
  float getMaxTmidNuc() const {
    return maxTmidNuc;
  }

  __host__ __device__ inline
  float getMinRatioDrift() const {
    return minRatioDrift;
  }

  __host__ __device__ inline
  float getMinCopyDrift() const {
    return minCopyDrift;
  }

  __host__ __device__ inline
  float getMaxCopyDrift() const {
    return maxCopyDrift;
  }

  __host__ __device__ inline
  void print(){
    printf("ConstantRegParamBounds GPU tmidNuc(min:%f, max:%f), ratioDrift(min:%f), copyDrift(min:%f, max:%f) \n", 
        minTmidNuc, maxTmidNuc, minRatioDrift, minCopyDrift, maxCopyDrift);
  }
};*/


class PerFlowParamsGlobal {

  //int flowIdx; //ToDo: remove as soon as data is only copied by flow
  int realFnum;
  int NucId;

public:

  __host__
  void setRealFnum(int realFnum) {
    this->realFnum = realFnum;
  }
  //__host__
  //void setFlowIdx(int flowIdx) {
  //  this->flowIdx = flowIdx;
  //}

  __host__
  void setNucId(int nucId) {
    NucId = nucId;
  }



  //__host__ __device__ inline
  //int getFlowIdx() const {
  //  return flowIdx;
  //}

  __host__ __device__ inline
  int getNucId() const {
    return NucId;
  }

  __host__ __device__ inline
  int getRealFnum() const {
    return realFnum;
  }


  __host__ __device__ inline
  void print() const {
    //printf("PerFlowParamsGlobal\n flowIdx %d realFnum %d NucId %d\n", flowIdx, realFnum, NucId );
    printf("PerFlowParamsGlobal\n realFnum %d NucId %d\n", realFnum, NucId );
  }

  friend ostream& operator<<(ostream& os, const PerFlowParamsGlobal& obj);

};







// END CONSTATN MEMORY OBJECTS
////////////////////////////////////////////////////////////
// GLOBAL MEMORY OBJECTS PER REGION


class ConstantParamsRegion {

  float sens; // conversion b/w protons generated and signal - no reason why this should vary by nuc as hydrogens are hydrogens.
  float tau_R_m;  // relationship of empty to bead slope
  float tau_R_o;  // relationship of empty to bead offset
  float tauE;
  float molecules_to_micromolar_conversion; // depends on volume of well
  float time_start;  //time_c.time_start
  float t0Frame;
  float minTmidNuc;
  float maxTmidNuc;
  float minRatioDrift;
  float maxRatioDrift;
  float minCopyDrift;
  float maxCopyDrift;
  //float R; // currently not used after first 20
  //float Copies; //currently not used after first 20
  //float sigma;  //currently not used after first 20

public:

  __host__
  void setSens(float sens) {
    this->sens = sens;
  }
  __host__
  void setTauRM(float tauRM) {
    tau_R_m = tauRM;
  }
  __host__
  void setTauRO(float tauRO) {
    tau_R_o = tauRO;
  }
  __host__
  void setTauE(float tauE) {
    this->tauE = tauE;
  }
  __host__
  void setMoleculesToMicromolarConversion(
      float moleculesToMicromolarConversion) {
    molecules_to_micromolar_conversion = moleculesToMicromolarConversion;
  }
  __host__
  void setTimeStart(float timeStart) {
    time_start = timeStart;
  }

  __host__
  void setT0Frame(float t0_start) {
    t0Frame = t0_start;
  }

  __host__
  void setMinTmidNuc(float minTmidNuc) {
    this->minTmidNuc = minTmidNuc;
  }

  __host__
  void setMaxTmidNuc(float maxTmidNuc) {
    this->maxTmidNuc = maxTmidNuc;
  }

  __host__
  void setMaxRatioDrift(float maxRatioDrift) {
    this->maxRatioDrift = maxRatioDrift;
  }

  __host__
  void setMinRatioDrift(float minRatioDrift) {
    this->minRatioDrift = minRatioDrift;
  }

  __host__
  void setMaxCopyDrift(float maxCopyDrift) {
    this->maxCopyDrift = maxCopyDrift;
  }

  __host__
  void setMinCopyDrift(float minCopyDrift) {
    this->minCopyDrift = minCopyDrift;
  }

  __host__ __device__ inline
  float getSens() const {
    return LDG_MEMBER(sens);
  }
  __host__ __device__ inline
  float getTauRM() const {
    return LDG_MEMBER(tau_R_m);
  }
  __host__ __device__ inline
  float getTauRO() const {
    return LDG_MEMBER(tau_R_o);
  }
  __host__ __device__ inline
  float getTauE() const {
    return LDG_MEMBER(tauE);
  }
  __host__ __device__ inline
  float getMoleculesToMicromolarConversion() const {
    return LDG_MEMBER(molecules_to_micromolar_conversion);
  }
  __host__ __device__ inline
  float getTimeStart() const {
    return LDG_MEMBER(time_start);
  }
  __host__ __device__ inline
  float getT0Frame() const {
    return LDG_MEMBER(t0Frame);
  }

  __host__ __device__ inline
  float getMaxTmidNuc() const {
    return LDG_MEMBER(maxTmidNuc);
  }

  __host__ __device__ inline
  float getMinTmidNuc() const {
    return LDG_MEMBER(minTmidNuc);
  }

  __host__ __device__ inline
  float getMaxRatioDrift() const {
    return LDG_MEMBER(maxRatioDrift);
  }

  __host__ __device__ inline
  float getMinRatioDrift() const {
    return LDG_MEMBER(minRatioDrift);
  }

  __host__ __device__ inline
  float getMaxCopyDrift() const {
    return LDG_MEMBER(maxCopyDrift);
  }

  __host__ __device__ inline
  float getMinCopyDrift() const {
    return LDG_MEMBER(minCopyDrift);
  }

  __host__ __device__ inline
  void print() const {
    printf("ConstantParamsRegion sense %f tau_R_m %f tau_R_o %f tauE %f molecules_to_micromolar_conversion %f time_start %f t0Frame %f tmidNuc(min:%f, max:%f) ratiodrift(min:%f, max: %f) copydrift(min:%f max:%f)\n",
        getSens(), getTauRM(), getTauRO(), getTauE(), getMoleculesToMicromolarConversion(),getTimeStart(), getT0Frame(), getMinTmidNuc(), getMaxTmidNuc(), getMinRatioDrift(), getMaxRatioDrift(), getMinCopyDrift(), getMaxCopyDrift());
  }

  friend ostream& operator<<(ostream& os, const ConstantParamsRegion& obj);

};



//ToDO: load with ldg?
class PerFlowParamsRegion {

  int fineStart; //[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  int coarseStart; //[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  float sigma;
  float tshift;
  float CopyDrift;
  float RatioDrift;
  float t_mid_nuc; //updated per block of 20 only use [0] of host buffer
  float t_mid_nuc_shift;
  float darkness; //[MAX_NUM_FLOWS_IN_BLOCK_GPU] only [0] was used now single value

public:

  __host__ __device__ inline
  void setTshift(float tshift) {
    this->tshift = tshift;
  }
  __host__ __device__ inline
  void setSigma(float sigma) {
    this->sigma = sigma;
  }
  __host__ __device__ inline
  void setCopyDrift(float copyDrift) {
    CopyDrift = copyDrift;
  }
  __host__ __device__ inline
  void setDarkness(float darkness) {
    this->darkness = darkness;
  }
  __host__ __device__ inline
  void setRatioDrift(float ratioDrift) {
    RatioDrift = ratioDrift;
  }
  __host__ __device__ inline
  void setFineStart(int start) {
    this->fineStart = start;
  }
  __host__ __device__ inline
  void setCoarseStart(int start) {
    this->coarseStart = start;
  }
  __host__ __device__ inline
  void setTMidNuc(float tMidNuc) {
    t_mid_nuc = tMidNuc;
  }
  __host__ __device__ inline
  void setTMidNucShift(float tMidNucShift) {
    t_mid_nuc_shift = tMidNucShift;
  }
  __host__ __device__ inline
  float getCopyDrift() const {
    //return LDG_MEMBER(CopyDrift);
    return CopyDrift;
  }
  __host__ __device__ inline
  float getDarkness() const {
    //return LDG_MEMBER(darkness);
    return darkness;
  }
  __host__ __device__ inline
  float getRatioDrift() const {
    //return LDG_MEMBER(RatioDrift);
    return RatioDrift;
  }
  __host__ __device__ inline
  int getFineStart() const {
    return fineStart;
  }
  __host__ __device__ inline
  int getCoarseStart() const {
    return coarseStart;
  }
  __host__ __device__ inline
  float getTMidNuc() const {
    //return LDG_MEMBER(t_mid_nuc);
    return t_mid_nuc;
  }
  __host__ __device__ inline
  float getTMidNucShift() const {
    //return LDG_MEMBER(t_mid_nuc_shift);
    return t_mid_nuc_shift;
  }
  __host__ __device__ inline
  float getSigma() const {
    //return LDG_MEMBER(sigma);
    return sigma;
  }
  __host__ __device__ inline
  float getTshift() const {
    //return LDG_MEMBER(tshift);
    return tshift;
  }
  __host__ __device__ inline
  void print() const {
    printf("PerFlowParamsRegion fineStart %d coarseStart %d sigma %f tshift %f CopyDrift %f RatioDrift %f t_mid_nuc %f t_mid_nuc_shift %f darkness %f\n",
        getFineStart(), getCoarseStart(), getSigma(), getTshift(), getCopyDrift(), getRatioDrift(), getTMidNuc(), getTMidNucShift(), getDarkness());
  }

  friend ostream& operator<<(ostream& os, const PerFlowParamsRegion& obj);

private:

  friend class boost::serialization::access;
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version) {

    ar &
    fineStart &
    coarseStart &
    sigma &
    tshift &
    CopyDrift &
    RatioDrift &
    t_mid_nuc &
    t_mid_nuc_shift &
    darkness;
  }

};



//store as plane of regions per nuc
class PerNucParamsRegion {
  float d;    // dNTP diffusion rate
  float kmax;  // saturation per nuc action rate
  float krate;  // rate of incorporation
  float t_mid_nuc_delay;
  float NucModifyRatio;  // buffering modifier per nuc
  float C; // dntp in uM 
  float sigma_mult; // concentration

public:

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

  __host__
  void setC(float C) {
    this->C = C;
  }

  __host__ 
  void setSigmaMult(float sigma_mult) {
    this->sigma_mult = sigma_mult;
  }

  __host__ __device__ inline
  float getD() const {
    return LDG_MEMBER(d);
  }
  __host__ __device__ inline
  float getKmax() const {
    return LDG_MEMBER(kmax);
  }
  __host__ __device__ inline
  float getKrate() const {
    return LDG_MEMBER(krate);
  }
  __host__ __device__ inline
  float getNucModifyRatio() const {
    return LDG_MEMBER(NucModifyRatio);
  }
  __host__ __device__ inline
  float getTMidNucDelay() const {
    return LDG_MEMBER(t_mid_nuc_delay);
  }

  __host__ __device__ inline
  float getC() const {
    return LDG_MEMBER(C);
  }

  __host__ __device__ inline
  float getSigmaMult() const {
    return LDG_MEMBER(sigma_mult);
  }

  __host__ __device__ inline
  void print() const {
    printf("PerNucParamsRegion d %f kmax %f krate %f t_mid_nuc_delay %f NucModifyRatio %f C %f simga_mult %f\n",
        getD(), getKmax(), getKrate(), getTMidNucDelay(), getNucModifyRatio(), getC(), getSigmaMult());
  }

  friend ostream& operator<<(ostream& os, const PerNucParamsRegion& obj);
};



/*
 * Assumption so far was that structure would be better since most threads will be working on the same region so structure stays in cache
 * on the other hand if multiple regions are handled in one multiprocessor a cube layout might be more beneficial to keep the most
 * frequently used params for multiple regions in texture cache
 */
class PerNucParamsRegionDeviceCubeWrapper {
  const float * base;
  const size_t stride;

public:

  PerNucParamsRegionDeviceCubeWrapper(const float * ptr, const int NucId, const size_t RegId, const size_t numRegs):stride(numRegs)
{
    base = ptr + NucId *(stride * Nuc_NUM_PARAMS) + RegId;
}

  __device__ inline
  float getD() const {
    return LDG_LOAD(base+NucD*stride);
  }
  __device__ inline
  float getKmax() const {
    return LDG_LOAD(base+NucKmax*stride);
  }
  __device__ inline
  float getKrate() const {
    return LDG_LOAD(base+NucKrate*stride);
  }
  __device__ inline
  float getNucModifyRatio() const {
    return LDG_LOAD(base+NucModifyRatio*stride);
  }
  __device__ inline
  float getTMidNucDelay() const {
    return LDG_LOAD(base + NucT_mid_nuc_delay*stride);
  }

  __device__ inline
  void print() const {
    printf("PerNucParamsRegion d %f kmax %f krate %f t_mid_nuc_delay %f NucModifyRatio %f\n",
        getD(), getKmax(), getKrate(), getTMidNucDelay(), getNucModifyRatio());
  }

};






/////////////////////



#endif /* REGIONPARAMSGPU_H_ */


