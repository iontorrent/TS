/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * HostParamDefines.h
 *
 * contains host side classes derived from device side classes in DeviceParamDefines.h that are meant to be used as symbols in constant device memory
 * Therefore here we can implement host side cosntructors/destructors are not allowed to have non-const functions, constructors or destructors.
 *
 *
 *  Created on: May 28, 2015
 *      Author: Jakob Siegel
 */

#ifndef HOSTPARAMDEFINES_H_
#define HOSTPARAMDEFINES_H_

#include "CudaDefines.h"
#include "DeviceParamDefines.h"



//constructor and non const host only functions are implemented here:
class WellsLevelXTalkParamsHost : public WellsLevelXTalkParamsConst<MAX_WELL_XTALK_SPAN,MAX_WELL_XTALK_SPAN>{

public:

  __host__
  WellsLevelXTalkParamsHost(){

    this->spanX = 2;
    this->spanY = 2;

    float nn_even_phase_map_defaultPI[] = {
        0.000,0.000,0.010,0.000,0.000,
        0.015,0.031,0.131,0.026,0.015,
        0.026,0.127,0.000,0.111,0.025,
        0.015,0.117,0.133,0.110,0.013,
        0.000,0.021,0.010,0.020,0.000,
    };
    //from WellXtalk.h
    memcpy(evenPhaseMap,nn_even_phase_map_defaultPI,sizeof(float)*getMapSize());
    OddFromEvenHexCol();
    NormalizeCoeffs(evenPhaseMap);
    NormalizeCoeffs(oddPhaseMap);
  }

  __host__
  WellsLevelXTalkParamsHost( const float * oddphasemap, const float * evenphasemap, const int xtalkSpanX, const int xtalkSpanY){

    if (xtalkSpanX > MAX_WELL_XTALK_SPAN || xtalkSpanY > MAX_WELL_XTALK_SPAN){
      std::cout << "CUDA ERROR: requested XTalks span of " << xtalkSpanX << "," << xtalkSpanY << " is larger than defined MAX_WELL_XTALK_SPAN of " << MAX_WELL_XTALK_SPAN << "," << MAX_WELL_XTALK_SPAN << "!" << std::endl;
      exit(-1);
    }
    this->spanX = xtalkSpanX;
    this->spanY = xtalkSpanY;
    //from WellXtalk.h
    memcpy(evenPhaseMap,evenphasemap,sizeof(float)*getMapSize());
    memcpy(oddPhaseMap,oddphasemap,sizeof(float)*getMapSize());
  }

  ~WellsLevelXTalkParamsHost(){};

  __host__
  void OddFromEvenHexCol(){
    for (int ly=0; ly<getMapHeight(); ly++){
      for (int lx=0; lx<getMapWidth(); lx++){
        int tx = lx;
        int ty = ly;
        int col_phase = lx &1;
        if (col_phase)
          ty = ly+1;
        if (ty<getMapHeight()){
          oddPhaseMap[Hash(lx,ly)] = evenPhaseMap[Hash(tx,ty)];
        } else {
          oddPhaseMap[Hash(lx,ly)] = 0.0f;
        }
      }
    }
  }
  __host__
  void NormalizeCoeffs(float *c_map){
    float coeff_sum = 0.0f;
    for (int i=0; i<getMapSize(); i++)
      coeff_sum += c_map[i];
    for (int i=0; i<getMapSize(); i++){
      c_map[i] /= coeff_sum;
    }
  }
};





//template<int numNeighbours>
class XTalkNeighbourStatsHost : public XTalkNeighbourStatsConst<MAX_XTALK_NEIGHBOURS>
{


protected:

  //all input values are set through constructor only hence setters for those are protected
  __host__
  void setCoords( const std::vector<int>& vcx, const std::vector<int>& vcy)
  {

    for(size_t i=0; i< numN; i++)
    {
      this->cx[i]=vcx[i];
      this->cy[i]=vcy[i];
    }

  }

  // not used for simple XTalk
  //void setTauTop( std::vector<float>& tau_top){ for(size_t i; i< numNeighbours && i<tau_top.size(); i++)  tauTop[i]=tau_top[i]; }
  //void setTauFluid( std::vector<float>& tau_fluid){ for(size_t i; i< numNeighbours && i<tau_fluid.size(); i++)  tauFluid[i]=tau_fluid[i]; }
  __host__
  void setMultiplier( const std::vector<float>& mult){ for(size_t i=0; i < numN; i++)  multiplier[i]=mult[i];}

  // not used for simple XTalk
  //void setTauTop(int nid, float tau_top){tauTop[nid]=tau_top;}
  //void setTauFluid(int nid, float tau_fluid){tauFluid[nid]=tau_fluid; }
  __host__
  void setMultiplier( int nid, float mult) { multiplier[nid]=mult; }


public:
  __host__
  XTalkNeighbourStatsHost( const std::vector<int>& vcx, const std::vector<int>& vcy, const std::vector<float>& multi){

    this->numN=vcx.size();

    if (  this->numN > MAX_XTALK_NEIGHBOURS){
      std::cout << "CUDA ERROR: requested XTalks neighbours of " << this->numN << " is larger than defined MAX_XTALK_NEIGHBOURS of " << MAX_XTALK_NEIGHBOURS << "!" << endl;
      exit(-1);
    }

    setCoords(vcx,vcy);
    setMultiplier(multi);
    setThreeSeries(false); // Default is not three series but P0
    setHexPacked(true); // only before 3series
  }

  __host__
  void setHexPacked(bool hex_packed){hexPacked=hex_packed;}
  __host__
  void setThreeSeries(bool three_series){threeSeries=three_series;}
  __host__
  void setInitialPhase(int initial_phase){ initialPhase = initial_phase;}

};
















#endif /* HOSTPARAMDEFINES_H_ */
