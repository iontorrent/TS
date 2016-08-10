/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * DeviceParamDefines.cpp
 *
 *  Created on: Sep 17, 2015
 *      Author: Jakob Siegel
 */

#include "DeviceParamDefines.h"
#include <bitset>


ostream& operator<<(ostream& os, const PerFlowParamsGlobal& obj)
{
  char dl = MY_STD_DELIMITER;
  os << "PerFlowParamsGlobal[" << dl;
  //os << obj.flowIdx << dl;
  os << obj.realFnum << dl;
  os << obj.NucId << dl;
  os << ']';
  return os;
}


ostream& operator<<(ostream& os, const ConfigParams& obj){

  char dl = MY_STD_DELIMITER;
  std::bitset<16> bs(obj.maskvalue);
  os << "ConfigParams[" << dl;
  os << bs << dl;
  os << ']';
  return os;
}

ostream& operator<<(ostream& os, const ConstantFrameParams& obj)
{
  char dl = MY_STD_DELIMITER;
  os << "ConstantFrameParams[" << dl;
  os << obj.rawFrames << dl;
  os << obj.uncompFrames << dl;
  os << obj.maxCompFrames << dl;

  os << "interpolatedFrames[" << dl;
  for (int i=0; i < obj.uncompFrames; i++)
    os << obj.interpolatedFrames[i] << dl;
  os << ']' << dl;
  os << "interpolatedMult[" << dl ;
  for (int i=0; i < obj.uncompFrames; i++)
    os << obj.interpolatedMult[i] << dl;
  os << ']' << dl;
  os << "interpolatedDiv[" << dl;
  for (int i=0; i < obj.uncompFrames; i++)
    os << obj.interpolatedDiv[i] << dl;
  os << ']';

  return os;

}


ostream& operator<<(ostream& os, const ConstantParamsRegion& obj)
{
  char dl = MY_STD_DELIMITER;
  os << "ConstantParamsRegion[" << dl;
  os << obj.getSens() << dl;
  os << obj.getTauRM() << dl;
  os << obj.getTauRO() << dl;
  os << obj.getTauE() << dl;
  os << obj.getMoleculesToMicromolarConversion() << dl;
  os << obj.getTimeStart() << dl;
  os << obj.getT0Frame() << dl;
  os << obj.getMinTmidNuc() << dl;
  os << obj.getMaxTmidNuc() << dl;
  os << obj.getMinRatioDrift() << dl;
  os << obj.getMaxRatioDrift() << dl;
  os << obj.getMinCopyDrift() << dl;
  os << obj.getMaxCopyDrift() << dl;
  os << ']';
  return os;
}


ostream& operator<<(ostream& os, const PerFlowParamsRegion& obj)
{
  char dl = MY_STD_DELIMITER;
  os << "PerFlowParamsRegion[" << dl;
  os << obj.getFineStart() << dl;
  os << obj.getCoarseStart() << dl;
  os << obj.getSigma() << dl;
  os << obj.getTshift() << dl;
  os << obj.getCopyDrift() << dl;
  os << obj.getRatioDrift() << dl;
  os << obj.getTMidNuc() << dl;
  os << obj.getTMidNucShift() << dl;
  os << obj.getDarkness() << dl;
  os << ']';
  return os;
}


ostream& operator<<(ostream& os, const PerNucParamsRegion& obj)
{
  char dl = MY_STD_DELIMITER;
  os << "PerNucParamsRegion[" << dl;
  os << obj.getD() << dl;
  os << obj.getKmax() << dl;
  os << obj.getKrate() << dl;
  os << obj.getTMidNucDelay() << dl;
  os << obj.getNucModifyRatio() << dl;
  os << obj.getC() << dl;
  os << obj.getSigmaMult() << dl;
  os << ']';
  return os;
}

ostream& operator<<(ostream& os, const  ConstantParamsGlobal& obj)
{
  char dl = MY_STD_DELIMITER;
  os << "ConstantParamsGlobal" << dl;
  os << obj.getMagicDivisorForTiming() << dl;
  os << obj.getNucFlowSpan() << dl;
  os << obj.getValveOpen() << dl;
  os << obj.getAdjKmult() << dl;
  os << obj.getMaxTauB() << dl;
  os << obj.getMaxKmult() << dl;
  os << obj.getMinTauB() << dl;
  os << obj.getScaleLimit() << dl;
  os << obj.getTailDClowerBound() << dl;
  os << obj.getMinAmpl() << dl;
  os << obj.getMinKmult() << dl;
  os << obj.getEmphWidth() << dl;
  os << obj.getEmphAmpl() << dl;
  os << "empParams[" << dl ;
    for (int i=0; i <NUMEMPHASISPARAMETERS; i++)
      os << obj.empParams[i] << dl;
    os << ']' << dl;
  os << obj.getClonalFilterFirstFlow() << dl;
  os << obj.getClonalFilterLastFlow() << dl;
  os << ']';
  return os;
}
