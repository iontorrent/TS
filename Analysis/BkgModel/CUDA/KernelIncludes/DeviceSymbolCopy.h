/*
 * DeviceSymbolCopy.h
 *
 *  Created on: Jun 10, 2014
 *      Author: jakob
 */

#ifndef DEVICESYMBOLCOPY_H_
#define DEVICESYMBOLCOPY_H_

#include "CudaDefines.h"
#include "ImgRegParams.h"
#include "DeviceParamDefines.h"

class SampleCollectionConst;

void copySymbolsToDevice( const ConstantFrameParams& ciP);
void copySymbolsToDevice( const ImgRegParamsConst& irP );
void copySymbolsToDevice( const ConstantParamsGlobal& pl);
void copySymbolsToDevice( const ConfigParams& cp);
void copySymbolsToDevice( const PerFlowParamsGlobal& fp);
void copySymbolsToDevice( const WellsLevelXTalkParamsConst<MAX_WELL_XTALK_SPAN,MAX_WELL_XTALK_SPAN> & cXtP);
void copySymbolsToDevice(const XTalkNeighbourStatsConst<MAX_XTALK_NEIGHBOURS> & cTlXtP);
void copySymbolsToDevice(const SampleCollectionConst& smplCol );
//void copySymbolsToDevice( const ConstantRegParamBounds& fp);
void CreatePoissonApproxOnDevice(int device);


#endif /* DEVICESYMBOLCOPY_H_ */
