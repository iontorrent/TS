/*
 * DeviceSymbolCopy.h
 *
 *  Created on: Jun 10, 2014
 *      Author: jakob
 */

#ifndef DEVICESYMBOLCOPY_H_
#define DEVICESYMBOLCOPY_H_

#include "ImgRegParams.h"
#include "DeviceParamDefines.h"

void copySymbolsToDevice( const ConstantFrameParams& ciP);
void copySymbolsToDevice( const ImgRegParams& irP );
void copySymbolsToDevice( const ConstantParamsGlobal& pl);
void copySymbolsToDevice( const ConfigParams& cp);
void copySymbolsToDevice( const PerFlowParamsGlobal& fp);
void copySymbolsToDevice( const WellsLevelXTalkParams& cXtP);
//void copySymbolsToDevice( const ConstantRegParamBounds& fp);
void CreatePoissonApproxOnDevice(int device);


#endif /* DEVICESYMBOLCOPY_H_ */
