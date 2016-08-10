/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 *
 * ConstantSymbolDeclare.h
 *
 *  Created on: Mar 18, 2014
 *      Author: jakob
 */

#ifndef CONSTANTSYMBOLDECLARE_H_
#define CONSTANTSYMBOLDECLARE_H_

#include "CudaDefines.h"
#include "DeviceParamDefines.h"
#include "SampleHistory.h"

//Todo: break up in logical sub structures and remove unused parameters
__constant__ ConfigParams ConfigP;

//new symbols
__constant__ ImgRegParamsConst ImgRegP;
__constant__ ConstantFrameParams ConstFrmP;
__constant__ ConstantParamsGlobal ConstGlobalP;


__constant__ PerFlowParamsGlobal ConstFlowP;


__constant__ HistoryCollectionConst ConstHistCol;

//__constant__ ConstantRegParamBounds ConstBoundRegP;

__constant__ static float4 * POISS_APPROX_LUT_CUDA_BASE;


//XTALK Contparams
__constant__ WellsLevelXTalkParamsConst<MAX_WELL_XTALK_SPAN,MAX_WELL_XTALK_SPAN> ConstXTalkP;

__constant__ XTalkNeighbourStatsConst<MAX_XTALK_NEIGHBOURS> ConstTraceXTalkP;

#endif /* CONSTANTSYMBOLDECLARE_H_ */
