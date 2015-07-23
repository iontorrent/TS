/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 *
 * ConstantSymbolDeclare.h
 *
 *  Created on: Mar 18, 2014
 *      Author: jakob
 */

#ifndef CONSTANTSYMBOLDECLARE_H_
#define CONSTANTSYMBOLDECLARE_H_

//Todo: break up in logical sub structures and remove unused parameters
__constant__ ConfigParams ConfigP;

//new symbols
__constant__ ImgRegParams ImgRegP;
__constant__ ConstantFrameParams ConstFrmP;
__constant__ ConstantParamsGlobal ConstGlobalP;


__constant__ PerFlowParamsGlobal ConstFlowP;

//__constant__ ConstantRegParamBounds ConstBoundRegP;

__constant__ static float4 * POISS_APPROX_LUT_CUDA_BASE;


//XTALK Contparams
__constant__ WellsLevelXTalkParams ConstXTalkP;


#endif /* CONSTANTSYMBOLDECLARE_H_ */
