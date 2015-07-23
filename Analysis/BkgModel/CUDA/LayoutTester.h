/*
 * LayoutTester.h
 *
 *  Created on: Feb 19, 2014
 *      Author: jakob
 */

#ifndef LAYOUTTESTER_H_
#define LAYOUTTESTER_H_

#include "cudaWrapper.h"

void testLayout(const char * ImgProcessingFileName, const char * bkgModelFileName, const char * resultfileName, float epsilon, int cacheSetting,int blockW,int blockH );

bool blockLevelSignalProcessing(BkgModelWorkInfo* fitterInfo, int flowBlockSize, int deviceId);

#endif /* LAYOUTTESTER_H_ */
