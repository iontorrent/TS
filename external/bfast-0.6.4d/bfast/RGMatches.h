#ifndef RGMATCHES_H_
#define RGMATCHES_H_

#include <stdio.h>
#include <zlib.h>
#include "BLibDefinitions.h"

int32_t RGMatchesRead(gzFile, RGMatches*);
int32_t RGMatchesReadWithOffsets(gzFile, RGMatches*);
int32_t RGMatchesReadText(FILE*, RGMatches*);
void RGMatchesPrint(gzFile, RGMatches*);
void RGMatchesPrintWithOffsets(gzFile, RGMatches*);
void RGMatchesPrintText(FILE*, RGMatches*);
void RGMatchesPrintFastq(FILE*, RGMatches*);
void RGMatchesRemoveDuplicates(RGMatches*, int32_t);
int32_t RGMatchesMergeFilesAndOutput(gzFile*, int32_t, gzFile, int32_t, int32_t);
int32_t RGMatchesMergeThreadTempFilesIntoOutputTempFile(gzFile*, int32_t, gzFile);
int32_t RGMatchesCompareAtIndex(RGMatches*, int32_t, RGMatches*, int32_t);
void RGMatchesAppend(RGMatches*, RGMatches*);
void RGMatchesAllocate(RGMatches*, int32_t);
void RGMatchesReallocate(RGMatches*, int32_t);
void RGMatchesFree(RGMatches*);
void RGMatchesInitialize(RGMatches*);
void RGMatchesMirrorPairedEnd(RGMatches*, RGBinary *rg, int32_t, int32_t, int32_t);
void RGMatchesCheck(RGMatches*, RGBinary*);
void RGMatchesFilterOutOfRange(RGMatches*, int32_t);
int32_t RGMatchesMergeIndexBins(gzFile*, int32_t, gzFile, RGIndex*, int32_t, int32_t); 

#endif

