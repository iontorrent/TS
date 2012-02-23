#ifndef RGINDEX_H_
#define RGINDEX_H_

#include <stdio.h>
#include <zlib.h>
#include "RGBinary.h"
#include "RGRanges.h"
#include "BLibDefinitions.h"

void RGIndexCreate(char*, RGIndexLayout*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, RGIndexExons*, int32_t, int32_t, int32_t, char*);
void RGIndexCreateSingle(char*, RGIndexLayout*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, RGIndexExons*, int32_t, int32_t, int32_t, char*);
void RGIndexCreateSplit(char*, RGIndexLayout*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, RGIndexExons*, int32_t, int32_t, int32_t, char*);
void RGIndexCreateHelper(RGIndex*, RGBinary*, FILE**, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
void RGIndexCreateHash(RGIndex*, RGBinary*);
void RGIndexSort(RGIndex*, RGBinary*, int32_t, char*);
void *RGIndexMergeSort(void*);
void RGIndexMergeSortHelper(RGIndex*, RGBinary*, int64_t, int64_t, int32_t, double*, int64_t, int64_t, int64_t, char*);
void RGIndexShellSort(RGIndex*, RGBinary*, int64_t, int64_t);
void *RGIndexMerge(void*);
void RGIndexMergeHelper(RGIndex*, RGBinary*, int64_t, int64_t, int64_t, int64_t, char*);
void RGIndexMergeHelperInMemoryContig_8(RGIndex*, RGBinary*, int64_t, int64_t, int64_t);
void RGIndexMergeHelperInMemoryContig_32(RGIndex*, RGBinary*, int64_t, int64_t, int64_t);
void RGIndexMergeHelperFromDiskContig_8(RGIndex*, RGBinary*, int64_t, int64_t, int64_t, char*);
void RGIndexMergeHelperFromDiskContig_32(RGIndex*, RGBinary*, int64_t, int64_t, int64_t, char*);

void RGIndexDelete(RGIndex*);
double RGIndexGetSize(RGIndex*, int32_t);
void RGIndexPrint(gzFile, RGIndex*);
void RGIndexRead(RGIndex*, char*);
void RGIndexPrintInfo(char*);
void RGIndexPrintHeader(gzFile, RGIndex*);
void RGIndexGetHeader(char*, RGIndex*);
void RGIndexReadHeader(gzFile, RGIndex*);
int64_t RGIndexGetRanges(RGIndex*, RGBinary*, int8_t*, int32_t, int64_t*, int64_t*);
int32_t RGIndexGetRangesBothStrands(RGIndex*, RGBinary*, int8_t*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, RGRanges*);
int64_t RGIndexGetIndex(RGIndex*, RGBinary*, int8_t*, int32_t, int64_t*, int64_t*);
void RGIndexSwapAt(RGIndex*, int64_t, int64_t);
int64_t RGIndexGetPivot(RGIndex*, RGBinary*, int64_t, int64_t);
int32_t RGIndexCompareContigPos(RGIndex*, RGBinary*, uint32_t, uint32_t, uint32_t, uint32_t, int);
int32_t RGIndexCompareAt(RGIndex*, RGBinary*, int64_t, int64_t, int);
int32_t RGIndexCompareRead(RGIndex*, RGBinary*, int8_t*, int64_t, int32_t, int32_t*, int);
uint32_t RGIndexGetHashIndex(RGIndex*, RGBinary*, uint32_t, int);
uint32_t RGIndexGetHashIndexFromRead(RGIndex*, RGBinary*, int8_t*, int32_t, int);
void RGIndexPrintReadMasked(RGIndex*, char*, int, FILE*);
void RGIndexInitialize(RGIndex*);
void RGIndexInitializeFull(RGIndex*, RGBinary*, RGIndexLayout*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
gzFile RGIndexOpenForWriting(char*, RGIndex*);
#endif

