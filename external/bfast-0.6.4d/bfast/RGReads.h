#ifndef RGREADS_H_
#define RGREADS_H_

#include <stdio.h>
#include "BLibDefinitions.h"
#include "RGMatch.h"
#include "RGIndex.h"

void RGReadsFindMatches(RGIndex*, RGBinary*, RGMatch*, int, int*, int, int, int, int, int, int, int, int, int, int);
void RGReadsGenerateReads(char*, int, RGIndex*, RGReads*, int*, int, int, int, int, int, int, int);
void RGReadsGeneratePerfectMatch(char*, int, int, RGIndex*, RGReads*);
void RGReadsGenerateMismatches(char*, int, int, int, RGIndex*, RGReads*);
void RGReadsGenerateMismatchesHelper(char*, int, int, int, char*, int, RGIndex*, RGReads*);
void RGReadsGenerateDeletions(char*, int, int, int, RGIndex*, RGReads*);
void RGReadsGenerateDeletionsHelper(char*, int, int, int, int, int, char*, int, RGIndex*, RGReads*);
void RGReadsGenerateInsertions(char*, int, int, int, RGIndex*, RGReads*);
void RGReadsGenerateInsertionsHelper(char*, int, int, int, int, int, char*, int, RGIndex*, RGReads*);
void RGReadsGenerateGapDeletions(char*, int, int, int, RGIndex*, RGReads*);
void RGReadsGenerateGapDeletionsHelper(char*, int, int, int, char*, RGIndex*, RGReads*);
void RGReadsGenerateGapInsertions(char*, int, int, int, RGIndex*, RGReads*);
void RGReadsGenerateGapInsertionsHelper(char*, int, int, int, char*, RGIndex*, RGReads*);
void RGReadsRemoveDuplicates(RGReads*);
void RGReadsQuickSort(RGReads*, int, int);
void RGReadsShellSort(RGReads*, int, int);
int RGReadsCompareAtIndex(RGReads*, int, RGReads*, int);
void RGReadsCopyAtIndex(RGReads*, int, RGReads*, int);
void RGReadsAllocate(RGReads*, int);
void RGReadsReallocate(RGReads*, int);
void RGReadsFree(RGReads*);
void RGReadsInitialize(RGReads*);
void RGReadsAppend(RGReads*, char*, int32_t, int32_t);
void RGReadsPrint(RGReads*, RGIndex*);

#endif

