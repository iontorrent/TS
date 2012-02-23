#ifndef BINDEXDIST_H_
#define BINDEXDIST_H_

#include "../bfast/RGIndex.h"
#include "../bfast/RGBinary.h"
#include "../bfast/RGMatch.h"

typedef struct {
	char **reads;
	int64_t *readCounts;
	int64_t low;
	int64_t high;
	char *tmpDir;
	int readLength;
	int showPercentComplete;
	int threadID;
} ThreadData;

void PrintDistribution(RGIndex*, RGBinary*, int, char*, int);
void GetMatchesFromContigPos(RGIndex*, RGBinary*, uint32_t, uint32_t, int64_t*, int64_t*, char**, char**);
void *MergeSortReads(void *arg);
void MergeSortReadsHelper(char**, int64_t*, int64_t, int64_t, int64_t, int64_t, int, double*, char*, int);
void MergeHelper(char**, int64_t*, int64_t, int64_t, int64_t, char*, int);

#endif
