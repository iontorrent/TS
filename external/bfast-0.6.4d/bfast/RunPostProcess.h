#ifndef RUNPOSTPROCESS_H_
#define RUNPOSTPROCESS_H_

#include "AlignedRead.h"


/* Paired End Distance Bins */
// This distance of the second end minus the first end
typedef struct {
	int32_t minDistance;
	int32_t maxDistance;
	int32_t bins[MAX_PEDBINS_DISTANCE - MIN_PEDBINS_DISTANCE + 1];
	int32_t numDistances;
	double std;
	double avg;
	int32_t inversionCount;
	double invRatio;
} PEDBins;

typedef struct {
	PEDBins *bins;
	int algorithm;
	int unpaired;
	int reversePaired;
	int avgMismatchQuality;
	int randomBest;
	int mismatchScore;
	int queueLength;
	int8_t *foundTypes;
	AlignedRead *alignQueue;
	int32_t *alignQueueThreadIDs;
	int32_t **numEntries;
	int32_t *numEntriesN;
	int32_t numThreads;
	int32_t threadID;
} PostProcessThreadData;

void ReadInputFilterAndOutput(RGBinary *rg,
		char *inputFileName,
		int algorithm,
		int space,
		int unpaired,
		int reversePaired,
		int avgMismatchQuality,
		char *scoringMatrixFileName,
	int randomBest,
		int numThreads,
		int queueLength,
		int outputFormat,
		char *outputID,
		char *readGroup,
		char *unmappedFileName,
		FILE *fpOut);

void *ReadInputFilterAndOutputThread(void*);

int32_t GetPEDBins(char*, int, int, PEDBins*);

int32_t GetAlignedReads(gzFile, AlignedRead*, int32_t);

int FilterAlignedRead(AlignedRead *a,
		int algorithm,
		int unpaired,
		int reversePaired,
		int avgMismatchQuality,
		int randomBest,
		int mismatchScore,
		PEDBins *b);

void PEDBinsInitialize(PEDBins*);
void PEDBinsFree(PEDBins*);
void PEDBinsInsert(PEDBins*, char, char, int32_t);
void PEDBinsPrintStatistics(PEDBins*, FILE*);

#endif
