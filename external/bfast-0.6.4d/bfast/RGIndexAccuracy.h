#ifndef RGINDEXACCURACY_H_
#define RGINDEXACCURACY_H_

#include "BLibDefinitions.h"

#define RGINDEXACCURACY_MIN_PERCENT_FOUND 95
#define RGINDEXACCURACY_NUM_TO_SAMPLE 100000 

/* Functions */
void RunSearchForRGIndexAccuracies(int, int, int, int, int, int, int, int, int, int);
void RunEvaluateRGIndexAccuracies(char*, int, int, int, int, int, int);
void RunEvaluateRGIndexAccuraciesNTSpace(RGIndexAccuracySet*, int, int, int, int);
void RunEvaluateRGIndexAccuraciesColorSpace(RGIndexAccuracySet*, int, int, int, int, int);
int32_t RunEvaluteRGIndexes(RGIndexAccuracySet*, int, int, int);
int32_t GetNumCorrect(RGIndexAccuracySet*, int, int, int, int, int, int, int);
/* RGIndexAccuracyMismatchProfile */
void RGIndexAccuracyMismatchProfileInitialize(RGIndexAccuracyMismatchProfile*);
void RGIndexAccuracyMismatchProfileAdd(RGIndexAccuracyMismatchProfile*, RGIndexAccuracySet*, int32_t, int32_t);
void RGIndexAccuracyMismatchProfilePrint(FILE*, RGIndexAccuracyMismatchProfile*);
void RGIndexAccuracyMismatchProfileRead(FILE*, RGIndexAccuracyMismatchProfile*);
void RGIndexAccuracyMismatchProfileFree(RGIndexAccuracyMismatchProfile*);
/* RGIndexAccuracySet functions */
int RGIndexAccuracySetContains(RGIndexAccuracySet*, RGIndexAccuracy*);
int32_t RGIndexAccuracySetCheckRead(RGIndexAccuracySet*, Read*);
void RGIndexAccuracySetPush(RGIndexAccuracySet*, RGIndexAccuracy*);
void RGIndexAccuracySetPop(RGIndexAccuracySet*);
void RGIndexAccuracySetSeed(RGIndexAccuracySet*, int);
void RGIndexAccuracySetInitialize(RGIndexAccuracySet*);
void RGIndexAccuracySetFree(RGIndexAccuracySet*);
void RGIndexAccuracySetPrint(RGIndexAccuracySet*, FILE*);
void RGIndexAccuracySetRead(RGIndexAccuracySet*, char*);
int32_t RGIndexAccuracySetCopyFrom(RGIndexAccuracySet*, RGIndex*, int32_t, int32_t);
/* RGIndexAccuracy functions */
void RGIndexAccuracySetReadFromRGIndexes(RGIndexAccuracySet*, char**, int, char**, int);
void RGIndexAccuracySetCopyFromRGIndex(RGIndexAccuracySet*, RGIndex*);
int RGIndexAccuracyCompare(RGIndexAccuracy*, RGIndexAccuracy*);
int32_t RGIndexAccuracyCheckRead(RGIndexAccuracy*, Read*);
void RGIndexAccuracyCopy(RGIndexAccuracy*, RGIndexAccuracy*);
void RGIndexAccuracyGetRandom(RGIndexAccuracy*, int, int);
void RGIndexAccuracyAllocate(RGIndexAccuracy*, int, int);
void RGIndexAccuracyInitialize(RGIndexAccuracy*);
void RGIndexAccuracyFree(RGIndexAccuracy*);
void RGIndexAccuracyPrint(RGIndexAccuracy*, FILE*);
int RGIndexAccuracyRead(RGIndexAccuracy*, FILE*);
void RGIndexAccuracyCopyFrom(RGIndexAccuracy*, RGIndex*, int32_t);
/* Accuracy Profile functions */
int AccuracyProfileCompare(RGIndexAccuracySet*, AccuracyProfile*, RGIndexAccuracySet*, AccuracyProfile*, int, int, int, int, int, int);
void AccuracyProfileCopy(AccuracyProfile*, AccuracyProfile*);
void AccuracyProfileAllocate(AccuracyProfile*, int, int, int);
void AccuracyProfileInitialize(AccuracyProfile*);
void AccuracyProfileFree(AccuracyProfile*);
/* Read functions */
void ReadSplit(Read*, Read*, Read*, int, int);
void ReadGetRandom(Read*, int, int, int, int);
void ReadInitialize(Read*);
void ReadAllocate(Read*, int);
void ReadFree(Read*);
void ReadPrint(Read*, FILE*);

#endif
