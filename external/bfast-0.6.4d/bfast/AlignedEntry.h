#ifndef ALIGNEDENTRY_H_
#define ALIGNEDENTRY_H_

#include <stdio.h>
#include <zlib.h>
#include "BLibDefinitions.h"
#include "RGBinary.h"

int32_t AlignedEntryPrint(AlignedEntry*, gzFile);
int32_t AlignedEntryPrintText(AlignedEntry*, FILE*);
int32_t AlignedEntryRead(AlignedEntry*, gzFile);
int32_t AlignedEntryReadText(AlignedEntry*, FILE*);
void AlignedEntryQuickSort(AlignedEntry**, int32_t, int32_t, int32_t, int32_t, double*, int32_t);
void AlignedEntryShellSort(AlignedEntry**, int32_t, int32_t, int32_t);
void AlignedEntryCopyAtIndex(AlignedEntry*, int32_t, AlignedEntry*, int32_t);
int32_t AlignedEntryCompareAtIndex(AlignedEntry*, int32_t, AlignedEntry*, int32_t, int32_t);
int32_t AlignedEntryCompare(AlignedEntry*, AlignedEntry*, int32_t);
int32_t AlignedEntryGetOneRead(AlignedEntry**, FILE*);
int32_t AlignedEntryGetAll(AlignedEntry**, FILE*);
void AlignedEntryCopy(AlignedEntry*, AlignedEntry*);
void AlignedEntryFree(AlignedEntry*);
void AlignedEntryInitialize(AlignedEntry*);
void AlignedEntryCheckReference(AlignedEntry*, RGBinary*, int32_t);
int32_t AlignedEntryGetPivot(AlignedEntry*, int32_t, int32_t, int32_t);
void AlignedEntryUpdateAlignment(AlignedEntry *a, uint32_t, double, int32_t, int32_t, char*, char*);
int32_t AlignedEntryGetAlignment(AlignedEntry*, RGBinary*, char alignment[3][SEQUENCE_LENGTH], char*, int32_t, int32_t);

#endif
