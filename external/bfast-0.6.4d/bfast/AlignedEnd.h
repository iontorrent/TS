#ifndef ALIGNEDEND_H_
#define ALIGNEDEND_H_

#include <stdio.h>
#include <zlib.h>

#include "BLibDefinitions.h"
#include "RGBinary.h"

int32_t AlignedEndPrint(AlignedEnd*, gzFile);
int32_t AlignedEndPrintText(AlignedEnd*, FILE*);
int32_t AlignedEndRead(AlignedEnd*, gzFile);
int32_t AlignedEndReadText(AlignedEnd*, FILE*);
int32_t AlignedEndRemoveDuplicates(AlignedEnd*, int32_t);
void AlignedEndQuickSort(AlignedEnd*, int32_t, int32_t);
int32_t AlignedEndCompare(AlignedEnd*, AlignedEnd*, int32_t);
void AlignedEndCopyAtIndex(AlignedEnd*, int32_t, AlignedEnd*, int32_t);
void AlignedEndCopy(AlignedEnd*, AlignedEnd*);
void AlignedEndAllocate(AlignedEnd*, char*, char*, int32_t);
void AlignedEndReallocate(AlignedEnd*, int32_t);
void AlignedEndFree(AlignedEnd*);
void AlignedEndInitialize(AlignedEnd*);
void AlignedEndUpdateMappingQuality(AlignedEnd*, double, int32_t);

#endif
