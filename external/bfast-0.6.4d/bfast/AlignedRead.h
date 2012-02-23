#ifndef ALIGNEDREAD_H_
#define ALIGNEDREAD_H_

#include <stdio.h>
#include <zlib.h>
#include "BLibDefinitions.h"
#include "RGBinary.h"

void AlignedReadPrint(AlignedRead*, gzFile);
void AlignedReadPrintText(AlignedRead*, FILE*);
int AlignedReadRead(AlignedRead*, gzFile);
int AlignedReadReadText(AlignedRead*, FILE*);
void AlignedReadRemoveDuplicates(AlignedRead*, int32_t);
void AlignedReadReallocate(AlignedRead*, int32_t);
void AlignedReadAllocate(AlignedRead*, char*, int32_t, int32_t);
void AlignedReadFree(AlignedRead*);
void AlignedReadInitialize(AlignedRead*);
void AlignedReadCopy(AlignedRead*, AlignedRead*);
int32_t AlignedReadCompareAll(AlignedRead*, AlignedRead*);
void AlignedReadUpdateMappingQuality(AlignedRead*, double, int32_t);
#endif

