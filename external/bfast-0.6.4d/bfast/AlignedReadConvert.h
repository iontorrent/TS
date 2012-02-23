#ifndef ALIGNEDREADCONVERT_H_
#define ALIGNEDREADCONVERT_H_

#include <stdlib.h>
#include <stdio.h>
#include <zlib.h>

#include "AlignedRead.h"
#include "AlignedEntry.h"
#include "BError.h"

void AlignedReadConvertPrintHeader(FILE*, RGBinary*, int, char*);
void AlignedReadConvertPrintOutputFormat(AlignedRead*, RGBinary*, FILE*, gzFile, char*, char*, int, int*, int, int);
void AlignedReadConvertPrintSAM(AlignedRead*, RGBinary*, int32_t, int32_t*, char*, char*, FILE*);
void AlignedReadConvertPrintAlignedEntryToSAM(AlignedRead*, RGBinary*, int32_t, int32_t, int32_t, int32_t*, char*, char*, FILE*);
void AlignedReadConvertPrintAlignedEntryToCIGAR(AlignedEntry*, char alignment[3][SEQUENCE_LENGTH], int32_t, int32_t, char*, char*, int32_t*, FILE*);

#endif
