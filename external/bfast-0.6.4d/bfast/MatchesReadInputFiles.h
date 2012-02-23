#ifndef MATCHESREADINPUTFILES_H_
#define MATCHESREADINPUTFILES_H_

#include <stdio.h>
#include <zlib.h>
#include "RGMatches.h"
#include "RGIndex.h"
#include "aflib.h"

int WriteRead(FILE*, RGMatches*);
int WriteReadAFILE(AFILE*, RGMatches*);
void WriteReadsToTempFile(AFILE*, gzFile*, char**, int, int, char*, int*, int32_t);
int ReadTempReadsAndOutput(gzFile, char*, gzFile, AFILE*); 
void ReadRGIndex(char*, RGIndex*, int);
int GetIndexFileNames(char*, int32_t, char*, char***, int32_t***);
int32_t ReadOffsets(char*, int32_t**);
int32_t GetReads(gzFile, RGMatches*, int32_t, int32_t);

#endif
