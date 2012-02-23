#ifndef BLIB_H_
#define BLIB_H_

#include <stdio.h>
#include <stdint.h>
#include <zlib.h>
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include "RGIndex.h"
#include "BLibDefinitions.h"

extern char DNA[5];

int GetFastaHeaderLine(FILE*, char*);
int ParseFastaHeaderLine(char*);
char ToLower(char);
void ToLowerRead(char*, int);
char ToUpper(char);
void ToUpperRead(char*, int);
void ReverseRead(char*, char*, int);
void ReverseReadFourBit(int8_t*, int8_t*, int);
void GetReverseComplimentAnyCase(char*, char*, int);
char GetReverseComplimentAnyCaseBase(char);
void GetReverseComplimentFourBit(int8_t*, int8_t*, int);
int ValidateBasePair(char);
int IsAPowerOfTwo(unsigned int);
uint32_t Log2(uint32_t);
char TransformFromIUPAC(char);
void CheckRGIndexes(char**, int, char**, int, int32_t*, int32_t*, int32_t*, int32_t*, int32_t);
FILE *OpenTmpFile(char*, char**);
void CloseTmpFile(FILE **, char**);
gzFile OpenTmpGZFile(char*, char**);
void CloseTmpGZFile(gzFile*, char**, int32_t);
void ReopenTmpGZFile(gzFile*, char**);
void PrintPercentCompleteShort(double);
void PrintPercentCompleteLong(double);
int PrintContigPos(FILE*, int32_t, int32_t);
int32_t IsValidRead(RGMatches*, int);
int32_t IsValidMatch(RGMatches*);
int UpdateRead(char*, int, int);
int CheckReadAgainstIndex(RGIndex*, char*, int);
int CheckReadBase(char);
void ConvertSequenceToIntegers(char*, int8_t*, int32_t seqLength);
int32_t BaseToInt(char);
int ConvertBaseToColorSpace(char, char, char*);
int ConvertBaseAndColor(char, char, char*);
int ConvertReadFromColorSpace(char*, int);
void ConvertReadToColorSpace(char**, int*);
void NormalizeRead(char**, int*, char);
void NormalizeColorSpaceRead(char *read, int, char);
void ConvertColorsToStorage(char*, int);
char ConvertColorToStorage(char);
void ConvertColorsFromStorage(char*, int);
char ConvertColorFromStorage(char);
char ConvertIntColorToCharColor(char);
void AdjustBounds(RGBinary*, int32_t*, int32_t*, int32_t*, int32_t*);
int WillGenerateValidKey(RGIndex*, int8_t*, int);
int ValidatePath(char*);
int ValidateFileName(char*);
void StringCopyAndReallocate(char**, const char*);
int StringTrimWhiteSpace(char*);
int IsWhiteSpace(char);
void ParsePackageVersion(char*, int*, int*, int*);
void CheckPackageCompatibility(char*, int);
void KnuthMorrisPrattCreateTable(char*, int, int*);
int32_t KnuthMorrisPratt(char*, int, char*, int);
int32_t NaiveSubsequence(char*, int, char*, int);
int CompareContigPos(int32_t, int32_t, int32_t, int32_t);
int WithinRangeContigPos(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
char *StrStrGetLast(char*, const char*);
void ParseRange(Range*, char*);
int32_t CheckRange(Range*, int32_t, int32_t);
int32_t CheckRangeWithinRange(Range*, Range*);
void RangeCopy(Range*, Range*);
double AddLog10(double, double);
int64_t gzwrite64(gzFile, void*, int64_t);
int64_t gzread64(gzFile, void*, int64_t);
char *GetBRGFileName(char*, int32_t);
char *GetBIFName(char*, int32_t, int32_t, int32_t);
int32_t FileExists(char*);
int32_t GetBIFMaximumBin(char*, int32_t);
int32_t *GetNumbersFromString(char*, int32_t*);
#ifndef HAVE_STRTOK_R
char *strtok_r(char *s1, const char *s2, char **lasts);
#endif
char *ReadInReadGroup(char *);
char *ParseReadGroup(char*);

#endif
