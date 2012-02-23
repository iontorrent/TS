#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <zlib.h>
#include "BError.h"
#include "BLib.h"
#include "AlignedEnd.h"
#include "AlignedRead.h"

/* TODO */
void AlignedReadPrint(AlignedRead *a,
		gzFile outputFP)
{
	char *FnName = "AlignedReadPrint";
	int32_t i;

	assert(a!=NULL);
	a->readNameLength = (int)strlen(a->readName);
	if(gzwrite64(outputFP, &a->readNameLength, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(outputFP, a->readName, sizeof(char)*a->readNameLength)!=sizeof(char)*a->readNameLength ||
			gzwrite64(outputFP, &a->space, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(outputFP, &a->numEnds, sizeof(int32_t))!=sizeof(int32_t)) {
		PrintError(FnName, NULL, "Could not write to file", Exit, WriteFileError);
	}

	for(i=0;i<a->numEnds;i++) {
		if(EOF == AlignedEndPrint(&a->ends[i],
					outputFP)) {
			PrintError(FnName, "a->ends[i]", "Could not write to file", Exit, WriteFileError);
		}
	}
}

/* TODO */
void AlignedReadPrintText(AlignedRead *a,
		FILE *outputFP)
{
	char *FnName = "AlignedReadPrintText";
	int32_t i;

	/* Print the read name and paired end flag */
	if(fprintf(outputFP, "@%s\t%d\t%d\n",
				a->readName,
				a->space,
				a->numEnds) < 0) {
		PrintError(FnName, NULL, "Could not write to file", Exit, WriteFileError);
	}

	for(i=0;i<a->numEnds;i++) {
		if(EOF == AlignedEndPrintText(&a->ends[i],
					outputFP)) {
			PrintError(FnName, "a->ends[i]", "Could not write to file", Exit, WriteFileError);
		}
	}
}

/* TODO */
int32_t AlignedReadRead(AlignedRead *a,
		gzFile inputFP)
{
	char *FnName = "AlignedReadRead";
	int32_t i;

	assert(a != NULL);

	/* Allocate memory for the read name */
	a->readName = malloc(sizeof(char)*SEQUENCE_NAME_LENGTH);
	if(a->readName == NULL) {
		if(NULL == a->readName) {
			PrintError(FnName, "a->readName", "Could not allocate memory", Exit, MallocMemory);
		}
	}

	/* Read the read name, paired end flag, space flag, and the number of entries for both entries */
	if(gzread64(inputFP, &a->readNameLength, sizeof(int32_t))!=sizeof(int32_t)) {
		/* Free read name before leaving */
		free(a->readName);
		a->readName=NULL;
		return EOF;
	}
	if(gzread64(inputFP, a->readName, sizeof(char)*a->readNameLength)!=sizeof(char)*a->readNameLength||
			gzread64(inputFP, &a->space, sizeof(int32_t))!=sizeof(int32_t)||
			gzread64(inputFP, &a->numEnds, sizeof(int32_t))!=sizeof(int32_t)) {
		PrintError(FnName, NULL, "Could not read from file", Exit, ReadFileError);
	}
	/* Add the null terminator */
	a->readName[a->readNameLength]='\0';

	/* Reallocate to conserve memory */
	if(0 < a->readNameLength) {
		a->readName = realloc(a->readName, sizeof(char)*(a->readNameLength+1));
		if(NULL == a->readName) {
			PrintError(FnName, "a->readName", "Could not reallocate memory", Exit, ReallocMemory);
		}
	}
	else {
		free(a->readName);
		a->readName=NULL;
	}

	/* Allocate memory for the ends */ 
	a->ends = malloc(sizeof(AlignedEnd)*a->numEnds);
	if(NULL==a->ends) {
		PrintError(FnName, "a->ends", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Read the alignment */
	for(i=0;i<a->numEnds;i++) {
		AlignedEndInitialize(&a->ends[i]);
		if(EOF==AlignedEndRead(&a->ends[i],
					inputFP)) {
			PrintError(FnName, NULL, "Could not read a->ends[i]", Exit, EndOfFile);
		}
	}

	return 1;
}

/* TODO */
int32_t AlignedReadReadText(AlignedRead *a,
		FILE *inputFP) 
{
	char *FnName = "AlignedReadReadText";
	int32_t i;

	assert(a != NULL);

	/* Allocate memory for the read name */
	a->readName = malloc(sizeof(char)*SEQUENCE_NAME_LENGTH);
	if(a->readName == NULL) {
		if(NULL == a->readName) {
			PrintError(FnName, "a->readName", "Could not allocate memory", Exit, MallocMemory);
		}
	}

	/* Read the read name, paired end flag, space flag, and the number of entries for both entries */
	if(fscanf(inputFP, "@%s %d %d",
				a->readName,
				&a->space,
				&a->numEnds)<3) {
		/* Free read name before leaving */
		free(a->readName);
		a->readName=NULL;
		return EOF;
	}
	a->readNameLength = (int)strlen(a->readName);

	/* Reallocate to conserve memory */
	if(0 < a->readNameLength) {
		a->readName = realloc(a->readName, sizeof(char)*(a->readNameLength+1));
		if(NULL == a->readName) {
			PrintError(FnName, "a->readName", "Could not reallocate memory", Exit, ReallocMemory);
		}
	}
	else {
		free(a->readName);
		a->readName=NULL;
	}

	/* Allocate memory for the ends */ 
	a->ends = malloc(sizeof(AlignedEnd)*a->numEnds);
	if(NULL==a->ends) {
		PrintError(FnName, "a->ends", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Read the alignment */
	for(i=0;i<a->numEnds;i++) {
		AlignedEndInitialize(&a->ends[i]);
		if(EOF==AlignedEndReadText(&a->ends[i],
					inputFP)) {
			PrintError(FnName, NULL, "Could not read a->ends[i]", Exit, EndOfFile);
		}
	}

	return 1;
}

/* TODO */
void AlignedReadRemoveDuplicates(AlignedRead *a,
		int32_t sortOrder)
{
	int32_t i;
	/* First entry */
	for(i=0;i<a->numEnds;i++) {
		AlignedEndRemoveDuplicates(&a->ends[i],
				sortOrder);
	}
}

/* TODO */
void AlignedReadReallocate(AlignedRead *a,
		int32_t numEnds)
{
	char *FnName = "AlignedReadReallocate";
	int32_t i;

	/* we have to free if we are reducing the number of entries */
	if(numEnds < a->numEnds) {
		for(i=numEnds;i<a->numEnds;i++) {
			AlignedEndFree(&a->ends[i]);
		}
	}
	a->numEnds = numEnds;

	/* Allocate memory for the entries */ 
	a->ends = realloc(a->ends, sizeof(AlignedEnd)*a->numEnds);
	if(a->numEnds > 0 && NULL==a->ends) {
		if(NULL == a->ends) {
			PrintError(FnName, "a->ends", "Could not allocate memory", Exit, MallocMemory);
		}
	}
}

/* TODO */
void AlignedReadAllocate(AlignedRead *a,
		char *readName,
		int32_t numEnds,
		int32_t space)
{
	char *FnName = "AlignedReadAllocate";
	int32_t i;

	a->space = space;
	a->numEnds = numEnds;
	a->readNameLength = (int)strlen(readName);
	a->readName = malloc(sizeof(char)*(a->readNameLength+1));
	if(a->readName == NULL) {
		if(NULL == a->readName) {
			PrintError(FnName, "a->readName", "Could not allocate memory", Exit, MallocMemory);
		}
	}
	/* Copy over */
	strcpy(a->readName, readName);

	/* Allocate memory for the entries */ 
	a->ends = malloc(sizeof(AlignedEnd)*a->numEnds);
	if(0 < a->numEnds && a->ends == NULL) {

		PrintError(FnName, "a->ends", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Initialize */
	for(i=0;i<a->numEnds;i++) {
		AlignedEndInitialize(&a->ends[i]);
	}
}

/* TODO */
void AlignedReadFree(AlignedRead *a)
{
	int32_t i;
	for(i=0;i<a->numEnds;i++) {
		AlignedEndFree(&a->ends[i]);
	}
	free(a->ends);
	free(a->readName);
	AlignedReadInitialize(a);
}

/* TODO */
void AlignedReadInitialize(AlignedRead *a) 
{
	a->readNameLength=0;
	a->readName=NULL;
	a->numEnds=0;
	a->ends=NULL;
	a->space=NTSpace;
}

void AlignedReadCopy(AlignedRead *dest, AlignedRead *src) 
{
	int32_t i;

	/* Free and Allocate destination */
	AlignedReadFree(dest);
	AlignedReadAllocate(dest,
			src->readName,
			src->numEnds,
			src->space);
	/* Copy over */
	for(i=0;i<src->numEnds;i++) {
		AlignedEndCopy(&dest->ends[i], &src->ends[i]);
	}
}

/* TODO */
int32_t AlignedReadCompareAll(AlignedRead *one, AlignedRead *two)
{
	char *FnName="AlignedReadCompareAll";
	/* Compare by chr/pos */ 
	int32_t cmp=0;
	int32_t numLeft, i;
	int32_t minIndexOne, minIndexTwo;
	AlignedEnd **oneA=NULL;
	AlignedEnd **twoA=NULL;

	if(1 == one->numEnds &&
			1 == two->numEnds) {
		assert(1 == one->ends[0].numEntries);
		assert(1 == two->ends[0].numEntries);

		return AlignedEndCompare(&one->ends[0], &two->ends[0], AlignedEntrySortByContigPos);
	}
	else {
		assert(one->numEnds == two->numEnds);
		oneA = malloc(sizeof(AlignedEnd*)*one->numEnds);
		if(NULL == oneA) {
			PrintError(FnName, "oneA", "Could not allocate memory", Exit, MallocMemory);
		}
		for(i=0;i<one->numEnds;i++) {
			assert(1 == one->ends[i].numEntries);
			oneA[i] = &one->ends[i];
		}
		twoA = malloc(sizeof(AlignedEnd*)*two->numEnds);
		if(NULL == twoA) {
			PrintError(FnName, "twoA", "Could not allocate memory", Exit, MallocMemory);
		}
		for(i=0;i<two->numEnds;i++) {
			assert(1 == two->ends[i].numEntries);
			twoA[i] = &two->ends[i];
		}

		numLeft = one->numEnds;
		while(0 < numLeft) {
			/* Get min on one */
			minIndexOne=0;
			for(i=1;i<numLeft;i++) {
				if(0 < AlignedEndCompare(oneA[i], oneA[minIndexOne], AlignedEntrySortByContigPos)) {
					minIndexOne = i;
				}
			}
			/* Get min on two */
			minIndexTwo=0;
			for(i=1;i<numLeft;i++) {
				if(0 < AlignedEndCompare(twoA[i], twoA[minIndexTwo], AlignedEntrySortByContigPos)) {
					minIndexTwo = i;
				}
			}
			/* Compare */
			cmp = AlignedEndCompare(oneA[minIndexOne], twoA[minIndexTwo], AlignedEntrySortByContigPos);
			if(cmp != 0) {
				/* Exit out of the loop */
				numLeft=0;
			}
			else {
				numLeft--;
			}
			/* Remove */
			if(numLeft != minIndexOne) {
				oneA[minIndexOne] = oneA[numLeft];
				oneA[numLeft]=NULL;
			}
			if(numLeft != minIndexTwo) {
				twoA[minIndexTwo] = twoA[numLeft];
				twoA[numLeft]=NULL;
			}
		}

		free(oneA);
		free(twoA);

		return cmp;
	}
}

void AlignedReadUpdateMappingQuality(AlignedRead *a,
		double mismatchScore,
		int32_t avgMismatchQuality)
{
	int32_t i;

	for(i=0;i<a->numEnds;i++) {
		AlignedEndUpdateMappingQuality(&a->ends[i],
				mismatchScore,
				avgMismatchQuality);
	}
}
