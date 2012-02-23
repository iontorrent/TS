#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <limits.h>

#include "BError.h"
#include "BLibDefinitions.h"
#include "BLib.h"
#include "RGIndexAccuracy.h"

/* Is a utility that tests, searches for, and compares layouts for indexes against certain events,
 * such as errors, mismatches and insertions.
 * */

#define SAMPLE_ROTATE_NUM 10
char Colors[5] = "01234";

/* TODO */
void RunSearchForRGIndexAccuracies(int readLength,
		int numEventsToSample,
		int numRGIndexAccuraciesToSample,
		int keySize,
		int maxKeyWidth,
		int maxRGIndexAccuracySetSize,
		int accuracyThreshold,
		int space,
		int maxNumMismatches,
		int maxNumColorErrors)
{
	int i, j;
	RGIndexAccuracySet curSet, bestSet;
	RGIndexAccuracy curRGIndexAccuracy;
	AccuracyProfile curP, bestP;

	/* Seed random number */
	srand(time(NULL));

	/* Initialize index set */
	RGIndexAccuracySetInitialize(&curSet);
	RGIndexAccuracySetInitialize(&bestSet);

	/* Will always seed with contiguous 1s mask */
	RGIndexAccuracySetSeed(&curSet,
			keySize);
	RGIndexAccuracySetSeed(&bestSet,
			keySize);

	fprintf(stderr, "Currently on [index set size, sample number]\n0");
	for(i=2;i<=maxRGIndexAccuracySetSize;i++) { /* Add one index to the set */
		/* Initialize */
		RGIndexAccuracyInitialize(&curRGIndexAccuracy);
		AccuracyProfileInitialize(&curP);
		AccuracyProfileInitialize(&bestP);

		/* Sample the space of possible indexes */
		for(j=0;j<numRGIndexAccuraciesToSample;j++) { 
			if(j%SAMPLE_ROTATE_NUM == 0) {
				fprintf(stderr, "\r%-3d,%-9d",
						i,
						j);
			}
			/* Initialze cur */
			RGIndexAccuracyInitialize(&curRGIndexAccuracy);
			AccuracyProfileInitialize(&curP);

			/* Get random index */
			do {
				RGIndexAccuracyFree(&curRGIndexAccuracy);
				RGIndexAccuracyGetRandom(&curRGIndexAccuracy,
						keySize,
						maxKeyWidth);
			}
			while(1==RGIndexAccuracySetContains(&curSet, &curRGIndexAccuracy));

			/* Push random index onto the current set */
			RGIndexAccuracySetPush(&curSet,
					&curRGIndexAccuracy);
			/* Check if this is the first time */
			if(bestSet.numRGIndexAccuracies < curSet.numRGIndexAccuracies) {
				RGIndexAccuracySetPush(&bestSet, &curRGIndexAccuracy);
			}
			else {
				assert(j>0);
				assert(bestSet.numRGIndexAccuracies == curSet.numRGIndexAccuracies);
				/* Compare accuracy profile */
				if(AccuracyProfileCompare(&bestSet, 
							&bestP,
							&curSet, 
							&curP,
							readLength,
							numEventsToSample,
							space,
							maxNumMismatches,
							maxNumColorErrors,
							accuracyThreshold) < 0) {
					/* Copy index over to the current best set */
					RGIndexAccuracySetPop(&bestSet);
					RGIndexAccuracySetPush(&bestSet, &curRGIndexAccuracy);
					AccuracyProfileCopy(&bestP, &curP);
				}
			}
			/* Pop the index off the current set */
			RGIndexAccuracySetPop(&curSet);
			/* Free cur index */
			RGIndexAccuracyFree(&curRGIndexAccuracy);
			/* Free profile */
			AccuracyProfileFree(&curP);
		}
		/* Free accuracy profile */
		AccuracyProfileFree(&bestP);
		/* Copy best index over to cur set */
		RGIndexAccuracySetPush(&curSet, &bestSet.indexes[bestSet.numRGIndexAccuracies-1]);
	}
	fprintf(stderr, "\r--------------completed\n");

	/* Print */
	RGIndexAccuracySetPrint(&bestSet, stdout);

	/* Free */
	RGIndexAccuracySetFree(&curSet);
	RGIndexAccuracySetFree(&bestSet);
}

/* TODO */
void RunEvaluateRGIndexAccuracies(char *inputFileName,
		int readLength,
		int numEventsToSample,
		int space,
		int maxNumMismatches,
		int maxInsertionLength,
		int maxNumColorErrors)
{
	char *FnName="RunEvaluateRGIndexAccuracies";
	int setSize, i;
	int minKeySize, maxKeySize, maxKeyWidth;
	RGIndexAccuracySet set, curSet;

	assert(space == 1 || maxNumColorErrors == 0);

	/* Seed random number */
	srand(time(NULL));

	RGIndexAccuracySetInitialize(&curSet);

	/* Read in */
	RGIndexAccuracySetRead(&set, inputFileName);

	/* Get the min key size, max key size, and max key width */
	minKeySize = INT_MAX;
	maxKeySize = maxKeyWidth = 0;
	for(i=0;i<set.numRGIndexAccuracies;i++) {
		minKeySize = (set.indexes[i].keySize < minKeySize)?(set.indexes[i].keySize):minKeySize;
		maxKeySize = (set.indexes[i].keySize > maxKeySize)?(set.indexes[i].keySize):maxKeySize;
		maxKeyWidth = (set.indexes[i].keyWidth > maxKeyWidth)?(set.indexes[i].keyWidth):maxKeyWidth;
	}

	/* Print header of the file */
	if(minKeySize == maxKeySize) {
		fprintf(stdout, "BFAST MASK SET THEORETICAL ACCURACY\nREAD LENGTH = %d\nINDEX KEY SIZE = %d\nMAX WIDTH = %d\n",
				readLength,
				minKeySize,
				maxKeyWidth);
	}
	else {
		fprintf(stdout, "BFAST MASK SET THEORETICAL ACCURACY\nREAD LENGTH = %d\nMIN INDEX KEY SIZE = %d\nMAX INDEX KEY SIZE = %d\nMAX WIDTH = %d\n",
				readLength,
				minKeySize,
				maxKeySize,
				maxKeyWidth);
	}

	fprintf(stdout, "\n...MASKS IN BFAST FORMAT\n");
	RGIndexAccuracySetPrint(&set, stdout);
	fprintf(stdout, "\n");

	for(setSize=1;setSize<=set.numRGIndexAccuracies;setSize++) { /* For increasing set size */

		/* Add an index to the set */
		RGIndexAccuracySetPush(&curSet, &set.indexes[setSize-1]); 

		switch(space) {
			case NTSpace:
				RunEvaluateRGIndexAccuraciesNTSpace(&curSet,
						readLength,
						numEventsToSample,
						maxNumMismatches,
						maxInsertionLength);
				break;
			case ColorSpace:
				RunEvaluateRGIndexAccuraciesColorSpace(&curSet,
						readLength,
						numEventsToSample,
						maxNumMismatches,
						maxInsertionLength,
						maxNumColorErrors);
				break;
			default:
				PrintError(FnName, "space", "Could not understand space", Exit, OutOfRange);
		}
	}

	/* Free memory */
	RGIndexAccuracySetFree(&set);
	RGIndexAccuracySetFree(&curSet);
}

void RunEvaluateRGIndexAccuraciesNTSpace(RGIndexAccuracySet *set,
		int readLength,
		int numEventsToSample,
		int maxNumMismatches,
		int maxInsertionLength)
{
	int i, j;

	assert(numEventsToSample > 0);

	/* Print the header */
	fprintf(stdout, "N Masks = %d\n", set->numRGIndexAccuracies);
	fprintf(stdout, "%-5s\t", "MM"); /* # of Mismatches */
	fprintf(stdout, "%-5s\t", "BP:0"); /* Mismatches accuracy */
	fprintf(stdout, "%-5s\t", "1=del"); /* Deletions */
	for(j=1;j<=maxInsertionLength;j++) {
		fprintf(stdout, "%2s%-3d\t", "2i", j); /* Insertions */
	}
	fprintf(stdout, "\n");

	/* SNPs - include no SNPs */
	/* Mismatches including zero */
	for(i=0;i<=maxNumMismatches;i++) {
		fprintf(stdout, "%-5d\t", i);
		fprintf(stdout, "%1.3lf\t",
				GetNumCorrect(set,
					readLength,
					numEventsToSample,
					i,
					0,
					NoIndelType,
					0,
					NTSpace)/((double)numEventsToSample));
		/* Deletion with Mismatches */
		fprintf(stdout, "%1.3lf\t",
				GetNumCorrect(set,
					readLength,
					numEventsToSample,
					i,
					0,
					DeletionType,
					0,
					NTSpace)/((double)numEventsToSample));
		/* Insertions with Mismatches */
		for(j=1;j<=maxInsertionLength;j++) {
			fprintf(stdout, "%1.3lf\t",
					GetNumCorrect(set,
						readLength,
						numEventsToSample,
						i,
						0,
						InsertionType,
						j,
						NTSpace)/((double)numEventsToSample));
		}
		fprintf(stdout, "\n");
	}
	fflush(stdout);
}

void RunEvaluateRGIndexAccuraciesColorSpace(RGIndexAccuracySet *set,
		int readLength,
		int numEventsToSample,
		int maxNumMismatches,
		int maxInsertionLength,
		int maxNumColorErrors)
{
	int i, j;
	assert(numEventsToSample > 0);

	/* Print the header */
	fprintf(stdout, "N Masks = %d\n", set->numRGIndexAccuracies);
	fprintf(stdout, "%-5s\t", "CE"); /* # of Color Errors */
	for(j=0;j<=maxNumMismatches;j++) { /* # of SNPs */
		fprintf(stdout, "%-3s%-2d\t", "MM=", j);
	}
	fprintf(stdout, "%-5s\t", "1=del"); /* Deletions */
	for(j=1;j<=maxInsertionLength;j++) {
		fprintf(stdout, "%2s%-3d\t", "2i", j); /* Insertions */
	}
	fprintf(stdout, "\n");

	/* Get accuracy and print out */
	for(i=0;i<=maxNumColorErrors;i++) {
		fprintf(stdout, "%-5d\t", i);
		/* SNPs with color errors - include no SNPs */
		for(j=0;j<=maxNumMismatches;j++) {
			fprintf(stdout, "%1.3lf\t",
					GetNumCorrect(set,
						readLength,
						numEventsToSample,
						j,
						i,
						NoIndelType,
						0,
						ColorSpace)/((double)numEventsToSample));
		}
		/* Deletion with color errors */
		fprintf(stdout, "%1.3lf\t",
				GetNumCorrect(set,
					readLength,
					numEventsToSample,
					0,
					i,
					DeletionType,
					0,
					ColorSpace)/((double)numEventsToSample));
		/* Insertions with color errors */
		for(j=1;j<=maxInsertionLength;j++) {
			fprintf(stdout, "%1.3lf\t",
					GetNumCorrect(set,
						readLength,
						numEventsToSample,
						0,
						i,
						InsertionType,
						j,
						ColorSpace)/((double)numEventsToSample));
		}
		fprintf(stdout, "\n");
	}
}

int32_t RunEvaluateRGIndexes(RGIndexAccuracySet *set,
		int readLength,
		int numEventsToSample,
		int space)
{
	/*
	   char *FnName="RunEvaluateRGIndexes";
	   */
	int numMismatches = 0;
	int found = 0;
	int numCorrect = 0;

	/* Seed random number */
	srand(time(NULL));

	numMismatches=found=numCorrect=0;
	while(0 == found) {
		numCorrect = GetNumCorrect(set,
				readLength,
				numEventsToSample,
				(NTSpace==space)?(numMismatches+1):0,
				(ColorSpace==space)?(numMismatches+1):0,
				NoIndelType,
				0,
				space);
		if(((double)numCorrect*100)/numEventsToSample < RGINDEXACCURACY_MIN_PERCENT_FOUND) {
			found = 1;
		}
		else {
			numMismatches++;
		}
	}

	return numMismatches;
}

int32_t GetNumCorrect(RGIndexAccuracySet *set,
		int readLength,
		int numEventsToSample,
		int numSNPs,
		int numColorErrors,
		int indelType,
		int insertionLength,
		int space)
{
	char *FnName="GetNumCorrect";
	assert(space == 1 || numColorErrors == 0);
	assert(insertionLength <= 0 || indelType == InsertionType);

	int32_t i;
	int32_t numCorrect = 0;
	int32_t breakpoint;
	Read curRead, r1, r2;

	for(i=0;i<numEventsToSample;i++) {
		ReadInitialize(&curRead);
		ReadInitialize(&r1);
		ReadInitialize(&r2);
		/* Get random read with SNPs and ColorErrors */
		ReadGetRandom(&curRead,
				readLength,
				numSNPs,
				numColorErrors,
				space);
		/* Get the breakpoint:
		 * SNPs - no breakpoint (0)
		 * Deletion - breakpoint within the read 
		 * Insertion - breakpoint within the read, including start
		 * */
		switch(indelType) {
			case NoIndelType:
				/* Only SNPs and color errors */
				assert(insertionLength == 0);
				/* Check read */
				numCorrect += RGIndexAccuracySetCheckRead(set, &curRead);
				break;
			case DeletionType:
				assert(insertionLength == 0);
				/* Get where the break point occured for the deletion */
				breakpoint = ( rand()%(readLength - 1) ) + 1;
				assert(breakpoint > 0);
				assert(readLength - breakpoint > 0);
				/* Split read into two reads based on the breakpoint */
				ReadSplit(&curRead, &r1, &r2, breakpoint, 0);
				/* In color space, unless we are deleting at the end of a run of ex. As,
				 * the color at the break point represents the composition of the "deleted colors".
				 * Thus we should flip the color at the breakpoint */
				if(space==1) {
					/* A color error at the break point */
					if(r1.length > 0) {
						r1.profile[r1.length-1] = 1;
					}
				}
				/* Check either end of the read after split */
				if(1==RGIndexAccuracySetCheckRead(set, &r1) ||
						1==RGIndexAccuracySetCheckRead(set, &r2)) {
					numCorrect++;
				}
				/* Free read */
				ReadFree(&r1);
				ReadFree(&r2);
				break;
			case InsertionType:
				assert(insertionLength > 0);
				if(readLength > insertionLength) {
					/* Get where the insertion occured relative to the start of hte read */
					breakpoint = (rand()%(readLength-insertionLength));
					/* Split read into two reads */
					ReadSplit(&curRead, &r1, &r2, breakpoint, insertionLength);
					/* In color space, unless we are inserting at the end of a run of ex. As,
					 * an insertion of length "n" will cause "n+1" colors to be inserted.
					 * Thus we should flip the color before and after the breakpoint */
					if(space==1) {
						/* A color error before the break point */
						if(r1.length > 0) {
							r1.profile[r1.length-1] = 1;
						}
						/* A color error after the break point */
						if(r2.length > 0) {
							r2.profile[0] = 1;
						}
					}
					/* Check either end of the read after split, substracting the insertion */
					if(1==RGIndexAccuracySetCheckRead(set, &r1) ||
							1==RGIndexAccuracySetCheckRead(set, &r2)) {
						numCorrect++;
					}
					/* Free read */
					ReadFree(&r1);
					ReadFree(&r2);
				}
				break;
			default:
				PrintError(FnName, "indelType", "Could not understand indel type", Exit, OutOfRange);
		}
		/* Free read */
		ReadFree(&curRead);
	}
	return numCorrect;
}

void RGIndexAccuracyMismatchProfileInitialize(RGIndexAccuracyMismatchProfile *p) 
{
	p->maxReadLength = -1;
	p->maxMismatches = NULL;
}

void RGIndexAccuracyMismatchProfileAdd(RGIndexAccuracyMismatchProfile *p,
		RGIndexAccuracySet *set,
		int32_t readLength,
		int32_t space) 
{
	char *FnName="RGIndexAccuracyMismatchProfileAdd";
	int32_t i, prev;

	if(readLength <= p->maxReadLength &&
			0 < p->maxMismatches[readLength]) {
		/* Already computed */
		return;
	}
	else {
		if(p->maxReadLength < readLength) {
			prev = p->maxReadLength;
			p->maxReadLength = readLength;
			p->maxMismatches = realloc(p->maxMismatches, sizeof(int32_t)*(1+p->maxReadLength));
			if(NULL == p->maxMismatches) {
				PrintError(FnName, "p->maxMismatches", "Could not reallocate memory", Exit, ReallocMemory);
			}
			/* Initialize */
			for(i=prev+1;i<=p->maxReadLength;i++) {
				p->maxMismatches[i] = -1;
			}
		}
		p->maxMismatches[readLength] = RunEvaluateRGIndexes(set,  
				readLength,
				RGINDEXACCURACY_NUM_TO_SAMPLE,
				space);
	}
}

void RGIndexAccuracyMismatchProfilePrint(FILE *fp,
		RGIndexAccuracyMismatchProfile *p)
{
	char *FnName="RGIndexAccuracyMismatchProfilePrint";
	int32_t i;

	for(i=0;i<=p->maxReadLength;i++) {
		if(0 <= p->maxMismatches[i]) {
			if(fprintf(fp, "%d\t%d\n", i, p->maxMismatches[i]) < 0) {
				PrintError(FnName, "p->maxMismatches[i]", "Could not write to file", Exit, WriteFileError);
			}
		}
	}
}

void RGIndexAccuracyMismatchProfileRead(FILE *fp,
		RGIndexAccuracyMismatchProfile *p)
{
	char *FnName="RGIndexAccuracyMismatchProfileRead";
	int32_t i, mismatches, readLength, prev;

	while(EOF != fscanf(fp, "%d %d", &readLength, &mismatches)) {
		if(p->maxReadLength < readLength) {
			prev = p->maxReadLength;
			p->maxReadLength = readLength;
			p->maxMismatches = realloc(p->maxMismatches, sizeof(int32_t)*(1+p->maxReadLength));
			if(NULL == p->maxMismatches) {
				PrintError(FnName, "p->maxMismatches", "Could not reallocate memory", Exit, ReallocMemory);
			}
			/* Initialize */
			for(i=prev+1;i<=p->maxReadLength;i++) {
				p->maxMismatches[i] = -1;
			}
		}
		p->maxMismatches[readLength] = mismatches;
	}
}

void RGIndexAccuracyMismatchProfileFree(RGIndexAccuracyMismatchProfile *p) 
{
	free(p->maxMismatches);
	RGIndexAccuracyMismatchProfileInitialize(p);
}


void RGIndexAccuracySetReadFromRGIndexes(RGIndexAccuracySet *set,
		char **mainFileNames,
		int numMainFileNames,
		char **secondaryFileNames,
		int numSecondaryFileNames)
{
	char *FnName="RGIndexAccuracySetReadFromRGIndexes";
	RGIndex tempIndex;
	gzFile fp=NULL;
	int32_t i;

	/* Read in main indexes */
	for(i=0;i<numMainFileNames;i++) {
		/* Open file */
		if((fp=gzopen(mainFileNames[i], "r"))==0) {
			PrintError(FnName, mainFileNames[i], "Could not open file for reading", Exit, OpenFileError);
		}

		/* Get the header */
		RGIndexReadHeader(fp, &tempIndex);

		/* Add to index set */
		RGIndexAccuracySetCopyFromRGIndex(set, &tempIndex);

		/* Free masks */
		free(tempIndex.mask);
		tempIndex.mask=NULL;

		/* Close file */
		gzclose(fp);
	}

	/* Read in secondary indexes */
	for(i=0;i<numSecondaryFileNames;i++) {
		/* Open file */
		if((fp=gzopen(secondaryFileNames[i], "r"))==0) {
			PrintError(FnName, "secondaryFileNames[i]", "Could not open file for reading", Exit, OpenFileError);
		}

		/* Get the header */
		RGIndexReadHeader(fp, &tempIndex);

		/* Add to index set */
		RGIndexAccuracySetCopyFromRGIndex(set, &tempIndex);

		/* Free masks */
		free(tempIndex.mask);
		tempIndex.mask=NULL;

		/* Close file */
		gzclose(fp);
	}
}

void RGIndexAccuracySetCopyFromRGIndex(RGIndexAccuracySet *set,
		RGIndex *index) 
{
	char *FnName="RGIndexAccuracySetCopyFromRGIndex";
	int32_t i;
	set->numRGIndexAccuracies++;
	set->indexes = realloc(set->indexes, sizeof(RGIndexAccuracy)*set->numRGIndexAccuracies);
	if(NULL == set->indexes) {
		PrintError(FnName, "set->indexes", "Could not reallocate memory", Exit, ReallocMemory);
	}
	RGIndexAccuracyInitialize(&set->indexes[set->numRGIndexAccuracies-1]);
	/* Copy mask */
	set->indexes[set->numRGIndexAccuracies-1].keySize = index->keysize;
	set->indexes[set->numRGIndexAccuracies-1].keyWidth = index->width;
	set->indexes[set->numRGIndexAccuracies-1].mask = realloc(set->indexes[set->numRGIndexAccuracies-1].mask, 
			sizeof(int32_t)*set->indexes[set->numRGIndexAccuracies-1].keyWidth);
	if(NULL == set->indexes[set->numRGIndexAccuracies-1].mask) {
		PrintError(FnName, "set->indexes[set->numRGIndexAccuracies-1].mask", "Could not reallocate memory", Exit, ReallocMemory);
	}
	for(i=0;i<set->indexes[set->numRGIndexAccuracies-1].keyWidth;i++) {
		set->indexes[set->numRGIndexAccuracies-1].mask[i] = index->mask[i];
	}
}

int32_t RGIndexAccuracySetCopyFrom(RGIndexAccuracySet *r, RGIndex *indexes, int32_t numIndexes, int32_t keysize)
{
	char *FnName="RGIndexAccuracySetCopyFrom";
	int32_t i;

	r->numRGIndexAccuracies = numIndexes;

	/* Check valid keysize */
	for(i=0;i<numIndexes;i++) {
		if(keysize < indexes[i].hashWidth ||
				indexes[i].keysize < keysize) {
			return -1;
		}
	}

	r->indexes = malloc(sizeof(RGIndexAccuracy)*numIndexes);
	if(NULL == r->indexes) {
		PrintError(FnName, "r->indexes", "Could not allocate memory", Exit, MallocMemory);
	}

	for(i=0;i<numIndexes;i++) {
		RGIndexAccuracyCopyFrom(&r->indexes[i], &indexes[i], keysize);
	}

	return 1;
}

void RGIndexAccuracyCopyFrom(RGIndexAccuracy *r, RGIndex *index, int32_t keysize)
{
	char *FnName="RGIndexAccuracyCopyFrom";
	int32_t i, new_keysize, new_width = 0;

	assert(index->hashWidth <= keysize && keysize <= index->keysize);

	/* Get new keysize and width */
	if(keysize != index->keysize && 0 < keysize) {
		new_width = new_keysize = 0;
		while(new_keysize < keysize) {
			if(1 == index->mask[new_width]) {
				new_keysize++;
			}
			new_width++;
		}
		assert(new_keysize == keysize);
		r->keyWidth = new_width;
		r->keySize = new_keysize;
	}
	else {
		r->keyWidth = index->width;
		r->keySize = index->keysize;
	}

	r->mask = malloc(sizeof(int32_t)*r->keyWidth);
	if(NULL == r->mask) {
		PrintError(FnName, "r->mask", "Could not allocate memory", Exit, MallocMemory);
	}

	for(i=0;i<r->keyWidth;i++) {
		r->mask[i] = index->mask[i];
	}
}

int RGIndexAccuracySetContains(RGIndexAccuracySet *set,
		RGIndexAccuracy *index)
{
	int i;
	for(i=0;i<set->numRGIndexAccuracies;i++) {
		/* If they are the same, return 1 */
		if(RGIndexAccuracyCompare(index, &set->indexes[i])==0) {
			return 1;
		}
	}
	/* Return zero */
	return 0;
}

int32_t RGIndexAccuracySetCheckRead(RGIndexAccuracySet *set,
		Read *r)
{
	int i;
	for(i=0;i<set->numRGIndexAccuracies;i++) {
		if(RGIndexAccuracyCheckRead(&set->indexes[i],
					r) == 1) {
			return 1;
		}
	}
	return 0;
}

void RGIndexAccuracySetPush(RGIndexAccuracySet *set, 
		RGIndexAccuracy *index)
{
	char *FnName="RGIndexAccuracySetPush";
	set->numRGIndexAccuracies++;
	set->indexes = realloc(set->indexes, sizeof(RGIndexAccuracy)*set->numRGIndexAccuracies);
	if(NULL == set->indexes) {
		PrintError(FnName, "set->indexes", "Could not reallocate memory", Exit, ReallocMemory);
	}
	RGIndexAccuracyInitialize(&set->indexes[set->numRGIndexAccuracies-1]);
	RGIndexAccuracyCopy(&set->indexes[set->numRGIndexAccuracies-1], index);
}

void RGIndexAccuracySetPop(RGIndexAccuracySet *set)
{
	char *FnName="RGIndexAccuracySetPop";
	RGIndexAccuracyFree(&set->indexes[set->numRGIndexAccuracies-1]);
	set->numRGIndexAccuracies--;
	set->indexes = realloc(set->indexes, sizeof(RGIndexAccuracy)*set->numRGIndexAccuracies);
	if(NULL == set->indexes) {
		PrintError(FnName, "set->indexes", "Could not reallocate memory", Exit, ReallocMemory);
	}
}

/* Seed index set with one index with a contiguous mask */
void RGIndexAccuracySetSeed(RGIndexAccuracySet *set,
		int keySize)
{
	char *FnName="RGIndexAccuracySetSeed";
	int i;

	/* Allocate for the index set */
	set->numRGIndexAccuracies=1;
	set->indexes = malloc(sizeof(RGIndexAccuracy)*set->numRGIndexAccuracies);
	if(NULL == set->indexes) {
		PrintError(FnName, "set->indexes", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Allocate index */
	RGIndexAccuracyAllocate(&set->indexes[set->numRGIndexAccuracies-1],
			keySize,
			keySize);
	/* Initialize the mask to ones */
	for(i=0;i<set->indexes[set->numRGIndexAccuracies-1].keySize;i++) {
		set->indexes[set->numRGIndexAccuracies-1].mask[i] = 1;
	}
}

void RGIndexAccuracySetInitialize(RGIndexAccuracySet *set) 
{
	set->indexes=NULL;
	set->numRGIndexAccuracies=0;
}

void RGIndexAccuracySetFree(RGIndexAccuracySet *set) 
{
	int i;
	for(i=0;i<set->numRGIndexAccuracies;i++) {
		RGIndexAccuracyFree(&set->indexes[i]);
	}
	free(set->indexes);
	RGIndexAccuracySetInitialize(set);
}

void RGIndexAccuracySetPrint(RGIndexAccuracySet *set,
		FILE *fp)
{
	int i;
	for(i=0;i<set->numRGIndexAccuracies;i++) {
		RGIndexAccuracyPrint(&set->indexes[i],
				fp);
	}
}

int RGIndexAccuracyRead(RGIndexAccuracy *index, 
		FILE *fp)
{
	char *FnName = "RGIndexAccuracyRead";
	char tempMask[2056]="\0";
	int32_t i;

	/* Read */
	if(EOF == fscanf(fp, "%s", tempMask)) {
		return EOF;
	}
	RGIndexAccuracyAllocate(index,
			0,
			(int)strlen(tempMask));
	index->keySize = 0;
	for(i=0;i<index->keyWidth;i++) {
		switch(tempMask[i]) {
			case '0':
				index->mask[i] = 0;
				break;
			case '1':
				index->mask[i] = 1;
				index->keySize++;
				break;
			default:
				PrintError(FnName, "mask", "Could not read mask", Exit, OutOfRange);
		}
	}

	return 1;
}

void RGIndexAccuracySetRead(RGIndexAccuracySet *set,
		char *inputFileName)
{
	char *FnName="RGIndexAccuracySetRead";
	FILE *fp;
	RGIndexAccuracy index;

	if(!(fp=fopen(inputFileName, "rb"))) {
		PrintError(FnName, inputFileName, "Could not open file for reading", Exit, OpenFileError);
	}

	RGIndexAccuracySetInitialize(set);
	RGIndexAccuracyInitialize(&index);

	while(EOF!=RGIndexAccuracyRead(&index, fp)) {
		RGIndexAccuracySetPush(set, &index);
		RGIndexAccuracyFree(&index);
	}

	fclose(fp);
}

int RGIndexAccuracyCompare(RGIndexAccuracy *a, RGIndexAccuracy *b) 
{
	int i;
	if(a->keySize == b->keySize &&
			a->keyWidth == b->keyWidth) {
		/* key size and key width are the same */

		/* Compare masks */
		for(i=0;i<a->keyWidth;i++) {
			if(a->mask[i] != b->mask[i]) {
				/* Different */
				return 1;
			}
		}
		/* They must be the same */
		return 0;
	}
	else {
		return 1;
	}
}

int32_t RGIndexAccuracyCheckRead(RGIndexAccuracy *index,
		Read *r)
{
	int i, j;
	int success;

	if(index->keyWidth > r->length) {
		return 0;
	}

	for(i=0;i<r->length - index->keyWidth + 1;i++) { /* For all possible offsets */
		success = 1;
		for(j=0;1==success && j<index->keyWidth;j++) { /* Go over the index mask */
			if(index->mask[j] == 1 && r->profile[j+i] == 1) {
				success = 0;
			}
		}
		if(1==success) {
			return 1;
		}
	}
	return 0;
}

void RGIndexAccuracyCopy(RGIndexAccuracy *dest, RGIndexAccuracy *src)
{
	int i;

	if(NULL != dest->mask) {
		RGIndexAccuracyFree(dest);
	}

	RGIndexAccuracyAllocate(dest,
			src->keySize,
			src->keyWidth);
	for(i=0;i<src->keyWidth;i++) {
		dest->mask[i] = src->mask[i];
	}
}

void RGIndexAccuracyGetRandom(RGIndexAccuracy *index,
		int keySize,
		int maxKeyWidth)
{
	char *FnName="RGIndexAccuracyGetRandom";
	int i, j, k;
	int numLeft;
	int32_t *bins=NULL;
	int numBins = keySize-1;

	/* Generate random masks by having bins inbetween each "1".  We 
	 * then randomly insert zeros into the bins */

	/* Allocate memory for the bins */
	bins = malloc(sizeof(int32_t)*numBins);
	if(NULL == bins) {
		PrintError(FnName, "bins", "Could not allocate memory", Exit, OutOfRange);
	}
	/* Initialize bins */
	for(i=0;i<numBins;i++) {
		bins[i] = 0;
	}

	/* Choose a number of zeros to insert into the bins */
	numLeft = rand()%(maxKeyWidth - keySize + 1);
	assert(numLeft >=0 && numLeft <= maxKeyWidth - keySize);

	/* Allocate memory for the index */
	RGIndexAccuracyAllocate(index,
			keySize,
			keySize + numLeft);

	/* Insert into bins */
	while(numLeft > 0) {
		/* choose a bin between 1 and keySize-1 */
		i = (rand()%numBins); /* Note: this is not truly inform, but a good approximation */
		assert(i>=0 && i<numBins);
		bins[i]++;
		numLeft--;
	}

	/* Generate index based off the bins */
	/* First base is always a 1 */ 
	for(i=0, j=1, index->mask[0] = 1;
			i<index->keySize-1;
			i++, j++) {
		/* Insert zero based on the bin size */
		for(k=0;
				k<bins[i];
				k++, j++) {
			assert(j<index->keyWidth);
			index->mask[j] = 0;
		}
		/* Add a one */
		assert(j<index->keyWidth);
		index->mask[j] = 1;
	}
	assert(index->keyWidth == j);

	/* Free memory */
	free(bins);
	bins=NULL;
}

void RGIndexAccuracyAllocate(RGIndexAccuracy *index,
		int keySize,
		int keyWidth)
{
	char *FnName = "RGIndexAccuracyAllocate";
	if(NULL != index->mask) {
		RGIndexAccuracyFree(index);
	}
	index->keySize = keySize;
	index->keyWidth = keyWidth;
	index->mask = malloc(sizeof(int32_t)*index->keyWidth);
	if(NULL == index->mask) {
		PrintError(FnName, "index->mask", "Could not allocate memory", Exit, MallocMemory);
	}
}

void RGIndexAccuracyInitialize(RGIndexAccuracy *index)
{
	index->mask = NULL;
	index->keySize = 0;
	index->keyWidth = 0;
}

void RGIndexAccuracyFree(RGIndexAccuracy *index)
{
	free(index->mask);
	RGIndexAccuracyInitialize(index);
}

void RGIndexAccuracyPrint(RGIndexAccuracy *index,
		FILE *fp)
{
	int i;
	for(i=0;i<index->keyWidth;i++) {
		fprintf(fp, "%1d", index->mask[i]);
	}
	fprintf(fp, "\n");
}

int AccuracyProfileCompare(RGIndexAccuracySet *setA,
		AccuracyProfile *a,
		RGIndexAccuracySet *setB,
		AccuracyProfile *b,
		int readLength,
		int numEventsToSample,
		int space,
		int maxNumMismatches,
		int maxNumColorErrors,
		int accuracyThreshold)
{
	int i, j, ctr;

	/* Allocate memory for the profiles, if necessary */
	if(a->length == 0) {
		assert(a->accuracy == NULL);
		AccuracyProfileAllocate(a, maxNumMismatches, maxNumColorErrors, accuracyThreshold);
		a->numReads = numEventsToSample;
	}
	if(b->length == 0) {
		assert(b->accuracy == NULL);
		AccuracyProfileAllocate(b, maxNumMismatches, maxNumColorErrors, accuracyThreshold);
		b->numReads = numEventsToSample;
	}

	assert(a->accuracy != NULL);
	assert(b->accuracy != NULL);
	assert(a->length > 0);
	assert(b->length > 0);
	assert(a->length == b->length);
	assert(a->numColorErrors == b->numColorErrors);
	assert(a->numSNPs == b->numSNPs);
	assert(a->numReads > 0);
	assert(b->numReads > 0);
	assert(a->accuracyThreshold == accuracyThreshold);
	assert(b->accuracyThreshold == accuracyThreshold);
	assert(a->accuracyThreshold == b->accuracyThreshold);

	/* Optimization - check num above threshold */
	if(a->numAboveThreshold < b->numAboveThreshold) {
		return -1;
	}
	else if(a->numAboveThreshold > b->numAboveThreshold) {
		return 1;
	}
	else {
		assert(a->numAboveThreshold == b->numAboveThreshold);
		/* Must go through.  Start at those that are not above the accuracy threshold */
		for(i=0,ctr=0;i<=a->numColorErrors;i++) { /* color errors are prioritized */
			for(j=0;j<=a->numSNPs;j++) { /* SNPs are secondary */
				if(ctr >= a->numAboveThreshold) {
					/* Update accuracies if necessary */
					if(a->accuracy[ctr] < 0.0) {
						a->accuracy[ctr] = 100.0*GetNumCorrect(setA,
								readLength,
								numEventsToSample,
								j,
								i,
								NoIndelType,
								0,
								space)/a->numReads;
					}
					if(b->accuracy[ctr] < 0.0) {
						b->accuracy[ctr] = 100.0*GetNumCorrect(setB,
								readLength,
								numEventsToSample,
								j,
								i,
								NoIndelType,
								0,
								space)/b->numReads;
					}
					/* Compare */
					if(a->accuracy[ctr] < accuracyThreshold || b->accuracy[ctr] < accuracyThreshold) {
						if(a->accuracy[ctr] < b->accuracy[ctr]) {
							return -1;
						}
						else if(a->accuracy[ctr] > b->accuracy[ctr]) {
							return 1;
						}
					}
				}
				else {
					assert(!(a->accuracy[ctr] < 0.0));
					assert(!(b->accuracy[ctr] < 0.0));
				}
				ctr++;
			}
		}

		/* Equal */
		return 0;
	}
}

void AccuracyProfileCopy(AccuracyProfile *dest, AccuracyProfile *src) 
{
	int i;
	if(dest->accuracy != NULL) {
		AccuracyProfileFree(dest);
	}
	AccuracyProfileAllocate(dest, src->numSNPs, src->numColorErrors, src->accuracyThreshold);
	assert(dest->length == src->length);
	assert(dest->numSNPs == src->numSNPs);
	assert(dest->numColorErrors == src->numColorErrors);
	assert(dest->accuracyThreshold == src->accuracyThreshold);
	dest->numAboveThreshold = src->numAboveThreshold;
	dest->numReads = src->numReads;
	for(i=0;i<dest->length;i++) {
		dest->accuracy[i] = src->accuracy[i];
	}
}

void AccuracyProfileAllocate(AccuracyProfile *a,
		int numSNPs,
		int numColorErrors,
		int accuracyThreshold)
{
	char *FnName="AccuracyProfileAllocate";
	int i;
	if(a->accuracy != NULL) {
		AccuracyProfileFree(a);
	}
	AccuracyProfileInitialize(a);
	a->numSNPs = numSNPs;
	a->numColorErrors = numColorErrors;
	a->length = (a->numSNPs + 1)*(a->numColorErrors + 1);
	a->accuracy = malloc(sizeof(double)*a->length);
	a->numAboveThreshold = 0;
	a->accuracyThreshold = accuracyThreshold;
	if(NULL == a->accuracy) {
		PrintError(FnName, "a->accuracy", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Initialize all to -1 */
	for(i=0;i<a->length;i++) {
		a->accuracy[i] = -1.0;
	}
}

void AccuracyProfileInitialize(AccuracyProfile *a) 
{
	a->numReads = 0;
	a->accuracy = NULL;
	a->length = 0;
	a->numSNPs = -1;
	a->numColorErrors = -1;
	a->numAboveThreshold = 0;
	a->accuracyThreshold = 100;
}

void AccuracyProfileFree(AccuracyProfile *a)
{
	free(a->accuracy);
	AccuracyProfileInitialize(a);
}

void ReadSplit(Read *curRead,
		Read *r1,
		Read *r2,
		int breakpoint,
		int insertionLength)
{
	int i, ctr;

	/* Read 1 */
	if(breakpoint > 0) {
		ReadAllocate(r1, breakpoint);
	}
	for(i=0;i<breakpoint;i++) {
		assert(i < r1->length);
		r1->profile[i] = curRead->profile[i];
	}
	/* Read 2 */
	if(curRead->length - breakpoint - insertionLength > 0) {
		ReadAllocate(r2, curRead->length - breakpoint - insertionLength);
	}
	for(i=breakpoint+insertionLength, ctr=0;
			i<curRead->length;
			i++,ctr++) {
		assert(ctr < r2->length);
		r2->profile[ctr] = curRead->profile[i];
	}
}

/* Zeros are valid bases, Ones we can't index */
void ReadGetRandom(Read *r, 
		int readLength,
		int numSNPs,
		int numColorErrors,
		int space)
{
	char *FnName="ReadGetRandom";
	int i;
	int numSNPsLeft = numSNPs;
	int numColorErrorsLeft = numColorErrors;
	int index;
	char original;
	char *read=NULL;
	char *originalColorSpace = NULL;
	int tmpReadLength = 0;

	assert(numSNPs <= readLength);
	assert(numColorErrors <= readLength);

	/* Allocate memory for a read */
	ReadAllocate(r, readLength);

	/* Initialize to no SNPs or color errors */
	for(i=0;i<r->length;i++) {
		r->profile[i] = 0;
	}

	assert(space == 1 || numColorErrors == 0);
	if(space == 0) {
		/* Insert random SNPS */
		while(numSNPsLeft > 0) {
			/* Pick a position to convert */
			index = rand()%(r->length);

			if(r->profile[index] == 0) {
				r->profile[index] = 1;
				numSNPsLeft--;
			}
		}
	}
	else {

		read = malloc(sizeof(char)*(readLength+1));
		if(NULL == read) {
			PrintError(FnName, "read", "Could not allocate memory", Exit, MallocMemory);
		}
		originalColorSpace = malloc(sizeof(char)*(readLength+1));
		if(NULL == originalColorSpace) {
			PrintError(FnName, "originalColorSpace", "Could not allocate memory", Exit, MallocMemory);
		}

		/* Get a random NT read */
		for(i=0;i<readLength;i++) {
			read[i] = DNA[rand()%4];
		}
		read[readLength]='\0';

		/* Get the color space of the original read */
		strcpy(originalColorSpace, read);
		tmpReadLength = readLength;
		ConvertReadToColorSpace(&originalColorSpace,
				&tmpReadLength);

		/* Insert random SNPs */
		while(numSNPsLeft > 0) {
			/* Pick a position to convert */
			index = rand()%(r->length);

			if(r->profile[index] == 0) {
				r->profile[index] = 1;
				numSNPsLeft--;
				/* Modify base to a new base */
				for(original = read[index];
						original == read[index];
						read[index] = DNA[rand()%4]) {
				}
			}
		}
		/* Convert to color space */
		tmpReadLength = readLength;
		ConvertReadToColorSpace(&read,
				&tmpReadLength);
		/* Insert color errors */
		while(numColorErrorsLeft > 0) {
			/* Pick a position to convert */
			index = rand()%(r->length);

			if(r->profile[index] != 2) {
				r->profile[index] = 2;
				numColorErrorsLeft--;
				/* Modify base to a new color */
				for(original = read[index];
						original == read[index];
						read[index] = Colors[rand()%4]) {
				}
			}
		}
		/* Compare the two profiles to get an end profile */
		for(i=0;i<r->length;i++) {
			if(originalColorSpace[i+1] == read[i+1]) {
				r->profile[i] = 0;
			}
			else {
				r->profile[i] = 1;
			}
		}

		free(read);
		read = NULL;
		free(originalColorSpace);
		originalColorSpace = NULL;
	}
}

void ReadInitialize(Read *r)
{
	r->length = 0;
	r->profile = NULL;
}

void ReadAllocate(Read *r, 
		int readLength)
{
	char *FnName = "ReadAllocate";
	r->length = readLength;
	r->profile = malloc(sizeof(int32_t)*r->length);
	if(NULL == r->profile) {
		PrintError(FnName, "r->profile", "Could not allocate memory", Exit, MallocMemory);
	}
}

void ReadFree(Read *r)
{
	free(r->profile);
	ReadInitialize(r);
}

void ReadPrint(Read *r, FILE *fp) 
{
	int i;
	for(i=0;i<r->length;i++) {
		fprintf(fp, "%1d", r->profile[i]);
	}
	fprintf(fp, "\n");
}
