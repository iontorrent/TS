#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <string.h>

#include "BLib.h"
#include "BLibDefinitions.h"
#include "BError.h"
#include "RGMatch.h"

/* TODO */
int32_t RGMatchRead(gzFile fp,
		RGMatch *m)
{
	char *FnName = "RGMatchRead";
	int32_t i, numEntries;

	/* Read in the read length */
	if(gzread64(fp, &m->readLength, sizeof(int32_t))!=sizeof(int32_t)||
			gzread64(fp, &m->qualLength, sizeof(int32_t))!=sizeof(int32_t)) {
		if(feof(fp) != 0) {
			return EOF;
		}
		else {
			PrintError(FnName, "m->readLength", "Could not read in read length", Exit, ReadFileError);
		}
	}
	assert(m->readLength < SEQUENCE_LENGTH);
	assert(m->readLength > 0);

	/* Allocate memory for the read */
	m->read = malloc(sizeof(char)*(m->readLength+1));
	if(NULL==m->read) {
		PrintError(FnName, "read", "Could not allocate memory", Exit, MallocMemory);
	}
	m->qual = malloc(sizeof(char)*(m->qualLength+1));
	if(NULL==m->qual) {
		PrintError(FnName, "qual", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Read in the read */
	if(gzread64(fp, m->read, sizeof(char)*m->readLength)!=sizeof(char)*m->readLength||
			gzread64(fp, m->qual, sizeof(char)*m->qualLength)!=sizeof(char)*m->qualLength) {
		PrintError(FnName, "m->read", "Could not read in the read and qual", Exit, ReadFileError);
	}
	m->read[m->readLength]='\0';
	m->qual[m->qualLength]='\0';

	/* Read in if we have reached the maximum number of matches */
	if(gzread64(fp, &m->maxReached, sizeof(int32_t))!=sizeof(int32_t)) {
		PrintError(FnName, "m->maxReached", "Could not read in m->maxReached", Exit, ReadFileError);
	}
	assert(0 == m->maxReached || 1 == m->maxReached);

	/* Read in the number of matches */
	if(gzread64(fp, &numEntries, sizeof(int32_t))!=sizeof(int32_t)) {
		PrintError(FnName, "numEntries", "Could not read in numEntries", Exit, ReadFileError);
	}
	assert(numEntries >= 0);

	/* Allocate memory for the matches */
	RGMatchAllocate(m, numEntries);

	/* Read first sequence matches */
	if(gzread64(fp, m->contigs, sizeof(uint32_t)*m->numEntries)!=sizeof(uint32_t)*m->numEntries) {
		PrintError(FnName, "m->contigs", "Could not read in contigs", Exit, ReadFileError);
	}
	if(gzread64(fp, m->positions, sizeof(int32_t)*m->numEntries)!=sizeof(int32_t)*m->numEntries) {
		PrintError(FnName, "m->positions", "Could not read in positions", Exit, ReadFileError);
	}
	if(gzread64(fp, m->strands, sizeof(char)*m->numEntries)!=sizeof(char)*m->numEntries) {
		PrintError(FnName, "m->strands", "Could not read in strand", Exit, ReadFileError);
	}
	for(i=0;i<m->numEntries;i++) {
		if(gzread64(fp, m->masks[i], sizeof(char)*GETMASKNUMBYTES(m))!=sizeof(char)*GETMASKNUMBYTES(m)) {
			PrintError(FnName, "m->masks[i]", "Could not read in mask", Exit, ReadFileError);
		}
	}

	return 1;
}

/* TODO */
int32_t RGMatchReadText(FILE *fp,
		RGMatch *m)
{
	char *FnName = "RGMatchRead";
	int32_t i, numEntries;
	char read[SEQUENCE_LENGTH]="\0";
	char qual[SEQUENCE_LENGTH]="\0";
	char mask[SEQUENCE_LENGTH]="\0";

	/* Read the read and qual */
	if(fscanf(fp, "%s %s",
				read,
				qual)==EOF) {
		return EOF;
	}
	m->readLength = strlen(read);
	m->qualLength = strlen(qual);
	assert(m->readLength > 0);
	assert(m->readLength < SEQUENCE_LENGTH);

	/* Allocate memory for the read */
	m->read = malloc(sizeof(char)*(m->readLength+1));
	if(NULL==m->read) {
		PrintError(FnName, "read", "Could not allocate memory", Exit, MallocMemory);
	}
	m->qual = malloc(sizeof(char)*(m->qualLength+1));
	if(NULL==m->qual) {
		PrintError(FnName, "qual", "Could not allocate memory", Exit, MallocMemory);
	}
	strcpy(m->read, read);
	strcpy(m->qual, qual);

	/* Read in if we have reached the maximum number of matches */
	if(fscanf(fp, "%d", &m->maxReached)==EOF) {
		PrintError(FnName, "m->maxReached", "Could not read in m->maxReached", Exit, EndOfFile);
	}
	assert(1==m->maxReached || 0 == m->maxReached);

	/* Read in the number of matches */
	if(fscanf(fp, "%d", &numEntries)==EOF) {
		PrintError(FnName, "numEntries", "Could not read in numEntries", Exit, EndOfFile);
	}
	assert(numEntries >= 0);

	/* Allocate memory for the matches */
	RGMatchAllocate(m, numEntries);

	/* Read first sequence matches */
	for(i=0;i<m->numEntries;i++) {
		if(fscanf(fp, "%u %d %c %s", 
					&m->contigs[i],
					&m->positions[i],
					&m->strands[i],
					mask)==EOF) {
			PrintError(FnName, NULL, "Could not read in match", Exit, EndOfFile);
		}
		free(m->masks[i]); // since we reallocated
		m->masks[i] = RGMatchStringToMask(mask, m->readLength);
	}

	return 1;
}

/* TODO */
void RGMatchPrint(gzFile fp,
		RGMatch *m)
{
	char *FnName = "RGMatchPrint";
	assert(fp!=NULL);
	assert(m->readLength > 0);
	assert(m->qualLength > 0);
	int32_t i;

	/* Print the matches to the output file */
	/* Print read length, read, maximum reached, and number of entries. */
	if(gzwrite64(fp, &m->readLength, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &m->qualLength, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, m->read, sizeof(char)*m->readLength)!=sizeof(char)*m->readLength ||
			gzwrite64(fp, m->qual, sizeof(char)*m->qualLength)!=sizeof(char)*m->qualLength ||
			gzwrite64(fp, &m->maxReached, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &m->numEntries, sizeof(int32_t))!=sizeof(int32_t)) {
		PrintError(FnName, NULL, "Could not write m->readLength, m->qualLength, m->read, m->qual, m->maxReached, and m->numEntries", Exit, WriteFileError);
	}

	/* Print the contigs, positions, and strands */
	if(gzwrite64(fp, m->contigs, sizeof(uint32_t)*m->numEntries)!=sizeof(uint32_t)*m->numEntries ||
			gzwrite64(fp, m->positions, sizeof(int32_t)*m->numEntries)!=sizeof(int32_t)*m->numEntries ||
			gzwrite64(fp, m->strands, sizeof(char)*m->numEntries)!=sizeof(char)*m->numEntries) {
		PrintError(FnName, NULL, "Could not write contigs, positions and strands", Exit, WriteFileError);
	}
	for(i=0;i<m->numEntries;i++) {
		if(gzwrite64(fp, m->masks[i], sizeof(char)*GETMASKNUMBYTES(m))!=sizeof(char)*GETMASKNUMBYTES(m)) {
			PrintError(FnName, NULL, "Could not write masks[i]", Exit, WriteFileError);
		}
	}
}

/* TODO */
void RGMatchPrintText(FILE *fp,
		RGMatch *m)
{
	char *FnName = "RGMatchPrint";
	int32_t i;
	char *maskString=NULL;
	assert(fp!=NULL);
	assert(m->readLength > 0);
	assert(m->qualLength > 0);

	/* Print the matches to the output file */
	if(0 > fprintf(fp, "%s\t%s\t%d\t%d",
				m->read,
				m->qual,
				m->maxReached,
				m->numEntries)) {
		PrintError(FnName, NULL, "Could not write m->read, m->qual, m->maxReached, and m->numEntries", Exit, WriteFileError);
	}

	for(i=0;i<m->numEntries;i++) {
		assert(m->contigs[i] > 0);
		maskString=RGMatchMaskToString(m->masks[i], m->readLength);
		if(0 > fprintf(fp, "\t%u\t%d\t%c\t%s",
					m->contigs[i],
					m->positions[i],
					m->strands[i],
					maskString)) {
			PrintError(FnName, NULL, "Could not write m->contigs[i], m->positions[i], m->strands[i], and m->masks[i]", Exit, WriteFileError);
		}
		free(maskString);
		maskString=NULL;
	}
	if(0 > fprintf(fp, "\n")) {
		PrintError(FnName, NULL, "Could not write newline", Exit, WriteFileError);
	}
}

/* TODO */
void RGMatchPrintFastq(FILE *fp,
		char *readName,
		RGMatch *m)
{
	char *FnName = "RGMatchPrintFastq";
	assert(fp!=NULL);
	assert(m->readLength > 0);
	assert(m->qualLength > 0);

	if(0 > fprintf(fp, "@%s\n%s\n+\n%s\n",
				readName,
				m->read,
				m->qual)) {
		PrintError(FnName, NULL, "Could not to file", Exit, WriteFileError);
	}
}

/* TODO */
void RGMatchRemoveDuplicates(RGMatch *m,
		int32_t maxNumMatches)
{
	int32_t i;
	int32_t prevIndex=0;

	/* Check to see if the max has been reached.  If so free all matches and return.
	 * We should remove duplicates before checking against maxNumMatches. */
	if(1 == m->maxReached) {
		/* Clear the matches but don't free the read name */
		RGMatchClearMatches(m);
		return;
	}

	if(m->numEntries > 0) {

		/* Quick sort the data structure */
		RGMatchQuickSort(m, 0, m->numEntries-1);

		/* Remove duplicates */
		prevIndex=0;
		for(i=1;i<m->numEntries;i++) {
			if(RGMatchCompareAtIndex(m, prevIndex, m, i)==0) {
				/* union of masks */
				RGMatchUnionMasks(m, prevIndex, i);
				RGMatchUnionOffsets(m, prevIndex, i);
			}
			else {
				prevIndex++;
				/* Copy to prevIndex (incremented) */
				RGMatchCopyAtIndex(m, prevIndex, m, i);
			}
		}

		/* Reallocate pair */
		/* does not make sense if there are no entries */
		RGMatchReallocate(m, prevIndex+1);

		/* Check to see if we have too many matches */
		if(NULL == m->offsets && maxNumMatches < m->numEntries) {
			/* Clear the entries but don't free the read */
			RGMatchClearMatches(m);
			m->maxReached = 1;
		}
		else { 
			m->maxReached = 0;
		}
	}
}

/* TODO */
void RGMatchQuickSort(RGMatch *m, int32_t low, int32_t high)
{
	int32_t i;
	int32_t pivot=-1;
	RGMatch *temp=NULL;


	if(low < high) {

		if(high - low + 1 <= RGMATCH_SHELL_SORT_MAX) {
			RGMatchShellSort(m, low, high);
			return;
		}

		/* Allocate memory for the temp used for swapping */
		temp=malloc(sizeof(RGMatch));
		if(NULL == temp) {
			PrintError("RGMatchQuickSort", "temp", "Could not allocate memory", Exit, MallocMemory);
		}
		RGMatchInitialize(temp);
		temp->readLength = m->readLength;
		RGMatchAllocate(temp, 1);
		if(NULL != m->offsets) {
			temp->numOffsets = malloc(sizeof(int32_t));
			if(NULL == temp->numOffsets) {
				PrintError("RGMatchQuickSort", "temp->numOffsets", "Could not allocate memory", Exit, MallocMemory);
			}
			temp->numOffsets[0]=0;
			temp->offsets = malloc(sizeof(int32_t*)); // include offsets just in case
			if(NULL == temp->offsets) {
				PrintError("RGMatchQuickSort", "temp->offsets", "Could not allocate memory", Exit, MallocMemory);
			}
			temp->offsets[0]=NULL;
		}

		pivot = (low+high)/2;

		RGMatchCopyAtIndex(temp, 0, m, pivot);
		RGMatchCopyAtIndex(m, pivot, m, high);
		RGMatchCopyAtIndex(m, high, temp, 0);

		pivot = low;

		for(i=low;i<high;i++) {
			if(RGMatchCompareAtIndex(m, i, m, high) <= 0) {
				if(i!=pivot) {
					RGMatchCopyAtIndex(temp, 0, m, i);
					RGMatchCopyAtIndex(m, i, m, pivot);
					RGMatchCopyAtIndex(m, pivot, temp, 0);
				}
				pivot++;
			}
		}
		RGMatchCopyAtIndex(temp, 0, m, pivot);
		RGMatchCopyAtIndex(m, pivot, m, high);
		RGMatchCopyAtIndex(m, high, temp, 0);

		/* Free temp before the recursive call, otherwise we have a worst
		 * case of O(n) space (NOT IN PLACE) 
		 * */
		RGMatchFree(temp);
		free(temp);
		temp=NULL;

		RGMatchQuickSort(m, low, pivot-1);
		RGMatchQuickSort(m, pivot+1, high);
	}
}

/* TODO */
void RGMatchShellSort(RGMatch *m, int32_t low, int32_t high)
{
	char *FnName="RGMatchShellSort";
	int32_t i, j, inc;
	RGMatch *temp=NULL;

	inc = ROUND((high - low + 1) / 2);

	/* Allocate memory for the temp used for swapping */
	temp=malloc(sizeof(RGMatch));
	if(NULL == temp) {
		PrintError(FnName, "temp", "Could not allocate memory", Exit, MallocMemory);
	}
	RGMatchInitialize(temp);
	temp->readLength = m->readLength;
	RGMatchAllocate(temp, 1);
	if(NULL != m->offsets) {
		temp->numOffsets = malloc(sizeof(int32_t));
		if(NULL == temp->numOffsets) {
			PrintError("RGMatchQuickSort", "temp->numOffsets", "Could not allocate memory", Exit, MallocMemory);
		}
		temp->numOffsets[0]=0;
		temp->offsets = malloc(sizeof(int32_t*)); // include offsets just in case
		if(NULL == temp->offsets) {
			PrintError("RGMatchQuickSort", "temp->offsets", "Could not allocate memory", Exit, MallocMemory);
		}
		temp->offsets[0]=NULL;
	}

	while(0 < inc) {
		for(i=inc + low;i<=high;i++) {
			RGMatchCopyAtIndex(temp, 0, m, i);
			j = i;
			while(inc + low <= j && RGMatchCompareAtIndex(temp, 0, m, j - inc) < 0) {
				RGMatchCopyAtIndex(m, j, m, j - inc);
				j -= inc;
			}
			RGMatchCopyAtIndex(m, j, temp, 0);
		}
		inc = ROUND(inc / SHELL_SORT_GAP_DIVIDE_BY);
	}

	RGMatchFree(temp);
	free(temp);
	temp=NULL;
}

/* TODO */
int32_t RGMatchCompareAtIndex(RGMatch *mOne, int32_t indexOne, RGMatch *mTwo, int32_t indexTwo) 
{
	assert(indexOne >= 0 && indexOne < mOne->numEntries);
	assert(indexTwo >= 0 && indexTwo < mTwo->numEntries);
	int32_t cmp[3], i;

	cmp[0] = COMPAREINTS(mOne->contigs[indexOne], mTwo->contigs[indexTwo]);
	cmp[1] = COMPAREINTS(mOne->positions[indexOne], mTwo->positions[indexTwo]);
	cmp[2] = COMPAREINTS(mOne->strands[indexOne], mTwo->strands[indexTwo]);

	for(i=0;i<3;i++) {
		if(0 != cmp[i]) return cmp[i];
	}
	return 0;
}

/* TODO */
void RGMatchAppend(RGMatch *dest, RGMatch *src)
{
	char *FnName = "RGMatchAppend";
	int32_t i, start;

	/* Make sure we are not appending to ourselves */
	assert(src != dest);
	assert(NULL != dest);
	assert(NULL != src);

	/* Check to see if we need to copy over the read as well */
	if(dest->readLength <= 0) {
		assert(dest->read == NULL);
		dest->readLength = src->readLength;
		dest->qualLength = src->qualLength;
		/* Allocate memory */
		dest->read = malloc(sizeof(char)*(dest->readLength+1));
		if(NULL==dest->read) {
			PrintError(FnName, "dest->read", "Could not allocate memory", Exit, MallocMemory);
		}   
		assert(dest->qual == NULL);
		dest->qual = malloc(sizeof(char)*(dest->qualLength+1));
		if(NULL==dest->qual) {
			PrintError(FnName, "dest->qual", "Could not allocate memory", Exit, MallocMemory);
		}
		/* Copy over */
		strcpy(dest->read, src->read);
		strcpy(dest->qual, src->qual);
	}

	/* if the max has been reached by the start or dest, then ignore */
	if(1 != dest->maxReached && 1 != src->maxReached) { 
		/* Allocate memory for the entries */
		start = dest->numEntries;
		RGMatchReallocate(dest, dest->numEntries + src->numEntries);

		assert(dest->numEntries == start + src->numEntries);
		assert(start <= dest->numEntries);

		// Must allocate if we had no entries
		if(0 == start && NULL != src->offsets && NULL == dest->offsets) {
			dest->numOffsets = malloc(sizeof(int32_t)*dest->numEntries);
			if(NULL == dest->numOffsets) {
				PrintError(FnName, "dest->numOffsets", "Could not allocate memory", Exit, MallocMemory);
			}
			dest->offsets = malloc(sizeof(int32_t*)*dest->numEntries);
			if(NULL == dest->offsets) {
				PrintError(FnName, "dest->offsets", "Could not allocate memory", Exit, MallocMemory);
			}
			// initialize
			for(i=0;i<dest->numEntries;i++) {
				dest->numOffsets[i] = 0;
				dest->offsets[i] = NULL;
			}
		}

		/* Copy over the entries */
		for(i=start;i<dest->numEntries;i++) {
			RGMatchCopyAtIndex(dest, i, src, i-start);
		}
	}
	else {
		/* Clear matches and set max reached flag */
		RGMatchClearMatches(dest);
	}
}

/* TODO */
void RGMatchCopyAtIndex(RGMatch *dest, int32_t destIndex, RGMatch *src, int32_t srcIndex)
{
	char *FnName="RGMatchCopyAtIndex";
	int32_t i;
	assert(srcIndex >= 0 && srcIndex < src->numEntries);
	assert(destIndex >= 0 && destIndex < dest->numEntries);

	if(src != dest || srcIndex != destIndex) {
		dest->positions[destIndex] = src->positions[srcIndex];
		dest->contigs[destIndex] = src->contigs[srcIndex];
		dest->strands[destIndex] = src->strands[srcIndex];
		assert(GETMASKNUMBYTES(dest) == GETMASKNUMBYTES(src));
		assert(NULL != dest->masks[destIndex]);
		assert(NULL != src->masks[srcIndex]);
		for(i=0;i<GETMASKNUMBYTES(dest);i++) {
			dest->masks[destIndex][i] = src->masks[srcIndex][i];
		}
		if(NULL != src->offsets) {
			assert(NULL != dest->offsets);
			free(dest->offsets[destIndex]);
			dest->numOffsets[destIndex]=src->numOffsets[srcIndex];
			dest->offsets[destIndex] = malloc(sizeof(int32_t)*dest->numOffsets[destIndex]);
			if(NULL == dest->offsets[destIndex]) {
				PrintError(FnName, "dest->offsets[destIndex]", "Could not allocate memory", Exit, MallocMemory);
			}
			for(i=0;i<src->numOffsets[srcIndex];i++) {
				dest->offsets[destIndex][i] = src->offsets[srcIndex][i];
			}
		}
	}
}

/* TODO */
void RGMatchAllocate(RGMatch *m, int32_t numEntries)
{
	char *FnName = "RGMatchAllocate";
	int32_t i;
	assert(m->numEntries==0);
	m->numEntries = numEntries;
	assert(m->positions==NULL);
	m->positions = malloc(sizeof(int32_t)*numEntries); 
	if(NULL == m->positions) {
		PrintError(FnName, "m->positions", "Could not allocate memory", Exit, MallocMemory);
	}
	assert(m->contigs==NULL);
	m->contigs = malloc(sizeof(uint32_t)*numEntries); 
	if(NULL == m->contigs) {
		PrintError(FnName, "m->contigs", "Could not allocate memory", Exit, MallocMemory);
	}
	assert(m->strands==NULL);
	m->strands = malloc(sizeof(char)*numEntries); 
	if(NULL == m->strands) {
		PrintError(FnName, "m->strands", "Could not allocate memory", Exit, MallocMemory);
	}
	m->masks = malloc(sizeof(char*)*numEntries); 
	if(NULL == m->masks) {
		PrintError(FnName, "m->masks", "Could not allocate memory", Exit, MallocMemory);
	}
	for(i=0;i<m->numEntries;i++) {
		m->masks[i] = calloc(GETMASKNUMBYTES(m), sizeof(char)); 
		if(NULL == m->masks[i]) {
			PrintError(FnName, "m->masks[i]", "Could not allocate memory", Exit, MallocMemory);
		}
	}
}

/* TODO */
void RGMatchReallocate(RGMatch *m, int32_t numEntries)
{
	char *FnName = "RGMatchReallocate";
	int32_t i, prevNumEntries;
	if(numEntries > 0) {
		prevNumEntries = m->numEntries;
		m->numEntries = numEntries;
		m->positions = realloc(m->positions, sizeof(int32_t)*numEntries); 
		if(numEntries > 0 && NULL == m->positions) {
			/*
			   fprintf(stderr, "numEntries:%d\n", numEntries);
			   */
			PrintError(FnName, "m->positions", "Could not reallocate memory", Exit, ReallocMemory);
		}
		m->contigs = realloc(m->contigs, sizeof(uint32_t)*numEntries); 
		if(numEntries > 0 && NULL == m->contigs) {
			PrintError(FnName, "m->contigs", "Could not reallocate memory", Exit, ReallocMemory);
		}
		m->strands = realloc(m->strands, sizeof(char)*numEntries); 
		if(numEntries > 0 && NULL == m->strands) {
			PrintError(FnName, "m->strands", "Could not reallocate memory", Exit, ReallocMemory);
		}
		for(i=numEntries;i<prevNumEntries;i++) {
			free(m->masks[i]);
		}
		m->masks = realloc(m->masks, sizeof(char*)*numEntries); 
		if(NULL == m->masks) {
			PrintError(FnName, "m->masks", "Could not reallocate memory", Exit, ReallocMemory);
		}
		for(i=prevNumEntries;i<m->numEntries;i++) {
			m->masks[i] = calloc(GETMASKNUMBYTES(m), sizeof(char)); 
			if(NULL == m->masks[i]) {
				PrintError(FnName, "m->masks[i]", "Could not allocate memory", Exit, MallocMemory);
			}
		}
		if(NULL != m->offsets) {
			for(i=numEntries;i<prevNumEntries;i++) {
				free(m->offsets[i]);
				m->offsets[i]=NULL;
				m->numOffsets[i]=0;
			}
			m->numOffsets = realloc(m->numOffsets, sizeof(int32_t)*numEntries);
			if(NULL == m->numOffsets) {
				PrintError(FnName, "m->numOffsets", "Could not allocate memory", Exit, MallocMemory);
			}
			m->offsets = realloc(m->offsets, sizeof(int32_t*)*numEntries);
			if(NULL == m->offsets) {
				PrintError(FnName, "m->offsets", "Could not allocate memory", Exit, MallocMemory);
			}
			for(i=prevNumEntries;i<m->numEntries;i++) {
				m->numOffsets[i]=0;
				m->offsets[i]=NULL;
			}
		}
	}
	else {
		/* Free just the matches part, not the meta-data */
		RGMatchClearMatches(m);
	}
}

/* TODO */
/* Does not free read */
void RGMatchClearMatches(RGMatch *m) 
{
	int32_t i;
	/* Free */
	free(m->contigs);
	free(m->positions);
	free(m->strands);
	m->contigs=NULL;
	m->positions=NULL;
	m->strands=NULL;
	for(i=0;i<m->numEntries;i++) {
		free(m->masks[i]);
	}
	free(m->masks);
	m->masks=NULL;
	if(NULL != m->offsets) {
		free(m->numOffsets);
		m->numOffsets=NULL;
		for(i=0;i<m->numEntries;i++) {
			free(m->offsets[i]);
		}
		free(m->offsets);
		m->offsets=NULL;
	}
	m->numEntries=0;
}

/* TODO */
void RGMatchFree(RGMatch *m) 
{
	int32_t i;
	free(m->read);
	free(m->qual);
	free(m->contigs);
	free(m->positions);
	free(m->strands);
	for(i=0;i<m->numEntries;i++) {
		free(m->masks[i]);
	}
	free(m->masks);
	if(NULL != m->offsets) {
		free(m->numOffsets);
		m->numOffsets=NULL;
		for(i=0;i<m->numEntries;i++) {
			free(m->offsets[i]);
		}
		free(m->offsets);
		m->offsets=NULL;
	}
	RGMatchInitialize(m);
}

/* TODO */
void RGMatchInitialize(RGMatch *m)
{
	m->readLength=0;
	m->qualLength=0;
	m->read=NULL;
	m->qual=NULL;
	m->maxReached=0;
	m->numEntries=0;
	m->contigs=NULL;
	m->positions=NULL;
	m->strands=NULL;
	m->masks=NULL;
	m->numOffsets=NULL;
	m->offsets=NULL;
}

/* TODO */
int32_t RGMatchCheck(RGMatch *m, RGBinary *rg)
{
	char *FnName="RGMatchCheck";
	int32_t i, j;

	/* Basic asserts */
	assert(m->readLength >= 0);
	assert(m->qualLength >= 0);
	assert(m->maxReached == 0 || m->maxReached == 1);
	assert(m->maxReached == 0 || m->numEntries == 0);
	assert(m->numEntries >= 0);
	/* Check that if the read length is greater than zero the read is not null */
	if(m->readLength > 0 && m->read == NULL && m->qual == NULL) {
		PrintError(FnName, NULL, "m->readLength > 0 && m->read == NULL && m->qual == NULL", Warn, OutOfRange);
		return 0;
	}
	/* Check that the read length matches the read */
	if(((int)strlen(m->read)) != m->readLength) {
		PrintError(FnName, NULL, "m->readLength and strlen(m->read) do not match", Warn, OutOfRange);
		return 0;
	}
	/* Check that the qual length matches the qual */
	if(((int)strlen(m->qual)) != m->qualLength) {
		PrintError(FnName, NULL, "m->qualLength and strlen(m->qual) do not match", Warn, OutOfRange);
		return 0;
	}
	/* Check that if the max has been reached then there are no entries */
	if(1==m->maxReached && m->numEntries > 0) {
		PrintError(FnName, NULL, "1==m->maxReached and m->numEntries>0", Warn, OutOfRange);
		return 0;
	}
	/* Check that if the number of entries is greater than zero that the entries are not null */
	if(m->numEntries > 0 && (m->contigs == NULL || m->positions == NULL || m->strands == NULL)) {
		PrintError(FnName, NULL, "m->numEntries > 0 && (m->contigs == NULL || m->positions == NULL || m->strands == NULL)", Warn, OutOfRange);
		return 0;
	}

	/* Check mask */
	for(i=0;i<m->numEntries;i++) {
		char *mask = RGMatchMaskToString(m->masks[i], m->readLength);
		char reference[SEQUENCE_LENGTH]="\0";

		if(m->strands[i] == FORWARD) {

			if(ColorSpace == rg->space) {
				reference[0]='X';
				for(j=1;j<m->readLength;j++) { // ignore leading adaptor
					reference[j] = RGBinaryGetBase(rg, m->contigs[i], m->positions[i] + j - 1);
					reference[j] = (0 == reference[j]) ? '-' : ToUpper(reference[j]);
					reference[j] = COLORS[BaseToInt(reference[j])];
				}
				reference[j]='\0';
				for(j=1;j<m->readLength;j++) { // ignore leading adaptor

					if('1' == mask[j-1] && ToUpper(m->read[j]) != reference[j]) {
						fprintf(stderr, "\n%s%s\n%s\n%s\n",
								BREAK_LINE,
								reference,
								m->read,
								mask);
						fprintf(stderr, "%c:%d:%d\n",
								FORWARD,
								m->contigs[i],
								m->positions[i]);
						PrintError(FnName, "m->read[j]) != base", "Inconsistency with the mask", Warn, OutOfRange);
						return 0;
					}
				}
			}
			else {
				for(j=0;j<m->readLength;j++) {
					reference[j] = RGBinaryGetBase(rg, m->contigs[i], m->positions[i] + j);
					reference[j] = (0 == reference[j]) ? '-' : ToUpper(reference[j]);


					if('1' == mask[j] && ToUpper(m->read[j]) != reference[j]) {
						reference[j+1]='\0';
						char *r=NULL;
						assert(1 == RGBinaryGetSequence(rg,
									m->contigs[i],
									m->positions[i],
									m->strands[i],
									&r,
									m->readLength));
						fprintf(stderr, "\n%s%s\n%s\n%s\n%s\n",
								BREAK_LINE,
								reference,
								r,
								m->read,
								mask);
						fprintf(stderr, "%c:%d:%d\n",
								FORWARD,
								m->contigs[i],
								m->positions[i]);
						free(r);
						PrintError(FnName, "m->read[j]) != base", "Inconsistency with the mask", Warn, OutOfRange);
						return 0;
					}
				}
			}
		}
		else {
			if(NTSpace == rg->space) {
				for(j=0;j<m->readLength;j++) {
					char base = RGBinaryGetBase(rg, 
							m->contigs[i], 
							m->positions[i] + m->readLength - 1 - j);
					reference[j] = (0 == base) ? '-' : ToUpper(base);
					reference[j] = GetReverseComplimentAnyCaseBase(reference[j]);
					if('1' == mask[j] && ToUpper(m->read[j]) != reference[j]) {
						reference[j+1]='\0';
						char *r=NULL;
						assert(1 == RGBinaryGetSequence(rg,
									m->contigs[i],
									m->positions[i],
									m->strands[i],
									&r,
									m->readLength));
						fprintf(stderr, "\n%s%s\n%s\n%s\n%s\n",
								BREAK_LINE,
								reference,
								r,
								m->read,
								mask);
						fprintf(stderr, "%c:%d:%d\n",
								REVERSE,
								m->contigs[i],
								m->positions[i]);
						free(r);
						PrintError(FnName, "m->read[j]) != base", "Inconsistency with the mask", Warn, OutOfRange);
						return 0;
					}
				}
			}
			else {
				reference[0]='X';
				for(j=1;j<m->readLength;j++) { // skip adaptor
					char base = RGBinaryGetBase(rg, 
							m->contigs[i], 
							m->positions[i] + m->readLength - j);
					reference[j] = (0 == base) ? '-' : ToUpper(base);
					reference[j] = COLORS[BaseToInt(reference[j])];
				}
				reference[j]='\0';
				for(j=1;j<m->readLength;j++) { // skip adaptor
					char base = RGBinaryGetBase(rg, 
							m->contigs[i], 
							m->positions[i] + m->readLength - j);
					reference[j-1] = (0 == base) ? '-' : ToUpper(base);
					reference[j-1] = COLORS[BaseToInt(reference[j-1])];
					if('1' == mask[j-1] && ToUpper(m->read[j]) != reference[j-1]) {
						fprintf(stderr, "\n%s%s\n%s\n%s\n",
								BREAK_LINE,
								reference,
								mask,
								m->read);
						fprintf(stderr, "%c:%d:%d\n",
								REVERSE,
								m->contigs[i],
								m->positions[i]);
						PrintError(FnName, "m->read[j]) != base", "Inconsistency with the mask", Warn, OutOfRange);
						return 0;
					}
				}
			}
		}
		free(mask);
	}
	return 1;
}

/* TODO */
void RGMatchFilterOutOfRange(RGMatch *m,
		int32_t maxNumMatches)
{
	/* Filter based on the maximum number of matches */
	if(maxNumMatches != 0 && m->numEntries > maxNumMatches) {
		/* Do not align this one */
		RGMatchClearMatches(m);
		m->maxReached=1;
		assert(m->readLength > 0);
	}
}

char *RGMatchMaskToString(char *mask,
		int32_t readLength)
{
	char *FnName="RGMatchMaskToString";
	int32_t i, curByte, curByteIndex;
	uint8_t byte;

	char *string = malloc(sizeof(char)*(1+readLength));
	if(NULL == string) {
		PrintError(FnName, "string", "Could not allocate memory", Exit, MallocMemory);
	}

	for(i=0;i<readLength;i++) {
		curByte = GETMASKBYTE(i);
		curByteIndex = i % (8*sizeof(char));
		byte = mask[curByte];
		byte = byte << (8 - 1 - curByteIndex);
		byte = byte >> 7;
		assert(byte == 1 || byte == 0);
		string[i] = (0 < byte) ? '1' : '0';
	}
	string[readLength] = '\0';

	return string;
}

char *RGMatchStringToMask(char *string,
		int32_t readLength)
{
	char *FnName="RGMatchStringToMask";
	assert(readLength == strlen(string));
	int32_t i, curByte, curByteIndex;

	char *mask = calloc(GETMASKNUMBYTESFROMLENGTH(readLength), sizeof(char));
	if(NULL == mask) {
		PrintError(FnName, "mask", "Could not allocate memory", Exit, MallocMemory);
	}

	for(i=0;i<readLength;i++) {
		if('1' == string[i]) {
			curByte = GETMASKBYTE(i);
			curByteIndex = i % (8*sizeof(char));
			mask[curByte] |= (0x01 << curByteIndex);
		}
	}

	return mask;
}

void RGMatchUpdateMask(char *mask, int32_t pos)
{
	int32_t curByte, curByteIndex;
	curByte = GETMASKBYTE(pos);
	curByteIndex = pos % (8*sizeof(char));
	mask[curByte] |= (0x01 << curByteIndex);
}

void RGMatchUnionMasks(RGMatch *m, int32_t dest, int32_t src)
{
	int32_t i;
	for(i=0;i<GETMASKNUMBYTES(m);i++) {
		m->masks[dest][i] |= m->masks[src][i];
	}
}

void RGMatchUnionOffsets(RGMatch *m, int32_t dest, int32_t src)
{
	char *FnName="RGMatchUnionOffsets";
	int32_t i, prevNumOffsets;

	if(NULL == m->offsets) return;

	prevNumOffsets=m->numOffsets[dest];
	m->numOffsets[dest] += m->numOffsets[src];
	m->offsets[dest] = realloc(m->offsets[dest], sizeof(int32_t)*m->numOffsets[dest]);
	if(NULL == m->offsets[dest]) {
		PrintError(FnName, "m->offsets[dest]", "Could not allocate memory", Exit, MallocMemory);
	}
	for(i=0;i<m->numOffsets[src];i++) {
		m->offsets[dest][i+prevNumOffsets] = m->offsets[src][i];
	}
}
