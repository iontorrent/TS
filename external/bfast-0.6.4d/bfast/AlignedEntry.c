#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <zlib.h>
#include <ctype.h>
#include "BError.h"
#include "BLib.h"
#include "AlignedEntry.h"

// move to BLib.c
char getAlnReadBase(uint8_t *alnRead, int32_t i)
{
	uint8_t base;

	base = alnRead[(int)(i/2)];
	if(0 == (i & 1)) { // left-four bits
		base >>= 4;
	}
	else { // right-four bits
		base &= 0x0F;
	}

	if(5 < base) {
		return "ACGTN"[base % 6];
	}
	else {
		return "acgtn-"[base % 6];
	}

}

// move to BLib.c
void putAlnReadBase(uint8_t *alnRead, int32_t i, char b, int32_t ins) 
{
	uint8_t op = 0;

	switch(b) {
		case 'A':
		case 'a': 
			op = 0; break;
		case 'c':
		case 'C':
			op = 1; break;
		case 'g':
		case 'G':
			op = 2; break;
		case 't':
		case 'T':
			op = 3; break;
		case 'n':
		case 'N':
		case '.':
			op = 4; break;
		case '-':
			op = 5; break;
		default:
			assert(1==0); // should not go here
	}

	if(1 == ins) {
		assert(op != 5); // cannot have ins and del
		op += 6;
	}
	// op should range from 0-11
	if(0 == (i & 1)) { // left-four bits
		alnRead[(int)(i/2)] |= (op << 4);
	}
	else { // right-four bits
		alnRead[(int)(i/2)] |= op;
	}
}

/* TODO */
int32_t AlignedEntryPrint(AlignedEntry *a,
		gzFile outputFP)
{
	int len = (int)(a->alnReadLength/2 + 1);
	if(gzwrite64(outputFP, &a->contig, sizeof(uint32_t))!=sizeof(uint32_t)||
			gzwrite64(outputFP, &a->position, sizeof(uint32_t))!=sizeof(uint32_t)||
			gzwrite64(outputFP, &a->strand, sizeof(char))!=sizeof(char)||
			gzwrite64(outputFP, &a->score, sizeof(int32_t))!=sizeof(int32_t)||
			gzwrite64(outputFP, &a->mappingQuality, sizeof(uint8_t))!=sizeof(uint8_t)||
			gzwrite64(outputFP, &a->alnReadLength, sizeof(int32_t))!=sizeof(int32_t)||
			gzwrite64(outputFP, a->alnRead, sizeof(uint8_t)*len)!=sizeof(uint8_t)*len) {
		return EOF;
	}

	return 1;
}


/* TODO */
int32_t AlignedEntryPrintText(AlignedEntry *a,
		FILE *outputFP)
{
	int32_t i;
	if(fprintf(outputFP, "%u\t%u\t%c\t%d\t%d\t%d\t",
				a->contig,
				a->position,
				a->strand,
				a->score,
				a->mappingQuality,
				a->alnReadLength) < 0) {
		return EOF;
	}

	for(i=0;i<a->alnReadLength;i++) {
		if(fprintf(outputFP, "%c",
					getAlnReadBase(a->alnRead, i)) < 0) {
			return EOF;
		}
	}
	if(fprintf(outputFP, "\n") < 0) {
		return EOF;
	}

	return 1;
}

/* TODO */
int32_t AlignedEntryRead(AlignedEntry *a,
		gzFile inputFP)
{
	char *FnName = "AlignedEntryRead";
	int len;
	assert(NULL != a);

	if(gzread64(inputFP, &a->contig, sizeof(uint32_t))!=sizeof(uint32_t)||
			gzread64(inputFP, &a->position, sizeof(uint32_t))!=sizeof(uint32_t)||
			gzread64(inputFP, &a->strand, sizeof(char))!=sizeof(char)||
			gzread64(inputFP, &a->score, sizeof(int32_t))!=sizeof(int32_t)||
			gzread64(inputFP, &a->mappingQuality, sizeof(uint8_t))!=sizeof(uint8_t)||
			gzread64(inputFP, &a->alnReadLength, sizeof(int32_t))!=sizeof(int32_t)) {
		return EOF;
	}

	len = (int)(a->alnReadLength/2 + 1);
	a->alnRead = calloc(sizeof(uint8_t), len);
	if(NULL == a->alnRead) {
		PrintError(FnName, "a->alnRead", "Could not allocate memory", Exit, MallocMemory);
	}

	if(gzread64(inputFP, a->alnRead, sizeof(uint8_t)*len)!=sizeof(uint8_t)*len) {
		return EOF;
	}

	return 1;
}

/* TODO */
int32_t AlignedEntryReadText(AlignedEntry *a,
		FILE *inputFP)
{
	char *FnName = "AlignedEntryReadText";
	int32_t i, tmp, len;
	char alnRead[1024]="\0";
	assert(NULL != a);

	if(fscanf(inputFP, "%u\t%u\t%c\t%d\t%d\t%d\t",
				&a->contig,
				&a->position,
				&a->strand,
				&a->score,
				&tmp,
				&a->alnReadLength) < 0) {
		return EOF;
	}
	a->mappingQuality = tmp;

	len = (int)(a->alnReadLength/2 + 1);
	a->alnRead = calloc(sizeof(uint8_t), len);
	if(NULL == a->alnRead) {
		PrintError(FnName, "a->alnRead", "Could not allocate memory", Exit, MallocMemory);
	}

	if(NULL == fgets(alnRead, 1024, inputFP)) {
		return EOF;
	}

	for(i=0;i<a->alnReadLength;i++) {
		putAlnReadBase(a->alnRead, i, alnRead[i], (toupper(alnRead[i]) == alnRead[i]) ? 1: 0);
	}

	return 1;
}

/* TODO */
/* Log-n space */
void AlignedEntryQuickSort(AlignedEntry **a,
		int32_t low,
		int32_t high,
		int32_t sortOrder,
		int32_t showPercentComplete,
		double *curPercent,
		int32_t total)
{
	char *FnName = "AlignedEntryQuickSort";
	int32_t i;
	int32_t pivot=-1;
	AlignedEntry *temp=NULL;

	if(low < high) {

		if(high - low + 1 <= ALIGNEDENTRY_SHELL_SORT_MAX) {
			AlignedEntryShellSort(a, low, high, sortOrder);
			return;
		}

		/* Allocate memory for the temp used for swapping */
		temp=malloc(sizeof(AlignedEntry));
		if(NULL == temp) {
			PrintError(FnName, "temp", "Could not allocate temp", Exit, MallocMemory);
		}
		AlignedEntryInitialize(temp);

		pivot = AlignedEntryGetPivot((*a),
				sortOrder,
				low,
				high);
		if(showPercentComplete == 1 && VERBOSE >= 0) {
			assert(NULL!=curPercent);
			if((*curPercent) < 100.0*((double)low)/total) {
				while((*curPercent) < 100.0*((double)low)/total) {
					(*curPercent) += SORT_ROTATE_INC;
				}
				PrintPercentCompleteShort((*curPercent));
			}
		}

		AlignedEntryCopyAtIndex(temp, 0, (*a), pivot);
		AlignedEntryCopyAtIndex((*a), pivot, (*a), high);
		AlignedEntryCopyAtIndex((*a), high, temp, 0);

		pivot = low;

		for(i=low;i<high;i++) {
			if(AlignedEntryCompareAtIndex((*a), i, (*a), high, sortOrder) <= 0) {
				if(i!=pivot) {
					AlignedEntryCopyAtIndex(temp, 0, (*a), i);
					AlignedEntryCopyAtIndex((*a), i, (*a), pivot);
					AlignedEntryCopyAtIndex((*a), pivot, temp, 0);
				}
				pivot++;
			}
		}
		AlignedEntryCopyAtIndex(temp, 0, (*a), pivot);
		AlignedEntryCopyAtIndex((*a), pivot, (*a), high);
		AlignedEntryCopyAtIndex((*a), high, temp, 0);

		/* Free temp before the recursive call, otherwise we have a worst
		 * case of O(n) space (NOT IN PLACE) 
		 * */
		AlignedEntryFree(temp);
		free(temp);
		temp=NULL;

		AlignedEntryQuickSort(a, low, pivot-1, sortOrder, showPercentComplete, curPercent, total);
		if(showPercentComplete == 1 && VERBOSE >= 0) {
			assert(NULL!=curPercent);
			if((*curPercent) < 100.0*((double)pivot)/total) {
				while((*curPercent) < 100.0*((double)pivot)/total) {
					(*curPercent) += SORT_ROTATE_INC;
				}
				PrintPercentCompleteShort((*curPercent));
			}
		}
		AlignedEntryQuickSort(a, pivot+1, high, sortOrder, showPercentComplete, curPercent, total);
		if(showPercentComplete == 1 && VERBOSE >= 0) {
			assert(NULL!=curPercent);
			if((*curPercent) < 100.0*((double)high)/total) {
				while((*curPercent) < 100.0*((double)high)/total) {
					(*curPercent) += SORT_ROTATE_INC;
				}
				PrintPercentCompleteShort((*curPercent));
			}
		}
	}
}

void AlignedEntryShellSort(AlignedEntry **a,
		int32_t low,
		int32_t high,
		int32_t sortOrder)
{
	char *FnName = "AlignedEntryShellSort";
	int32_t i, j, inc;
	AlignedEntry *temp=NULL;

	inc = ROUND((high - low + 1) / 2);

	/* Allocate memory for the temp used for swapping */
	temp=malloc(sizeof(AlignedEntry));
	if(NULL == temp) {
		PrintError(FnName, "temp", "Could not allocate temp", Exit, MallocMemory);
	}
	AlignedEntryInitialize(temp);

	while(0 < inc) {
		for(i=inc + low;i<=high;i++) {
			AlignedEntryCopyAtIndex(temp, 0, (*a), i);
			j = i;
			while(inc + low <= j && AlignedEntryCompareAtIndex(temp, 0, (*a), j - inc, sortOrder) < 0) {
				AlignedEntryCopyAtIndex((*a), j, (*a), j - inc);
				j -= inc;
			}
			AlignedEntryCopyAtIndex((*a), j, temp, 0);
		}
		inc = ROUND(inc / SHELL_SORT_GAP_DIVIDE_BY);
	}
	AlignedEntryFree(temp);
	free(temp);
	temp=NULL;

}

/* TODO */
int32_t AlignedEntryCompareAtIndex(AlignedEntry *a, int32_t indexA, AlignedEntry *b, int32_t indexB, int32_t sortOrder)
{
	return AlignedEntryCompare(&(a[indexA]), &(b[indexB]), sortOrder);
}

/* TODO */
int32_t AlignedEntryCompare(AlignedEntry *a, AlignedEntry *b, int32_t sortOrder)
{
	int32_t cmp[5];
	int32_t i;
	int32_t top;

	if(sortOrder == AlignedEntrySortByAll) {

		/* If there are multiple alignments to the same starting chr/pos/strand with the same score,
		 * this will pick ensure that we will only pick one of them.
		 * */
		cmp[0] = (a->contig <= b->contig)?((a->contig<b->contig)?-1:0):1;
		cmp[1] = (a->position <= b->position)?((a->position<b->position)?-1:0):1;
		cmp[2] = (a->strand <= b->strand)?((a->strand<b->strand)?-1:0):1;
		cmp[3] = (a->score <= b->score)?((a->score<b->score)?-1:0):1;

		top = 4;
	}
	else {
		assert(sortOrder == AlignedEntrySortByContigPos);
		cmp[0] = (a->contig <= b->contig)?((a->contig<b->contig)?-1:0):1;
		cmp[1] = (a->position <= b->position)?((a->position<b->position)?-1:0):1;

		top = 2;
	}

	/* ingenious */
	for(i=0;i<top;i++) {
		if(cmp[i] != 0) {
			return cmp[i];
		}
	}

	return 0;
}

/* TODO */
void AlignedEntryCopyAtIndex(AlignedEntry *dest, int32_t destIndex, AlignedEntry *src, int32_t srcIndex)
{
	if(dest != src || srcIndex != destIndex) {
		AlignedEntryCopy(&(dest[destIndex]), &(src[srcIndex]));
	}
}

/* TODO */
void AlignedEntryCopy(AlignedEntry *dest, AlignedEntry *src)
{
	char *FnName = "AlignedEntryCopy";
	int32_t i, len;
	if(src != dest) {
		/* Metadata */
		dest->contig = src->contig;
		dest->position = src->position;
		dest->strand = src->strand;
		dest->score = src->score;
		dest->mappingQuality = src->mappingQuality;
		// alnRead
		if(0 < dest->alnReadLength) {
			free(dest->alnRead);
			dest->alnRead=NULL;
			dest->alnReadLength=0;
		}
		dest->alnReadLength = src->alnReadLength;
		len = (int)(dest->alnReadLength/2 + 1);
		dest->alnRead = calloc(sizeof(uint8_t), len);
		if(NULL == dest->alnRead) {
			PrintError(FnName, "dest->alnRead", "Could not allocate memory", Exit, MallocMemory);
		}
		for(i=0;i<len;i++) {
			dest->alnRead[i] = src->alnRead[i];
		}
	}
}

void AlignedEntryFree(AlignedEntry *a)
{
	free(a->alnRead);
	AlignedEntryInitialize(a);
}

void AlignedEntryInitialize(AlignedEntry *a) 
{
	a->contig=0;
	a->position=0;
	a->strand=0;
	a->score=0;
	a->mappingQuality=0;
	a->alnReadLength=0;
	a->alnRead=NULL;
}

int32_t AlignedEntryGetPivot(AlignedEntry *a,
		int32_t sortOrder,
		int32_t low,
		int32_t high) 
{
	int32_t cmp[3];
	int32_t pivot = (low + high)/2;
	cmp[0] = AlignedEntryCompareAtIndex(a, low, a, pivot, sortOrder); 
	cmp[1] = AlignedEntryCompareAtIndex(a, low, a, high, sortOrder); 
	cmp[2] = AlignedEntryCompareAtIndex(a, pivot, a, high, sortOrder); 

	if(cmp[0] <= 0) {
		/* low <= pivot */
		if(cmp[1] >= 0) {
			/* high <= low */
			/* so high <= low <= pivot */
			pivot = low;
		}
		else {
			/* low < high */
			if(cmp[2] <= 0) {
				/* pivot <= high */
				/* so low <= pivot <= high */
				/* choose pivot */
			}
			else {
				/* high < pivot */
				/* so low < high < pivot */
				pivot = high;
			}
		}
	}
	else {
		/* pivot < low */
		if(cmp[1] <= 0) {
			/* low <= high */
			/* so pivot < low <= high */
			pivot = low;
		}
		else {
			/* high < low */
			if(cmp[2] <= 0) {
				/* pivot <= high */
				/* so pivot <= high < low */
				pivot = high;
			}
			else {
				/* high < pivot */
				/* so high < pivot < low */
				/* choose pivot */
			}
		}
	}
	return pivot;
}

void AlignedEntryUpdateAlignment(AlignedEntry *a,
		uint32_t position,
		double score,
		int32_t referenceLength,
		int32_t length,
		char *read,
		char *reference)
{
	char *FnName="AlignedEntryAllocate";
	int32_t i, len;

	// contig updated earlier (I hope)
	a->position = position;
	// strand updated earlier (I hope)
	a->score = score;

	// make a->alnRead
	a->alnReadLength = length;
	len = (int)(a->alnReadLength/2 + 1);
	a->alnRead = calloc(sizeof(uint8_t), len);
	if(NULL == a->alnRead) {
		PrintError(FnName, "a->alnRead", "Could not allocate memory", Exit, MallocMemory);
	}
	for(i=0;i<length;i++) {
		putAlnReadBase(a->alnRead, 
				i,
				read[i],
				(GAP == reference[i]) ? 1 : 0);
	}
}

int32_t AlignedEntryGetAlignment(AlignedEntry *a,
		RGBinary *rg,
		char alignment[3][SEQUENCE_LENGTH],
		char *origRead,
		int32_t origReadLength,
		int32_t space)
{
	char *FnName="AlignedEntryGetAlignment";
	int32_t i, j;
	int32_t length=0, referenceLength=0;
	char *reference=NULL;

	// Get read, and reference length
	length = a->alnReadLength;
	for(i=0;i<a->alnReadLength;i++) {
		alignment[1][i] = getAlnReadBase(a->alnRead, i);
		if(GAP == alignment[1][i] || // deleted from read
				alignment[1][i] != toupper(alignment[1][i])) { // not an inserted base into the read
			referenceLength++;
		}
	}
	alignment[1][length]='\0';

	// get reference
	if(0 == RGBinaryGetSequence(rg, a->contig, a->position, a->strand, &reference, referenceLength)) {
		PrintError(FnName, NULL, "Could not get reference sequence", Exit, OutOfRange);
	}
	for(i=j=0;i<length;i++) {
		if(GAP == alignment[1][i] ||
				alignment[1][i] != toupper(alignment[1][i])) { // not an inserted base into the read
			referenceLength++;
			alignment[0][i] = reference[j];
			j++;
		}
		else {
			alignment[0][i] = GAP;
		}
		alignment[1][i] = toupper(alignment[1][i]);
	}
	free(reference);
	alignment[0][length]='\0';

	// get color error string (how) ?
	if(ColorSpace == space) {
		// Copy over 
		char prevBase = origRead[0];
		for(i=0,j=1;i<length;i++) {
			if(GAP != alignment[1][i]) { // not a deletion
				if(GAP != alignment[0][i]) { // not an insertion 
					char c;
					assert(1 == ConvertBaseToColorSpace(prevBase, alignment[1][i], &c));
					if("01234"[(int)c] == origRead[j]) {
						alignment[2][i]=GAP;
					}
					else {
						alignment[2][i]=origRead[j];
					}
				}
				else {
					alignment[2][i]=GAP;
				}
				prevBase = alignment[1][i];
				j++;
			}
			else {
				alignment[2][i]=GAP;
			}
		}

		alignment[2][length]='\0';
	}
	else {
		// maybe check that origRead matches?
		alignment[0][length]='\0';
		alignment[1][length]='\0';
		alignment[2][0]='\0';
	}
	return length;
}
