#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "BLibDefinitions.h"
#include "BError.h"
#include "RGIndex.h"
#include "RGBinary.h"
#include "RGIndexExons.h"


void RGIndexExonsRead(char *exonsFileName,
		RGIndexExons *e)
{
	char *FnName = "RGIndexExonsRead";
	FILE *fp;
	uint32_t prevEndContig, prevEndPos;
	uint32_t curStartContig, curStartPos, curEndContig, curEndPos;

	RGIndexExonsInitialize(e);

	/* Open the file */
	if(VERBOSE >= 0) {
		fprintf(stderr, "Reading in exons from %s.\n",
				exonsFileName);
	}
	if(!(fp=fopen(exonsFileName, "rb"))) {
		PrintError(FnName, exonsFileName, "Could not open file for reading", Exit, OpenFileError);
	}

	/* Read in the exons */ 
	prevEndContig = prevEndPos = 0;
	while(EOF!=fscanf(fp, "%u %u %u %u", &curStartContig, &curStartPos, &curEndContig, &curEndPos)) {

		/* Check valid range */
		if(curStartContig > curEndContig ||
				(curStartContig == curEndContig && curStartPos > curEndPos)) {
			PrintError(FnName, NULL, "Exon range is not valid", Exit, OutOfRange);
		}
		/* Check versus previous */
		if(curStartContig < prevEndContig ||
				(curStartContig == prevEndContig && curStartPos < prevEndPos)) {
			fprintf(stderr, "%u %u %u %u\n",
					curStartContig,
					curStartPos,
					curEndContig,
					curEndPos);
			fprintf(stderr, "previous was %u %u\n",
					prevEndContig,
					prevEndPos);
			PrintError(FnName, NULL, "Exons must be in increasing order and non-overlapping", Exit, OutOfRange);
		}
		/* Copy over */
		e->numExons++;
		e->exons = realloc(e->exons, sizeof(RGIndexExon)*e->numExons);
		if(NULL == e->exons) {
			PrintError(FnName, "e->exons", "Could not allocate memory", Exit, MallocMemory);
		}
		e->exons[e->numExons-1].startContig = curStartContig;
		e->exons[e->numExons-1].startPos = curStartPos;
		e->exons[e->numExons-1].endContig = curEndContig;
		e->exons[e->numExons-1].endPos = curEndPos;
		/* Update previous */
		prevEndContig = curEndContig;
		prevEndPos = curEndPos;
	}

	/* Close the file */
	fclose(fp);

	if(VERBOSE >= 0) {
		fprintf(stderr, "Read in %d exons.\n",
				e->numExons);
	}
}

int RGIndexExonsWithin(RGIndexExons *e,
		uint32_t startContig,
		uint32_t startPos,
		uint32_t endContig,
		uint32_t endPos)
{
	int64_t low, mid, high, found;

	assert(startPos <= endPos);

	/* Check if it falls within range */
	/* Binary search for start pos */
	low = 0;
	mid = -1;
	high = e->numExons-1;
	found = 0;
	while(low <= high && found == 0) {
		mid = (low + high)/2;
		if(startContig < e->exons[mid].startContig ||
				(startContig == e->exons[mid].startContig && startPos < e->exons[mid].startPos)) {
			high = mid - 1;
		}
		else if(startContig > e->exons[mid].endContig ||
				(startContig == e->exons[mid].endContig && startPos > e->exons[mid].endPos)) {
			low = mid + 1;
		}
		else {
			found = 1;
		}
	}
	if(found == 1) {
		/* Move to front */
		while(mid>=0 &&
				(startContig > e->exons[mid].startContig || 
				 (startContig == e->exons[mid].startContig && startPos >= e->exons[mid].startPos))) {
			mid--;
		}
		/* Check within bounds */
		if((e->exons[mid].startContig < startContig ||
					(e->exons[mid].startContig == startContig && e->exons[mid].startPos <= startPos)) &&
				(e->exons[mid].endContig > endContig || 
				 (e->exons[mid].endContig == endContig && e->exons[mid].endPos >= endPos))) {
			found = 1;
		}
		else {
			found = 0;
		}
	}
	return found;
}

void RGIndexExonsInitialize(RGIndexExons *e)
{
	e->numExons=0;
	e->exons=NULL;
}

void RGIndexExonsDelete(RGIndexExons *e)
{
	free(e->exons);
	RGIndexExonsInitialize(e);
}
