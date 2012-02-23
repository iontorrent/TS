#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include "BLibDefinitions.h"
#include "BError.h"
#include "BLib.h"
#include "RGIndex.h"
#include "RGMatch.h"
#include "RGRanges.h"
#include "RGReads.h"

/* TODO 
 * */

char ALPHABET[ALPHABET_SIZE] = "acgt";

/* TODO */
void RGReadsFindMatches(RGIndex *index, 
		RGBinary *rg,
		RGMatch *match,
		int copyOffsets,
		int *offsets,
		int numOffsets,
		int space,
		int numMismatches,
		int numInsertions,
		int numDeletions,
		int numGapInsertions,
		int numGapDeletions,
		int maxKeyMatches,
		int maxNumMatches,
		int strands)
{
	int64_t i;
	int readLength=0;
	int8_t read[SEQUENCE_LENGTH];
	RGReads reads;
	RGRanges ranges;
	int readOffset = 0;

	/* Initialize */
	RGReadsInitialize(&reads);
	RGRangesInitialize(&ranges);

	readLength = match->readLength;
	if(space==ColorSpace) {
		/* First letter is adapter, second letter is the color (unusable) */
		readOffset += 2;
		readLength -= 2;
	}

	/* Copy over */
	/* Convert bases/colors to 0-4 */
	ConvertSequenceToIntegers(match->read + readOffset,
			read,
			readLength);

	/* Merge all reads */
	/* This may be necessary for a large number of generated reads, but omit for now */
	/*
	   if(numMismatches > 0 || 
	   numInsertions > 0 ||
	   numDeletions > 0 ||
	   numGapInsertions > 0 ||
	   numGapDeletions > 0) {
	   RGReadsRemoveDuplicates(reads);
	   }
	   */

	if(0 < numOffsets) { /* Go through the offsets */
		for(i=0;0 == match->maxReached && // have not reached the maximum
				i<numOffsets && // offsets remaining
				index->width <= (readLength - offsets[i]); // offsets is within bounds (assumes sorted) 
				i++) {
			match->maxReached = RGIndexGetRangesBothStrands(index, 
					rg,
					read + offsets[i],
					index->width,
					offsets[i],
					(0 == copyOffsets) ? maxKeyMatches : INT_MAX,
					maxNumMatches,
					space,
					strands,
					&ranges);
		}
	}
	else { /* Use all offsets */
		for(i=0;0 == match->maxReached && // have not reached the maximum
				index->width <= (readLength - i); // offsets is within bounds (assumes sorted) 
				i++) {
			match->maxReached = RGIndexGetRangesBothStrands(index, 
					rg,
					read + i,
					index->width,
					i,
					(0 == copyOffsets) ? maxKeyMatches : INT_MAX,
					maxNumMatches,
					space,
					strands,
					&ranges);
		}
	}

	/* Transfer ranges to matches */
	RGRangesCopyToRGMatch(&ranges,
			index,
			match,
			space,
			copyOffsets);

	/* Remove duplicates */
	RGMatchRemoveDuplicates(match,
			maxNumMatches);

	/* Free memory */
	RGRangesFree(&ranges);
	RGReadsFree(&reads);
}

/* TODO */
/* We may want to include enumeration of SNPs in color space */
void RGReadsGenerateReads(char *read,
		int readLength,
		RGIndex *index,
		RGReads *reads,
		int *offsets,
		int numOffsets,
		int space,
		int numMismatches,
		int numInsertions,
		int numDeletions,
		int numGapInsertions,
		int numGapDeletions)
{
	int i;

	/* Go through all offsets */
	for(i=0;i<numOffsets && (readLength - offsets[i]) >= index->width;i++) {

		/* Insert the perfect match */
		RGReadsGeneratePerfectMatch(read,
				readLength,
				offsets[i],
				index,
				reads);

		/* Go through all mismatches */
		/* Note: we allow any number (including zero) of mismatches up to
		 * numMismatches.  
		 * */
		if(numMismatches > 0) {
			RGReadsGenerateMismatches(read,
					readLength,
					offsets[i],
					numMismatches,
					index,
					reads);
		}

		/* Go through all deletions */
		/* Note: we allow only contiguous deletions of length up to 
		 * numDeletions.  We also always start from the offset.  We
		 * must add base to the reads, and therfore we enumerate
		 * over all possible deletions in the entire read.
		 * */
		/*
		   if(numDeletions > 0) {
		   RGReadsGenerateDeletions(read,
		   readLength,
		   offsets[i],
		   numDeletions,
		   index,
		   reads);
		   }
		   */

		/* Go through all insertions */
		/* Note: we allow only contiguous insertions of length up to
		 * numInsertions.  We also always start from the offset.  When
		 * we model insertions, we have to remove a certain number of
		 * bases in the read, and therefore we enumerate over all
		 * possible insertions in the entire read.
		 * */
		/*
		   if(numInsertions > 0) {
		   RGReadsGenerateInsertions(read,
		   readLength,
		   offsets[i],
		   numInsertions,
		   index,
		   reads);
		   }
		   */

		/* Go through all possible insertions in the gap between
		 * the pair of l-mers.  If there is a gap insertion,
		 * then we will delete bases in the gap.
		 * */
		/*
		   if(numGapInsertions > 0) {
		   RGReadsGenerateGapInsertions(read,
		   readLength,
		   offsets[i],
		   numGapInsertions,
		   index,
		   reads);
		   }
		   */

		/* Go through all possible deletions in the gap between
		 * the pair of l-mers.  If there is a gap deletion, 
		 * then we will add bases to the gap.
		 * */
		/*
		   if(numGapDeletions > 0) {
		   RGReadsGenerateGapDeletions(read,
		   readLength,
		   offsets[i],
		   numGapDeletions,
		   index,
		   reads);
		   }
		   */
	}

}

/* TODO */
void RGReadsGeneratePerfectMatch(char *read,
		int readLength,
		int offset,
		RGIndex *index,
		RGReads *reads)
{
	int32_t i;
	char *curRead;

	/* Check bounds */
	if(readLength < index->width + offset) {
		return;
	}

	curRead = malloc(sizeof(char)*(index->width+1));
	if(NULL == curRead) {
		PrintError("RGReadsPerfectMatchesHelper", "curRead", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Copy over */
	for(i=offset;i<index->width + offset;i++) {
		curRead[i-offset] = read[i];
	}
	curRead[index->width]='\0';

	/* Append */
	RGReadsAppend(reads, curRead, index->width, offset);

	free(curRead);
	curRead = NULL;
}

/* TODO */
void RGReadsGenerateMismatches(char *read,
		int readLength,
		int offset,
		int numMismatches,
		RGIndex *index,
		RGReads *reads)
{
	char *curRead=NULL;

	/* Check bounds */
	if(readLength < index->width+offset) {
		return;
	}

	/* Allocate memory */
	curRead = malloc(sizeof(char)*(index->width+1));
	if(NULL == curRead) {
		PrintError("RGReadsGenerateMismatches", "curRead", "Could not allocate memory", Exit, MallocMemory);
	}

	RGReadsGenerateMismatchesHelper(read,
			readLength,
			offset,
			numMismatches,
			curRead,
			0,
			index,
			reads);

	/* Free memory */
	free(curRead);
}

/* TODO */
void RGReadsGenerateMismatchesHelper(char *read,
		int readLength,
		int offset,
		int numMismatchesLeft,
		char *curRead,
		int curReadIndex,
		RGIndex *index,
		RGReads *reads)
{
	int i;

	if(curReadIndex > index->width) {
		return;
	}

	if(numMismatchesLeft > 0) {
		assert(curReadIndex <= index->width);
		/* No more to print */
		if(curReadIndex == index->width) {
			curRead[index->width]='\0';
			RGReadsAppend(reads, curRead, index->width, offset);
			return;
		}
		else {
			assert(curReadIndex < index->width);
			/* use mismatches */
			for(i=0;i<ALPHABET_SIZE;i++) {
				int tempReadIndex = curReadIndex;
				while(index->mask[tempReadIndex] == 0 &&
						tempReadIndex < readLength) {
					curRead[tempReadIndex] = read[offset+tempReadIndex];
					tempReadIndex++;
				}
				assert(tempReadIndex < readLength);
				if(index->mask[tempReadIndex] == 0) {
					return;
				}
				curRead[tempReadIndex] = ALPHABET[i];
				if(read[offset+tempReadIndex] == ALPHABET[i]) {
					/* No mismatch */
					/* Keep going */
					RGReadsGenerateMismatchesHelper(read,
							readLength,
							offset,
							numMismatchesLeft,
							curRead,
							tempReadIndex+1,
							index,
							reads);
				}
				else {
					/* Mismatch */

					/* Keep going */
					RGReadsGenerateMismatchesHelper(read,
							readLength,
							offset,
							numMismatchesLeft-1,
							curRead,
							tempReadIndex+1,
							index,
							reads);
				}
			}
		}
	}
	else {
		/* print remaining */                                           
		while(curReadIndex < index->width) {
			curRead[curReadIndex] = read[curReadIndex+offset];
			curReadIndex++;

		}
		assert(curReadIndex == index->width);
		curRead[index->width]='\0';
		/* Append */
		RGReadsAppend(reads, curRead, index->width, offset);
	}
}

/* TODO */
/* Note: Deletions have occured, so insert bases */
void RGReadsGenerateDeletions(char *read,
		int readLength,
		int offset,
		int numDeletions,
		RGIndex *index,
		RGReads *reads)
{
	char *curRead=NULL;
	int i;

	/* Allocate memory */
	curRead = malloc(sizeof(char)*(index->width+1));
	if(NULL == curRead) {
		PrintError("RGReadsGenerateDeletions", "curRead", "Could not allocate memory", Exit, MallocMemory);
	}

	for(i=1;i<=numDeletions;i++) {
		RGReadsGenerateDeletionsHelper(read,
				readLength,
				offset,
				i,
				i,
				0,
				curRead,
				0,
				index,
				reads);
	}

	/* Free memory */
	free(curRead);
}

/* TODO */
/* NOTE: no error checking yet! */
/* Deletion occured, so insert bases */
void RGReadsGenerateDeletionsHelper(char *read,
		int readLength,
		int offset,
		int numDeletionsLeft,
		int numDeletions,
		int deletionOffset,
		char *curRead,
		int curReadIndex,
		RGIndex *index,
		RGReads *reads)
{
	int i;

	if(curReadIndex > index->width) {
		return;
	}

	if(numDeletionsLeft > 0) {
		/* No more to print */
		if(curReadIndex == index->width) {
			if(numDeletionsLeft != numDeletions) {
				curRead[index->width]='\0';
				/* Append */
				RGReadsAppend(reads, curRead, index->width, offset);
			}
			return;
		}
		else {
			/* Update curReadIndex etc. based on current tile */
			int tempReadIndex = curReadIndex;
			while(index->mask[tempReadIndex] == 0 &&
					tempReadIndex < readLength) {
				curRead[tempReadIndex] = read[offset+tempReadIndex-deletionOffset];
				tempReadIndex++;
			}
			assert(tempReadIndex < readLength);
			if(index->mask[tempReadIndex] == 0) {
				return;
			}
			/* try inserting a base - do not insert at the beginning or the end of a read */
			if(curReadIndex > 0 && curReadIndex < readLength-1) {
				for(i=0;i<ALPHABET_SIZE;i++) {
					curRead[curReadIndex] = ALPHABET[i];
					/* Use on first read */
					RGReadsGenerateDeletionsHelper(read,
							readLength,
							offset,
							numDeletionsLeft-1,
							numDeletions,
							deletionOffset+1,
							curRead,
							tempReadIndex+1,
							index,
							reads);
				}
			}
			/* This will enforce that insertions occur together */
			if(numDeletionsLeft == numDeletions) {
				/* Try not inserting a base */
				curRead[curReadIndex] = read[offset+curReadIndex-deletionOffset];
				RGReadsGenerateDeletionsHelper(read,
						readLength,
						offset,
						numDeletionsLeft,
						numDeletions,
						deletionOffset,
						curRead,
						tempReadIndex+1,
						index,
						reads);
			}
		}
	}
	else {
		/* print remaining */                                           
		while(curReadIndex < index->width) {
			curRead[curReadIndex] = read[curReadIndex+offset-deletionOffset];
			curReadIndex++;
		}
		curRead[index->width]='\0';
		assert(curReadIndex == index->width);
		/* Append */
		RGReadsAppend(reads, curRead, index->width, offset);
		return;

	}
}

/* TODO */
/* Note: Insertions have occured, so delete bases */
void RGReadsGenerateInsertions(char *read,
		int readLength,
		int offset,
		int numInsertions,
		RGIndex *index,
		RGReads *reads)
{
	char *curRead=NULL;
	int maxNumInsertions = 0;
	int i;

	/* Get the total number of insertions (delete bases) possible */
	maxNumInsertions = readLength - index->width;

	if(maxNumInsertions <= 0) {
		/* Cannot delete any bases */
		return;
	}
	else if(maxNumInsertions < numInsertions) {
		/* Can only delete a certain # of bases */
		numInsertions = maxNumInsertions;
	}

	/* Allocate memory */
	curRead = malloc(sizeof(char)*(index->width+1));
	if(NULL == curRead) {
		PrintError("RGReadsGenerateInsertions", "curRead", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Try up to the number of insertions */
	for(i=1;i<=numInsertions;i++) {
		RGReadsGenerateInsertionsHelper(read,
				readLength,
				offset,
				i,
				i,
				0,
				curRead,
				0,
				index,
				reads);
	}

	/* Free memory */
	free(curRead);
}

/* TODO */
/* NOTE: no error checking yet! */
/* Try deleting bases from the read */
void RGReadsGenerateInsertionsHelper(char *read,
		int readLength,
		int offset,
		int numInsertionsLeft,
		int numInsertions,
		int insertionOffset,
		char *curRead,
		int curReadIndex,
		RGIndex *index,
		RGReads *reads)
{
	if(curReadIndex > index->width) {
		return;
	}

	if(numInsertionsLeft > 0) {
		/* No more to print */
		if(curReadIndex >= index->width) {
			if(numInsertionsLeft != numInsertions) {
				curRead[index->width]='\0';
				/* Append */
				RGReadsAppend(reads, curRead, index->width, offset);
			}
			return;
		}
		/* try deleting a base */
		if(curReadIndex == 0 || 
				read[curReadIndex-1] != read[curReadIndex]) {
			RGReadsGenerateInsertionsHelper(read,
					readLength,
					offset,
					numInsertionsLeft-1,
					numInsertions,
					insertionOffset+1,
					curRead,
					curReadIndex,
					index,
					reads);
		}
		/* Try not deleting a base */
		/* Only do this if we haven't started deleting */ 
		if(numInsertionsLeft == numInsertions) {
			int tempReadIndex = curReadIndex;
			while(index->mask[tempReadIndex] == 0 &&
					tempReadIndex < readLength) {
				curRead[tempReadIndex] = read[offset+tempReadIndex+insertionOffset];
				tempReadIndex++;
			}
			assert(tempReadIndex < readLength);
			if(index->mask[tempReadIndex] == 0) {
				return;
			}
			curRead[tempReadIndex] = read[offset+tempReadIndex+insertionOffset];
			RGReadsGenerateInsertionsHelper(read,
					readLength,
					offset,
					numInsertionsLeft,
					numInsertions,
					insertionOffset,
					curRead,
					tempReadIndex+1,
					index,
					reads);
		}
	}
	else {
		/* print remaining */                                           
		while(curReadIndex<index->width) {
			curRead[curReadIndex] = read[curReadIndex+offset+insertionOffset];
			curReadIndex++;
		}
		curRead[index->width]='\0';
		assert(curReadIndex == index->width);
		/* Append */
		RGReadsAppend(reads, curRead, index->width, offset);
		return;
	}
}

/* TODO */
/* Note: Deletions have occured, so insert bases in the gaps */
void RGReadsGenerateGapDeletions(char *read,
		int readLength,
		int offset,
		int numGapDeletions,
		RGIndex *index,
		RGReads *reads)
{
	char *curRead = NULL;

	if(numGapDeletions <= 0) {
		return;
	}
	/* Allocate memory */
	curRead = malloc(sizeof(char)*(readLength+1));
	if(NULL == curRead) {
		PrintError("RGReadsGenerateGapDeletions", "curRead", "Could not allocate memory", Exit, MallocMemory);
	}
	RGReadsGenerateGapDeletionsHelper(read,
			readLength,
			offset,
			numGapDeletions,
			curRead,
			index,
			reads);

	/* Free memory */
	free(curRead);
}

/* TODO */
/* NOTE: no error checking yet! */
/* We assume that all insertions in the gap are grouped together */
void RGReadsGenerateGapDeletionsHelper(char *read,
		int readLength,
		int offset,
		int numGapDeletions,
		char *curRead,
		RGIndex *index,
		RGReads *reads)
{
	int i, j, k;
	int readPos;
	int curReadPos;

	int prevType=0;
	int gapLength=0;
	int numToInsert=0;

	/* Choose a gap to insert bases */
	readPos = offset;
	curReadPos = 0;
	for(i=0;i<index->width;i++) {
		/* Previous base was a 1 and we are not at a 0 */
		if(prevType == 1 &&
				index->mask[i] == 0) {

			/* Get the gap length */
			gapLength = 0;
			for(j=i;index->mask[j]==0;j++) {
				gapLength++;
			}
			/* Insert the min(gapLength, numGapDeletions) bases into the gap, since
			 * we require the bases in the gap */
			numToInsert = (gapLength < numGapDeletions)?gapLength:numGapDeletions;

			/* j is the current number of bases we are inserting */
			for(j=1;j<=numToInsert;j++) {
				int tempCurReadPos = curReadPos;
				int tempReadPos = readPos;

				/* Insert bases into the gap */
				for(k=0;k<j;k++) {
					curRead[tempCurReadPos] = NULL_LETTER;
					tempCurReadPos++;
				}
				/* Copy over the bases after the inserted bases */
				while(tempCurReadPos < index->width) {
					curRead[tempCurReadPos] = read[tempReadPos];
					tempCurReadPos++;
					tempReadPos++;
				}
				assert(tempCurReadPos == index->width);
				curRead[index->width]='\0';
				/* Append */
				RGReadsAppend(reads, curRead, index->width, offset);
			}
		}
		/* Copy base */
		curRead[curReadPos] = read[readPos];
		curReadPos++;
		readPos++;
		prevType = index->mask[i];
	}
}

/* TODO */
/* Note: Insertions have occured, so delete bases */
void RGReadsGenerateGapInsertions(char *read,
		int readLength,
		int offset,
		int numGapInsertions,
		RGIndex *index,
		RGReads *reads)
{
	char *curRead=NULL;

	if(numGapInsertions <= 0) {
		/* Out of bounds.  Don't add anything. */
		return;
	}

	/* Allocate memory */
	curRead = malloc(sizeof(char)*(readLength+1));
	if(NULL == curRead) {
		PrintError("RGReadsGenerateGapInsertions", "curRead", "Could not allocate memory", Exit, MallocMemory);
	}

	RGReadsGenerateGapInsertionsHelper(read,
			readLength,
			offset,
			numGapInsertions,
			curRead,
			index,
			reads);

	/* Free memory */
	free(curRead);
}

/* TODO */
/* NOTE: no error checking yet! */
/* Delete bases in the gaps */
void RGReadsGenerateGapInsertionsHelper(char *read,
		int readLength,
		int offset,
		int numGapInsertions,
		char *curRead,
		RGIndex *index,
		RGReads *reads)
{
	int i, j;
	int readPos;
	int curReadPos;

	int gapLength = 0;
	int prevType = 1;
	int numToDelete = 0;

	/* Choose a gap to try to remove bases */
	readPos = offset;
	curReadPos = 0;
	for(i=0;i<index->width;i++) {
		/* Previous base was a 1 and we are not at a 0 */
		if(prevType == 1 &&
				index->mask[i] == 0) {

			/* Get the gap length */
			gapLength = 0;
			for(j=i;index->mask[j]==0;j++) {
				gapLength++;
			}

			/* Delete min(gapLength, (readLength - readPos) - (index->width - curReadPos)).  We can only 
			 * remove as many bases as we can shift into the gap. 
			 */
			numToDelete = (gapLength < (readLength - readPos) - (index->width - curReadPos))?gapLength:(readLength - readPos) - (index->width - curReadPos);
			/* j is the current number of bases we are deleting */
			for(j=1;j<=numToDelete;j++) {
				int tempCurReadPos = curReadPos;
				int tempReadPos = readPos+j;

				/* Copy over the bases after the deleted bases */
				while(tempCurReadPos < index->width) {
					assert(tempReadPos < readLength);
					curRead[tempCurReadPos] = read[tempReadPos];
					tempCurReadPos++;
					tempReadPos++;
				}
				assert(tempCurReadPos == index->width);
				curRead[index->width]='\0';
				/* Append */
				RGReadsAppend(reads, curRead, index->width, offset);
			}
		}
		/* Copy base */
		curRead[curReadPos] = read[readPos];
		curReadPos++;
		readPos++;
		prevType = index->mask[i];
	}
}

/* TODO */
void RGReadsRemoveDuplicates(RGReads *s)
{
	int32_t i;
	int32_t prevIndex=0;

	if(s->numReads <= 0) {
		return;
	}

	/* Sort the data structure */
	RGReadsQuickSort(s, 0, s->numReads-1);

	/* Remove duplicates */
	prevIndex=0;
	for(i=1;i<s->numReads;i++) {
		if(RGReadsCompareAtIndex(s, prevIndex, s, i)==0) { 
			/* Ignore */
		}
		else {
			prevIndex++;
			/* Copy over to temporary pair */
			RGReadsCopyAtIndex(s, prevIndex, s, i);
		}
	}

	/* Reallocate pair */
	RGReadsReallocate(s, prevIndex+1);

}

/* TO DO */
void RGReadsQuickSort(RGReads *s, int low, int high)
{
	char *FnName="RGReadsQuickSort";
	int32_t i;
	int32_t pivot=-1;
	RGReads *temp;

	if(low < high) {

		if(high - low + 1 <= RGREADS_SHELL_SORT_MAX) {
			RGReadsShellSort(s, low, high);
			return;
		}

		/* Allocate memory for the temp RGReads indexes */
		temp = malloc(sizeof(RGReads));
		if(NULL == temp) {
			PrintError(FnName, "temp", "Could not allocate memory", Exit, MallocMemory);
		}
		RGReadsInitialize(temp);
		RGReadsAllocate(temp, 1);
		temp->reads[0] = malloc(sizeof(char)*SEQUENCE_LENGTH);
		if(NULL == temp->reads[0]) {
			PrintError("RGReadsQuickSort", "temp->reads[0]", "Could not allocate memory", Exit, MallocMemory);
		}
		temp->reads[0][0]='\0';
		assert(temp->numReads == 1);

		pivot = (low + high)/2;

		RGReadsCopyAtIndex(temp, 0, s, pivot);
		RGReadsCopyAtIndex(s, pivot, s, high);
		RGReadsCopyAtIndex(s, high, temp, 0);

		pivot = low;

		for(i=low;i<high;i++) {
			if(RGReadsCompareAtIndex(s, i, s, high) <= 0) {
				if(i!=pivot) {
					RGReadsCopyAtIndex(temp, 0, s, i);
					RGReadsCopyAtIndex(s, i, s, pivot);
					RGReadsCopyAtIndex(s, pivot, temp, 0);
				}
				pivot++;
			}
		}
		RGReadsCopyAtIndex(temp, 0, s, pivot);
		RGReadsCopyAtIndex(s, pivot, s, high);
		RGReadsCopyAtIndex(s, high, temp, 0);

		/* Free memory before recursive call */
		assert(temp->numReads == 1);
		RGReadsFree(temp);
		free(temp);
		temp=NULL;

		RGReadsQuickSort(s, low, pivot-1);
		RGReadsQuickSort(s, pivot+1, high);
	}
}

void RGReadsShellSort(RGReads *s, int low, int high)
{
	char *FnName="RGReadsShellSort";
	int32_t i, j, inc;
	RGReads *temp;

	inc = ROUND((high - low + 1) / 2);

	/* Allocate memory for the temp RGReads indexes */
	temp = malloc(sizeof(RGReads));
	if(NULL == temp) {
		PrintError(FnName, "temp", "Could not allocate memory", Exit, MallocMemory);
	}
	RGReadsInitialize(temp);
	RGReadsAllocate(temp, 1);
	temp->reads[0] = malloc(sizeof(char)*SEQUENCE_LENGTH);
	if(NULL == temp->reads[0]) {
		PrintError("RGReadsQuickSort", "temp->reads[0]", "Could not allocate memory", Exit, MallocMemory);
	}
	temp->reads[0][0]='\0';
	assert(temp->numReads == 1);

	while(0 < inc) {
		for(i=inc + low;i<=high;i++) {
			RGReadsCopyAtIndex(temp, 0, s, i);
			j = i;
			while(inc + low <= j && RGReadsCompareAtIndex(temp, 0, s, j - inc) < 0) {
				RGReadsCopyAtIndex(s, j, s, j - inc);
				j -= inc;
			}
			RGReadsCopyAtIndex(s, j, temp, 0);
		}
		inc = ROUND(inc / SHELL_SORT_GAP_DIVIDE_BY);
	}

	RGReadsFree(temp);
	free(temp);
	temp=NULL;
}

int RGReadsCompareAtIndex(RGReads *pOne, int iOne, RGReads *pTwo, int iTwo) 
{
	int cmp;

	cmp = strcmp(pOne->reads[iOne], pTwo->reads[iTwo]);
	if(cmp < 0 ||
			(cmp == 0 && pOne->offset[iOne] < pTwo->offset[iTwo])) { 
		return -1;
	}
	else if(cmp == 0 && pOne->offset[iOne] == pTwo->offset[iTwo]) {
		return 0;
	}
	else {
		return 1;
	}
}

void RGReadsCopyAtIndex(RGReads *dest, int destIndex, RGReads *src, int srcIndex)
{
	if(dest != src || srcIndex != destIndex) {
		strcpy(dest->reads[destIndex], src->reads[srcIndex]);
		dest->readLength[destIndex] = src->readLength[srcIndex];
		dest->offset[destIndex] = src->offset[srcIndex];
	}
}

void RGReadsAllocate(RGReads *reads, int numReads)
{
	assert(reads->numReads == 0);
	reads->numReads = numReads;
	reads->reads = malloc(sizeof(char*)*reads->numReads);
	if(NULL == reads->reads) {
		PrintError("RGReadsAllocate", "reads->reads", "Could not allocate memory", Exit, MallocMemory);
	}
	reads->readLength= malloc(sizeof(int32_t)*(reads->numReads));
	if(NULL == reads->readLength) {
		PrintError("RGReadsAllocate", "reads->readLength", "Could not allocate memory", Exit, MallocMemory);
	}
	reads->offset = malloc(sizeof(int32_t)*(reads->numReads));
	if(NULL == reads->offset) {
		PrintError("RGReadsAllocate", "reads->offset", "Could not allocate memory", Exit, MallocMemory);
	}
}

void RGReadsReallocate(RGReads *reads, int numReads) 
{
	int i;
	if(numReads > 0) {
		/* Remember to free the reads that will be reallocated if we go to less */
		if(numReads < reads->numReads) {
			for(i=numReads;i<reads->numReads;i++) {
				free(reads->reads[i]);
			}
		}
		reads->numReads = numReads;
		reads->reads = realloc(reads->reads, sizeof(char*)*(reads->numReads));
		if(NULL == reads->reads) {
			PrintError("RGReadsReallocate", "reads->reads", "Could not reallocate memory", Exit, MallocMemory);
		}
		reads->readLength = realloc(reads->readLength, sizeof(int32_t)*(reads->numReads));
		if(NULL == reads->readLength) {
			PrintError("RGReadsReallocate", "reads->readLength", "Could not reallocate memory", Exit, MallocMemory);
		}
		reads->offset = realloc(reads->offset, sizeof(int32_t)*(reads->numReads));
		if(NULL == reads->offset) {
			PrintError("RGReadsReallocate", "reads->offset", "Could not reallocate memory", Exit, MallocMemory);
		}
	}
	else {
		RGReadsFree(reads);
	}
}

void RGReadsFree(RGReads *reads) 
{
	int i;

	/* Free memory from reads */
	for(i=0;i<reads->numReads;i++) {
		free(reads->reads[i]);
		reads->reads[i] = NULL;
	}
	free(reads->reads);
	free(reads->readLength);
	free(reads->offset);
	RGReadsInitialize(reads);
}

void RGReadsInitialize(RGReads *reads) 
{
	reads->reads=NULL;
	reads->readLength=NULL;
	reads->offset=NULL;
	reads->numReads=0;
}

void RGReadsAppend(RGReads *reads, 
		char *read,
		int32_t readLength,
		int32_t offset) 
{
	char *FnName = "RGReadsAppend";

	/* Allocate memory */
	RGReadsReallocate(reads, reads->numReads+1);
	/* Allocate memory for read */
	reads->reads[reads->numReads-1] = malloc(sizeof(char)*(readLength+1));
	if(NULL == reads->reads[reads->numReads-1]) {
		PrintError(FnName, "reads->reads[reads->numReads-1]", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Copy over */
	strcpy(reads->reads[reads->numReads-1], read);
	reads->readLength[reads->numReads-1] = readLength;
	reads->offset[reads->numReads-1] = offset;
}

/* TODO */
/* Debugging procedure */
void RGReadsPrint(RGReads *reads, RGIndex *index) 
{
	int i;
	for(i=0;i<reads->numReads;i++) {
		RGIndexPrintReadMasked(index, reads->reads[i], 0, stderr);
		fprintf(stderr, "%s\t%d\t%d\n",
				reads->reads[i],
				reads->readLength[i],
				reads->offset[i]);
	}
}
