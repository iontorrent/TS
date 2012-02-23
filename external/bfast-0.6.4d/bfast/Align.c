#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <math.h>
#include <config.h>
#include "BLib.h"
#include "BLibDefinitions.h"
#include "BError.h"
#include "RGMatches.h"
#include "AlignedRead.h"
#include "AlignedEnd.h"
#include "AlignedEntry.h"
#include "ScoringMatrix.h"
#include "AlignNTSpace.h"
#include "AlignColorSpace.h"
#include "RGMatch.h"
#include "Align.h"

int AlignRGMatches(RGMatches *m,
		RGBinary *rg,
		AlignedRead *a,
		int32_t space,
		int32_t offset,
		ScoringMatrix *sm,
		int32_t ungapped,
		int32_t unconstrained,
		int32_t bestOnly,
		int32_t usePairedEndLength,
		int32_t pairedEndLength,
		int32_t mirroringType,
		int32_t forceMirroring,
		AlignMatrix *matrix)
{
	double bestScore;
	int32_t i;
	int32_t numLocalAlignments = 0;
	int32_t numAligned=0;

	/* Check to see if we should try to align one read with no candidate
	 * locations if the other one has candidate locations.
	 *
	 * This assumes that the the first read is 5'->3' before the second read.
	 * */
	if(2 == m->numEnds && usePairedEndLength == 1) {
		RGMatchesMirrorPairedEnd(m,
				rg,
				pairedEndLength,
				mirroringType,
				forceMirroring);
	}

	AlignedReadAllocate(a,
			m->readName,
			m->numEnds,
			space);

	/* Align each end individually */
	for(i=0;i<m->numEnds;i++) {
		/* Align an end */
		AlignRGMatchesOneEnd(&m->ends[i],
				rg,
				&a->ends[i],
				space,
				offset,
				sm,
				ungapped,
				unconstrained,
				bestOnly,
				&bestScore,
				&numAligned,
				matrix);
		if(BestOnly == bestOnly) {
			numLocalAlignments += AlignRGMatchesKeepBestScore(&a->ends[i],
					bestScore);
		}
		else {
			numLocalAlignments += numAligned;
		}
	}
	return numLocalAlignments;
}

/* TODO */
void AlignRGMatchesOneEnd(RGMatch *m,
		RGBinary *rg,
		AlignedEnd *end,
		int32_t space,
		int32_t offset,
		ScoringMatrix *sm,
		int32_t ungapped,
		int32_t unconstrained,
		int32_t bestOnly,
		double *bestScore,
		int32_t *numAligned,
		AlignMatrix *matrix)
{
	char *FnName="AlignRGMatchOneEnd";
	int32_t i;
	char **references=NULL;
	char **masks=NULL;
	int32_t *referenceLengths=NULL;
	int32_t *referencePositions=NULL;
	int32_t *referenceOffsets=NULL;
	int32_t *readStartInsertionLengths=NULL;
	int32_t *readEndInsertionLengths=NULL;
	char read[SEQUENCE_LENGTH]="\0";
	char colors[SEQUENCE_LENGTH]="\0";
	int32_t readLength;
	int32_t ctr=0;
	int32_t numberFound = 0;
	int32_t prevIndex;

	(*bestScore)=NEGATIVE_INFINITY;

	strcpy(read, m->read);

	if(NTSpace == space) {
		readLength = m->readLength;
	}
	else {
		// Copy over the colors
		strcpy(colors, m->read);
		readLength = m->readLength;
		NormalizeColorSpaceRead(colors, readLength, COLOR_SPACE_START_NT);
		// This modifies the "read"
		readLength = ConvertReadFromColorSpace(read, readLength);
		// Remove the adaptor from the colors
		for(i=0;i<readLength;i++) { 
			// remember to convert to '4's
			switch(colors[i+1]) {
				case '0':
				case '1':
				case '2':
				case '3':
					colors[i] = colors[i+1]; break;
				default:
					colors[i] = '4';
			}
		}
		colors[i]='\0';
		// Both read and colors now have the same length
	}
	if(matrix->nrow < readLength+1) {
		AlignMatrixReallocate(matrix, readLength+1, GETMAX(matrix->ncol, readLength+1));
	}

	/* Allocate */
	AlignedEndAllocate(end,
			m->read,
			m->qual,
			m->numEntries);

	/* Get all the references */
	references = malloc(sizeof(char*)*m->numEntries);
	if(NULL==references) {
		PrintError(FnName, "references", "Could not allocate memory", Exit, MallocMemory);
	}
	masks = malloc(sizeof(char*)*m->numEntries);
	if(NULL==masks) {
		PrintError(FnName, "masks", "Could not allocate memory", Exit, MallocMemory);
	}
	referenceLengths = malloc(sizeof(int32_t)*m->numEntries);
	if(NULL==referenceLengths) {
		PrintError(FnName, "referenceLengths", "Could not allocate memory", Exit, MallocMemory);
	}
	referenceOffsets = malloc(sizeof(int32_t)*m->numEntries);
	if(NULL==referenceOffsets) {
		PrintError(FnName, "referenceOffsets", "Could not allocate memory", Exit, MallocMemory);
	}
	readStartInsertionLengths = malloc(sizeof(int32_t)*m->numEntries);
	if(NULL==readStartInsertionLengths) {
		PrintError(FnName, "readStartInsertionLengths", "Could not allocate memory", Exit, MallocMemory);
	}
	readEndInsertionLengths = malloc(sizeof(int32_t)*m->numEntries);
	if(NULL==readEndInsertionLengths) {
		PrintError(FnName, "readEndInsertionLengths", "Could not allocate memory", Exit, MallocMemory);
	}
	referencePositions = malloc(sizeof(int32_t)*m->numEntries);
	if(NULL==referencePositions) {
		PrintError(FnName, "referencePositions", "Could not allocate memory", Exit, MallocMemory);
	}
	for((*numAligned)=0,i=0,ctr=0;i<m->numEntries;i++) {
		references[ctr]=NULL; /* This is needed for RGBinaryGetReference */

		/* Get references */
		RGBinaryGetReference(rg,
				m->contigs[i],
				m->positions[i],
				m->strands[i], 
				offset,
				&references[ctr],
				readLength,
				&referenceLengths[ctr],
				&referencePositions[ctr]);

		assert(referenceLengths[ctr] > 0);
		/* Initialize entries */
		if(readLength <= referenceLengths[ctr] || Gapped == ungapped) {
			readStartInsertionLengths[ctr]=0;
			readEndInsertionLengths[ctr]=0;
			referenceOffsets[ctr]=offset;

			if(referencePositions[ctr] <= m->positions[i] &&
					m->positions[i] + readLength <= referencePositions[ctr] + referenceLengths[ctr]) {
			}
			else if(FORWARD == m->strands[i]) {
				if(m->positions[i] < 1) {
					readStartInsertionLengths[ctr] = 1 - m->positions[i];
				}
				if(referencePositions[ctr] + referenceLengths[ctr] < 
						m->positions[i] +  readLength) {
					readEndInsertionLengths[ctr] = m->positions[i] +  readLength -
						referencePositions[ctr] - referenceLengths[ctr];
				}
			}
			else {
				if(m->positions[i] < 1) {
					readEndInsertionLengths[ctr] = referencePositions[ctr] - m->positions[i];
				}
				if(referencePositions[ctr] + referenceLengths[ctr] < 
						m->positions[i] +  readLength) {
					readStartInsertionLengths[ctr] = m->positions[i] +  readLength -
						referencePositions[ctr] - referenceLengths[ctr];
				}
			}
			assert(readStartInsertionLengths[ctr] + readEndInsertionLengths[ctr] <= readLength);
			
			if(FORWARD == m->strands[i]) {
				referenceOffsets[ctr] = m->positions[i] - referencePositions[ctr];
			}
			else {
				referenceOffsets[ctr]  = referencePositions[ctr] + referenceLengths[ctr] - 
					m->positions[i] - readLength; 
			}
			referenceOffsets[ctr] += readStartInsertionLengths[ctr];
			
			if(matrix->nrow < readLength + 
					readStartInsertionLengths[ctr] + 
					readEndInsertionLengths[ctr] + 1) {
				AlignMatrixReallocate(matrix, 
						readLength + readStartInsertionLengths[ctr] + 
						readEndInsertionLengths[ctr] + 1,
						GETMAX(matrix->ncol, 
							readLength + readStartInsertionLengths[ctr] + 
							readEndInsertionLengths[ctr] + 1));
			}
			/* Copy over mask */
			masks[ctr] = RGMatchMaskToString(m->masks[i], m->readLength);
			/* Update contig name and strand */
			end->entries[ctr].contig = m->contigs[i];
			end->entries[ctr].strand = m->strands[i];
			/* The rest should be filled in later */
			end->entries[ctr].position = -1; 
			end->entries[ctr].score=NEGATIVE_INFINITY;
			end->entries[ctr].alnRead = NULL;
			end->entries[ctr].alnReadLength = 0;

			if(matrix->ncol < referenceLengths[ctr]+1) {
				AlignMatrixReallocate(matrix, matrix->nrow, referenceLengths[ctr]+1);
			}

			(*numAligned)++;
			ctr++;
		}
		else {
			/* Free retrieved reference sequence */
			free(references[ctr]);
			references[ctr]=NULL;
			masks[ctr]=NULL;
		}
	}

	/* Reallocate entries if necessary */
	if(ctr < end->numEntries) {
		AlignedEndReallocate(end,
				ctr);
	}

	/* Idea: 
	 * - do exact
	 *
	 * Unconstrained 
	 *   Ungapped
	 *     - if exact not found, do ungapped
	 *   Gapped
	 *     - if exact not found, do ungapped
	 *     - bound gapped with ungapped results
	 * Constrained
	 *   Ungapped
	 *     - if exact not found, do ungapped
	 *   Gapped 
	 *     - if exact not found, do gapped
	 *     */

#ifndef UNOPTIMIZED_SMITH_WATERMAN
	int32_t foundExact;
	foundExact = 0;
	/* Try exact alignment */
	for(i=0;i<end->numEntries;i++) {
		if(1==AlignExact(read, 
					readLength, 
					references[i], 
					referenceLengths[i],
					sm,
					&end->entries[i],
					space,
					referenceOffsets[i],
					referencePositions[i],
					end->entries[i].strand)) {
			foundExact=1;
			numberFound++;
			if((*bestScore) < end->entries[i].score) {
				(*bestScore) = end->entries[i].score;
			}
		}
	}

	/* If we are to only output the best alignments and we have found an exact alignment, return */
	if(1==foundExact && bestOnly == BestOnly) {
		for(i=0;i<end->numEntries;i++) {
			free(references[i]);
			free(masks[i]);
		}
		free(references);
		free(masks);
		free(referenceLengths);
		free(referencePositions);
		free(referenceOffsets);
		free(readStartInsertionLengths);
		free(readEndInsertionLengths);
		return;
	}
#endif

#ifdef UNOPTIMIZED_SMITH_WATERMAN
	if(Ungapped == ungapped) {
#endif
		/* Only do ungapped local alignment if we are not doing gapped
		 * local alignment (i.e. ungapped) or if we are not doing 
		 * constrained local alignment (i.e unconstrained).  This is because ungapped
		 * local alignment is not being used to bound the gappel local alignment.
		 * */
		if(Gapped != ungapped || Constrained != unconstrained) {
			for(i=0;i<end->numEntries;i++) {
				if(!(NEGATIVE_INFINITY < end->entries[i].score)) { // If we did not find an exact match
					numberFound += AlignUngapped(read,
							colors,
							masks[i],
							readLength,
							references[i],
							referenceLengths[i],
							unconstrained,
							sm,
							&end->entries[i],
							space,
							referenceOffsets[i],
							referencePositions[i],
							end->entries[i].strand);
					if((*bestScore) < end->entries[i].score) {
						(*bestScore) = end->entries[i].score;
					}
				}
			}
		}

		/* Return if we are only to be searching for mismatches */
#ifndef UNOPTIMIZED_SMITH_WATERMAN
		if(Ungapped == ungapped) {
#endif
			// Remove empty alignments
			if(numberFound != end->numEntries) {
				for(prevIndex=i=0;i<end->numEntries;i++) {
					if(NEGATIVE_INFINITY < end->entries[i].score) { 
						// Alignment exits
						if(prevIndex != i) { // We are not going to copy to the same location
							// Copy to prevIndex
							AlignedEntryCopy(&end->entries[prevIndex], &end->entries[i]);
							AlignedEntryFree(&end->entries[i]); // Free
						}
						prevIndex++;
					}
					else {
						// No alignment, free
						AlignedEntryFree(&end->entries[i]);
					}
				}
				assert(prevIndex == numberFound);
				assert(prevIndex < end->numEntries);
				// Reallocate
				AlignedEndReallocate(end, prevIndex);
			}
			for(i=0;i<end->numEntries;i++) {
				free(references[i]);
				free(masks[i]);
			}
			free(references);
			free(masks);
			free(referenceLengths);
			free(referencePositions);
			free(referenceOffsets);
			free(readStartInsertionLengths);
			free(readEndInsertionLengths);

			return;
			/* These compiler commands aren't necessary, but are here for vim tab indenting */
#ifdef UNOPTIMIZED_SMITH_WATERMAN
		}
#else 
	}
#endif

	/* Run Gapped */
	for(i=0;i<end->numEntries;i++) {
		AlignGapped(read,
				colors,
				masks[i],
				readLength,
				references[i],
				referenceLengths[i],
				unconstrained,
				sm,
				&end->entries[i],
				matrix,
				space,
				referenceOffsets[i],
				readStartInsertionLengths[i],
				readEndInsertionLengths[i],
				referencePositions[i],
				end->entries[i].strand,
				(BestOnly == bestOnly)?(*bestScore):end->entries[i].score);
		if((*bestScore) < end->entries[i].score) {
			(*bestScore) = end->entries[i].score;
		}
	}

	for(i=0;i<end->numEntries;i++) {
		free(references[i]);
		free(masks[i]);
	}
	free(references);
	free(masks);
	free(referenceLengths);
	free(referencePositions);
	free(referenceOffsets);
	free(readStartInsertionLengths);
	free(readEndInsertionLengths);
}

/* TODO */
int32_t AlignExact(char *read,
		int32_t readLength,
		char *reference,
		int32_t referenceLength,
		ScoringMatrix *sm,
		AlignedEntry *a,
		int32_t space,
		int32_t offset,
		int32_t position,
		char strand)
{
	char *FnName="AlignExact";
	int32_t i;
	int32_t foundOffset = -1;

	// Use this when we want to perform exact substring matching 
	//foundOffset = KnuthMorrisPratt(read, readLength, reference, referenceLength);

	// Use this when we want to perform exact string matching
	for(i=0, foundOffset=offset;i<readLength;i++) {
		if(ToUpper(read[i]) != ToUpper(reference[offset+i])) {
			foundOffset = -1;
			break;
		}
	}

	if(foundOffset < 0) {
		return -1;
	}
	else {
		char prevReadBase = COLOR_SPACE_START_NT;
		char referenceAligned[SEQUENCE_LENGTH]="\0";
		char colorErrorAligned[SEQUENCE_LENGTH]="\0";
		int32_t score=0;

		for(i=0;i<readLength;i++) {
			if(ColorSpace == space) {
				char curColor='X';
				if(0 == ConvertBaseToColorSpace(prevReadBase, read[i], &curColor)) {
					PrintError(FnName, "curColor", "Could not convert base to color space", Exit, OutOfRange);
				}
				curColor = COLORFROMINT(curColor);
				/* Add score for color error, if any */
				score += ScoringMatrixGetColorScore(curColor,
						curColor,
						sm);

				colorErrorAligned[i] = GAP;
			}
			assert(ToLower(read[i]) == ToLower(reference[i+foundOffset])); 
			score += ScoringMatrixGetNTScore(read[i], read[i], sm);
			referenceAligned[i] = reference[i+foundOffset];
		}
		referenceAligned[readLength]='\0';
		colorErrorAligned[readLength]='\0';

		AlignedEntryUpdateAlignment(a,
				(FORWARD==strand) ? (position + foundOffset) : (position + referenceLength - readLength - foundOffset),
				score,
				readLength,
				readLength,
				read,
				referenceAligned);
	}

	return 1;
}

int32_t AlignUngapped(char *read,
		char *colors,
		char *mask,
		int32_t readLength,
		char *reference,
		int32_t referenceLength,
		int32_t unconstrained,
		ScoringMatrix *sm,
		AlignedEntry *a,
		int32_t space,
		int32_t offset,
		int32_t position,
		char strand)
{
	// Note: we do not allow any wiggle room in ungapped alignment
	char *FnName="AlignUngapped";

	switch(space) {
		case NTSpace:
			return AlignNTSpaceUngapped(read,
					mask,
					readLength,
					reference,
					referenceLength,
					unconstrained,
					sm,
					a,
					offset, //(Unconstrained == unconstrained) ? 0 : offset,
					position,
					strand);
			break;
		case ColorSpace:
			return AlignColorSpaceUngapped(colors,
					mask,
					readLength,
					reference,
					referenceLength,
					unconstrained,
					sm,
					a,
					offset, //(Unconstrained == unconstrained) ? 0 : offset,
					position,
					strand);
			break;
		default:
			PrintError(FnName, "space", "Could not understand space", Exit, OutOfRange);
			break;
	}
	return 0;
}

void AlignGapped(char *read,
		char *colors,
		char *mask,
		int32_t readLength,
		char *reference,
		int32_t referenceLength,
		int32_t unconstrained,
		ScoringMatrix *sm,
		AlignedEntry *a,
		AlignMatrix *matrix,
		int32_t space,
		int32_t referenceOffset,
		int32_t readStartInsertionLength,
		int32_t readEndInsertionLength,
		int32_t position,
		char strand,
		double lowerBound)
{
	char *FnName="AlignGapped";
	int64_t maxH, maxV;

	/* Get the maximum number of vertical and horizontal moves allowed */
	maxV = maxH = 0;
	if(sm->gapOpenPenalty < sm->gapExtensionPenalty) {
		if(NTSpace == space) {
			/* b = nt match score */
			/* p = gap open */
			/* e = gap extend */
			/* N = read length */
			/* Find x such that b(N) + p + e(x - 1) < Bound */
			/* x < (e + Bound - p - b(N)) / e */
			maxH = GETMAX(0, (int32_t)ceil((lowerBound - sm->ntMatch*readLength  - sm->gapOpenPenalty + sm->gapExtensionPenalty) / sm->gapExtensionPenalty));
			/* Find x such that b(N - x) + p + e(x - 1) < lowerBound */
			/* b(N) - x(b) + p + e(x) - e < lowerBound */
			/* - x(b) + e(x) < lowerBound + e -p - b(N) */
			/* x < (lowerBound - e - p - b(N) ) / (e - b)*/
			maxV = GETMAX(0, (int32_t)ceil((lowerBound - sm->ntMatch*readLength  - sm->gapOpenPenalty + sm->gapExtensionPenalty) / (sm->gapExtensionPenalty - sm->ntMatch)));
		}
		else {
			/* c = color match score */
			/* b = nt match score */
			/* p = gap open */
			/* e = gap extend */
			/* N = read length */
			/* Find x such that (c + b)N + p + e(x - 1) < Bound */
			maxH = GETMAX(0, (int32_t)ceil((lowerBound - (sm->colorMatch + sm->ntMatch)*readLength  - sm->gapOpenPenalty + sm->gapExtensionPenalty) / sm->gapExtensionPenalty));
			/* Find x such that (c + b)(N - x) + p + e(x - 1) < lowerBound */
			maxV = GETMAX(0, (int32_t)ceil((lowerBound - (sm->colorMatch + sm->ntMatch)*readLength  - sm->gapOpenPenalty + sm->gapExtensionPenalty) / (sm->gapExtensionPenalty - sm->colorMatch - sm->ntMatch)));
		}
		assert(maxH >= 0 && maxV >= 0);
	}
	else {
		// Default to maximums
		maxH = GETMIN(maxH, readLength);
		maxV = GETMIN(maxV, readLength);
		/*
		   PrintError(FnName, PACKAGE_BUGREPORT, "This is currently not implemented, please report", Exit, OutOfRange);
		   */
	}
	if(maxH == 0 && maxV == 0) {
		/* Use result from searching only mismatches */
		return;
	}

	/* Re-bound the maximum number of vertical and horizontal moves */
	maxH = GETMIN(maxH, readLength);
	maxV = GETMIN(maxV, readLength);

	/* Free relevant entries */
	free(a->alnRead);
	a->alnRead = NULL;
	a->alnReadLength = 0;
	assert(NULL == a->alnRead);
	
	switch(unconstrained) {
		case Unconstrained:
			AlignGappedBounded(read,
					colors,
					readLength,
					reference,
					referenceLength,
					sm,
					a,
					matrix,
					space,
					position,
					strand,
					lowerBound,
					maxH, 
					maxV);
			break;
		case Constrained:
			AlignGappedConstrained(read,
					colors,
					mask,
					readLength,
					reference,
					referenceLength,
					sm,
					a,
					matrix,
					space,
					referenceOffset,
					readStartInsertionLength,
					readEndInsertionLength,
					position,
					strand);
			break;
		default:
			PrintError(FnName, "unconstrained", "Could not understand unconstrained", Exit, OutOfRange);
			break;
	}
}

void AlignGappedBounded(char *read,
		char *colors,
		int32_t readLength,
		char *reference,
		int32_t referenceLength,
		ScoringMatrix *sm,
		AlignedEntry *a,
		AlignMatrix *matrix,
		int32_t space,
		int32_t position,
		char strand,
		double lowerBound,
		int32_t maxH,
		int32_t maxV)
{
	char *FnName="AlignGappedBounded";

	switch(space) {
		case NTSpace:
			AlignNTSpaceGappedBounded(read,
					readLength,
					reference,
					referenceLength,
					sm,
					a,
					matrix,
					position,
					strand,
					maxH,
					maxV);
			break;
		case ColorSpace:
			AlignColorSpaceGappedBounded(colors,
					readLength,
					reference,
					referenceLength,
					sm,
					a,
					matrix,
					position,
					strand,
					maxH,
					maxV);
			break;
		default:
			PrintError(FnName, "space", "Could not understand space", Exit, OutOfRange);
			break;
	}
}

void AlignGappedConstrained(char *read,
		char *colors,
		char *mask,
		int32_t readLength,
		char *reference,
		int32_t referenceLength,
		ScoringMatrix *sm,
		AlignedEntry *a,
		AlignMatrix *matrix,
		int32_t space,
		int32_t referenceOffset,
		int32_t readStartInsertionLength,
		int32_t readEndInsertionLength,
		int32_t position,
		char strand)
{
	char *FnName="AlignGappedConstrained";
	
	switch(space) {
		case NTSpace:
			AlignNTSpaceGappedConstrained(read,
					mask,
					readLength,
					reference,
					referenceLength,
					sm,
					a,
					matrix,
					referenceOffset,
					readStartInsertionLength,
					readEndInsertionLength,
					position,
					strand);
			break;
		case ColorSpace:
			AlignColorSpaceGappedConstrained(colors,
					mask,
					readLength,
					reference,
					referenceLength,
					sm,
					a,
					matrix,
					referenceOffset,
					readStartInsertionLength,
					readEndInsertionLength,
					position,
					strand);
			break;
		default:
			PrintError(FnName, "space", "Could not understand space", Exit, OutOfRange);
			break;
	}
}

int32_t AlignRGMatchesKeepBestScore(AlignedEnd *end,
		double bestScore)
{
	char *FnName="AlignRGMatchesKeepBestScore";
	int32_t curIndex, i;
	int32_t numLocalAlignments = 0;

	for(curIndex=0, i=0;
			i<end->numEntries;
			i++) {
		if(NEGATIVE_INFINITY < end->entries[i].score) {
			numLocalAlignments++;
		}
		if(bestScore < end->entries[i].score) {
			PrintError(FnName, "bestScore", "Best score is incorrect", Exit, OutOfRange);
		}
		else if(!(end->entries[i].score < bestScore)) {
			/* Free */
			AlignedEntryFree(&end->entries[i]);
		}
		else {
			/* Copy over to cur index */
			AlignedEntryCopyAtIndex(end->entries, curIndex, end->entries, i);
			curIndex++;
		}
	}
	assert(curIndex > 0);

	end->numEntries = curIndex;
	end->entries = realloc(end->entries, sizeof(AlignedEntry*)*end->numEntries);
	if(NULL == end->entries) {
		PrintError(FnName, "end->entries", "Could not reallocate memory", Exit, MallocMemory);
	}

	return numLocalAlignments;
}
