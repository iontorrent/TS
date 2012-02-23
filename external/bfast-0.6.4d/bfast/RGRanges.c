#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <string.h>
#include "BLibDefinitions.h"
#include "BError.h"
#include "RGMatch.h"
#include "RGRanges.h"

/* TODO */
void RGRangesCopyToRGMatch(RGRanges *r,
		RGIndex *index,
		RGMatch *m,
		int32_t space,
		int32_t copyOffsets)
{
	char *FnName="RGRangesCopyToRGMatch";
	int64_t i, j, counter, numEntries, prevNumEntries;
	int32_t k;

	if(0 < r->numEntries) {
		prevNumEntries = m->numEntries;
		for(i=numEntries=0;i<r->numEntries;i++) {
			numEntries += r->endIndex[i] - r->startIndex[i] + 1;
		}
		RGMatchReallocate(m, prevNumEntries + numEntries); 
		if(1 == copyOffsets && NULL == m->offsets) {
			assert(0 == prevNumEntries);
			m->numOffsets = malloc(sizeof(int32_t)*m->numEntries);
			if(NULL == m->numOffsets) {
				PrintError(FnName, "m->numOffsets", "Could not allocate memory", Exit, MallocMemory);
			}
			m->offsets = malloc(sizeof(int32_t*)*m->numEntries);
			if(NULL == m->offsets) {
				PrintError(FnName, "m->offsets", "Could not allocate memory", Exit, MallocMemory);
			}
			for(i=0;i<m->numEntries;i++) {
				m->numOffsets[i] = 0;
				m->offsets[i] = NULL;
			}
		}
		/* Copy over for each range */
		counter = prevNumEntries;
		for(i=0;i<r->numEntries;i++) {
			/* Copy over for the given range */
			for(j=r->startIndex[i];j<=r->endIndex[i];j++) {
				assert(j>=0 && j<index->length);
				assert(counter >= 0 && counter < m->numEntries);
				/* Get contig number */ 
				if(index->contigType == Contig_8) {
					m->contigs[counter] = index->contigs_8[j];
				}
				else {
					m->contigs[counter] = index->contigs_32[j];
				}
				/* Adjust position with the offset */
				if(FORWARD == r->strand[i]) {
					m->positions[counter] = index->positions[j] - r->offset[i];
				}
				else {
					m->positions[counter] = index->positions[j] + index->width + r->offset[i] - m->readLength;
				}
				m->strands[counter] = r->strand[i];
				if(ColorSpace == space) {
					/* In color space we removed the first base/color so we need to 
					 * decrement the positions by one.
					 * */
					m->positions[counter]--;
				}
				// Update mask
				if(FORWARD == m->strands[counter]) {
					for(k=0;k<index->width;k++) {
						if(1 == index->mask[k]) {
							int32_t offset = r->offset[i] + k;
							if(ColorSpace == space) offset++;
							RGMatchUpdateMask(m->masks[counter], 
									offset);
						}
					}
				}
				else {
					for(k=0;k<index->width;k++) {
						if(1 == index->mask[index->width - k - 1]) {
							int32_t offset = r->offset[i] + k;
							if(ColorSpace == space) offset--;
							RGMatchUpdateMask(m->masks[counter], 
									offset);
						}
					}
				}
				// Copy offsets if necessary
				if(1 == copyOffsets) {
					assert(0 == m->numOffsets[counter]);
					m->numOffsets[counter]=1;
					m->offsets[counter] = malloc(sizeof(int32_t));
					if(NULL == m->offsets[counter]) {
						PrintError(FnName, "m->offsets[counter]", "Could not allocate memory", Exit, MallocMemory);
					}
					m->offsets[counter][0] = r->offset[i];
					// Adjust for color space
					if(ColorSpace == space) {
						if(FORWARD == m->strands[counter]) m->offsets[counter][0]++;
						else m->offsets[counter][0]--;
					}
				}
				counter++;
			}
		}
		assert(counter == m->numEntries);
	}
}

void RGRangesAllocate(RGRanges *r, int32_t numEntries)
{
	assert(r->numEntries==0);
	r->numEntries = numEntries;
	assert(r->startIndex==NULL);
	r->startIndex = malloc(sizeof(int64_t)*numEntries); 
	if(NULL == r->startIndex) {
		PrintError("RGRangesAllocate", "r->startIndex", "Could not allocate memory", Exit, MallocMemory);
	}
	assert(r->endIndex==NULL);
	r->endIndex = malloc(sizeof(int64_t)*numEntries); 
	if(NULL == r->endIndex) {
		PrintError("RGRangesAllocate", "r->endIndex", "Could not allocate memory", Exit, MallocMemory);
	}
	assert(r->strand==NULL);
	r->strand = malloc(sizeof(char)*numEntries); 
	if(NULL == r->strand) {
		PrintError("RGRangesAllocate", "r->strand", "Could not allocate memory", Exit, MallocMemory);
	}
	assert(r->offset==NULL);
	r->offset = malloc(sizeof(int32_t)*numEntries); 
	if(NULL == r->offset) {
		PrintError("RGRangesAllocate", "r->offset", "Could not allocate memory", Exit, MallocMemory);
	}
}

void RGRangesReallocate(RGRanges *r, int32_t numEntries)
{
	if(numEntries > 0) {
		r->numEntries = numEntries;
		r->startIndex = realloc(r->startIndex, sizeof(int64_t)*numEntries); 
		if(numEntries > 0 && NULL == r->startIndex) {
			PrintError("RGRangesReallocate", "r->startIndex", "Could not reallocate memory", Exit, ReallocMemory);
		}
		r->endIndex = realloc(r->endIndex, sizeof(int64_t)*numEntries); 
		if(numEntries > 0 && NULL == r->endIndex) {
			PrintError("RGRangesReallocate", "r->endIndex", "Could not reallocate memory", Exit, ReallocMemory);
		}
		r->strand = realloc(r->strand, sizeof(char)*numEntries); 
		if(numEntries > 0 && NULL == r->strand) {
			PrintError("RGRangesReallocate", "r->strand", "Could not reallocate memory", Exit, ReallocMemory);
		}
		r->offset = realloc(r->offset, sizeof(int32_t)*numEntries); 
		if(numEntries > 0 && NULL == r->offset) {
			PrintError("RGRangesReallocate", "r->offset", "Could not reallocate memory", Exit, ReallocMemory);
		}
	}
	else {
		RGRangesFree(r);
	}
}

void RGRangesFree(RGRanges *r) 
{
	if(r->numEntries>0) {
		free(r->startIndex);
		free(r->endIndex);
		free(r->strand);
		free(r->offset);
	}
	RGRangesInitialize(r);
}

void RGRangesInitialize(RGRanges *r)
{
	r->numEntries=0;
	r->startIndex=NULL;
	r->endIndex=NULL;
	r->strand=NULL;
	r->offset=NULL;
}
