#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#include "BLibDefinitions.h"
#include "BLib.h"
#include "BError.h"
#include "RGMatch.h"
#include "RGMatches.h"

#define RGMATCHES_CHECK 0

/* TODO */
int32_t RGMatchesRead(gzFile fp,
		RGMatches *m)
{
	char *FnName = "RGMatchesRead";
	int32_t i;

	/* Read read name length */
	if(gzread64(fp, &m->readNameLength, sizeof(int32_t))!=sizeof(int32_t)) {
		return EOF;
	}
	assert(m->readNameLength < SEQUENCE_NAME_LENGTH);
	assert(m->readNameLength > 0);

	/* Allocate memory for the read name */
	m->readName = malloc(sizeof(char)*(m->readNameLength + 1));
	if(NULL == m->readName) {
		PrintError(FnName, "m->readName", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Read in read name */
	if(gzread64(fp, m->readName, sizeof(char)*m->readNameLength)!=sizeof(char)*m->readNameLength) {
		PrintError(FnName, "m->readName", "Could not read in read name", Exit, ReadFileError);
	}
	m->readName[m->readNameLength]='\0';
	/* Read numEnds */
	if(gzread64(fp, &m->numEnds, sizeof(int32_t))!=sizeof(int32_t)) {
		PrintError(FnName, "numEnds", "Could not read in numEnds", Exit, ReadFileError);
	}

	/* Allocate the ends */
	m->ends = malloc(sizeof(RGMatch)*m->numEnds);
	if(NULL == m->ends) {
		PrintError(FnName, "m->ends", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Read each end */
	for(i=0;i<m->numEnds;i++) {
		/* Initialize */
		RGMatchInitialize(&m->ends[i]);
		/* Read */
		RGMatchRead(fp,
				&m->ends[i]);
	}

	return 1;
}

/* TODO */
int32_t RGMatchesReadWithOffsets(gzFile fp,
		RGMatches *m)
{
	char *FnName = "RGMatchesReadWithOffsets";
	int32_t i, j;

	if(1 != RGMatchesRead(fp, m)) {
		return EOF;
	}

	/* Read each end */
	for(i=0;i<m->numEnds;i++) {
		m->ends[i].numOffsets = malloc(sizeof(int32_t)*m->ends[i].numEntries);
		if(NULL == m->ends[i].numOffsets) {
			PrintError(FnName, "numOffsets", "Could not allocate memory", Exit, MallocMemory);
		}
		m->ends[i].offsets = malloc(sizeof(int32_t*)*m->ends[i].numEntries);
		if(NULL == m->ends[i].offsets) {
			PrintError(FnName, "offsets", "Could not allocate memory", Exit, MallocMemory);
		}
		if(gzread64(fp, m->ends[i].numOffsets, sizeof(int32_t)*m->ends[i].numEntries) != sizeof(int32_t)*m->ends[i].numEntries) {
			PrintError(FnName, "numOffsets", "Could not read from file", Exit, ReadFileError);
		}
		for(j=0;j<m->ends[i].numEntries;j++) {
			m->ends[i].offsets[j] = malloc(sizeof(int32_t)*m->ends[i].numOffsets[j]);
			if(NULL == m->ends[i].offsets[j]) {
				PrintError(FnName, "offsets", "Could not allocate memory", Exit, MallocMemory);
			}
			if(gzread64(fp, m->ends[i].offsets[j], sizeof(int32_t)*m->ends[i].numOffsets[j]) != sizeof(int32_t)*m->ends[i].numOffsets[j]) {
				PrintError(FnName, "offsets[j]", "Could not read from file", Exit, ReadFileError);
			}
		}
	}

	return 1;
}

int32_t RGMatchesReadText(FILE *fp,
		RGMatches *m)
{
	char *FnName = "RGMatchesReadText";
	int32_t i;
	char readName[SEQUENCE_NAME_LENGTH]="\0";

	/* Read the matches from the input file */
	if(fscanf(fp, "%s %d", 
				readName,
				&m->numEnds) == EOF) {
		return EOF;
	}
	for(i=0;i<strlen(readName);i++) {
		readName[i] = readName[i+1];
	}
	m->readNameLength = strlen(readName);
	assert(m->readNameLength < SEQUENCE_NAME_LENGTH);
	assert(m->readNameLength > 0);
	/* Allocate memory for the read name */
	m->readName = malloc(sizeof(char)*(m->readNameLength + 1));
	if(NULL == m->readName) {
		PrintError(FnName, "m->readName", "Could not allocate memory", Exit, MallocMemory);
	}
	strcpy(m->readName, readName);

	/* Allocate the ends */
	m->ends = malloc(sizeof(RGMatch)*m->numEnds);
	if(NULL == m->ends) {
		PrintError(FnName, "m->ends", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Read each end */
	for(i=0;i<m->numEnds;i++) {
		/* Initialize */
		RGMatchInitialize(&m->ends[i]);
		/* Read */
		RGMatchReadText(fp,
				&m->ends[i]);
	}

	return 1;
}

/* TODO */
void RGMatchesPrint(gzFile fp,
		RGMatches *m)
{
	char *FnName = "RGMatchesPrint";
	int32_t i;
	assert(fp!=NULL);

	/* Print num ends, read name length, and read name */
	if(gzwrite64(fp, &m->readNameLength, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, m->readName, sizeof(char)*m->readNameLength)!=sizeof(char)*m->readNameLength ||
			gzwrite64(fp, &m->numEnds, sizeof(int32_t))!=sizeof(int32_t))  {
		PrintError(FnName, NULL, "Could not write m->readNameLength, m->readName, and m->numEnds", Exit, WriteFileError);
	}

	/* Print each end */
	for(i=0;i<m->numEnds;i++) {
		RGMatchPrint(fp,
				&m->ends[i]);
	}
}

void RGMatchesPrintWithOffsets(gzFile fp,
		RGMatches *m)
{
	char *FnName = "RGMatchesPrintWithOffsets";
	int32_t i, j;
	assert(fp!=NULL);

	RGMatchesPrint(fp, m);

	for(i=0;i<m->numEnds;i++) {
		if(gzwrite64(fp, m->ends[i].numOffsets, sizeof(int32_t)*m->ends[i].numEntries) != sizeof(int32_t)*m->ends[i].numEntries) {
			PrintError(FnName, "numOffsets", "Could not write to file", Exit, WriteFileError);
		}
		for(j=0;j<m->ends[i].numEntries;j++) {
			if(gzwrite64(fp, m->ends[i].offsets[j], sizeof(int32_t)*m->ends[i].numOffsets[j]) != sizeof(int32_t)*m->ends[i].numOffsets[j]) {
				PrintError(FnName, "offsets[j]", "Could not write to file", Exit, WriteFileError);
			}
		}
	}
}
/* TODO */
void RGMatchesPrintText(FILE *fp,
		RGMatches *m)
{
	char *FnName = "RGMatchesPrintText";
	int32_t i;
	assert(fp!=NULL);

	/* Print the matches to the output file */
	/* Print read name length, read name, and num ends*/
	if(0 > fprintf(fp, "@%s %d\n",
				m->readName,
				m->numEnds)) {
		PrintError(FnName, NULL, "Could not write m->readNameLength, m->readName, and m->numEnds", Exit, WriteFileError);
	}

	/* Print each end */
	for(i=0;i<m->numEnds;i++) {
		RGMatchPrintText(fp,
				&m->ends[i]);
	}
}

/* TODO */
void RGMatchesPrintFastq(FILE *fp,
		RGMatches *m)
{
	/*
	   char *FnName = "RGMatchesPrintFastq";
	   */
	int32_t i;
	assert(fp!=NULL);

	/* Print the matches to the output file */
	/* Print read name length, read name, and num ends*/
	for(i=0;i<m->numEnds;i++) {
		RGMatchPrintFastq(fp,
				m->readName,
				&m->ends[i]);
	}
}

/* TODO */
void RGMatchesRemoveDuplicates(RGMatches *m,
		int32_t maxNumMatches)
{
	int32_t i;
	for(i=0;i<m->numEnds;i++) {
		RGMatchRemoveDuplicates(&m->ends[i], maxNumMatches);
		assert(m->ends[i].numEntries <= maxNumMatches);
	}
}

/* TODO */
/* Merges matches from the same read */
int32_t RGMatchesMergeFilesAndOutput(gzFile *tempFPs,
		int32_t numFiles,
		gzFile outputFP,
		int32_t maxNumMatches,
		int32_t queueLength)
{
	char *FnName="RGMatchesMergeFilesAndOutput";
	int32_t i, j, k, minRead;
	int32_t counter;
	int32_t numMatches=0;
	int32_t numFinished = 0;
	RGMatches *matchQueue=NULL;
	int32_t matchQueueLength = (queueLength / numFiles) + 1;
	
	if(matchQueueLength < 1) {
		matchQueueLength = 1;
	}

	// Allocate
	matchQueue = malloc(sizeof(RGMatches)*matchQueueLength*numFiles);
	if(NULL == matchQueue) {
		PrintError(FnName, "matchQueue", "Could not allocate memory", Exit, MallocMemory);
	}

	// Initialize
	for(i=0;i<matchQueueLength*numFiles;i++) {
		RGMatchesInitialize(&matchQueue[i]);
	}

	/* Read in each sequence/match one at a time */
	counter = 0;
	if(VERBOSE >=0) {
		fputs("\r[0]", stderr);
	}
	while(0 == numFinished) {
		if(VERBOSE >= 0) {
			fprintf(stderr, "\r[%d]", counter);
		}

		// Read matchQueueLength matches for each file
		minRead = -1;
		for(i=j=k=0;i<numFiles;i++) {
			for(j=0;j<matchQueueLength;j++,k++) {
				assert(k < matchQueueLength*numFiles);
				if(RGMatchesRead(tempFPs[i], &matchQueue[k])==EOF) {
					if(minRead < 0) {
						minRead = j;
					}
					else if(j != minRead) {
						PrintError(FnName, "j != minRead", "Did not read in the correct # of entries", Exit, OutOfRange);
					}
					numFinished++;
					break;
				}
			}
		}
		if(minRead < 0) {
			minRead = matchQueueLength;
		}
		// We must finish all at the same time
		assert(numFinished == 0 || numFinished == numFiles);


		// Merge
		for(i=j=k=0;i<minRead;i++) {
			k = minRead + i; // store in first minRead matches 
			// Append
			for(j=1;j<numFiles;j++,k+=minRead) {
				if(0 != strcmp(matchQueue[i].readName, matchQueue[k].readName)) {
					PrintError(FnName, NULL, "Read names do not match", Exit, OutOfRange);
				}
				RGMatchesAppend(&matchQueue[i], &matchQueue[k]); 
				// Free
				RGMatchesFree(&matchQueue[k]);
			}
			// Remove duplicates
			RGMatchesRemoveDuplicates(&matchQueue[i], maxNumMatches);
			// Count matches
			for(j=0;j<matchQueue[i].numEnds;j++) {
				if(0 < matchQueue[i].ends[j].numEntries) {
					numMatches++;
					break;
				}
			}
		}

		// Print
		for(i=0;i<minRead;i++) {
			RGMatchesPrint(outputFP, &matchQueue[i]);
		}

		// Free rest in the queue
		for(i=0;i<minRead;i++) {
			RGMatchesFree(&matchQueue[i]);
		}

		counter += minRead;

	}

	// Free
	free(matchQueue);

	if(VERBOSE >=0) {
		fprintf(stderr, "\r[%d]... completed.\n", counter);
	}

	return numMatches;
}

/* TODO */
void RGMatchesAppend(RGMatches *dest, RGMatches *src)
{
	char *FnName = "RGMatchesAppend";
	int32_t i;
	/* Check that we are not appending to ourselves */
	assert(src != dest);
	assert(NULL != src);
	assert(NULL != dest);

	/* Check to see if we need to add in the read name */
	if(dest->readNameLength <= 0) {
		assert(dest->readName == NULL);
		dest->readNameLength = src->readNameLength;

		/* Allocate memory for the read name */
		dest->readName = malloc(sizeof(char)*(dest->readNameLength+1));
		if(NULL==dest->readName) {
			PrintError(FnName, "dest->readName", "Could not allocate memory", Exit, MallocMemory);
		}
		strcpy(dest->readName, src->readName);

		assert(dest->numEnds <= src->numEnds);
		if(dest->numEnds < src->numEnds) {
			dest->ends = realloc(dest->ends, sizeof(RGMatch)*src->numEnds);
			if(NULL==dest->ends) {
				PrintError(FnName, "dest->ends", "Could not reallocate memory", Exit, ReallocMemory);
			}
			for(i=dest->numEnds;i<src->numEnds;i++) {
				RGMatchInitialize(&dest->ends[i]);
			}
			dest->numEnds = src->numEnds;
		}
	}
	assert(src->numEnds == dest->numEnds);

	/* Append the matches */
	assert(src->numEnds == dest->numEnds);
	for(i=0;i<dest->numEnds;i++) {
		RGMatchAppend(&dest->ends[i], &src->ends[i]);
	}
}

/* TODO */
void RGMatchesReallocate(RGMatches *m,
		int32_t numEnds)
{
	char *FnName="RGMatchesReallocate";
	int32_t i;

	if(numEnds < m->numEnds) {
		for(i=numEnds;i<m->numEnds;i++) {
			RGMatchFree(&m->ends[i]);
		}
	}

	m->ends = realloc(m->ends, sizeof(RGMatch)*numEnds);
	if(NULL == m->ends) {
		PrintError(FnName, "m->ends", "Could not allocate memory", Exit, MallocMemory);
	}

	for(i=m->numEnds;i<numEnds;i++) {
		RGMatchInitialize(&m->ends[i]);
	}
	m->numEnds = numEnds;
}

/* TODO */
void RGMatchesFree(RGMatches *m) 
{
	int32_t i;
	free(m->readName);
	for(i=0;i<m->numEnds;i++) {
		RGMatchFree(&m->ends[i]);
	}
	free(m->ends);
	RGMatchesInitialize(m);
}

/* TODO */
void RGMatchesInitialize(RGMatches *m)
{
	m->readNameLength = 0;
	m->readName = NULL;
	m->numEnds = 0;
	m->ends=NULL;
}

/* TODO */
void RGMatchesMirrorPairedEnd(RGMatches *m,
		RGBinary *rg,
		int32_t pairedEndLength,
		int32_t mirroringType,
		int32_t forceMirroring)
{
	int i, tempNumEntries;
	int numEntriesOne, numEntriesTwo;

	/* For paired end only */
	if(2 == m->numEnds) { 
		numEntriesOne = m->ends[0].numEntries;
		numEntriesTwo = m->ends[1].numEntries;

		/* Copy matches from first to second */
		if(forceMirroring == 1 || 
				(m->ends[0].numEntries > 0 && m->ends[1].numEntries <= 0)) {
			/* Copy forward */
			if(MirrorBoth == mirroringType ||
					MirrorForward == mirroringType) {
				RGMatchReallocate(&m->ends[1], numEntriesOne + numEntriesTwo);
				for(i=0;i<numEntriesOne;i++) {
					m->ends[1].contigs[i+numEntriesTwo] = m->ends[0].contigs[i];
					m->ends[1].strands[i+numEntriesTwo] = m->ends[0].strands[i];
					/* Adjust position */
					m->ends[1].positions[i+numEntriesTwo] = m->ends[0].positions[i] + m->ends[0].readLength + pairedEndLength;
				}
			}
			/* Copy reverse */
			if(MirrorBoth == mirroringType || 
					MirrorReverse == mirroringType) {
				tempNumEntries = m->ends[1].numEntries;
				RGMatchReallocate(&m->ends[1], numEntriesOne + tempNumEntries);
				for(i=0;i<numEntriesOne;i++) {
					m->ends[1].contigs[i+tempNumEntries] = m->ends[0].contigs[i];
					m->ends[1].strands[i+tempNumEntries] = m->ends[0].strands[i];
					/* Adjust position */
					m->ends[1].positions[i+tempNumEntries] = m->ends[0].positions[i] - m->ends[0].readLength - pairedEndLength;
				}
			}
		}
		/* Copy matches from second to first */
		if(forceMirroring == 1 || 
				(m->ends[0].numEntries <= 0 && m->ends[1].numEntries > 0)) {
			/* Copy forward */
			if(MirrorBoth == mirroringType ||
					MirrorForward == mirroringType) {
				RGMatchReallocate(&m->ends[0], numEntriesOne + numEntriesTwo);
				for(i=0;i<numEntriesTwo;i++) {
					m->ends[0].contigs[i+numEntriesOne] = m->ends[1].contigs[i];
					m->ends[0].strands[i+numEntriesOne] = m->ends[1].strands[i];
					/* Adjust position */
					m->ends[0].positions[i+numEntriesOne] = m->ends[1].positions[i] - m->ends[0].readLength - pairedEndLength;
				}
			}
			/* Copy reverse */
			if(MirrorBoth == mirroringType || 
					MirrorReverse == mirroringType) {
				tempNumEntries = m->ends[0].numEntries;
				RGMatchReallocate(&m->ends[0], tempNumEntries + numEntriesTwo);
				for(i=0;i<numEntriesTwo;i++) {
					m->ends[0].contigs[i+tempNumEntries] = m->ends[1].contigs[i];
					m->ends[0].strands[i+tempNumEntries] = m->ends[1].strands[i];
					/* Adjust position */
					m->ends[0].positions[i+tempNumEntries] = m->ends[1].positions[i] + m->ends[0].readLength + pairedEndLength;
				}
			}
		}

		RGMatchesRemoveDuplicates(m, INT_MAX);
		/* Adjust positions in case they trail off the end of the contigs */
		for(i=0;i<m->ends[0].numEntries;i++) {
			m->ends[0].positions[i] = GETMAX(1, GETMIN(m->ends[0].positions[i], rg->contigs[m->ends[0].contigs[i]-1].sequenceLength));
		}
		for(i=0;i<m->ends[1].numEntries;i++) {
			m->ends[1].positions[i] = GETMAX(1, GETMIN(m->ends[1].positions[i], rg->contigs[m->ends[1].contigs[i]-1].sequenceLength));
		}
	}
}

/* TODO */
void RGMatchesCheck(RGMatches *m, RGBinary *rg) 
{
	char *FnName="RGMatchesCheck";
	int32_t i;
	/* Basic asserts */
	if(m->numEnds < 0) { 
		PrintError(FnName, NULL, "m->numEnds < 0", Exit, OutOfRange);
	}
	/* Check that the read name length is the same as the length of the read name */
	if(((int)strlen(m->readName)) != m->readNameLength) {
		PrintError(FnName, NULL, "strlen(m->readName)) != m->readNameLength", Exit, OutOfRange);
	}
	for(i=0;i<m->numEnds;i++) {
		if(0 == RGMatchCheck(&m->ends[i], rg)) {
			fprintf(stderr, "End %d failed!\n", i+1);
			RGMatchesPrintText(stderr, m);
			PrintError(FnName, "RGMatchCheck", "The match failed the consistency check", Exit, OutOfRange);
		}
	}
}

/* TODO */
void RGMatchesFilterOutOfRange(RGMatches *m,
		int32_t maxNumMatches)
{
	int32_t i;
	for(i=0;i<m->numEnds;i++) {
		RGMatchFilterOutOfRange(&m->ends[i],
				maxNumMatches);
	}
}

int32_t RGMatchesMergeIndexBins(gzFile *tempOutputIndexBinFPs,
		int32_t numBins,
		gzFile tempOutputIndexFP,
		RGIndex *index,
		int32_t maxKeyMatches,
		int32_t maxNumMatches) 
{
	char *FnName="RGMatchesMergeIndexBins";
	int32_t i, j, k, l, m, numMatches=0;
	int32_t counter;
	RGMatches matches;
	RGMatches tempMatches;
	int32_t numFinished = 0;
	int32_t numKeyMatches[SEQUENCE_LENGTH];

	/* Initialize matches */
	RGMatchesInitialize(&matches);
	RGMatchesInitialize(&tempMatches);

	/* Read in each sequence/match one at a time */
	counter = 0;
	if(VERBOSE >=0) {
		fprintf(stderr, "\r[%d]", 0);
	}
	while(numFinished == 0) {
		if(VERBOSE >=0 && counter%RGMATCH_MERGE_ROTATE_NUM == 0) {
			fprintf(stderr, "\r[%d]", counter);
		}
		counter++;

		/* Read matches for one read from each file */ 
		for(i=0;i<numBins;i++) {
			if(RGMatchesReadWithOffsets(tempOutputIndexBinFPs[i],
						&tempMatches)==EOF) {
				numFinished++;
			}
			else {
				if(matches.readName != NULL &&
						strcmp(matches.readName, tempMatches.readName)!=0) {
					PrintError(FnName, NULL, "Read names do not match", Exit, OutOfRange);
				}

				/* Append temp matches to matches */
				RGMatchesAppend(&matches, &tempMatches);
			}

			/* Free temp matches */
			RGMatchesFree(&tempMatches);
		}
		/* We must finish all at the same time */
		assert(numFinished == 0 || numFinished == numBins);

		if(numFinished == 0) {
			/* Finalize each end */
			for(i=0;i<matches.numEnds;i++) {
				RGMatchRemoveDuplicates(&matches.ends[i], maxNumMatches);

				for(j=0;j<matches.ends[i].readLength;j++) { // initialize
					numKeyMatches[j]=0;
				}
				for(j=0;j<matches.ends[i].numEntries;j++) { // count # of matches per offset
					for(k=0;k<matches.ends[i].numOffsets[j];k++) {
						numKeyMatches[matches.ends[i].offsets[j][k]]++;
					}
				}
				for(j=k=0;j<matches.ends[i].numEntries;j++) {
					// Find any offset that is below the bound
					for(l=0;l<matches.ends[i].numOffsets[j];l++) {
						if(numKeyMatches[matches.ends[i].offsets[j][l]] <= maxKeyMatches) break;
					}
					if(l<matches.ends[i].numOffsets[j]) { // Found one
						if(k != j) {
							matches.ends[i].contigs[k] = matches.ends[i].contigs[j];
							matches.ends[i].positions[k] = matches.ends[i].positions[j];
							matches.ends[i].strands[k] = matches.ends[i].strands[j];
						}
						// Zero out mask
						for(l=0;l<GETMASKNUMBYTESFROMLENGTH(matches.ends[i].readLength);l++) {
							matches.ends[i].masks[k][l] = 0;
						}
						// Copy over masks based on kept offsets
						for(l=0;l<matches.ends[i].numOffsets[j];l++) { // for each offset
							if(numKeyMatches[matches.ends[i].offsets[j][l]] <= maxKeyMatches) {
								// Add ot the mask
								for(m=0;m<index->width;m++) {
									if(FORWARD == matches.ends[i].strands[j]) {
										if(1 == index->mask[m]) {
											int32_t offset = matches.ends[i].offsets[j][l] + m; 
											// Color space already adjusted
											//if(ColorSpace == index->space) offset++;
											RGMatchUpdateMask(matches.ends[i].masks[k], offset); 
										}
									}
									else {
										if(1 == index->mask[index->width - m - 1]) {
											int32_t offset = matches.ends[i].offsets[j][l] + m; 
											// Color space already adjusted
											//if(ColorSpace == index->space) offset--;
											RGMatchUpdateMask(matches.ends[i].masks[k], offset); 
										}
									}
								}
							}
						}
						k++;
					}
				}
				// remove offsets
				for(j=0;j<matches.ends[i].numEntries;j++) {
					free(matches.ends[i].offsets[j]);
					matches.ends[i].offsets[j]=NULL;
				}
				free(matches.ends[i].numOffsets);
				matches.ends[i].numOffsets=NULL;
				free(matches.ends[i].offsets);
				matches.ends[i].offsets=NULL;
				// reallocate
				RGMatchReallocate(&matches.ends[i], k); // important that k is preserved up to this point
				// check if there were too many matches by removing duplicates
				// this will also union the masks
				RGMatchRemoveDuplicates(&matches.ends[i], maxNumMatches);
			}
			for(i=0;i<matches.numEnds;i++) {
				if(0 < matches.ends[i].numEntries) {
					numMatches++;
					break;
				}
			}
			RGMatchesPrint(tempOutputIndexFP,
					&matches);
		}
		/* Free memory */
		RGMatchesFree(&matches);
	}

	if(VERBOSE >=0) {
		fprintf(stderr, "\r[%d]... completed.\n", counter-1);
	}

	return numMatches;
}
