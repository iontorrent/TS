#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <zlib.h>
#include "BError.h"
#include "BLib.h"
#include "RGMatch.h"
#include "RGMatches.h"
#include "aflib.h"
#include "kseq.h"
#include "MatchesReadInputFiles.h"

KSEQ_INIT(AFILE*, AFILE_afread2)

	/* TODO */
int WriteRead(FILE *fp, RGMatches *m)
{
	int32_t i;

	/* Print read */
	for(i=0;i<m->numEnds;i++) {
		if(fprintf(fp, "@%s\n%s\n+\n%s\n", 
					m->readName,
					m->ends[i].read,
					m->ends[i].qual) < 0) { 
			return EOF;
		}
	}
	return 1;
}

/* TODO */
int WriteReadAFILE(AFILE *afp_output, RGMatches *m)
{
	int32_t i, j;
	char at = '@';
	char plus = '+';
	char new_line = '\n';

	/* Print read */
	for(i=0;i<m->numEnds;i++) {
		// Name
		AFILE_afwrite(&at, sizeof(char), 1, afp_output);
		for(j=0;j<m->readNameLength;j++) {
			AFILE_afwrite(&m->readName[j], sizeof(char), 1, afp_output);
		}
		AFILE_afwrite(&new_line, sizeof(char), 1, afp_output);

		// Sequence
		for(j=0;j<m->ends[i].readLength;j++) {
			AFILE_afwrite(&m->ends[i].read[j], sizeof(char), 1, afp_output);
		}
		AFILE_afwrite(&new_line, sizeof(char), 1, afp_output);

		// Comment
		AFILE_afwrite(&plus, sizeof(char), 1, afp_output);
		AFILE_afwrite(&new_line, sizeof(char), 1, afp_output);

		// Quality
		for(j=0;j<m->ends[i].qualLength;j++) {
			AFILE_afwrite(&m->ends[i].qual[j], sizeof(char), 1, afp_output);
		}
		AFILE_afwrite(&new_line, sizeof(char), 1, afp_output);
	}
	return 1;
}

static void kseq_AppendToRGMatches(RGMatches *m,
		kseq_t *seq)
{
	char *FnName="kseq_AppendToRGMatches";
	if(0 == m->numEnds) {
		/* Allocate memory */
		m->readNameLength = seq->name.l;
		m->readName = malloc(sizeof(char)*(m->readNameLength+1));
		if(NULL == m->readName) {
			PrintError(FnName, "m->readName", "Could not allocate memory", Exit, MallocMemory);
		}
		strcpy(m->readName, seq->name.s);
	}

	/* Reallocate */
	m->numEnds++;
	m->ends = realloc(m->ends, sizeof(RGMatch)*m->numEnds);
	if(NULL == m->ends) {
		PrintError(FnName, "m->ends", "Could not reallocate memory", Exit, ReallocMemory);
	}
	RGMatchInitialize(&m->ends[m->numEnds-1]);
	m->ends[m->numEnds-1].readLength = seq->seq.l;
	m->ends[m->numEnds-1].qualLength = seq->qual.l;
	m->ends[m->numEnds-1].read = malloc(sizeof(char)*(m->ends[m->numEnds-1].readLength+1));
	if(NULL == m->ends[m->numEnds-1].read) {
		PrintError(FnName, "m->ends[m->numEnds-1].read", "Could not allocate memory", Exit, MallocMemory);
	}
	strcpy(m->ends[m->numEnds-1].read, seq->seq.s);
	m->ends[m->numEnds-1].qual = malloc(sizeof(char)*(m->ends[m->numEnds-1].qualLength+1));
	if(NULL == m->ends[m->numEnds-1].qual) {
		PrintError(FnName, "m->ends[m->numEnds-1].qual", "Could not allocate memory", Exit, MallocMemory);
	}
	strcpy(m->ends[m->numEnds-1].qual, seq->qual.s);
}

/* TODO */
void WriteReadsToTempFile(AFILE *seqFP,
		gzFile *tmpSeqFP, 
		char **tmpSeqFileName,
		int startReadNum, 
		int endReadNum, 
		char *tmpDir,
		int *numWritten,
		int32_t space)
{
	int curReadNum = 1;
	RGMatches m;
	kseq_t *seq=NULL;

	// Open temporary file
	(*tmpSeqFP) = OpenTmpGZFile(tmpDir, tmpSeqFileName);

	seq = kseq_init(seqFP);
	RGMatchesInitialize(&m);
	(*numWritten)=0;
	curReadNum=1;
	while(0 <= kseq_read(seq, space)) {
		// compare with previous
		if(0 == m.numEnds || 0 == strcmp(m.readName, seq->name.s)) {
			// append
			kseq_AppendToRGMatches(&m, seq);
		}
		else {
			// print
			if(startReadNum <= curReadNum && curReadNum <= endReadNum) {
				RGMatchesPrint((*tmpSeqFP), &m);
				(*numWritten)++;
			}
			curReadNum++;

			// make new
			RGMatchesFree(&m);
			RGMatchesInitialize(&m);
			kseq_AppendToRGMatches(&m, seq);
		}
	}
	if(0 < m.numEnds) {
		if(startReadNum <= curReadNum && curReadNum <= endReadNum) {
			RGMatchesPrint((*tmpSeqFP), &m);
			(*numWritten)++;
		}
		curReadNum++;

		// make new
		RGMatchesFree(&m);
	}

	/* reset pointer to temp files to the beginning of the file */
	ReopenTmpGZFile(tmpSeqFP, tmpSeqFileName);

	// destroy
	kseq_destroy(seq);
}

/* TODO */
/* Go through the temporary output file and output those reads that have 
 * at least one match to the final output file.  For those reads that have
 * zero matches, output them to the temporary read file *
 * */
int ReadTempReadsAndOutput(gzFile tempOutputFP,
		char *tempOutputFileName,
		gzFile outputFP,
		AFILE *tempRGMatchesFP)
{
	char *FnName = "ReadTempReadsAndOutput";
	RGMatches m;
	int32_t i;
	int numReads = 0;
	int numOutputted=0;
	int hasEntries=0;

	/* Initialize */
	RGMatchesInitialize(&m);

	/* Go to the beginning of the temporary output file */
	ReopenTmpGZFile(&tempOutputFP,
			&tempOutputFileName);

	while(RGMatchesRead(tempOutputFP, 
				&m)!=EOF) {
		/* Output if any end has more than one entry */
		for(i=hasEntries=0;0==hasEntries && i<m.numEnds;i++) {
			if(0 < m.ends[i].numEntries) {
				hasEntries=1;
			}
		}
		/* Output to final output file */
		if(1 == hasEntries) {
			RGMatchesPrint(outputFP,
					&m);
			numOutputted++;
		}
		else {
			/* Put back in the read file */
			if(EOF == WriteReadAFILE(tempRGMatchesFP, &m)) {
				PrintError(FnName, NULL, "Could not write read.", Exit, WriteFileError);
			}
			numReads++;
		}

		/* Free memory */
		RGMatchesFree(&m);
	}
	return numReads;
}

void ReadRGIndex(char *rgIndexFileName, RGIndex *index, int space)
{

	/* Read from file */
	RGIndexRead(index, rgIndexFileName);

	if(index->space != space) {
		PrintError("space", rgIndexFileName, "The index has a different space parity than specified", Exit, OutOfRange);
	}
}

/* TODO */
int GetIndexFileNames(char *fastaFileName, 
		int32_t space, 
		char *indexes, 
		char ***fileNames,
		int32_t ***indexIDs)
{
	char *FnName="GetIndexFileNames";
	char prefix[MAX_FILENAME_LENGTH]="\0";
	int32_t i, j, numFiles=0;
	int32_t *indexNumbers=NULL;
	int32_t numIndexNumbers=0;
	int32_t maxBin;

	assert(NULL != fastaFileName);

	/* Build the index file names */
	strcpy(prefix, fastaFileName);
	strcat(prefix, ".");
	strcat(prefix, SPACENAME(space));

	if(NULL != indexes) { // Tokenize
		indexNumbers = GetNumbersFromString(indexes, &numIndexNumbers);
		for(i=0;i<numIndexNumbers;i++) {
			if(indexNumbers[i] <= 0) {
				PrintError(FnName, indexes, "Could not understand index number", Exit, OutOfRange);
			}
			maxBin = GetBIFMaximumBin(prefix, indexNumbers[i]);
			if(0 == maxBin) {
				fprintf(stderr, "Index number: %d\n", indexNumbers[i]);
				PrintError(FnName, NULL, "The index does not exist", Exit, OutOfRange);			
			}			
			(*fileNames) = realloc((*fileNames), sizeof(char*)*(numFiles+maxBin));			
			if(NULL == (*fileNames)) {				
				PrintError(FnName, "fileNames", "Could not reallocate memory", Exit, ReallocMemory);
			}
			/* Insert file names */
			for(j=1;j<=maxBin;j++) {
				(*fileNames)[numFiles] = malloc(sizeof(char)*MAX_FILENAME_LENGTH);
				if(NULL == (*fileNames)[numFiles]) {
					PrintError(FnName, "fileNames[j]", "Could not allocate memory", Exit, MallocMemory);				
				}				
				sprintf((*fileNames)[numFiles], "%s.%d.%d.%s", prefix, indexNumbers[i], 
						j,
						BFAST_INDEX_FILE_EXTENSION);
				if(0 == FileExists((*fileNames)[numFiles])) {
					PrintError(FnName, (*fileNames)[numFiles], "The index does not exist", Exit, OutOfRange);				
				}				
				(*indexIDs) = realloc((*indexIDs), sizeof(int32_t*)*(1+numFiles));
				if(NULL == (*indexIDs)) {
					PrintError(FnName, "(*indexIDs)", "Could not reallocate memory", Exit, ReallocMemory);
				}
				(*indexIDs)[numFiles] = malloc(sizeof(int32_t)*2);
				if(NULL == (*indexIDs)[numFiles]) {
					PrintError(FnName, "(*indexIDs)[numFiles]", "Could not allocate memory", Exit, MallocMemory);
				}
				(*indexIDs)[numFiles][0] = indexNumbers[i];
				(*indexIDs)[numFiles][1] = j;
				numFiles++;			
			}		
		}
		free(indexNumbers);
	}
	else {
		i=1;
		numIndexNumbers=0;
		while(1) { // ^^
			maxBin = GetBIFMaximumBin(prefix, i);
			if(0 == maxBin) {
				break;
			}
			(*fileNames) = realloc((*fileNames), sizeof(char*)*(numFiles+maxBin));
			if(NULL == (*fileNames)) {
				PrintError(FnName, "fileNames", "Could not reallocate memory", Exit, ReallocMemory);			
			}			
			/* Insert file names */			
			for(j=1;j<=maxBin;j++) {				
				(*fileNames)[numFiles] = malloc(sizeof(char)*MAX_FILENAME_LENGTH);
				if(NULL == (*fileNames)[numFiles]) {
					PrintError(FnName, "fileNames[j]", "Could not allocate memory", Exit, MallocMemory);				
				}				
				sprintf((*fileNames)[numFiles], "%s.%d.%d.%s", prefix, i, 
						j,
						BFAST_INDEX_FILE_EXTENSION);
				if(0 == FileExists((*fileNames)[numFiles])) {
					PrintError(FnName, (*fileNames)[numFiles], "Missing Bin: The index does not exist", Exit, OutOfRange);				
				}				

				(*indexIDs) = realloc((*indexIDs), sizeof(int32_t*)*(1+numFiles));
				if(NULL == (*indexIDs)) {
					PrintError(FnName, "(*indexIDs)", "Could not reallocate memory", Exit, ReallocMemory);
				}
				(*indexIDs)[numFiles] = malloc(sizeof(int32_t)*2);
				if(NULL == (*indexIDs)[numFiles]) {
					PrintError(FnName, "(*indexIDs)[numFiles]", "Could not allocate memory", Exit, MallocMemory);
				}
				(*indexIDs)[numFiles][0] = i;
				(*indexIDs)[numFiles][1] = j;

				numFiles++;			
			}			

			i++;
			numIndexNumbers++;
		}
		free(indexNumbers);
	}

	if(0 == numIndexNumbers) {
		PrintError(FnName, prefix, "Could not find any indexes with the given prefix", Exit, OutOfRange);	
	}	
	//for(i=0;i<numFiles;i++) fprintf(stderr, "f[%d]=[%s]\n", i, (*fileNames)[i]);
	if(VERBOSE>=0) {
		if(1 == numIndexNumbers && 1 == numFiles) fprintf(stderr, "Found %d index (%d file).\n", numIndexNumbers, numFiles);
		else if(1 == numIndexNumbers && 1 != numFiles) fprintf(stderr, "Found %d index (%d total files).\n", numIndexNumbers, numFiles);
		else fprintf(stderr, "Found %d index (%d total files).\n", numIndexNumbers, numFiles);
	}

	return numFiles;
}

/* TODO */
int32_t ReadOffsets(char *offsetsInput, int32_t **offsets) 
{
	char *FnName="ReadOffsets";
	int numOffsets=0;
	int32_t i;

	(*offsets) = GetNumbersFromString(offsetsInput, &numOffsets);

	if(NULL == (*offsets)) {
		PrintError(FnName, offsetsInput, "Could not parse the offsets", Exit, OutOfRange);
	}
	else if(numOffsets <= 0) {
		PrintError(FnName, NULL, "Could not find any offsets", Exit, OutOfRange);
	}

	// Check input
	for(i=0;i<numOffsets;i++) {
		if((*offsets)[i] < 0) {			
			PrintError(FnName, offsetsInput, "Offset was negative", Exit, OutOfRange);		
		}
		else if(0 < i && (*offsets)[i] <= (*offsets)[i-1]) {
			PrintError(FnName, offsetsInput, "Offset were not in increasing order", Exit, OutOfRange);		
		}
	}

	if(VERBOSE>=0) {
		fprintf(stderr, "Read %d offsets.\n", numOffsets);
	}

	return numOffsets;
}

int32_t GetReads(gzFile inRGMatchesFP, RGMatches *m, int32_t maxToRead, int32_t space) 
{
	int32_t numRead = 0;

	while(numRead < maxToRead) {
		RGMatchesInitialize(&(m[numRead]));
		if(EOF == RGMatchesRead(inRGMatchesFP, &(m[numRead]))) return numRead;
		numRead++;
	}
	return numRead;
}
