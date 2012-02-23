#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <errno.h>
#include <limits.h>
#include <zlib.h>
#include "BLibDefinitions.h"
#include "BError.h"
#include "BLib.h"
#include "RGBinary.h"
#include "RGIndex.h"
#include "RGReads.h"
#include "RGMatch.h"
#include "RGMatches.h"
#include "MatchesReadInputFiles.h"
#include "aflib.h"
#include "RunMatch.h"

static pthread_mutex_t matchQueueMutex = PTHREAD_MUTEX_INITIALIZER;

/* TODO */
void RunMatch(
		char *fastaFileName,
		char *mainIndexes,
		char *secondaryIndexes,
		char *readFileName, 
		char *offsetsInput,
		int loadAllIndexes,
		int compression,
		int space,
		int startReadNum,
		int endReadNum,
		int keySize,
		int maxKeyMatches,
		int maxNumMatches,
		int whichStrand,
		int numThreads,
		int queueLength,
		char *tmpDir,
		int timing,
		FILE *fpOut
		)
{
	char *FnName="RunMatch";
	int numMainIndexes=0;
	char **mainIndexFileNames=NULL;
	int32_t **mainIndexIDs=NULL;

	int numSecondaryIndexes=0;
	char **secondaryIndexFileNames=NULL;
	int32_t **secondaryIndexIDs=NULL;

	int32_t *offsets=NULL;
	int32_t numOffsets=0;

	AFILE *seqFP=NULL;
	gzFile tmpSeqFP=NULL; // for secondary index search
	char *tmpSeqFileName=NULL; // for secondary index search
	gzFile outputFP;
	int i;

	int numMatches;
	int numReads;

	time_t startTime, endTime;
	int seconds, minutes, hours;
	int totalReadRGTime = 0;
	int totalDataStructureTime = 0; /* This will only give the to load and deleted the indexes (excludes searching and other things) */
	int totalSearchTime = 0; /* This will only give the time searching (excludes load times and other things) */
	int totalOutputTime = 0; /* This wll only give the total time to merge and output */

	RGMatches tempRGMatches;
	RGBinary rg;
	int startChr, startPos, endChr, endPos;

	/* Read in the main RGIndex File Names */
	if(0<=VERBOSE) {
		fprintf(stderr, "Searching for main indexes...\n");
	}
	numMainIndexes=GetIndexFileNames(fastaFileName, space, mainIndexes, &mainIndexFileNames, &mainIndexIDs);
	if(numMainIndexes<=0) {
		PrintError(FnName, "numMainIndexes", "Read zero indexes", Exit, OutOfRange);
	}

	/* Read in the secondary RGIndex File Names */
	if(NULL != secondaryIndexes) {
		if(0<=VERBOSE) {
			fprintf(stderr, "Searching for secondary indexes...\n");
		}
		numSecondaryIndexes=GetIndexFileNames(fastaFileName, space, secondaryIndexes, &secondaryIndexFileNames, &secondaryIndexIDs);
	}
	else {
		if(0<=VERBOSE) {
			fprintf(stderr, "Not using secondary indexes.\n");
		}
		numSecondaryIndexes=0;
	}

	/* Check the indexes.
	 * 1. We want the two sets of files to have the same range.
	 * */
	if(numSecondaryIndexes > 0) {
		CheckRGIndexes(mainIndexFileNames, 
				numMainIndexes,
				secondaryIndexFileNames,
				numSecondaryIndexes,
				&startChr,
				&startPos,
				&endChr,
				&endPos,
				space);
	}

	/* Read in the reference genome */
	startTime = time(NULL);
	RGBinaryReadBinary(&rg,
			space,
			fastaFileName);
	assert(rg.space == space);
	endTime = time(NULL);
	totalReadRGTime = endTime - startTime;

	/* Read in the offsets */
	numOffsets = (NULL == offsetsInput) ? 0 : ReadOffsets(offsetsInput, &offsets);

	/* open read file */
	if(NULL == readFileName) {
		if(0 == (seqFP = AFILE_afdopen(fileno(stdin), "r", compression))) {
			PrintError(FnName, "stdin", "Could not open stdin for reading", Exit, OpenFileError);
		}
	}
	else {
		if(0 == (seqFP = AFILE_afopen(readFileName, "r", compression))) {
			PrintError(FnName, readFileName, "Could not open readFileName for reading", Exit, OpenFileError);
		}
	}
	/* Read the reads to the thread temp files */
	if(VERBOSE >= 0) {
		fprintf(stderr, "Reading %s into a temp file.\n",
				(readFileName == NULL) ? "stdin" : readFileName);
	}
	/* This will close the reads file */
	WriteReadsToTempFile(seqFP,
			&tmpSeqFP,
			&tmpSeqFileName,
			startReadNum,
			endReadNum,
			tmpDir,
			&numReads,
			space);
	/* Close the read file */
	AFILE_afclose(seqFP);
	if(VERBOSE >= 0) {
		fprintf(stderr, "Will process %d reads.\n",
				numReads);
	}

	/* Open output file */
	if(0 == (outputFP=gzdopen(fileno(fpOut), "wb"))) {
		PrintError(FnName, "stdout", "Could not open stdout for writing", Exit, OpenFileError);
	}

	/* Do step 1: search the main indexes for all reads */
	numMatches=FindMatchesInIndexSet(mainIndexFileNames,
			mainIndexIDs,
			numMainIndexes,
			&rg,
			offsets,
			numOffsets,
			loadAllIndexes,
			space,
			keySize,
			maxKeyMatches,
			maxNumMatches,
			whichStrand,
			numThreads,
			queueLength,
			&tmpSeqFP,
			&tmpSeqFileName,
			outputFP,
			(0 < numSecondaryIndexes)?CopyForNextSearch:EndSearch,
			MainIndexes,
			tmpDir,
			timing,
			&totalDataStructureTime,
			&totalSearchTime,
			&totalOutputTime
				);

	/* Do secondary index search */

	if(0 < numSecondaryIndexes) { /* Only if there are secondary indexes */
		if(0 < numReads - numMatches) { /* Only if enough reads are left */
			if(VERBOSE >= 0) {
				fprintf(stderr, "%s", BREAK_LINE);
			}

			/* Do step 2: search the indexes for all reads */
			numMatches+=FindMatchesInIndexSet(secondaryIndexFileNames,
					secondaryIndexIDs,
					numSecondaryIndexes,
					&rg,
					offsets,
					numOffsets,
					loadAllIndexes,
					space,
					keySize,
					maxKeyMatches,
					maxNumMatches,
					whichStrand,
					numThreads,
					queueLength,
					&tmpSeqFP,
					&tmpSeqFileName,
					outputFP,
					EndSearch,
					SecondaryIndexes,
					tmpDir,
					timing,
					&totalDataStructureTime,
					&totalSearchTime,
					&totalOutputTime
						);
		}
		else {
			// Output the reads not aligned and close the temporary read files
			ReopenTmpGZFile(&tmpSeqFP, &tmpSeqFileName);
			RGMatchesInitialize(&tempRGMatches);
			while(EOF!=RGMatchesRead(tmpSeqFP, &tempRGMatches)) {
				RGMatchesPrint(outputFP,
						&tempRGMatches);
				RGMatchesFree(&tempRGMatches);
			}
			CloseTmpGZFile(&tmpSeqFP, &tmpSeqFileName, 1);
			gzclose(outputFP);
		}
	}

	if(VERBOSE>=0) {
		fprintf(stderr, "%s", BREAK_LINE);
		fprintf(stderr, "%s", BREAK_LINE);
		fprintf(stderr, "In total, found matches for %d out of %d reads.\n", 
				numMatches,
				numReads);
		fprintf(stderr, "%s", BREAK_LINE);
	}

	/* Free main RGIndex file names */
	for(i=0;i<numMainIndexes;i++) {
		free(mainIndexFileNames[i]);
		free(mainIndexIDs[i]);
	}
	free(mainIndexFileNames);
	free(mainIndexIDs);

	/* Free RGIndex file names */
	for(i=0;i<numSecondaryIndexes;i++) {
		free(secondaryIndexFileNames[i]);
		free(secondaryIndexIDs[i]);
	}
	free(secondaryIndexFileNames);
	free(secondaryIndexIDs);

	/* Free reference genome */
	RGBinaryDelete(&rg);

	/* Free offsets */
	free(offsets);

	/* Print timing */
	if(timing == 1) {
		/* Read RG time */
		seconds = totalReadRGTime;
		hours = seconds/3600;
		seconds -= hours*3600;
		minutes = seconds/60;
		seconds -= minutes*60;
		fprintf(stderr, "Total time loading the reference genome: %d hour, %d minutes and %d seconds.\n",
				hours,
				minutes,
				seconds);
		/* Data structure time */
		seconds = totalDataStructureTime;
		hours = seconds/3600;
		seconds -= hours*3600;
		minutes = seconds/60;
		seconds -= minutes*60;
		fprintf(stderr, "Total time loading and deleting index%s: %d hour, %d minutes and %d seconds.\n",
				(1 == numMainIndexes + numSecondaryIndexes) ? "" : "es",
				hours,
				minutes,
				seconds);
		/* Search time */
		seconds = totalSearchTime;
		hours = seconds/3600;
		seconds -= hours*3600;
		minutes = seconds/60;
		seconds -= minutes*60;
		fprintf(stderr, "Total time searching index%s: %d hour, %d minutes and %d seconds.\n",
				(1 == numMainIndexes + numSecondaryIndexes) ? "" : "es",
				hours,
				minutes,
				seconds);
		/* Output time */
		seconds = totalOutputTime;
		hours = seconds/3600;
		seconds -= hours*3600;
		minutes = seconds/60;
		seconds -= minutes*60;
		fprintf(stderr, "Total time merging and writing output: %d hour, %d minutes and %d seconds.\n",
				hours,
				minutes,
				seconds);
	}
}

int FindMatchesInIndexSet(char **indexFileNames,
		int32_t **indexIDs,
		int numIndexes,
		RGBinary *rg,
		int32_t *offsets,
		int numOffsets,
		int loadAllIndexes,
		int space,
		int keySize,
		int maxKeyMatches,
		int maxNumMatches,
		int whichStrand,
		int numThreads,
		int queueLength,
		gzFile *tmpSeqFP,
		char **tmpSeqFileName,
		gzFile outputFP,
		int copyForNextSearch,
		int indexesType,
		char *tmpDir,
		int timing,
		int *totalDataStructureTime,
		int *totalSearchTime,
		int *totalOutputTime)
{
	char *FnName = "FindMatchesInIndexSet";
	int i;
	gzFile tempOutputFP;
	char *tempOutputFileName=NULL;
	gzFile *tempOutputIndexFPs=NULL;
	char **tempOutputIndexFileNames=NULL;
	int numWritten=0, numReads=0;
	int numMatches = 0;
	time_t startTime, endTime;
	int seconds, minutes, hours;
	AFILE tempRGMatchesAFP;
	char *tempRGMatchesFileName=NULL;
	int32_t numUniqueIndexes = 1;
	int32_t indexNum, numBins, uniqueIndexCtr, uniqueIndexBinCtr;
	gzFile *tempOutputIndexBinFPs=NULL;
	char **tempOutputIndexBinFileNames=NULL;
	RGIndex tempIndex;

	/* IDEA: for each index, split search into threads generating one output file per thread.
	 * After the threads have finished their searches, merge their output into one output file
	 * specific for each index.  After all indexes have been searched, merge the index specific
	 * output.
	 * */

	for(i=1;i<numIndexes;i++) {
		if(indexIDs[i-1][0] != indexIDs[i][0]) {
			numUniqueIndexes++;
		}
	}

	/* Allocate memory for the index specific file pointers */
	tempOutputIndexFPs = malloc(sizeof(gzFile)*numUniqueIndexes);
	if(NULL == tempOutputIndexFPs) {
		PrintError(FnName, "tempOutputIndexFPs", "Could not allocate memory", Exit, MallocMemory);
	}
	tempOutputIndexFileNames = malloc(sizeof(char*)*numUniqueIndexes);
	if(NULL == tempOutputIndexFileNames) {
		PrintError(FnName, "tempOutputIndexFileNames", "Could not allocate memory", Exit, MallocMemory);
	}
	/* If we are ending the search, output to the final output file.  Otherwise,
	 * output to a temporary file.
	 * */
	if(CopyForNextSearch == copyForNextSearch) {
		/* Open temporary file for the entire index search */
		tempOutputFP=OpenTmpGZFile(tmpDir, &tempOutputFileName);
	}
	else {
		assert(EndSearch == copyForNextSearch);
		/* Write directly to the output file */
		tempOutputFP=outputFP;
	}

	/* If we have only one index or if we are processing all indexes at once, output the temp output file */
	if(1 == numUniqueIndexes || IndexesMemoryAll == loadAllIndexes) {
		tempOutputIndexFPs[0] = tempOutputFP;
	}
	else {
		/* Open tmp files for each index */
		for(i=0;i<numUniqueIndexes;i++) {
			tempOutputIndexFPs[i] = OpenTmpGZFile(tmpDir, &tempOutputIndexFileNames[i]); 
		}
	}

	/* Only if there are more than one index, otherwise this defaults below */
	if(1 < numIndexes && IndexesMemoryAll == loadAllIndexes) {
		if(VERBOSE >= 0) {
			fprintf(stderr, "Searching index files 1-%d...\n", numIndexes);
		}
		// Process all indexes at once
		numMatches = FindMatches(indexFileNames,
				numIndexes,
				rg,
				offsets,
				numOffsets,
				loadAllIndexes,
				space,
				keySize,
				maxKeyMatches,
				maxNumMatches,
				whichStrand,
				numThreads,
				queueLength,
				tmpSeqFP,
				tmpSeqFileName,
				tempOutputFP,
				0,
				tmpDir,
				timing,
				totalDataStructureTime,
				totalSearchTime,
				totalOutputTime
					);
		if(VERBOSE >= 0) {
			fprintf(stderr, "Searching index files 1-%d... complete\n", numIndexes);
		}
	}
	else {
		indexNum=0;
		// for each unique index
		for(uniqueIndexCtr=0;uniqueIndexCtr<numUniqueIndexes;uniqueIndexCtr++) {

			// get the # of bins for this index
			numBins = 1;
			for(i=indexNum+1;i<numIndexes && indexIDs[i-1][0] == indexIDs[i][0];i++) {
				numBins++;
			}

			if(1 == numBins) { // don't bother storing since we have one bin 
				if(VERBOSE >= 0) {
					fprintf(stderr, "%s", BREAK_LINE);
					fprintf(stderr, "Searching index file %d/%d (index #%d, bin #%d)...\n", 
							indexNum+1, numIndexes,
							indexIDs[indexNum][0], indexIDs[indexNum][1]);
				}
				numMatches = FindMatches(&indexFileNames[indexNum],
						1,
						rg,
						offsets,
						numOffsets,
						loadAllIndexes,
						space,
						keySize,
						maxKeyMatches,
						maxNumMatches,
						whichStrand,
						numThreads,
						queueLength,
						tmpSeqFP,
						tmpSeqFileName,
						tempOutputIndexFPs[uniqueIndexCtr],
						0,
						tmpDir,
						timing,
						totalDataStructureTime,
						totalSearchTime,
						totalOutputTime
							);
				if(VERBOSE >= 0) {
					fprintf(stderr, "Searching index file %d/%d (index #%d, bin #%d) complete...\n", 
							indexNum+1, numIndexes,
							indexIDs[indexNum][0], indexIDs[indexNum][1]);
				}
				indexNum++;
			}
			else {
				tempOutputIndexBinFPs = malloc(sizeof(gzFile)*numBins);
				if(NULL == tempOutputIndexBinFPs) {
					PrintError(FnName, "tempOutputIndexBinFPs", "Could not allocate memory", Exit, MallocMemory);
				}
				tempOutputIndexBinFileNames = malloc(sizeof(char*)*numBins);
				if(NULL == tempOutputIndexBinFileNames) {
					PrintError(FnName, "tempOutputIndexBinFileNames", "Could not allocate memory", Exit, MallocMemory);
				}

				for(i=0;i<numBins;i++) {
					tempOutputIndexBinFPs[i]=OpenTmpGZFile(tmpDir, &tempOutputIndexBinFileNames[i]);
				}

				// search each bin
				for(uniqueIndexBinCtr=0;uniqueIndexBinCtr<numBins;uniqueIndexBinCtr++) {

					assert(indexNum < numIndexes);
					if(VERBOSE >= 0) {
						if(1 == indexIDs[indexNum][1]) fprintf(stderr, "%s", BREAK_LINE);
						fprintf(stderr, "Searching index file %d/%d (index #%d, bin #%d)...\n", 
								indexNum+1, numIndexes,
								indexIDs[indexNum][0], indexIDs[indexNum][1]);
					}
					FindMatches(&indexFileNames[indexNum],
							1,
							rg,
							offsets,
							numOffsets,
							loadAllIndexes,
							space,
							keySize,
							maxKeyMatches,
							maxNumMatches,
							whichStrand,
							numThreads,
							queueLength,
							tmpSeqFP,
							tmpSeqFileName,
							tempOutputIndexBinFPs[uniqueIndexBinCtr],
							1,
							tmpDir,
							timing,
							totalDataStructureTime,
							totalSearchTime,
							totalOutputTime
								);
					if(VERBOSE >= 0) {
						fprintf(stderr, "Searching index file %d/%d (index #%d, bin #%d) complete...\n", 
								indexNum+1, numIndexes,
								indexIDs[indexNum][0], indexIDs[indexNum][1]);
					}

					// seek to start for merge
					ReopenTmpGZFile(&tempOutputIndexBinFPs[uniqueIndexBinCtr],
							&tempOutputIndexBinFileNames[uniqueIndexBinCtr]);

					indexNum++;
				}

				if(VERBOSE >= 0) {
					fprintf(stderr, "Merging the output from each bin...\n");
				}

				RGIndexInitialize(&tempIndex);
				RGIndexGetHeader(indexFileNames[indexNum-1], &tempIndex); // use previous

				startTime=time(NULL);
				numMatches = RGMatchesMergeIndexBins(tempOutputIndexBinFPs,
						numBins,
						tempOutputIndexFPs[uniqueIndexCtr],
						&tempIndex,
						maxKeyMatches,
						maxNumMatches);
				endTime=time(NULL);
				if(VERBOSE >= 0 && timing == 1) {
					seconds = (int)(endTime - startTime);
					hours = seconds/3600;
					seconds -= hours*3600;
					minutes = seconds/60;
					seconds -= minutes*60;
					fprintf(stderr, "Merging matches from the index bins took: %d hours, %d minutes and %d seconds\n",
							hours,
							minutes,
							seconds);
				}
				(*totalOutputTime)+=endTime-startTime;

				// Destroy
				for(i=0;i<numBins;i++) {
					CloseTmpGZFile(&tempOutputIndexBinFPs[i],
							&tempOutputIndexBinFileNames[i],
							1);
				}

				RGIndexDelete(&tempIndex);
				free(tempOutputIndexBinFPs);
				free(tempOutputIndexBinFileNames);

			}
			if(VERBOSE >= 0) {
				fprintf(stderr, "Found %d matches.\n", numMatches);
			}
		}

		/* Merge temporary output from each index and output to the output file. */
		if(numUniqueIndexes > 1) {
			if(VERBOSE >= 0) {
				fprintf(stderr, "%s", BREAK_LINE);
				fprintf(stderr, "Merging the output from each index...\n");
			}
			for(i=0;i<numUniqueIndexes;i++) {
				ReopenTmpGZFile(&tempOutputIndexFPs[i], 
						&tempOutputIndexFileNames[i]);
			}

			startTime=time(NULL);
			/* Merge the temp index files into the all indexes file */
			numWritten=RGMatchesMergeFilesAndOutput(tempOutputIndexFPs,
					numUniqueIndexes,
					tempOutputFP,
					maxNumMatches,
					queueLength);
			endTime=time(NULL);
			if(VERBOSE >= 0 && timing == 1) {
				seconds = (int)(endTime - startTime);
				hours = seconds/3600;
				seconds -= hours*3600;
				minutes = seconds/60;
				seconds -= minutes*60;
				fprintf(stderr, "Merging matches from the indexes took: %d hours, %d minutes and %d seconds\n",
						hours,
						minutes,
						seconds);
			}
			(*totalOutputTime)+=endTime-startTime;

			/* If we had more than one index, then this is the total merged number of matches */
			numMatches = numWritten;

			/* Close the temporary index files */
			for(i=0;i<numUniqueIndexes;i++) {
				CloseTmpGZFile(&tempOutputIndexFPs[i],
						&tempOutputIndexFileNames[i],
						1);
			}
		}
	}
	if(VERBOSE >= 0) {
		fprintf(stderr, "Found matches for %d reads.\n", numMatches);
	}

	/* Close the temporary read files */
	CloseTmpGZFile(tmpSeqFP, tmpSeqFileName, 1);

	if(CopyForNextSearch == copyForNextSearch) {
		/* Go through the temporary output file and output those reads that have 
		 * at least one match to the final output file.  For those reads that have
		 * zero matches, output them to the temporary read file */

		if(VERBOSE >= 0) {
			fprintf(stderr, "Copying unmatched reads for secondary index search.\n");
		}

		/* Open a new temporary read file */
		tempRGMatchesAFP.fp = NULL; tempRGMatchesAFP.gz = NULL; 
#ifndef DISABLE_BZ2
		tempRGMatchesAFP.bz2 = NULL;
#endif
		tempRGMatchesAFP.c = AFILE_GZ_COMPRESSION;
		tempRGMatchesAFP.gz = OpenTmpGZFile(tmpDir, &tempRGMatchesFileName);

		startTime=time(NULL);
		assert(tempOutputFP != outputFP); // this is very important
		numWritten=ReadTempReadsAndOutput(tempOutputFP,
				tempOutputFileName,
				outputFP,
				&tempRGMatchesAFP);
		endTime=time(NULL);
		(*totalOutputTime)+=endTime-startTime;

		/* Move to the beginning of the read file */
		ReopenTmpGZFile(&tempRGMatchesAFP.gz, &tempRGMatchesFileName);

		if(VERBOSE >= 0) {
			fprintf(stderr, "Splitting unmatched reads into temp files.\n");
		}
		/* Now apportion the remaining reads into temp files for the threads when 
		 * searching the secondary indexes 
		 * */
		WriteReadsToTempFile(&tempRGMatchesAFP,
				tmpSeqFP,
				tmpSeqFileName,
				0,
				INT_MAX,
				tmpDir,
				&numReads,
				space);
		/* In this case, all the reads should be valid so we should apportion all reads */
		assert(numReads == numWritten);

		/* Close the tempRGMatchesAFP */
		CloseTmpGZFile(&tempRGMatchesAFP.gz, &tempRGMatchesFileName, 1);
		/* Close the temporary output file */
		CloseTmpGZFile(&tempOutputFP, &tempOutputFileName, 1);
	}
	else {
		gzclose(tempOutputFP);
	}

	/* Free memory for temporary file pointers */
	free(tempOutputIndexFPs);
	free(tempOutputIndexFileNames);

	if(VERBOSE >= 0) {
		if(MainIndexes == indexesType) {
			fprintf(stderr, "Searching main index%s complete.\n",
					(1 == numIndexes) ? "" : "es");
		}
		else {
			fprintf(stderr, "Searching secondary index%s complete.\n",
					(1 == numIndexes) ? "" : "es");
		}
	}

	return numMatches;
}

int FindMatches(char **indexFileName,
		int32_t numIndexes,
		RGBinary *rg,
		int32_t *offsets,
		int numOffsets,
		int loadAllIndexes,
		int space,
		int keySize,
		int maxKeyMatches,
		int maxNumMatches,
		int whichStrand,
		int numThreads,
		int queueLength,
		gzFile *tmpSeqFP,
		char **tmpSeqFileName,
		gzFile outputFP,
		int outputOffsets,
		char *tmpDir,
		int timing,
		int *totalDataStructureTime,
		int *totalSearchTime,
		int *totalOutputTime)
{
	char *FnName = "FindMatches";
	int i, j, k;
	RGIndex *indexes=NULL;
	int numMatches = 0;
	time_t startTime, endTime;
	int errCode;
	ThreadIndexData *data=NULL;
	pthread_t *threads=NULL;
	void *status;
	RGMatches *matchQueue=NULL;
	int32_t *matchQueueThreadIDs=NULL; // TODO: could make this more memory efficient
	int32_t matchQueueLength=queueLength;
	int32_t returnNumMatches=0, numReadsProcessed=0;

	/* Allocate memory for threads */
	threads=malloc(sizeof(pthread_t)*numThreads);
	if(NULL==threads) {
		PrintError(FnName, "threads", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Allocate memory to pass data to threads */
	data=malloc(sizeof(ThreadIndexData)*numThreads);
	if(NULL==data) {
		PrintError(FnName, "data", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Allocate memory for the indexes */
	indexes=malloc(sizeof(RGIndex)*numIndexes);
	if(NULL==indexes) {
		PrintError(FnName, "indexes", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Initialize indexes */
	for(i=0;i<numIndexes;i++) {
		RGIndexInitialize(&indexes[i]);
	}

	/* Read in the RG Index */
	startTime = time(NULL);
	for(i=0;i<numIndexes;i++) {
		ReadRGIndex(indexFileName[i], &indexes[i], space);
		if(IndexesMemoryAll == loadAllIndexes && 0 < indexes[i].depth) {
			PrintError(FnName, "index[i].depth", "Cannot use binned indexes when loading all into memory", Exit, OutOfRange);
		}
		/* Check that depth = 0 if we have more than one index */
		/* Adjust if necessary */
		if(0 < keySize &&
				indexes[i].hashWidth <= keySize &&
				keySize < indexes[i].keysize) {
			/* Adjust key size and width */
			for(j=k=0;k < indexes[i].width && j < keySize;k++) {
				if(1 == indexes[i].mask[k]) {
					j++;
				}
			}
			assert(j == keySize);
			indexes[i].width = k;
			indexes[i].keysize = keySize;
		}
	}
	endTime = time(NULL);
	(*totalDataStructureTime)+=endTime - startTime;	

	/* Set position to read from the beginning of the file */
	ReopenTmpGZFile(tmpSeqFP, tmpSeqFileName);

	/* Allocate match queue */
	matchQueue = malloc(sizeof(RGMatches)*matchQueueLength); 
	if(NULL == matchQueue) {
		PrintError(FnName, "matchQueue", "Could not allocate memory", Exit, MallocMemory);
	}
	matchQueueThreadIDs = malloc(sizeof(int32_t)*matchQueueLength); 
	if(NULL == matchQueueThreadIDs) {
		PrintError(FnName, "matchQueueThreadIDs", "Could not allocate memory", Exit, MallocMemory);
	}

	/* For each read */
	if(VERBOSE >= 0) {
		fprintf(stderr, "Reads processed: 0");
	}

	// Run
	startTime = time(NULL);
	while(0!=(numMatches = GetReads((*tmpSeqFP), matchQueue, matchQueueLength, space))) { // Read in data
		endTime = time(NULL);
		(*totalOutputTime)+=endTime - startTime;
	
	/* Initialize match structures */
		for(i=0;i<matchQueueLength;i++) {
		matchQueueThreadIDs[i] = -1;
	}

		// Initialize arguments to threads 
		for(i=0;i<numThreads;i++) {
			data[i].matchQueue = matchQueue;
			data[i].matchQueueThreadIDs = matchQueueThreadIDs;
			data[i].matchQueueLength = numMatches;
			data[i].numThreads = numThreads;
			data[i].indexes = indexes;
			data[i].numIndexes = numIndexes;
			data[i].rg = rg;
			data[i].offsets = offsets;
			data[i].numOffsets = numOffsets;
			data[i].space = space;
			data[i].maxKeyMatches = maxKeyMatches;
			data[i].maxNumMatches = maxNumMatches;
			data[i].whichStrand = whichStrand;
			data[i].outputOffsets = outputOffsets;
			data[i].threadID = i;
		}
		// Spawn threads
		startTime = time(NULL);
		/* Open threads */
		for(i=0;i<numThreads;i++) {
			/* Start thread */
			errCode = pthread_create(&threads[i], /* thread struct */
					NULL, /* default thread attributes */
					FindMatchesThread, /* start routine */
					&data[i]); /* data to routine */
			if(0!=errCode) {
				PrintError(FnName, "pthread_create: errCode", "Could not start thread", Exit, ThreadError);
			}
		}
		/* Wait for threads to return */
		for(i=0;i<numThreads;i++) {
			/* Wait for the given thread to return */
			errCode = pthread_join(threads[i],
					&status);
			/* Check the return code of the thread */
			if(0!=errCode) {
				PrintError(FnName, "pthread_join: errCode", "Thread returned an error", Exit, ThreadError);
			}
			returnNumMatches += data[i].numMatches;
		}
		endTime = time(NULL);
		(*totalSearchTime)+=endTime - startTime;

		/* Output to file */
		startTime = time(NULL);
		for(i=0;i<numMatches;i++) {
			if(0 == outputOffsets) {
				RGMatchesPrint(outputFP, 
						&matchQueue[i]);
			}
			else {
				RGMatchesPrintWithOffsets(outputFP, 
						&matchQueue[i]);
			}
		}
		endTime = time(NULL);
		(*totalOutputTime)+=endTime - startTime;

		numReadsProcessed += numMatches;
		if(VERBOSE >= 0) {
			fprintf(stderr, "\rReads processed: %d", numReadsProcessed);
		}

		/* Free matches */
		for(i=0;i<numMatches;i++) {
			RGMatchesFree(&matchQueue[i]);
			matchQueueThreadIDs[i] = -1;
		}

		// For reading
		startTime = time(NULL);
	}
	endTime = time(NULL);
	(*totalOutputTime)+=endTime - startTime;

	if(VERBOSE >= 0) {
		fprintf(stderr, "\rReads processed: %d\n", numReadsProcessed);
	}

	/* Free memory of the RGIndex */
	if(VERBOSE >= 0) {
		fprintf(stderr, "Cleaning up index%s.\n",
				(1 == numIndexes) ? "" : "es");
	}
	startTime = time(NULL);
	for(i=0;i<numIndexes;i++) {
		RGIndexDelete(&indexes[i]);
	}
	free(indexes);
	endTime = time(NULL);
	(*totalDataStructureTime)+=endTime - startTime;	

	// Free match queue
	free(matchQueue);
	free(matchQueueThreadIDs);

	/* Free thread data */
	free(threads);
	free(data);

	return returnNumMatches;
}

/* TODO */
void *FindMatchesThread(void *arg)
{
	//char *FnName="FindMatchesThread";
	int32_t i, j, k, l;
	int foundMatch = 0;
	ThreadIndexData *data=(ThreadIndexData*)arg;
	/* Function arguments */
	RGMatches *matchQueue = data->matchQueue;
	int32_t *matchQueueThreadIDs = data->matchQueueThreadIDs;
	int32_t matchQueueLength = data->matchQueueLength;
	int32_t numThreads = data->numThreads; 
	RGIndex *indexes = data->indexes;
	int32_t numIndexes = data->numIndexes;
	RGBinary *rg = data->rg;
	int32_t *offsets = data->offsets;
	int numOffsets = data->numOffsets;
	int space = data->space;
	int maxKeyMatches = data->maxKeyMatches;
	int maxNumMatches = data->maxNumMatches;
	int whichStrand = data->whichStrand;
	int outputOffsets = data->outputOffsets;
	int threadID = data->threadID;
	data->numMatches = 0;

	i=0;
	while(i<matchQueueLength) {
		if(1 < numThreads) {
			pthread_mutex_lock(&matchQueueMutex);
			if(matchQueueThreadIDs[i] < 0) {
				// mark this block
				for(j=i;j<matchQueueLength && j<i+BFAST_MATCH_THREAD_BLOCK_SIZE;j++) {
					matchQueueThreadIDs[j] = threadID;
				}
			}
			else if(matchQueueThreadIDs[i] != threadID) {
				pthread_mutex_unlock(&matchQueueMutex);
				i+=BFAST_MATCH_THREAD_BLOCK_SIZE;
				// skip this block
				continue;
			}
			pthread_mutex_unlock(&matchQueueMutex);
		}

		// Process the block
		for(l=0;l<BFAST_MATCH_THREAD_BLOCK_SIZE && i<matchQueueLength;l++,i++) {
			assert(numThreads <= 1 || matchQueueThreadIDs[i] == threadID); 
			/* Read */
			foundMatch = 0;
			for(j=0;j<matchQueue[i].numEnds;j++) {
				if(1 == numIndexes) {
					RGReadsFindMatches(&indexes[0],
							rg,
							&matchQueue[i].ends[j], 
							outputOffsets,
							offsets,
							numOffsets,
							space,
							0,
							0,
							0,
							0,
							0,
							maxKeyMatches,
							maxNumMatches,
							whichStrand);
				}
				else {
					for(k=0;k<numIndexes && 1!=matchQueue[i].ends[j].maxReached;k++) {
						RGReadsFindMatches(&indexes[k],
								rg,
								&matchQueue[i].ends[j], 
								outputOffsets,
								offsets,
								numOffsets,
								space,
								0,
								0,
								0,
								0,
								0,
								maxKeyMatches,
								maxNumMatches,
								whichStrand);
					}
				}
				if(0 < matchQueue[i].ends[j].numEntries && 1 != matchQueue[i].ends[j].maxReached) {
					foundMatch = 1;
				}
			}
			if(1 == foundMatch) {
				data->numMatches++;
				//DEBUGGING
				//RGMatchesCheck(&matchQueue[i], rg);
			}
		}


	}

	return arg;
}
