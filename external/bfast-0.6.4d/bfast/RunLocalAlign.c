#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include "BLibDefinitions.h"
#include "BError.h"
#include "BLib.h"
#include "RGBinary.h"
#include "RGMatch.h" 
#include "RGMatches.h" 
#include "AlignedRead.h"
#include "AlignedEnd.h" 
#include "AlignedEntry.h" 
#include "ScoringMatrix.h"
#include "Align.h"
#include "RunLocalAlign.h"

static pthread_mutex_t matchQueueMutex = PTHREAD_MUTEX_INITIALIZER;

/* TODO */
void RunAligner(char *fastaFileName,
		char *matchFileName,
		char *scoringMatrixFileName,
		int32_t ungapped,
		int32_t unconstrained,
		int32_t bestOnly,
		int32_t space,
		int32_t startReadNum,
		int32_t endReadNum,
		int32_t offsetLength,
		int32_t maxNumMatches,
		int32_t avgMismatchQuality,
		int32_t numThreads,
		int32_t queueLength,
		int32_t usePairedEndLength,
		int32_t pairedEndLength,
		int32_t mirroringType,
		int32_t forceMirroring,
		int32_t timing,
		FILE *fpOut)
{
	char *FnName = "RunAligner";
	gzFile outputFP=NULL;
	gzFile matchFP=NULL;
	int32_t startTime, endTime;
	RGBinary rg;
	int32_t totalReferenceGenomeTime;
	int32_t totalAlignedTime;
	int32_t totalFileHandlingTime;
	int32_t seconds, minutes, hours;

	startTime = time(NULL);
	RGBinaryReadBinary(&rg,
			NTSpace, // always NT space
			fastaFileName);
	endTime = time(NULL);
	/* Unpack */
	/*
	   RGBinaryUnPack(&rg);
	   */
	totalReferenceGenomeTime = endTime - startTime;

	/* Check rg to make sure it is in NT Space */
	if(rg.space != NTSpace) {
		PrintError(FnName, "rg->space", "The reference genome must be in NT space", Exit, OutOfRange);
	}

	/* Open output file */
	if((outputFP=gzdopen(fileno(fpOut), "wb"))==0) {
		PrintError(FnName, "stdout", "Could not open stdout file for writing", Exit, OpenFileError);
	}

	if(0 <= VERBOSE) {
		fprintf(stderr, "%s", BREAK_LINE);
		fprintf(stderr, "Reading match file from %s.\n",
				(NULL == matchFileName) ? "stdin" : matchFileName);
	}

	/* Open current match file */
	if(NULL == matchFileName) {
		if((matchFP=gzdopen(fileno(stdin), "rb"))==0) {
			PrintError(FnName, "stdin", "Could not open stdin for reading", Exit, OpenFileError);
		}
	}
	else {
		if((matchFP=gzopen(matchFileName, "rb"))==0) {
			PrintError(FnName, matchFileName, "Could not open file for reading", Exit, OpenFileError);
		}
	}

	RunDynamicProgramming(matchFP,
			&rg,
			scoringMatrixFileName,
			ungapped,
			unconstrained,
			bestOnly,
			space,
			startReadNum,
			endReadNum,
			offsetLength,
			maxNumMatches,
			avgMismatchQuality,
			numThreads,
			queueLength,
			usePairedEndLength,
			pairedEndLength,
			mirroringType,
			forceMirroring,
			outputFP,
			&totalAlignedTime,
			&totalFileHandlingTime);

	if(0 <= VERBOSE) {
		fprintf(stderr, "%s", BREAK_LINE);
	}

	/* Close the match file */
	gzclose(matchFP);

	/* Close output file */
	gzclose(outputFP);

	/* Free the Reference Genome */
	RGBinaryDelete(&rg);

	if(1 == timing) {
		/* Output loading reference genome time */                        
		seconds = totalReferenceGenomeTime;
		hours = seconds/3600;
		seconds -= hours*3600;
		minutes = seconds/60;
		seconds -= minutes*60;
		if(0 <= VERBOSE) {
			fprintf(stderr, "Reference Genome loading time took: %d hours, %d minutes and %d seconds.\n"
					,
					hours,
					minutes,
					seconds
				   );
		}

		/* Output aligning time */
		seconds = totalAlignedTime;
		hours = seconds/3600;
		seconds -= hours*3600;
		minutes = seconds/60;
		seconds -= minutes*60;
		if(0 <= VERBOSE) {
			fprintf(stderr, "Align time took: %d hours, %d minutes and %d seconds.\n",
					hours,
					minutes,
					seconds
				   );
		}

		/* Output file handling time */
		seconds = totalFileHandlingTime;
		hours = seconds/3600;
		seconds -= hours*3600;
		minutes = seconds/60;
		seconds -= minutes*60;
		if(0 <= VERBOSE) {
			fprintf(stderr, "File handling time took: %d hours, %d minutes and %d seconds.\n",
					hours,
					minutes,
					seconds
				   );

		}
	}
}

/* TODO */
void RunDynamicProgramming(gzFile matchFP,
		RGBinary *rg,
		char *scoringMatrixFileName,
		int32_t ungapped,
		int32_t unconstrained,
		int32_t bestOnly,
		int32_t space,
		int32_t startReadNum,
		int32_t endReadNum,
		int32_t offsetLength,
		int32_t maxNumMatches,
		int32_t avgMismatchQuality,
		int32_t numThreads,
		int32_t queueLength,
		int32_t usePairedEndLength,
		int32_t pairedEndLength,
		int32_t mirroringType,
		int32_t forceMirroring,
		gzFile outputFP,
		int32_t *totalAlignedTime,
		int32_t *totalFileHandlingTime)
{
	char *FnName="RunDynamicProgramming";
	/* local variables */
	ScoringMatrix sm;
	double mismatchScore;
	RGMatches m;
	int32_t i;

	int32_t numAligned=0;
	int32_t numNotAligned=0;
	int32_t startTime, endTime;
	int64_t numLocalAlignments=0;
	/* Thread specific data */
	ThreadData *data;
	pthread_t *threads=NULL;
	int32_t errCode;
	void *status;
	RGMatches *matchQueue=NULL;
	int32_t *matchQueueThreadIDs=NULL;
	AlignedRead *alignedQueue=NULL;
	int32_t matchQueueLength=0;
	int32_t matchFPctr = 1;
	int32_t outputCtr = 0;
	int32_t numReadsProcessed = 0, numMatchesRead = 0;

	/* Initialize */
	RGMatchesInitialize(&m);
	ScoringMatrixInitialize(&sm);

	/* Allocate match queue */
	matchQueue = malloc(sizeof(RGMatches)*queueLength);
	if(NULL == matchQueue) {
		PrintError(FnName, "matchQueue", "Could not allocate memory", Exit, MallocMemory);
	}
	matchQueueThreadIDs = malloc(sizeof(int32_t)*queueLength);
	if(NULL == matchQueueThreadIDs) {
		PrintError(FnName, "matchQueueThreadIDs", "Could not allocate memory", Exit, MallocMemory);
	}
	alignedQueue = malloc(sizeof(AlignedRead)*queueLength);
	if(NULL == alignedQueue) {
		PrintError(FnName, "alignedQueue", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Allocate memory for thread arguments */
	data = malloc(sizeof(ThreadData)*numThreads);
	if(NULL==data) {
		PrintError(FnName, "data", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Allocate memory for thread ids */
	threads = malloc(sizeof(pthread_t)*numThreads);
	if(NULL==threads) {
		PrintError(FnName, "threads", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Start file handling timer */
	startTime = time(NULL);

	/* Read in scoring matrix */
	if(NULL != scoringMatrixFileName) {
		ScoringMatrixRead(scoringMatrixFileName, &sm, space); 
	}
	/* End file handling timer */
	endTime = time(NULL);
	totalFileHandlingTime += endTime - startTime;

	/* Calculate mismatch score */
	/* Assumes all match scores are the same and all substitution scores are the same */
	if(space == NTSpace) {
		mismatchScore = sm.ntMatch - sm.ntMismatch;
	}
	else {
		mismatchScore = sm.colorMatch - sm.colorMismatch;
	}

	// Skip matches
	startTime = time(NULL);
	SkipMatches(matchFP, &matchFPctr, startReadNum);
	endTime = time(NULL);
	(*totalFileHandlingTime) += endTime - startTime;


	if(0 <= VERBOSE) {
		fprintf(stderr, "%s", BREAK_LINE);
		fprintf(stderr, "Performing alignment...\n");
		fprintf(stderr, "Reads processed: 0");
	}

	startTime = time(NULL);
	while(0 != (numMatchesRead = GetMatches(matchFP, &matchFPctr, startReadNum, endReadNum, matchQueue, queueLength))) {
		endTime = time(NULL);
		(*totalFileHandlingTime) += endTime - startTime;

		numReadsProcessed += numMatchesRead;
		matchQueueLength = numMatchesRead;

		/* Initialize match structures */
		for(i=0;i<matchQueueLength;i++) {
			matchQueueThreadIDs[i] = -1;
		}

		/* Initialize thread arguments */
		for(i=0;i<numThreads;i++) {
			data[i].rg=rg;
			data[i].space=space;
			data[i].offsetLength=offsetLength;
			data[i].usePairedEndLength = usePairedEndLength;
			data[i].pairedEndLength = pairedEndLength;
			data[i].mirroringType = mirroringType;
			data[i].forceMirroring = forceMirroring;
			data[i].sm = &sm;
			data[i].ungapped = ungapped;
			data[i].unconstrained = unconstrained;
			data[i].bestOnly = bestOnly;
			data[i].numLocalAlignments = 0;
			data[i].avgMismatchQuality = avgMismatchQuality;
			data[i].mismatchScore = mismatchScore;
			data[i].queueLength = matchQueueLength;
			data[i].threadID = i;
			data[i].numThreads = numThreads;
			data[i].numAligned = 0;
			data[i].numNotAligned = 0;
			data[i].matchQueue = matchQueue;
			data[i].matchQueueThreadIDs = matchQueueThreadIDs;
			data[i].alignedQueue = alignedQueue;
		}

		/* Create threads */
		startTime = time(NULL);
		for(i=0;i<numThreads;i++) {
			/* Start thread */
			errCode = pthread_create(&threads[i], /* thread struct */
					NULL, /* default thread attributes */
					RunDynamicProgrammingThread, /* start routine */
					&data[i]); /* data to routine */
			if(0!=errCode) {
				PrintError(FnName, "pthread_create: errCode", "Could not start thread", Exit, ThreadError);
			}
			if(VERBOSE >= DEBUG) {
				fprintf(stderr, "Created threadID:%d\n",
						i);
			}
		}

		/* Wait for the threads to finish */
		for(i=0;i<numThreads;i++) {
			/* Wait for the given thread to return */
			errCode = pthread_join(threads[i],
					&status);
			if(VERBOSE >= DEBUG) {
				fprintf(stderr, "Thread returned with errCode:%d\n",
						errCode);
			}
			/* Check the return code of the thread */
			if(0!=errCode) {
				PrintError(FnName, "pthread_join: errCode", "Thread returned an error", Exit, ThreadError);
			}
		}
		endTime = time(NULL);
		(*totalAlignedTime) += (endTime - startTime);

		// Output to file 
		startTime = time(NULL);
		for(i=0;i<matchQueueLength;i++) {
			AlignedReadPrint(&alignedQueue[i],
					outputFP);

			/* Free memory */
			AlignedReadFree(&alignedQueue[i]);
			RGMatchesFree(&matchQueue[i]);
			outputCtr++;
		}
		endTime = time(NULL);
		(*totalFileHandlingTime) += endTime - startTime;

		/* Sum up statistics */
		for(i=0;i<numThreads;i++) {
			numAligned += data[i].numAligned;
			numNotAligned += data[i].numNotAligned;
			numLocalAlignments += data[i].numLocalAlignments;
		}

		if(VERBOSE >= 0) {
			fprintf(stderr, "\rReads processed: %d", numReadsProcessed);
		}

		startTime = time(NULL);
	}


	if(0 <= VERBOSE) {
		fprintf(stderr, "\rReads processed: %d\n", numReadsProcessed);
		fprintf(stderr, "Alignment complete.\n");
	}

	endTime = time(NULL);

	if(VERBOSE >=0) {
		fprintf(stderr, "Performed %lld local alignments.\n", (long long int)numLocalAlignments);
		fprintf(stderr, "Outputted alignments for %d reads.\n", numAligned);
		fprintf(stderr, "Outputted %d reads for which there were no alignments.\n", numNotAligned); 
		fprintf(stderr, "Outputting complete.\n");
	}

	/* Free memory */
	free(matchQueue);
	free(matchQueueThreadIDs);
	free(alignedQueue);
	free(data);
	free(threads);
}

/* TODO */
void *RunDynamicProgrammingThread(void *arg)
{
	/* Recover arguments */
	ThreadData *data = (ThreadData *)(arg);
	RGBinary *rg=data->rg;
	int32_t space=data->space;
	int32_t offsetLength=data->offsetLength;
	int32_t usePairedEndLength=data->usePairedEndLength;
	int32_t pairedEndLength=data->pairedEndLength;
	int32_t mirroringType=data->mirroringType;
	int32_t forceMirroring=data->forceMirroring;
	ScoringMatrix *sm = data->sm;
	int32_t ungapped=data->ungapped;
	int32_t unconstrained=data->unconstrained;
	int32_t bestOnly=data->bestOnly;
	int32_t threadID=data->threadID;
	int32_t numThreads=data->numThreads;
	int32_t avgMismatchQuality=data->avgMismatchQuality;
	double mismatchScore=data->mismatchScore;
	int32_t queueLength=data->queueLength;
	AlignedRead *alignedQueue=data->alignedQueue;
	RGMatches *matchQueue=data->matchQueue;
	int32_t *matchQueueThreadIDs = data->matchQueueThreadIDs;
	/* Local variables */
	//char *FnName = "RunDynamicProgrammingThread";
	int32_t i, j, wasAligned, queueIndex;
	AlignMatrix matrix;
	
	/* Initialize */
	AlignMatrixInitialize(&matrix);

	/* Go through each read in the match file */
	queueIndex=0;
	while(queueIndex<queueLength) {
		if(1 < numThreads) {
			pthread_mutex_lock(&matchQueueMutex);
			if(matchQueueThreadIDs[queueIndex] < 0) {
				// mark this block
				for(j=queueIndex;j<queueLength && j<queueIndex+BFAST_LOCALALIGN_THREAD_BLOCK_SIZE;j++) {
					matchQueueThreadIDs[j] = threadID;
				}
			}
			else if(matchQueueThreadIDs[queueIndex] != threadID) {
				pthread_mutex_unlock(&matchQueueMutex);
				queueIndex+=BFAST_LOCALALIGN_THREAD_BLOCK_SIZE;
				// skip this block
				continue;
			}
			pthread_mutex_unlock(&matchQueueMutex);
		}

		// Process this block
		for(i=0;i<BFAST_LOCALALIGN_THREAD_BLOCK_SIZE && queueIndex<queueLength;i++,queueIndex++) {
			assert(numThreads <= 1 || matchQueueThreadIDs[queueIndex] == threadID);
			AlignedReadInitialize(&alignedQueue[queueIndex]);

			wasAligned=0;
			if(1 == IsValidMatch(&matchQueue[queueIndex])) {

				/* Update the number of local alignments performed */
				data->numLocalAlignments += AlignRGMatches(&matchQueue[queueIndex],
						rg,
						&alignedQueue[queueIndex],
						space,
						offsetLength,
						sm,
						ungapped,
						unconstrained,
						bestOnly,
						usePairedEndLength,
						pairedEndLength,
						mirroringType,
						forceMirroring,
						&matrix);

				for(j=wasAligned=0;j<alignedQueue[queueIndex].numEnds;j++) {
					if(0 < alignedQueue[queueIndex].ends[j].numEntries) {
						wasAligned = 1;
					}
				}
			}

			if(1 == wasAligned) {
				/* Remove duplicates */
				AlignedReadRemoveDuplicates(&alignedQueue[queueIndex],
						AlignedEntrySortByAll);
				/* Updating mapping quality */
				AlignedReadUpdateMappingQuality(&alignedQueue[queueIndex], 
						mismatchScore, 
						avgMismatchQuality);
			}
			else {
				/* Copy over to alignedQueue[queueIndex] */
				AlignedReadAllocate(&alignedQueue[queueIndex],
						matchQueue[queueIndex].readName,
						matchQueue[queueIndex].numEnds,
						space);
				for(j=0;j<matchQueue[queueIndex].numEnds;j++) {
					AlignedEndAllocate(&alignedQueue[queueIndex].ends[j],
							matchQueue[queueIndex].ends[j].read,
							matchQueue[queueIndex].ends[j].qual,
							0);
				}
			}

			if(0 == wasAligned) {
				data->numNotAligned++;
			}
			else {
				data->numAligned++;
			}

			/* Free memory */
			RGMatchesFree(&matchQueue[queueIndex]);
		}
	}
	/* Free the matrix, free your mind */
	AlignMatrixFree(&matrix);

	return arg;
}

int32_t GetMatches(gzFile matchFP, int32_t *matchFPctr, int32_t startReadNum, int32_t endReadNum, RGMatches *m, int32_t maxToRead)
{
	char *FnName="GetMatches";
	int32_t numRead = 0;

	if((*matchFPctr) < startReadNum) {
		PrintError(FnName, "matchFPctr < startReadNum", "The start read number was greater the actual number of reads found", Warn, OutOfRange);
		numRead=0; // to be explicit
	}
	else {
		while(numRead < maxToRead && (*matchFPctr) <= endReadNum) {
			RGMatchesInitialize(&(m[numRead]));
			if(EOF == RGMatchesRead(matchFP, &(m[numRead]))) {
				break;
			}
			(*matchFPctr)++;
			numRead++;
		}
	}
	return numRead;
}

void SkipMatches(gzFile matchFP, int32_t *matchFPctr, int32_t startReadNum)
{
	RGMatches m;

	if(startReadNum <= 1) {
		return;
	}

	if(0 <= VERBOSE) {
		fprintf(stderr, "Skipping matches...\nCurrently on:\n0");
	}

	RGMatchesInitialize(&m);
	while((*matchFPctr) < startReadNum && EOF != RGMatchesRead(matchFP, &m)) {
		if(0 <= VERBOSE && (*matchFPctr)%ALIGN_SKIP_ROTATE_NUM==0) {
			fprintf(stderr, "\r%d", (*matchFPctr));
		}
		RGMatchesFree(&m);
		(*matchFPctr)++;
	}
	if(0 <= VERBOSE) {
		fprintf(stderr, "\r%d\n", (*matchFPctr));
	}
}
