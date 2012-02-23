#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <config.h>
#include <pthread.h>
#include <unistd.h>  

#include "../bfast/BLibDefinitions.h"
#include "../bfast/BLib.h"
#include "../bfast/BError.h"
#include "../bfast/RGIndex.h"
#include "../bfast/RGRanges.h"
#include "../bfast/RGMatch.h"
#include "../bfast/RGReads.h"
#include "bindexdist.h"

#define Name "bindexdist"
#define BINDEXDIST_ROTATE_NUM 100000
#define BINDEXDIST_SORT_ROTATE_INC 0.01

/* Prints each unique read from the genome and the number 
 * of times it occurs, where the genome is contained in 
 * the bfast index file.
 * */

int PrintUsage()
{
	fprintf(stderr, "%s %s\n", "bfast", PACKAGE_VERSION);
	fprintf(stderr, "\nUsage:%s [options]\n", Name);
	fprintf(stderr, "\t-f\tFILE\tSpecifies the file name of the FASTA reference genome\n");
	fprintf(stderr, "\t-i\tFILE\tSpecifies the bfast index file name\n");
	fprintf(stderr, "\t-s\tINT\tStrands 0: both strands 1: forward only 2: reverse only\n");
	fprintf(stderr, "\t-n\tINT\tSpecifies the number of threads to use (Default 1)\n");
	fprintf(stderr, "\t-A\tINT\t0: NT space 1: Color space\n");
	fprintf(stderr, "\t-T\tDIR\tSpecifies the directory in which to store temporary file\n");
	fprintf(stderr, "\t-h\t\tprints this help message\n");
	fprintf(stderr, "\nsend bugs to %s\n",
			PACKAGE_BUGREPORT);
	return 1;
}

int main(int argc, char *argv[]) 
{
	char *indexFileName=NULL;
	char *fastaFileName=NULL;
	char tmpDir[MAX_FILENAME_LENGTH]="./";
	int numThreads = 1;
	int whichStrand = 0;
	int space = NTSpace;
	int c;
	RGBinary rg;
	RGIndex index;

	while((c = getopt(argc, argv, "f:i:n:s:A:T:h")) >= 0) {
		switch(c) {
			case 'f': fastaFileName=strdup(optarg); break;
			case 'h': return PrintUsage();
			case 'i': indexFileName=strdup(optarg); break;
			case 's': whichStrand=atoi(optarg); break;
			case 'n': numThreads=atoi(optarg); break;
			case 'A': space=atoi(optarg); break;
			case 'T': strcpy(tmpDir, optarg); break;
			default: fprintf(stderr, "Unrecognized option: -%c\n", c); return 1;
		}
	}

	if(1 == argc || argc != optind) {
		return PrintUsage();
	}

	if(NULL == indexFileName) {
		PrintError(Name, "indexFileName", "Command line option", Exit, InputArguments);
	}
	if(NULL == fastaFileName) {
		PrintError(Name, "fastaFileName", "Command line option", Exit, InputArguments);
	}

	assert(whichStrand == BothStrands || whichStrand == ForwardStrand || whichStrand == ReverseStrand);

	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "../Starting %s.\n", Name);

	/* Read in the rg binary file */
	RGBinaryReadBinary(&rg, space, fastaFileName);

	/* Read the index */
	RGIndexRead(&index, indexFileName);

	if(index.space != space) {
		PrintError(Name, "space", "The index and space do not match", Exit, InputArguments); 
	}

	fprintf(stderr, "%s", BREAK_LINE);
	PrintDistribution(&index, 
			&rg, 
			whichStrand,
			tmpDir,
			numThreads); 
	fprintf(stderr, "%s", BREAK_LINE);

	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "Cleaning up.\n");
	/* Delete the index */
	RGIndexDelete(&index);
	/* Delete the rg */
	RGBinaryDelete(&rg);
	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "Terminating successfully!\n");
	fprintf(stderr, "%s", BREAK_LINE);

	return 0;
}

void PrintDistribution(RGIndex *index, 
		RGBinary *rg,
		int whichStrand,
		char *tmpDir,
		int numThreads)
{
	char *FnName = "PrintDistribution";
	FILE *fp;
	int64_t startIndex = 0;
	int64_t endIndex = index->length-1;
	int64_t curIndex=0, nextIndex=0;
	int64_t counter=0;
	int64_t numDifferent = 0;
	int64_t numForward, numReverse;
	int64_t prevIndex=0;
	char *read=NULL;
	char *reverseRead=NULL;
	int64_t i, j;
	char **reads=NULL;
	int64_t *readCounts=NULL;
	int64_t numReads=0;
	pthread_t *threads=NULL;
	ThreadData *data=NULL;
	int errCode;
	void *status;

	fprintf(stderr, "Out of %lld, currently on:\n0",
			(long long int)(endIndex - startIndex + 1));
	/* Go through every possible read in the genome using the index */
	for(curIndex=startIndex, nextIndex=startIndex, counter=0, numDifferent=0;
			curIndex <= endIndex;
			curIndex = nextIndex) {
		if(counter >= BINDEXDIST_ROTATE_NUM) {
			fprintf(stderr, "\r%10lld", 
					(long long int)(curIndex-startIndex));
			counter -= BINDEXDIST_ROTATE_NUM;
		}
		/* Get the matches for the contig/pos */
		GetMatchesFromContigPos(index,
				rg,
				(index->contigType==Contig_8)?(index->contigs_8[curIndex]):(index->contigs_32[curIndex]),
				index->positions[curIndex],
				&numForward, 
				&numReverse,
				&read,
				&reverseRead);
		assert(numForward + numReverse> 0);

		nextIndex += numForward;
		counter += numForward;
		/* In case reverse is zero */
		if(numForward <= 0) {
			nextIndex++;
			counter++;
		}

		/* Reallocate memory */
		prevIndex = numReads;
		if((BothStrands == whichStrand || ReverseStrand == whichStrand) &&
				numReverse == 0) {
			numReads+=2; /* One for both strands */
		}
		else {
			numReads++; /* Only for + strand */
		}
		reads = realloc(reads, sizeof(char*)*numReads);
		if(NULL==reads) {
			PrintError(FnName, "reads", "Could not allocate memory", Exit, MallocMemory);
		}
		while(prevIndex < numReads) {
			reads[prevIndex] = malloc(sizeof(char)*(index->width+1));
			if(NULL==reads[prevIndex]) {
				PrintError(FnName, "reads[prevIndex]", "Could not allocate memory", Exit, MallocMemory);
			}
			prevIndex++;
		}
		readCounts = realloc(readCounts, sizeof(int64_t)*numReads);
		if(NULL==readCounts) {
			PrintError(FnName, "readCounts", "Could not allocate memory", Exit, MallocMemory);
		}
		/* Copy over */
		assert(strlen(read) < SEQUENCE_LENGTH);
		assert(strlen(reverseRead) < SEQUENCE_LENGTH);
		ToLowerRead(read, index->width+1); 
		strcpy(reads[numReads-1], read);
		readCounts[numReads-1] = numForward+numReverse;
		if((BothStrands == whichStrand || ReverseStrand == whichStrand) &&
				numReverse == 0) {
			if(0 == strcmp(reverseRead, read)) {
				fprintf(stderr, "read=%s\nreverseRead=%s\n",
						read,
						reverseRead);
			}
			assert(0 != strcmp(reverseRead, read));
			ToLowerRead(reverseRead, index->width+1); 
			strcpy(reads[numReads-2], reverseRead);
			readCounts[numReads-2] = numForward+numReverse;
		}

		/* Free memory */
		free(read);
		read = NULL;
		free(reverseRead);
		reverseRead = NULL;
	}
	fprintf(stderr, "\r%10lld\n", 
			(long long int)(curIndex-startIndex+1));

	/* Allocate memory for threads */
	threads=malloc(sizeof(pthread_t)*numThreads);
	if(NULL==threads) {
		PrintError(FnName, "threads", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Allocate memory to pass data to threads */
	data=malloc(sizeof(ThreadData)*numThreads);
	if(NULL==data) {
		PrintError(FnName, "data", "Could not allocate memory", Exit, MallocMemory);
	}

	for(i=0;i<numThreads;i++) {
		data[i].reads = reads;
		data[i].readCounts = readCounts;
		data[i].low = i*(numReads/numThreads);
		data[i].high = (i+1)*(numReads/numThreads)-1;
		data[i].tmpDir = tmpDir;
		data[i].readLength = index->width;
		data[i].showPercentComplete = 0;
	}
	data[0].low = 0;
	data[numThreads-1].high = numReads-1;
	data[numThreads-1].showPercentComplete = 1;

	/* Open threads */
	for(i=0;i<numThreads;i++) {
		/* Start thread */
		errCode = pthread_create(&threads[i], /* thread struct */
				NULL, /* default thread attributes */
				MergeSortReads, /* start routine */
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
	}

	/* Merge results from the sorts */
	fprintf(stderr, "\rMerging sorts from threads...    \n");
	for(j=1;j<numThreads;j=j*2) {
		for(i=0;i<numThreads;i+=2*j) {
			MergeHelper(reads,
					readCounts,
					data[i].low,
					data[i+j].low-1,
					data[i+2*j-1].high,
					tmpDir,
					index->width);
		}
	}
	fprintf(stderr, "../bfast/Sorting complete.\n");
	fprintf(stderr, "%s", BREAK_LINE);

	/* Remove duplicates */
	/*
	   fprintf(stderr, "%s", BREAK_LINE);
	   fprintf(stderr, "../bfast/Removing duplicates.\n");
	   prevIndex = 0;
	   for(i=1;i<numReads;i++) {
	   if(strcmp(reads[prevIndex], reads[i]) == 0) {
	   free(reads[i]);
	   reads[i] = NULL;
	   readCounts[i] = 0;
	   }
	   else {
	   prevIndex++;
	   if(i != prevIndex) {
	   assert(reads[prevIndex]==NULL);
	   assert(reads[i]!=NULL);
	   reads[prevIndex]=reads[i];
	   reads[i]=NULL;
	   }
	   readCounts[prevIndex]=readCounts[i];
	   }
	   }
	   numReads = prevIndex+1;
	   reads = realloc(reads, sizeof(char*)*numReads);
	   if(NULL==reads) {
	   PrintError(FnName, "reads", "Could not reallocate memory", Exit, ReallocMemory);
	   }
	   readCounts = realloc(readCounts, sizeof(int64_t)*numReads);
	   if(NULL==readCounts) {
	   PrintError(FnName, "readCounts", "Could not reallocate memory", Exit, ReallocMemory);
	   }
	   fprintf(stderr, "../bfast/Removing duplicates complete.\n");
	   fprintf(stderr, "%s", BREAK_LINE);
	   */

	/* Open the output file */
	if(!(fp = fdopen(fileno(stdout), "w"))) {
		PrintError(FnName, "stdout", "Could not open stdout for writing", Exit, OpenFileError);
	}

	/* Print */
	for(i=0;i<numReads;i++) {
		/* Duplicates */
		fprintf(fp, "%s\t%lld\n",
				reads[i],
				(long long int)readCounts[i]);
	}

	/* Close the file */
	fclose(fp);

	/* Free memory */
	for(i=0;i<numReads;i++) {
		free(reads[i]);
		reads[i]=NULL;
	}
	free(reads);
	reads=NULL;
	free(readCounts);
	readCounts=NULL;
}

void *MergeSortReads(void *arg)
{
	ThreadData *data = (ThreadData*)arg;
	char **reads = data->reads;
	int64_t *readCounts = data->readCounts;
	int64_t low = data->low;
	int64_t high = data->high;
	char *tmpDir = data->tmpDir;
	int readLength = data->readLength;

	double curPercentComplete = 0.0;
	/* Call helper */
	if(data->showPercentComplete == 1) {
		fprintf(stderr, "\r%3.2lf percent complete", 0.0);
	}

	MergeSortReadsHelper(reads,
			readCounts,
			low,
			high,
			low,
			high - low,
			data->showPercentComplete,
			&curPercentComplete,
			tmpDir,
			readLength);
	if(data->showPercentComplete == 1) {
		fprintf(stderr, "\r");
		fprintf(stderr, "thread %3.2lf percent complete", 100.0);
	}

	return NULL;
}

void MergeSortReadsHelper(char **reads,
		int64_t *readCounts,
		int64_t low,
		int64_t high,
		int64_t startLow,
		int64_t total,
		int showPercentComplete,
		double *curPercentComplete,
		char *tmpDir,
		int readLength)
{
	int64_t mid = (low + high)/2;
	if(low >= high) {

		if(showPercentComplete == 1) {
			assert(NULL!=curPercentComplete);
			if((*curPercentComplete) < 100.0*((double)(low - startLow))/total) {
				while((*curPercentComplete) < 100.0*((double)(low - startLow))/total) {
					(*curPercentComplete) += BINDEXDIST_SORT_ROTATE_INC;
				}
				PrintPercentCompleteShort((*curPercentComplete));
			}
		}
		return;
	}

	/* Sort recursively */
	MergeSortReadsHelper(reads,
			readCounts,
			low,
			mid,
			startLow, 
			total,
			showPercentComplete,
			curPercentComplete,
			tmpDir,
			readLength);
	MergeSortReadsHelper(reads,
			readCounts,
			mid+1,
			high,
			startLow, 
			total,
			showPercentComplete,
			curPercentComplete,
			tmpDir,
			readLength);

	/* Merge the two lists */
	MergeHelper(reads,
			readCounts,
			low,
			mid,
			high,
			tmpDir,
			readLength);
}

void MergeHelper(char **reads,
		int64_t *readCounts,
		int64_t low,
		int64_t mid,
		int64_t high,
		char *tmpDir,
		int readLength)
{
	char *FnName = "MergeHelper";
	int64_t i=0;
	char **tmpReads=NULL;
	int64_t *tmpReadCounts=NULL;
	int64_t startUpper, startLower, endUpper, endLower;
	int64_t ctr=0;
	FILE *tmpLowerFP=NULL;
	FILE *tmpUpperFP=NULL;
	char *tmpLowerFileName=NULL;
	char *tmpUpperFileName=NULL;
	char tmpLowerRead[SEQUENCE_LENGTH]="\0";
	char tmpUpperRead[SEQUENCE_LENGTH]="\0";
	long long int tmpLowerReadCount=0;
	long long int tmpUpperReadCount=0;
	int eofLower, eofUpper;

	/* Merge the two lists */
	/* Since we want to keep space requirement small, use an upper bound on memory,
	 * so that we use tmp files when memory requirements become to large */
	if( (high-low+1)*(sizeof(int64_t) + sizeof(char*)) <= ONE_GIGABYTE) {

		/* Use memory */
		tmpReads = malloc(sizeof(char*)*(high-low+1));
		if(NULL == tmpReads) {
			PrintError(FnName, "tmpReads", "Could not allocate memory", Exit, MallocMemory);
		}
		tmpReadCounts = malloc(sizeof(int64_t)*(high-low+1));
		if(NULL == tmpReadCounts) {
			PrintError(FnName, "tmpReadCounts", "Could not allocate memory", Exit, MallocMemory);
		}

		/* Merge */
		startLower = low;
		endLower = mid;
		startUpper = mid+1;
		endUpper = high;
		ctr=0;
		while( (startLower <= endLower) && (startUpper <= endUpper) ) {
			if(strcmp(reads[startLower], reads[startUpper]) <= 0) {
				tmpReads[ctr] = reads[startLower];
				tmpReadCounts[ctr] = readCounts[startLower];
				startLower++;
			}
			else {
				tmpReads[ctr] = reads[startUpper];
				tmpReadCounts[ctr] = readCounts[startUpper];
				startUpper++;
			}
			ctr++;
		}
		while(startLower <= endLower) {
			tmpReads[ctr] = reads[startLower];
			tmpReadCounts[ctr] = readCounts[startLower];
			startLower++;
			ctr++;
		}
		while(startUpper <= endUpper) {
			tmpReads[ctr] = reads[startUpper];
			tmpReadCounts[ctr] = readCounts[startUpper];
			startUpper++;
			ctr++;
		}
		/* Copy back */
		for(i=low, ctr=0;
				i<=high;
				i++, ctr++) {
			reads[i] = tmpReads[ctr];
			readCounts[i] = tmpReadCounts[ctr];
		}

		/* Free memory */
		free(tmpReads);
		tmpReads=NULL;
		free(tmpReadCounts);
		tmpReadCounts=NULL;
	}
	else {
		/* Use tmp files */
		assert(sizeof(int64_t) == sizeof(long long int));

		/* Open tmp files */
		tmpLowerFP = OpenTmpFile(tmpDir, &tmpLowerFileName);
		tmpUpperFP = OpenTmpFile(tmpDir, &tmpUpperFileName);

		/* Print to tmp files */
		for(i=low;i<=mid;i++) {
			if(0 > fprintf(tmpLowerFP, "%s\t%lld\n", 
						reads[i],
						(long long int)readCounts[i])) { 
				PrintError(FnName, NULL, "Could not write to tmp lower file", Exit, WriteFileError);
			}
		}
		for(i=mid+1;i<=high;i++) {
			if(0 > fprintf(tmpUpperFP, "%s\t%lld\n",
						reads[i],
						(long long int)readCounts[i])) {
				PrintError(FnName, NULL, "Could not write to tmp upper file", Exit, WriteFileError);
			}
		}

		/* Move to beginning of the files */
		fseek(tmpLowerFP, 0 , SEEK_SET);
		fseek(tmpUpperFP, 0 , SEEK_SET);

		/* Merge tmp files back into index */
		/* Get first contig/pos */
		if(0 > fscanf(tmpLowerFP, "%s %lld\n",
					tmpLowerRead,
					&tmpLowerReadCount)) {
			PrintError(FnName, NULL, "Could not read in tmp lower", Exit, ReadFileError);
		}
		if(0 > fscanf(tmpUpperFP, "%s %lld\n",
					tmpUpperRead,
					&tmpUpperReadCount)) {
			PrintError(FnName, NULL, "Could not read in tmp upper", Exit, ReadFileError);
		}
		for(i=low, ctr=0, eofLower = 0, eofUpper = 0;
				i<=high &&
				eofLower == 0 &&
				eofUpper == 0;
				i++, ctr++) {
			if(strcmp(tmpLowerRead, tmpUpperRead) <= 0) {
				/* Copy lower */
				strcpy(reads[i], tmpLowerRead);
				readCounts[i] = tmpLowerReadCount;
				/* Get new tmpLower */
				if(0 > fscanf(tmpLowerFP, "%s %lld\n",
							tmpLowerRead,
							&tmpLowerReadCount)) {
					eofLower = 1;
				}
			}
			else {
				/* Copy upper */
				strcpy(reads[i], tmpUpperRead);
				readCounts[i] = tmpUpperReadCount;
				/* Get new tmpUpper */
				if(0 > fscanf(tmpUpperFP, "%s %lld\n",
							tmpUpperRead,
							&tmpUpperReadCount)) {
					eofUpper = 1;
				}
			}
		}
		while(eofLower != 1) {
			/* Copy lower */
			strcpy(reads[i], tmpLowerRead);
			readCounts[i] = tmpLowerReadCount;
			/* Get new tmpLower */
			if(0 > fscanf(tmpLowerFP, "%s %lld\n",
						tmpLowerRead,
						&tmpLowerReadCount)) {
				eofLower = 1;
			}
			i++;
			ctr++;
		}
		while(eofUpper != 1) {
			/* Copy upper */
			strcpy(reads[i], tmpUpperRead);
			readCounts[i] = tmpUpperReadCount;
			/* Get new tmpUpper */
			if(0 > fscanf(tmpUpperFP, "%s %lld\n",
						tmpUpperRead,
						&tmpUpperReadCount)) {
				eofUpper = 1;
			}
			i++;
			ctr++;
		}
		assert(ctr == (high - low + 1));
		assert(i == high + 1);

		/* Close tmp files */
		CloseTmpFile(&tmpLowerFP, &tmpLowerFileName);
		CloseTmpFile(&tmpUpperFP, &tmpUpperFileName);
	}
}

/* Get the matches for the contig/pos */
void GetMatchesFromContigPos(RGIndex *index,
		RGBinary *rg,
		uint32_t curContig,
		uint32_t curPos,
		int64_t *numForward,
		int64_t *numReverse, 
		char **read,
		char **reverseRead)
{
	char *FnName = "GetMatchesFromContigPos";
	int returnLength, returnPosition;
	RGRanges ranges;
	RGMatch match;
	int readLength = index->width;
	int32_t i;
	int8_t readInt[SEQUENCE_LENGTH];

	/* Initialize */
	RGRangesInitialize(&ranges);
	RGMatchInitialize(&match);

	/* Get the read */
	RGBinaryGetReference(rg,
			curContig,
			curPos,
			FORWARD,
			0,
			read,
			readLength,
			&returnLength,
			&returnPosition);
	assert(returnLength == readLength);
	assert(returnPosition == curPos);
	RGBinaryGetReference(rg,
			curContig,
			curPos,
			REVERSE,
			0,
			reverseRead,
			readLength,
			&returnLength,
			&returnPosition);
	assert(returnLength == readLength);
	assert(returnPosition == curPos);

	ConvertSequenceToIntegers((*read), readInt, readLength);
	RGIndexGetRangesBothStrands(index,
			rg,
			readInt,
			readLength,
			0,
			INT_MAX,
			INT_MAX,
			rg->space,
			BothStrands,
			&ranges);

	/* Transfer ranges to matches */
	RGRangesCopyToRGMatch(&ranges,
			index,
			&match,
			rg->space,
			0);

	/* Remove duplicates */
	RGMatchRemoveDuplicates(&match,
			INT_MAX);

	assert(0 < ranges.numEntries);
	(*numForward) = (*numReverse) = 0;
	for(i=0;i<match.numEntries;i++) {
		switch(match.strands[i]) {
			case FORWARD:
				(*numForward)++;
				break;
			case REVERSE:
				(*numReverse)++;
				break;
			default:
				PrintError(FnName, NULL, "Could not understand strand", Exit, OutOfRange);
				break;
		}
	}

	assert((*numForward)>0);
	assert((*numReverse) >= 0);

	if(ColorSpace == rg->space) {
		ConvertColorsFromStorage((*read), readLength);
		ConvertColorsFromStorage((*reverseRead), readLength);
	}

	RGRangesFree(&ranges);
	RGMatchFree(&match);
}
