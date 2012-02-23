#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>
#include <string.h>
#include <pthread.h>
#include <config.h>
#include <zlib.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "BLibDefinitions.h"
#include "BError.h"
#include "BLib.h"
#include "RGBinary.h"
#include "RGRanges.h"
#include "RGIndexExons.h"
#include "RGIndex.h"

/* TODO */
void RGIndexCreate(char *fastaFileName,
		RGIndexLayout *layout, 
		int32_t space,
		int32_t indexNumber,
		int32_t startContig,
		int32_t startPos,
		int32_t endContig,
		int32_t endPos,
		int32_t useExons,
		RGIndexExons *e,
		int32_t numThreads,
		int32_t repeatMasker,
		int32_t includeNs,
		char *tmpDir) 
{

	/* The sort will take care of most of the work.  We just want 
	 * to make sure that we only include sequences that agree with
	 * repeatMasker and includeNs
	 * */
	//char *FnName = "RGIndexCreate";
	int32_t numFiles = pow(ALPHABET_SIZE, layout->depth);

	if(1 == numFiles) {
		/* Create like normal */
		RGIndexCreateSingle(fastaFileName,
				layout,
				space,
				indexNumber,
				startContig,
				startPos,
				endContig,
				endPos,
				useExons,
				e,
				numThreads,
				repeatMasker,
				includeNs,
				tmpDir);
	}
	else {
		/* Bin and create from each bin ... */
		RGIndexCreateSplit(fastaFileName,
				layout,
				space,
				indexNumber,
				startContig,
				startPos,
				endContig,
				endPos,
				useExons,
				e,
				numThreads,
				repeatMasker,
				includeNs,
				tmpDir);
	}
}

/* TODO */
void RGIndexCreateSingle(char *fastaFileName,
		RGIndexLayout *layout, 
		int32_t space,
		int32_t indexNumber,
		int32_t startContig,
		int32_t startPos,
		int32_t endContig,
		int32_t endPos,
		int32_t useExons,
		RGIndexExons *e,
		int32_t numThreads,
		int32_t repeatMasker,
		int32_t includeNs,
		char *tmpDir) 
{
	//char *FnName = "RGIndexCreateSingle";
	int64_t i;
	RGIndex index;
	RGBinary rg;
	gzFile gzOut;

	/* Get brg */
	RGBinaryReadBinary(&rg, space, fastaFileName);
	/* Make sure we have the correct reference genome */
	assert(rg.space == space);
	assert(4 == ALPHABET_SIZE);

	if(VERBOSE >=0) {
		fprintf(stderr, "Creating the index...\n");
	}

	/* Adjust bounds */
	AdjustBounds(&rg,
			&startContig,
			&startPos,
			&endContig,
			&endPos);

	/* Initialize the index */
	RGIndexInitializeFull(&index,
			&rg,
			layout,
			space,
			1,
			indexNumber,
			startContig,
			startPos,
			endContig,
			endPos,
			repeatMasker);

	/* Open output file before creation, so that if it exists we
	 * know before all the work is performed. */
	gzOut = RGIndexOpenForWriting(fastaFileName, &index); 

	/* Add locations to the index */
	if(VERBOSE >= 0) {
		fprintf(stderr, "Currently on [contig,pos]:\n");
		PrintContigPos(stderr,
				0,
				0);
	}
	if(UseExons == useExons) { /* Use only bases within the exons */
		for(i=0;i<e->numExons;i++) { /* For each exon */
			RGIndexCreateHelper(&index,
					&rg,
					NULL,
					e->exons[i].startContig,
					e->exons[i].startPos,
					e->exons[i].endContig,
					e->exons[i].endPos,
					repeatMasker,
					includeNs);
		}
	}
	else {
		RGIndexCreateHelper(&index,
				&rg,
				NULL,
				startContig,
				startPos,
				endContig,
				endPos,
				repeatMasker,
				includeNs);
	}

	if(VERBOSE >= 0) {
		PrintContigPos(stderr, 
				index.endContig,
				index.endPos);
		fprintf(stderr, "\n");
	}

	assert(index.length > 0);

	/* Sort the nodes in the index */
	RGIndexSort(&index, &rg, numThreads, tmpDir);

	/* Create hash table from the index */
	RGIndexCreateHash(&index, &rg);

	/* Write */ 
	RGIndexPrint(gzOut, &index);

	if(VERBOSE >= 0) {
		fprintf(stderr, "Index created.\n");
		fprintf(stderr, "Index size is %.3lfGB.\n",
				RGIndexGetSize(&index, GIGABYTES));
	}

	/* TODO: output Messages */
	/* Free memory */
	RGIndexDelete(&index);
	RGBinaryDelete(&rg);
}

/* TODO */
void RGIndexCreateSplit(char *fastaFileName,
		RGIndexLayout *layout, 
		int32_t space,
		int32_t indexNumber,
		int32_t startContig,
		int32_t startPos,
		int32_t endContig,
		int32_t endPos,
		int32_t useExons,
		RGIndexExons *e,
		int32_t numThreads,
		int32_t repeatMasker,
		int32_t includeNs,
		char *tmpDir) 
{
	char *FnName = "RGIndexCreateSplit";
	int64_t i;
	RGIndex index;
	RGBinary rg;
	int32_t numFiles = pow(ALPHABET_SIZE, layout->depth);
	FILE **tmpFPs=NULL;
	char **tmpFileNames=NULL;
	uint8_t contig_8;
	uint32_t contig_32;
	uint32_t position;
	gzFile *gzOuts=NULL;

	/* Get brg */
	RGBinaryReadBinary(&rg, space, fastaFileName);
	/* Make sure we have the correct reference genome */
	assert(rg.space == space);

	assert(4 == ALPHABET_SIZE);

	if(VERBOSE >=0) {
		fprintf(stderr, "Creating the index...\n");
	}

	/* Adjust bounds */
	AdjustBounds(&rg,
			&startContig,
			&startPos,
			&endContig,
			&endPos);

	gzOuts = malloc(sizeof(gzFile)*numFiles);
	if(NULL == gzOuts) {
		PrintError(FnName, "gzOuts", "Could not allocate memory", Exit, MallocMemory);	
	}	
	/* Open output file before creation, so that if it exists we	 
	 * * know before all the work is performed. */
	for(i=0;i<numFiles;i++) {
		/* Initialize */
		RGIndexInitializeFull(&index,
				&rg,
				layout,
				space,
				i+1,
				indexNumber,
				startContig,
				startPos,
				endContig,
				endPos,
				repeatMasker);
		/* Open file */
		gzOuts[i] = RGIndexOpenForWriting(fastaFileName, &index); 
		/* Delete */
		RGIndexDelete(&index);
	}

	/* Bin and create from each bin ... */
	tmpFPs = malloc(sizeof(FILE*)*numFiles);
	if(NULL == tmpFPs) {
		PrintError(FnName, "tmpFPs", "Could not allocate memory", Exit, MallocMemory);	
	}	
	tmpFileNames = malloc(sizeof(char*)*numFiles);	
	if(NULL == tmpFileNames) {		
		PrintError(FnName, "tmpFileNames", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Open tmp files */
	for(i=0;i<numFiles;i++) {
		tmpFPs[i] = OpenTmpFile(tmpDir, &tmpFileNames[i]);
	}

	/* Create dummy index for binning */
	RGIndexInitializeFull(&index,
			&rg,
			layout,
			space,
			1,
			indexNumber,
			startContig,
			startPos,
			endContig,
			endPos,
			repeatMasker);

	/* Create bins */
	if(UseExons == useExons) { /* Use only bases within the exons */
		for(i=0;i<e->numExons;i++) { /* For each exon */
			RGIndexCreateHelper(&index,
					&rg,
					tmpFPs,
					e->exons[i].startContig,
					e->exons[i].startPos,
					e->exons[i].endContig,
					e->exons[i].endPos,
					repeatMasker,
					includeNs);
		}
	}
	else {
		RGIndexCreateHelper(&index,
				&rg,
				tmpFPs,
				startContig,
				startPos,
				endContig,
				endPos,
				repeatMasker,
				includeNs);
	}
	if(VERBOSE >= 0) {
		PrintContigPos(stderr, 
				endContig,
				endPos);
		fprintf(stderr, "\n");
	}

	/* Delete dummy index */
	RGIndexDelete(&index);

	/* Process each bin */
	for(i=0;i<numFiles;i++) {
		if(VERBOSE >= 0) {
			fprintf(stderr, "%s", BREAK_LINE);
			fprintf(stderr, "Creating index (bin %d/%d)\n",
					(int)(i+1), numFiles);
		}
		/* Initialize */
		RGIndexInitializeFull(&index,
				&rg,
				layout,
				space,
				i+1,
				indexNumber,
				startContig,
				startPos,
				endContig,
				endPos,
				repeatMasker);

		/* Read in from tmp file */
		index.length=0;
		fseek(tmpFPs[i], 0, SEEK_SET);
		if(Contig_8 == index.contigType) {
			while(fread(&contig_8, sizeof(uint8_t), 1, tmpFPs[i]) == 1 &&
					fread(&position, sizeof(uint32_t), 1, tmpFPs[i]) == 1) {
				/* Insert */
				index.length++;

				/* Reallocate memory */
				index.positions = realloc(index.positions, sizeof(uint32_t)*index.length);
				if(NULL == index.positions) {
					PrintError("RGBinaryCreate", "index.positions", "Could not reallocate memory", Exit, ReallocMemory);
				}
				index.contigs_8 = realloc(index.contigs_8, sizeof(uint8_t)*index.length);
				if(NULL == index.contigs_8) {
					PrintError("RGBinaryCreate", "index.contigs_8", "Could not reallocate memory", Exit, ReallocMemory);
				}

				/* Copy over */
				index.positions[index.length-1] = position;
				index.contigs_8[index.length-1] = contig_8;
			}
		}
		else {
			while(fread(&contig_32, sizeof(uint32_t), 1, tmpFPs[i]) == 1 &&
					fread(&position, sizeof(uint32_t), 1, tmpFPs[i]) == 1) {

				/* Insert */
				index.length++;

				/* Reallocate memory */
				index.positions = realloc(index.positions, sizeof(uint32_t)*index.length);
				if(NULL == index.positions) {
					PrintError("RGBinaryCreate", "index.positions", "Could not reallocate memory", Exit, ReallocMemory);
				}
				index.contigs_32 = realloc(index.contigs_32, sizeof(uint32_t)*index.length);
				if(NULL == index.contigs_32) {
					PrintError("RGBinaryCreate", "index.contigs_32", "Could not reallocate memory", Exit, ReallocMemory);
				}

				/* Copy over */
				index.positions[index.length-1] = position;
				index.contigs_32[index.length-1] = contig_32;
			}
		}

		/* Sort the nodes in the index */
		RGIndexSort(&index, &rg, numThreads, tmpDir);

		/* Create hash table from the index */
		RGIndexCreateHash(&index, &rg);

		/* Write */
		RGIndexPrint(gzOuts[i], &index);
		/* TODO: output Messages */

		if(VERBOSE >= 0) {
			fprintf(stderr, "Index created.\n");
			fprintf(stderr, "Index size is %.3lfGB.\n",
					RGIndexGetSize(&index, GIGABYTES));
		}

		/* Free memory */
		RGIndexDelete(&index);

		/* Close the tmp file */
		CloseTmpFile(&tmpFPs[i], &tmpFileNames[i]);
	}

	if(VERBOSE >= 0) {
		fprintf(stderr, "%s", BREAK_LINE);
	}

	/* Free memory */
	free(gzOuts);
	free(tmpFPs);
	free(tmpFileNames);
	RGBinaryDelete(&rg);
}

/* TODO */
/* Two functions:
 * 1. Create an index without binning.
 * 2. Bin the contig/pos.
 * */
void RGIndexCreateHelper(RGIndex *index,
		RGBinary *rg,
		FILE **tmpFPs,
		int32_t startContig,
		int32_t startPos,
		int32_t endContig,
		int32_t endPos,
		int32_t repeatMasker,
		int32_t includeNs)
{
	/* For storing the bases */
	char *FnName="RGIndexCreateHelper";
	char bases[SEQUENCE_LENGTH]="\0";
	int32_t basesLength=0;
	int32_t basesIndex=0;
	int32_t curBasesPos=0; 
	int32_t toInsert=1;
	int32_t curPos=-1;
	int32_t curStartPos=-1;
	int32_t curEndPos=-1;
	uint32_t curContig=0;
	uint8_t curContig_8;
	uint32_t keyStartPos, keyEndPos;
	int64_t i;
	int32_t toBin=0;
	int32_t binNumber=0;
	int32_t curDepth=0;

	if(0 == index->depth) {
		toBin=0;
		assert(NULL == tmpFPs);
	}
	else {
		toBin=1;
		assert(NULL != tmpFPs);
	}
	assert(NULL != index);

	/* For each contig */
	for(curContig=startContig;
			curContig <= endContig;
			curContig++) {
		/* Update start and end bounds for this contig */
		curStartPos = (curContig==startContig)?startPos:1;
		curEndPos = (curContig==endContig)?endPos:(rg->contigs[curContig-1].sequenceLength);

		/* Initialize variables */
		basesLength = 0; /* Have not looked at any bases */
		basesIndex = 0; 

		/* For each position */
		for(curPos=curStartPos;curPos<=curEndPos;curPos++) {
			if(VERBOSE >= 0) {
				if(curPos%RGINDEX_ROTATE_NUM==0) {
					PrintContigPos(stderr, 
							curContig,
							curPos);
				}
			}

			/* Get the current base and insert into bases */
			basesLength++;
			bases[basesIndex] = RGBinaryGetBase(rg,
					curContig,
					curPos);
			/* Update where to put the next base */
			basesIndex = (basesIndex+1)%index->width;

			/* Check if we have enough bases */
			if(basesLength < index->width) {
				/* Do nothing since we do not have enough bases */
			}
			else {
				basesLength=index->width;

				/* Find the starting position, this is equal to the current position since period is the same as the total length */
				curBasesPos = basesIndex;
				toInsert = 1;

				keyStartPos = curPos - index->width + 1;
				keyEndPos = curPos;

				for(i=curDepth=0;i<index->width && 1==toInsert;i++) { /* For each base in the mask */
					if(1==index->mask[i]) {
						if(1==repeatMasker && 1==RGBinaryIsBaseRepeat(bases[curBasesPos])) {
							/* Did not pass */
							toInsert = 0;
						}
						else if(0==includeNs && 1==RGBinaryIsBaseN(bases[curBasesPos])) {
							/* Did not pass */
							toInsert = 0;
						}
					}
					/* Update position in bases */
					curBasesPos = (curBasesPos+1)%index->width;
				}

				/* See if we should insert into the index.  We should have enough consecutive bases. */
				if(1==toInsert) {
					if(0 == toBin) {
						/* Insert */
						index->length++;

						/* Reallocate memory */
						/* Copy over.  Remember that we are at the end of the read. */
						index->positions = realloc(index->positions, sizeof(uint32_t)*index->length);
						if(NULL == index->positions) {
							PrintError("RGBinaryCreate", "index->positions", "Could not reallocate memory", Exit, ReallocMemory);
						}
						index->positions[index->length-1] = keyStartPos;
						/* Reallocate memory for the contigs based on contig type and copy over. */
						if(index->contigType == Contig_8) {
							index->contigs_8 = realloc(index->contigs_8, sizeof(uint8_t)*index->length);
							if(NULL == index->contigs_8) {
								PrintError("RGBinaryCreate", "index->contigs_8", "Could not reallocate memory", Exit, ReallocMemory);
							}
							index->contigs_8[index->length-1] = curContig;
						}
						else {
							index->contigs_32 = realloc(index->contigs_32, sizeof(uint32_t)*index->length);
							if(NULL == index->contigs_32) {
								PrintError("RGBinaryCreate", "index->contigs_32", "Could not reallocate memory", Exit, ReallocMemory);
							}
							index->contigs_32[index->length-1] = curContig;
						}
					}
					else {
						/* Get bin number */
						curBasesPos = basesIndex;
						keyStartPos = curPos - index->width + 1;
						keyEndPos = curPos;
						binNumber=0;
						for(i=curDepth=0;i<index->width && curDepth < index->depth;i++) { 
							if(1==index->mask[i]) {
								binNumber = binNumber << 2; /* Only works with a four letter alphabet */
								switch(tolower(bases[curBasesPos])) {
									case 'a':
										break;
									case 'c':
										binNumber += 1; 
										break;
									case 'g':
										binNumber += 2; 
										break;
									case 't':
										binNumber += 3;
										break;
									default:
										PrintError(FnName, NULL, "Unrecognized base while binning", Exit, OutOfRange);
										break;
								}
								curDepth++;
							}
							/* Update position in bases */
							curBasesPos = (curBasesPos+1)%index->width;
						}

						/* Print to the Bin */
						if(index->contigType == Contig_8) {
							curContig_8 = (uint8_t)curContig;
							if(fwrite(&curContig_8, sizeof(uint8_t), 1, tmpFPs[binNumber]) != 1 ||
									fwrite(&keyStartPos, sizeof(uint32_t), 1, tmpFPs[binNumber]) != 1) {
								PrintError(FnName, "curContig8 and keyStartPos", "Could not write to file", Exit, WriteFileError);							
							}						
						}						
						else {							
							if(fwrite(&curContig, sizeof(uint32_t), 1, tmpFPs[binNumber]) != 1 ||
									fwrite(&keyStartPos, sizeof(uint32_t), 1, tmpFPs[binNumber]) != 1) {
								PrintError(FnName, "curContig and keyStartPos", "Could not write to file", Exit, WriteFileError);							}						}					}				}
			}
		}
		if(VERBOSE >= 0) {
			PrintContigPos(stderr, 
					curContig,
					curPos);
		}
	}
}

/* TODO */
void RGIndexCreateHash(RGIndex *index, RGBinary *rg)
{
	char *FnName = "RGIndexCreateHash";
	uint32_t curHash, prevHash, prevStart;
	int64_t i;

	if(index->length >= UINT_MAX) {
		PrintError(FnName, "index->length", "Index length has reached its maximum", Exit, OutOfRange);
	}

	/* Allocate memory for the hash */
	index->starts = malloc(sizeof(uint32_t)*index->hashLength);
	if(NULL==index->starts) {
		PrintError(FnName, "index->starts", "Could not allocate memory", Exit, MallocMemory);
	}

	/* initialize */
	for(i=0;i<index->hashLength;i++) {
		/* Can't use -1, so use UINT_MAX */
		index->starts[i] = UINT_MAX;
	}

	/* Go through index and update the hash */
	if(VERBOSE >= 0) {
		fprintf(stderr, "Creating a hash.\nPass 1 out of 2.  Out of %u, currently on:\n0",
				(uint32_t)index->length);
	}

	prevHash = UINT_MAX;
	for(i=0;i<index->length;i++) {
		if(VERBOSE >= 0 && i%RGINDEX_ROTATE_NUM==0) {
			fprintf(stderr, "\r%lld", 
					(long long int)i);
		}

		curHash = RGIndexGetHashIndex(index, rg, i, 0);
		if(prevHash == curHash) {
			/* Ignore */
		}
		else {
			/* Update */
			assert(i < UINT_MAX);
			assert(0 <= curHash && curHash < index->hashLength);
			index->starts[curHash] = i;
			prevHash = curHash;
		}
	}
	/* Test pass 1 has creation */
	/*
	   for(i=0;i<index->hashLength;i++) {
	   assert(0 <= index->starts[i]);
	   assert(index->starts[i] < index->length || index->starts[i] == UINT_MAX);
	   }
	   */

	if(VERBOSE >= 0) {
		fprintf(stderr, "\r%lld\n", 
				(long long int)i);
		fprintf(stderr, "Pass 2 of 2.  Out of %lld, currently on:\n0",
				(long long int)index->hashLength);
	}

	/* Go through hash and reset all UINT_MAX starts */
	for(i=index->hashLength-1, prevStart=UINT_MAX;
			0<=i;
			i--) {
		if(VERBOSE >=0 && (index->hashLength-i)%RGINDEX_ROTATE_NUM == 0) {
			fprintf(stderr, "\r%lld", 
					(long long int)(index->hashLength-i));
		}
		if(UINT_MAX == index->starts[i]) {
			index->starts[i] = prevStart;
		}
		else {
			prevStart = index->starts[i];
		}
	}
	if(VERBOSE >=0) {
		fprintf(stderr, "\r%lld\n", 
				(long long int)(index->hashLength));
	}

	/* Test pass 2 has creation */
	/*
	   for(i=0;i<index->hashLength;i++) {
	   if(UINT_MAX == index->starts[i]) {
	   break;
	   }
	   assert(0 <= index->starts[i]);
	   assert(index->starts[i] < index->length);
	   }
	   while(i<index->hashLength) { // Last entries must be the maximum 
	   assert(index->starts[i] == UINT_MAX);
	   i++;
	   }
	   */
	if(VERBOSE >= 0) {
		fprintf(stderr, "\rHash created.\n");
	}
}

/* TODO */
void RGIndexSort(RGIndex *index, RGBinary *rg, int32_t numThreads, char* tmpDir)
{
	char *FnName = "RGIndexSort";
	int64_t i, j;
	ThreadRGIndexSortData *sortData=NULL;
	ThreadRGIndexMergeData *mergeData=NULL;
	pthread_t *threads=NULL;
	int32_t errCode;
	void *status=NULL;
	double curPercentComplete = 0.0;
	int32_t curNumThreads = numThreads;
	int32_t curMergeIteration, curThread;

	/* Only use threads if we want to divide and conquer */
	if(numThreads > 1) {
		if(VERBOSE >= 0) {
			fprintf(stderr, "Sorting by thread...\n");
		}
		/* Should check that the number of threads is a power of 4 since we split
		 * in half in both sorts. */
		assert(IsAPowerOfTwo(numThreads)==1);

		/* Allocate memory for the thread arguments */
		sortData = malloc(sizeof(ThreadRGIndexSortData)*numThreads);
		if(NULL==sortData) {
			PrintError(FnName, "sortData", "Could not allocate memory", Exit, MallocMemory);
		}
		/* Allocate memory for the thread point32_ters */
		threads = malloc(sizeof(pthread_t)*numThreads);
		if(NULL==threads) {
			PrintError(FnName, "threads", "Could not allocate memory", Exit, MallocMemory);
		}

		/* Merge sort with tmp file I/O */

		/* Initialize sortData */
		for(i=0;i<numThreads;i++) {
			sortData[i].index = index;
			sortData[i].rg = rg;
			sortData[i].threadID = i;
			sortData[i].low = i*(index->length/numThreads);
			sortData[i].high = (i+1)*(index->length/numThreads)-1;
			sortData[i].showPercentComplete = 0;
			sortData[i].tmpDir = tmpDir;
			/* Divide the maximum overhead by the number of threads */
			sortData[i].mergeMemoryLimit = MERGE_MEMORY_LIMIT/((int64_t)numThreads); 
			assert(sortData[i].low >= 0 && sortData[i].high < index->length);
		}
		sortData[0].low = 0;
		sortData[numThreads-1].high = index->length-1;
		sortData[numThreads-1].showPercentComplete = 1;

		/* Check that we split correctly */
		for(i=1;i<numThreads;i++) {
			assert(sortData[i-1].high < sortData[i].low);
		}

		/* Create threads */
		for(i=0;i<numThreads;i++) {
			/* Start thread */
			errCode = pthread_create(&threads[i], /* thread struct */
					NULL, /* default thread attributes */
					RGIndexMergeSort, /* start routine */
					(void*)(&sortData[i])); /* sortData to routine */
			if(0!=errCode) {
				PrintError(FnName, "pthread_create: errCode", "Could not start thread", Exit, ThreadError);
			}
		}

		/* Wait for the threads to finish */
		for(i=0;i<numThreads;i++) {
			/* Wait for the given thread to return */
			errCode = pthread_join(threads[i],
					&status);
			/* Check the return code of the thread */
			if(0!=errCode) {
				PrintError(FnName, "pthread_join: errCode", "Thread returned an error", Exit, ThreadError);
			}
			if(i==numThreads-1 && VERBOSE >= 0) {
				fprintf(stderr, "\rWaiting for other threads to complete...");
			}
		}

		/* Free memory for the threads */
		free(threads);
		threads=NULL;

		/* Now we must merge the results from the threads */
		/* Merge intelligently i.e. merge recursively so 
		 * there are only nlogn merges where n is the 
		 * number of threads. */
		curNumThreads = numThreads;
		if(VERBOSE >= 0) {
			fprintf(stderr, "\rMerging sorts from threads...                          \n");
			fprintf(stderr, "Out of %d required merges, currently on:\n0", Log2(numThreads));
		}
		for(i=1, curMergeIteration=1;i<numThreads;i=i*2, curMergeIteration++) { /* The number of merge iterations */
			if(VERBOSE >= 0) {
				fprintf(stderr, "\r%d", curMergeIteration);
			}
			curNumThreads /= 2; /* The number of threads to spawn */
			/* Allocate memory for the thread arguments */
			mergeData = malloc(sizeof(ThreadRGIndexMergeData)*curNumThreads);
			if(NULL==mergeData) {
				PrintError(FnName, "mergeData", "Could not allocate memory", Exit, MallocMemory);
			}
			/* Allocate memory for the thread point32_ters */
			threads = malloc(sizeof(pthread_t)*curNumThreads);
			if(NULL==threads) {
				PrintError(FnName, "threads", "Could not allocate memory", Exit, MallocMemory);
			}
			/* Initialize data for threads */
			for(j=0,curThread=0;j<numThreads;j+=2*i,curThread++) {
				mergeData[curThread].index = index;
				mergeData[curThread].rg = rg;
				mergeData[curThread].threadID = curThread;
				/* Use the same bounds as was used in the sort */
				mergeData[curThread].low = sortData[j].low;
				mergeData[curThread].mid = sortData[i+j].low-1;
				mergeData[curThread].high = sortData[j+2*i-1].high;
				mergeData[curThread].mergeMemoryLimit = MERGE_MEMORY_LIMIT/((int64_t)curNumThreads); 
				mergeData[curThread].tmpDir = tmpDir;
			}
			/* Check that we split correctly */
			for(j=1;j<curNumThreads;j++) {
				if(mergeData[j-1].high >= mergeData[j].low) {
					PrintError(FnName, NULL, "mergeData[j-1].high >= mergeData[j].low", Exit, OutOfRange);
				}
			}

			/* Create threads */
			for(j=0;j<curNumThreads;j++) {
				/* Start thread */
				errCode = pthread_create(&threads[j], /* thread struct */
						NULL, /* default thread attributes */
						RGIndexMerge, /* start routine */
						(void*)(&mergeData[j])); /* sortData to routine */
				if(0!=errCode) {
					PrintError(FnName, "pthread_create: errCode", "Could not start thread", Exit, ThreadError);
				}
			}

			/* Wait for the threads to finish */
			for(j=0;j<curNumThreads;j++) {
				/* Wait for the given thread to return */
				errCode = pthread_join(threads[j],
						&status);
				/* Check the return code of the thread */
				if(0!=errCode) {
					PrintError(FnName, "pthread_join: errCode", "Thread returned an error", Exit, ThreadError);
				}
			}

			/* Free memory for the merge data */
			free(mergeData);
			mergeData=NULL;
			/* Free memory for the threads */
			free(threads);
			threads=NULL;
		}
		if(VERBOSE >= 0) {
			fprintf(stderr, "\nMerge complete.\n");
		}

		/* Free memory for sort data */
		free(sortData);
		sortData=NULL;
	}
	else {
		if(VERBOSE >= 0) {
			fprintf(stderr, "Sorting...\n");
		}
		if(VERBOSE >= 0) {
			fprintf(stderr, "\r0 percent complete");
		}
		RGIndexMergeSortHelper(index,
				rg,
				0,
				index->length-1,
				1,
				&curPercentComplete,
				0,
				index->length-1,
				MERGE_MEMORY_LIMIT,
				tmpDir);
		if(VERBOSE >= 0) {
			fprintf(stderr, "\r100.00 percent complete\n");
		}
	}

	/* Test that we sorted correctly */
	/*
	if(1 == TEST_RGINDEX_SORT) {
		for(i=1;i<index->length;i++) {
			if(0 < RGIndexCompareAt(index, rg, i-1, i, 0)) {
				RGIndexCompareAt(index, rg, i-1, i, 1);
			}
			assert(RGIndexCompareAt(index, rg, i-1, i, 0) <= 0);
		}
	}
	*/
	if(VERBOSE >= 0) {
		fprintf(stderr, "Sorted.\n");
	}
}

/* TODO */
void *RGIndexMergeSort(void *arg)
{
	/* thread arguments */
	ThreadRGIndexSortData *data = (ThreadRGIndexSortData*)(arg);
	double curPercentComplete = 0.0;

	/* Call helper */
	if(data->showPercentComplete == 1 && VERBOSE >= 0) {
		fprintf(stderr, "\r%3.3lf percent complete", 0.0);
	}
	RGIndexMergeSortHelper(data->index,
			data->rg,
			data->low,
			data->high,
			data->showPercentComplete,
			&curPercentComplete,
			data->low,
			data->high - data->low,
			data->mergeMemoryLimit,
			data->tmpDir);
	if(data->showPercentComplete == 1 && VERBOSE >= 0) {
		fprintf(stderr, "\r");
		fprintf(stderr, "thread %3.3lf percent complete", 100.0);
	}

	return arg;
}

/* TODO */
void RGIndexShellSort(RGIndex *index,
		RGBinary *rg,
		int64_t low,
		int64_t high)
{
	int64_t i, j, inc;
	uint8_t tempContig_8;
	uint32_t tempContig_32, tempPosition;

	inc = ROUND((high - low + 1) / 2);

	if(Contig_8 == index->contigType) {
		while(0 < inc) {
			for(i=inc + low;i<=high;i++) {
				tempContig_8 = index->contigs_8[i]; 
				tempPosition = index->positions[i];
				j = i;            
				while(inc + low <= j 
						&& RGIndexCompareContigPos(index,
							rg,
							tempContig_8, 
							tempPosition, 
							index->contigs_8[j-inc], 
							index->positions[j-inc],
							0) < 0) {
					index->contigs_8[j] = index->contigs_8[j-inc];
					index->positions[j] = index->positions[j-inc];
					j -= inc;
				}
				index->contigs_8[j] = tempContig_8;
				index->positions[j] = tempPosition;
			}
			inc = ROUND(inc / SHELL_SORT_GAP_DIVIDE_BY);
		}
	}
	else {
		while(0 < inc) {
			for(i=inc + low;i<=high;i++) {
				tempContig_32 = index->contigs_32[i]; 
				tempPosition = index->positions[i];
				j = i;            
				while(inc + low <= j 
						&& RGIndexCompareContigPos(index,
							rg,
							tempContig_32, 
							tempPosition, 
							index->contigs_32[j-inc], 
							index->positions[j-inc],
							0) < 0) {
					index->contigs_32[j] = index->contigs_32[j-inc];
					index->positions[j] = index->positions[j-inc];
					j -= inc;
				}
				index->contigs_32[j] = tempContig_32;
				index->positions[j] = tempPosition;
			}
			inc = ROUND(inc / SHELL_SORT_GAP_DIVIDE_BY);
		}
	}
}

/* TODO */
/* Call stack was getting too big, implement non-recursive sort */
void RGIndexMergeSortHelper(RGIndex *index,
		RGBinary *rg,
		int64_t low,
		int64_t high,
		int32_t showPercentComplete,
		double *curPercentComplete,
		int64_t startLow,
		int64_t total,
		int64_t mergeMemoryLimit,
		char *tmpDir)
{
	/* Local Variables */
	int64_t mid = (low + high)/2;

	if(high <= low) {
		if(VERBOSE >= 0 &&
				showPercentComplete == 1) {
			assert(NULL!=curPercentComplete);
			if((*curPercentComplete) < 100.0*((double)(low - startLow))/total) {
				while((*curPercentComplete) < 100.0*((double)(low - startLow))/total) {
					(*curPercentComplete) += RGINDEX_SORT_ROTATE_INC;
				}
				PrintPercentCompleteLong((*curPercentComplete));
			}
		}
		return;
	}

	if(high - low + 1 <= RGINDEX_SHELL_SORT_MAX) {
		RGIndexShellSort(index, rg, low, high);

		if(VERBOSE >= 0 &&
				showPercentComplete == 1) {
			assert(NULL!=curPercentComplete);
			if((*curPercentComplete) < 100.0*((double)(low - startLow))/total) {
				while((*curPercentComplete) < 100.0*((double)(low - startLow))/total) {
					(*curPercentComplete) += RGINDEX_SORT_ROTATE_INC;
				}
				PrintPercentCompleteLong((*curPercentComplete));
			}
		}
		return;
	}

	/* Partition the list into two lists and sort them recursively */
	RGIndexMergeSortHelper(index,
			rg,
			low,
			mid,
			showPercentComplete,
			curPercentComplete,
			startLow,
			total,
			mergeMemoryLimit,
			tmpDir);
	RGIndexMergeSortHelper(index,
			rg,
			mid+1,
			high,
			showPercentComplete,
			curPercentComplete,
			startLow,
			total,
			mergeMemoryLimit,
			tmpDir);

	/* Merge the two lists */
	RGIndexMergeHelper(index,
			rg,
			low,
			mid,
			high,
			mergeMemoryLimit,
			tmpDir);
}

/* TODO */
void *RGIndexMerge(void *arg)
{
	ThreadRGIndexMergeData *data = (ThreadRGIndexMergeData*)arg;

	/* Merge the data */
	RGIndexMergeHelper(data->index,
			data->rg,
			data->low,
			data->mid,
			data->high,
			data->mergeMemoryLimit,
			data->tmpDir);

	return arg;
}

/* TODO */
void RGIndexMergeHelper(RGIndex *index,
		RGBinary *rg,
		int64_t low,
		int64_t mid,
		int64_t high,
		int64_t mergeMemoryLimit, /* In bytes */
		char *tmpDir)
{
	/*
	   char *FnName = "RGIndexMergeHelper";
	   */

	/* Merge the two lists */
	/* Since we want to keep space requirement small, use an upper bound on memory,
	 * so that we use tmp files when memory requirements become to large */
	if(index->contigType == Contig_8) {
		if((high-low+1)*(sizeof(uint32_t) + sizeof(uint8_t)) <= mergeMemoryLimit) {
			/* Use memory */
			RGIndexMergeHelperInMemoryContig_8(index, rg, low, mid, high);
		}
		else {
			/* Use tmp files */
			RGIndexMergeHelperFromDiskContig_8(index, rg, low, mid, high, tmpDir);
		}
	}
	else {
		if((high-low+1)*(sizeof(uint32_t) + sizeof(uint32_t)) <= mergeMemoryLimit) {
			RGIndexMergeHelperInMemoryContig_32(index, rg, low, mid, high);
		}
		else {
			/* Use tmp files */
			RGIndexMergeHelperFromDiskContig_32(index, rg, low, mid, high, tmpDir);
		}
	}
	/* Test merge */
	/*
	   for(i=low+1;i<=high;i++) {
	   assert(RGIndexCompareAt(index, rg, i-1, i, 0) <= 0);
	   }
	   */
}

/* TODO */
void RGIndexMergeHelperInMemoryContig_8(RGIndex *index,
		RGBinary *rg,
		int64_t low,
		int64_t mid,
		int64_t high)
{
	char *FnName = "RGIndexMergeHelperInMemoryContig_8";
	int64_t i=0;
	uint32_t *tmpPositions=NULL;
	uint8_t *tmpContigs_8=NULL;
	int64_t startUpper, startLower, endUpper, endLower;
	int64_t ctr=0;

	assert(index->contigType == Contig_8);
	assert(index->contigs_8 != NULL);

	/* Merge the two lists using memory */

	/* Use memory */
	tmpPositions = malloc(sizeof(uint32_t)*(high-low+1));
	if(NULL == tmpPositions) {
		PrintError(FnName, "tmpPositions", "Could not allocate memory", Exit, MallocMemory);
	}
	tmpContigs_8 = malloc(sizeof(uint8_t)*(high-low+1));
	if(NULL == tmpContigs_8) {
		PrintError(FnName, "tmpContigs_8", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Merge */
	startLower = low;
	endLower = mid;
	startUpper = mid+1;
	endUpper = high;
	ctr=0;
	while( (startLower <= endLower) && (startUpper <= endUpper) ) {
		if(RGIndexCompareAt(index, rg, startLower, startUpper, 0) <= 0) {
			tmpPositions[ctr] = index->positions[startLower];
			tmpContigs_8[ctr] = index->contigs_8[startLower];
			startLower++;
		}
		else {
			tmpPositions[ctr] = index->positions[startUpper];
			tmpContigs_8[ctr] = index->contigs_8[startUpper];
			startUpper++;
		}
		ctr++;
	}
	while(startLower <= endLower) {
		tmpPositions[ctr] = index->positions[startLower];
		tmpContigs_8[ctr] = index->contigs_8[startLower];
		startLower++;
		ctr++;
	}
	while(startUpper <= endUpper) {
		tmpPositions[ctr] = index->positions[startUpper];
		tmpContigs_8[ctr] = index->contigs_8[startUpper];
		startUpper++;
		ctr++;
	}
	/* Copy back */
	for(i=low, ctr=0;
			i<=high;
			i++, ctr++) {
		index->positions[i] = tmpPositions[ctr];
		index->contigs_8[i] = tmpContigs_8[ctr];
	}

	/* Free memory */
	free(tmpPositions);
	tmpPositions=NULL;
	free(tmpContigs_8);
	tmpContigs_8=NULL;
}

/* TODO */
void RGIndexMergeHelperInMemoryContig_32(RGIndex *index,
		RGBinary *rg,
		int64_t low,
		int64_t mid,
		int64_t high)
{
	char *FnName = "RGIndexMergeHelperInMemoryContig_32";
	int64_t i=0;
	uint32_t *tmpPositions=NULL;
	uint32_t *tmpContigs_32=NULL;
	int64_t startUpper, startLower, endUpper, endLower;
	int64_t ctr=0;

	assert(index->contigType == Contig_32);
	assert(index->contigs_32 != NULL);

	/* Merge the two lists using memory */

	/* Use memory */
	tmpPositions = malloc(sizeof(uint32_t)*(high-low+1));
	if(NULL == tmpPositions) {
		PrintError(FnName, "tmpPositions", "Could not allocate memory", Exit, MallocMemory);
	}
	tmpContigs_32 = malloc(sizeof(uint32_t)*(high-low+1));
	if(NULL == tmpContigs_32) {
		PrintError(FnName, "tmpContigs_32", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Merge */
	startLower = low;
	endLower = mid;
	startUpper = mid+1;
	endUpper = high;
	ctr=0;
	while( (startLower <= endLower) && (startUpper <= endUpper) ) {
		if(RGIndexCompareAt(index, rg, startLower, startUpper, 0) <= 0) {
			tmpPositions[ctr] = index->positions[startLower];
			tmpContigs_32[ctr] = index->contigs_32[startLower];
			startLower++;
		}
		else {
			tmpPositions[ctr] = index->positions[startUpper];
			tmpContigs_32[ctr] = index->contigs_32[startUpper];
			startUpper++;
		}
		ctr++;
	}
	while(startLower <= endLower) {
		tmpPositions[ctr] = index->positions[startLower];
		tmpContigs_32[ctr] = index->contigs_32[startLower];
		startLower++;
		ctr++;
	}
	while(startUpper <= endUpper) {
		tmpPositions[ctr] = index->positions[startUpper];
		tmpContigs_32[ctr] = index->contigs_32[startUpper];
		startUpper++;
		ctr++;
	}
	/* Copy back */
	for(i=low, ctr=0;
			i<=high;
			i++, ctr++) {
		index->positions[i] = tmpPositions[ctr];
		index->contigs_32[i] = tmpContigs_32[ctr];
	}

	/* Free memory */
	free(tmpPositions);
	tmpPositions=NULL;
	free(tmpContigs_32);
	tmpContigs_32=NULL;
}

/* TODO */
void RGIndexMergeHelperFromDiskContig_8(RGIndex *index,
		RGBinary *rg,
		int64_t low,
		int64_t mid,
		int64_t high,
		char *tmpDir)
{
	char *FnName = "RGIndexMergeHelperFromDiskContig_8";
	int64_t i=0;
	int64_t ctr=0;
	FILE *tmpLowerFP=NULL;
	FILE *tmpUpperFP=NULL;
	char *tmpLowerFileName=NULL;
	char *tmpUpperFileName=NULL;
	uint32_t tmpLowerPosition=0;
	uint32_t tmpUpperPosition=0;
	uint8_t tmpLowerContig_8=0;
	uint8_t tmpUpperContig_8=0;

	assert(index->contigType == Contig_8);
	assert(index->contigs_8 != NULL);

	/* Merge the two lists */
	/* Since we want to keep space requirement small, use an upper bound on memory,
	 * so that we use tmp files when memory requirements become to large */
	/* Use tmp files */

	/* Open tmp files */
	tmpLowerFP = OpenTmpFile(tmpDir, &tmpLowerFileName);
	tmpUpperFP = OpenTmpFile(tmpDir, &tmpUpperFileName);

	/* Print to tmp files */
	for(i=low;i<=mid;i++) {
		if(1 != fwrite(&index->positions[i], sizeof(uint32_t), 1, tmpLowerFP)) {
			PrintError(FnName, "index->positions", "Could not write positions to tmp lower file", Exit, WriteFileError);
		}
		if(1 != fwrite(&index->contigs_8[i], sizeof(uint8_t), 1, tmpLowerFP)) {
			PrintError(FnName, "index->contigs_8", "Could not write contigs_8 to tmp lower file", Exit, WriteFileError);
		}
	}
	for(i=mid+1;i<=high;i++) {
		if(1 != fwrite(&index->positions[i], sizeof(uint32_t), 1, tmpUpperFP)) {
			PrintError(FnName, "index->positions", "Could not write positions to tmp upper file", Exit, WriteFileError);
		}
		if(1 != fwrite(&index->contigs_8[i], sizeof(uint8_t), 1, tmpUpperFP)) {
			PrintError(FnName, "index->contigs_8", "Could not write contigs_8 to tmp upper file", Exit, WriteFileError);
		}
	}

	/* Move to beginning of the files */
	fseek(tmpLowerFP, 0 , SEEK_SET);
	fseek(tmpUpperFP, 0 , SEEK_SET);

	/* Merge tmp files back into index */
	/* Get first contig/pos */

	if(1!=fread(&tmpLowerPosition, sizeof(uint32_t), 1, tmpLowerFP) ||
			1!=fread(&tmpLowerContig_8, sizeof(uint8_t), 1, tmpLowerFP)) {
		PrintError(FnName, NULL, "Could not read in tmp lower", Exit, ReadFileError);
	}
	if(1!=fread(&tmpUpperPosition, sizeof(uint32_t), 1, tmpUpperFP) ||
			1!=fread(&tmpUpperContig_8, sizeof(uint8_t), 1, tmpUpperFP)) {
		PrintError(FnName, NULL, "Could not read in tmp upper", Exit, ReadFileError);
	}

	for(i=low, ctr=0;
			i<=high &&
			tmpLowerPosition != 0 &&
			tmpUpperPosition != 0;
			i++, ctr++) {
		if(RGIndexCompareContigPos(index,
					rg,
					tmpLowerContig_8,
					tmpLowerPosition,
					tmpUpperContig_8,
					tmpUpperPosition,
					0)<=0) {
			/* Copy lower */
			index->positions[i] = tmpLowerPosition;
			index->contigs_8[i] = tmpLowerContig_8;
			/* Get new tmpLower */
			if(1!=fread(&tmpLowerPosition, sizeof(uint32_t), 1, tmpLowerFP) ||
					1!=fread(&tmpLowerContig_8, sizeof(uint8_t), 1, tmpLowerFP)) {
				tmpLowerPosition = 0;
				tmpLowerContig_8 = 0;
			}
		}
		else {
			/* Copy upper */
			index->positions[i] = tmpUpperPosition;
			index->contigs_8[i] = tmpUpperContig_8;
			/* Get new tmpUpper */
			if(1!=fread(&tmpUpperPosition, sizeof(uint32_t), 1, tmpUpperFP) ||
					1!=fread(&tmpUpperContig_8, sizeof(uint8_t), 1, tmpUpperFP)) {
				tmpUpperPosition = 0;
				tmpUpperContig_8 = 0;
			}
		}
	}
	while(tmpLowerPosition != 0 && tmpUpperPosition == 0) {
		/* Copy lower */
		index->positions[i] = tmpLowerPosition;
		index->contigs_8[i] = tmpLowerContig_8;
		/* Get new tmpLower */
		if(1!=fread(&tmpLowerPosition, sizeof(uint32_t), 1, tmpLowerFP) ||
				1!=fread(&tmpLowerContig_8, sizeof(uint8_t), 1, tmpLowerFP)) {
			tmpLowerPosition = 0;
			tmpLowerContig_8 = 0;
		}
		i++;
		ctr++;
	}
	while(tmpLowerPosition == 0 && tmpUpperPosition != 0) {
		/* Copy upper */
		index->positions[i] = tmpUpperPosition;
		index->contigs_8[i] = tmpUpperContig_8;
		/* Get new tmpUpper */
		if(1!=fread(&tmpUpperPosition, sizeof(uint32_t), 1, tmpUpperFP) ||
				1!=fread(&tmpUpperContig_8, sizeof(uint8_t), 1, tmpUpperFP)) {
			tmpUpperPosition = 0;
			tmpUpperContig_8 = 0;
		}
		i++;
		ctr++;
	}
	assert(ctr == (high - low + 1));
	assert(i == high + 1);

	/* Close tmp files */
	CloseTmpFile(&tmpLowerFP, &tmpLowerFileName);
	CloseTmpFile(&tmpUpperFP, &tmpUpperFileName);
	/* Test merge */
	/*
	   for(i=low+1;i<=high;i++) {
	   assert(RGIndexCompareAt(index, rg, i-1, i, 0) <= 0);
	   }
	   */
}

/* TODO */
void RGIndexMergeHelperFromDiskContig_32(RGIndex *index,
		RGBinary *rg,
		int64_t low,
		int64_t mid,
		int64_t high,
		char *tmpDir)
{
	char *FnName = "RGIndexMergeHelperFromDiskContig_32";
	int64_t i=0;
	int64_t ctr=0;
	FILE *tmpLowerFP=NULL;
	FILE *tmpUpperFP=NULL;
	char *tmpLowerFileName=NULL;
	char *tmpUpperFileName=NULL;
	uint32_t tmpLowerPosition=0;
	uint32_t tmpUpperPosition=0;
	uint32_t tmpLowerContig_32=0;
	uint32_t tmpUpperContig_32=0;

	assert(index->contigType == Contig_32);
	assert(index->contigs_32 != NULL);

	/* Merge the two lists */
	/* Since we want to keep space requirement small, use an upper bound on memory,
	 * so that we use tmp files when memory requirements become to large */
	/* Use tmp files */

	/* Open tmp files */
	tmpLowerFP = OpenTmpFile(tmpDir, &tmpLowerFileName);
	tmpUpperFP = OpenTmpFile(tmpDir, &tmpUpperFileName);

	/* Print to tmp files */
	for(i=low;i<=mid;i++) {
		if(1 != fwrite(&index->positions[i], sizeof(uint32_t), 1, tmpLowerFP)) {
			PrintError(FnName, "index->positions", "Could not write positions to tmp lower file", Exit, WriteFileError);
		}
		if(1 != fwrite(&index->contigs_32[i], sizeof(uint32_t), 1, tmpLowerFP)) {
			PrintError(FnName, "index->contigs_32", "Could not write contigs_32 to tmp lower file", Exit, WriteFileError);
		}
	}
	for(i=mid+1;i<=high;i++) {
		if(1 != fwrite(&index->positions[i], sizeof(uint32_t), 1, tmpUpperFP)) {
			PrintError(FnName, "index->positions", "Could not write positions to tmp upper file", Exit, WriteFileError);
		}
		if(1 != fwrite(&index->contigs_32[i], sizeof(uint32_t), 1, tmpUpperFP)) {
			PrintError(FnName, "index->contigs_32", "Could not write contigs_32 to tmp upper file", Exit, WriteFileError);
		}
	}

	/* Move to beginning of the files */
	fseek(tmpLowerFP, 0 , SEEK_SET);
	fseek(tmpUpperFP, 0 , SEEK_SET);

	/* Merge tmp files back into index */
	/* Get first contig/pos */

	if(1!=fread(&tmpLowerPosition, sizeof(uint32_t), 1, tmpLowerFP) ||
			1!=fread(&tmpLowerContig_32, sizeof(uint32_t), 1, tmpLowerFP)) {
		PrintError(FnName, NULL, "Could not read in tmp lower", Exit, ReadFileError);
	}
	if(1!=fread(&tmpUpperPosition, sizeof(uint32_t), 1, tmpUpperFP) ||
			1!=fread(&tmpUpperContig_32, sizeof(uint32_t), 1, tmpUpperFP)) {
		PrintError(FnName, NULL, "Could not read in tmp upper", Exit, ReadFileError);
	}

	for(i=low, ctr=0;
			i<=high &&
			tmpLowerPosition != 0 &&
			tmpUpperPosition != 0;
			i++, ctr++) {
		if(RGIndexCompareContigPos(index,
					rg,
					tmpLowerContig_32,
					tmpLowerPosition,
					tmpUpperContig_32,
					tmpUpperPosition,
					0)<=0) {
			/* Copy lower */
			index->positions[i] = tmpLowerPosition;
			index->contigs_32[i] = tmpLowerContig_32;
			/* Get new tmpLower */
			if(1!=fread(&tmpLowerPosition, sizeof(uint32_t), 1, tmpLowerFP) ||
					1!=fread(&tmpLowerContig_32, sizeof(uint32_t), 1, tmpLowerFP)) {
				tmpLowerPosition = 0;
				tmpLowerContig_32 = 0;
			}
		}
		else {
			/* Copy upper */
			index->positions[i] = tmpUpperPosition;
			index->contigs_32[i] = tmpUpperContig_32;
			/* Get new tmpUpper */
			if(1!=fread(&tmpUpperPosition, sizeof(uint32_t), 1, tmpUpperFP) ||
					1!=fread(&tmpUpperContig_32, sizeof(uint32_t), 1, tmpUpperFP)) {
				tmpUpperPosition = 0;
				tmpUpperContig_32 = 0;
			}
		}
	}
	while(tmpLowerPosition != 0 && tmpUpperPosition == 0) {
		/* Copy lower */
		index->positions[i] = tmpLowerPosition;
		index->contigs_32[i] = tmpLowerContig_32;
		/* Get new tmpLower */
		if(1!=fread(&tmpLowerPosition, sizeof(uint32_t), 1, tmpLowerFP) ||
				1!=fread(&tmpLowerContig_32, sizeof(uint32_t), 1, tmpLowerFP)) {
			tmpLowerPosition = 0;
			tmpLowerContig_32 = 0;
		}
		i++;
		ctr++;
	}
	while(tmpLowerPosition == 0 && tmpUpperPosition != 0) {
		/* Copy upper */
		index->positions[i] = tmpUpperPosition;
		index->contigs_32[i] = tmpUpperContig_32;
		/* Get new tmpUpper */
		if(1!=fread(&tmpUpperPosition, sizeof(uint32_t), 1, tmpUpperFP) ||
				1!=fread(&tmpUpperContig_32, sizeof(uint32_t), 1, tmpUpperFP)) {
			tmpUpperPosition = 0;
			tmpUpperContig_32 = 0;
		}
		i++;
		ctr++;
	}
	assert(ctr == (high - low + 1));
	assert(i == high + 1);

	/* Close tmp files */
	CloseTmpFile(&tmpLowerFP, &tmpLowerFileName);
	CloseTmpFile(&tmpUpperFP, &tmpUpperFileName);
	/* Test merge */
	/*
	   for(i=low+1;i<=high;i++) {
	   assert(RGIndexCompareAt(index, rg, i-1, i, 0) <= 0);
	   }
	   */
}

/* TODO */
void RGIndexDelete(RGIndex *index)
{
	/* Free memory and initialize */
	if(index->contigType == Contig_8) {
		free(index->contigs_8);
	}
	else {
		free(index->contigs_32);
	}
	free(index->positions);
	free(index->mask);
	free(index->starts);
	free(index->packageVersion);

	RGIndexInitialize(index);
}

/* TODO */
double RGIndexGetSize(RGIndex *index, int32_t outputSize) 
{
	double total=0.0;

	/* memory used by positions */
	total += (index->contigType==Contig_8)?(sizeof(uint8_t)*index->length):(sizeof(uint32_t)*index->length);
	/* memory used by positions */
	total += sizeof(uint32_t)*index->length;
	/* memory used by the mask */
	total += sizeof(int32_t)*index->width;
	/* memory used by starts */
	total += sizeof(uint32_t)*index->hashLength;
	/* memory used by the index base structure */
	total += sizeof(RGIndex); 

	switch(outputSize) {
		case KILOBYTES:
			return (total/pow(2, 10));
			break;
		case MEGABYTES:
			return (total/pow(2, 20));
			break;
		case GIGABYTES:
			return (total/pow(2, 30));
			break;
		default:
			return total;
			break;
	}
}

/* TODO */
void RGIndexPrint(gzFile fp, RGIndex *index)
{
	char *FnName="RGIndexPrint";

	/* Print header */
	RGIndexPrintHeader(fp, index);

	if(index->contigType == Contig_8) {
		/* Print positions */
		if(gzwrite64(fp, index->positions, sizeof(uint32_t)*index->length)!=sizeof(uint32_t)*index->length || 
				/* Print chomosomes */
				gzwrite64(fp, index->contigs_8, sizeof(uint8_t)*index->length)!=sizeof(uint8_t)*index->length ||
				/* Print the starts */
				gzwrite64(fp, index->starts, sizeof(uint32_t)*index->hashLength)!=sizeof(uint32_t)*index->hashLength) {
			PrintError(FnName, NULL, "Could not write index and hash", Exit, WriteFileError);
		}
	}
	else {
		/* Print positions */
		if(gzwrite64(fp, index->positions, sizeof(uint32_t)*index->length)!=sizeof(uint32_t)*index->length || 
				/* Print chomosomes */
				gzwrite64(fp, index->contigs_32, sizeof(uint32_t)*index->length)!=sizeof(uint32_t)*index->length ||
				/* Print the starts */
				gzwrite64(fp, index->starts, sizeof(uint32_t)*index->hashLength)!=sizeof(uint32_t)*index->hashLength) {
			PrintError(FnName, NULL, "Could not write index and hash", Exit, WriteFileError);
		}
	}

	gzclose(fp);
}

/* TODO */
void RGIndexRead(RGIndex *index, char *rgIndexFileName)
{
	char *FnName="RGIndexRead";

	gzFile fp;

	if(VERBOSE >= 0) {
		fprintf(stderr, "Reading index from %s.\n",
				rgIndexFileName);
	}

	/* open file */
	if(!(fp=gzopen(rgIndexFileName, "rb"))) {
		PrintError(FnName, rgIndexFileName, "Could not open rgIndexFileName for reading", Exit, OpenFileError);
	}

	/* Read in the header */
	RGIndexReadHeader(fp, index);

	assert(index->length > 0);

	/* Allocate memory for the positions */
	index->positions = malloc(sizeof(uint32_t)*index->length);
	if(NULL == index->positions) {
		PrintError(FnName, "index->positions", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Allocate memory for the contigs */
	if(index->contigType == Contig_8) {
		index->contigs_8 = malloc(sizeof(uint8_t)*index->length);
		if(NULL == index->contigs_8) {
			PrintError(FnName, "index->contigs", "Could not allocate memory", Exit, MallocMemory);
		}
	}
	else {
		index->contigs_32 = malloc(sizeof(uint32_t)*index->length);
		if(NULL == index->contigs_32) {
			PrintError(FnName, "index->contigs", "Could not allocate memory", Exit, MallocMemory);
		}
	}

	/* Read in positions */
	if(gzread64(fp, index->positions, sizeof(uint32_t)*index->length)!=sizeof(uint32_t)*index->length) {
		PrintError(FnName, NULL, "Could not read in positions", Exit, ReadFileError);
	}

	/* Read in the contigs */
	if(index->contigType == Contig_8) {
		if(gzread64(fp, index->contigs_8, sizeof(uint8_t)*index->length)!=sizeof(uint8_t)*index->length) {
			PrintError(FnName, NULL, "Could not read in contigs_8", Exit, ReadFileError);
		}
	}
	else {
		if(gzread64(fp, index->contigs_32, sizeof(uint32_t)*index->length)!=sizeof(uint32_t)*index->length) {
			PrintError(FnName, NULL, "Could not read in contigs_32", Exit, ReadFileError);
		}
	}

	/* Allocate memory for the starts */
	index->starts = malloc(sizeof(uint32_t)*index->hashLength);
	if(NULL == index->starts) {
		PrintError(FnName, "index->starts", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Read in starts */
	if(gzread64(fp, index->starts, sizeof(uint32_t)*index->hashLength)!=sizeof(uint32_t)*index->hashLength) {
		PrintError(FnName, NULL, "Could not read in starts", Exit, ReadFileError);
	}

	/* close file */
	gzclose(fp);

	if(VERBOSE >= 0) {
		fprintf(stderr, "Read index from %s.\n",
				rgIndexFileName);
	}
}

/* TODO */
/* Debugging function */
void RGIndexPrintInfo(char *inputFileName)
{
	char *FnName = "RGIndexPrintInfo";
	gzFile fp;
	int64_t i;
	RGIndex index;
	char contigType[2][256] = {"1 byte", "4 byte"};
	char Space[3][256] = {"NT Space", "Color Space", "Space Last Type"};
	FILE *fpOut=stdout;


	/* Open the file */
	if(!(fp=gzopen(inputFileName, "rb"))) {
		PrintError(FnName, inputFileName, "Could not open file for reading", Exit, OpenFileError);
	}

	/* Read in the header */
	RGIndexReadHeader(fp, &index);

	/* Print the info */
	fprintf(fpOut, "version:\t\t%s\n", index.packageVersion);
	fprintf(fpOut, "start contig:\t\t%d\n", index.startContig);
	fprintf(fpOut, "start position:\t\t%d\n", index.startPos);
	fprintf(fpOut, "end contig:\t\t%d\n", index.endContig);
	fprintf(fpOut, "end position:\t\t%d\n", index.endPos);
	fprintf(fpOut, "index length:\t\t%lld\n", (long long int)index.length);
	fprintf(fpOut, "contig type:\t\t%d\t\t[%s]\n", index.contigType, contigType[index.contigType]);
	fprintf(fpOut, "repeat masker:\t\t%d\n", index.repeatMasker);
	fprintf(fpOut, "space:\t\t\t%d\t\t[%s]\n", index.space, Space[index.space]);
	fprintf(fpOut, "depth:\t\t\t%d\n", index.depth);
	fprintf(fpOut, "binNumber:\t\t%d\n", index.binNumber);
	fprintf(fpOut, "indexNumber:\t\t%d\n", index.indexNumber);
	fprintf(fpOut, "hash width:\t\t%u\n", index.hashWidth);
	fprintf(fpOut, "hash length:\t\t%lld\n", (long long int)index.hashLength);
	fprintf(fpOut, "width:\t\t\t%d\n", index.width);
	fprintf(fpOut, "keysize:\t\t%d\n", index.keysize);
	fprintf(fpOut, "mask:\t\t\t");
	for(i=0;i<index.width;i++) {
		fprintf(fpOut, "%1d", index.mask[i]);
	}
	fprintf(fpOut, "\n");

	/* Free masks and initialize */
	free(index.mask);
	RGIndexInitialize(&index);

	/* Close the file */
	gzclose(fp);
}

/* TODO */
void RGIndexPrintHeader(gzFile fp, RGIndex *index)
{
	char *FnName="RGIndexPrintHeader";
	/* Print Header */
	if(gzwrite64(fp, &index->id, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->packageVersionLength, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, index->packageVersion, sizeof(char)*index->packageVersionLength)!=sizeof(char)*index->packageVersionLength ||
			gzwrite64(fp, &index->length, sizeof(int64_t))!=sizeof(int64_t) || 
			gzwrite64(fp, &index->contigType, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->startContig, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->startPos, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->endContig, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->endPos, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->width, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->keysize, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->repeatMasker, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->space, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->depth, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->binNumber, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->indexNumber, sizeof(int32_t))!=sizeof(int32_t) ||
			gzwrite64(fp, &index->hashWidth, sizeof(uint32_t))!=sizeof(uint32_t) ||
			gzwrite64(fp, &index->hashLength, sizeof(int64_t))!=sizeof(int64_t) ||
			gzwrite64(fp, index->mask, sizeof(int32_t)*index->width)!=sizeof(int32_t)*index->width) {
		PrintError(FnName, NULL, "Could not write header", Exit, WriteFileError);
	}
}

void RGIndexGetHeader(char *inputFileName, RGIndex *index)
{
	char *FnName="RGIndexGetHeader";
	gzFile fp=NULL;

	/* Open the file */
	if(!(fp=gzopen(inputFileName, "rb"))) {
		PrintError(FnName, inputFileName, "Could not open file for reading", Exit, OpenFileError);
	}

	RGIndexReadHeader(fp, index);

	gzclose(fp);
}

/* TODO */
void RGIndexReadHeader(gzFile fp, RGIndex *index) 
{
	char *FnName = "RGIndexReadHeader";
	/* Read in header */
	if(gzread64(fp, &index->id, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->packageVersionLength, sizeof(int32_t))!=sizeof(int32_t)) {
		PrintError(FnName, NULL, "Could not read header", Exit, ReadFileError);
	}
	index->packageVersion = malloc(sizeof(char)*(index->packageVersionLength+1));
	if(NULL==index->packageVersion) {
		PrintError(FnName, "index->packageVersion", "Could not allocate memory", Exit, MallocMemory);
	}

	if(gzread64(fp, index->packageVersion, sizeof(char)*index->packageVersionLength)!=sizeof(char)*index->packageVersionLength ||
			gzread64(fp, &index->length, sizeof(int64_t))!=sizeof(int64_t) || 
			gzread64(fp, &index->contigType, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->startContig, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->startPos, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->endContig, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->endPos, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->width, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->keysize, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->repeatMasker, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->space, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->depth, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->binNumber, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->indexNumber, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fp, &index->hashWidth, sizeof(uint32_t))!=sizeof(uint32_t) ||
			gzread64(fp, &index->hashLength, sizeof(int64_t))!=sizeof(int64_t)) {
		PrintError(FnName, NULL, "Could not read header", Exit, ReadFileError);
	}
	index->packageVersion[index->packageVersionLength]='\0';
	/* Allocate memory for the mask */
	index->mask = malloc(sizeof(int32_t)*index->width);
	if(NULL==index->mask) {
		PrintError(FnName, "index->mask", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Read the mask */
	if(gzread64(fp, index->mask, sizeof(int32_t)*index->width)!=sizeof(int32_t)*index->width) {
		PrintError(FnName, NULL, "Could not read header", Exit, ReadFileError);
	}

	/* Error checking */
	assert(index->id == (int)BFAST_ID);
	CheckPackageCompatibility(index->packageVersion, BFASTIndexFile);
	assert(index->length > 0);
	assert(index->contigType == Contig_8 || index->contigType == Contig_32);
	assert(index->startContig > 0);
	assert(index->startPos > 0);
	assert(index->endContig > 0);
	assert(index->endPos > 0);
	assert(index->width > 0);
	assert(index->keysize > 0);
	assert(index->repeatMasker == 0 || index->repeatMasker == 1);
	assert(index->space == NTSpace || index->space == ColorSpace);
	assert(index->hashWidth > 0);
	assert(index->hashLength > 0);
}

/* TODO */
/* We will append the matches if matches have already been found */
int64_t RGIndexGetRanges(RGIndex *index, RGBinary *rg, int8_t *read, int32_t readLength, int64_t *startIndex, int64_t *endIndex) 
{
	int64_t foundIndex=0;

	if(1!=WillGenerateValidKey(index, read, readLength)) {
		return 0;
	}

	/* Search the index using the bounds from the hash */
	foundIndex = RGIndexGetIndex(index, 
			rg, 
			read,
			readLength,
			startIndex,
			endIndex);
	/*
	if(foundIndex > 0) {
		assert((*endIndex) >= (*startIndex));
		assert((*startIndex) >= 0 && (*startIndex) < index->length);
		assert((*endIndex) >= 0 && (*endIndex) < index->length);
	}
	*/
	return foundIndex;
}

/* TODO */
/* We will append the matches if matches have already been found */
int32_t RGIndexGetRangesBothStrands(RGIndex *index, RGBinary *rg, int8_t *read, int32_t readLength, int32_t offset, int32_t maxKeyMatches, int32_t maxNumMatches, int32_t space, int32_t strands, RGRanges *r)
{
	int64_t startIndexForward=0;
	int64_t startIndexReverse=0;
	int64_t endIndexForward=-1;
	int64_t endIndexReverse=-1;
	int64_t foundIndexForward=0;
	int64_t foundIndexReverse=0;
	int64_t numMatches=0;
	int toAdd=0;
	int8_t reverseRead[SEQUENCE_LENGTH];

	/* Forward */
	if(BothStrands == strands || ForwardStrand == strands) {
		foundIndexForward = RGIndexGetRanges(index,
				rg,
				read,
				readLength,
				&startIndexForward,
				&endIndexForward);
	}
	/* Reverse */
	if(BothStrands == strands || ReverseStrand == strands) {
		if(space==ColorSpace) {
			/* In color space, the reverse compliment is just the reverse of the colors */
			ReverseReadFourBit(read, reverseRead, readLength);
		}
		else {
			//assert(space==NTSpace);
			/* Get the reverse compliment */
			GetReverseComplimentFourBit(read, reverseRead, readLength);
		}

		foundIndexReverse = RGIndexGetRanges(index,
				rg,
				reverseRead,
				readLength,
				&startIndexReverse,
				&endIndexReverse);
	}

	/* Update the number of matches */
	numMatches = (0 < foundIndexForward)?(endIndexForward - startIndexForward + 1):0;
	numMatches += (0 < foundIndexReverse)?(endIndexReverse - startIndexReverse + 1):0;

	/* Check if the key has too many matches */
	if(numMatches <= 0) {
		/* No matches */
		return 0;
	}
	else if(maxKeyMatches < numMatches) {
		/* Ignore the key since it had too many matches */
		//assert(0 < numMatches);
		return 0;
	}
	else if(maxNumMatches < numMatches) {
		/* Too many matches, return 1 */
		//assert(0 < numMatches);
		return 1;
	}
	else {
		toAdd = (0 < foundIndexForward)?1:0;
		toAdd += (0 < foundIndexReverse)?1:0;
		//assert(1 <= toAdd && toAdd <= 2);
		/* (Re)Allocate memory for the new range */
		RGRangesReallocate(r, r->numEntries + toAdd);
		/* Copy over to the range list */
		if(0 < foundIndexForward) {
			r->startIndex[r->numEntries-toAdd] = startIndexForward;
			r->endIndex[r->numEntries-toAdd] = endIndexForward;
			r->strand[r->numEntries-toAdd] = FORWARD;
			r->offset[r->numEntries-toAdd] = offset;
		}
		if(0 < foundIndexReverse) {
			r->startIndex[r->numEntries-1] = startIndexReverse;
			r->endIndex[r->numEntries-1] = endIndexReverse;
			r->strand[r->numEntries-1] = REVERSE;
			/* Must adjust for being the reverse */
			r->offset[r->numEntries-1] = offset;
			if(ColorSpace == space) {
				/* Must adjust for color space, since removed one color as well as being off by one
				 * in general for color space */
				r->offset[r->numEntries-1] += 2;
			}
		}
	}
	return 0;
}

/* TODO */
int64_t RGIndexGetIndex(RGIndex *index,
		RGBinary *rg,
		int8_t *read,
		int32_t readLength,
		int64_t *startIndex,
		int64_t *endIndex)
{
	int32_t cmp;
	int32_t cont = 1;
	int64_t tmpLow, tmpMid, tmpHigh;
	int32_t tmpLowNumBasesEqual, tmpHighNumBasesEqual, tmpMidNumBasesEqual;
	int64_t low, high, mid=-1;
	int32_t lowNumBasesEqual, highNumBasesEqual, midNumBasesEqual;
	uint32_t hashIndex;

	/* Use hash to restrict low and high */
	hashIndex = RGIndexGetHashIndexFromRead(index, rg, read, readLength, 0);
	if(UINT_MAX == hashIndex) {
		/* Did not fall in this bin */
		return 0;
	}
	//assert(0 <= hashIndex && hashIndex < index->hashLength);
	if(UINT_MAX == index->starts[hashIndex]) {
		/* The hash from this point on does not index anything */
		return 0;
	}
	else if(index->hashLength - 1 == hashIndex) {
		/* The end must point to entries in the index */
		low = index->starts[hashIndex];
		high = index->length - 1;
	}
	else if(index->starts[hashIndex] < index->starts[hashIndex+1]) {
		low = index->starts[hashIndex];
		/* Check to see if this goes all the way to the end of the index */
		if(UINT_MAX == index->starts[hashIndex+1]) {
			high = index->length - 1;
		}
		else {
			high = index->starts[hashIndex+1] - 1;
		}
	}
	else {
		return 0;
	}

	/*
	   int i;
	   fprintf(stderr, "\n");
	   for(i=0;i<readLength;i++) {
	   fprintf(stderr, "%1d", read[i]);
	   }
	   fprintf(stderr, "\n");
	   for(i=0;i<readLength;i++) {
	   fprintf(stderr, "%1c", "ACGTN"[read[i]]);
	   }
	   fprintf(stderr, "\n");
	   fprintf(stderr, "hashIndex=%u\n", hashIndex);
	   */

	//assert(low <= high);
	//assert(low==0 || 0 < RGIndexCompareRead(index, rg, read, low-1, 0, NULL, 0));
	//assert(high==index->length-1 || RGIndexCompareRead(index, rg, read, high+1, 0, NULL, 0) < 0); 

	// Assume that the first X # of bases are the same given the hash width and depth
	lowNumBasesEqual=highNumBasesEqual=midNumBasesEqual=index->hashWidth+index->depth;
	while(low <= high && cont==1) {
		mid = (low+high)/2;
		cmp = RGIndexCompareRead(index, rg, read, mid, GETMIN(lowNumBasesEqual, highNumBasesEqual), &midNumBasesEqual, 0);
		if(VERBOSE >= DEBUG) {
			fprintf(stderr, "low:%lld\tmid:%lld\thigh:%lld\tcmp:%d\n",
					(long long int)low,
					(long long int)mid,
					(long long int)high,
					cmp);
		}
		if(cmp == 0) {
			cont = 0;
		}
		else if(cmp < 0) {
			high = mid-1;
			highNumBasesEqual = midNumBasesEqual;
		}
		else {
			low = mid + 1;
			lowNumBasesEqual = midNumBasesEqual;
		}
	}
	/* If we found an entry that matches, get the bounds (start and end indexes */
	if(cont == 0) {
		//assert(low==0 || RGIndexCompareRead(index, rg, read, low-1, 0, NULL, 0) > 0);
		//assert(high==index->length-1 || RGIndexCompareRead(index, rg, read, high+1, 0, NULL, 0) < 0); 
		//assert(RGIndexCompareRead(index, rg, read, mid, 0, NULL, 0) == 0);
		tmpLow = low;
		tmpMid = mid;
		tmpHigh = high;
		tmpLowNumBasesEqual=lowNumBasesEqual;
		tmpHighNumBasesEqual=highNumBasesEqual;
		tmpMidNumBasesEqual=midNumBasesEqual;
		/*
		   fprintf(stderr, "Getting start and end:\t%lld\t%lld\t%lld\n",
		   low,
		   mid,
		   high);
		   */
		/* Get lower start Index */
		low = tmpLow;
		high = tmpMid;
		lowNumBasesEqual = tmpLowNumBasesEqual;
		highNumBasesEqual = tmpMidNumBasesEqual;
		while(low < high) {
			mid = (low+high)/2;
			cmp = RGIndexCompareRead(index, rg, read, mid, GETMIN(lowNumBasesEqual, highNumBasesEqual), &midNumBasesEqual, 0);
			//assert(cmp >= 0);
			/*
			   fprintf(stderr, "start:%lld\t%lld\t%lld\t%d\n",
			   low,
			   mid,
			   high,
			   cmp);
			   */
			if(cmp == 0) {
				high = mid;
				highNumBasesEqual = midNumBasesEqual;
			}
			else {
				/* mid is less than */
				low = mid+1;
				lowNumBasesEqual = midNumBasesEqual;
			}
		}
		(*startIndex) = low;
		//assert(low == high);
		//assert(RGIndexCompareRead(index, rg, read, (*startIndex), 0, NULL, 0)==0);
		//assert((*startIndex) == 0 || RGIndexCompareRead(index, rg, read, (*startIndex)-1, 0, NULL, 0)>0);
		/* Get upper start Index */
		low = tmpMid;
		high = tmpHigh;
		lowNumBasesEqual = tmpMidNumBasesEqual;
		highNumBasesEqual = tmpHighNumBasesEqual;
		while(low < high) {
			mid = (low+high)/2+1;
			cmp = RGIndexCompareRead(index, rg, read, mid, GETMIN(lowNumBasesEqual, highNumBasesEqual), &midNumBasesEqual, 0);
			//assert(cmp <= 0);
			/*
			   fprintf(stderr, "end:%lld\t%lld\t%lld\t%d\n",
			   low,
			   mid,
			   high,
			   cmp);
			   */
			if(cmp == 0) {
				low = mid;
				lowNumBasesEqual = midNumBasesEqual;
			}
			else {
				/* mid is less than */
				high = mid-1;
				highNumBasesEqual = midNumBasesEqual;
			}
		}
		//assert(low == high);
		/* adjust endIndex */
		(*endIndex) = low;
		//assert(RGIndexCompareRead(index, rg, read, (*endIndex), 0, NULL, 0)==0);
		//assert((*endIndex) == index->length-1 || RGIndexCompareRead(index, rg, read, (*endIndex)+1, 0, NULL, 0)<0);
		return 1;
	}
	else {
		return 0;
	}

}

/* TODO */
void RGIndexSwapAt(RGIndex *index, int64_t a, int64_t b)
{
	uint32_t tempContig, tempPos;

	tempPos = index->positions[a];
	index->positions[a] = index->positions[b];
	index->positions[b] = tempPos;

	if(index->contigType == Contig_8) {
		tempContig = index->contigs_8[a];
		index->contigs_8[a] = index->contigs_8[b];
		index->contigs_8[b] = tempContig;
	}
	else {
		tempContig = index->contigs_32[a];
		index->contigs_32[a] = index->contigs_32[b];
		index->contigs_32[b] = tempContig;
	}
}

/* TODO */
int64_t RGIndexGetPivot(RGIndex *index, RGBinary *rg, int64_t low, int64_t high)
{
	int64_t pivot = (low+high)/2;
	int32_t cmp[3];
	cmp[0] = RGIndexCompareAt(index, rg, low, pivot, 0);
	cmp[1] = RGIndexCompareAt(index, rg, low, high, 0);
	cmp[2] = RGIndexCompareAt(index, rg, pivot, high, 0);

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

/* TODO */
int32_t RGIndexCompareContigPos(RGIndex *index,
		RGBinary *rg,
		uint32_t aContig,
		uint32_t aPos,
		uint32_t bContig,
		uint32_t bPos,
		int debug)
{
	char *FnName="RGIndexCompareContigPos";
	int64_t i;
	char aBase;
	char bBase;

	/*
	assert(aContig >= index->startContig && aContig <= index->endContig);
	assert( (aContig != index->startContig || aPos >= index->startPos) &&
			(aContig != index->endContig || aPos <= index->endPos));
	assert(bContig >= index->startContig && bContig <= index->endContig);
	assert( (bContig != index->startContig || bPos >= index->startPos) &&
			(bContig != index->endContig || bPos <= index->endPos));
			*/

	/* Initialize for color space */

	if(debug == 1) {
		fprintf(stderr, "[%d,%d]\t[%d,%d]\n",
				(int)aContig,
				aPos,
				(int)bContig,
				bPos);
		char *seq=NULL;
		int32_t length;
		length = RGBinaryGetSequence(rg, aContig, aPos, FORWARD, &seq, index->width);
		RGIndexPrintReadMasked(index, seq, 0, stderr);
		free(seq); seq=NULL;
		length = RGBinaryGetSequence(rg, bContig, bPos, FORWARD, &seq, index->width);
		RGIndexPrintReadMasked(index, seq, 0, stderr);
		free(seq); seq=NULL;
	}

	/* Go across the mask */
	for(i=0;i<index->width;i++) {
		switch(index->mask[i]) {
			case 0:
				/* Ignore base */
				break;
			case 1:
				/* Get bases */
				aBase = ToLower(RGBinaryGetBase(rg,
							aContig,
							aPos + i));
				bBase = ToLower( RGBinaryGetBase(rg,
							bContig,
							bPos + i));
				/* Compare */
				if(aBase < bBase) {
					return -1;
				}
				else if(aBase > bBase) {
					return 1;
				}
				/* Continue if the current bases are equal */
				break;
			default:
				PrintError(FnName, NULL, "Could not understand mask", Exit, OutOfRange);
		}
	}

	/* All bases were equal, return 0 */
	return 0;
}

/* TODO */
int32_t RGIndexCompareAt(RGIndex *index,
		RGBinary *rg,
		int64_t a,
		int64_t b, 
		int debug)
{
	/*
	assert(a>=0 && a<index->length);
	assert(b>=0 && b<index->length);
	*/

	if(index->contigType == Contig_8) {
		return RGIndexCompareContigPos(index,
				rg,
				index->contigs_8[a],
				index->positions[a],
				index->contigs_8[b],
				index->positions[b],
				debug);
	}
	else {
		return RGIndexCompareContigPos(index,
				rg,
				index->contigs_32[a],
				index->positions[a],
				index->contigs_32[b],
				index->positions[b],
				debug);
	}
}

/* TODO */
int32_t RGIndexCompareRead(RGIndex *index,
		RGBinary *rg,
		int8_t* read,
		int64_t a,
		int32_t skip, // skip this number of bases in the prefix assuming they are equal
		int32_t *numBasesEqual, // returns the number of bases that were equal
		int debug)
{
	char *FnName="RGIndexCompareRead";
	//assert(a>=0 && a<index->length);

	int32_t i;
	uint32_t aContig = (index->contigType==Contig_8)?index->contigs_8[a]:index->contigs_32[a];
	uint32_t aPos = index->positions[a];

	uint8_t aBase;

	/*
	   if(debug > 0) {
	   fprintf(stderr, "%d\n%s", 
	   index->width,
	   BREAK_LINE);
	   fprintf(stderr, "read[%d]:%s\n", 
	   (int)strlen(read),
	   read);
	   }
	   */

	/* Go across the mask */
	//for(i=0;i<index->width;i++) {
	if(NULL != numBasesEqual) {
		(*numBasesEqual) = skip;
	}
	for(i=skip;i<index->width;i++) { // we do not need to compare the bases used by the hash 
		switch(index->mask[i]) {
			case 0:
				/* Ignore base */
				break;
			case 1:
				/* Get bases */
				aBase = RGBinaryGetFourBit(rg, aContig, aPos + i);
				aBase = ((aBase >> 2) == 2) ? 4 : (aBase & 0x03);
				/* Compare */
				if(read[i] < aBase) {
					return -1;
				}
				else if(read[i] > aBase) {
					return 1;
				}
				else if(NULL != numBasesEqual) {
					(*numBasesEqual) = i+1;
				}
				break;
			default:
				PrintError(FnName, NULL, "Could not understand mask", Exit, OutOfRange);
		}
	}

	/* All bases were equal, return 0 */
	return 0;
}

/* TODO */
uint32_t RGIndexGetHashIndex(RGIndex *index,
		RGBinary *rg,
		uint32_t a, // index in the index
		int debug)
{
	//assert(a>=0 && a<index->length);

	char *FnName = "RGIndexGetHashIndex";

	int32_t i;
	uint32_t aContig = (index->contigType==Contig_8)?index->contigs_8[a]:index->contigs_32[a];
	uint32_t aPos = index->positions[a];
	char aBase;
	int32_t cur = index->hashWidth-1;
	uint32_t hashIndex = 0;
	assert(ALPHABET_SIZE == 4);

	for(cur=i=0;cur<index->depth;i++) { // Skip over the first (depth) bases 
		switch(index->mask[i]) {
			case 0:
				break;
			case 1:
				cur++; 
				break;
			default:
				PrintError(FnName, NULL, "Could not understand mask", Exit, OutOfRange);
		}
	}

	/* Go across the mask */
	for(cur=index->hashWidth-1;0 <= cur && i<index->width;i++) {
		switch(index->mask[i]) {
			case 0:
				/* Ignore base */
				break;
			case 1:
				aBase = ToLower(RGBinaryGetBase(rg,
							aContig,
							aPos + i));
				switch(aBase) {
					case 0:
					case 'a':
						/* Do nothing since a is zero base 4 */
						break;
					case 1:
					case 'c':
						hashIndex += pow(ALPHABET_SIZE, cur);
						break;
					case 2:
					case 'g':
						hashIndex += pow(ALPHABET_SIZE, cur)*2;
						break;
					case 3:
					case 't':
						hashIndex += pow(ALPHABET_SIZE, cur)*3;
						break;
					default:
						PrintError(FnName, "aBase", "Could not understand base", Exit, OutOfRange);
						break;
				}
				/* Update */
				cur--;
				break;
			default:
				PrintError(FnName, NULL, "Could not understand mask", Exit, OutOfRange);
		}
	}

	return hashIndex;
}

/* TODO */
uint32_t RGIndexGetHashIndexFromRead(RGIndex *index,
		RGBinary *rg,
		int8_t *read,
		int32_t readLength,
		int debug)
{
	char *FnName = "RGIndexGetHashIndexFromRead";
	int32_t i=0;
	int32_t cur = 0;
	uint32_t hashIndex = 0;

	if(0 < index->depth) {
		/* Check if we are in the correct bin */
		for(cur=index->depth-1,i=0;0 <= cur && i < index->width;i++) { /* Skip over the first (depth) bases */
			switch(index->mask[i]) {
				case 0:
					break;
				case 1:
					/* Only works with a four letter alphabet */
					hashIndex = hashIndex << 2;
					hashIndex += read[i];
					cur--;
					break;
				default:
					PrintError(FnName, NULL, "Could not understand mask", Exit, OutOfRange);
			}
		}
		if(hashIndex != index->binNumber - 1) {
			return UINT_MAX;
		}
	}

	/* Go across the mask */
	hashIndex = 0;
	for(cur=index->hashWidth-1;0 <= cur && i<index->width;i++) {
		switch(index->mask[i]) {
			case 0:
				break;
			case 1:
				/* Only works with a four letter alphabet */
				hashIndex = hashIndex << 2;
				hashIndex += read[i];
				cur--;
				break;
			default:
				PrintError(FnName, NULL, "Could not understand mask", Exit, OutOfRange);
		}
	}

	return hashIndex;
}

/* TODO */
/* Debug function */
void RGIndexPrintReadMasked(RGIndex *index, char *read, int offset, FILE *fp) 
{
	char *FnName="RGIndexPrintReadMasked";
	int i;
	for(i=0;i<index->width;i++) {
		switch(index->mask[i]) {
			case 0:
				/* Ignore base */
				break;
			case 1:
				fprintf(stderr, "%c", read[i]);
				break;
			default:
				PrintError(FnName, NULL, "Could not understand mask", Exit, OutOfRange);
		}
	}   
	fprintf(fp, "\n");
}

/* TODO */
void RGIndexInitialize(RGIndex *index)
{
	index->id = 0;

	index->contigs_8 = NULL;
	index->contigs_32 = NULL;
	index->positions = NULL;
	index->length = 0;
	index->contigType = 0;

	index->startContig = 0;
	index->startPos = 0;
	index->endContig = 0;
	index->endPos = 0;

	index->width = 0;
	index->keysize = 0;
	index->mask = NULL;

	index->repeatMasker = 0;
	index->space = 0;
	index->depth = 0;
	index->binNumber = 0;
	index->indexNumber = 0;

	index->hashWidth = 0;
	index->hashLength = 0;
	index->starts = NULL;
}

void RGIndexInitializeFull(RGIndex *index,
		RGBinary *rg,
		RGIndexLayout *layout,
		int32_t space,
		int32_t binNumber,
		int32_t indexNumber,
		int32_t startContig,
		int32_t startPos,
		int32_t endContig,
		int32_t endPos,
		int32_t repeatMasker) 
{
	char *FnName="RGIndexInitializeFull";

	int32_t i;
	RGIndexInitialize(index);

	/* Copy over index information from the rg */
	assert(startContig <= endContig);
	assert(startContig < endContig || (startContig == endContig && startPos <= endPos));
	index->startContig = startContig;
	index->startPos = startPos;
	index->endContig = endContig;
	index->endPos = endPos;
	assert(index->startContig > 0);
	assert(index->endContig > 0);

	/* Copy over other metadata */
	index->id = BFAST_ID;
	index->packageVersionLength = (int)strlen(PACKAGE_VERSION);
	index->packageVersion = malloc(sizeof(char)*(index->packageVersionLength+1));
	if(NULL==index->packageVersion) {
		PrintError(FnName, "index->packageVersion", "Could not allocate memory", Exit, MallocMemory);
	}
	strcpy(index->packageVersion, PACKAGE_VERSION);
	index->repeatMasker = repeatMasker;
	index->space = space;
	index->depth = layout->depth; 
	index->binNumber = binNumber;
	index->indexNumber = indexNumber;

	/* Copy over index information from the layout */
	index->hashWidth = layout->hashWidth;
	assert(index->hashWidth > 0);
	index->width = layout->width;
	assert(index->width > 0);
	index->keysize = layout->keysize;
	assert(index->keysize > 0);
	index->mask = malloc(sizeof(int32_t)*layout->width);
	if(NULL == index->mask) {
		PrintError(FnName, "index->mask", "Could not allocate memory", Exit, MallocMemory);
	}
	/* Copy over mask */
	for(i=0;i<layout->width;i++) {
		index->mask[i] = layout->mask[i];
	}
	/* Infer the length of the hash */
	index->hashLength = pow(4, index->hashWidth);
	assert(index->hashLength > 0);
	/* Decide if we should use 1 byte or 4 byte to store the contigs. 
	 * We subtract one when comparing to UCHAR_MAX because we index 
	 * starting at one, not zero. */ 
	index->contigType = (rg->numContigs < UCHAR_MAX)?Contig_8:Contig_32;
}

gzFile RGIndexOpenForWriting(char *fastaFileName, RGIndex *index) 
{
	char *FnName="RGIndexOpenForWriting";
	gzFile gz;
	char *bifName=NULL;
	int fd;

	bifName=GetBIFName(fastaFileName, index->space, index->binNumber, index->indexNumber);
	if((fd = open(bifName, 
					O_WRONLY | O_CREAT | O_EXCL, 
					S_IRUSR | S_IRGRP | S_IROTH | S_IWUSR | S_IWGRP)) < 0) {
		/* File exists */
		PrintError(FnName, bifName, "Could not open bifName for writing", Exit, OpenFileError);
	}

	if(!(gz=gzdopen(fd, "wb"))) {
		PrintError(FnName, bifName, "Could not open bifName for writing", Exit, OpenFileError);
	}

	free(bifName);

	return gz;
}
