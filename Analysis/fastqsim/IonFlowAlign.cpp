/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static char *refSeq = NULL;
static int refBytes = 0;
static int refFlows = 0;
static char *flowOrder1 = "TACGN";
static char *flowOrder2 = "GCATN";
static bool verbose = false;
static int verboseCount = 0;

int testLen = 18; // try and match at least this many bp
int errorsToStopAt = 0; // stop when we EXCEED this many errors
int errorsToAccept = 0; // read is good if this many errors or less


static char testSeq[1024];

void FreeGenome()
{
        if (refSeq != NULL)
                free(refSeq);
        refSeq = NULL;
}

void RefFlowAdd(char *bases, int numBases)
{
// printf("Adding <%s>\n", bases);
	// grow ref seq as necessary
	if (numBases*4+refFlows > refBytes) {
		refBytes += numBases*4;
		refSeq = (char *)realloc(refSeq, refBytes);
	}

	int i = 0;
	while (i < numBases) {
		if (bases[i] == 'N') {
			i++;
			continue;
		}
		while(bases[i] != flowOrder1[refFlows%4]) {
			refSeq[refFlows] = 0;
			refFlows++;
		}
		refSeq[refFlows] = 0;
		while(i < numBases && bases[i] == flowOrder1[refFlows%4]) {
			refSeq[refFlows]++;
			i++;
		}
		refFlows++;
	}
}

void LoadGenome(char *fileName)
{
        if (refSeq != NULL)
                FreeGenome();

	// load ref genome & convert on the fly to flow space

        FILE *fp = fopen(fileName, "r");
        if (fp) {
                fseek(fp, 0, SEEK_END);
                refBytes = 2 * ftell(fp); // initial guess at mem we will need
                refSeq = (char *)malloc(refBytes);
                fseek(fp, 0, SEEK_SET);
                char line[512];
                assert(fgets(line, sizeof(line), fp));
                while (fgets(line, sizeof(line), fp)) {
                        int len = strlen(line);
                        while (len > 0 && (line[len-1] == '\r' || line[len-1] == '\n')) { // strip new lines & carriage returns
                                line[len-1] = 0;
                                len--;
                        }
			RefFlowAdd(line, len);
                }
                fclose(fp);
                printf("Loaded reference genome into %d bytes and %d flows.\n", refBytes, refFlows);
        }

	if (verbose && (verboseCount > 2)) {
		int i;
		for(i=0;i<refFlows;i++)
			printf("%d", refSeq[i]);
		printf("\n");
	}
}

int align(char *line, int len, bool reverse)
{
	// note that for searches in the 'reverse' direction, the input sequence is assumed to be supplied already in reverse direction
	// this algorithm assumes that more errors will be made later in the input sequence, so will start at the first base, or
	// last base when running in reverse, in order to maximize the likelyhood of a match

	if (verbose && (verboseCount > 1))
		printf("Trying to align: <%s>\n", line);

	// convert read into flowspace
	int numFlows = 999; // this gets set to the number of flows we will need to test this read against to get the desired readlength
	int testFlows = 0;
	int i = 0;
	// offset is the location we can start at in the ref genome, we only need to check every 4th starting location for alignments
	int offset = 0; // this offset can be used as the starting point in the ref genome, holds for either direction alignment
	while (line[0] != flowOrder1[offset])
		offset++;
	if (verbose && (verboseCount > 1))
		printf("We will skip to ref offset: %d\n", offset);

	char *flowOrder = flowOrder1;
	if (reverse)
		flowOrder = flowOrder2;

	// skip to the first base as per the flow order (don't care or want leading zeros)
	int readFlowOffset = 0;
	while(line[0] != flowOrder[readFlowOffset%4])
		readFlowOffset++;
	if (verbose && (verboseCount > 1)) {
		printf("Skipped to flow offset: %d\n", readFlowOffset);
	}

	while (i < len) {
		while(line[i] != flowOrder[(testFlows+readFlowOffset)%4]) {
			testSeq[testFlows] = 0;
			testFlows++;
		}
		testSeq[testFlows] = 0;
		while(i < len && line[i] == flowOrder[(testFlows+readFlowOffset)%4]) {
// printf("Match\n");
			testSeq[testFlows]++;
// printf("testFlows: %d val: %d\n", testFlows, testSeq[testFlows]);
			i++;
			if (i == testLen)
				numFlows = testFlows;
		}
		testFlows++;
	}

	if (verbose && (verboseCount > 1)) {
		for(i=0;i<testFlows;i++)
			printf("%d", testSeq[i]);
		printf("\n");
	}

	// search for *good* match

// printf("Using %d flows to align\n", numFlows);

	int sum;
	int hit;
	int j=0, j2=0;
	bool found = false;
	if (!reverse) {
		for(i=offset;i<refFlows-testFlows;i+=4) {
			sum = 0;
			for(j=0;j<testFlows;j++) {
				// we take a hit for every mismatch
				// and will match even when homopolymer lengths differ
				hit = (refSeq[i+j] == 0 ? (testSeq[j] == 0 ? 0 : 1) : (testSeq[j] > 0 ? 0 : 1));
				sum += hit;
				if (sum > errorsToStopAt)
					break;
			}
			if (sum <= errorsToAccept && j >= numFlows) {
				found = true;
				break;
			}
		}
	} else {
		for(i=(refFlows/4)*4+offset;i>testFlows;i-=4) { // not sure which version is faster?
		// for(i=offset+((testFlows/4)*4);i<refFlows-testFlows;i+=4) {
			sum = 0;
			for(j=0;j<testFlows;j++) {
				// we take a hit for every mismatch
				// and will match even when homopolymer lengths differ
				hit = (refSeq[i-j] == 0 ? (testSeq[j] == 0 ? 0 : 1) : (testSeq[j] > 0 ? 0 : 1));
				sum += hit;
				if (sum > errorsToStopAt)
					break;
			}
			if (sum <= errorsToAccept && j >= numFlows) {
				found = true;
				break;
			}
		}
	}

	if (verbose && (verboseCount > 0) && found) {
		printf("%s match at %d with len: %d\n", (reverse ? "Reverse" : "Forward"), i, j);

		// generate the alignment strings
		char refBuf[256];
		char testBuf[256];
		int k;
		int nextChar = 0;

		int alignFlows = j;
		for(j=0;j<alignFlows;j++) {
			if (reverse)
				j2 = -j;
			else
				j2 = j;
			if (refSeq[i+j2] > 0 || testSeq[j] > 0) {
				int max = (refSeq[i+j2] > testSeq[j] ? refSeq[i+j2] : testSeq[j]);
				int refCount = 0;
				int testCount = 0;
				for(k=0;k<max;k++) {
					if (refCount < refSeq[i+j2]) {
						refBuf[nextChar] = flowOrder1[(i+j2)%4];
					} else {
						refBuf[nextChar] = '-';
					}
					refCount++;

					if (testCount < testSeq[j]) {
						testBuf[nextChar] = flowOrder1[(i+j2)%4];
					} else {
						testBuf[nextChar] = '-';
					}
					testCount++;

					nextChar++;
				}
			}
		}

		refBuf[nextChar] = 0;
		testBuf[nextChar] = 0;
		printf("Input seq: %s\n", line);
		printf("%s\n%s\n", refBuf, testBuf);
	}

	if (found)
		return 0;
	return 99;
}

int main(int argc, char *argv[])
{
	char *refGenome = "CP000948.fna";
	char *reads = "";
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'g':
				argcc++;
				refGenome = argv[argcc];
			break;

			case 't': // min test len to find
				argcc++;
				testLen = atoi(argv[argcc]);
			break;

			case 'v':
				verbose = true;
				verboseCount++;
			break;

			case 'e': // errors
				argcc++;
				errorsToStopAt = atoi(argv[argcc]);
				argcc++;
				errorsToAccept = atoi(argv[argcc]);
			break;

			case 'r':
				argcc++;
				reads = argv[argcc];
			break;
		}
		argcc++;
	}

	// load up the ref genome
	LoadGenome(refGenome);

	// loop through our fastq reads, one at a time, and attempt to align each
	char line[256];
	int len;
	FILE *fp = fopen(reads, "r");
	int forwardAlignCount = 0;
	int reverseAlignCount = 0;
	int alignCount = 0;
	int readCount = 0;
	if (fp) {
		while(fgets(line, sizeof(line), fp)) {
			if (line[0] == '@') {
				assert(fgets(line, sizeof(line), fp));
				len = strlen(line);
				len--;
				line[len] = 0; // remove the return char
				int ret = align(line, len, false);
				if (ret < 3) {
					forwardAlignCount++;
					alignCount++;
				} else {
					// try reverse
					ret = align(line, len, true);
					if (ret < 3) {
						reverseAlignCount++;
						alignCount++;
					}
				}
				readCount++;
				if (verbose && (verboseCount > 2) && (readCount % 1 == 0)) {
					printf(".");
					fflush(stdout);
				}
			}
		}
		fclose(fp);
	}

	printf("Aligned %d of %d reads (forward: %d  reverse: %d)\n", alignCount, readCount, forwardAlignCount, reverseAlignCount);
}

