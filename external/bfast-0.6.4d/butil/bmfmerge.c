#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <config.h>
#include <zlib.h>
#include <unistd.h>  

#include "../bfast/BLibDefinitions.h"
#include "../bfast/BError.h"
#include "../bfast/AlignedRead.h"
#include "../bfast/RGMatches.h"
#include "../bfast/RGMatch.h"
#include "../bfast/MatchesReadInputFiles.h"


#define Name "bmfmerge"

/* Merges bmf files produced by different indexes produced by searching the same set of reads
 * */

int PrintUsage()
{
	fprintf(stderr, "%s %s\n", "bfast", PACKAGE_VERSION);
	fprintf(stderr, "\nUsage:%s [options] <bmf files>\n", Name);
	fprintf(stderr, "\t-M\tINT\tSpecifies the maximum total number of matches to consider (default: %d).\n", MAX_NUM_MATCHES);
	fprintf(stderr, "\t-Q\tINT\tSpecifies the number of reads to cache (default: %d).\n", DEFAULT_MATCHES_QUEUE_LENGTH);
	fprintf(stderr, "\t-h\t\tprints this help message\n");
	fprintf(stderr, "\nsend bugs to %s\n",
			PACKAGE_BUGREPORT);
	return 1;
}

int main(int argc, char *argv[]) 
{
	int32_t queueLength = DEFAULT_MATCHES_QUEUE_LENGTH;
	int32_t maxNumMatches = MAX_NUM_MATCHES;
	int c, i, numWritten;
	int startTime, endTime, seconds, minutes, hours;
	gzFile *inputFPs=NULL;
	gzFile outputFP=NULL;
	int32_t numInputFPs=0;

	while((c = getopt(argc, argv, "Q:h")) >= 0) {
		switch(c) {
			case 'h': return PrintUsage();
			case 'Q': queueLength=atoi(optarg); break;
			default: fprintf(stderr, "Unrecognized option: -%c\n", c); return 1;
		}
	}

	if(1 == argc || argc == optind) {
		return PrintUsage();
	}

	// allocate memory for bmf file pointers
	numInputFPs=(argc-optind);
	inputFPs = malloc(sizeof(gzFile)*numInputFPs);
	if(NULL == inputFPs) {
		PrintError(Name, "inputFPs", "Could not allocate memory", Exit, MallocMemory);
	}

	// open bmf files
	for(i=0;i<numInputFPs;i++) {
		if(!(inputFPs[i] = gzopen(argv[optind+i], "rb"))) {
			PrintError(Name, argv[optind+i], "Could not open file for reading", Exit, OpenFileError);
		}
	}

	/* Open output file */
	if(!(outputFP=gzdopen(fileno(stdout), "wb"))) {
		PrintError(Name, "stdout", "Could not open stdout for writing", Exit, OpenFileError);
	}

	// process
	if(VERBOSE >= 0) {
		fprintf(stderr, "%s", BREAK_LINE);
		fprintf(stderr, "Merging the output from each index...\n");
	}
	startTime=time(NULL);
	numWritten = RGMatchesMergeFilesAndOutput(inputFPs,
			numInputFPs,
			outputFP,
			maxNumMatches,
			queueLength);
	endTime=time(NULL);
	if(VERBOSE >= 0) {
		seconds = (int)(endTime - startTime);
		hours = seconds/3600;
		seconds -= hours*3600;
		minutes = seconds/60;
		seconds -= minutes*60;
		fprintf(stderr, "Merging %d matches from the indexes took: %d hours, %d minutes and %d seconds\n",
				numWritten,
				hours,
				minutes,
				seconds);
	}

	// close output file
	gzclose(outputFP);

	// close bmf files
	for(i=0;i<numInputFPs;i++) {
		gzclose(inputFPs[i]);
	}

	// free
	free(inputFPs);

	return 0;
}
