#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <config.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#include "BError.h"
#include "BLib.h"
#include "RunMatch.h"
#include "aflib.h"
#include "BfastMatch.h"

/*
   OPTIONS.  Field 1 in ARGP.
   Order of fields: {NAME, KEY, ARG, FLAGS, DOC, OPTIONAL_GROUP_NAME}.
   */
enum { 
	DescInputFilesTitle, DescFastaFileName, DescMainIndexes, DescSecondaryIndexes, DescReadsFileName, DescOffsets,  DescLoadAllIndexes, 
#ifndef DISABLE_BZ2
	DescCompressionBZ2, 
#endif
	DescCompressionGZ,
	DescAlgoTitle, DescSpace, DescStartReadNum, DescEndReadNum, 
	DescKeySize, DescMaxKeyMatches, DescMaxTotalMatches, DescWhichStrand, DescNumThreads, DescQueueLength, 
	DescOutputTitle, DescTmpDir, DescTiming,
	DescMiscTitle, DescParameters, DescHelp
};

static struct argp_option options[] = {
	{0, 0, 0, 0, "=========== Input Files =============================================================", 1},
	{"fastaFileName", 'f', "fastaFileName", 0, "Specifies the file name of the FASTA reference genome", 1},
	{"mainIndexes", 'i', "mainIndexes", 0, "The index numbers for the main bif files (comma separated)", 1},
	{"secondaryIndexes", 'I', "secondaryIndexes", 0, "The index numbers for the secondary bif files (comma"
		"\n\t\t\t\t  separated)", 1},
	{"readsFileName", 'r', "readsFileName", 0, "Specifies the file name for the reads", 1}, 
	{"offsets", 'o', "offsets", 0, "Specifies the offsets", 1},
	{"loadAllIndexes", 'l', "loadAllIndexes", 0, "Specifies to load all main or secondary indexes into memory", 1},
#ifndef DISABLE_BZ2
	{"bz2", 'j', "bz2", 0, "Specifies that the input reads are bz2 compressed (bzip2)", 1},
#endif
	{"gz", 'z', "gz", 0, "Specifies that the input reads are gz compressed (gzip)", 1},
	{0, 0, 0, 0, "=========== Algorithm Options: (Unless specified, default value = 0) ================", 2},
	{"space", 'A', "space", 0, "0: NT space 1: Color space", 2},
	{"startReadNum", 's', "startReadNum", 0, "Specifies the read to begin with (skip the first"
		"\n\t\t\t  startReadNum-1 reads)", 2},
	{"endReadNum", 'e', "endReadNum", 0, "Specifies the last read to use (inclusive)", 2},
	{"keySize", 'k', "keySize", 0, "Specifies to truncate all indexes to have the given key size"
		"\n\t\t\t  (must be greater than the hash width)", 2},
	{"maxKeyMatches", 'K', "maxKeyMatches", 0, "Specifies the maximum number of matches to allow before a key"
		"\n\t\t\t is ignored", 2},
	{"maxNumMatches", 'M', "maxNumMatches", 0, "Specifies the maximum total number of matches to consider"
		"\n\t\t\t before the read is discarded", 2},
	{"whichStrand", 'w', "whichStrand", 0, "0: consider both strands 1: forward strand only 2: reverse"
		"\n\t\t\t strand only", 2},
	{"numThreads", 'n', "numThreads", 0, "Specifies the number of threads to use (Default 1)", 2},
	{"queueLength", 'Q', "queueLength", 0, "Specifies the number of reads to cache", 2},
	{0, 0, 0, 0, "=========== Output Options ==========================================================", 3},
	{"tmpDir", 'T', "tmpDir", 0, "Specifies the directory in which to store temporary files", 3},
	{"timing", 't', 0, OPTION_NO_USAGE, "Specifies to output timing information", 3},
	{0, 0, 0, 0, "=========== Miscellaneous Options ===================================================", 4},
	{"Parameters", 'p', 0, OPTION_NO_USAGE, "Print program parameters", 4},
	{"Help", 'h', 0, OPTION_NO_USAGE, "Display usage summary", 4},
	{0, 0, 0, 0, 0, 0}
};

static char OptionString[]=
#ifndef DISABLE_BZ2
"e:f:i:k:m:n:o:r:s:w:A:I:K:M:Q:T:hjlptz";
#else
"e:f:i:k:m:n:o:r:s:w:A:I:K:M:Q:T:hlptz";
#endif

	int
BfastMatch(int argc, char **argv)
{
	struct arguments arguments;
	time_t startTime = time(NULL);
	time_t endTime;

	if(argc>1) {
		/* Set argument defaults. (overriden if user specifies them)  */ 
		BfastMatchAssignDefaultValues(&arguments);

		/* Parse command line args */
		if(BfastMatchGetOptParse(argc, argv, OptionString, &arguments)==0)
		{
			switch(arguments.programMode) {
				case ExecuteGetOptHelp:
					BfastMatchGetOptHelp();
					break;
				case ExecutePrintProgramParameters:
					BfastMatchPrintProgramParameters(stderr, &arguments);
					break;
				case ExecuteProgram:
					if(BfastMatchValidateInputs(&arguments)) {
						if(0 <= VERBOSE) {
							fprintf(stderr, "**** Input arguments look good!\n");
							fprintf(stderr, BREAK_LINE);
						}
					}
					else {
						PrintError("PrintError", NULL, "validating command-line inputs", Exit, InputArguments);

					}
					BfastMatchPrintProgramParameters(stderr, &arguments);
					/* Execute Program */

					/* Run Matches */
					RunMatch(
							arguments.fastaFileName,
							arguments.mainIndexes,
							arguments.secondaryIndexes,
							arguments.readsFileName,
							arguments.offsets,
							arguments.loadAllIndexes,
							arguments.compression,
							arguments.space,
							arguments.startReadNum,
							arguments.endReadNum,
							arguments.keySize,
							arguments.maxKeyMatches,
							arguments.maxNumMatches,
							arguments.whichStrand,
							arguments.numThreads,
							arguments.queueLength,
							arguments.tmpDir,
							arguments.timing,
							stdout);

					if(arguments.timing == 1) {
						endTime = time(NULL);
						int seconds = endTime - startTime;
						int hours = seconds/3600;
						seconds -= hours*3600;
						int minutes = seconds/60;
						seconds -= minutes*60;
						if(0 <= VERBOSE) {
							fprintf(stderr, "Total time elapsed: %d hours, %d minutes and %d seconds.\n",
									hours,
									minutes,
									seconds
								   );
						}
					}
					if(0 <= VERBOSE) {
						fprintf(stderr, "Terminating successfully!\n");
						fprintf(stderr, "%s", BREAK_LINE);
					}
					break;
				default:
					PrintError("PrintError", "programMode", "Could not determine program mode", Exit, OutOfRange);
			}
		}
		else {
			PrintError("PrintError", NULL, "Could not parse command line arguments", Exit, InputArguments);
		}
		/* Free program parameters */
		BfastMatchFreeProgramParameters(&arguments);
	}
	else {
		BfastMatchGetOptHelp();
	}

	return 0;
}

/* TODO */
int BfastMatchValidateInputs(struct arguments *args) {

	char *FnName="BfastMatchValidateInputs";

	/* Check if we are piping */
	if(NULL == args->readsFileName) {
		VERBOSE = -1;
	}

	if(0<=VERBOSE) {
		fprintf(stderr, BREAK_LINE);
		fprintf(stderr, "Checking input parameters supplied by the user ...\n");
	}

	if(args->fastaFileName!=0) {
		if(0<=VERBOSE) {
			fprintf(stderr, "Validating fastaFileName %s. \n",
					args->fastaFileName);
		}
		if(ValidateFileName(args->fastaFileName)==0)
			PrintError(FnName, "fastaFileName", "Command line argument", Exit, IllegalFileName);	
	}	
	else {		
		PrintError(FnName, "fastaFileName", "Required command line argument", Exit, IllegalFileName);	
	}

	if(args->readsFileName!=0) {
		if(0<=VERBOSE) {
			fprintf(stderr, "Validating readsFileName %s. \n",
					args->readsFileName);
		}
		if(ValidateFileName(args->readsFileName)==0)
			PrintError(FnName, "readsFileName", "Command line argument", Exit, IllegalFileName);	
	}	

	if(args->space != NTSpace && args->space != ColorSpace) {
		PrintError(FnName, "space", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->keySize < 0) {		
		PrintError(FnName, "keySize", "Command line argument", Exit, OutOfRange);
	}

	if(args->maxKeyMatches < 0) {
		PrintError(FnName, "maxKeyMatches", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->maxNumMatches < 0) {		
		PrintError(FnName, "maxNumMatches", "Command line argument", Exit, OutOfRange);
	}

	if(!(args->whichStrand == BothStrands || 
				args->whichStrand == ForwardStrand || 
				args->whichStrand == ReverseStrand)) {
		PrintError(FnName, "whichStrand", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->numThreads<=0) {		
		PrintError(FnName, "numThreads", "Command line argument", Exit, OutOfRange);
	} 

	if(args->queueLength<=0) {
		PrintError(FnName, "queueLength", "Command line argument", Exit, OutOfRange);	
	} 	
	if(args->tmpDir!=0) {		
		if(0 <= VERBOSE) {
			fprintf(stderr, "Validating tmpDir path %s. \n", 
					args->tmpDir);
		}
		if(ValidatePath(args->tmpDir)==0)
			PrintError(FnName, "tmpDir", "Command line argument", Exit, IllegalPath);	
	}	
	/* If this does not hold, we have done something wrong internally */	
	assert(args->timing == 0 || args->timing == 1);
	assert(IndexesMemorySerial == args->loadAllIndexes || IndexesMemoryAll == args->loadAllIndexes);

	return 1;
}

/* TODO */
	void 
BfastMatchAssignDefaultValues(struct arguments *args)
{
	/* Assign default values */

	args->programMode = ExecuteProgram;

	args->fastaFileName = NULL;

	args->mainIndexes = NULL;
	args->secondaryIndexes = NULL;
	args->readsFileName = NULL;
	args->offsets = NULL;
	args->loadAllIndexes = IndexesMemorySerial;
	args->compression = AFILE_NO_COMPRESSION;

	args->space = NTSpace;

	args->startReadNum = 1;
	args->endReadNum = INT_MAX;
	args->keySize = 0;
	args->maxKeyMatches = MAX_KEY_MATCHES;
	args->maxNumMatches = MAX_NUM_MATCHES;
	args->whichStrand = BothStrands;
	args->numThreads = 1;
	args->queueLength = DEFAULT_MATCHES_QUEUE_LENGTH;

	args->tmpDir =
		(char*)malloc(sizeof(DEFAULT_OUTPUT_DIR));
	assert(args->tmpDir!=0);
	strcpy(args->tmpDir, DEFAULT_OUTPUT_DIR);

	args->timing = 0;

	return;
}

/* TODO */
	void 
BfastMatchPrintProgramParameters(FILE* fp, struct arguments *args)
{
	if(0 <= VERBOSE) {
		fprintf(fp, BREAK_LINE);
		fprintf(fp, "Printing Program Parameters:\n");
		fprintf(fp, "programMode:\t\t\t\t%s\n", PROGRAMMODE(args->programMode));
		fprintf(fp, "fastaFileName:\t\t\t\t%s\n", FILEREQUIRED(args->fastaFileName));
		fprintf(fp, "mainIndexes\t\t\t\t%s\n", (NULL == args->mainIndexes) ? "[Auto-recognizing]" : args->mainIndexes);
		fprintf(fp, "secondaryIndexes\t\t\t%s\n", FILEUSING(args->secondaryIndexes));
		fprintf(fp, "readsFileName:\t\t\t\t%s\n", FILESTDIN(args->readsFileName));
		fprintf(fp, "offsets:\t\t\t\t%s\n", (NULL == args->offsets) ? "[Using All]" : args->offsets);
		fprintf(fp, "loadAllIndexes:\t\t\t\t%s\n", INTUSING(args->loadAllIndexes));
		fprintf(fp, "compression:\t\t\t\t%s\n", COMPRESSION(args->compression));
		fprintf(fp, "space:\t\t\t\t\t%s\n", SPACE(args->space));
		fprintf(fp, "startReadNum:\t\t\t\t%d\n", args->startReadNum);
		fprintf(fp, "endReadNum:\t\t\t\t%d\n", args->endReadNum);
		if(0 < args->keySize) fprintf(fp, "keySize:\t\t\t\t%d\n", args->keySize);
		else fprintf(fp, "keySize:\t\t\t\t%s\n", INTUSING(0));
		fprintf(fp, "maxKeyMatches:\t\t\t\t%d\n", args->maxKeyMatches);
		fprintf(fp, "maxNumMatches:\t\t\t\t%d\n", args->maxNumMatches);
		fprintf(fp, "whichStrand:\t\t\t\t%s\n", WHICHSTRAND(args->whichStrand));
		fprintf(fp, "numThreads:\t\t\t\t%d\n", args->numThreads);
		fprintf(fp, "queueLength:\t\t\t\t%d\n", args->queueLength);
		fprintf(fp, "tmpDir:\t\t\t\t\t%s\n", args->tmpDir);
		fprintf(fp, "timing:\t\t\t\t\t%s\n", INTUSING(args->timing));
		fprintf(fp, BREAK_LINE);
	}
	return;
}

/* TODO */
void BfastMatchFreeProgramParameters(struct arguments *args)
{
	free(args->fastaFileName);
	args->fastaFileName=NULL;
	free(args->mainIndexes);
	args->mainIndexes=NULL;
	free(args->secondaryIndexes);
	args->secondaryIndexes=NULL;
	free(args->readsFileName);
	args->readsFileName=NULL;
	free(args->offsets);
	args->offsets=NULL;
	free(args->tmpDir);
	args->tmpDir=NULL;
}

/* TODO */
void
BfastMatchGetOptHelp() {

	struct argp_option *a=options;
	fprintf(stderr, "\nUsage: bfast match [options]\n");
	while((*a).group>0) {
		switch((*a).key) {
			case 0:
				fprintf(stderr, "\n%s\n", (*a).doc); break;
			default:
				if((*a).arg != 0) {
					fprintf(stderr, "-%c\t%12s\t%s\n", (*a).key, (*a).arg, (*a).doc); 
				}
				else {
					fprintf(stderr, "-%c\t%12s\t%s\n", (*a).key, "", (*a).doc); 
				}
				break;
		}
		a++;
	}
	fprintf(stderr, "\nsend bugs to %s\n", 
			PACKAGE_BUGREPORT);
	return;
}

/* TODO */
	int
BfastMatchGetOptParse(int argc, char** argv, char OptionString[], struct arguments* arguments) 
{
	int key;
	int OptErr=0;
	while((OptErr==0) && ((key = getopt (argc, argv, OptionString)) != -1)) {
		/*
		   fprintf(stderr, "Key is %c and OptErr = %d\n", key, OptErr);
		   */
		switch (key) {
			case 'e':
				arguments->endReadNum = atoi(optarg); break;
			case 'f':
				arguments->fastaFileName = strdup(optarg); break;
			case 'h':
				arguments->programMode=ExecuteGetOptHelp; break;
			case 'i':
				arguments->mainIndexes=strdup(optarg); break;
#ifndef DISABLE_BZ2
			case 'j':
				arguments->compression=AFILE_BZ2_COMPRESSION; break;
#endif
			case 'k':
				arguments->keySize = atoi(optarg); break;
			case 'l':
				arguments->loadAllIndexes = IndexesMemoryAll; break;
			case 'n':
				arguments->numThreads=atoi(optarg); break;
			case 'o':
				arguments->offsets=strdup(optarg); break;
			case 'p':
				arguments->programMode=ExecutePrintProgramParameters; break;
			case 'r':
				arguments->readsFileName=strdup(optarg); break;
			case 's':
				arguments->startReadNum = atoi(optarg); break;
			case 't':
				arguments->timing = 1; break;
			case 'w':
				arguments->whichStrand = atoi(optarg); break;
			case 'z':
				arguments->compression=AFILE_GZ_COMPRESSION; break;
			case 'A':
				arguments->space=atoi(optarg); break;
			case 'I':
				arguments->secondaryIndexes=strdup(optarg); break;
			case 'K':
				arguments->maxKeyMatches=atoi(optarg); break;
			case 'M':
				arguments->maxNumMatches=atoi(optarg); break;
			case 'Q':
				arguments->queueLength=atoi(optarg); break;
			case 'T':
				StringCopyAndReallocate(&arguments->tmpDir, optarg); break;
			default:
				OptErr=1;
		} /* while */
	} /* switch */
	return OptErr;
}
