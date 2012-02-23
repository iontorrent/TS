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
#include "RunAlign.h"
#include "aflib.h"
#include "BfastAlign.h"

/*
   OPTIONS.  Field 1 in ARGP.
   Order of fields: {NAME, KEY, ARG, FLAGS, DOC, OPTIONAL_GROUP_NAME}.
   */
enum { 
	DescInputFilesTitle, DescFastaFileName, DescReadsFileName, DescLoadAllIndexes, DescCompressionBZ2, DescCompressionGZ,
	DescAlgoTitle, DescSpace, DescNumThreads, 
	DescOutputTitle, DescTmpDir, DescTiming,
	DescMiscTitle, DescParameters, DescHelp
};

static struct argp_option options[] = {
	{0, 0, 0, 0, "=========== Input Files =============================================================", 1},
	{"fastaFileName", 'f', "fastaFileName", 0, "Specifies the file name of the FASTA reference genome", 1},
	{"readsFileName", 'r', "readsFileName", 0, "Specifies the file name for the reads", 1}, 
	{"bz2", 'j', "bz2", 0, "Specifies that the input reads are bz2 compressed (bzip2)", 1},
	{"gz", 'z', "gz", 0, "Specifies that the input reads are gz compressed (gzip)", 1},
	{0, 0, 0, 0, "=========== Algorithm Options: (Unless specified, default value = 0) ================", 2},
	{"space", 'A', "space", 0, "0: NT space 1: Color space", 2},
	{"numThreads", 'n', "numThreads", 0, "Specifies the number of threads to use (Default 1)", 2},
	{0, 0, 0, 0, "=========== Output Options ==========================================================", 3},
	{"tmpDir", 'T', "tmpDir", 0, "Specifies the directory in which to store temporary files", 3},
	{"timing", 't', 0, OPTION_NO_USAGE, "Specifies to output timing information", 3},
	{0, 0, 0, 0, "=========== Miscellaneous Options ===================================================", 4},
	{"Parameters", 'p', 0, OPTION_NO_USAGE, "Print program parameters", 4},
	{"Help", 'h', 0, OPTION_NO_USAGE, "Display usage summary", 4},
	{0, 0, 0, 0, 0, 0}
};

static char OptionString[]=
"f:n:r:A:T:hjptz";

	int
BfastAlign(int argc, char **argv)
{
	struct arguments arguments;
	time_t startTime = time(NULL);
	time_t endTime;

	if(argc>1) {
		/* Set argument defaults. (overriden if user specifies them)  */ 
		BfastAlignAssignDefaultValues(&arguments);

		/* Parse command line args */
		if(BfastAlignGetOptParse(argc, argv, OptionString, &arguments)==0)
		{
			switch(arguments.programMode) {
				case ExecuteGetOptHelp:
					BfastAlignGetOptHelp();
					break;
				case ExecutePrintProgramParameters:
					BfastAlignPrintProgramParameters(stderr, &arguments);
					break;
				case ExecuteProgram:
					if(BfastAlignValidateInputs(&arguments)) {
						if(0 <= VERBOSE) {
							fprintf(stderr, "**** Input arguments look good!\n");
							fprintf(stderr, BREAK_LINE);
						}
					}
					else {
						PrintError("PrintError", NULL, "validating command-line inputs", Exit, InputArguments);

					}
					BfastAlignPrintProgramParameters(stderr, &arguments);
					/* Execute Program */

					/* Run Matches */
					RunAlign(
							arguments.fastaFileName,
							arguments.readsFileName,
							arguments.compression,
							arguments.space,
							arguments.numThreads,
							arguments.tmpDir,
							arguments.timing);

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
		BfastAlignFreeProgramParameters(&arguments);
	}
	else {
		BfastAlignGetOptHelp();
	}

	return 0;
}

/* TODO */
int BfastAlignValidateInputs(struct arguments *args) {

	char *FnName="BfastAlignValidateInputs";

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
	
	if(args->numThreads<=0) {		
		PrintError(FnName, "numThreads", "Command line argument", Exit, OutOfRange);
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

	return 1;
}

/* TODO */
	void 
BfastAlignAssignDefaultValues(struct arguments *args)
{
	/* Assign default values */

	args->programMode = ExecuteProgram;

	args->fastaFileName = NULL;

	args->readsFileName = NULL;
	args->compression = AFILE_NO_COMPRESSION;
	args->space = NTSpace;
	args->numThreads = 1;

	args->tmpDir =
		(char*)malloc(sizeof(DEFAULT_OUTPUT_DIR));
	assert(args->tmpDir!=0);
	strcpy(args->tmpDir, DEFAULT_OUTPUT_DIR);

	args->timing = 0;

	return;
}

/* TODO */
	void 
BfastAlignPrintProgramParameters(FILE* fp, struct arguments *args)
{
	if(0 <= VERBOSE) {
		fprintf(fp, BREAK_LINE);
		fprintf(fp, "Printing Program Parameters:\n");
		fprintf(fp, "programMode:\t\t\t\t%s\n", PROGRAMMODE(args->programMode));
		fprintf(fp, "fastaFileName:\t\t\t\t%s\n", FILEREQUIRED(args->fastaFileName));
		fprintf(fp, "readsFileName:\t\t\t\t%s\n", FILESTDIN(args->readsFileName));
		fprintf(fp, "compression:\t\t\t\t%s\n", COMPRESSION(args->compression));
		fprintf(fp, "space:\t\t\t\t\t%s\n", SPACE(args->space));
		fprintf(fp, "numThreads:\t\t\t\t%d\n", args->numThreads);
		fprintf(fp, "tmpDir:\t\t\t\t\t%s\n", args->tmpDir);
		fprintf(fp, "timing:\t\t\t\t\t%s\n", INTUSING(args->timing));
		fprintf(fp, BREAK_LINE);
	}
	return;
}

/* TODO */
void BfastAlignFreeProgramParameters(struct arguments *args)
{
	free(args->fastaFileName);
	args->fastaFileName=NULL;
	free(args->readsFileName);
	args->readsFileName=NULL;
	free(args->tmpDir);
	args->tmpDir=NULL;
}

/* TODO */
void
BfastAlignGetOptHelp() {

	struct argp_option *a=options;
	fprintf(stderr, "\nUsage: bfast easyalign [options]\n");
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
BfastAlignGetOptParse(int argc, char** argv, char OptionString[], struct arguments* arguments) 
{
	int key;
	int OptErr=0;
	while((OptErr==0) && ((key = getopt (argc, argv, OptionString)) != -1)) {
		/*
		   fprintf(stderr, "Key is %c and OptErr = %d\n", key, OptErr);
		   */
		switch (key) {
			case 'f':
				arguments->fastaFileName = strdup(optarg); break;
			case 'h':
				arguments->programMode=ExecuteGetOptHelp; break;
			case 'j':
				arguments->compression=AFILE_BZ2_COMPRESSION; break;
			case 'n':
				arguments->numThreads=atoi(optarg); break;
			case 'p':
				arguments->programMode=ExecutePrintProgramParameters; break;
			case 'r':
				arguments->readsFileName=strdup(optarg); break;
			case 't':
				arguments->timing = 1; break;
			case 'z':
				arguments->compression=AFILE_GZ_COMPRESSION; break;
			case 'A':
				arguments->space=atoi(optarg); break;
			case 'T':
				StringCopyAndReallocate(&arguments->tmpDir, optarg); break;
			default:
				OptErr=1;
		} /* while */
	} /* switch */
	return OptErr;
}
