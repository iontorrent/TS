#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <config.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#include "BLibDefinitions.h"
#include "RGBinary.h"
#include "RGIndexLayout.h"
#include "RGIndexExons.h"
#include "BError.h"
#include "BLib.h"
#include "BfastIndex.h"

/*
   OPTIONS.  Field 1 in ARGP.
   Order of fields: {NAME, KEY, ARG, FLAGS, DOC, OPTIONAL_GROUP_NAME}.
   */
enum { 
	DescInputFilesTitle, DescFastaFileName, DescIndexLayoutFileName,  
	DescAlgoTitle, DescSpace, DescNumThreads, DescRepeatMasker, DescStartContig, DescStartPos, DescEndContig, DescEndPos, DescExonFileName, 
	DescOutputTitle, DescTmpDir, DescTiming,
	DescMiscTitle, DescParameters, DescHelp
};

static struct argp_option options[] = {
	{0, 0, 0, 0, "=========== Input Files =============================================================", 1},
	{"fastaFileName", 'f', "fastaFileName", 0, "Specifies the file name of the FASTA reference genome", 1},
	{0, 0, 0, 0, "=========== Algorithm Options =======================================================", 2},
	{"space", 'A', "space", 0, "0: NT space 1: Color space", 2},
	{"mask", 'm', "mask", 0, "The mask or spaced seed to use", 2}, 
	{"hashWidth", 'w', "hashWidth", 0, "The hash width for the index", 2},
	{"depth", 'd', "depth", 0, "The depth of splitting (d).  The index will be split into"
		"\n\t\t\t  4^d parts.", 2},
	{"indexNumber", 'i', "indexNumber", 0, "Specifies this is the ith index you are creating", 2},
	{"repeatMasker", 'R', 0, OPTION_NO_USAGE, "Specifies that lower case bases will be ignored", 2},
	{"startContig", 's', "startContig", 0, "Specifies the start contig", 2},
	{"startPos", 'S', "startPos", 0, "Specifies the end position", 2},
	{"endContig", 'e', "endContig", 0, "Specifies the end contig", 2},
	{"endPos", 'E', "endPos", 0, "Specifies the end postion", 2},
	{"exonsFileName", 'x', "exonsFileName", 0, "Specifies the file name that specifies the exon-like ranges to"
		"\n\t\t\t  include in the index", 2},
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
"d:e:f:i:m:n:s:w:x:A:E:S:T:hptR";

	int
BfastIndex(int argc, char **argv)
{
	struct arguments arguments;
	RGIndexLayout rgLayout;
	RGIndexExons exons;
	time_t startTime = time(NULL);
	time_t endTime;

	if(argc>1) {
		/* Set argument defaults. (overriden if user specifies them)  */ 
		BfastIndexAssignDefaultValues(&arguments);

		/* Parse command line args */
		if(BfastIndexGetOptParse(argc, argv, OptionString, &arguments)==0)
		{
			switch(arguments.programMode) {
				case ExecuteGetOptHelp:
					BfastIndexGetOptHelp();
					break;
				case ExecutePrintProgramParameters:
					BfastIndexPrintProgramParameters(stderr, &arguments);
					break;
				case ExecuteProgram:
					if(BfastIndexValidateInputs(&arguments)) {
						fprintf(stderr, "Input arguments look good!\n");
						fprintf(stderr, BREAK_LINE);
					}
					else {
						PrintError("PrintError", NULL, "validating command-line inputs", Exit, InputArguments);
					}
					BfastIndexPrintProgramParameters(stderr, &arguments);

					/* Read in the RGIndex layout */
					RGIndexLayoutCreate(arguments.mask, 
							arguments.hashWidth, 
							arguments.depth,
							&rgLayout);
					/* Read exons, if necessary */
					if(NULL != arguments.exonsFileName) {
						RGIndexExonsRead(arguments.exonsFileName,
								&exons);
					}

					/* Generate the indexes */
					RGIndexCreate(arguments.fastaFileName,
							&rgLayout,
							arguments.space,
							arguments.indexNumber,
							arguments.startContig,
							arguments.startPos,
							arguments.endContig,
							arguments.endPos,
							(NULL == arguments.exonsFileName) ? 0 : 1,
							&exons,
							arguments.numThreads,
							arguments.repeatMasker,
							0,
							arguments.tmpDir);

					/* Free the RGIndex layout */
					RGIndexLayoutDelete(&rgLayout);
					/* Free exons, if necessary */
					if(NULL != arguments.exonsFileName) {
						RGIndexExonsDelete(&exons);
					}
					else {
						/* Free exons file name if we did not use it */
						free(arguments.exonsFileName);
						arguments.exonsFileName=NULL;
					}

					if(arguments.timing == 1) {
						/* Get the time information */
						endTime = time(NULL);
						int seconds = endTime - startTime;
						int hours = seconds/3600;
						seconds -= hours*3600;
						int minutes = seconds/60;
						seconds -= minutes*60;
						fprintf(stderr, "Total time elapsed: %d hours, %d minutes and %d seconds.\n",
								hours,
								minutes,
								seconds
							   );
					}
					fprintf(stderr, "Terminating successfully!\n");
					fprintf(stderr, "%s", BREAK_LINE);
					break;
				default:
					PrintError("PrintError", "programMode", "Could not determine program mode", Exit, OutOfRange);
			}

		}
		else {
			PrintError("PrintError", NULL, "Could not parse command line arguments", Exit, InputArguments);
		}
		/* Free program parameters */
		BfastIndexFreeProgramParameters(&arguments);
	}
	else {
		BfastIndexGetOptHelp();
	}
	return 0;
}

/* TODO */
int BfastIndexValidateInputs(struct arguments *args) {

	char *FnName="BfastIndexValidateInputs";

	fprintf(stderr, BREAK_LINE);
	fprintf(stderr, "Checking input parameters supplied by the user ...\n");

	if(args->fastaFileName!=0) {
		fprintf(stderr, "Validating fastaFileName %s. \n",
				args->fastaFileName);
		if(ValidateFileName(args->fastaFileName)==0)
			PrintError(FnName, "fastaFileName", "Command line argument", Exit, IllegalFileName);	
	}	
	else {			
		PrintError(FnName, "fastaFileName", "Required command line argument", Exit, IllegalFileName);	
	}
	
	if(args->mask==NULL) {
		PrintError(FnName, "mask", "Required command line argument", Exit, IllegalFileName);	
	}	
	if(args->space != NTSpace && args->space != ColorSpace) {		
		PrintError(FnName, "space", "Command line argument", Exit, OutOfRange);
	}

	if(args->hashWidth <= 0) {
		PrintError(FnName, "hashWidth", "Command line argument", Exit, OutOfRange);	
	}		
	if(args->depth < 0) {		
			PrintError(FnName, "depth", "Command line argument", Exit, OutOfRange);
	}
	
	if(args->indexNumber <= 0) {
		PrintError(FnName, "indexNumber", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->startContig < 0) {		
		PrintError(FnName, "startContig", "Command line argument", Exit, OutOfRange);
	}

	if(args->startPos < 0) {
		PrintError(FnName, "startPos", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->endContig < 0) {		
		PrintError(FnName, "endContig", "Command line argument", Exit, OutOfRange);
	}

	if(args->endPos < 0) {
		PrintError(FnName, "endPos", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->exonsFileName!=NULL) {		
		fprintf(stderr, "Validating exonsFileName %s. \n", 
				args->exonsFileName);
		if(ValidateFileName(args->exonsFileName)==0)
			PrintError(FnName, "exonsFileName", "Command line argument", Exit, IllegalFileName);	
	}	
	if(args->numThreads<=0) {		
		PrintError(FnName, "numThreads", "Command line argument", Exit, OutOfRange);
	}

	if(args->tmpDir!=0) {
		fprintf(stderr, "Validating tmpDir path %s. \n",
				args->tmpDir);
		if(ValidatePath(args->tmpDir)==0)
			PrintError(FnName, "tmpDir", "Command line argument", Exit, IllegalPath);	
	}	
	/* If this does not hold, we have done something wrong internally */	
	assert(args->timing == 0 || args->timing == 1);
	assert(args->repeatMasker == 0 || args->repeatMasker == 1);

	/* Cross-check arguments */
	if(args->startContig > args->endContig) {
		PrintError(FnName, "startContig > endContig", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->startContig == args->endContig && args->startPos > args->endPos) {		
		PrintError(FnName, "endPos < startPos with startContig == endContig", "Command line argument", Exit, OutOfRange);	
	}
	if(NULL != args->exonsFileName && args->startContig > 0) {
		PrintError(FnName, "Cannot use -s with -x", "Command line argument", Exit, OutOfRange);	
	}	
	if(NULL != args->exonsFileName && args->startPos > 0) {		
		PrintError(FnName, "Cannot use -S with -x", "Command line argument", Exit, OutOfRange);	
	}
	if(NULL != args->exonsFileName && args->endContig < INT_MAX) {
		PrintError(FnName, "Cannot use -e with -x", "Command line argument", Exit, OutOfRange);	
	}	
	if(NULL != args->exonsFileName && args->endPos < INT_MAX) {		
		PrintError(FnName, "Cannot use -E with -x", "Command line argument", Exit, OutOfRange);	
	}

	return 1;
}

/* TODO */
	void
BfastIndexAssignDefaultValues(struct arguments *args)
{
	/* Assign default values */

	args->programMode = ExecuteProgram;
	args->fastaFileName = NULL;
	args->space = NTSpace;
	args->mask = NULL;
	args->hashWidth=0;
	args->depth=0;
	args->indexNumber=1;
	args->repeatMasker=0;
	args->startContig=0;
	args->startPos=0;
	args->endContig=INT_MAX;
	args->endPos=INT_MAX;
	args->exonsFileName = NULL;
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
BfastIndexPrintProgramParameters(FILE* fp, struct arguments *args)
{
	fprintf(fp, BREAK_LINE);
	fprintf(fp, "Printing Program Parameters:\n");
	fprintf(fp, "programMode:\t\t\t\t%s\n", PROGRAMMODE(args->programMode));
	fprintf(fp, "fastaFileName:\t\t\t\t%s\n", FILEREQUIRED(args->fastaFileName));
	fprintf(fp, "space:\t\t\t\t\t%s\n", SPACE(args->space));
	fprintf(fp, "mask:\t\t\t\t\t%s\n", FILEREQUIRED(args->mask));
	fprintf(fp, "depth:\t\t\t\t\t%d\n", args->depth);
	if(0 < args->hashWidth) {
		fprintf(fp, "hashWidth:\t\t\t\t%d\n", args->hashWidth);
	}
	else {
		fprintf(fp, "hashWidth:\t\t\t\t%s\n", FILEREQUIRED(NULL));
	}
	fprintf(fp, "indexNumber:\t\t\t\t%d\n", args->indexNumber);
	fprintf(fp, "repeatMasker:\t\t\t\t%s\n", INTUSING(args->repeatMasker));
	fprintf(fp, "startContig:\t\t\t\t%d\n", args->startContig);
	fprintf(fp, "startPos:\t\t\t\t%d\n", args->startPos);
	fprintf(fp, "endContig:\t\t\t\t%d\n", args->endContig);
	fprintf(fp, "endPos:\t\t\t\t\t%d\n", args->endPos);
	fprintf(fp, "exonsFileName:\t\t\t\t%s\n", FILEUSING(args->exonsFileName));
	fprintf(fp, "numThreads:\t\t\t\t%d\n", args->numThreads);
	fprintf(fp, "tmpDir:\t\t\t\t\t%s\n", args->tmpDir);
	fprintf(fp, "timing:\t\t\t\t\t%s\n", INTUSING(args->timing));
	fprintf(fp, BREAK_LINE);
	return;
}

/* TODO */
void BfastIndexFreeProgramParameters(struct arguments *args) 
{
	free(args->fastaFileName);
	args->fastaFileName=NULL;
	free(args->mask);
	args->mask=NULL;
	free(args->exonsFileName);
	args->exonsFileName=NULL;
	free(args->tmpDir);
	args->tmpDir=NULL;
}

/* TODO */
void
BfastIndexGetOptHelp() {

	struct argp_option *a=options;
	fprintf(stderr, "\nUsage: bfast index [options]\n");
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
BfastIndexGetOptParse(int argc, char** argv, char OptionString[], struct arguments* arguments) 
{
	int key;
	int OptErr=0;
	while((OptErr==0) && ((key = getopt (argc, argv, OptionString)) != -1)) {
		/*
		   fprintf(stderr, "Key is %c and OptErr = %d\n", key, OptErr);
		   */
		switch (key) {
			case 'd':
				arguments->depth=atoi(optarg);break;
			case 'e':
				arguments->endContig=atoi(optarg);break;
			case 'f':
				arguments->fastaFileName = strdup(optarg); break;
			case 'h':
				arguments->programMode=ExecuteGetOptHelp;break;
			case 'i':
				arguments->indexNumber=atoi(optarg);break;
			case 'm':
				arguments->mask = strdup(optarg); break;
			case 'n':
				arguments->numThreads=atoi(optarg); break;
			case 'p':
				arguments->programMode=ExecutePrintProgramParameters;break;
			case 's':
				arguments->startContig=atoi(optarg);break;
			case 't':
				arguments->timing = 1;break;
			case 'w':
				arguments->hashWidth=atoi(optarg);break;
			case 'x':
				arguments->exonsFileName=strdup(optarg);break;
			case 'A':
				arguments->space=atoi(optarg);break;
			case 'E':
				arguments->endPos=atoi(optarg);break;
			case 'R':
				arguments->repeatMasker=1;break;
			case 'S':
				arguments->startPos=atoi(optarg);break;
			case 'T':
				StringCopyAndReallocate(&arguments->tmpDir, optarg);
				break;
			default:
				OptErr=1;
		} /* while */
	} /* switch */
	return OptErr;
}
