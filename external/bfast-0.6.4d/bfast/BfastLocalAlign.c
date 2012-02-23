#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <config.h>
#include <unistd.h>
#include <time.h>

#include "BLibDefinitions.h"
#include "BError.h"
#include "RGBinary.h"
#include "BLib.h"
#include "RunLocalAlign.h"
#include "BfastLocalAlign.h"

/*
   OPTIONS.  Field 1 in ARGP.
   Order of fields: {NAME, KEY, ARG, FLAGS, DOC, OPTIONAL_GROUP_NAME}.
   */
enum { 
	DescInputFilesTitle, DescFastaFileName, DescMatchFileName, DescScoringMatrixFileName, 
	DescAlgoTitle, DescUngapped, DescUnconstrained, DescSpace, DescStartReadNum, DescEndReadNum, DescOffsetLength, DescMaxNumMatches, DescAvgMismatchQuality, DescNumThreads, DescQueueLength,
	DescPairedEndOptionsTitle, DescPairedEndLength, DescMirroringType, DescForceMirroring, 
	DescOutputTitle, DescTiming, 
	DescMiscTitle, DescHelp
};

static struct argp_option options[] = {
	{0, 0, 0, 0, "=========== Input Files =============================================================", 1},
	{"fastaFileName", 'f', "fastaFileName", 0, "Specifies the file name of the FASTA reference genome", 1},
	{"matchFileName", 'm', "matchFileName", 0, "Specifies the bfast matches file", 1},
	{"scoringMatrixFileName", 'x', "scoringMatrixFileName", 0, "Specifies the file name storing the scoring matrix", 1},
	{0, 0, 0, 0, "=========== Algorithm Options =======================================================", 1},
	{"ungapped", 'u', 0, OPTION_NO_USAGE, "Do ungapped local alignment (the default is gapped).", 2},
	{"unconstrained", 'U', 0, OPTION_NO_USAGE, "Do not use mask constraints from the match step", 2},
	{"space", 'A', "space", 0, "0: NT space 1: Color space", 2},
	{"startReadNum", 's', "startReadNum", 0, "Specifies the read to begin with (skip the first" 
		"\n\t\t\t  startReadNum-1 reads)", 2},
	{"endReadNum", 'e', "endReadNum", 0, "Specifies the last read to use (inclusive)", 2},
	{"offsetLength", 'o', "offset", 0, "Specifies the number of bases before and after the match to"
		"\n\t\t\t  include in the reference genome", 2},
	{"maxNumMatches", 'M', "maxNumMatches", 0, "Specifies the maximum number of candidates to initiate"
		"\n\t\t\t  alignment for a given match", 2},
	{"avgMismatchQuality", 'q', "avgMismatchQuality", 0, "Specifies the average mismatch quality", 2},
	{"numThreads", 'n', "numThreads", 0, "Specifies the number of threads to use (Default 1)", 2},
	{"queueLength", 'Q', "queueLength", 0, "Specifies the number of reads to cache", 2},
	/*
	{0, 0, 0, 0, "=========== Paired End Options ======================================================", 3},
	{"pairedEndLength", 'l', "pairedEndLength", 0, "Specifies that if one read of the pair has CALs and the other"
		"\n\t\t\t  does not,"
			"\n\t\t\t  this distance will be used to infer the latter read's CALs", 3},
	{"mirroringType", 'L', "mirroringType", 0, "Specifies how to infer the other end (with -l)"
		"\n\t\t\t  0: No mirroring should occur"
			"\n\t\t\t  1: specifies that we assume that the first end is before the"
			"\n\t\t\t    second end (5'->3')"
			"\n\t\t\t  2: specifies that we assume that the second end is before"
			"\n\t\t\t    the first end (5'->3')"
			"\n\t\t\t  3: specifies that we mirror CALs in both directions", 3},
	{"forceMirroring", 'F', 0, OPTION_NO_USAGE, "Specifies that we should always mirror CALs using the distance"
		"\n\t\t\t  from -l", 3},
		*/
	{0, 0, 0, 0, "=========== Output Options ==========================================================", 4},
	{"timing", 't', 0, OPTION_NO_USAGE, "Specifies to output timing information", 4},
	{0, 0, 0, 0, "=========== Miscellaneous Options ===================================================", 5},
	{"Parameters", 'p', 0, OPTION_NO_USAGE, "Print program parameters", 5},
	{"Help", 'h', 0, OPTION_NO_USAGE, "Display usage summary", 5},
	{0, 0, 0, 0, 0, 0}
};

static char OptionString[]=
"e:f:m:n:o:q:s:x:A:M:Q:T:hptuU";
//"e:f:l:m:n:o:q:s:x:A:L:M:Q:T:hptuFU";

	int
BfastLocalAlign(int argc, char **argv)
{
	struct arguments arguments;
	time_t startTotalTime = time(NULL);
	time_t endTotalTime;
	int seconds, minutes, hours;
	if(argc>1) {
		/* Set argument defaults. (overriden if user specifies them)  */ 
		BfastLocalAlignAssignDefaultValues(&arguments);

		/* Parse command line args */
		if(BfastLocalAlignGetOptParse(argc, argv, OptionString, &arguments)==0)
		{
			switch(arguments.programMode) {
				case ExecuteGetOptHelp:
					BfastLocalAlignGetOptHelp();
					break;
				case ExecutePrintProgramParameters:
					BfastLocalAlignPrintProgramParameters(stderr, &arguments);
					break;
				case ExecuteProgram:
					if(BfastLocalAlignValidateInputs(&arguments)) {
						if(0 <= VERBOSE) {
							fprintf(stderr, "**** Input arguments look good! *****\n");
							fprintf(stderr, BREAK_LINE);
						}
					}
					else {
						PrintError("PrintError", NULL, "validating command-line inputs", Exit, InputArguments);

					}
					BfastLocalAlignPrintProgramParameters(stderr, &arguments);
					/* Execute Program */
					/* Run the aligner */
					RunAligner(arguments.fastaFileName,
							arguments.matchFileName,
							arguments.scoringMatrixFileName,
							arguments.ungapped,
							arguments.unconstrained,
							AllAlignments, // hard-coded
							arguments.space,
							arguments.startReadNum,
							arguments.endReadNum,
							arguments.offsetLength,
							arguments.maxNumMatches,
							arguments.avgMismatchQuality,
							arguments.numThreads,
							arguments.queueLength,
							arguments.usePairedEndLength,
							arguments.pairedEndLength,
							arguments.mirroringType,
							arguments.forceMirroring,
							arguments.timing,
							stdout);

					if(arguments.timing == 1) {

						/* Output total time */
						endTotalTime = time(NULL);
						seconds = endTotalTime - startTotalTime;
						hours = seconds/3600;
						seconds -= hours*3600;
						minutes = seconds/60;
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
		BfastLocalAlignFreeProgramParameters(&arguments);
	}
	else {
		BfastLocalAlignGetOptHelp();
	}
	return 0;
}

int BfastLocalAlignValidateInputs(struct arguments *args) {

	char *FnName="BfastLocalAlignValidateInputs";
	
	/* Check if we are piping */
	if(NULL == args->matchFileName) {
		VERBOSE = -1;
	}

	if(0 <= VERBOSE) {
		fprintf(stderr, BREAK_LINE);
		fprintf(stderr, "Checking input parameters supplied by the user ...\n");
	}

	if(args->fastaFileName!=0) {
		if(0 <= VERBOSE) {
			fprintf(stderr, "Validating fastaFileName %s. \n",
					args->fastaFileName);
		}
		if(ValidateFileName(args->fastaFileName)==0)
			PrintError(FnName, "fastaFileName", "Command line argument", Exit, IllegalFileName);        }   
	else {
		PrintError(FnName, "fastaFileName", "Required command line argument", Exit, IllegalFileName);   
	}

	if(args->matchFileName!=0) {
		if(0 <= VERBOSE) {
			fprintf(stderr, "Validating matchFileName%s. \n", 
					args->matchFileName);
		}
		if(ValidateFileName(args->matchFileName)==0)
			PrintError(FnName, "matchFileName", "Command line argument", Exit, IllegalFileName);	
	}	
	if(args->scoringMatrixFileName!=0) {		
		if(0 <= VERBOSE) {
			fprintf(stderr, "Validating scoringMatrixFileName path %s. \n", 
					args->scoringMatrixFileName);
		}
		if(ValidateFileName(args->scoringMatrixFileName)==0)
			PrintError(FnName, "scoringMatrixFileName", "Command line argument", Exit, IllegalFileName);	
	}	
	if(args->ungapped != Gapped && args->ungapped != Ungapped) {		
		PrintError(FnName, "ungapped", "Command line argument", Exit, OutOfRange);
	}

	if(args->space != NTSpace && args->space != ColorSpace) {		
		PrintError(FnName, "space", "Command line argument", Exit, OutOfRange);
	}

	if(args->startReadNum < 0) {
		PrintError(FnName, "startReadNum", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->endReadNum < 0) {		
		PrintError(FnName, "endReadNum", "Command line argument", Exit, OutOfRange);
	}

	if(args->offsetLength < 0) {
		PrintError(FnName, "offsetLength", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->maxNumMatches < 0) {		
		PrintError(FnName, "maxNumMatches", "Command line argument", Exit, OutOfRange);
	}

	if(args->avgMismatchQuality <= 0) {
		PrintError(FnName, "avgMismatchQuality", "Command line argument", Exit, OutOfRange);	
	}	
	if(args->numThreads<=0) {		
		PrintError(FnName, "numThreads", "Command line argument", Exit, OutOfRange);
	} 

	if(args->queueLength<=0) {
		PrintError(FnName, "queueLength", "Command line argument", Exit, OutOfRange);	
	}	

	/* If this does not hold, we have done something wrong internally */	
	assert(args->timing == 0 || args->timing == 1);
	assert(args->usePairedEndLength == 0 || args->usePairedEndLength == 1);
	assert(args->forceMirroring == 0 || args->forceMirroring == 1);
	assert(NoMirroring <= args->mirroringType && args->mirroringType <= MirrorBoth);
	assert(Ungapped == args->ungapped || Gapped == args->ungapped);
	assert(Unconstrained == args->unconstrained || Constrained == args->unconstrained);
	if(args->mirroringType != NoMirroring && args->usePairedEndLength == 0) {
		PrintError(FnName, "pairedEndLength", "Must specify a paired end length when using mirroring", Exit, OutOfRange);	
	}	
	if(args->forceMirroring == 1 && args->usePairedEndLength == 0) {		
		PrintError(FnName, "pairedEndLength", "Must specify a paired end length when using force mirroring", Exit, OutOfRange);	
	}

	return 1;
}

	void 
BfastLocalAlignAssignDefaultValues(struct arguments *args)
{
	/* Assign default values */

	args->programMode = ExecuteProgram;

	args->fastaFileName = NULL;
	args->matchFileName = NULL;
	args->scoringMatrixFileName=NULL;

	args->ungapped = Gapped;
	args->unconstrained = Constrained; 
	args->space = NTSpace;
	args->startReadNum=1;
	args->endReadNum=INT_MAX;
	args->offsetLength=OFFSET_LENGTH;
	args->maxNumMatches=MAX_NUM_MATCHES;
	args->avgMismatchQuality=AVG_MISMATCH_QUALITY;
	args->numThreads = 1;
	args->queueLength = DEFAULT_LOCALALIGN_QUEUE_LENGTH;
	args->usePairedEndLength = 0;
	args->pairedEndLength = 0;
	args->mirroringType = NoMirroring;
	args->forceMirroring = 0;

	args->timing = 0;

	return;
}

	void 
BfastLocalAlignPrintProgramParameters(FILE* fp, struct arguments *args)
{
	if(0 <= VERBOSE) {
		fprintf(fp, BREAK_LINE);
		fprintf(fp, "Printing Program Parameters:\n");
		fprintf(fp, "programMode:\t\t\t\t%s\n", PROGRAMMODE(args->programMode));
		fprintf(fp, "fastaFileName:\t\t\t\t%s\n", FILEREQUIRED(args->fastaFileName));
		fprintf(fp, "matchFileName:\t\t\t\t%s\n", FILESTDIN(args->matchFileName));
		fprintf(fp, "scoringMatrixFileName:\t\t\t%s\n", FILEUSING(args->scoringMatrixFileName));
		fprintf(fp, "ungapped:\t\t\t\t%s\n", INTUSING(args->ungapped));
		fprintf(fp, "unconstrained:\t\t\t\t%s\n", INTUSING(args->unconstrained));
		fprintf(fp, "space:\t\t\t\t\t%s\n", SPACE(args->space));
		fprintf(fp, "startReadNum:\t\t\t\t%d\n", args->startReadNum);
		fprintf(fp, "endReadNum:\t\t\t\t%d\n", args->endReadNum);
		fprintf(fp, "offsetLength:\t\t\t\t%d\n", args->offsetLength);
		fprintf(fp, "maxNumMatches:\t\t\t\t%d\n", args->maxNumMatches);
		fprintf(fp, "avgMismatchQuality:\t\t\t%d\n", args->avgMismatchQuality); 
		fprintf(fp, "numThreads:\t\t\t\t%d\n", args->numThreads);
		fprintf(fp, "queueLength:\t\t\t\t%d\n", args->queueLength);
		/*
		if(1 == args->usePairedEndLength) fprintf(fp, "pairedEndLength:\t\t\t%d\n", args->pairedEndLength);
		else fprintf(fp, "pairedEndLength:\t\t\t%s\n", INTUSING(args->usePairedEndLength));
		fprintf(fp, "mirroringType:\t\t\t\t%s\n", MIRRORINGTYPE(args->mirroringType));
		fprintf(fp, "forceMirroring:\t\t\t\t%s\n", INTUSING(args->forceMirroring));
		*/
		fprintf(fp, "timing:\t\t\t\t\t%s\n", INTUSING(args->timing));
		fprintf(fp, BREAK_LINE);
	}
	return;

}

/* TODO */
void BfastLocalAlignFreeProgramParameters(struct arguments *args)
{
	free(args->fastaFileName);
	args->fastaFileName=NULL;
	free(args->matchFileName);
	args->matchFileName=NULL;
	free(args->scoringMatrixFileName);
	args->scoringMatrixFileName=NULL;
}

void
BfastLocalAlignGetOptHelp() {

	struct argp_option *a=options;
	fprintf(stderr, "\nUsage: bfast localalign [options]\n");
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
	fprintf(stderr, "\n send bugs to %s\n", PACKAGE_BUGREPORT);
	return;
}

	int
BfastLocalAlignGetOptParse(int argc, char** argv, char OptionString[], struct arguments* arguments) 
{
	int key;
	int OptErr=0;
	while((OptErr==0) && ((key = getopt (argc, argv, OptionString)) != -1)) {
		/*
		   fprintf(stderr, "Key is %c and OptErr = %d\n", key, OptErr);
		   */
		switch (key) {
			case 'e':
				arguments->endReadNum=atoi(optarg);break;
			case 'f':
				arguments->fastaFileName=strdup(optarg);break;
			case 'h':
				arguments->programMode=ExecuteGetOptHelp; break;
				/*
			case 'l':
				arguments->usePairedEndLength=1;
				arguments->pairedEndLength = atoi(optarg);break;
				*/
			case 'm':
				arguments->matchFileName=strdup(optarg);break;
			case 'n':
				arguments->numThreads=atoi(optarg); break;
			case 'o':
				arguments->offsetLength=atoi(optarg);break;
			case 'p':
				arguments->programMode=ExecutePrintProgramParameters; break;
			case 'q':
				arguments->avgMismatchQuality = atoi(optarg); break;
			case 's':
				arguments->startReadNum=atoi(optarg);break;
			case 't':
				arguments->timing = 1;break;
			case 'u':
				arguments->ungapped = Ungapped; break;
			case 'x':
				StringCopyAndReallocate(&arguments->scoringMatrixFileName, optarg);
				break;
			case 'A':
				arguments->space=atoi(optarg);break;
				/*
			case 'F':
				arguments->forceMirroring=1;break;
			case 'L':
				arguments->mirroringType=atoi(optarg);break;
				*/
			case 'M':
				arguments->maxNumMatches=atoi(optarg);break;
			case 'Q':
				arguments->queueLength=atoi(optarg);break;
			case 'U':
				arguments->unconstrained=Unconstrained;break;
			default:
				OptErr=1;
		} /* while */
	} /* switch */
	return OptErr;
}
