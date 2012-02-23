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
#include "BfastFasta2BRG.h"

/*
   OPTIONS.  Field 1 in ARGP.
   Order of fields: {NAME, KEY, ARG, FLAGS, DOC, OPTIONAL_GROUP_NAME}.
   */
enum { 
	DescInputFilesTitle, DescFastaFileName, 
	DescAlgoTitle, DescSpace, 
	DescOutputTitle, DescTiming,
	DescMiscTitle, DescParameters, DescHelp
};

static struct argp_option options[] = {
	{0, 0, 0, 0, "=========== Input Files =============================================================", 1},
	{"fastaFileName", 'f', "fastaFileName", 0, "Specifies the file name of the FASTA reference genome", 1},
	{0, 0, 0, 0, "=========== Algorithm Options: ======================================================", 2},
	{"space", 'A', "space", 0, "0: NT space 1: Color space", 2},
	{0, 0, 0, 0, "=========== Output Options ==========================================================", 3},
	{"timing", 't', 0, OPTION_NO_USAGE, "Specifies to output timing information", 3},
	{0, 0, 0, 0, "=========== Miscellaneous Options ===================================================", 4},
	{"Parameters", 'p', 0, OPTION_NO_USAGE, "Print program parameters", 4},
	{"Help", 'h', 0, OPTION_NO_USAGE, "Display usage summary", 4},
	{0, 0, 0, 0, 0, 0}
};

static char OptionString[]=
"d:f:o:A:hpt";

	int
BfastFasta2BRG(int argc, char **argv)
{
	struct arguments arguments;
	RGBinary rg;
	time_t startTime = time(NULL);
	time_t endTime;

	if(argc>1) {
		/* Set argument defaults. (overriden if user specifies them)  */ 
		BfastFasta2BRGAssignDefaultValues(&arguments);

		/* Parse command line args */
		if(BfastFasta2BRGGetOptParse(argc, argv, OptionString, &arguments)==0)
		{
			switch(arguments.programMode) {
				case ExecuteGetOptHelp:
					BfastFasta2BRGGetOptHelp();
					break;
				case ExecutePrintProgramParameters:
					BfastFasta2BRGPrintProgramParameters(stderr, &arguments);
					break;
				case ExecuteProgram:
					if(BfastFasta2BRGValidateInputs(&arguments)) {
						fprintf(stderr, "Input arguments look good!\n");
						fprintf(stderr, BREAK_LINE);
					}
					else {
						PrintError("PrintError", NULL, "validating command-line inputs", Exit, InputArguments);
					}
					BfastFasta2BRGPrintProgramParameters(stderr, &arguments);

					/* Read fasta files */
					RGBinaryRead(arguments.fastaFileName, 
							&rg,
							arguments.space);
					/* Write binary */
					RGBinaryWriteBinary(&rg,
							arguments.space,
							arguments.fastaFileName);

					/* Free the Reference Genome */
					RGBinaryDelete(&rg);

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
		BfastFasta2BRGFreeProgramParameters(&arguments);
	}
	else {
		BfastFasta2BRGGetOptHelp();
	}
	return 0;
}

/* TODO */
int BfastFasta2BRGValidateInputs(struct arguments *args) {

	char *FnName="BfastFasta2BRGValidateInputs";

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

	if(args->space != NTSpace && args->space != ColorSpace) {
		PrintError(FnName, "space", "Command line argument", Exit, OutOfRange);	
	}	
	assert(args->timing == 0 || args->timing == 1);
	return 1;
}

/* TODO */
	void
BfastFasta2BRGAssignDefaultValues(struct arguments *args)
{
	/* Assign default values */

	args->programMode = ExecuteProgram;
	args->fastaFileName = NULL;
	args->space = NTSpace;

	args->timing = 0;

	return;
}


/* TODO */
	void 
BfastFasta2BRGPrintProgramParameters(FILE* fp, struct arguments *args)
{
	fprintf(fp, BREAK_LINE);
	fprintf(fp, "Printing Program Parameters:\n");
	fprintf(fp, "programMode:\t\t\t\t%s\n", PROGRAMMODE(args->programMode));
	fprintf(fp, "fastaFileName:\t\t\t\t%s\n", FILEREQUIRED(args->fastaFileName));
	fprintf(fp, "space:\t\t\t\t\t%s\n", SPACE(args->space));
	fprintf(fp, "timing:\t\t\t\t\t%s\n", INTUSING(args->timing));
	fprintf(fp, BREAK_LINE);
	return;
}

/* TODO */
void BfastFasta2BRGFreeProgramParameters(struct arguments *args) 
{
	free(args->fastaFileName);
	args->fastaFileName=NULL;
}

/* TODO */
void
BfastFasta2BRGGetOptHelp() {

	struct argp_option *a=options;
	fprintf(stderr, "\nUsage: bfast fasta2brg [options]\n");
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
BfastFasta2BRGGetOptParse(int argc, char** argv, char OptionString[], struct arguments* arguments) 
{
	int key;
	int OptErr=0;
	while((OptErr==0) && ((key = getopt (argc, argv, OptionString)) != -1)) {
		/*
		   fprintf(stderr, "Key is %c and OptErr = %d\n", key, OptErr);
		   */
		switch (key) {
			case 'f':
				arguments->fastaFileName=strdup(optarg);break;
			case 'h':
				arguments->programMode=ExecuteGetOptHelp; break;
			case 'p':
				arguments->programMode=ExecutePrintProgramParameters; break;
			case 't':
				arguments->timing = 1; break;
			case 'A':
				arguments->space=atoi(optarg); break;
			default:
				fprintf(stderr, "Key is %c and OptErr = %d\n", key, OptErr);
				OptErr=1;
		} /* while */
	} /* switch */
	return OptErr;
}
