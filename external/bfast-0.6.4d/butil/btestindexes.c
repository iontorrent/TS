#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <config.h>
#include <assert.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>  

#include "../bfast/BError.h"
#include "../bfast/BLibDefinitions.h"
#include "../bfast/RGIndexAccuracy.h"
#include "../bfast/BLib.h"
#include "btestindexes.h"

#define Name "btestindexes"

/* Is a utility that tests, searches for, and compares layouts for indexes against certain events,
 * such as errors, mismatches and insertions.
 * */

void PrintUsage()
{
	fprintf(stderr, "%s %s\n", "bfast", PACKAGE_VERSION);
	fprintf(stderr, "\nUsage:%s [options] <files>\n", Name);
	fprintf(stderr, "******************************* Algorithm Options (no defaults) *******************************\n");
	fprintf(stderr, "\t-a\tINT\talgorithm\n\t\t\t\t0: search for indexes\n\t\t\t\t1: evaluate indexes\n");
	fprintf(stderr, "\t-r\tINT\tread length (for all) \n");
	fprintf(stderr, "\t-S\tINT\tnumber of events to sample\n");
	fprintf(stderr, "\t-A\tINT\tspace 0: nucleotide space 1: color space\n");
	fprintf(stderr, "******************************* Search Options (for -a 0) *************************************\n");
	fprintf(stderr, "\t-s\tINT\tnumber of indexes to sample\n");
	fprintf(stderr, "\t-l\tINT\tkey size\n");
	fprintf(stderr, "\t-w\tINT\tmaximum key width\n");
	fprintf(stderr, "\t-n\tINT\tmaximum index set size\n");
	fprintf(stderr, "\t-t\tINT\taccuracy percent threshold (0-100)\n");
	fprintf(stderr, "******************************* Evaluate Options (for -a 1) ***********************************\n");
	fprintf(stderr, "\t-f\tSTRING\tinput file name\n");
	fprintf(stderr, "\t-I\tINT\tmaximum insertion length (-a 1)\n");
	fprintf(stderr, "******************************* Event Options (default =0 ) ***********************************\n");
	fprintf(stderr, "\t-M\tINT\tmaximum number of mismatches\n");
	fprintf(stderr, "\t-E\tINT\tmaximum number of color errors (-A 1)\n");
	fprintf(stderr, "******************************* Miscellaneous Options  ****************************************\n");
	fprintf(stderr, "\t-p\tNULL\tprints the program parameters\n");
	fprintf(stderr, "\t-h\tNULL\tprints this message\n");
	fprintf(stderr, "\nsend bugs to %s\n",
			PACKAGE_BUGREPORT);
}

void PrintProgramParameters(arguments *args)
{
	/* Print program parameters */
	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "Printing program parameters:\n");
	fprintf(stderr, "algorithm:\t\t\t%d\t[%s]\n", args->algorithm, Algorithm[args->algorithm]); 
	fprintf(stderr, "read length:\t\t\t%d\n", args->readLength);
	fprintf(stderr, "number of events to sample:\t%d\n", args->numEventsToSample);
	fprintf(stderr, "space:\t\t\t\t%d\n", args->space);
	fprintf(stderr, "number of indexes to sample:\t%d\n", args->numIndexesToSample);
	fprintf(stderr, "key size:\t\t\t%d\n", args->keySize);
	fprintf(stderr, "key width:\t\t\t%d\n", args->maxKeyWidth);
	fprintf(stderr, "max index set size:\t\t%d\n", args->maxIndexSetSize);
	fprintf(stderr, "accuracy percent threshold:\t%d\n", args->accuracyThreshold);
	fprintf(stderr, "input file name:\t\t%s\n", args->inputFileName);
	fprintf(stderr, "maximum insertion length:\t%d\n", args->maxInsertionLength);
	fprintf(stderr, "maximum number of mismatches:\t%d\n", args->maxNumMismatches);
	fprintf(stderr, "maximum number of color errors:\t%d\n", args->maxNumColorErrors);
	fprintf(stderr, "%s", BREAK_LINE);
}

void AssignDefaultValues(arguments *args) 
{
	args->algorithm=0;
	strcpy(args->inputFileName, "\0");
	args->readLength=0;
	args->numEventsToSample=0;
	args->numIndexesToSample=0;
	args->keySize=0;
	args->maxKeyWidth=0;
	args->maxIndexSetSize=0;
	args->accuracyThreshold=0;
	args->space=0;
	args->maxNumMismatches=0;
	args->maxInsertionLength=0;
	args->maxNumColorErrors=0;
}

void ValidateArguments(arguments *args)
{
	char *FnName="ValidateArguments";

	if(args->algorithm < 0 || args->algorithm > 2) {
		PrintError(FnName, "Command line argument", "algorithm", Exit, OutOfRange);	}	if(args->readLength <= 0) {		PrintError(FnName, "Command line argument", "readLength", Exit, OutOfRange);	}
	if(args->numEventsToSample <= 0) {
		PrintError(FnName, "Command line argument", "numEventsToSample", Exit, OutOfRange);	}	if(args->numIndexesToSample < 0 ||			(args->algorithm == 0 && args->numIndexesToSample <= 0)) {		PrintError(FnName, "Command line argument", "numIndexesToSample", Exit, OutOfRange);
		}
	if(args->algorithm == 0) {
		if(args->keySize <= 0) {
			PrintError(FnName, "Command line argument", "keySize", Exit, OutOfRange);		}		if(args->maxKeyWidth <= 0) {			PrintError(FnName, "Command line argument", "maxKeyWidth", Exit, OutOfRange);		}
		if(args->maxIndexSetSize <= 0) {
			PrintError(FnName, "Command line argument", "maxIndexSetSize", Exit, OutOfRange);		}		if(args->accuracyThreshold < 0) {			PrintError(FnName, "Command line argument", "accuracyThreshold", Exit, OutOfRange);		}
		if(args->keySize > args->maxKeyWidth) {
			PrintError(FnName, "Command line argument", "keySize > maxKeyWidth", Exit, OutOfRange);		}		if(args->keySize > args->readLength) {			PrintError(FnName, "Command line argument", "keySize > readLength", Exit, OutOfRange);		}
		if(args->maxKeyWidth > args->readLength) {
			PrintError(FnName, "Command line argument", "maxKeyWidth > readLength", Exit, OutOfRange);		}	}	if(args->space < 0 || 1 < args->space) {		PrintError(FnName, "Command line argument", "space", Exit, OutOfRange);
			}
		if(args->maxNumMismatches < 0) {
			PrintError(FnName, "Command line argument", "maxNumMismatches", Exit, OutOfRange);	}	if(args->maxInsertionLength < 0) {		PrintError(FnName, "Command line argument", "maxInsertionLength", Exit, OutOfRange);	}
		if(args->space == 1) {
			if(args->maxNumColorErrors < 0) {
				PrintError(FnName, "Command line argument", "maxNumColorErrors", Exit, OutOfRange);		}	}	else {		if(args->maxNumColorErrors > 0) {
					PrintError(FnName, "Command line argument", "maxNumColorErrors", Exit, OutOfRange);		}	}}
void ParseCommandLineArguments(int argc, char *argv[], arguments *args) 
{
	int i;
	if(argc==1) {
		PrintUsage();
		exit(1);
	}
	for(i=1;i<argc;i+=2) {
		if(argv[i][0] != '-' ||
				strlen(argv[i]) != 2) {
			fprintf(stderr, "*** Error.  Could not understand command line option %s.  Terminating! ***\n",
					argv[i]);
			exit(1);
		}
		switch(argv[i][1]) {
			case 'a':
				args->algorithm = atoi(argv[i+1]);
				break;
			case 'A':
				args->space = atoi(argv[i+1]);
				break;
			case 'E':
				args->maxNumColorErrors = atoi(argv[i+1]);
				break;
			case 'f':
				strcpy(args->inputFileName, argv[i+1]);
				break;
			case 'I':
				args->maxInsertionLength = atoi(argv[i+1]);
				break;
			case 'h':
				PrintUsage();
				exit(1);
				break;
			case 'l':
				args->keySize = atoi(argv[i+1]);
				break;
			case 'M':
				args->maxNumMismatches = atoi(argv[i+1]);
				break;
			case 'n':
				args->maxIndexSetSize = atoi(argv[i+1]);
				break;
			case 'p':
				args->algorithm = ProgramParameters;
				break;
			case 'r':
				args->readLength = atoi(argv[i+1]);
				break;
			case 's':
				args->numIndexesToSample = atoi(argv[i+1]);
				break;
			case 'S':
				args->numEventsToSample = atoi(argv[i+1]);
				break;
			case 't':
				args->accuracyThreshold = atoi(argv[i+1]);
				break;
			case 'w':
				args->maxKeyWidth = atoi(argv[i+1]);
				break;
			default:
				fprintf(stderr, "*** Error.  Could not understand command line option %s.  Terminating! ***\n",
						argv[i]);
				exit(1);
				break;

		}
	}
}

int main(int argc, char *argv[])
{
	/* Command line arguments */
	arguments args;

	/* Assign default values */
	AssignDefaultValues(&args);

	/* Parse command line arguments */
	ParseCommandLineArguments(argc, argv, &args);

	/* Validate command line arguments */
	ValidateArguments(&args);

	/* Print program parameters */
	PrintProgramParameters(&args);

	switch(args.algorithm) {
		case SearchForRGIndexAccuracies:
			RunSearchForRGIndexAccuracies(args.readLength,
					args.numEventsToSample,
					args.numIndexesToSample,
					args.keySize,
					args.maxKeyWidth,
					args.maxIndexSetSize,
					args.accuracyThreshold,
					args.space,
					args.maxNumMismatches,
					args.maxNumColorErrors);
			break;
		case EvaluateRGIndexAccuracies:
			RunEvaluateRGIndexAccuracies(args.inputFileName,
					args.readLength,
					args.numEventsToSample,
					args.space,
					args.maxNumMismatches,
					args.maxInsertionLength,
					args.maxNumColorErrors);
			break;
		case ProgramParameters:
			/* Do nothing */
			break;
		default:
			fprintf(stderr, "Error.  Could not understand program mode [%d].  Terminating!\n", args.algorithm);
			exit(1);

	}
	return 0;
}
