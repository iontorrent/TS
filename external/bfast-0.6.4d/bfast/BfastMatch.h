#ifndef BFASTMATCH_H_
#define BFASTMATCH_H_

#include <stdio.h>

/* This structure is used by main to communicate with parse_opt. */
struct arguments
{
	char *args[1];							/* No arguments to this function */
	char *fastaFileName;                   	/* -f */
	char *mainIndexes;						/* -i */
	char *secondaryIndexes;					/* -I */
	char *readsFileName;					/* -r */
	char *offsets;							/* -o */
	int loadAllIndexes;						/* -l */
	int compression;						/* -j, -z */ 
	int space;								/* -A */
	int startReadNum;						/* -s */
	int endReadNum;							/* -e */
	int keySize;							/* -k */
	int maxKeyMatches;						/* -K */
	int maxNumMatches;						/* -M */
	int whichStrand;						/* -w */
	int numThreads;							/* -n */
	int queueLength;						/* -Q */
	char *tmpDir;							/* -T */
	int timing;								/* -t */
	int programMode;						/* -h */ 
};

/* Local functions */
int BfastMatchValidateInputs(struct arguments*);
void BfastMatchAssignDefaultValues(struct arguments*);
void BfastMatchPrintProgramParameters(FILE*, struct arguments*);
void BfastMatchFreeProgramParameters(struct arguments *args);
void BfastMatchPrintGetOptHelp();
void BfastMatchGetOptHelp();
void BfastMatchPrintGetOptHelp();
struct argp_option {
	char *name; /* Arg name */
	int key;
	char *arg; /* arg symbol */
	int flags; 
	char *doc; /* short info about the arg */
	int group;
};
int BfastMatchGetOptParse(int, char**, char*, struct arguments*); 
#endif
