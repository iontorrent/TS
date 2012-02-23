#ifndef BFASTALIGN_H_
#define BFASTALIGN_H_

#include <stdio.h>

/* This structure is used by main to communicate with parse_opt. */
struct arguments
{
	char *args[1];							/* No arguments to this function */
	char *fastaFileName;                   	/* -f */
	char *readsFileName;					/* -r */
	int compression;						/* -j, -z */ 
	int space;								/* -A */
	int numThreads;							/* -n */
	char *tmpDir;							/* -T */
	int timing;								/* -t */
	int programMode;						/* -h */ 
};

/* Local functions */
int BfastAlignValidateInputs(struct arguments*);
void BfastAlignAssignDefaultValues(struct arguments*);
void BfastAlignPrintProgramParameters(FILE*, struct arguments*);
void BfastAlignFreeProgramParameters(struct arguments *args);
void BfastAlignPrintGetOptHelp();
void BfastAlignGetOptHelp();
void BfastAlignPrintGetOptHelp();
struct argp_option {
	char *name; /* Arg name */
	int key;
	char *arg; /* arg symbol */
	int flags; 
	char *doc; /* short info about the arg */
	int group;
};
int BfastAlignGetOptParse(int, char**, char*, struct arguments*); 
#endif
