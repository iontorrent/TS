#ifndef BFASTINDEX_H_
#define BFASTINDEX_H_

/* This structure is used by main to communicate with parse_opt. */
struct arguments
{
	char *args[1];							/* No arguments to this function */
	char *fastaFileName;					/* -f */
	int space;								/* -A */
	char *mask;								/* -m */
	int hashWidth;							/* -w */
	int depth;								/* -D */
	int indexNumber;						/* -i */
	int numThreads;                         /* -n */
	int repeatMasker;						/* -R */
	int startContig;						/* -s */
	unsigned int startPos;					/* -S */
	int endContig;							/* -e */
	unsigned int endPos;					/* -E */
	char *exonsFileName;					/* -x */
	char *tmpDir;                           /* -T */
	int timing;                             /* -t */
	int programMode;						/* -h */ 
};

/* Local functions */
int BfastIndexValidateInputs(struct arguments *args);
void BfastIndexAssignDefaultValues(struct arguments*);
void BfastIndexPrintProgramParameters(FILE*, struct arguments*);
void BfastIndexFreeProgramParameters(struct arguments *args);
void BfastIndexPrintGetOptHelp();
void BfastIndexGetOptHelp();
void BfastIndexPrintGetOptHelp();
struct argp_option {
	char *name; /* Arg name */
	int key;
	char *arg; /* arg symbol */
	int flags; 
	char *doc; /* short info about the arg */
	int group;
};
int BfastIndexGetOptParse(int, char**, char*, struct arguments*); 
#endif
