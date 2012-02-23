#ifndef BFASTFASTA2BRG_H_ 
#define BFASTFASTA2BRG_H_ 
/* This structure is used by main to communicate with parse_opt. */
struct arguments
{
	char *args[1];							/* No arguments to this function */
	char *fastaFileName;					/* -f */
	int space;								/* -A */
	int timing;                             /* -t */
	int programMode;						/* -h */ 
};

/* Local functions */
int BfastFasta2BRGValidateInputs(struct arguments *args);
void BfastFasta2BRGAssignDefaultValues(struct arguments*);
void BfastFasta2BRGPrintProgramParameters(FILE*, struct arguments*);
void BfastFasta2BRGFreeProgramParameters(struct arguments *args);
void BfastFasta2BRGPrintGetOptHelp();
void BfastFasta2BRGGetOptHelp();
void BfastFasta2BRGPrintGetOptHelp();
struct argp_option {
	char *name; /* Arg name */
	int key;
	char *arg; /* arg symbol */
	int flags; 
	char *doc; /* short info about the arg */
	int group;
};
int BfastFasta2BRGGetOptParse(int, char**, char*, struct arguments*); 
#endif
