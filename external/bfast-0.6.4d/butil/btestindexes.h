#ifndef BTESTINDEXES_H_
#define BTESTINDEXES_H_

char Algorithm[3][2048] = {"../bfast/bfast/Search for indexes", "Evaluate indexes", "Print Program Parameters"};

typedef struct {
	int algorithm;
	char inputFileName[MAX_FILENAME_LENGTH];
	int readLength;
	int numEventsToSample;
	int numIndexesToSample;
	int keySize;
	int maxKeyWidth;
	int maxIndexSetSize;
	int accuracyThreshold;
	int space;
	int maxNumMismatches;
	int maxInsertionLength;
	int maxNumColorErrors;
} arguments;

/* Command line functions */
void PrintUsage();
void PrintProgramParmeters(arguments *args);
void AssignDefaultValues(arguments *args);
void ValidateArguments(arguments *args);
void ParseCommandLineArguments(int argc, char *argv[], arguments *args);

#endif
