#include <stdio.h>
#include <stdlib.h>
#include "BError.h"
#include "BLibDefinitions.h"

int32_t VERBOSE = 0;

static char ErrorString[][20]=
{ "\0", "OutOfRange", "InputArguments", "IllegalFileName", "IllegalPath", "OpenFileError", "EndOfFile", "ReallocMemory", "MallocMemory", "ThreadError", "ReadFileError", "WriteFileError", "DeleteFileError"}; 
static char ActionType[][20]={"Fatal Error", "Warning"};

	void
PrintError(char* FunctionName, char *VariableName, char* Message, int Action, int type)
{	
	fprintf(stderr, "%s\rIn function \"%s\": %s[%s]. ", 
			BREAK_LINE, FunctionName, ActionType[Action], ErrorString[type]);

	/* Only print variable name if is available */
	if(VariableName) {
		fprintf(stderr, "Variable/Value: %s.\n", VariableName);
	}
	/* Only print message name if is available */
	if(Message) { 
		fprintf(stderr, "Message: %s.\n", Message);
	}
	if(type == ReadFileError || 
			type == OpenFileError || 
			type == WriteFileError) {
		perror("The file stream error was:");
	}

	switch(Action) {
		case Exit: 
			fprintf(stderr, " ***** Exiting due to errors *****\n"); 
			fprintf(stderr, "%s", BREAK_LINE);
			exit(EXIT_FAILURE); 
			break; /* Not necessary actually! */
		case Warn:
			fprintf(stderr, " ***** Warning *****\n");
			fprintf(stderr, "%s", BREAK_LINE);
			break;
		default:
			fprintf(stderr, "Trouble!!!\n");
			fprintf(stderr, "%s", BREAK_LINE);
	}
}
