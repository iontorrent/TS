#include <stdio.h>
#include <stdlib.h>
#include "ion_error.h"

static char 
error_string[][64] =
{ 
  "value out of range",
  "command line argument",
  "could not re-allocate memory",
  "could not allocate memory",
  "could not open file",
  "could not read from file",
  "could not write to file",
  "encountered early end-of-file",
  "last error type"
};	   

static char 
action_string[][20] =
{"Fatal Error", "Warning", "LastActionType"};

void 
ion_error(const char *function_name, const char *variable_name, int action_type, int error_type) 
{
  fprintf(stderr, "%s\rIn function \"%s\":\n\t%s: %s.\n", 
          BREAK_LINE, function_name, action_string[action_type], error_string[error_type]);

  /* Only print variable name if is available */
  if(NULL != variable_name) {
      fprintf(stderr, "\tVariable/Value: %s.\n", variable_name);
  }
  if(error_type == ReadFileError 
     || error_type == OpenFileError 
     || error_type == WriteFileError 
     || error_type == EndOfFile) {
      perror("The file stream error was:");
  }

  switch(action_type) {
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
