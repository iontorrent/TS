#ifndef BERROR_H_
#define BERROR_H_

/* Action */
enum {Exit, Warn, LastActionType};

/* Type */
enum {
	Dummy,
	OutOfRange, /* e.g. command line args */
	InputArguments,
	IllegalFileName,   
	IllegalPath,
	OpenFileError,
	EndOfFile,
	ReallocMemory,
	MallocMemory,
	ThreadError,
	ReadFileError,
	WriteFileError,
	DeleteFileError,
	LastErrorType,
};

void PrintError(char*, char*, char*, int, int);

#endif
