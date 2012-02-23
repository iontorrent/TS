#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <zlib.h>
#include <config.h>
#include <unistd.h>

#include "RGMatches.h"
#include "BLibDefinitions.h"
#include "BError.h"
#include "BLib.h"

#define Name "bfast bmfconvert"
#define BMFCONVERT_ROTATE_NUM 100000
#define BMFCONVERT_FASTQ 2

/* Converts a bmatches file from binary to plaintext or vice versa.
 * */

int BfastBMFConvertUsage()
{
		fprintf(stderr, "\nUsage:%s [options] <files>\n", Name);
		fprintf(stderr, "\t-O\t\toutput type:\n"
				"\t\t\t\t0-BMF text to BMF binary\n"
				"\t\t\t\t1-BMF binary to BMF text\n"
				"\t\t\t\t2-BMF binary to FASTQ\n");
		fprintf(stderr, "\t-h\t\tprints this help message\n");
		fprintf(stderr, "\nsend bugs to %s\n",
				PACKAGE_BUGREPORT);
		return 1;
}

int BfastBMFConvert(int argc, char *argv[])
{

	FILE *fpIn=NULL, *fpOut=NULL;
	gzFile fpInGZ=NULL, fpOutGZ=NULL; 
	int binaryInput = 0, binaryOutput = 0;
	long long int counter;
	char *inputFileName=NULL;
	char outputFileName[MAX_FILENAME_LENGTH]="\0";
	int outputType = 1;
	int c, argnum;
	char *last;
	RGMatches m;
	char fileExtension[256]="\0";

	// Get parameters
	while((c = getopt(argc, argv, "O:h")) >= 0) {
		switch(c) {
			case 'O': outputType=atoi(optarg); break;
			case 'h':
					  BfastBMFConvertUsage(); return 1;
			default: fprintf(stderr, "Unrecognized option: -%c\n", c); return 1;
		}
	}

	if(argc == optind) {
		BfastBMFConvertUsage(); 
		return 1;
	}
		
	switch(outputType) {			
			case 0:
				binaryInput = TextInput;
				binaryOutput = BinaryOutput;
				strcat(fileExtension, BFAST_MATCHES_FILE_EXTENSION);
				break;
			case 1:
				binaryInput = BinaryInput;
				binaryOutput = TextOutput;
				strcat(fileExtension, "txt");
				break;
			case 2:
				binaryInput = BinaryInput;
				strcat(fileExtension, BFAST_MATCHES_READS_FILTERED_FILE_EXTENSION);
				break;
			default:
				PrintError(Name, NULL, "Could not understand output type", Exit, OutOfRange);		
		}		

	for(argnum = optind; argnum < argc; argnum++) {

		assert(argnum<argc);
		inputFileName = strdup(argv[argnum]);

		/* Create the file output file name */
		last = StrStrGetLast(inputFileName,
				BFAST_MATCHES_FILE_EXTENSION);
		if(NULL == last) {
			last = StrStrGetLast(inputFileName, "txt");
			if(NULL == last) {
				PrintError(Name, inputFileName, "Could not recognize file extension", Exit, OutOfRange);		
			} 		
		}
		outputFileName[0]='\0';
		strncpy(outputFileName, inputFileName, (last - inputFileName));		
		outputFileName[(last-inputFileName)]='\0';
		strcat(outputFileName, fileExtension);
		
		/* Open the input file */		
		if(TextInput == binaryInput) {			
			if(!(fpIn=fopen(inputFileName, "rb"))) {
				PrintError(Name, inputFileName, "Could not open file for reading", Exit, OpenFileError);			
			}		
		}		
		else {			
			if(!(fpInGZ=gzopen(inputFileName, "rb"))) {
				PrintError(Name, inputFileName, "Could not open file for reading", Exit, OpenFileError);			
			}		
		}		
		/* Open the output file */		
		if(TextOutput == binaryOutput) {
			if(!(fpOut=fopen(outputFileName, "wb"))) {
				PrintError(Name, outputFileName, "Could not open file for writing", Exit, OpenFileError);			
			}		
		}		
		else {			
			if(!(fpOutGZ=gzopen(outputFileName, "wb"))) {
				PrintError(Name, outputFileName, "Could not open file for writing", Exit, OpenFileError);			
			}		
		}	
		/* Initialize */
		RGMatchesInitialize(&m);
		counter = 0;
		fprintf(stderr, "Input:%s\nOutput:%s\n", inputFileName, outputFileName);
		fprintf(stderr, "Currently on:\n0");
		/* Read in each match */
		while((TextInput == binaryInput && EOF != RGMatchesReadText(fpIn, &m)) ||
				(BinaryInput == binaryInput && EOF != RGMatchesRead(fpInGZ, &m))) {
			if(counter%BMFCONVERT_ROTATE_NUM==0) {
				fprintf(stderr, "\r%lld",
						counter);
			}
			counter++;
			/* Print each match */
			switch(outputType) {
				case 0:
					RGMatchesPrint(fpOutGZ, &m);
					break;
				case 1:
					RGMatchesPrintText(fpOut, &m);
					break;
				case 2:
					RGMatchesPrintFastq(fpOut, &m);
					break;
				default:
					PrintError(Name, NULL, "Could not understand output type", Exit, OutOfRange);		
			}		
			RGMatchesFree(&m);	
		}	
		fprintf(stderr, "\r%lld\n", 
				counter);
		/* Close the input file */
		if(TextInput == binaryInput) {
			fclose(fpIn);
		}
		else {
			gzclose(fpInGZ);
		}
		/* Close the output file */
		if(TextOutput == binaryOutput) {
			fclose(fpOut);
		}
		else {
			gzclose(fpOutGZ);
		}
		free(inputFileName);
	}

	fprintf(stderr, "Terminating successfully!\n");
	return 0;
}
