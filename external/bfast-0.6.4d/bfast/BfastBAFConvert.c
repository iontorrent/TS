#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <zlib.h>
#include <config.h>
#include <unistd.h>
#include "BLibDefinitions.h"
#include "AlignedRead.h"
#include "AlignedReadConvert.h"
#include "BError.h"
#include "BLib.h"

#define Name "bfast bafconvert"
#define BAFCONVERT_ROTATE_NUM 100000

/* Converts a balign file to the specified output format.
 * */
void BfastBAFConvertUsage()
{
	fprintf(stderr, "\nUsage:%s [options] <files>\n", Name);
	fprintf(stderr, "\t-O\t\toutput type:\n"
			"\t\t\t\t0-BAF text to BAF binary\n"
			"\t\t\t\t1-BAF binary to BAF text\n"
			"\t\t\t\t2-BAF binary to SAM (v.%s)\n",
			BFAST_SAM_VERSION
		   );
	fprintf(stderr, "\t-f\t\tSpecifies the file name of the FASTA reference genome\n");
	fprintf(stderr, "\t-o\t\toutput ID to append to the read name (SAM only)\n");
	fprintf(stderr, "\t-r\t\tSpecifies the file that contains the read group" 
			"\n\t\t\t  to add to the SAM header and reads (SAM only)\n");
	fprintf(stderr, "\t-h\t\tprints this help message\n");
	fprintf(stderr, "\nsend bugs to %s\n",
			PACKAGE_BUGREPORT);
}

int BfastBAFConvert(int argc, char *argv[])
{
	FILE *fpIn=NULL, *fpOut=NULL;
	gzFile fpInGZ=NULL, fpOutGZ=NULL;
	long long int counter;
	char inputFileName[MAX_FILENAME_LENGTH]="\0";
	char outputFileName[MAX_FILENAME_LENGTH]="\0";
	char fastaFileName[MAX_FILENAME_LENGTH]="\0";
	char outputID[MAX_FILENAME_LENGTH]="\0";
	char *readGroupFileName=NULL, *readGroup=NULL, *readGroupString=NULL;
	char *last;
	int outputType=1; // BAF2TEXT
	int outputSubType=TextOutput;
	int inputType=BinaryInput;
	int c, argnum;
	AlignedRead a;
	RGBinary rg;
	char fileExtension[256]="\0";

	// Get parameters
	while((c = getopt(argc, argv, "f:o:r:O:h")) >= 0) {
		switch(c) {
			case 'O': outputType = atoi(optarg); break;
			case 'f': strcpy(fastaFileName, optarg); break;
			case 'o': strcpy(outputID, optarg); break;
			case 'r': readGroupFileName=strdup(optarg); break;
			case 'h':
					  BfastBAFConvertUsage(); return 1;
			default: fprintf(stderr, "Unrecognized option: -%c\n", c); return 1;
		}
	}

	if(argc == optind) {
		BfastBAFConvertUsage(); return 1;
	}

	/* Only read in the brg if necessary */
	switch(outputType) {
		case 2:
		case 3:
		case 4:
			if(0 == strlen(fastaFileName)) {
				PrintError(Name, "fastaFileName", "Required command line argument", Exit, InputArguments);
			}
			RGBinaryReadBinary(&rg,
					NTSpace,
					fastaFileName);
			break;
		default:
			break;
	}

	/* Set types and file extension */
	switch(outputType) {
		case 0:
			outputType=BAF;
			inputType=TextInput;
			outputSubType=BinaryOutput;
			strcat(fileExtension, BFAST_ALIGNED_FILE_EXTENSION);
			break;
		case 1:
			outputType=BAF;
			inputType=BinaryInput;
			outputSubType=TextOutput;
			strcat(fileExtension, "txt");
			break;
		case 2:
			outputType=SAM;
			inputType=BinaryInput;
			outputSubType=TextOutput;
			strcat(fileExtension, BFAST_SAM_FILE_EXTENSION);
			if(NULL != readGroupFileName) {
				readGroup=ReadInReadGroup(readGroupFileName);
				readGroupString=ParseReadGroup(readGroup);
			}
			break;
		default:
			PrintError(Name, NULL, "Could not understand output type", Exit, OutOfRange);
	}

	for(argnum=optind;argnum<argc;argnum++) {
		strcpy(inputFileName, argv[argnum]);

		/* Create output file name */
		last = StrStrGetLast(inputFileName,
				BFAST_ALIGNED_FILE_EXTENSION);
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
		if(BinaryInput == inputType) {
			if(!(fpInGZ=gzopen(inputFileName, "rb"))) {
				PrintError(Name, inputFileName, "Could not open file for reading", Exit, OpenFileError);
			}
		}
		else {
			if(!(fpIn=fopen(inputFileName, "rb"))) {
				PrintError(Name, inputFileName, "Could not open file for reading", Exit, OpenFileError);
			}
		}
		/* Open the output file */
		if(BinaryOutput == outputSubType) {
			if(!(fpOutGZ=gzopen(outputFileName, "wb"))) {
				PrintError(Name, outputFileName, "Could not open file for writing", Exit, OpenFileError);
			}
		}
		else {
			if(!(fpOut=fopen(outputFileName, "wb"))) {
				PrintError(Name, outputFileName, "Could not open file for writing", Exit, OpenFileError);
			}
		}

		fprintf(stderr, "Input:%s\nOutput:%s\n", inputFileName, outputFileName);

		/* Print Header */
		AlignedReadConvertPrintHeader(fpOut, &rg, outputType, readGroup);
		/* Initialize */
		AlignedReadInitialize(&a);
		counter = 0;
		fprintf(stderr, "Currently on:\n0");
		/* Read in each match */
		while((TextInput == inputType && EOF != AlignedReadReadText(&a, fpIn)) ||
				(BinaryInput == inputType && EOF != AlignedReadRead(&a, fpInGZ))) {
			if(counter%BAFCONVERT_ROTATE_NUM==0) {
				fprintf(stderr, "\r%lld",
						counter);
			}
			counter++;
			/* Print each match */
			AlignedReadConvertPrintOutputFormat(&a,
					&rg,
					fpOut,
					fpOutGZ,
					outputID,
					readGroupString,
					-1,
					NULL,
					outputType,
					outputSubType);
			AlignedReadFree(&a);
		}
		fprintf(stderr, "\r%lld\n",
				counter);
		/* Close the input file */
		if(TextInput == inputType) {
			fclose(fpIn);
		}
		else {
			gzclose(fpInGZ);
		}
		/* Close the output file */
		if(TextOutput == outputSubType) {
			fclose(fpOut);
		}
		else {
			gzclose(fpOutGZ);
		}
	}
	if(SAM == outputType) {
		RGBinaryDelete(&rg);
	}
	free(readGroupFileName);
	free(readGroup);
	free(readGroupString);

	fprintf(stderr, "Terminating successfully!\n");
	return 0;
}
