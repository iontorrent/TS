#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <config.h>
#include <time.h>
#include <unistd.h>  

#include "../bfast/BError.h"
#include "../bfast/BLib.h"
#include "SimRead.h"
#include "bgeneratereads.h"

#define READS_ROTATE_NUM 10000
#define Name "bgeneratereads"

/* Generate synthetic reads given a number of variants and errors
 * from a reference genome. */


int PrintUsage()
{
	fprintf(stderr, "%s %s\n", "bfast", PACKAGE_VERSION);
	fprintf(stderr, "\nUsage:%s [options]\n", Name);
	fprintf(stderr, "\t-i\tFILE\tinput specification file\n");
	fprintf(stderr, "\t-f\tFILE\tSpecifies the file name of the FASTA reference genome\n");
	fprintf(stderr, "\t-A\tINT\t0: NT space 1: Color space\n");
	fprintf(stderr, "\t-h\t\tprints this help message\n");

	fprintf(stderr, "\nInput specification file:\n");
	fprintf(stderr, "\tThe input specification file is line-orientated.  Each line\n");
	fprintf(stderr, "\tcontains the specification for one set of simulated reads.\n");
	fprintf(stderr, "\tEach set of reads has 9 fields (all specified on one line).\n");
	fprintf(stderr, "\t\t#1 - 0: no indel 1: deletion 2: insertion\n");
	fprintf(stderr, "\t\t#2 - indel length (if #2 is an indel)\n");
	fprintf(stderr, "\t\t#3 - include errors within insertion 0: false 1: true\n");
	fprintf(stderr, "\t\t#4 - # of SNPs\n");
	fprintf(stderr, "\t\t#5 - # of errors\n");
	fprintf(stderr, "\t\t#6 - read length\n");
	fprintf(stderr, "\t\t#7 - paired end 0: false1: true\n");
	fprintf(stderr, "\t\t#8 - paired end length\n");
	fprintf(stderr, "\t\t#9 - number of reads\n");
	fprintf(stderr, "\nsend bugs to %s\n",
			PACKAGE_BUGREPORT);
	return 1;
}

int main(int argc, char *argv[]) 
{
	RGBinary rg;
	char *fastaFileName=NULL;
	int space = NTSpace;
	int indel = 0;
	int indelLength = 0;
	int withinInsertion = 0;
	int numSNPs = 0;
	int numErrors = 0;
	int readLength = 0;
	int pairedEnd = 0;
	int pairedEndLength = 0;
	int numReads = 0;
	int lineNumber = 0;
	char *inputFileName=NULL;
	int c;
	FILE *fpIn=NULL;

	while((c = getopt(argc, argv, "f:i:A:h")) >= 0) {
		switch(c) {
			case 'f': fastaFileName=strdup(optarg);break;
			case 'h': return PrintUsage();
			case 'i': inputFileName = strdup(optarg); break;
			case 'A': space=atoi(optarg); break;
			default: fprintf(stderr, "Unrecognized option: -%c\n", c); return 1;
		}
	}

	if(1 == argc || argc != optind) {
		return PrintUsage();
	}

	if(NULL == fastaFileName) {
		PrintError(Name, "fastaFileName", "Command line argument", Exit, OutOfRange);
	}
	if(NULL == inputFileName) {
		PrintError(Name, "inputFileName", "Command line argument", Exit, OutOfRange);
	}

	/* Get reference genome */
	RGBinaryReadBinary(&rg,
			NTSpace, // Always in NT Space
			fastaFileName);

	if(!(fpIn = fopen(inputFileName, "r"))) {
		PrintError(Name, inputFileName, "Could not open file for reading", Exit, OpenFileError);
	}
	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "Currently on [line, read, numReads]:\n0");
	while(0 == feof(fpIn)) {
		if(fscanf(fpIn, "%d %d %d %d %d %d %d %d %d",
					&indel,
					&indelLength,
					&withinInsertion,
					&numSNPs,
					&numErrors,
					&readLength,
					&pairedEnd,
					&pairedEndLength,
					&numReads) < 0) {
			break;
		}
		lineNumber++;

		/* Generate reads */
		GenerateReadsFP(&rg,
				space,
				indel,
				indelLength,
				withinInsertion,
				numSNPs,
				numErrors,
				readLength,
				(0 == pairedEnd)?1:2,
				pairedEndLength,
				numReads,
				lineNumber,
				stdout);
	}
	fprintf(stderr, "\nTerminating successfully!\n");
	fprintf(stderr, "%s", BREAK_LINE);
	fclose(fpIn);

	/* Free */
	RGBinaryDelete(&rg);
	free(inputFileName);
	free(fastaFileName);
	return 0;
}

/* TODO */
void GenerateReads(RGBinary *rg,
		int space,
		int indel,
		int indelLength,
		int withinInsertion,
		int numSNPs,
		int numErrors,
		int readLength,
		int numEnds,
		int pairedEndLength,
		int numReads)
{
	char *FnName="GenerateReads";
	SimRead r;
	char outFileName[MAX_FILENAME_LENGTH]="\0";
	FILE *fp=NULL;
	int i;
	int64_t rgLength = 0;

	if(NTSpace != rg->space) {
		PrintError(FnName, "rg->space", "The reference genome must be given in nucleotide space", Exit, OutOfRange);
	}

	/* Seed random number */
	srand(time(NULL));

	/* Get the reference genome length */
	for(i=0;i<rg->numContigs;i++) {
		rgLength += rg->contigs[i].sequenceLength;
	}

	/* Create output file name */
	sprintf(outFileName, "reads.%d.%d.%d.%d.%d.%d.%d.%d.%d.%d.fastq",
			space,
			indel,
			indelLength,
			withinInsertion,
			numSNPs,
			numErrors,
			readLength,
			numEnds,
			pairedEndLength,
			numReads);

	/* Open output file */
	if(!(fp=fopen(outFileName, "wb"))) {
		PrintError(FnName, outFileName, "Could not open output file for writing.\n", Exit, OpenFileError);
	}

	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "Outputting to %s.\n",
			outFileName);
	fprintf(stderr, "%s", BREAK_LINE);

	/* Initialize */
	r.numEnds = numEnds;
	r.pairedEndLength = pairedEndLength;
	SimReadInitialize(&r);

	/* Generate the reads */
	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "Out of %d reads, currently on:\n0", numReads);
	for(i=0;i<numReads;i++) {
		if((i+1) % READS_ROTATE_NUM==0) {
			fprintf(stderr, "\r%d",
					(i+1));
		}
		/* Get the read */
		SimReadGetRandom(rg,
				rgLength,
				&r,
				space,
				indel,
				indelLength,
				withinInsertion,
				numSNPs,
				numErrors,
				readLength,
				numEnds,
				pairedEndLength);
		/* Output */
		r.readNum = i+1;
		SimReadPrint(&r,
				fp);

		/* Initialize read */
		SimReadDelete(&r);
	}
	fprintf(stderr, "\r%d\n%s",
			numReads,
			BREAK_LINE);

	/* Close output file */
	fclose(fp);
}

/* TODO */
void GenerateReadsFP(RGBinary *rg,
		int space,
		int indel,
		int indelLength,
		int withinInsertion,
		int numSNPs,
		int numErrors,
		int readLength,
		int numEnds,
		int pairedEndLength,
		int numReads,
		int lineNumber,
		FILE *fp)
{
	char *FnName="GenerateReadsFP";
	SimRead r;
	int i;
	int64_t rgLength = 0;

	if(NTSpace != rg->space) {
		PrintError(FnName, "rg->space", "The reference genome must be given in nucleotide space", Exit, OutOfRange);
	}

	/* Seed random number */
	srand(time(NULL));

	/* Get the reference genome length */
	for(i=0;i<rg->numContigs;i++) {
		rgLength += rg->contigs[i].sequenceLength;
	}

	/* Initialize */
	r.numEnds = numEnds;
	r.pairedEndLength = pairedEndLength;
	SimReadInitialize(&r);

	/* Generate the reads */
	for(i=0;i<numReads;i++) {
		if((i+1) % READS_ROTATE_NUM==0) {
			fprintf(stderr, "\r[%6d,%10d,%10d]",
					lineNumber,
					(i+1),
					numReads);
		}
		/* Get the read */
		SimReadGetRandom(rg,
				rgLength,
				&r,
				space,
				indel,
				indelLength,
				withinInsertion,
				numSNPs,
				numErrors,
				readLength,
				numEnds,
				pairedEndLength);
		/* Output */
		r.readNum = i+1;
		SimReadPrint(&r,
				fp);

		/* Initialize read */
		SimReadDelete(&r);
	}
	fprintf(stderr, "\r[%6d,%10d,%10d]",
			lineNumber,
			numReads,
			numReads);
}
