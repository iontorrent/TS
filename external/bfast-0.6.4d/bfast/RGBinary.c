#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#ifdef HAVE_CONFIG_H
#include <config.h>
#include <zlib.h>
#endif
#include "BError.h"
#include "BLib.h"
#include "BLibDefinitions.h"
#include "RGBinary.h"

/* TODO */
/* Read from fasta file */
void RGBinaryRead(char *fastaFileName, 
		RGBinary *rg,
		int32_t space)
{
	char *FnName="RGBinaryRead";
	FILE *fpRG=NULL;
	int32_t i;
	char c;
	char original;
	int64_t numPosRead=0;
	int32_t byteIndex;
	int32_t numCharsPerByte;

	char curLine[MAX_FASTA_LINE_LENGTH]="\0";
	char *buffer=NULL;
	int64_t bufferLength=0, prevBufferLength=0;
	int64_t bufferIndex=0;
	char header[MAX_CONTIG_NAME_LENGTH]="\0";
	char nextHeader[MAX_CONTIG_NAME_LENGTH]="\0";
	char prevBase = COLOR_SPACE_START_NT; /* For color space */

	/* We assume that we can hold 2 [acgt] (nts) in each byte */
	assert(ALPHABET_SIZE==4);
	numCharsPerByte=ALPHABET_SIZE/2;

	/* Initialize the data structure for holding the rg */
	rg->id=BFAST_ID;
	rg->packageVersionLength = (int)strlen(PACKAGE_VERSION);
	rg->packageVersion = malloc(sizeof(char)*(rg->packageVersionLength+1));
	if(NULL==rg->packageVersion) {
		PrintError(FnName, "rg->packageVersion", "Could not allocate memory", Exit, MallocMemory);
	}
	strcpy(rg->packageVersion, PACKAGE_VERSION);
	rg->packed=RGBinaryPacked;
	rg->contigs=NULL;
	rg->numContigs=0;
	rg->space=space;

	if(VERBOSE>=0) {
		fprintf(stderr, "%s", BREAK_LINE);
		fprintf(stderr, "Reading from %s.\n",
				fastaFileName);
	}

	/* open file */
	if(!(fpRG=fopen(fastaFileName, "rb"))) {
		PrintError(FnName, fastaFileName, "Could not open file for reading", Exit, OpenFileError);
	}

	/*****/
	/* Read in the sequence for each contig. */
	/*****/
	if(VERBOSE >= 0) {
		fprintf(stderr, "Reading in [contig,pos]:\n0");
		PrintContigPos(stderr,
				1,
				1);
	}
	rg->numContigs=0;
	// Get a header
	if(NULL==fgets(nextHeader, MAX_CONTIG_NAME_LENGTH, fpRG)) {
		PrintError(FnName, "nextHeader", "Could not find a fasta header", Exit, OutOfRange);
	}
	int32_t eofReached = 0;
	while(0 == eofReached) {
		strcpy(header, nextHeader);
		ParseFastaHeaderLine(header);

		/* Get sequence */
		curLine[0]='\0';
		buffer=NULL;
		bufferLength=prevBufferLength=0;
		while(NULL!=fgets(curLine, MAX_FASTA_LINE_LENGTH, fpRG)) {

			if('>' == curLine[0]) {
				strcpy(nextHeader, curLine);
				break;
			}
			if('\n' == curLine[strlen(curLine)-1]) {
				curLine[strlen(curLine)-1]='\0';
			}

			prevBufferLength=bufferLength;
			bufferLength += strlen(curLine);
			buffer = realloc(buffer, sizeof(char)*(1+bufferLength));
			if(NULL == buffer) {
				PrintError(FnName, "buffer", "Could not reallocate memory", Exit, ReallocMemory);
			}
			for(i=prevBufferLength;i<bufferLength;i++) {
				buffer[i] = curLine[i-prevBufferLength];
			}
			buffer[bufferLength]='\0';
		}

		if(bufferLength <= 0) {
			break;
		}

		rg->numContigs++;

		/* Reallocate memory to store one more contig. */
		rg->contigs = realloc(rg->contigs, rg->numContigs*sizeof(RGBinaryContig));
		if(NULL == rg->contigs) {
			PrintError(FnName, "rg->contigs", "Could not reallocate memory", Exit, ReallocMemory);
		}
		/* Allocate memory for contig name */
		rg->contigs[rg->numContigs-1].contigNameLength=strlen(header);
		rg->contigs[rg->numContigs-1].contigName = malloc(sizeof(char)*(rg->contigs[rg->numContigs-1].contigNameLength+1));
		if(NULL==rg->contigs[rg->numContigs-1].contigName) {
			PrintError(FnName, "rg->contigs[rg->numContigs-1].contigName", "Could not allocate memory", Exit, MallocMemory);
		}
		/* Copy over contig name */
		strcpy(rg->contigs[rg->numContigs-1].contigName, header); 
		/* Initialize contig */
		rg->contigs[rg->numContigs-1].sequence = NULL;
		rg->contigs[rg->numContigs-1].sequenceLength = 0;
		rg->contigs[rg->numContigs-1].numBytes = 0;

		prevBase = COLOR_SPACE_START_NT; /* For color space */
		for(bufferIndex=0;bufferIndex<bufferLength;bufferIndex++) {
			original=buffer[bufferIndex];
			/* original - will be the original sequence.  Possibilities include:
			 * Non-repeat sequence: a,c,g,t
			 * Repeat sequence: A,C,G,T
			 * Null Character: N,n
			 * */

			/* Transform IUPAC codes */
			original=TransformFromIUPAC(original);

			if(ColorSpace==space) {
				/* Convert to color space */
				/* Convert color space to A,C,G,T */
				if(0 == ConvertBaseToColorSpace(prevBase, original, &c)) {
					fprintf(stderr, "bufferIndex=%lld\n", (long long int)bufferIndex);
					fprintf(stderr, "prevBase=[%c]\toriginal=[%c]\n",
							prevBase,
							original);
					PrintError(FnName, "c", "Could not convert base to color space", Exit, OutOfRange);
				}
				/* Convert to nucleotide equivalent for storage */
				/* Store 0=A, 1=C, 2=G, 3=T, else N */
				c = ConvertColorToStorage(c);
				/* Update if it is a repeat */
				/* For repeat sequence, if both the previous base and 
				 * current base are non-repeat, the color is non-repeat.
				 * Otherwise, it is repeat sequence */
				if(RGBinaryIsBaseRepeat(prevBase) == 1 || RGBinaryIsBaseRepeat(original) == 1) {
					/* Repeat */
					prevBase = original; /* Update the previous base */
					original = ToUpper(c);
				}
				else {
					/* Non-repeat */
					prevBase = original; /* Update the previous base */
					original = ToLower(c);
				}
			}

			if(VERBOSE >= 0) {
				if(0==rg->contigs[rg->numContigs-1].sequenceLength % READ_ROTATE_NUM) {
					PrintContigPos(stderr,
							rg->numContigs,
							rg->contigs[rg->numContigs-1].sequenceLength);
				}
			}

			/* Validate base pair */
			if(ValidateBasePair(original)==0) {
				fprintf(stderr, "Base:[%c]\n", original);
				PrintError(FnName, "original", "Not a valid base pair", Exit, OutOfRange);
			}
			/* Get which byte to insert */
			byteIndex = rg->contigs[rg->numContigs-1].sequenceLength%numCharsPerByte;
			/* Check if we must allocate a new byte */
			if(byteIndex==0) {
				/* Update the number of bytes */
				rg->contigs[rg->numContigs-1].numBytes++;
				/* Reallocate a new byte */
				rg->contigs[rg->numContigs-1].sequence = realloc(rg->contigs[rg->numContigs-1].sequence, sizeof(char)*(rg->contigs[rg->numContigs-1].numBytes));
				if(NULL == rg->contigs[rg->numContigs-1].sequence) {
					PrintError(FnName, "rg->contigs[rg->numContigs-1].sequence", "Could not reallocate memory", Exit, ReallocMemory);
				}
				/* Initialize the byte */
				rg->contigs[rg->numContigs-1].sequence[rg->contigs[rg->numContigs-1].numBytes-1] = 0;
			}
			/* Insert the sequence correctly (as opposed to incorrectly) */
			RGBinaryInsertBase(&rg->contigs[rg->numContigs-1].sequence[rg->contigs[rg->numContigs-1].numBytes-1], 
					byteIndex, 
					original);
			rg->contigs[rg->numContigs-1].sequenceLength++;
			numPosRead++;
			if(rg->contigs[rg->numContigs-1].sequenceLength >= UINT_MAX) {
				PrintError(FnName, "sequenceLength", "Maximum sequence length for a given contig was reached", Exit, OutOfRange);
			}
		}
		/* Update our our output */
		if(VERBOSE >= 0) {
			PrintContigPos(stderr,
					rg->numContigs, 
					rg->contigs[rg->numContigs-1].sequenceLength);
		}

		/* Reallocate to reduce memory (fit exactly) */
		assert(numCharsPerByte==2);
		/* we must add one since there could be an odd number of positions */
		rg->contigs[rg->numContigs-1].sequence = realloc(rg->contigs[rg->numContigs-1].sequence, sizeof(char)*(rg->contigs[rg->numContigs-1].numBytes));
		if(NULL == rg->contigs[rg->numContigs-1].sequence) {
			PrintError(FnName, "rg->contigs[numContigs-1].sequence", "Could not reallocate memory", Exit, ReallocMemory);
		}
		/* End loop for the current contig */
		free(buffer);
		buffer=NULL;
		bufferLength=0;
		if(0 != feof(fpRG)) {
			eofReached=1;
		}
	}
	if(VERBOSE >= 0) {
		PrintContigPos(stderr,
				rg->numContigs, 
				rg->contigs[rg->numContigs-1].sequenceLength);
		fprintf(stderr, "\n");
	}

	/* Close file */
	fclose(fpRG);

	if(VERBOSE>=0) {
		fprintf(stderr, "In total read %d contigs for a total of %lld bases\n",
				rg->numContigs,
				(long long int)numPosRead);
		fprintf(stderr, "%s", BREAK_LINE);
	}
}

/* TODO */
/* Read from file */
void RGBinaryReadBinaryHeader(RGBinary *rg,
		gzFile fpRG)
{
	char *FnName="RGBinaryReadBinaryHeader";
	int32_t i;
	int32_t numCharsPerByte;

	numCharsPerByte=ALPHABET_SIZE/2;

	/* Read RGBinary information */
	if(gzread64(fpRG, &rg->id, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fpRG, &rg->packageVersionLength, sizeof(int32_t))!=sizeof(int32_t)) {
		PrintError(FnName, NULL, "Could not read RGBinary information", Exit, ReadFileError);
	}
	assert(0<rg->packageVersionLength);
	rg->packageVersion = malloc(sizeof(char)*(rg->packageVersionLength+1));
	if(NULL==rg->packageVersion) {
		PrintError(FnName, "rg->packageVersion", "Could not allocate memory", Exit, MallocMemory);
	}
	if(gzread64(fpRG, rg->packageVersion, sizeof(char)*rg->packageVersionLength)!=sizeof(char)*rg->packageVersionLength ||
			gzread64(fpRG, &rg->numContigs, sizeof(int32_t))!=sizeof(int32_t) ||
			gzread64(fpRG, &rg->space, sizeof(int32_t))!=sizeof(int32_t)) {
		PrintError(FnName, NULL, "Could not read RGBinary information", Exit, ReadFileError);
	}
	rg->packageVersion[rg->packageVersionLength]='\0';

	/* Check id */
	if(BFAST_ID != rg->id) {
		PrintError(FnName, "rg->id", "The id did not match", Exit, OutOfRange);
	}
	CheckPackageCompatibility(rg->packageVersion,
			BFASTReferenceGenomeFile);

	assert(rg->numContigs > 0);
	assert(rg->space == NTSpace|| rg->space == ColorSpace);
	rg->packed = RGBinaryPacked;

	/* Allocate memory for the contigs */
	rg->contigs = malloc(sizeof(RGBinaryContig)*rg->numContigs);
	if(NULL==rg->contigs) {
		PrintError(FnName, "rg->contigs", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Read each contig info */
	for(i=0;i<rg->numContigs;i++) {
		/* Read contig name length */
		if(gzread64(fpRG, &rg->contigs[i].contigNameLength, sizeof(int32_t))!=sizeof(int32_t)) {
			PrintError(FnName, NULL, "Could not read contig name length", Exit, ReadFileError);
		}
		assert(rg->contigs[i].contigNameLength > 0);
		/* Allocate memory */
		rg->contigs[i].contigName = malloc(sizeof(char)*(rg->contigs[i].contigNameLength+1));
		if(NULL==rg->contigs[i].contigName) {
			PrintError(FnName, "contigName", "Could not allocate memory", Exit, MallocMemory);
		}
		/* Read RGContig information */
		if(gzread64(fpRG, rg->contigs[i].contigName, sizeof(char)*rg->contigs[i].contigNameLength) != sizeof(char)*rg->contigs[i].contigNameLength ||
				gzread64(fpRG, &rg->contigs[i].sequenceLength, sizeof(int32_t))!=sizeof(int32_t) ||
				gzread64(fpRG, &rg->contigs[i].numBytes, sizeof(uint32_t))!=sizeof(uint32_t)) {
			PrintError(FnName, NULL, "Could not read RGContig information", Exit, ReadFileError);
		}
		rg->contigs[i].sequence = NULL;
	}
}

/* TODO */
/* Read from file */
void RGBinaryReadBinary(RGBinary *rg,
		int32_t space,
		char *fastaFileName)
{
	char *FnName="RGBinaryReadBinary";
	gzFile fpRG;
	int32_t i;
	int32_t numCharsPerByte;
	int64_t numPosRead=0;
	char *brgFileName=NULL;
	/* We assume that we can hold 2 [acgt] (nts) in each byte */
	assert(ALPHABET_SIZE==4);
	numCharsPerByte=ALPHABET_SIZE/2;

	brgFileName=GetBRGFileName(fastaFileName, space);

	if(VERBOSE>=0) {
		fprintf(stderr, "%s", BREAK_LINE);
		fprintf(stderr, "Reading in reference genome from %s.\n", brgFileName);
	}

	/* Open output file */
	if((fpRG=gzopen(brgFileName, "rb"))==0) {
		PrintError(FnName, brgFileName, "Could not open brgFileName for reading", Exit, OpenFileError);
	}

	RGBinaryReadBinaryHeader(rg, fpRG);

	/* Read each contig sequence */
	for(i=0;i<rg->numContigs;i++) {
		/* It should be packed */
		assert(numCharsPerByte == (rg->contigs[i].sequenceLength + (rg->contigs[i].sequenceLength % 2))/rg->contigs[i].numBytes);
		/* Add null terminator */
		rg->contigs[i].contigName[rg->contigs[i].contigNameLength]='\0';
		/* Allocate memory for the sequence */
		rg->contigs[i].sequence = malloc(sizeof(char)*rg->contigs[i].numBytes);
		if(NULL==rg->contigs[i].sequence) {
			PrintError(FnName, "rg->contigs[i].sequence", "Could not allocate memory", Exit, MallocMemory);
		}
		/* Read sequence */
		if(gzread64(fpRG, rg->contigs[i].sequence, sizeof(char)*rg->contigs[i].numBytes)!=sizeof(char)*rg->contigs[i].numBytes) {
			PrintError(FnName, NULL, "Could not read sequence", Exit, ReadFileError);
		}

		numPosRead += rg->contigs[i].sequenceLength;
	}

	/* Close the output file */
	gzclose(fpRG);

	if(VERBOSE>=0) {
		fprintf(stderr, "In total read %d contigs for a total of %lld bases\n",
				rg->numContigs,
				(long long int)numPosRead);
		fprintf(stderr, "%s", BREAK_LINE);
	}

	free(brgFileName);
}

void RGBinaryWriteBinaryHeader(RGBinary *rg,
		gzFile fpRG)
{
	char *FnName="RGBinaryWriteBinaryHeader";
	int i;
	int32_t numCharsPerByte;
	/* We assume that we can hold 2 [acgt] (nts) in each byte */
	assert(ALPHABET_SIZE==4);
	assert(RGBinaryPacked == rg->packed);
	numCharsPerByte=ALPHABET_SIZE/2;

	/* Output RGBinary information */
	if(gzwrite64(fpRG, &rg->id, sizeof(int32_t)) != sizeof(int32_t) ||
			gzwrite64(fpRG, &rg->packageVersionLength, sizeof(int32_t)) != sizeof(int32_t) ||
			gzwrite64(fpRG, rg->packageVersion, rg->packageVersionLength*sizeof(char)) != rg->packageVersionLength*sizeof(char) ||
			gzwrite64(fpRG, &rg->numContigs, sizeof(int32_t)) != sizeof(int32_t) ||
			gzwrite64(fpRG, &rg->space, sizeof(int32_t)) != sizeof(int32_t)) {
		PrintError(FnName, NULL, "Could not output rg header", Exit, WriteFileError);
	}

	/* Output each contig */
	for(i=0;i<rg->numContigs;i++) {
		/* Output RGContig information */
		if(gzwrite64(fpRG, &rg->contigs[i].contigNameLength, sizeof(int32_t)) != sizeof(int32_t) ||
				gzwrite64(fpRG, rg->contigs[i].contigName, sizeof(char)*rg->contigs[i].contigNameLength) != sizeof(char)*rg->contigs[i].contigNameLength ||
				gzwrite64(fpRG, &rg->contigs[i].sequenceLength, sizeof(int32_t)) != sizeof(int32_t) ||
				gzwrite64(fpRG, &rg->contigs[i].numBytes, sizeof(uint32_t)) != sizeof(uint32_t)) {
			PrintError(FnName, NULL, "Could not output rg contig", Exit, WriteFileError);
		}
	}
}

void RGBinaryWriteBinary(RGBinary *rg,
		int32_t space,
		char *fastaFileName) 
{
	char *FnName="RGBinaryWriteBinary";
	gzFile fpRG;
	int i;
	int32_t numCharsPerByte;
	char *brgFileName=NULL;
	/* We assume that we can hold 2 [acgt] (nts) in each byte */
	assert(ALPHABET_SIZE==4);
	assert(RGBinaryPacked == rg->packed);
	numCharsPerByte=ALPHABET_SIZE/2;

	brgFileName=GetBRGFileName(fastaFileName, space);

	if(0 <= VERBOSE) {
		fprintf(stderr, "%s", BREAK_LINE);
		fprintf(stderr, "Outputting to %s\n", brgFileName);
	}

	/* Open output file */
	if((fpRG=gzopen(brgFileName, "wb"))==0) {
		PrintError(FnName, brgFileName, "Could not open brgFileName for writing", Exit, OpenFileError);
	}

	RGBinaryWriteBinaryHeader(rg, fpRG);

	for(i=0;i<rg->numContigs;i++) {
		if(0 <= VERBOSE) {
			fprintf(stderr, "Outputting %s\n", rg->contigs[i].contigName);
		}
		/* Output RGContig sequence */
		if(gzwrite64(fpRG, rg->contigs[i].sequence, sizeof(char)*rg->contigs[i].numBytes) != sizeof(char)*rg->contigs[i].numBytes) {
			PrintError(FnName, NULL, "Could not output rg contig", Exit, WriteFileError);
		}
	}
	gzclose(fpRG);

	free(brgFileName);

	fprintf(stderr, "Output complete.\n");
	fprintf(stderr, "%s", BREAK_LINE);
}

/* TODO */
void RGBinaryDelete(RGBinary *rg)
{
	int32_t i;

	/* Free each contig */
	for(i=0;i<rg->numContigs;i++) {
		free(rg->contigs[i].sequence);
		rg->contigs[i].sequence=NULL;
		free(rg->contigs[i].contigName);
		rg->contigs[i].contigName=NULL;
	}
	/* Free the contigs */
	free(rg->contigs);
	rg->contigs = NULL;

	free(rg->packageVersion);
	rg->packageVersion=NULL;

	/* Initialize structure */
	rg->packageVersionLength=0;
	rg->packed = RGBinaryPacked;
	rg->id = 0;
	rg->numContigs = 0;
	rg->space = NTSpace;
}

/* TODO */
void RGBinaryInsertBase(char *dest,
		int32_t byteIndex,
		char base)
{
	/*********************************
	 * Packed Version:
	 * In four bits we hold two no:
	 *
	 * left two bits:
	 * 		0 - No repat and [acgt]
	 * 		1 - Repeat and [acgt]
	 * 		2 - [nN]
	 * 		3 - undefined
	 *
	 * right-most:
	 * 		0 - aAnN
	 * 		1 - cC
	 * 		2 - gG
	 * 		3 - tt
	 *********************************
	 * */
	int32_t numCharsPerByte;
	/* We assume that we can hold 2 [acgt] (nts) in each byte */
	assert(ALPHABET_SIZE==4);

	numCharsPerByte=ALPHABET_SIZE/2;
	switch(byteIndex%numCharsPerByte) {
		case 0:
			(*dest)=0;
			/* left-most 2-bits will hold the repeat*/
			switch(base) {
				case 'a':
				case 'c':
				case 'g':
				case 't':
					/* zero */
					(*dest) = (*dest) | 0x00;
					break;
				case 'A':
				case 'C':
				case 'G':
				case 'T':
					/* one */
					(*dest) = (*dest) | 0x40;
					break;
				case 'N':
				case 'n':
					/* two */
					(*dest) = (*dest) | 0x80;
					break;
				default:
					PrintError("RGBinaryInsertSequenceLetterIntoByte", NULL, "Could not understand case 0 base", Exit, OutOfRange);
			}
			/* third and fourth bits from the left will hold the sequence */
			switch(base) {
				case 'N':
				case 'n': /* This does not matter if the base is an n */
				case 'A':
				case 'a':
					(*dest) = (*dest) | 0x00;
					break;
				case 'C':
				case 'c':
					(*dest) = (*dest) | 0x10;
					break;
				case 'G':
				case 'g':
					(*dest) = (*dest) | 0x20;
					break;
				case 'T':
				case 't':
					(*dest) = (*dest) | 0x30;
					break;
				default:
					PrintError("RGBinaryInsertSequenceLetterIntoByte", NULL, "Could not understand case 0 base", Exit, OutOfRange);
					break;
			}
			break;
		case 1:
			/* third and fourth bits from the right will hold the repeat*/
			switch(base) {
				case 'a':
				case 'c':
				case 'g':
				case 't':
					/* zero */
					(*dest) = (*dest) | 0x00;
					break;
				case 'A':
				case 'C':
				case 'G':
				case 'T':
					/* one */
					(*dest) = (*dest) | 0x04;
					break;
				case 'N':
				case 'n':
					/* two */
					(*dest) = (*dest) | 0x08;
					break;
				default:
					PrintError("RGBinaryInsertSequenceLetterIntoByte", NULL, "Could not understand case 1 repeat", Exit, OutOfRange);
			}
			/* right most 2-bits will hold the sequence */
			switch(base) {
				case 'N':
				case 'n': /* This does not matter if the base is an n */
				case 'A':
				case 'a':
					(*dest) = (*dest) | 0x00;
					break;
				case 'C':
				case 'c':
					(*dest) = (*dest) | 0x01;
					break;
				case 'G':
				case 'g':
					(*dest) = (*dest) | 0x02;
					break;
				case 'T':
				case 't':
					(*dest) = (*dest) | 0x03;
					break;
				default:
					PrintError("RGBinaryInsertSequenceLetterIntoByte", NULL, "Could not understand case 1 base", Exit, OutOfRange);
			}
			break;
		default:
			PrintError("RGBinaryInsertSequenceLetterIntoByte", NULL, "Could not understand byteIndex", Exit, OutOfRange);
	}
}

/* TODO */
int32_t RGBinaryGetSequence(RGBinary *rg,
		int32_t contig,
		int32_t position,
		char strand,
		char **sequence,
		int32_t sequenceLength)
{
	char *FnName="RGBinaryGetSequence";
	char *reverseCompliment;
	int32_t curPos;

	assert(ALPHABET_SIZE==4);
	if(contig <= 0 || rg->numContigs < contig) {
		return 0;
	}

	/* Allocate memory for the reference */
	assert((*sequence)==NULL);
	(*sequence) = malloc(sizeof(char)*(sequenceLength+1));
	if(NULL==(*sequence)) {
		PrintError(FnName, "sequence", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Copy over bases */
	for(curPos=position;curPos < position + sequenceLength;curPos++) {
		(*sequence)[curPos-position] = RGBinaryGetBase(rg, contig, curPos);
		if(0==(*sequence)[curPos-position]) {
			/* Free memory */
			free((*sequence));
			(*sequence) = NULL;
			return 0;
		}
	}
	(*sequence)[sequenceLength] = '\0';

	/* Get the reverse compliment if necessary */
	if(strand == FORWARD) {
		/* ignore */
	}
	else if(strand == REVERSE) {
		/* Allocate memory for the reverse compliment */
		reverseCompliment = malloc(sizeof(char)*(sequenceLength+1));
		if(NULL == reverseCompliment) {
			PrintError(FnName, "reverseCompliment", "Could not allocate memory", Exit, MallocMemory);
		}
		if(NTSpace == rg->space) {
			/* Get the reverse compliment */
			GetReverseComplimentAnyCase((*sequence), reverseCompliment, sequenceLength);
		}
		else {
			ReverseRead((*sequence), reverseCompliment, sequenceLength);
		}
		free((*sequence)); /* Free memory pointed to by sequence */
		(*sequence) = reverseCompliment; /* Point sequence to reverse compliment's memory */
		reverseCompliment=NULL; /* Destroy the pointer for reverse compliment */
	}
	else {
		fprintf(stderr, "stand=%c\n", strand);
		PrintError(FnName, "strand", "Could not understand strand", Exit, OutOfRange);
	}
	return 1;
}

/* TODO */
void RGBinaryGetReference(RGBinary *rg,
		int32_t contig,
		int32_t position,
		char strand,
		int32_t offsetLength,
		char **reference,
		int32_t readLength,
		int32_t *returnReferenceLength,
		int32_t *returnPosition)
{
	char *FnName="RGBinaryGetReference";
	int32_t startPos, endPos;
	int success;

	assert(ALPHABET_SIZE==4);
	assert(contig > 0 && contig <= rg->numContigs);

	/* Get bounds for the sequence to return */
	startPos = position - offsetLength;
	endPos = position + readLength - 1 + offsetLength;

	/* Check contig bounds */
	if(contig < 1 || contig > rg->numContigs) {
		PrintError(FnName, NULL, "Contig is out of range", Exit, OutOfRange);
	}

	/* Check position bounds */
	if(startPos < 1) {
		startPos = 1;
	}
	if(endPos > rg->contigs[contig-1].sequenceLength) {
		endPos = rg->contigs[contig-1].sequenceLength;
	}

	/* Check that enough bases remain */
	if(endPos - startPos + 1 <= 0) {
		/* Return just one base = N */
		assert((*reference)==NULL);
		(*reference) = malloc(sizeof(char)*(2));
		if(NULL==(*reference)) {
			PrintError(FnName, "reference", "Could not allocate memory", Exit, MallocMemory);
		}
		(*reference)[0] = 'N';
		(*reference)[1] = '\0';

		(*returnReferenceLength) = 1;
		(*returnPosition) = 1;
	}
	else {
		/* Get reference */
		success = RGBinaryGetSequence(rg,
				contig,
				startPos,
				strand,
				reference,
				endPos - startPos + 1);

		if(0 == success) {
			PrintError(FnName, NULL, "Could not get reference", Exit, OutOfRange);
		}

		/* Update start pos and reference length */
		(*returnReferenceLength) = endPos - startPos + 1;
		(*returnPosition) = startPos;
	}
}

/* TODO */
char RGBinaryGetBase(RGBinary *rg,
		int32_t contig,
		int32_t position) 
{
	char *FnName = "RGBinaryGetBase";
	int32_t numCharsPerByte=ALPHABET_SIZE/2;
	char curChar=0;
	uint8_t curByte;

	if(contig < 1 ||
			contig > rg->numContigs ||
			position < 1 ||
			position > rg->contigs[contig-1].sequenceLength) {
		return 0;
	}

	if(RGBinaryUnPacked == rg->packed) {
		/* Simple */
		curChar = rg->contigs[contig-1].sequence[position-1];
	}
	else {
		curByte = RGBinaryGetFourBit(rg, contig, position);
		assert(numCharsPerByte == 2);
		/* Update based on repeat */
		switch((curByte >> 2)) {
			case 0:
				/* not a repeat, convert char to lower */
				curChar="acgt"[(curByte & 0x03)];
				break;
			case 1:
				/* repeat, convert char to upper */
				curChar="ACGT"[(curByte & 0x03)];
				break;
			case 2:
				/* N character */
				curChar='N';
				break;
			default:
				PrintError(FnName, "repeat", "Could not understand repeat", Exit, OutOfRange);
				break;
		}
		// Error check 
		/*
		   switch(curChar) {
		   case 'a':
		   case 'c':
		   case 'g':
		   case 't':
		   case 'A':
		   case 'C':
		   case 'G':
		   case 'T':
		   case 'N':
		   break;
		   default:
		   PrintError(FnName, NULL, "Could not understand base", Exit, OutOfRange);
		   }
		   */
	}
	return curChar;
}

uint8_t RGBinaryGetFourBit(RGBinary *rg,
		int32_t contig,
		int32_t position) 
{
	char *FnName = "RGBinaryGetBaseFourBit";
	int32_t numCharsPerByte=ALPHABET_SIZE/2;
	char curChar;
	uint8_t curByte=0;

	curChar = 0;
	if(contig < 1 ||
			contig > rg->numContigs ||
			position < 1 ||
			position > rg->contigs[contig-1].sequenceLength) {
		return 0;
	}

	if(RGBinaryUnPacked == rg->packed) {
		curByte = 0;
		// Note: no error checking!
		switch(rg->contigs[contig-1].sequence[position-1]) {
			case 'A':
			case 'C':
			case 'G':
			case 'T':
				/* one */
				curByte = curByte | 0x04;
				break;
			case 'N':
			case 'n':
				/* two */
				curByte = curByte | 0x08;
				break;
			default:
				break;
		}
		switch(rg->contigs[contig-1].sequence[position-1]) {
			case 'C':
			case 'c':
				curByte = curByte | 0x01;
				break;
			case 'G':
			case 'g':
				curByte = curByte | 0x03;
				break;
			case 'T':
			case 't':
				curByte = curByte | 0x03;
				break;
			default:
				break;
		}
		return curByte;
	}
	else {
		/* For DNA */
		assert(numCharsPerByte == 2);

		/* The index in the sequence for the given position */
		int32_t byteIndex = (position-1)%numCharsPerByte; /* Which bits in the byte */
		int32_t posIndex = (position - 1 - byteIndex)/numCharsPerByte; /* Get which byte */

		/* Get the current byte */
		assert(posIndex >= 0 && posIndex < rg->contigs[contig-1].numBytes);
		curByte= rg->contigs[contig-1].sequence[posIndex];

		/* Extract base */
		switch(byteIndex) {
			case 0:
				/* left-most 4-bits */
				return (curByte >> 4);
				break;
			case 1:
				/* right-most-four bits */
				return (curByte & 0x0F);
				break;
			default:
				PrintError(FnName, "byteIndex", "Could not understand byteIndex", Exit, OutOfRange);
		}
	}
	return 0;
}

/* TODO */
/*
   int32_t RGBinaryIsRepeat(RGBinary *rg,
   int32_t contig,
   int32_t position)
   {
   char curBase = RGBinaryGetBase(rg,
   contig,
   position);

   return RGBinaryIsBaseRepeat(curBase);
   }
   */

int32_t RGBinaryIsBaseRepeat(char curBase)
{
	switch(curBase) {
		/* Lower case is repat */
		case 'a':
		case 'c':
		case 'g':
		case 't':
			return 1;
			break;
		case 'A':
		case 'G':
		case 'C':
		case 'T':
		default:
			return 0;
			break;
	}
}

/* TODO */
/*
   int32_t RGBinaryIsN(RGBinary *rg,
   int32_t contig, 
   int32_t position) 
   {
   char curBase = RGBinaryGetBase(rg,
   contig,
   position);

   return RGBinaryIsBaseN(curBase);
   }
   */

/* TODO */
/*
   int32_t RGBinaryIsBaseN(char curBase)
   {
   return ( (curBase == 'n' || curBase == 'N')?1:0);
   }
   */

/* TODO */
void RGBinaryPrintInfo(char *brgFileName)
{
	char *FnName="RGBinaryPrintInfo";
	int32_t i;
	RGBinary rg;
	char Space[3][256] = {"NT Space", "Color Space", "Space Last Type"};
	gzFile fpRG;
	FILE *fp=stdout;
	int64_t totalBases=0;

	/* Open output file */
	if((fpRG=gzopen(brgFileName, "rb"))==0) {
		PrintError(FnName, brgFileName, "Could not open brgFileNamefor reading", Exit, OpenFileError);
	}

	/* Read in the reference genome */
	RGBinaryReadBinaryHeader(&rg, fpRG);

	/* Close the output file */
	gzclose(fpRG);

	/* Print details */
	for(i=0;i<rg.numContigs;i++) {
		fprintf(fp, "contig:%6d\tlength:\t%12d\tname:\t%s\n", i+1, rg.contigs[i].sequenceLength, rg.contigs[i].contigName);
		totalBases+=rg.contigs[i].sequenceLength;
	}
	fprintf(fp, "number of contigs:\t%d\n", rg.numContigs);
	fprintf(fp, "total number of bases:\t%lld\n", (long long int)totalBases);
	fprintf(fp, "version:\t\t%s\n", rg.packageVersion);
	fprintf(fp, "space:\t\t\t%d\t\t[%s]\n", rg.space, Space[rg.space]);

	RGBinaryDelete(&rg);
}

/* TODO */
void RGBinaryUnPack(RGBinary *rg) 
{
	char *FnName="RGBinaryUnPack";
	int32_t i, j;
	char *tempSequence=NULL;

	if(RGBinaryUnPacked == rg->packed) {
		return;
	}

	for(i=0;i<rg->numContigs;i++) {
		tempSequence = malloc(sizeof(char)*rg->contigs[i].sequenceLength);
		if(NULL==tempSequence) {
			PrintError(FnName, "tempSequence", "Could not allocate memory", Exit, MallocMemory);
		}
		/* Unpack this */
		for(j=1;j<=rg->contigs[i].sequenceLength;j++) {
			tempSequence[j-1] = RGBinaryGetBase(rg, i+1, j);
		}
		/* Free sequence and copy over */
		free(rg->contigs[i].sequence);
		rg->contigs[i].sequence=tempSequence;
		tempSequence=NULL;
		rg->contigs[i].numBytes = rg->contigs[i].sequenceLength;
	}

	rg->packed = RGBinaryUnPacked;
}
