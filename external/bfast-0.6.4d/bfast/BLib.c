#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <ctype.h>
#include <zlib.h>
#include <limits.h>

#include "BLibDefinitions.h"
#include "RGIndex.h"
#include "BError.h"
#include "BLib.h"

char DNA[5] = "ACGTN";
char COLORS[5] = "01234";

/* TODO */
int GetFastaHeaderLine(FILE *fp,
		char *header)
{
	/*
	   char *FnName="GetFastaHeaderLine";
	   */
	char *ret;

	/* Read in the line */
	ret = fgets(header,  MAX_CONTIG_NAME_LENGTH, fp);

	/* Check the return value */
	if(ret != header) {
		return EOF;
	}
	return 1;
}

/* TODO */
int ParseFastaHeaderLine(char *header)
{
	char *FnName="ParseFastaHeaderLine";
	int i;

	/* Check that the first character is a ">" */
	if(header[0] != '>') {
		PrintError(FnName, "header", "Header of a contig must start with a '>'", Exit, OutOfRange);
	}

	/* Shift over to remove the ">" and trailing EOL */
	for(i=1;i<strlen(header);i++) {
		if(1==IsWhiteSpace(header[i])) break;
		header[i-1] = header[i];
	}
	header[i-1] = '\0';

	return 1;
}

/* TODO */
char ToLower(char a) 
{
	switch(a) {
		case 'A':
			return 'a';
			break;
		case 'C':
			return 'c';
			break;
		case 'G':
			return 'g';
			break;
		case 'T':
			return 't';
			break;
		case 'N':
			return 'n';
			break;
		default:
			return a;
	}
}

/* TODO */
void ToLowerRead(char *r, int readLength) 
{
	int i;
	for(i=0;i<readLength;i++) {
		r[i] = ToLower(r[i]);
	}
}

/* TODO */
char ToUpper(char a)
{
	switch(a) {
		case 'a':
			return 'A';
			break;
		case 'c':
			return 'C';
			break;
		case 'g':
			return 'G';
			break;
		case 't':
			return 'T';
			break;
		case 'n':
			return 'N';
			break;
		default:
			return a;
	}
}

/* TODO */
void ToUpperRead(char *r, int readLength) 
{
	int i;
	for(i=0;i<readLength;i++) {
		r[i] = ToUpper(r[i]);
	}
}

/* TODO */
void ReverseRead(char *s,
		char *r,
		int length)
{       
	int i;
	/* Get reverse */
	for(i=length-1;i>=0;i--) {
		r[i] = s[length-1-i];
	}
	r[length]='\0';
}

void ReverseReadFourBit(int8_t *s,
		int8_t *r,
		int length)
{       
	int i;
	/* Get reverse */
	for(i=length-1;i>=0;i--) {
		r[i] = s[length-1-i];
	}
	r[length]='\0';
}

/* TODO */
void GetReverseComplimentAnyCase(char *s,
		char *r,
		int length)
{       
	int i;
	/* Get reverse compliment sequence */
	for(i=length-1;i>=0;i--) {
		r[i] = GetReverseComplimentAnyCaseBase(s[length-1-i]);
	}
	r[length]='\0';
}

void GetReverseComplimentFourBit(int8_t *s,
		int8_t *r, 
		int length) 
{
	int i;
	/* Get reverse compliment sequence */
	for(i=length-1;i>=0;i--) {
		r[i] = (4 != s[length-i-1]) ? 3 - s[length-i-1] : 4;
	}
	r[length]='\0';
}

char GetReverseComplimentAnyCaseBase(char a) 
{
	char *FnName = "GetReverseComplimentAnyCaseBase";
	switch(a) {
		case 'a':
			return 't';
			break;
		case 'c':
			return 'g';
			break;
		case 'g':
			return 'c';
			break;
		case 't':
			return 'a';
			break;
		case 'n':
			return 'n';
			break;
		case 'A':
			return 'T';
			break;
		case 'C':
			return 'G';
			break;
		case 'G':
			return 'C';
			break;
		case 'T':
			return 'A';
			break;
		case '.':
		case 'N':
			return 'N';
			break;
		case GAP:
			return GAP;
			break;
		default:
			fprintf(stderr, "\n[%c]\t[%d]\n",
					a,
					(int)a);
			PrintError(FnName, NULL, "Could not understand sequence base", Exit, OutOfRange);
			break;
	}
	PrintError(FnName, NULL, "Control should not reach here", Exit, OutOfRange);
	return '0';
}

/* TODO */
int ValidateBasePair(char c) {
	switch(c) {
		case 'a':
		case 'c':
		case 'g':
		case 't':
		case 'A':
		case 'C':
		case 'G':
		case 'T':
		case 'n':
		case 'N':
		case '.':
			return 1;
			break;
		default:
			return 0;
			break;
	}
}

int IsAPowerOfTwo(unsigned int a) {
	int i;

	for(i=0;i<8*sizeof(unsigned int);i++) {
		/*
		   fprintf(stderr, "i:%d\ta:%d\tshifted:%d\tres:%d\n",
		   i,
		   a,
		   a>>i,
		   (a >> i)%2);
		   */
		if( (a >> i) == 2) {
			return 1;
		}
		else if( (a >> i)%2 != 0) {
			return 0;
		}
	}
	return 1;
}

/* TODO */
uint32_t Log2(uint32_t num) 
{
	char *FnName = "Log2";
	int i;

	if(IsAPowerOfTwo(num)==0) {
		PrintError(FnName, "num", "Num is not a power of 2", Exit, OutOfRange);
	}
	/* Not the most efficient but we are not going to use this often */
	for(i=0;num>1;i++,num/=2) {
	}
	return i;
}

char TransformFromIUPAC(char a) 
{
	switch(a) {
		case 'U':
			return 'T';
			break;
		case 'u':
			return 't';
			break;
		case 'R':
		case 'Y':
		case 'M':
		case 'K':
		case 'W':
		case 'S':
		case 'B':
		case 'D':
		case 'H':
		case 'V':
			return 'N';
			break;
		case 'r':
		case 'y':
		case 'm':
		case 'k':
		case 'w':
		case 's':
		case 'b':
		case 'd':
		case 'h':
		case 'v':
			return 'n';
			break;
		default:
			return a;
			break;
	}
}

void CheckRGIndexes(char **mainFileNames,
		int numMainFileNames,
		char **secondaryFileNames,
		int numSecondaryFileNames,
		int32_t *startContig,
		int32_t *startPos,
		int32_t *endContig,
		int32_t *endPos,
		int32_t space)
{
	int i;
	int32_t mainStartContig, mainStartPos, mainEndContig, mainEndPos;
	int32_t secondaryStartContig, secondaryStartPos, secondaryEndContig, secondaryEndPos;
	int32_t mainColorSpace=space;
	int32_t secondaryColorSpace=space;
	int32_t mainContigType=0;
	int32_t secondaryContigType=0;
	mainStartContig = mainStartPos = mainEndContig = mainEndPos = 0;
	secondaryStartContig = secondaryStartPos = secondaryEndContig = secondaryEndPos = 0;

	RGIndex tempIndex;
	gzFile fp;

	/* Read in main indexes */
	for(i=0;i<numMainFileNames;i++) {
		/* Open file */
		if((fp=gzopen(mainFileNames[i], "rb"))==0) {
			PrintError("CheckRGIndexes", mainFileNames[i], "Could not open file for reading", Exit, OpenFileError);
		}

		/* Get the header */
		RGIndexReadHeader(fp, &tempIndex);

		assert(tempIndex.startContig < tempIndex.endContig ||
				(tempIndex.startContig == tempIndex.endContig && tempIndex.startPos <= tempIndex.endPos));

		if(i==0) {
			mainContigType = tempIndex.contigType;
			mainStartContig = tempIndex.startContig;
			mainStartPos = tempIndex.startPos;
			mainEndContig = tempIndex.endContig;
			mainEndPos = tempIndex.endPos;
			mainColorSpace = tempIndex.space;
		}
		else {
			/* Update bounds if necessary */
			assert(mainContigType == tempIndex.contigType);
			if(tempIndex.startContig < mainStartContig ||
					(tempIndex.startContig == mainStartContig && tempIndex.startPos < mainStartPos)) {
				mainStartContig = tempIndex.startContig;
				mainStartPos = tempIndex.startPos;
			}
			if(tempIndex.endContig > mainEndContig ||
					(tempIndex.endContig == mainEndContig && tempIndex.endPos > mainEndPos)) {
				mainEndContig = tempIndex.endContig;
				mainEndPos = tempIndex.endPos;
			}
			assert(mainColorSpace == tempIndex.space);
		}

		/* Free masks */
		free(tempIndex.mask);
		free(tempIndex.packageVersion);
		tempIndex.packageVersion=NULL;
		tempIndex.mask=NULL;

		/* Close file */
		gzclose(fp);
	}
	/* Read in secondary indexes */
	for(i=0;i<numSecondaryFileNames;i++) {
		/* Open file */
		if((fp=gzopen(secondaryFileNames[i], "rb"))==0) {
			PrintError("CheckRGIndexes", "secondaryFileNames[i]", "Could not open file for reading", Exit, OpenFileError);
		}

		/* Get the header */
		RGIndexReadHeader(fp, &tempIndex);

		assert(tempIndex.startContig < tempIndex.endContig ||
				(tempIndex.startContig == tempIndex.endContig && tempIndex.startPos <= tempIndex.endPos));

		if(i==0) {
			secondaryContigType = tempIndex.contigType;
			secondaryStartContig = tempIndex.startContig;
			secondaryStartPos = tempIndex.startPos;
			secondaryEndContig = tempIndex.endContig;
			secondaryEndPos = tempIndex.endPos;
			secondaryColorSpace = tempIndex.space;
		}
		else {
			/* Update bounds if necessary */
			assert(secondaryContigType == tempIndex.contigType);
			if(tempIndex.startContig < secondaryStartContig ||
					(tempIndex.startContig == secondaryStartContig && tempIndex.startPos < secondaryStartPos)) {
				secondaryStartContig = tempIndex.startContig;
				secondaryStartPos = tempIndex.startPos;
			}
			if(tempIndex.endContig > secondaryEndContig ||
					(tempIndex.endContig == secondaryEndContig && tempIndex.endPos > secondaryEndPos)) {
				secondaryEndContig = tempIndex.endContig;
				secondaryEndPos = tempIndex.endPos;
			}
			assert(secondaryColorSpace == tempIndex.space);
		}

		/* Free masks */
		free(tempIndex.mask);
		tempIndex.mask=NULL;

		/* Close file */
		gzclose(fp);
	}

	/* Check the bounds between main and secondary indexes */
	assert(numSecondaryFileNames == 0 ||
			mainContigType == secondaryContigType);
	if(mainStartContig != secondaryStartContig ||
			mainStartPos != secondaryStartPos ||
			mainEndContig != secondaryEndContig ||
			mainEndPos != secondaryEndPos ||
			mainColorSpace != secondaryColorSpace) {
		PrintError("CheckRGIndexes", NULL, "The ranges between main and secondary indexes differ", Exit, OutOfRange);
	}

	(*startContig) = mainStartContig;
	(*startPos) = mainStartPos;
	(*endContig) = mainEndContig;
	(*endPos) = mainEndPos;

	assert(mainColorSpace == space);
	assert(secondaryColorSpace == space);
}

/* TODO */
FILE *OpenTmpFile(char *tmpDir,
		char **tmpFileName)
{
	char *FnName = "OpenTmpFile";
	FILE *fp=NULL;
	int fd;

	/* Allocate memory */
	(*tmpFileName) = malloc(sizeof(char)*MAX_FILENAME_LENGTH);
	if(NULL == (*tmpFileName)) {
		PrintError(FnName, "tmpFileName", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Create the templated */
	/* Copy over tmp directory */
	strcpy((*tmpFileName), tmpDir);
	/* Copy over the tmp name */
	strcat((*tmpFileName), BFAST_TMP_TEMPLATE);

	if(-1 == (fd = mkstemp((*tmpFileName))) ||
			NULL == (fp = fdopen(fd, "wb+"))) {
		/* Check if the fd was open */ 
		if(-1 != fd) {
			/* Remove the file and close */
			unlink((*tmpFileName));
			close(fd);
			PrintError(FnName, (*tmpFileName), "Could not open temporary file", Exit, OpenFileError);
		}
		else {
			PrintError(FnName, (*tmpFileName), "Could not create a tmp file name", Exit, IllegalFileName);
		}
	}

	/* Create a new tmp file name */
	/*
	   if(NULL == mktemp((*tmpFileName))) {
	   PrintError(FnName, (*tmpFileName), "Could not create a tmp file name", Exit, IllegalFileName);
	   }
	   */

	/* Open a new file */
	/*
	   if(!(fp = fopen((*tmpFileName), "wb+"))) {
	   PrintError(FnName, (*tmpFileName), "Could not open temporary file", Exit, OpenFileError);
	   }
	   */

	return fp;
}

/* TODO */
void CloseTmpFile(FILE **fp,
		char **tmpFileName)
{
	char *FnName="CloseTmpFile";

	/* Close the file */
	assert((*fp)!=NULL);
	fclose((*fp));
	(*fp)=NULL;

	/* Remove the file */
	assert((*tmpFileName)!=NULL);
	if(0!=remove((*tmpFileName))) {
		PrintError(FnName, (*tmpFileName), "Could not delete temporary file", Exit, DeleteFileError);
	}

	/* Free file name */
	free((*tmpFileName));
	(*tmpFileName) = NULL;
}

/* TODO */
gzFile OpenTmpGZFile(char *tmpDir,
		char **tmpFileName)
{
	char *FnName = "OpenTmpGZFile";
	int fd;
	gzFile fp = NULL;

	/* Allocate memory */
	(*tmpFileName) = malloc(sizeof(char)*MAX_FILENAME_LENGTH);
	if(NULL == (*tmpFileName)) {
		PrintError(FnName, "tmpFileName", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Create the templated */
	/* Copy over tmp directory */
	strcpy((*tmpFileName), tmpDir);
	/* Copy over the tmp name */
	strcat((*tmpFileName), BFAST_TMP_TEMPLATE);

	if(-1 == (fd = mkstemp((*tmpFileName))) ||
			NULL == (fp = gzdopen(fd, "wb+"))) {
		/* Check if the fd was open */ 
		if(-1 != fd) {
			/* Remove the file and close */
			unlink((*tmpFileName));
			close(fd);
			PrintError(FnName, (*tmpFileName), "Could not open temporary file", Exit, OpenFileError);
		}
		else {
			PrintError(FnName, (*tmpFileName), "Could not create a tmp file name", Exit, IllegalFileName);
		}
	}

	return fp;
}

/* TODO */
void CloseTmpGZFile(gzFile *fp,
		char **tmpFileName,
		int32_t removeFile)
{
	char *FnName="CloseTmpGZFile";

	/* Close the file */
	assert((*fp)!=NULL);
	gzclose((*fp));
	(*fp)=NULL;

	if(1 == removeFile) {
		/* Remove the file */
		assert((*tmpFileName)!=NULL);
		if(0!=remove((*tmpFileName))) {
			PrintError(FnName, (*tmpFileName), "Could not delete temporary file", Exit, DeleteFileError);
		}

		/* Free file name */
		free((*tmpFileName));
		(*tmpFileName) = NULL;
	}
}

void ReopenTmpGZFile(gzFile *fp, char **tmpFileName)
{
	char *FnName="ReopenTmpGZFile";
	CloseTmpGZFile(fp, tmpFileName, 0);
	if(!((*fp) = gzopen((*tmpFileName), "rb"))) {
		PrintError(FnName, (*tmpFileName), "Could not re-open file for reading", Exit, OpenFileError);
	}
}

/* TODO */
void PrintPercentCompleteShort(double percent)
{
	/* back space the " percent complete" */
	fprintf(stderr, "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
	/* back space the "%3.2lf" */
	if(percent < 10.0) {
		/* Only really %1.2lf */
		fprintf(stderr, "\b\b\b\b");
	}
	else if(percent < 100.0) {
		/* Only really %2.2lf */
		fprintf(stderr, "\b\b\b\b\b");
	}
	else {
		fprintf(stderr, "\b\b\b\b\b\b");
	}
	fprintf(stderr, "%3.2lf percent complete", percent);
}

/* TODO */
void PrintPercentCompleteLong(double percent)
{
	/* back space the " percent complete" */
	fprintf(stderr, "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
	/* back space the "%3.3lf" */
	if(percent < 10.0) {
		/* Only really %1.3lf */
		fprintf(stderr, "\b\b\b\b\b");
	}
	else if(percent < 100.0) {
		/* Only really %2.3lf */
		fprintf(stderr, "\b\b\b\b\b\b");
	}
	else {
		fprintf(stderr, "\b\b\b\b\b\b\b");
	}
	fprintf(stderr, "%3.3lf percent complete", percent);
}

int PrintContigPos(FILE *fp,
		int32_t contig,
		int32_t position)
{
	int i;
	int numPrinted = 0;
	int contigLog10 = (int)floor(log10(contig));
	int positionLog10 = (int)floor(log10(position));

	contigLog10 = (contig < 1)?0:((int)floor(log10(contig)));
	positionLog10 = (position < 1)?0:((int)floor(log10(position)));

	assert(contigLog10 <= MAX_CONTIG_LOG_10);
	assert(positionLog10 <= MAX_POSITION_LOG_10);

	numPrinted += fprintf(fp, "\r[");
	for(i=0;i<(MAX_CONTIG_LOG_10-contigLog10);i++) {
		numPrinted += fprintf(fp, "-");
	}
	numPrinted += fprintf(fp, "%d,",
			contig);
	for(i=0;i<(MAX_POSITION_LOG_10-positionLog10);i++) {
		numPrinted += fprintf(fp, "-");
	}
	numPrinted += fprintf(fp, "%d",
			position);
	numPrinted += fprintf(fp, "]");

	return numPrinted;
}

int32_t IsValidRead(RGMatches *m, int space) 
{
	int32_t i;
	for(i=0;i<m->numEnds;i++) {
		if(0 == UpdateRead(m->ends[i].read,
					m->ends[i].readLength,
					space)) {
			return 0;
		}
	}
	return 1;
}

int32_t IsValidMatch(RGMatches *m)
{
	int32_t i;
	for(i=0;i<m->numEnds;i++) {
		if(1 != m->ends[i].maxReached && 0 < m->ends[i].numEntries) {
			return 1;
		}
	}
	return 0;
}

/* TODO */
int UpdateRead(char *read, int readLength, int space) 
{
	char *FnName="UpdateRead";
	int i;

	if(readLength <= 0) {
		/* Ignore zero length reads */
		return 0;
	}

	/* Update the read if possible to lower case, 
	 * if we encounter a base we do not recognize, 
	 * return 0 
	 * */

	/* Check color space adaptor and colors */
	if(ColorSpace == space) {
		switch(read[0]) {
			case 'a':
			case 'c':
			case 'g':
			case 't':
				break;
			case 'A':
			case 'C':
			case 'G':
			case 'T':
				read[0] = tolower(read[0]);
				break;
			default:
				return 0;
				break;
		}
		for(i=1;i<readLength;i++) {
			switch(read[i]) {
				case '0':
				case '1':
				case '2':
				case '3':
				case '4':
					break;
				case '.':
					read[i] = '4';
					break;
				case '\r':
				case '\n':
					PrintError(FnName, "read[i]", "Read was improperly trimmed", Exit, OutOfRange);
					break;
				default:
					return 0;
					break;
			}
		}
	}
	else {

		for(i=0;i<readLength;i++) {
			switch(read[i]) {
				case 'a':
				case 'c':
				case 'g':
				case 't':
				case 'n':
					break;
				case 'A':
				case 'C':
				case 'G':
				case 'T':
				case 'N':
					read[i] = tolower(read[i]);
					break;
				case '.':
					read[i] = 'n';
					break;
				case '\r':
				case '\n':
					PrintError(FnName, "read[i]", "Read was improperly trimmed", Exit, OutOfRange);
					break;
				default:
					return 0;
					break;
			}
		}
	}
	return 1;
}

/* TODO */
/* Debugging function */
int CheckReadAgainstIndex(RGIndex *index,
		char *read,
		int readLength)
{
	char *FnName = "CheckReadAgainstIndex";
	int i;
	for(i=0;i<index->width;i++) {
		switch(CheckReadBase(read[i])) {
			case 0:
				return 0;
				break;
			case 1:
				break;
			default:
				PrintError(FnName, NULL, "Could not understand return value of CheckReadBase", Exit, OutOfRange);
				break;
		}
	}
	return 1;
}

/* TODO */
/* Debugging function */
int CheckReadBase(char base) 
{
	/* Do not include "n"s */
	switch(base) {
		case 'a':
		case 'c':
		case 'g':
		case 't':
			return 1;
			break;
		default:
			return 0;
			break;
	}
}

/* TODO */
/* Two bases */
/* If either of the two bases is an "N" or an "n", then
 * we return the color code 4 */
int ConvertBaseToColorSpace(char A, 
		char B,
		char *C)
{
	/* 
	   char *FnName = "ConvertBaseToColorSpace";
	   */
	int start=0;
	int by=0;
	int result=0;

	switch(A) {
		case 'A':
		case 'a':
			start = 0;
			by = 1;
			break;
		case 'C':
		case 'c':
			start = 1;
			by = -1;
			break;
		case 'G':
		case 'g':
			start = 2;
			by = 1;
			break;
		case 'T':
		case 't':
			start = 3;
			by = -1;
			break;
		case 'N':
		case 'n':
		case '.':
			(*C) = 4;
			return 1;
			break;
		default:
			return 0;
			break;
	}

	switch(B) {
		case 'A':
		case 'a':
			result = start;
			break;
		case 'C':
		case 'c':
			result = start + by;
			break;
		case 'G':
		case 'g':
			result = start + 2*by;
			break;
		case 'T':
		case 't':
			result = start + 3*by;
			break;
		case 'N':
		case 'n':
		case '.':
			(*C) = 4;
			return 1;
			break;
		default:
			return 0;
			break;
	}

	if(result < 0) {
		(*C) =  ALPHABET_SIZE - ( (-1*result)% ALPHABET_SIZE);
	}
	else {
		(*C) = (result%ALPHABET_SIZE);
	}
	return 1;
}

/* TODO */
/* color must be an integer, and a base a character */
int ConvertBaseAndColor(char base, char color, char *B)
{
	/* sneaky */
	char C;

	if(0==ConvertBaseToColorSpace(base, DNA[(int)color], &C)) {
		return 0;
	}
	else {
		(*B) = DNA[(int)C];
	}
	return 1;
}

/* TODO */
/* Include the first letter adaptor */
/* Does not reallocate memory */
int ConvertReadFromColorSpace(char *read,
		int readLength)
{
	char *FnName="ConvertReadFromColorSpace";
	int i;
	char prevBase;

	if(readLength <= 0) return 0;

	/* Convert color character numbers to 8-bit ints */
	for(i=1;i<readLength;i++) { // ignore the adaptor
		switch(read[i]) {
			case '0':
				read[i] = 0;
				break;
			case '1':
				read[i] = 1;
				break;
			case '2':
				read[i] = 2;
				break;
			case '3':
				read[i] = 3;
				break;
			default:
				read[i] = 4;
				break;
		}
	}

	assert(0 < readLength);
	prevBase=read[0];
	for(i=0;i<readLength-1;i++) { 
		if(0 == ConvertBaseAndColor(prevBase, read[i+1], &read[i])) {
			PrintError(FnName, "read", "Could not convert base and color", Exit, OutOfRange);
		}
		/* If the base is an 'N', meaning the color was a '4' (uncalled), then
		 * use the previously non-N base as the previous base (this will be
		 * correct 1/4 of the time). */
		if('N' != read[i]) {
			prevBase = read[i];
		}
	}
	read[readLength-1] = '\0';
	readLength--;

	return readLength;
}

void ConvertSequenceToIntegers(char *seq,
		int8_t *dest,
		int32_t seqLength)
{
	int i;
	for(i=0;i<seqLength;i++) {
		dest[i] = BaseToInt(seq[i]);
	}
}

int32_t BaseToInt(char base) 
{
	switch(base) {
		case 0:
		case '0':
		case 'A':
		case 'a':
			return 0;
		case 1:
		case '1':
		case 'C':
		case 'c':
			return 1;
		case 2:
		case '2':
		case 'G':
		case 'g':
			return 2;
		case 3:
		case '3':
		case 'T':
		case 't':
			return 3;
		default:
			return 4;
	}
}

/* TODO */
/* Must reallocate memory */
/* NT read to color space */
void ConvertReadToColorSpace(char **read,
		int *readLength)
{
	char *FnName="ConvertReadToColorSpace";
	int i;
	char *tempRead=NULL;

	assert(0 < (*readLength));

	tempRead = malloc(sizeof(char)*(1 + (*readLength)));
	if(NULL == tempRead) {
		PrintError(FnName, "tempRead", "Could not allocate memory", Exit, MallocMemory);
	}

	/* Initialize */
	tempRead[0] =  COLOR_SPACE_START_NT;
	if(0==ConvertBaseToColorSpace(tempRead[0], (*read)[0], &tempRead[1])) {
		fprintf(stderr, "tempRead[0]=%c\t(*read)[0]=%c\n", 
				tempRead[0],
				(*read)[0]);
		PrintError(FnName, NULL, "Could not initialize color", Exit, OutOfRange);
	}

	/* Convert to colors represented as integers */
	for(i=1;i<(*readLength);i++) {
		if(0==ConvertBaseToColorSpace((*read)[i-1], (*read)[i], &tempRead[i+1])) {
			fprintf(stderr, "(*read)[i-1]=%c\t(*read)[i]=%c\n(*read)=%s\n",
					(*read)[i-1],
					(*read)[i],
					(*read));
			PrintError(FnName, NULL, "Could not convert base to color space", Exit, OutOfRange);
		}
	}

	(*readLength)++;

	/* Convert integers to characters */
	for(i=1;i<(*readLength);i++) {
		assert(tempRead[i] <= 4);
		tempRead[i] = COLORS[(int)(tempRead[i])];
	}
	tempRead[(*readLength)]='\0';

	/* Reallocate read to make sure */
	(*read) = realloc((*read), sizeof(char)*(1 + (*readLength)));
	if((*read)==NULL) {
		PrintError(FnName, "(*read)", "Could not allocate memory", Exit, MallocMemory);
	}

	strcpy((*read), tempRead);

	free(tempRead);
}

/* TODO */
/* Takes in a NT read, converts to color space,
 * and then converts back to NT space using the
 * start NT */
void NormalizeRead(char **read,
		int *readLength,
		char startNT)
{
	int i;
	char prevOldBase, prevNewBase;
	char tempColor=0;
	char *FnName = "NormalizeRead";

	prevOldBase = startNT;
	prevNewBase = COLOR_SPACE_START_NT;
	for(i=0;i<(*readLength);i++) {
		/* Convert to color space using the old previous NT and current old NT */
		if(0 == ConvertBaseToColorSpace(prevOldBase, (*read)[i], &tempColor)) {
			fprintf(stderr, "prevOldBase=%c\t(*read)[i]=%c\n(*read)=%s\n",
					prevOldBase,
					(*read)[i],
					(*read));
			PrintError(FnName, NULL, "Could not convert base to color space", Exit, OutOfRange);
		}
		prevOldBase = (*read)[i];
		/* Convert to NT space but using the new previous NT and current color */
		if(0 == ConvertBaseAndColor(prevNewBase, tempColor, &(*read)[i])) {
			fprintf(stderr, "prevNewBase=%c\t(*read)[i]=%c\n(*read)=%s\n",
					prevNewBase,
					(*read)[i],
					(*read));
			PrintError(FnName, NULL, "Could not convert base and color", Exit, OutOfRange);
		}
		prevNewBase = (*read)[i];
	}
}

/* TODO */
/* Takes in a Color space read, converts to nt space,
 * and then converts back to Color space.
 * Must include the start NT in the read.
 * */
void NormalizeColorSpaceRead(char *read,
		int readLength,
		char startNT)
{
	char *FnName="NormalizeColorSpaceRead";
	char firstBase, firstColor;
	firstBase = firstColor = 0;


	if(readLength <= 1) {
		return;
	}

	/* Note: we only need to update the start base and the first color */

	/* Convert first color to integer */
	switch(read[1]) {
		case '0':
			firstColor=0;
			break;
		case '1':
			firstColor=1;
			break;
		case '2':
			firstColor=2;
			break;
		case '3':
			firstColor=3;
			break;
		default:
			firstColor=4;
			break;
	}

	/* Get first base */
	if(0 == ConvertBaseAndColor(read[0], firstColor, &firstBase)) {
		PrintError(FnName, NULL, "Could not convert base and color", Exit, OutOfRange);
	}
	/* Now rencode first color */
	read[0] = startNT; 
	if(0 == ConvertBaseToColorSpace(read[0], firstBase, &firstColor)) {
		PrintError(FnName, NULL, "Could not convert base to color space", Exit, OutOfRange);
	}
	read[1] = COLORS[(int)firstColor];
}

/* TODO */
void ConvertColorsToStorage(char *colors, int length)
{
	int i;
	for(i=0;i<length;i++) {
		colors[i] = ConvertColorToStorage(colors[i]);
	}
}

/* TODO */
char ConvertColorToStorage(char c)
{
	switch(c) {
		case 0:
		case '0':
			c = 'A';
			break;
		case 1:
		case '1':
			c = 'C';
			break;
		case 2:
		case '2':
			c = 'G';
			break;
		case 3:
		case '3':
			c = 'T';
			break;
		default:
			c = 'N';
			break;
	}
	return c;
}

/* TODO */
void ConvertColorsFromStorage(char *colors, int length)
{
	int i;
	for(i=0;i<length;i++) {
		colors[i] = ConvertColorFromStorage(colors[i]);
	}
}

/* TODO */
char ConvertIntColorToCharColor(char c)
{
	switch(c) {
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
			break;
		case GAP:
			c = GAP;
			break;
		case 0:
		case 1:
		case 2:
		case 3:
		case 4:
			c = COLORFROMINT(c);
			break;
		default:
			c = '4';
			break;
	}
	return c;
}

/* TODO */
char ConvertColorFromStorage(char c)
{
	return COLORFROMINT(c);
}

/* TODO */
void AdjustBounds(RGBinary *rg,
		int32_t *startContig,
		int32_t *startPos,
		int32_t *endContig,
		int32_t *endPos
		)
{
	char *FnName = "AdjustBounds";

	/* Adjust start and end based on reference genome */
	/* Adjust start */
	if((*startContig) <= 0) {
		if(VERBOSE >= 0) {
			fprintf(stderr, "%s", BREAK_LINE);
			fprintf(stderr, "Warning: startContig was less than zero.\n");
			fprintf(stderr, "Defaulting to contig=%d and position=%d.\n",
					1,
					1);
			fprintf(stderr, "%s", BREAK_LINE);
		}
		(*startContig) = 1;
		(*startPos) = 1;
	}
	else if((*startPos) <= 0) {
		if(VERBOSE >= 0) {
			fprintf(stderr, "%s", BREAK_LINE);
			fprintf(stderr, "Warning: startPos was less than zero.\n");
			fprintf(stderr, "Defaulting to position=%d.\n",
					1);
		}
		(*startPos) = 1;
	}

	/* Adjust end */
	if((*endContig) > rg->numContigs) {
		if(VERBOSE >= 0) {
			fprintf(stderr, "%s", BREAK_LINE);
			fprintf(stderr, "Warning: endContig was greater than the number of contigs in the reference genome.\n");
			fprintf(stderr, "Defaulting to reference genome's end contig=%d and position=%d.\n",
					rg->numContigs,
					rg->contigs[rg->numContigs-1].sequenceLength);
			fprintf(stderr, "%s", BREAK_LINE);
		}
		(*endContig) = rg->numContigs;
		(*endPos) = rg->contigs[rg->numContigs-1].sequenceLength;
	}
	else if((*endContig) <= rg->numContigs && 
			(*endPos) > rg->contigs[(*endContig)-1].sequenceLength) {
		if(VERBOSE >= 0) {
			fprintf(stderr, "%s", BREAK_LINE);
			fprintf(stderr, "Warning: endPos was greater than reference genome's contig %d end position.\n",
					(*endContig));
			fprintf(stderr, "Defaulting to reference genome's contig %d end position: %d.\n",
					(*endContig),
					rg->contigs[(*endContig)-1].sequenceLength);
			fprintf(stderr, "%s", BREAK_LINE);
		}
		(*endPos) = rg->contigs[(*endContig)-1].sequenceLength;
	}

	/* Check that the start and end bounds are ok */
	if((*startContig) > (*endContig)) {
		PrintError(FnName, NULL, "The start contig is greater than the end contig", Exit, OutOfRange);
	}
	else if((*startContig) == (*endContig) &&
			(*startPos) > (*endPos)) {
		PrintError(FnName, NULL, "The start position is greater than the end position on the same contig", Exit, OutOfRange);
	}
}

/* TODO */
int WillGenerateValidKey(RGIndex *index,
		int8_t *read,
		int readLength)
{
	int i;

	for(i=0;i<index->width;i++) {
		if(i >= readLength ||
				(1 == index->mask[i] && 4 == read[i])) {
			return 0;
		}
	}
	return 1;
}

int ValidatePath(char *Name)
{
	/* 
	 *        Checking that strings are good: FileName = [a-zA-Z_0-9][a-zA-Z0-9-.]+/
	 *                      */
	// Check it as a file
	if(0 == ValidateFileName(Name)) return 0;
	// Must end with a trailing forward slash
	if('/' != Name[strlen(Name)-1]) return 0;

	return 1;
}

/* TODO */
int ValidateFileName(char *Name)
{
	/* 
	 *        Checking that strings are good: FileName = [a-zA-Z_0-9.][a-zA-Z0-9-.]+
	 *               FileName can start with only [a-zA-Z_0-9]
	 *                      */

	char *ptr=Name;
	int counter=0;
	/*   fprintf(stderr, "Validating FileName %s with length %d\n", ptr, strlen(Name));  */

	assert(ptr!=0);

	while(*ptr) {
		if((isalnum(*ptr) || (*ptr=='_') || (*ptr=='+') ||
					((*ptr=='.')) ||
					((*ptr=='/')) || /* Make sure that we can navigate through folders */
					((*ptr=='-') && (counter>0)))) {
			ptr++;
			counter++;
		}
		else return 0;
	}
	return 1;
}

/* TODO */
void StringCopyAndReallocate(char **dest, const char *src)
{
	char *FnName="StringCopyAndReallocate";
	/* Reallocate dest */
	(*dest) = realloc((*dest), sizeof(char*)*((int)strlen(src)));
	if(NULL==(*dest)) {
		PrintError(FnName, "(*dest)", "Could not reallocate memory", Exit, ReallocMemory);
	}
	/* Copy */
	strcpy((*dest), src);
}

/* TODO */
int StringTrimWhiteSpace(char *s)
{
	int i;
	int length = strlen(s);

	/* Leading whitespace ignored */

	/* Ending whitespace */
	for(i=length-1;
			1==IsWhiteSpace(s[i]);
			i--) {
		length--;
	}

	s[length]='\0';
	return length;
}

/* TODO */
int IsWhiteSpace(char c) 
{
	switch(c) {
		case ' ':
		case '\t':
		case '\n':
		case '\r':
			return 1;
		default:
			return 0;
	}
	return 0;
}

void ParsePackageVersion(char *packageVersion,
		int *v1,
		int *v2,
		int *v3)
{
	char *FnName="ParsePackageVersion";
	assert(NULL != packageVersion);

	if(3 != sscanf(packageVersion, "%d.%d.%d",
				v1,
				v2,
				v3)) {
		PrintError(FnName, packageVersion, "Could not parse package version", Exit, OutOfRange);
	}
}

/* TODO */
void CheckPackageCompatibility(char *packageVersion, int fileType) 
{
	char *FnName="CheckPackageCompatibility";
	int version[3]={0, 0, 0};

	assert(NULL != packageVersion);

	ParsePackageVersion(packageVersion, &version[0], &version[1], &version[2]);

	switch(fileType) {
		case BFASTReferenceGenomeFile:
			if(version[0] < 0 ||
					version[1] < 2) {
				fprintf(stderr, "%d.%d.%d\n",
						version[0],
						version[1],
						version[2]);
				PrintError(FnName, packageVersion, "File was created using too old of a package", Exit, OutOfRange);
			}
			break;
		case BFASTIndexFile:
			if(version[0] < 0 ||
					version[1] < 2) {
				PrintError(FnName, packageVersion, "File was created using too old of a package", Exit, OutOfRange);
			}
			break;
		default:
			PrintError(FnName, "fileType", "Unrecognized file type given", Exit, OutOfRange);
	}
}

void KnuthMorrisPrattCreateTable(char *read,
		int readLength,
		int *kmp_table)
{
	int cur, next;
	kmp_table[0] = -1;
	kmp_table[1] = 0;

	cur=2;
	next=0;
	while(cur < readLength) {
		if(read[cur-1] == read[next]) {
			kmp_table[cur] = next + 1;
			cur++;
			next++;
		}
		else if(0 < next) {
			next = kmp_table[next];
		}
		else {
			kmp_table[cur] = 0;
			cur++;
		}
	}
}

int32_t KnuthMorrisPratt(char *read,
		int readLength,
		char *reference,
		int referenceLength)
{
	int kmp_table[2056];
	int i, m;

	KnuthMorrisPrattCreateTable(read, readLength, kmp_table);

	i = m = 0;
	while(m + i < referenceLength) {
		if(ToUpper(read[i]) == ToUpper(reference[m + i])) {
			i++;
			if(i == readLength) {
				return m;
			}
		}
		else {
			m += i - kmp_table[i];
			if(0 < i) {
				i = kmp_table[i];
			}
		}
	}
	return -1;
}

/* strstr */
int NaiveSubsequence(char *read, 
		int32_t readLength,
		char *reference,
		int32_t referenceLength)
{
	int i, j, found;
	for(i=0;i<referenceLength-readLength+1;i++) {
		for(j=0, found=1;1==found && j<readLength;j++) {
			if(ToUpper(read[j]) == ToUpper(reference[i+j])) {
				found = 0;
			}
		}
		if(1==found) {
			return i;
		}
	}
	return -1;
}

/* TODO */
/* Need to modify all code to use this function */
int CompareContigPos(int32_t contigOne,
		int32_t positionOne,
		int32_t contigTwo,
		int32_t positionTwo)
{
	if(contigOne < contigTwo ||
			(contigOne == contigTwo && positionOne < positionTwo)) {
		return -1;
	}
	else if(contigOne == contigTwo && positionOne == positionTwo) {
		return 0;
	}
	else {
		return 1;
	}
}

/* TODO */
/* Need to modify all code to use this function */
int WithinRangeContigPos(int32_t contig,
		int32_t position,
		int32_t startContig,
		int32_t startPosition,
		int32_t endContig,
		int32_t endPosition)
{
	if(CompareContigPos(startContig, startPosition, contig, position) <= 0 &&
			CompareContigPos(contig, position, endContig, endPosition) <= 0) {
		return 1;
	}
	else {
		return 0;
	}
}

char *StrStrGetLast(char * str1, 
		const char * str2)
{
	char *ptr=str1;
	char *prev=NULL;

	while(ptr != NULL) {
		prev = ptr;
		ptr = strstr(ptr+1, str2);
	}

	return (prev != str1)?prev:NULL;
}

/* TODO */
void ParseRange(Range *r,
		char *string)
{
	char *FnName="ParseRange";
	if(4 != sscanf(string, "%d-%d:%d-%d\n",
				&r->contigStart,
				&r->contigEnd,
				&r->positionStart,
				&r->positionEnd)) {
		PrintError(FnName, string, "Could not parse string.  Should be in %d-%d:%d-%d format", Exit, OutOfRange);
	}
	if(CompareContigPos(r->contigEnd, r->positionEnd, r->contigStart, r->positionStart) < 0) {
		PrintError(FnName, string, "End range was out of bounds", Exit, OutOfRange);
	}
}

/* TODO */
int32_t CheckRange(Range *r,
		int32_t contig,
		int32_t position)
{
	if(1==WithinRangeContigPos(contig,
				position,
				r->contigStart,
				r->positionStart,
				r->contigEnd,
				r->positionEnd)) {
		return 1;
	}
	return 0;
}

int32_t CheckRangeWithinRange(Range *outside,
		Range *inside){
	if(1==CheckRange(outside,
				inside->contigStart,
				inside->positionStart) &&
			1==CheckRange(outside,
				inside->contigEnd,
				inside->positionEnd)) {
		return 1;
	}
	return 0;
}

void RangeCopy(Range *dest,
		Range *src) 
{
	dest->contigStart = src->contigStart; 
	dest->contigEnd = src->contigEnd; 
	dest->positionStart = src->positionStart; 
	dest->positionEnd = src->positionEnd; 
}

double AddLog10(double a, double b)
{
	if(a < b) {
		return AddLog10(b, a);
	}
	else {
		return a + log10(1.0 + pow(10.0, b-a));
	}
}

int64_t gzwrite64(gzFile file, void *buf, int64_t len) 
{
	int64_t count = 0;
	uint32_t numBytesWritten = 0; 
	uint32_t numBytesToWrite = 0; 

	while(count < len) {
		numBytesToWrite = GETMIN(INT_MAX, (len - count));
		numBytesWritten = gzwrite(file, 
				buf + count, 
				numBytesToWrite);
		if(numBytesWritten != numBytesToWrite) {
			return count;
		}
		count += numBytesWritten;
	}

	return count;
}

int64_t gzread64(gzFile file, void *buf, int64_t len)
{
	int64_t count = 0;
	uint32_t numBytesRead = 0; 
	uint32_t numBytesToRead = 0; 

	while(count < len) {
		numBytesToRead = GETMIN(INT_MAX, (len - count));
		numBytesRead = gzread(file, 
				buf + count, 
				numBytesToRead);
		if(numBytesRead != numBytesToRead) {
			return count;
		}
		count += numBytesRead;
	}

	return count;
}

char *GetBRGFileName(char *fastaFileName, int32_t space)
{
	char *FnName="GetBRGFileName";
	char *rgFileName=NULL;
	assert(NTSpace == space || ColorSpace == space);

	rgFileName=malloc(sizeof(char)*MAX_FILENAME_LENGTH);
	if(NULL == rgFileName) {
		PrintError(FnName, "rgFileName", "Could not allocate memory", Exit, MallocMemory);
	}

	sprintf(rgFileName, "%s.%s.%s",
			fastaFileName,
			SPACENAME(space),
			BFAST_RG_FILE_EXTENSION);

	return rgFileName;
}

char *GetBIFName(char *fastaFileName, 
		int32_t space,
		int32_t depthNumber,
		int32_t indexNumber)
{
	char *FnName="GetBIFName";
	char *bifName=NULL;
	assert(NTSpace == space || ColorSpace == space);

	bifName=malloc(sizeof(char)*MAX_FILENAME_LENGTH);
	if(NULL == bifName) {
		PrintError(FnName, "bifName", "Could not allocate memory", Exit, MallocMemory);
	}

	sprintf(bifName, "%s.%s.%d.%d.%s",
			fastaFileName,
			SPACENAME(space),
			indexNumber,
			depthNumber,
			BFAST_INDEX_FILE_EXTENSION);

	return bifName;
}

/* Do not use this if you want to open the file writing afterwards */
int32_t FileExists(char *fileName)
{
	FILE *fp=NULL;
	if(0==(fp=fopen(fileName, "r"))) {
		return 0;
	}
	fclose(fp);
	return 1;
}

int32_t GetBIFMaximumBin(char *prefix, int32_t indexNumber)
{
	int32_t curExp = 0;
	char bifName[MAX_FILENAME_LENGTH]="\0";

	while(1) { /* ^^ */
		/* Check that it exists */
		sprintf(bifName, "%s.%d.%d.%s",
				prefix,
				indexNumber,
				(int)pow(ALPHABET_SIZE, curExp),
				BFAST_INDEX_FILE_EXTENSION);
		if(0==FileExists(bifName)) {
			break;
		}   
		curExp++;
	}
	if(0 == curExp) {
		return 0;
	}

	return (int)pow(ALPHABET_SIZE, curExp-1);
}

/* This works for positive numbers only.  Inputing negative
 * numbers will have weird behavior. */
int32_t *GetNumbersFromString(char *string, int32_t *length)
{
	char *FnName="GetNumbersFromString";
	char *pch1, *pch2, *tmp=NULL;
	char *saveptr1=NULL, *saveptr2=NULL;

	int32_t* numbers=NULL;
	int32_t prevLength, start, end, state, i, number;

	(*length) = 0;
	if(NULL == string) {
		return NULL;
	}

	pch1 = strtok_r(string, ",", &saveptr1);
	while(pch1 != NULL) { // Split by commas
		number=atoi(pch1); // Could be a negative number
		if(0 <= number && NULL != strchr(pch1, '-')) { // Entry is a range
			tmp=strdup(pch1);
			state = 0;
			pch2 = strtok_r(tmp, "-", &saveptr2);
			end = start = -1;
			while(pch2 != NULL) {
				switch(state) {
					case 0:
						start=atoi(pch2); break;
					case 1:
						end=atoi(pch2); break;
					default:
						PrintError(FnName, pch1, "The range was in improper format", Warn, OutOfRange);
						// Error, return NULL
						free(numbers); free(tmp); return NULL;
				}
				state++;
				pch2=strtok_r(NULL, "-", &saveptr2);
			}
			free(tmp);tmp=NULL;
			if(2 != state) {
				free(numbers); return NULL;
			}
			// Error, return NULL
			if(end < start) {
				PrintError(FnName, pch1, "The start of the range was greater than the end of the range", Warn, OutOfRange);
				free(numbers); return NULL;
			}
			prevLength = (*length);
			(*length) += (end - start + 1);
			numbers=realloc(numbers, sizeof(int32_t)*(*length));
			if(NULL == numbers) {
				PrintError(FnName, "numbers", "Could not reallocate memory", Exit, OutOfRange);
			}
			for(i=prevLength;i<(*length);i++) {
				numbers[i] = start + i - prevLength;
			}
		}
		else { // Entry is a number
			(*length)++;
			numbers=realloc(numbers, sizeof(int32_t)*(*length));
			if(NULL == numbers) {
				PrintError(FnName, "numbers", "Could not reallocate memory", Exit, OutOfRange);
			}
			numbers[(*length)-1] = atoi(pch1);
			if(0 == numbers[(*length)-1] && 0 != strcmp("0", pch1)) {
				PrintError(FnName, pch1, "The input was not a valid number", Warn, OutOfRange);
				free(numbers); return NULL;
			}
		}
		pch1 = strtok_r(NULL, ",", &saveptr1);
	}

	return numbers;
}

#ifndef HAVE_STRTOK_R
char * strtok_r(char *s1, const char *s2, char **lasts)
{
	char *ret;

	if (s1 == NULL)
		s1 = *lasts;
	while(*s1 && strchr(s2, *s1))
		++s1;
	if(*s1 == '\0')
		return NULL;
	ret = s1;
	while(*s1 && !strchr(s2, *s1))
		++s1;
	if(*s1)
		*s1++ = '\0';
	*lasts = s1;
	return ret;
}
#endif

char *ReadInReadGroup(char *readGroupFileName)
{
	char *FnName="readInReadGroup";
	FILE *fp=NULL;
	char readGroup[4096]="\0";

	if(!(fp=fopen(readGroupFileName, "r"))) {
		PrintError(FnName, readGroupFileName, "Could not open file for reading", Exit, OpenFileError);
	}
	if(NULL == fgets(readGroup, 4096, fp)) {
		PrintError(FnName, readGroupFileName, "Could not open read from file", Exit, ReadFileError);
	}
	fclose(fp);

	StringTrimWhiteSpace(readGroup);

	return strdup(readGroup);
}

char *ParseReadGroup(char *readGroup)
{
	char *FnName="ParseReadGroup";
	int32_t m=1, state, foundReadGroupID=0;
	char *pch=NULL, *saveptr=NULL, *ptr=NULL;
	char *readGroupString=NULL;
	char *tmpReadGroup=NULL;
	char ID[32]="\0";

	tmpReadGroup=strdup(readGroup);
	if(NULL == tmpReadGroup) {
		PrintError(FnName, "tmpReadGroup", "Could not allocate memory", Exit, MallocMemory);
	}

	// INitialize
	readGroupString=malloc(sizeof(char)*m);
	if(NULL == readGroupString) {
		PrintError(FnName, "readGroupString", "Could not allocate memory", Exit, MallocMemory);
	}
	readGroupString[0]='\0';

	pch = strtok_r(tmpReadGroup, "\t", &saveptr);
	if(NULL == pch) {
		PrintError(FnName, "readGroup", "Could not parse read group", Exit, OutOfRange);
	}
	strcpy(readGroup, pch);
	pch = strtok_r(NULL, "\t", &saveptr);
	while(NULL != pch) {
		strcat(readGroup, "\t");
		strcat(readGroup, pch);
		state = 0;
		if(NULL != (ptr = strstr(pch, "ID:"))  && ptr == pch) {
			state = foundReadGroupID = 1;
			strcpy(ID, "RG:Z:");
		}
		else if(NULL != (ptr = strstr(pch, "LB:"))  && ptr == pch) {
			state = 2;
			strcpy(ID, "LB:Z:");
		}
		else if(NULL != (ptr = strstr(pch, "PU:"))  && ptr == pch) {
			state = 3;
			strcpy(ID, "PU:Z:");
		}
		if(0 < state) {
			// Found ID
			m += 1 + strlen(ID) + strlen(pch);  // tab, Tag + ':' + Type + ':', Tag value
			readGroupString=realloc(readGroupString, sizeof(char)*m);
			if(NULL == readGroupString) {
				PrintError(FnName, "readGroupString", "Could not allocate memory", Exit, MallocMemory);
			}
			strcat(readGroupString, "\t");
			strcat(readGroupString, ID);
			strcat(readGroupString, pch+3);
		}
		pch = strtok_r(NULL, "\t", &saveptr);
	}

	if(0 == foundReadGroupID) {
		PrintError(FnName, "readGroupString", "Could not parse read group", Exit, OutOfRange);
	}
	
	free(tmpReadGroup);

	return readGroupString;
}
