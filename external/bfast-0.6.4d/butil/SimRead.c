#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>  

#include "../bfast/BError.h"
#include "../bfast/BLib.h"
#include "SimRead.h"

static char *Colors = "01234";

/* Do not change read length, paired end, or paired end length */
void SimReadInitialize(SimRead *r)
{
	int i;
	r->readOne = NULL;
	r->readTwo = NULL;
	r->contig = 0;
	r->pos = 0;
	r->strand = 0;
	r->whichReadVariants = -1;
	r->startIndel = -1;
	for(i=0;i<SEQUENCE_LENGTH-1;i++) {
		r->readOneType[i] = Default;
		r->readTwoType[i] = Default;
	}

}

/* Do not change read length, paired end, or paired end length */
void SimReadDelete(SimRead *r) 
{
	free(r->readOne);
	free(r->readTwo);
	SimReadInitialize(r);
}

char *SimReadGetName(SimRead *r)
{
	char *FnName="../bfast/SimReadGetName";
	char *name=NULL;
	char tmp[32]="\0";
	int i;

	name = malloc(sizeof(char)*(SEQUENCE_LENGTH+1));
	if(NULL == name) {
		PrintError(FnName, "name", "Could not allocate memory", Exit, MallocMemory);
	}
	if(sprintf(name, "readNum=%d_strand=%c_contig=%d_pos=%d_numends=%d_pel=%d_rl=%d_wrv=%d_si=%d_il=%d",
				r->readNum,
				r->strand,
				r->contig,
				r->pos,
				r->numEnds,
				r->pairedEndLength,
				r->readLength,
				r->whichReadVariants,
				r->startIndel,
				r->indelLength) < 0) {
		PrintError(FnName, "name", "Could not create name", Exit, MallocMemory);
	}
	strcat(name, "_r1=");
	for(i=0;i<r->readLength;i++) {
		sprintf(tmp, "%1d", r->readOneType[i]);
		tmp[1]='\0';
		strcat(name, tmp);
	}
	if(r->numEnds==2) {
		strcat(name, "_r2=");
		for(i=0;i<r->readLength;i++) {
			sprintf(tmp, "%1d", r->readTwoType[i]);
			tmp[1]='\0';
			strcat(name, tmp);
		}
	}

	return name;
}

void SimReadPrint(SimRead *r, 
		FILE *fp)
{
	char *name=NULL;
	char quals[SEQUENCE_LENGTH]="\0";
	int32_t i;
	
	/* create quality values */
	for(i=0;i<r->readLength;i++) {
		quals[i] = QUAL2CHAR(SIMREAD_DEFAULT_QUAL);
	}
	quals[r->readLength]='\0';

	/* read name */
	name = SimReadGetName(r);

	/* read one */
	StringTrimWhiteSpace(r->readOne);
	fprintf(fp, "@%s\n%s\n+\n%s\n", name, r->readOne, quals);

	/* read two */
	if(2 == r->numEnds) {
		StringTrimWhiteSpace(r->readTwo);
		fprintf(fp, "@%s\n%s\n+\n%s\n", name, r->readTwo, quals);
	}

	/* Free memory */
	free(name);
	name = NULL;
}

/* TODO */
void SimReadGetRandom(RGBinary *rg,
		int64_t rgLength,
		SimRead *r,
		int space,
		int indel,
		int indelLength,
		int withinInsertion,
		int numSNPs,
		int numErrors,
		int readLength,
		int numEnds,
		int pairedEndLength)
{
	char *FnName="../bfast/SimReadGetRandom";
	int count = 0;
	int i;
	int hasNs=0;
	int readOneSuccess=0;
	int readTwoSuccess=0;
	int64_t ctr=0;
	int success;

	assert(1 <= numEnds && numEnds <= 2);
	r->indelLength = indelLength;
	r->readLength = readLength;
	r->numEnds = numEnds;
	r->pairedEndLength = pairedEndLength;

	for(ctr=0, success=0;0 == success && ctr < SIMREAD_MAX_MODIFY_FAILURES;ctr++) { 
		do {
			/* Avoid infinite loop */
			count++;
			if(SIMREAD_MAX_GETRANDOM_FAILURES < count) {
				PrintError(FnName, "count", "Could not get a random read", Exit, OutOfRange);
			}

			/* Initialize read */
			if(count > 1) {
				SimReadDelete(r);
			}

			/* Get the random contig and position */
			SimReadGetRandomContigPos(rg,
					rgLength - (r->readLength + ((1 == indel) ? indelLength : indel)),
					&r->contig,
					&r->pos,
					&r->strand);

			if(2 == r->numEnds) {
				/* Get the sequence for the first read */
				readOneSuccess = RGBinaryGetSequence(rg,
						r->contig,
						r->pos,
						r->strand,
						&r->readOne,
						r->readLength);
				/* Get the sequence for the second read */
				readTwoSuccess = RGBinaryGetSequence(rg,
						r->contig,
						r->pos + r->pairedEndLength + r->readLength,
						r->strand,
						&r->readTwo,
						r->readLength);
			}
			else {
				/* Get the sequence for the first read */
				readOneSuccess = RGBinaryGetSequence(rg,
						r->contig,
						r->pos,
						r->strand,
						&r->readOne,
						r->readLength);
			}

			/* Make sure there are no Ns */
			hasNs = 0;
			if(1==readOneSuccess) {
				for(i=0;0==hasNs && i<r->readLength;i++) {
					if(RGBinaryIsBaseN(r->readOne[i]) == 1) {
						hasNs = 1;
					}
				}
			}
			if(2 == r->numEnds && 1==readTwoSuccess) {
				for(i=0;0==hasNs && i<r->readLength;i++) {
					if(RGBinaryIsBaseN(r->readTwo[i]) == 1) {
						hasNs = 1;
					}
				}
			}

		} while(
				(readOneSuccess == 0) || /* read one was successfully read */
				(2 == r-> numEnds && readTwoSuccess == 0) || /* read two was successfully read */
				(hasNs == 1) /* Either read end has an "N" */
			   );
		/* Move to upper case */
		for(i=0;i<r->readLength;i++) {
			r->readOne[i] = ToUpper(r->readOne[i]);
			if(2 == r->numEnds) {
				r->readTwo[i] = ToUpper(r->readTwo[i]);
			}
		}

		success = SimReadModify(rg,
				r,
				space,
				indel,
				indelLength,
				withinInsertion,
				numSNPs,
				numErrors);
	}
	if(SIMREAD_MAX_MODIFY_FAILURES <= ctr) {
		PrintError(FnName, "../bfast/SimReadModify", "Could not modify read", Exit, OutOfRange);
	}
}

/* Get the random contig and position */
void SimReadGetRandomContigPos(RGBinary *rg,
		int64_t rgLength,
		int *contig,
		int *pos,
		char *strand)
{
	char *FnName = "GetRandomContigPos";
	int i;
	int64_t curI;
	int64_t low, mid, high;
	int value;
	int count = 0;

	low = 1;
	high = rgLength;

	/* Flip a coin for strand */
	(*strand) = ((rand()%2)==0)?FORWARD:REVERSE;

	/* Use coin flips to find position */
	mid = (low + high)/2;
	while(low < high) {
		mid = (low + high)/2;
		value = rand() % 2;
		if(value == 0) {
			/* lower */
			high = mid;
		}
		else {
			assert(value == 1);
			/* upper */
			low = mid;
		}
		/* To avoid an infinite loop */
		count++;
		if(count > SIMREAD_MAX_GETRANDOM_FAILURES) {
			PrintError(FnName, "count", "Could not get random contig and position", Exit, OutOfRange);
		}
	}

	/* Identify where it occurs */
	curI=0;
	for(i=0;i<rg->numContigs;i++) {
		curI += rg->contigs[i].sequenceLength;
		if(mid <= curI) {
			(*contig) = i+1;
			(*pos) = (curI - mid);
			return;
		}
	}

	PrintError(FnName, "mid", "Mid was out of range", Exit, OutOfRange);
}

/* TODO */
int SimReadModify(RGBinary *rg,
		SimRead *r,
		int space,
		int indel,
		int indelLength,
		int withinInsertion,
		int numSNPs,
		int numErrors)
{
	/* Apply them in this order:
	 * 1. insert an indel based on indel length 
	 * 2. insert a SNP based on include within insertion
	 * 3. convert to color space (if necessary)
	 * 4. insert errors (color errors if in color space, otherwise nt errors )
	 */
	char *FnName="../bfast/SimReadModify";
	int tempReadLength=r->readLength;
	int i;
	int curNumSNPs=0;
	int curNumErrors=0;
	int curInsertionLength=0;

	/* Which read should the variants be contained within */
	r->whichReadVariants= (1 == r->numEnds)?0:(rand()%2); 

	/* 1. Insert an indel based on the indel length */
	switch(indel) {
		case 0:
			/* Do nothing */
			break;
		case 1:
		case 2:
			if(0==SimReadInsertIndel(rg, r, indel, indelLength)) {
				/* Could not add an indel */
				return 0;
			}
			if(REVERSE == r->strand) {
				if(1 == indel) {
					r->pos -= indelLength;
				}
				else {
					r->pos += indelLength;
				}
			}
			break;
		default:
			PrintError(FnName, "indel", "indel out of range", Exit, OutOfRange);
	}

	/* 2. SNPs */
	SimReadInsertMismatches(r,
			numSNPs,
			SNP,
			0);

	if(ColorSpace == space) {
		/* 3. Convert to color space if necessary */
		tempReadLength = r->readLength;
		ConvertReadToColorSpace(&r->readOne,
				&tempReadLength);
		assert(tempReadLength == r->readLength+1);
		if(2==r->numEnds) {
			tempReadLength = r->readLength;
			ConvertReadToColorSpace(&r->readTwo,
					&tempReadLength);
			assert(tempReadLength == r->readLength+1);
		}

		/* 4. Insert errors if necessary */
		SimReadInsertColorErrors(r,
				numErrors,
				withinInsertion);
	}
	else {
		/* 4. Insert NT errors */
		SimReadInsertMismatches(r, 
				numErrors,
				Error,
				withinInsertion);
	}

	/* Check reads */
	if(space == ColorSpace) {
		assert(strlen(r->readOne) == r->readLength + 1);
		if(2==r->numEnds) {
			assert(strlen(r->readTwo) == r->readLength + 1);
		}
	}
	else {
		assert(strlen(r->readOne) == r->readLength);
		if(2==r->numEnds) {
			assert(strlen(r->readTwo) == r->readLength);
		}
	}

	/* Validate read - SNPs and Errors currently */
	curNumSNPs=curNumErrors=curInsertionLength=0;
	for(i=0;i<r->readLength;i++) {
		/* read one */
		switch(r->readOneType[i]) {
			case Default:
				break;
			case Insertion:
				curInsertionLength++;
				break;
			case SNP:
				curNumSNPs++;
				break;
			case Error:
				curNumErrors++;
				break;
			case InsertionAndSNP:
				curInsertionLength++;
				curNumSNPs++;
				break;
			case InsertionAndError:
				curInsertionLength++;
				curNumErrors++;
				break;
			case SNPAndError:
				curNumSNPs++;
				curNumErrors++;
				break;
			case InsertionSNPAndError:
				curInsertionLength++;
				curNumSNPs++;
				curNumErrors++;
				break;
			default:
				PrintError(FnName, "r->readOneType[i]", "Could not understand type", Exit, OutOfRange);
		}
		/* read two */
		if(2 == r->numEnds) {
			switch(r->readTwoType[i]) {
				case Default:
					break;
				case Insertion:
					curInsertionLength++;
					break;
				case SNP:
					curNumSNPs++;
					break;
				case Error:
					curNumErrors++;
					break;
				case InsertionAndSNP:
					curInsertionLength++;
					curNumSNPs++;
					break;
				case InsertionAndError:
					curInsertionLength++;
					curNumErrors++;
					break;
				case SNPAndError:
					curNumSNPs++;
					curNumErrors++;
					break;
				case InsertionSNPAndError:
					curInsertionLength++;
					curNumSNPs++;
					curNumErrors++;
					break;
				default:
					PrintError(FnName, "r->readTwoType[i]", "Could not understand type", Exit, OutOfRange);
			}
		}
	}
	assert(curNumSNPs == numSNPs);
	assert(curNumErrors == numErrors);
	assert(curInsertionLength == indelLength || indel != 2);

	return 1;
}

/* TODO */
int SimReadInsertIndel(RGBinary *rg,
		SimRead *r,
		int indel,
		int indelLength)
{
	char *FnName="InsertIndel";
	int i;
	int start; /* starting position within the read */
	int success=1;

	/* Pick a starting position within the read to insert */
	start = rand() % (r->readLength - indelLength + 1);

	if(indel == 1) {
		/* Deletion */
		/* Remove bases */
		if(r->whichReadVariants == 0) {
			/* Free read */
			free(r->readOne);
			r->readOne = NULL;
			/* Get new read */
			success = RGBinaryGetSequence(rg,
					r->contig,
					r->pos,
					r->strand,
					&r->readOne,
					r->readLength + indelLength);
			if(success == 1) {
				/* Shift over bases */
				for(i=start;i<r->readLength;i++) {
					r->readOne[i] = r->readOne[i+indelLength];
				}
				/* Reallocate memory */
				r->readOne = realloc(r->readOne, sizeof(char)*(r->readLength+1));
				if(NULL==r->readOne) {
					PrintError(FnName, "r->readOne", "Could not reallocate memory", Exit, ReallocMemory);
				}
				r->readOne[r->readLength]='\0';
				assert(strlen(r->readOne) == r->readLength);
				/* Adjust position if reverse strand */
				r->pos = (r->strand==REVERSE)?(r->pos + indelLength):r->pos;
			}
		}
		else {
			/* Insertion */
			/* Free read */
			free(r->readTwo);
			r->readTwo = NULL;
			/* Get new read */
			success = RGBinaryGetSequence(rg,
					r->contig,
					r->pos + r->pairedEndLength + r->readLength,
					r->strand,
					&r->readTwo,
					r->readLength + indelLength);
			if(success == 1) {
				/* Shift over bases */
				for(i=start;i<r->readLength;i++) {
					r->readTwo[i] = r->readTwo[i+indelLength];
				}
				/* Reallocate memory */
				r->readTwo = realloc(r->readTwo, sizeof(char)*(r->readLength+1));
				if(NULL==r->readTwo) {
					PrintError(FnName, "r->read", "Could not reallocate memory", Exit, ReallocMemory);
				}
				r->readTwo[r->readLength]='\0';
				assert(strlen(r->readTwo) == r->readLength);
				/* Adjust position if reverse strand */
				r->pos = (r->strand==REVERSE)?(r->pos + indelLength):r->pos;
			}
		}
	}
	else if(indel == 2) {
		/* Insertion */
		if(r->whichReadVariants == 0) {
			/* shift over all above */
			for(i = r->readLength-1;i >= start + indelLength;i--) {
				r->readOne[i] = r->readOne[i-indelLength];
			}
			/* insert random bases */
			for(i=start;i<start+indelLength;i++) {
				r->readOne[i] = DNA[rand()%4];
				r->readOneType[i] = Insertion;
			}
		}
		else {
			/* shift over all above */
			for(i = r->readLength-1;i >= start + indelLength;i--) {
				r->readTwo[i] = r->readTwo[i-indelLength];
			}
			/* insert random bases */
			for(i=start;i<start+indelLength;i++) {
				r->readTwo[i] = DNA[rand()%4];
				r->readTwoType[i] = Insertion;
			}
		}
	}
	else {
		PrintError(FnName, "indel", "indel out of range", Exit, OutOfRange);
	}

	/* Update the start of the indel */
	r->startIndel = start;


	return success;
}

void SimReadInsertMismatches(SimRead *r,
		int numMismatches,
		int type,
		int withinInsertion)
{
	char *FnName = "InsertMismatches";
	int i;
	switch(type) {
		case SNP:
			if(r->whichReadVariants == 0) {
				SimReadInsertMismatchesHelper(r->readOne,
						r->readLength,
						r->readOneType,
						numMismatches,
						type,
						withinInsertion);
			}
			else {
				SimReadInsertMismatchesHelper(r->readTwo,
						r->readLength,
						r->readTwoType,
						numMismatches,
						type,
						withinInsertion);
			}
			break;
		case Error:
			/* Insert errors one at a time randomly selecting which read */
			for(i=numMismatches;i>0;i--) {
				if(1 == r->numEnds || 
						0==(rand()%2)) {
					SimReadInsertMismatchesHelper(r->readOne,
							r->readLength,
							r->readOneType,
							1,
							type,
							withinInsertion);
				}
				else {
					SimReadInsertMismatchesHelper(r->readTwo,
							r->readLength,
							r->readTwoType,
							1,
							type,
							withinInsertion);
				}
			}
			break;
		default:
			PrintError(FnName, "type", "Could not understand type", Exit, OutOfRange);
	}
}

void SimReadInsertMismatchesHelper(char *read,
		int readLength,
		int *readType,
		int numMismatches,
		int type,
		int withinInsertion)
{
	char *FnName = "InsertMismatches";
	int numMismatchesLeft = numMismatches;
	int index;
	char original;
	int toAdd;

	assert(type == SNP || type == Error);

	while(numMismatchesLeft > 0) {
		/* Pick a base to modify */
		index = rand()%(readLength);
		toAdd = 0;

		switch(readType[index]) {
			case Default:
				readType[index] = type;
				toAdd = 1;
				break;
			case Insertion:
				if(withinInsertion == 1) {
					readType[index] = (type==SNP)?InsertionAndSNP:InsertionAndError;
					toAdd = 1;
				}
				break;
			case SNP:
				if(type == Error) {
					readType[index] = SNPAndError;
					toAdd = 1;
				}
				break;
			case Error:
				/* Nothing, since we assume that SNPs were applied before errors */
				assert(type != SNP);
				break;
			case InsertionAndSNP:
				if(withinInsertion == 1 && type == Error) {
					readType[index] = InsertionSNPAndError;
					toAdd = 1;
				}
				break;
			case SNPAndError:
				/* Nothing */
				break;
			case InsertionAndError:
				/* Nothing, since we assume that SNPs were applied before errors */
				assert(type != SNP);
				break;
			case InsertionSNPAndError:
				/* Nothing */
				break;
			default:
				PrintError(FnName, "readType[index]", "Could not understand type", Exit, OutOfRange);
				break;
		}
		if(1==toAdd) {
			/* Modify base to a new base */
			for(original = read[index];
					original == read[index];
					read[index] = DNA[rand()%4]) {
			}
			assert(read[index] != original);
			numMismatchesLeft--;
		}
	}
}

void SimReadInsertColorErrors(SimRead *r,
		int numErrors,
		int withinInsertion)
{
	/*
	   char *FnName = "InsertColorErrors";
	   */
	int numErrorsLeft = numErrors;
	int which, index;
	char original;
	int toAdd;

	while(numErrorsLeft > 0) {
		/* Pick a read */
		which = (1 == r->numEnds)?0:(rand()%2);
		/* Pick a color to modify */
		index = (rand()%(r->readLength) )+ 1;
		assert(index >= 1 && index < r->readLength + 1);
		toAdd = 0;

		/* Assumes the type can only be Default, Insertion, SNP, or InsertionAndSNP */
		if(which == 0) {
			if(withinInsertion == 1 && r->readOneType[index-1] == Insertion) {
				r->readOneType[index-1] = InsertionAndError;
				toAdd = 1;
			}
			if(withinInsertion == 1 && r->readOneType[index-1] == InsertionAndSNP) {
				r->readOneType[index-1] = InsertionSNPAndError;
				toAdd = 1;
			}
			else if(r->readOneType[index-1] == SNP) {
				r->readOneType[index-1] = SNPAndError;
				toAdd = 1;
			}
			else if(r->readOneType[index-1] == Default) {
				r->readOneType[index-1] = Error;
				toAdd = 1;
			}
			if(1==toAdd) {
				/* Modify color to a new color */
				for(original = r->readOne[index];
						original == r->readOne[index];
						r->readOne[index] = Colors[rand()%4]) {
				}
				numErrorsLeft--;
			}
		}
		else {
			if(withinInsertion == 1 && r->readTwoType[index-1] == Insertion) {
				r->readTwoType[index-1] = InsertionAndError;
				toAdd = 1;
			}
			if(withinInsertion == 1 && r->readTwoType[index-1] == InsertionAndSNP) {
				r->readTwoType[index-1] = InsertionSNPAndError;
				toAdd = 1;
			}
			else if(r->readTwoType[index-1] == SNP) {
				r->readTwoType[index-1] = SNPAndError;
				toAdd = 1;
			}
			else if(r->readTwoType[index-1] == Default) {
				r->readTwoType[index-1] = Error;
				toAdd = 1;
			}
			if(1==toAdd) {
				/* Modify color to a new color */
				for(original = r->readTwo[index];
						original == r->readTwo[index];
						r->readTwo[index] = Colors[rand()%4]) {
				}
				numErrorsLeft--;
			}
		}
	}
}
