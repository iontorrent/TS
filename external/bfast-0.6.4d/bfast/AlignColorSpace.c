#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include "BLib.h"
#include "BLibDefinitions.h"
#include "BError.h"
#include "RGMatches.h"
#include "AlignedEntry.h"
#include "ScoringMatrix.h"
#include "Align.h"
#include "AlignColorSpace.h"

// Remove debugging code
// Fill in end insertion

/* TODO */
int32_t AlignColorSpaceUngapped(char *colors,
		char *mask,
		int readLength,
		char *reference,
		int referenceLength,
		int unconstrained,
		ScoringMatrix *sm,
		AlignedEntry *a,
		int offset,
		int32_t position,
		char strand)
{
	/* read goes on the rows, reference on the columns */
	char *FnName = "AlignColorSpaceUngapped";
	int i, j, k, l;

	int offsetAligned=-1;
	int32_t prevScore[ALPHABET_SIZE+1];
	int prevNT[ALPHABET_SIZE+1][SEQUENCE_LENGTH];
	int32_t maxScore = NEGATIVE_INFINITY;
	int maxNT[SEQUENCE_LENGTH];
	char DNA[ALPHABET_SIZE+1] = "ACGTN";
	char readAligned[SEQUENCE_LENGTH]="\0";
	char referenceAligned[SEQUENCE_LENGTH]="\0";
	char Aligned[SEQUENCE_LENGTH]="\0";
	int32_t alphabetSize = ALPHABET_SIZE+1;

	assert(readLength <= referenceLength);

	alphabetSize = AlignColorSpaceGetAlphabetSize(colors, readLength, reference, referenceLength);

	for(i=offset;i<referenceLength-readLength-offset+1;i++) { /* Starting position */
		/* Initialize */
		for(j=0;j<alphabetSize;j++) {
			if(DNA[j] == COLOR_SPACE_START_NT) { 
				prevScore[j] = 0;
			}
			else {
				prevScore[j] = NEGATIVE_INFINITY;
			}
		}
		for(j=0;j<readLength;j++) { /* Position in the alignment */
			int32_t nextScore[ALPHABET_SIZE+1];
			char nextNT[ALPHABET_SIZE+1];
			for(k=0;k<alphabetSize;k++) { /* To NT */

				/* Get the best score to this NT */
				int32_t bestScore = NEGATIVE_INFINITY;
				int bestNT=-1;
				char bestColor = 'X';

				if(Constrained == unconstrained && '1' == mask[j]) { // If we are to use the constraint and it exists
					char fromNT;
					int32_t fromNTInt;

					if(0 == ConvertBaseAndColor(DNA[k], BaseToInt(colors[j]), &fromNT)) {
						PrintError(FnName, "fromNT", "Could not convert base and color space", Exit, OutOfRange);
					}
					fromNTInt = BaseToInt(fromNT);

					AlignColorSpaceUngappedGetBest(sm, 
							prevScore[fromNTInt],
							colors[j],
							reference[i+j],
							k,
							fromNTInt, // Use the from base (as an integer)
							alphabetSize,
							&bestScore,
							&bestNT,
							&bestColor);
				}
				else { // Ignore constraint, go through all possible transitions
					for(l=0;l<alphabetSize;l++) { /* From NT */
						AlignColorSpaceUngappedGetBest(sm, 
								prevScore[l],
								colors[j],
								reference[i+j],
								k,
								l,
								alphabetSize,
								&bestScore,
								&bestNT,
								&bestColor);
					}
				}
				nextScore[k] = bestScore;
				nextNT[k] = bestNT;
			}

			for(k=0;k<alphabetSize;k++) { /* To NT */
				prevScore[k] = nextScore[k];
				prevNT[k][j] = nextNT[k];
			}
		}
		/* Check if the score is better than the max */
		k=0;
		for(j=0;j<alphabetSize;j++) { /* To NT */
			if(prevScore[k] < prevScore[j]) {
				k=j;
			}
		}
		if(maxScore < prevScore[k]) {
			maxScore = prevScore[k];
			/* TO GET COLORS WE NEED TO BACKTRACK */
			for(j=readLength-1;0<=j;j--) {
				maxNT[j] = k;
				k=prevNT[k][j];
			}
			offsetAligned = i;
		}
	}

	if(NEGATIVE_INFINITY < maxScore) {
		for(i=0;i<readLength;i++) {
			char c[2];
			readAligned[i] = DNA[maxNT[i]];
			referenceAligned[i] = reference[i+offsetAligned];
			c[0] = colors[i];
			ConvertBaseToColorSpace((i==0)?COLOR_SPACE_START_NT:readAligned[i-1],
					readAligned[i],
					&c[1]);
			c[1] = COLORFROMINT(c[1]);
			Aligned[i] = ConvertIntColorToCharColor((c[0] == c[1])?GAP:c[0]); /* Keep original color */
		}
		readAligned[readLength]=referenceAligned[readLength]=Aligned[readLength]='\0';

		/* Copy over */
		AlignedEntryUpdateAlignment(a,
				(FORWARD==strand) ? (position + offsetAligned) : (position + referenceLength - readLength - offsetAligned),
				maxScore,
				readLength,
				readLength,
				readAligned,
				referenceAligned);
		return 1;
	}
	else {
		/* For a case where this actually occurs, think reads at the beginning 
		 * or end of a contig, with adaptor sequence!
		 * */
		return 0;
	}
}

void AlignColorSpaceUngappedGetBest(
		ScoringMatrix *sm,
		int32_t curScore, // previous score (prevScore[l])
		char curColor, // observed color
		char refBase, // reference base (reference[i+j])
		int32_t k, // To NT
		int32_t l, // From NT
		int32_t alphabetSize,
		int32_t *bestScore,
		int32_t *bestNT,
		char *bestColor)
{
	char *FnName="AlignColorSpaceUngappedGetBest";

	char convertedColor='X';
	/* Get color */
	if(0 == ConvertBaseToColorSpace(DNA[l], DNA[k], &convertedColor)) {
		PrintError(FnName, "convertedColor", "Could not convert base to color space", Exit, OutOfRange);
	}
	convertedColor=COLORFROMINT(convertedColor);
	/* Add score for color error, if any */
	curScore += ScoringMatrixGetColorScore(curColor,
			convertedColor,
			sm);
	/* Add score for NT */
	curScore += ScoringMatrixGetNTScore(refBase, DNA[k], sm);

	LOWERBOUNDSCORE(curScore);

	if((*bestScore) < curScore) {
		(*bestScore) = curScore;
		(*bestNT) = l;
		(*bestColor) = convertedColor;
	}
}

void AlignColorSpaceGappedBounded(char *colors,
		int readLength,
		char *reference,
		int referenceLength,
		ScoringMatrix *sm,
		AlignedEntry *a,
		AlignMatrix *matrix,
		int32_t position,
		char strand,
		int32_t maxH,
		int32_t maxV)
{
	//char *FnName = "AlignColorSpaceGappedBounded";
	int i, j;
	int alphabetSize=ALPHABET_SIZE;

	assert(0 < readLength);
	assert(0 < referenceLength);

	alphabetSize = AlignColorSpaceGetAlphabetSize(colors, readLength, reference, referenceLength);

	AlignColorSpaceInitializeAtStart(colors, matrix, sm, readLength, referenceLength, alphabetSize, COLOR_SPACE_START_NT);

	/* Fill in the matrix according to the recursive rules */
	for(i=0;i<readLength;i++) { /* read/rows */
		/* Get the current color */
		for(j=GETMAX(0, i - maxV);
				j <= GETMIN(referenceLength-1, referenceLength - (readLength - maxH) + i);
				j++) { /* reference/columns */
			assert(i-maxV <= j && j <= referenceLength - (readLength - maxH) + i);
			AlignColorSpaceFillInCell(colors, readLength, reference, referenceLength, sm, matrix, i, j, colors[i], maxH, maxV, alphabetSize);
		}
	}

	AlignColorSpaceRecoverAlignmentFromMatrix(a, matrix, colors, readLength, reference, 0, 0, referenceLength, readLength - maxV, position, strand, alphabetSize, 0);
}

void AlignColorSpaceGappedConstrained(char *colors,
		char *mask,
		int readLength,
		char *reference,
		int referenceLength,
		ScoringMatrix *sm,
		AlignedEntry *a,
		AlignMatrix *matrix,
		int32_t referenceOffset,
		int32_t readStartInsertionLength,
		int32_t readEndInsertionLength,
		int32_t position,
		char strand)
{
	char *FnName = "AlignColorSpaceGappedConstrained";
	int i, j, k;
	int alphabetSize=ALPHABET_SIZE;
	int32_t endRowStepOne, endColStepOne, endRowStepTwo, endColStepTwo;
	char *colorsAfterInsertion = colors + readStartInsertionLength;
	char *maskAfterInsertion = mask + readStartInsertionLength;
	int32_t readAfterInsertionLength = readLength - readStartInsertionLength;
	char prevBase, curBase;

	alphabetSize = AlignColorSpaceGetAlphabetSize(colors, readLength, reference, referenceLength);

	/* Get where to transition */
	endRowStepOne = endColStepOne = endRowStepTwo = endColStepTwo = -1;
	i=0;
	while(i<readAfterInsertionLength-readEndInsertionLength) {
		if('1' == maskAfterInsertion[i]) {
			endRowStepOne=i;
			endColStepOne=referenceOffset+i;
			break;
		}
		i++;
	}
	i=readAfterInsertionLength-readEndInsertionLength;
	while(0<=i) {
		if('1' == maskAfterInsertion[i]) {
			endRowStepTwo=i;
			endColStepTwo=referenceOffset+i;
			break;
		}
		i--;
	}

	assert(0 <= endRowStepOne && 0 <= endColStepOne);
	assert(0 <= endRowStepTwo && 0 <= endColStepTwo);

	// Get start NT after insertion
	prevBase = COLOR_SPACE_START_NT;
	for(i=0;i<readStartInsertionLength;i++) {
		if(0 == ConvertBaseAndColor(prevBase, BaseToInt(colors[i-1]), &curBase)) {
			PrintError(FnName, "curReadBase", "Could not convert base and color", Exit, OutOfRange);
		}
		prevBase = curBase;
	}

	/* Step 1 - upper left */
	AlignColorSpaceInitializeAtStart(colorsAfterInsertion, matrix, sm, endRowStepOne, endColStepTwo, alphabetSize, prevBase);
	for(i=0;i<endRowStepOne;i++) { /* read/rows */
		for(j=0;j<endColStepOne;j++) { /* reference/columns */
			AlignColorSpaceFillInCell(colorsAfterInsertion, readAfterInsertionLength, reference, referenceLength, sm, matrix, i, j, colorsAfterInsertion[i], readAfterInsertionLength, readAfterInsertionLength, alphabetSize);
		}
	}

	/* Step 2 - align along the mask */
	// Must consider ins, del, and match on first "color"
	for(i=endRowStepOne,j=endColStepOne;
			i<endRowStepTwo && j<endColStepTwo;
			i++,j++) {
		char curReferenceColor;
		/* Get the current color for the reference */
		if(0 == ConvertBaseToColorSpace((0 == j) ? COLOR_SPACE_START_NT : reference[j-1], reference[j], &curReferenceColor)) {
			PrintError(FnName, "curReferenceColor", "Could not convert base to color space", Exit, OutOfRange);
		}
		curReferenceColor=COLORFROMINT(curReferenceColor);
		/* Check the colors match */
		if('1' == maskAfterInsertion[i] && colorsAfterInsertion[i] != curReferenceColor) {
			PrintError(FnName, NULL, "read and reference did not match", Exit, OutOfRange);
		}
		for(k=0;k<alphabetSize;k++) { /* To NT */
			char fromNT;
			int32_t fromNTInt;
			int32_t curScore = 0;

			if('1' == maskAfterInsertion[i]) { // The mask matched 
				/* Get the from base */
				if(0 == ConvertBaseAndColor(DNA[k], BaseToInt(colorsAfterInsertion[i]), &fromNT)) { 
					PrintError(FnName, "fromNT", "Could not convert base and color to base", Exit, OutOfRange);
				}
				fromNTInt=BaseToInt(fromNT);

				// Add color score and nt score
				curScore = ScoringMatrixGetColorScore(colorsAfterInsertion[i], colorsAfterInsertion[i], sm);
				curScore += ScoringMatrixGetNTScore(reference[j], DNA[k], sm);

				/* Add score for NT */
				matrix->cells[i+1][j+1].s.score[k] = matrix->cells[i][j].s.score[fromNTInt] + curScore;
				matrix->cells[i+1][j+1].s.from[k] = fromNTInt + 1 + (ALPHABET_SIZE + 1); 
				matrix->cells[i+1][j+1].s.length[k] = matrix->cells[i][j].s.length[fromNTInt] + 1;

				assert(i+1 <= matrix->cells[i+1][j+1].s.length[k]);

				// Consider from an indel on the first extension
				if(i == endRowStepOne && j == endColStepOne) {

					/* From Horizontal - Deletion */
					if(matrix->cells[i+1][j+1].s.score[k] < curScore + matrix->cells[i][j].h.score[fromNTInt]) { 
						matrix->cells[i+1][j+1].s.score[k] = curScore + matrix->cells[i][j].h.score[fromNTInt];
						matrix->cells[i+1][j+1].s.from[k] = fromNTInt + 1;
						matrix->cells[i+1][j+1].s.length[k] = matrix->cells[i][j].h.length[fromNTInt] + 1;
					}

					/* From Vertical - Insertion */
					if(matrix->cells[i+1][j+1].s.score[k] < curScore + matrix->cells[i][j].v.score[fromNTInt]) { 
						matrix->cells[i+1][j+1].s.score[k] = curScore + matrix->cells[i][j].v.score[fromNTInt];
						matrix->cells[i+1][j+1].s.from[k] = fromNTInt + 1 + 2*(ALPHABET_SIZE + 1);
						matrix->cells[i+1][j+1].s.length[k] = matrix->cells[i][j].v.length[fromNTInt] + 1;
					}
				}

				matrix->cells[i+1][j+1].s.score[k] = LOWERBOUNDSCORE(matrix->cells[i+1][j+1].s.score[k]);
			}
			else{ // Consider all possible colors as the mask did not match
				int32_t maxScore = NEGATIVE_INFINITY-1;
				int maxFrom = -1;
				char max = GAP;
				int maxLength = -1;

				for(fromNTInt=0;fromNTInt<alphabetSize;fromNTInt++) {
					curScore=NEGATIVE_INFINITY+1;
					char convertedColor='X';

					/* Get color */
					if(0 == ConvertBaseToColorSpace(DNA[fromNTInt], DNA[k], &convertedColor)) {
						PrintError(FnName, "convertedColor", "Could not convert base to color space", Exit, OutOfRange);
					}
					convertedColor=COLORFROMINT(convertedColor);

					// Should not be here if the masks begins with a 1
					assert(i != endRowStepOne && j != endColStepOne);

					/* Get NT and Color scores */
					curScore = ScoringMatrixGetNTScore(reference[j], DNA[k], sm);
					curScore += ScoringMatrixGetColorScore(colorsAfterInsertion[i],
							convertedColor,
							sm);
					LOWERBOUNDSCORE(curScore);

					/* From Diagonal - Match/Mismatch */
					if(maxScore < matrix->cells[i][j].s.score[fromNTInt] + curScore) {
						maxScore = matrix->cells[i][j].s.score[fromNTInt] + curScore;
						maxFrom = fromNTInt + 1 + (ALPHABET_SIZE + 1); /* see the enum */ 
						max = (colorsAfterInsertion[i]  == convertedColor)?GAP:colorsAfterInsertion[i]; /* Keep original color */
						maxLength = matrix->cells[i][j].s.length[fromNTInt] + 1;
					}
				}
				/* Update */
				matrix->cells[i+1][j+1].s.score[k] = maxScore;
				matrix->cells[i+1][j+1].s.from[k] = maxFrom;
				//matrix->cells[i+1][j+1].s.[k] = maxColorError;
				matrix->cells[i+1][j+1].s.length[k] = maxLength;
			}
		}
	}
	for(k=0;k<alphabetSize;k++) {
		assert(1 + ALPHABET_SIZE < matrix->cells[endRowStepTwo][endColStepTwo].s.from[k] &&
				matrix->cells[endRowStepTwo][endColStepTwo].s.from[k] <= 2*(ALPHABET_SIZE + 1));
		assert(endRowStepTwo <= matrix->cells[endRowStepTwo][endColStepTwo].s.length[k]);
	}

	/* Step 3 - lower right */
	AlignColorSpaceInitializeToExtend(colorsAfterInsertion, matrix, sm, readAfterInsertionLength, referenceLength, endRowStepTwo, endColStepTwo, alphabetSize);
	// Note: we ignore any cells on row==endRowStepTwo or col==endRowStepTwo
	// since we assumed they were filled in by the previous re-initialization
	for(i=endRowStepTwo;i<readAfterInsertionLength-readEndInsertionLength;i++) { /* read/rows */
		/* Get the current color for the read */
		for(j=endColStepTwo;j<referenceLength;j++) { /* reference/columns */
			AlignColorSpaceFillInCell(colorsAfterInsertion, readAfterInsertionLength, reference, referenceLength, sm, matrix, i, j, colorsAfterInsertion[i], readAfterInsertionLength, readAfterInsertionLength, alphabetSize);
		}
	}

	/* Step 4 - recover alignment */
	AlignColorSpaceRecoverAlignmentFromMatrix(a, matrix, colors, readLength, reference, referenceLength, 
			readStartInsertionLength,
			readEndInsertionLength,
			endColStepTwo, position, strand, alphabetSize, 0);
}

void AlignColorSpaceRecoverAlignmentFromMatrix(AlignedEntry *a,
		AlignMatrix *matrix,
		char *colors,
		int readLength,
		char *reference,
		int referenceLength,
		int32_t readStartInsertionLength,
		int32_t readEndInsertionLength,
		int toExclude,
		int32_t position,
		char strand,
		int alphabetSize,
		int debug)
{
	char *FnName="AlignColorSpaceRecoverAlignmentFromMatrix";
	int curRow, curCol, startRow, startCol, startCell;
	char curReadBase;
	int nextRow, nextCol, nextCell, nextFrom;
	char prevBase, nextReadBase;
	int curFrom=-1;
	double maxScore;
	int i, j;
	int offset;
	char readAligned[SEQUENCE_LENGTH]="\0";
	char referenceAligned[SEQUENCE_LENGTH]="\0";
	int32_t referenceLengthAligned=0, length=0;

	curReadBase = nextReadBase = 'X';
	nextRow = nextCol = nextCell = -1;

	// Fill in the initial insertion
	prevBase = COLOR_SPACE_START_NT;
	for(i=0;i<readStartInsertionLength;i++) {
		if(0 == ConvertBaseAndColor(prevBase, BaseToInt(colors[i]), &curReadBase)) {
			PrintError(FnName, "curReadBase", "Could not convert base and color", Exit, OutOfRange);
		}
		readAligned[length] = curReadBase;
		referenceAligned[length] = GAP;
		prevBase = curReadBase;
		length++;
	}

	/* Get the best alignment.  We can find the best score in the last row and then
	 * trace back.  We choose the best score from the last row since we want to 
	 * align the read completely and only locally to the reference. */
	startRow=-1;
	startCol=-1;
	startCell=-1;
	maxScore = NEGATIVE_INFINITY-1;
	for(i=toExclude;i<referenceLength+1;i++) { 
		for(j=0;j<alphabetSize;j++) {
			/* Don't end with a Deletion in the read */

			/* End with a Match/Mismatch */
			if(maxScore < matrix->cells[readLength-readEndInsertionLength-readStartInsertionLength][i].s.score[j]) {
				maxScore = matrix->cells[readLength-readEndInsertionLength-readStartInsertionLength][i].s.score[j];
				startRow = readLength-readEndInsertionLength-readStartInsertionLength;
				startCol = i;
				startCell = j + 1 + (ALPHABET_SIZE + 1);
			}

			/* End with an Insertion */
			if(maxScore < matrix->cells[readLength-readEndInsertionLength-readStartInsertionLength][i].v.score[j]) {
				maxScore = matrix->cells[readLength-readEndInsertionLength-readStartInsertionLength][i].v.score[j];
				startRow = readLength-readEndInsertionLength-readStartInsertionLength;
				startCol = i;
				startCell = j + 1 + 2*(ALPHABET_SIZE + 1);
			}
		}
	}
	assert(startRow >= 0 && startCol >= 0 && startCell >= 0);

	/* Initialize variables for the loop */
	curRow=startRow;
	curCol=startCol;
	curFrom=startCell;

	assert(0 < curFrom);
	assert(curFrom <= 3*(ALPHABET_SIZE + 1));

	referenceLengthAligned=0;
	/* Init */
	if(curFrom <= (ALPHABET_SIZE + 1)) {
		PrintError(FnName, "curFrom", "Cannot end with a deletion", Exit, OutOfRange);
		i = matrix->cells[curRow][curCol].h.length[(curFrom - 1) % (ALPHABET_SIZE + 1)] - 1;
		length += matrix->cells[curRow][curCol].h.length[(curFrom - 1) % (ALPHABET_SIZE + 1)];
	}
	else if(2*(ALPHABET_SIZE + 1) < curFrom) {
		length += matrix->cells[curRow][curCol].v.length[(curFrom - 1) % (ALPHABET_SIZE + 1)];
		i = matrix->cells[curRow][curCol].v.length[(curFrom - 1) % (ALPHABET_SIZE + 1)] - 1;
	}
	else {
		length += matrix->cells[curRow][curCol].s.length[(curFrom - 1) % (ALPHABET_SIZE + 1)];
		i = matrix->cells[curRow][curCol].s.length[(curFrom - 1) % (ALPHABET_SIZE + 1)] - 1;
	}

	/* Now trace back the alignment using the "from" member in the matrix */
	while(0 <= i) {
		assert(0 <= curRow && 0 <= curCol);
		/* Where did the current cell come from */
		/* Get if there was a color error */
		if(curFrom <= (ALPHABET_SIZE + 1)) {
			nextFrom = matrix->cells[curRow][curCol].h.from[(curFrom - 1) % (ALPHABET_SIZE + 1)];
		}
		else if(2*(ALPHABET_SIZE + 1) < curFrom) {
			nextFrom = matrix->cells[curRow][curCol].v.from[(curFrom - 1) % (ALPHABET_SIZE + 1)];
		}
		else {
			nextFrom = matrix->cells[curRow][curCol].s.from[(curFrom - 1) % (ALPHABET_SIZE + 1)];
		}

		switch(curFrom) {
			case MatchA:
			case InsertionA:
				readAligned[readStartInsertionLength+i] = 'A';
				break;
			case MatchC:
			case InsertionC:
				readAligned[readStartInsertionLength+i] = 'C';
				break;
			case MatchG:
			case InsertionG:
				readAligned[readStartInsertionLength+i] = 'G';
				break;
			case MatchT:
			case InsertionT:
				readAligned[readStartInsertionLength+i] = 'T';
				break;
			case MatchN:
			case InsertionN:
				readAligned[readStartInsertionLength+i] = 'N';
				break;
			case DeletionA:
			case DeletionC:
			case DeletionG:
			case DeletionT:
			case DeletionN:
				readAligned[readStartInsertionLength+i] = GAP;
				break;
			default:
				PrintError(FnName, "curFrom", "Could not understand curFrom", Exit, OutOfRange);
		}

		switch(curFrom) {
			case InsertionA:
			case InsertionC:
			case InsertionG:
			case InsertionT:
			case InsertionN:
				referenceAligned[readStartInsertionLength+i] = GAP;
				break;
			default:
				referenceAligned[readStartInsertionLength+i] = reference[curCol-1];
				referenceLengthAligned++;
				break;
		}

		assert(readAligned[readStartInsertionLength+i] != GAP || readAligned[readStartInsertionLength+i] != referenceAligned[i+readStartInsertionLength]);

		/* Update next row/col */
		if(curFrom <= (ALPHABET_SIZE + 1)) {
			nextRow = curRow;
			nextCol = curCol-1;
		}
		else if(2*(ALPHABET_SIZE +1) < curFrom) {
			nextRow = curRow-1;
			nextCol = curCol;
		}
		else {
			nextRow = curRow-1;
			nextCol = curCol-1;
		}

		/* Update for next loop iteration */
		curFrom = nextFrom;
		curRow = nextRow;
		curCol = nextCol;
		i--;
	} /* End loop */
	assert(-1==i);
	
	// Fill in the end insertion
	prevBase = readAligned[length-1];
	for(i=0;i<readEndInsertionLength;i++) {
		if(0 == ConvertBaseAndColor(prevBase, BaseToInt(colors[readLength - readEndInsertionLength + i]), &curReadBase)) {
			PrintError(FnName, "curReadBase", "Could not convert base and color", Exit, OutOfRange);
		}
		readAligned[length] = curReadBase;
		referenceAligned[length] = GAP;
		prevBase = curReadBase;
		length++;
	}

	offset = curCol;
	readAligned[length]='\0';
	referenceAligned[length]='\0';

	/* Copy over */
	AlignedEntryUpdateAlignment(a,
			(FORWARD==strand) ? (position + offset) : (position + referenceLength - referenceLengthAligned - offset),
			maxScore,
			referenceLengthAligned,
			length,
			readAligned,
			referenceAligned);

	assert(readLength <= length);
}

void AlignColorSpaceInitializeAtStart(char *colors,
		AlignMatrix *matrix,
		ScoringMatrix *sm, 
		int32_t endRow, 
		int32_t endCol,
		int32_t alphabetSize,
		char colorSpaceStartNT)
{
	char *FnName="AlignColorSpaceInitializeAtStart";
	int32_t i, j, k;

	/* Normal initialization */
	/* Allow the alignment to start anywhere in the reference */
	for(j=0;j<endCol+1;j++) {
		for(k=0;k<alphabetSize;k++) {
			matrix->cells[0][j].h.score[k] = NEGATIVE_INFINITY;
			matrix->cells[0][j].h.from[k] = StartCS;
			matrix->cells[0][j].h.length[k] = 0;

			/* Assumes both DNA and colorSpaceStartNT are upper case */
			if(DNA[k] == colorSpaceStartNT) { 
				/* Starting adaptor NT */
				matrix->cells[0][j].s.score[k] = 0;
			}
			else {
				matrix->cells[0][j].s.score[k] = NEGATIVE_INFINITY;
			}
			matrix->cells[0][j].s.from[k] = StartCS;
			matrix->cells[0][j].s.length[k] = 0;

			matrix->cells[0][j].v.score[k] = NEGATIVE_INFINITY;
			matrix->cells[0][j].v.from[k] = StartCS;
			matrix->cells[0][j].v.length[k] = 0;
		}
	}
	/* Row i (i>0) column 0 should be negative infinity since we want to
	 * align the full read */
	char prevBase = colorSpaceStartNT;
	for(i=1;i<endRow+1;i++) {
		char curBase;
		if(0 == ConvertBaseAndColor(prevBase, BaseToInt(colors[i-1]), &curBase)) {
			PrintError(FnName, "curBase", "Could not convert base and color", Exit, OutOfRange);
		}
		for(k=0;k<alphabetSize;k++) {
			matrix->cells[i][0].h.score[k] = NEGATIVE_INFINITY;
			matrix->cells[i][0].h.from[k] = StartCS;
			matrix->cells[i][0].h.length[k] = 0;

			matrix->cells[i][0].s.score[k] = NEGATIVE_INFINITY;
			matrix->cells[i][0].s.from[k] = StartCS;
			matrix->cells[i][0].s.length[k] = 0;

			// Allow an insertion
			if(DNA[k] == curBase) { // Must be consistent with the read (no color errors please)
				if(i == 1) { // Allow for an insertion start
					matrix->cells[i][0].v.score[k] = matrix->cells[i-1][0].s.score[BaseToInt(colorSpaceStartNT)] + sm->gapOpenPenalty;
					matrix->cells[i][0].v.from[k] = BaseToInt(colorSpaceStartNT) + 1 + (ALPHABET_SIZE + 1); /* see the enum */
					matrix->cells[i][0].v.length[k] = matrix->cells[i-1][0].s.length[BaseToInt(colorSpaceStartNT)] + 1;
				}
				else { // Allow for an insertion extension
					int32_t fromNT = BaseToInt(prevBase); // previous NT
					matrix->cells[i][0].v.score[k] = matrix->cells[i-1][0].v.score[fromNT] + sm->gapExtensionPenalty;
					matrix->cells[i][0].v.from[k] = fromNT + 1 + 2*(ALPHABET_SIZE + 1); /* see the enum */
					matrix->cells[i][0].v.length[k] = matrix->cells[i-1][0].v.length[fromNT] + 1;
				}
				LOWERBOUNDSCORE(matrix->cells[i][0].v.score[k]);
			}
			else {
				matrix->cells[i][0].v.score[k] = NEGATIVE_INFINITY;
				matrix->cells[i][0].v.from[k] = StartCS;
				matrix->cells[i][0].v.length[k] = 0;
			}
		}
		prevBase = curBase;
	}
}

void AlignColorSpaceInitializeToExtend(char *colors,
		AlignMatrix *matrix,
		ScoringMatrix *sm, 
		int32_t readLength,
		int32_t referenceLength,
		int32_t startRow, 
		int32_t startCol,
		int32_t alphabetSize)
{
	char *FnName="AlignColorSpaceInitializeToExtend";
	int32_t i, j, k, endRow, endCol;

	assert(0 < startRow && 0 < startCol);

	endRow = readLength;
	endCol = referenceLength;

	/* Special initialization */

	/* Initialize the corner cell */
	// Check that the match has been filled in 
	for(k=0;k<alphabetSize;k++) {
		assert(1 + ALPHABET_SIZE < matrix->cells[startRow][startCol].s.from[k] &&
				matrix->cells[startRow][startCol].s.from[k] <= 2*(ALPHABET_SIZE + 1));
		assert(startRow <= matrix->cells[startRow][startCol].s.length[k]);
		// Do not allow a deletion or insertion
		matrix->cells[startRow][startCol].h.score[k] = matrix->cells[startRow][startCol].v.score[k] = NEGATIVE_INFINITY-1;
		matrix->cells[startRow][startCol].h.from[k] = matrix->cells[startRow][startCol].v.from[k] = StartNT;
		matrix->cells[startRow][startCol].h.length[k] = matrix->cells[startRow][startCol].v.length[k] = 0;
	}

	// TODO
	for(j=startCol+1;j<endCol+1;j++) { // Columns
		for(k=0;k<alphabetSize;k++) { // To NT
			if(j == startCol + 1) { // Allow for a deletion start
				matrix->cells[startRow][j].h.score[k] = matrix->cells[startRow][j-1].s.score[k] + sm->gapOpenPenalty;
				matrix->cells[startRow][j].h.length[k] = matrix->cells[startRow][j-1].s.length[k] + 1;
				matrix->cells[startRow][j].h.from[k] = k + 1 + (ALPHABET_SIZE + 1); /* see the enum */ 
			}
			else { // Allow for a deletion extension
				matrix->cells[startRow][j].h.score[k] = matrix->cells[startRow][j-1].h.score[k] + sm->gapExtensionPenalty;
				matrix->cells[startRow][j].h.length[k] = matrix->cells[startRow][j-1].h.length[k] + 1;
				matrix->cells[startRow][j].h.from[k] = k + 1;
			}
			LOWERBOUNDSCORE(matrix->cells[startRow][j].h.score[k]);

			// Do not allow for a match or an insertion
			matrix->cells[startRow][j].s.score[k] = matrix->cells[startRow][j].v.score[k] = NEGATIVE_INFINITY;
			matrix->cells[startRow][j].s.from[k] = matrix->cells[startRow][j].v.from[k] = StartNT;
			matrix->cells[startRow][j].s.length[k] = matrix->cells[startRow][j].v.length[k] = 0;
		}
	}

	/* Align the full read */
	for(i=startRow+1;i<endRow+1;i++) {
		char base;
		int32_t fromNT;
		/* Get the current color for the read */
		assert(1 < i); // Otherwise we should use the COLOR_SPACE_START_NT for colors
		for(k=0;k<alphabetSize;k++) {
			// Do not allow for a match or a deletion
			matrix->cells[i][startCol].h.score[k] = matrix->cells[i][startCol].s.score[k] = NEGATIVE_INFINITY;
			matrix->cells[i][startCol].h.from[k] = matrix->cells[i][startCol].s.from[k] = StartNT;
			matrix->cells[i][startCol].h.length[k] = matrix->cells[i][startCol].s.length[k] = 0;

			/* Get from base for extending an insertion */
			if(0 == ConvertBaseAndColor(DNA[k], BaseToInt(colors[i-1]), &base)) {
				PrintError(FnName, NULL, "Could not convert base and color", Exit, OutOfRange);
			}
			fromNT=BaseToInt(base);

			if(i == startRow + 1) { // Allow for an insertion start
				matrix->cells[i][startCol].v.score[k] = matrix->cells[i-1][startCol].s.score[fromNT] + sm->gapOpenPenalty;
				matrix->cells[i][startCol].v.length[k] = matrix->cells[i-1][startCol].s.length[fromNT] + 1;
				matrix->cells[i][startCol].v.from[k] = fromNT + 1 + (ALPHABET_SIZE + 1);
			}
			else { // Allow for an insertion extension
				matrix->cells[i][startCol].v.score[k] = matrix->cells[i-1][startCol].v.score[fromNT] + sm->gapExtensionPenalty;
				matrix->cells[i][startCol].v.length[k] = matrix->cells[i-1][startCol].v.length[fromNT] + 1;
				matrix->cells[i][startCol].v.from[k] = fromNT + 1 + 2*(ALPHABET_SIZE + 1);
			}
			LOWERBOUNDSCORE(matrix->cells[i][startCol].v.score[k]);
		}
	}
}

void AlignColorSpaceFillInCell(char *colors,
		int32_t readLength,
		char *reference,
		int32_t referenceLength,
		ScoringMatrix *sm,
		AlignMatrix *matrix,
		int32_t row,
		int32_t col,
		char curColor,
		int32_t maxH,
		int32_t maxV,
		int32_t alphabetSize)
{
	char *FnName = "AlignColorSpaceFillInCell";
	int32_t k, l;

	/* Deletion */
	if(maxV <= row - col) { // Out of bounds, do not consider
		for(k=0;k<alphabetSize;k++) { /* To NT */
			/* Update */
			matrix->cells[row+1][col+1].h.score[k] = NEGATIVE_INFINITY-1;
			matrix->cells[row+1][col+1].h.from[k] = NoFromCS;
			matrix->cells[row+1][col+1].h.length[k] = INT_MIN;
		}
	}
	else {
		for(k=0;k<alphabetSize;k++) { /* To NT */
			int32_t maxScore = NEGATIVE_INFINITY-1;
			int maxFrom = -1;
			char max = GAP;
			int maxLength = 0;

			int32_t curScore=NEGATIVE_INFINITY;
			int curLength=-1;

			/* Deletion starts or extends from the same base */

			/* New deletion */
			curLength = matrix->cells[row+1][col].s.length[k] + 1;
			/* Deletion - previous column */
			/* Ignore color error since one color will span the entire
			 * deletion.  We will consider the color at the end of the deletion.
			 * */
			curScore = matrix->cells[row+1][col].s.score[k] + sm->gapOpenPenalty;
			/* Make sure we aren't below infinity */
			LOWERBOUNDSCORE(curScore);
			if(curScore > maxScore) {
				maxScore = curScore;
				maxFrom = k + 1 + (ALPHABET_SIZE + 1); /* see the enum */ 
				max = GAP;
				maxLength = curLength;
			}

			/* Extend current deletion */
			curLength = matrix->cells[row+1][col].h.length[k] + 1;
			/* Deletion - previous column */
			curScore = matrix->cells[row+1][col].h.score[k] + sm->gapExtensionPenalty;
			/* Ignore color error since one color will span the entire
			 * deletion.  We will consider the color at the end of the deletion.
			 * */
			/* Make sure we aren't below infinity */
			LOWERBOUNDSCORE(curScore);
			if(curScore > maxScore) {
				maxScore = curScore;
				maxFrom = k + 1; /* see the enum */ 
				max = GAP;
				maxLength = curLength;
			}
			/* Update */
			matrix->cells[row+1][col+1].h.score[k] = maxScore;
			matrix->cells[row+1][col+1].h.from[k] = maxFrom;
			//matrix->cells[row+1][col+1].h.[k] = maxColorError;
			matrix->cells[row+1][col+1].h.length[k] = maxLength;
		}
	}

	/* Match/Mismatch */
	for(k=0;k<alphabetSize;k++) { /* To NT */
		int32_t maxScore = NEGATIVE_INFINITY-1;
		int maxFrom = -1;
		char max = GAP;
		int maxLength = -1;

		for(l=0;l<alphabetSize;l++) { /* From NT */
			int32_t curScore=NEGATIVE_INFINITY+1;
			int curLength=-1;
			char convertedColor='X';
			int32_t scoreNT, scoreColor;

			/* Get color */
			if(0 == ConvertBaseToColorSpace(DNA[l], DNA[k], &convertedColor)) {
				PrintError(FnName, "convertedColor", "Could not convert base to color space", Exit, OutOfRange);
			}
			convertedColor=COLORFROMINT(convertedColor);
			/* Get NT and Color scores */
			scoreNT = ScoringMatrixGetNTScore(reference[col], DNA[k], sm);
			scoreColor = ScoringMatrixGetColorScore(curColor,
					convertedColor,
					sm);

			/* From Horizontal - Deletion */
			curLength = matrix->cells[row][col].h.length[l] + 1;
			/* Add previous with current NT */
			curScore = matrix->cells[row][col].h.score[l] + scoreNT;
			/* Add score for color error, if any */
			curScore += scoreColor;
			/* Make sure we aren't below infinity */
			LOWERBOUNDSCORE(curScore);
			if(curScore > maxScore) {
				maxScore = curScore;
				maxFrom = l + 1; /* see the enum */ 
				max = (curColor == convertedColor)?GAP:curColor; /* Keep original color */
				maxLength = curLength;
			}

			/* From Vertical - Insertion */
			curLength = matrix->cells[row][col].v.length[l] + 1;
			/* Add previous with current NT */
			curScore = matrix->cells[row][col].v.score[l] + scoreNT;
			/* Add score for color error, if any */
			curScore += scoreColor;
			/* Make sure we aren't below infinity */
			LOWERBOUNDSCORE(curScore);
			if(curScore > maxScore) {
				maxScore = curScore;
				maxFrom = l + 1 + 2*(ALPHABET_SIZE + 1); /* see the enum */ 
				max = (curColor == convertedColor)?GAP:curColor; /* Keep original color */
				maxLength = curLength;
			}

			/* From Diagonal - Match/Mismatch */
			curLength = matrix->cells[row][col].s.length[l] + 1;
			/* Add previous with current NT */
			curScore = matrix->cells[row][col].s.score[l] + scoreNT;
			/* Add score for color error, if any */
			curScore += scoreColor;
			/* Make sure we aren't below infinity */
			LOWERBOUNDSCORE(curScore);
			if(curScore > maxScore) {
				maxScore = curScore;
				maxFrom = l + 1 + (ALPHABET_SIZE + 1); /* see the enum */ 
				max = (curColor == convertedColor)?GAP:curColor; /* Keep original color */
				maxLength = curLength;
			}
		}
		/* Update */
		matrix->cells[row+1][col+1].s.score[k] = maxScore;
		matrix->cells[row+1][col+1].s.from[k] = maxFrom;
		//matrix->cells[row+1][col+1].s.[k] = maxColorError;
		matrix->cells[row+1][col+1].s.length[k] = maxLength;
	}

	/* Insertion */
	if(maxH <= col - referenceLength + readLength + row) {
		/* We are on the boundary, do not consider an insertion */
		for(k=0;k<alphabetSize;k++) { /* To NT */
			/* Update */
			matrix->cells[row+1][col+1].v.score[k] = NEGATIVE_INFINITY-1;
			matrix->cells[row+1][col+1].v.from[k] = NoFromCS;
			//matrix->cells[row+1][col+1].v.[k] = GAP;
			matrix->cells[row+1][col+1].v.length[k] = INT_MIN;
		}
	}
	else {
		for(k=0;k<alphabetSize;k++) { /* To NT */
			int32_t maxScore = NEGATIVE_INFINITY-1;
			int maxFrom = -1;
			char max = GAP;
			int maxLength = 0;

			int32_t curScore=NEGATIVE_INFINITY;
			int curLength=-1;
			char B;
			int fromNT=-1;

			/* Get from base for extending an insertion */
			if(0 == ConvertBaseAndColor(DNA[k], BaseToInt(curColor), &B)) {
				PrintError(FnName, NULL, "Could not convert base and color", Exit, OutOfRange);
			}
			fromNT=BaseToInt(B);

			/* New insertion */
			curScore=NEGATIVE_INFINITY;
			curLength=-1;
			/* Get NT and Color scores */
			curLength = matrix->cells[row][col+1].s.length[fromNT] + 1;
			curScore = matrix->cells[row][col+1].s.score[fromNT] + sm->gapOpenPenalty;
			/*
			   curScore += ScoringMatrixGetColorScore(curColor,
			   convertedColor,
			   sm);
			   */
			/* Make sure we aren't below infinity */
			LOWERBOUNDSCORE(curScore);
			if(curScore > maxScore) {
				maxScore = curScore;
				maxFrom = fromNT + 1 + (ALPHABET_SIZE + 1); /* see the enum */ 
				max = GAP;
				maxLength = curLength;
			}

			/* Extend current insertion */
			curLength = matrix->cells[row][col+1].v.length[fromNT] + 1;
			/* Insertion - previous row */
			curScore = matrix->cells[row][col+1].v.score[fromNT] + sm->gapExtensionPenalty;
			curScore += ScoringMatrixGetColorScore(curColor,
					curColor,
					sm);
			/* Make sure we aren't below infinity */
			LOWERBOUNDSCORE(curScore);
			if(curScore > maxScore) {
				maxScore = curScore;
				maxFrom = fromNT + 1 + 2*(ALPHABET_SIZE + 1); /* see the enum */ 
				max = GAP;
				maxLength = curLength;
			}

			/* Update */
			matrix->cells[row+1][col+1].v.score[k] = maxScore;
			matrix->cells[row+1][col+1].v.from[k] = maxFrom;
			//matrix->cells[row+1][col+1].v.[k] = maxColorError;
			matrix->cells[row+1][col+1].v.length[k] = maxLength;
		}
	}
}

int32_t AlignColorSpaceGetAlphabetSize(char *colors,
		int32_t readLength,
		char *reference,
		int32_t referenceLength) 
{
	int32_t i;
	int32_t alphabetSize=ALPHABET_SIZE;
	/* Check if there are any Ns or 0s */
	for(i=0;i<readLength;i++) {
		if('4' == colors[i]) {
			return (1 + ALPHABET_SIZE);
		}
	}
	for(i=0;i<referenceLength && ALPHABET_SIZE==alphabetSize;i++) {
		if(1 == RGBinaryIsBaseN(reference[i])) {
			return (1 + ALPHABET_SIZE);
		}
	}
	return alphabetSize;
}
