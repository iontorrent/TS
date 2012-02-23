#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#include "BLib.h"
#include "BError.h"
#include "BLibDefinitions.h"
#include "ScoringMatrix.h"

/* TODO */
int ScoringMatrixRead(char *scoringMatrixFileName, 
		ScoringMatrix *sm,
		int space)
{
	char *FnName="ScoringMatrixRead";
	FILE *fp;

	/* Open the scoring matrix file */
	if((fp=fopen(scoringMatrixFileName, "r"))==0) {
		PrintError(FnName, scoringMatrixFileName, "Could not open scoringMatrixFileName for reading", Exit, OpenFileError);
	}

	/* Read in the gap open penalty,
	 * gap extension penalty,
	 * nt match score,
	 * nt mismatch score */
	if(fscanf(fp, "%d %d %d %d", &sm->gapOpenPenalty,
				&sm->gapExtensionPenalty,
				&sm->ntMatch,
				&sm->ntMismatch) == EOF) {
		PrintError(FnName, scoringMatrixFileName, "Could not read in the gap open penalty, gap extension penalty, nt match score, and nt mismatch score", Exit, OutOfRange);
	}

	if(space == 1) {
		if(fscanf(fp, "%d %d", &sm->colorMatch,
					&sm->colorMismatch) == EOF) {
			PrintError(FnName, scoringMatrixFileName, "Could not read in the color match score and color mismatch score", Exit, OutOfRange);
		}
	}

	ScoringMatrixCheck(sm, space);

	/* Close the file */
	fclose(fp);

	return 1;
}

/* TODO */
void ScoringMatrixInitialize(ScoringMatrix *sm)
{
	sm->gapOpenPenalty=SCORING_MATRIX_GAP_OPEN;
	sm->gapExtensionPenalty=SCORING_MATRIX_GAP_EXTEND;
	sm->ntMatch=SCORING_MATRIX_NT_MATCH;
	sm->ntMismatch=SCORING_MATRIX_NT_MISMATCH;
	sm->colorMatch=SCORING_MATRIX_COLOR_MATCH;
	sm->colorMismatch=SCORING_MATRIX_COLOR_MISMATCH;
}

/* TODO */
/* For color space only */
int32_t ScoringMatrixCheck(ScoringMatrix *sm,
		int space) {
	char *FnName="ScoringMatrixCheck";

	if(0 < sm->gapOpenPenalty) {
		PrintError(FnName, "sm->gapOpenPenalty", "Must be less than or equal to zero", Exit, OutOfRange);
	}
	if(0 < sm->gapExtensionPenalty) {
		PrintError(FnName, "sm->gapExtensionPenalty", "Must be less than or equal to zero", Exit, OutOfRange);
	}
	if(sm->gapExtensionPenalty < sm->gapOpenPenalty) {
		PrintError(FnName, "sm->gapExtensionPenalty < sm->gapOpenPenalty", "Gap extend must be greater than gap open", Exit, OutOfRange);
	}

	if(sm->gapExtensionPenalty < sm->ntMismatch) {
		PrintError(FnName, "sm->gapExtensionPenalty < sm->ntMismatch", "Gap extend must be greater than mismatch", Exit, OutOfRange);
	}

	if(sm->ntMismatch <= sm->gapOpenPenalty) {
		PrintError(FnName, "sm->ntMismatch <= sm->gapOpenPenalty", "Mismatch must be greater than one-base gap", Exit, OutOfRange);
	}

	if(sm->ntMatch < 0) {
		PrintError(FnName, "sm->ntMatch", "Must be greater than or equal to zero", Exit, OutOfRange);
	}
	if(ColorSpace == space && sm->colorMatch < 0) {
		PrintError(FnName, "sm->colorMatch", "Must be greater than or equal to zero", Exit, OutOfRange);
	}
	if(0 < sm->ntMismatch) {
		PrintError(FnName, "sm->ntMismatch", "Must be less than or equal to zero", Exit, OutOfRange);
	}
	if(ColorSpace == space && 0 < sm->colorMismatch) {
		PrintError(FnName, "sm->colorMismatch", "Must be less than or equal to zero", Exit, OutOfRange);
	}
	return 1;
}

int32_t ScoringMatrixGetNTScore(char a,
		char b,
		ScoringMatrix *sm)
{
	return (ToUpper(a) == ToUpper(b)) ? sm->ntMatch : sm->ntMismatch;
}

int32_t ScoringMatrixGetColorScore(char a, 
		char b, 
		ScoringMatrix *sm) 
{
	return (a == b) ? sm->colorMatch : sm->colorMismatch;
}
