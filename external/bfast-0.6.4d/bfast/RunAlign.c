#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <errno.h>
#include <limits.h>
#include <zlib.h>
#include "BLibDefinitions.h"
#include "BError.h"
#include "BLib.h"
#include "RGBinary.h"
#include "RGIndex.h"
#include "RGReads.h"
#include "RGMatch.h"
#include "RGMatches.h"
#include "MatchesReadInputFiles.h"
#include "aflib.h"
#include "RunMatch.h"
#include "RunLocalAlign.h"
#include "RunPostProcess.h"
#include "RunAlign.h"

/* TODO */
void RunAlign(
		char *fastaFileName,
		char *readFileName, 
		int compression,
		int space,
		int numThreads,
		char *tmpDir,
		int timing
		)
{
	char *FnName="RunAlign";
	char *tmpMatchFileName=NULL;
	FILE *tmpMatchFP=NULL;
	char *tmpLocalAlignFileName=NULL;
	FILE *tmpLocalAlignFP=NULL;
	int seconds, minutes, hours, startTotalTime, endTotalTime;

	startTotalTime = time(NULL);

	// Open a tmp file
	tmpMatchFP = OpenTmpFile(tmpDir, &tmpMatchFileName);

	// Run Match
	RunMatch(fastaFileName,
			NULL,
			NULL,
			readFileName,
			NULL,
			IndexesMemorySerial,
			compression,
			space,
			1,
			INT_MAX,
			0,
			MAX_KEY_MATCHES,
			MAX_NUM_MATCHES,
			BothStrands,
			numThreads,
			DEFAULT_MATCHES_QUEUE_LENGTH,
			tmpDir,
			timing,
			tmpMatchFP);

	// Close but do not delete
	fclose(tmpMatchFP);

	// Open a tmp file
	tmpLocalAlignFP = OpenTmpFile(tmpDir, &tmpLocalAlignFileName);

	// Run local alignment
	RunAligner(fastaFileName,
			tmpMatchFileName,
			NULL,
			Gapped,
			Constrained,
			AllAlignments,
			space,
			1,
			INT_MAX,
			OFFSET_LENGTH,
			MAX_NUM_MATCHES,
			AVG_MISMATCH_QUALITY,
			numThreads,
			DEFAULT_MATCHES_QUEUE_LENGTH,
			0,
			0,
			NoMirroring,
			0,
			timing,
			tmpLocalAlignFP);

	// Close but do not delete
	fclose(tmpLocalAlignFP);
	// Delete match file
	if(0 != remove(tmpMatchFileName)) {
		PrintError(FnName, tmpMatchFileName, "Could not delete temporary file", Exit, DeleteFileError);
	}

	// Run local aligner
	RGBinary rg;
	RGBinaryReadBinary(&rg, NTSpace, fastaFileName);
	ReadInputFilterAndOutput(&rg,
			tmpLocalAlignFileName,
			BestScore,
			space,
			0,
			0,
			AVG_MISMATCH_QUALITY,
			NULL,
			0,
			numThreads,
			DEFAULT_LOCALALIGN_QUEUE_LENGTH,
			SAM,
			NULL,
			NULL,
			NULL,
			stdout);
	RGBinaryDelete(&rg);

	// Delete local align file
	if(0 != remove(tmpLocalAlignFileName)) {
		PrintError(FnName, tmpLocalAlignFileName, "Could not delete temporary file", Exit, DeleteFileError);
	}

	free(tmpMatchFileName);
	free(tmpLocalAlignFileName);

	if(timing == 1) {
		/* Output total time */
		endTotalTime = time(NULL);
		seconds = endTotalTime - startTotalTime;
		hours = seconds/3600;
		seconds -= hours*3600;
		minutes = seconds/60;
		seconds -= minutes*60;
		if(0 <= VERBOSE) {
			fprintf(stderr, "Total time elapsed: %d hours, %d minutes and %d seconds.\n",
					hours,
					minutes,
					seconds
				   );
		}
	}
}
