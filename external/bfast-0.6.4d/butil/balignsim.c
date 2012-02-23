#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <config.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <zlib.h>
#include <unistd.h>  

#include "../bfast/BError.h"
#include "../bfast/BLib.h"
#include "../bfast/BLibDefinitions.h"
#include "../bfast/RGIndexAccuracy.h"
#include "../bfast/RGMatches.h"
#include "../bfast/AlignedRead.h"
#include "../bfast/ScoringMatrix.h"
#include "../bfast/RunLocalAlign.h"
#include "SimRead.h"
#include "balignsim.h"

#define round(x)(int)(x<0?ceil((x)-0.5):floor((x)+0.5))
#define READS_ROTATE_NUM 10000
#define Name "balignsim"

/* Generate synthetic reads given a number of variants and errors
 * from a reference genome and tests the various local alignment 
 * algorithms. */

int PrintUsage()
{
	fprintf(stderr, "%s %s\n", "bfast", PACKAGE_VERSION);
	fprintf(stderr, "\nUsage:%s [options] <files>\n", Name);
	fprintf(stderr, "\t-i\tFILE\tinput specification file\n");
	fprintf(stderr, "\t-f\tFILE\tSpecifies the file name of the FASTA reference genome\n");
	fprintf(stderr, "\t-x\tFILE\tSpecifies the file name storing the scoring matrix\n");
	fprintf(stderr, "\t-n\tINT\tSpecifies the number of threads to use (Default 1)\n");
	fprintf(stderr, "\t-A\tINT\t0: NT space 1: Color space\n");
	fprintf(stderr, "\t-T\tDIR\tSpecifies the directory in which to store temporary file\n");
	fprintf(stderr, "\t-h\t\tprints this help message\n");

	fprintf(stderr, "\nInput specification file:\n");
	fprintf(stderr, "\tThe input specification file is line-orientated.  Each line\n");
	fprintf(stderr, "\tcontains the specification for one set of simulated reads.\n");
	fprintf(stderr, "\tEach set of reads has 8 fields (all specified on one line).\n");
	fprintf(stderr, "\t\t#1 - 0: gapped 1: ungapped\n");
	fprintf(stderr, "\t\t#2 - 0: no indel 1: deletion 2: insertion\n");
	fprintf(stderr, "\t\t#3 - indel length (if #2 is an indel)\n");
	fprintf(stderr, "\t\t#4 - include errors within insertion 0: false 1: true\n");
	fprintf(stderr, "\t\t#5 - # of SNPs\n");
	fprintf(stderr, "\t\t#6 - # of errors\n");
	fprintf(stderr, "\t\t#7 - read length\n");
	fprintf(stderr, "\t\t#8 - number of reads\n");
	fprintf(stderr, "\nsend bugs to %s\n",
			PACKAGE_BUGREPORT);
	return 1;
}

int main(int argc, char *argv[]) 
{
	RGBinary rg;
	char *fastaFileName=NULL;
	char *scoringMatrixFileName=NULL;
	int alignmentType=Gapped;
	int space = NTSpace;
	int indel = 0;
	int indelLength = 0;
	int withinInsertion = 0;
	int numSNPs = 0;
	int numErrors = 0;
	int readLength = 0;
	int numReads = 0;
	int numThreads = 1;
	char tmpDir[MAX_FILENAME_LENGTH]="./";
	char *inputFileName=NULL;
	int c;
	FILE *fpIn=NULL;

	while((c = getopt(argc, argv, "f:i:n:x:A:T:h")) >= 0) {
		switch(c) {
			case 'f': fastaFileName=strdup(optarg); break;
			case 'i': inputFileName = strdup(optarg); break;
			case 'h': return PrintUsage();
			case 'x': scoringMatrixFileName=strdup(optarg); break;
			case 'n': numThreads=atoi(optarg); break;
			case 'A': space=atoi(optarg); break;
			case 'T': strcpy(tmpDir, optarg); break;
			default: fprintf(stderr, "Unrecognized option: -%c\n", c); return 1;
		}
	}

	if(1 == argc || argc != optind) {
		return PrintUsage();
	}

	if(NULL == inputFileName) {
		PrintError(Name, "inputFileName", "Command line option", Exit, InputArguments);
	}
	if(NULL == fastaFileName) {
		PrintError(Name, "fastaFileName", "Command line option", Exit, InputArguments);
	}


	/* Get reference genome */
	RGBinaryReadBinary(&rg,
			space,
			fastaFileName);

	if(!(fpIn = fopen(inputFileName, "r"))) {
		PrintError(Name, inputFileName, "Could not open file for reading", Exit, OpenFileError);
	}
	while(0 == feof(fpIn)) {
		if(fscanf(fpIn, "%d %d %d %d %d %d %d %d",
					&alignmentType,
					&indel,
					&indelLength,
					&withinInsertion,
					&numSNPs,
					&numErrors,
					&readLength,
					&numReads) < 0) {
			if(0 == feof(fpIn)) {
				PrintError(Name, NULL, "Could not understand line", Exit, OutOfRange);
			}
			break;
		}
		Run(&rg,
				scoringMatrixFileName,
				alignmentType,
				space,
				indel,
				indelLength,
				withinInsertion,
				numSNPs,
				numErrors,
				readLength,
				numReads,
				numThreads,
				tmpDir);
	}
	fclose(fpIn);
	free(inputFileName);
	free(scoringMatrixFileName);
	return 0;
}

/* TODO */
void Run(RGBinary *rg,
		char *scoringMatrixFileName,
		int alignmentType,
		int space,
		int indel,
		int indelLength,
		int withinInsertion,
		int numSNPs,
		int numErrors,
		int readLength,
		int numReads,
		int numThreads,
		char *tmpDir)
{
	char *FnName="../bfast/Run";
	int i, j;
	int64_t rgLength=0;
	gzFile matchesFP=NULL;
	char *matchesFileName=NULL;
	gzFile alignFP=NULL;
	char *alignFileName=NULL;
	gzFile notAlignedFP=NULL;
	char *notAlignedFileName=NULL;
	int32_t totalAlignTime = 0;
	int32_t totalFileHandlingTime = 0;
	double mismatchScore;
	ScoringMatrix sm;
	SimRead r;
	RGMatches m;
	AlignedRead a;
	int32_t score, prev, score_m, score_mm, score_cm, score_ce, wasInsertion;
	int32_t numScoreLessThan, numScoreEqual, numScoreGreaterThan;
	int insertionLength = (2==indel)?indelLength:0;
	int deletionLength = (1==indel)?indelLength:0;
	char string[4096]="\0";
	int ret=0;
	char *s=NULL;

	if(ColorSpace == space &&
			1 == withinInsertion && 
			0 < indelLength && 
			2 == indel) {
		PrintError(Name, "withinInsertion", "Incosistent results will occurs.  Try not using withinInsertion == 1.", Warn, OutOfRange);
	}


	score = prev = score_m = score_mm = score_cm = score_ce = 0;

	/* Check rg to make sure it is in NT Space */
	if(rg->space != NTSpace) {
		PrintError(FnName, "rg->space", "The reference genome must be in NT space", Exit, OutOfRange);
	}


	/* ********************************************************
	 * 1.
	 * Generate reads and create artificial matches file.
	 ********************************************************
	 */

	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "Generating reads and creating artificial matches.\n");

	/* Get the reference genome length */
	for(i=0;i<rg->numContigs;i++) {
		rgLength += rg->contigs[i].sequenceLength;
	}

	/* Open tmp files */
	matchesFP = OpenTmpGZFile(tmpDir, &matchesFileName);
	alignFP = OpenTmpGZFile(tmpDir, &alignFileName);
	notAlignedFP = OpenTmpGZFile(tmpDir, &notAlignedFileName);

	/* Get scoring matrix */
	ScoringMatrixInitialize(&sm);
	if(NULL != scoringMatrixFileName) {
		ScoringMatrixRead(scoringMatrixFileName,
				&sm,
				space);
	}

	/* In these sims we want the scoring matrix to have certain constraints:
	 * All scores for matches must be the same and all scores for mismatches 
	 * must be the same */
	if(space == NTSpace) {
		mismatchScore = sm.ntMatch - sm.ntMismatch;
	}
	else {
		mismatchScore = sm.colorMatch - sm.colorMismatch;
	}
	score_m = sm.ntMatch;
	score_mm = sm.ntMismatch;
	if(ColorSpace == space) {
		score_cm = sm.colorMatch;
		score_ce = sm.colorMismatch;
	}

	/* Seed random number */
	srand(time(NULL));

	/* Create RGMatches */
	RGMatchesInitialize(&m);
	SimReadInitialize(&r);
	fprintf(stderr, "Currently on:\n0");
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
				1,
				0);
		r.readNum = i+1;

		/* Convert into RGMatches */
		RGMatchesInitialize(&m);
		RGMatchesReallocate(&m, 1);
		/* Get score for proper alignment and store in read name */
		score = 0;
		prev = Default;
		wasInsertion=0;
		if(NTSpace == space) {
			for(j=0;j<r.readLength;j++) {
				switch(r.readOneType[j]) {
					case Insertion:
					case InsertionAndSNP:
					case InsertionAndError:
					case InsertionSNPAndError:
						if(Insertion == prev) {
							score += sm.gapExtensionPenalty;  
						}
						else {
							score += sm.gapOpenPenalty;  
						}
						wasInsertion=1;
						break;
					case SNP:
					case Error:
					case SNPAndError:
						score += score_mm;
						break;
					case Default:
						score += score_m;
						break;
					default:
						fprintf(stderr, "r.readOneType[%d]=%d\n", j, r.readOneType[j]);
						PrintError(FnName, "r.readOneType[j]", "Could not recognize type", Exit, OutOfRange);
				}
				prev = r.readOneType[j];
			}
		} 
		else {
			for(j=0;j<r.readLength;j++) {
				switch(r.readOneType[j]) {
					case Insertion:
					case InsertionAndSNP:
					case InsertionAndError:
					case InsertionSNPAndError:
						if(Insertion == prev) {
							score += sm.gapExtensionPenalty;  
						}
						else {
							score += sm.gapOpenPenalty;  
						}
						wasInsertion=1;
						break;
					case SNP:
						score += score_mm + score_cm;
						break;
					case Error:
						score += score_m + score_ce;
						break;
					case SNPAndError:
						score += score_mm + score_ce;
						break;
					case Default:
						score += score_m + score_cm;
						break;
					default:
						fprintf(stderr, "r.readOneType[%d]=%d\n", j, r.readOneType[j]);
						PrintError(FnName, "r.readOneType[j]", "Could not recognize type", Exit, OutOfRange);
				}
				prev = r.readOneType[j];
			}
		}
		if(0 < indelLength && 0 == wasInsertion) {
			/* Add in deletion */
			score += sm.gapOpenPenalty;
			score += (r.indelLength-1)*sm.gapExtensionPenalty;
		}
		m.readName = SimReadGetName(&r);
		sprintf(string, "_score=%d", score);
		strcat((char*)m.readName, string);
		/*
		   assert(NULL==m.readName);
		   m.readName = malloc(sizeof(int8_t)*(SEQUENCE_LENGTH+1));
		   if(NULL == m.readName) {
		   PrintError(FnName, "m.readName", "Could not allocate memory", Exit, MallocMemory);
		   }
		   assert(0 <= sprintf((char*)m.readName, ">%d", score));
		   */
		m.readNameLength = strlen((char*)m.readName);
		m.ends[0].numEntries = 1;
		m.ends[0].readLength = (int)strlen(r.readOne);
		assert(r.readLength > 0);
		m.ends[0].read = malloc(sizeof(int8_t)*(m.ends[0].readLength+1));
		if(NULL==m.ends[0].read) {
			PrintError(FnName, "m.ends[0].read", "Could not allocate memory", Exit, MallocMemory);
		}
		assert(m.ends[0].readLength > 0);
		strcpy((char*)m.ends[0].read, r.readOne); 
		if(ColorSpace == space) {
			m.ends[0].qualLength = m.ends[0].readLength-1;
		}
		else {
			m.ends[0].qualLength = m.ends[0].readLength;
		}
		m.ends[0].qual = malloc(sizeof(char)*(m.ends[0].qualLength + 1));
		if(NULL==m.ends[0].qual) {
			PrintError(FnName, "m.ends[0].qual", "Could not allocate memory", Exit, MallocMemory);
		}
		for(j=0;j<m.ends[0].qualLength;j++) {
			m.ends[0].qual[j] = 'I';
		}
		m.ends[0].maxReached = 0;
		m.ends[0].numEntries = 1;
		m.ends[0].contigs = malloc(sizeof(uint32_t));
		assert(NULL != m.ends[0].contigs);
		m.ends[0].positions = malloc(sizeof(int32_t));
		assert(NULL != m.ends[0].positions);
		m.ends[0].strands= malloc(sizeof(int8_t));
		assert(NULL != m.ends[0].strands);
		m.ends[0].contigs[0] = r.contig;
		m.ends[0].positions[0] = r.pos;
		m.ends[0].strands[0] = r.strand;

		/* Output */
		RGMatchesPrint(matchesFP,
				&m);

		/* Clean up */
		SimReadDelete(&r);
		RGMatchesFree(&m);
	}
	fprintf(stderr, "\r%d\n", numReads);
	fprintf(stderr, "%s", BREAK_LINE);

	/* Re-initialize */
	CloseTmpGZFile(&matchesFP,
			&matchesFileName,
			0);
	if(!(matchesFP=gzopen(matchesFileName, "rb"))) {
		PrintError(FnName, matchesFileName, "Could not re-open file for reading", Exit, OpenFileError);
	}

	/* Run "../bfast/RunDynamicProgramming" from balign */
	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "../bfast/Running local alignment.\n");
	RunDynamicProgramming(matchesFP,
			rg,
			scoringMatrixFileName,
			Gapped,
			Unconstrained,
			AllAlignments,
			space,
			0,
			INT_MAX,
			readLength, 
			INT_MAX,
			AVG_MISMATCH_QUALITY,
			numThreads,
			100000,
			0,
			0,
			0,
			0,
			alignFP,
			&totalAlignTime,
			&totalFileHandlingTime);
	fprintf(stderr, "%s", BREAK_LINE);

	/* Re-initialize */
	CloseTmpGZFile(&alignFP,
			&alignFileName,
			0);
	if(!(alignFP=gzopen(alignFileName, "rb"))) {
		PrintError(FnName, alignFileName, "Could not re-open file for reading", Exit, OpenFileError);
	}

	/* Read in output and sum up accuracy */
	fprintf(stderr, "%s", BREAK_LINE);
	fprintf(stderr, "../bfast/Summing up totals.\n");
	AlignedReadInitialize(&a);
	numScoreLessThan = numScoreEqual = numScoreGreaterThan = 0;
	while(EOF != AlignedReadRead(&a,
				alignFP)) {
		/* Get substring */
		s = strstr(a.readName, "score=");
		if(NULL == s) {
			PrintError(FnName, "a.readName", "Could not find \"score=\"", Exit, OutOfRange);
		}
		/* Extract score */
		ret = sscanf(s, "score=%d", &score);
		if(ret != 1) {
			fprintf(stderr, "ret=%d\nscore=%d\n", ret, score);
			fprintf(stderr, "a.readName=%s\n", a.readName);
			PrintError(FnName, "a.readName", "Could not parse read name", Exit, OutOfRange);
		}

		if(round(a.ends[0].entries[0].score) < score) {
			numScoreLessThan++;
			/*
			   if(FullAlignment == alignmentType) {
			   fprintf(stderr, "a.readName=%s\n", a.readName);
			   fprintf(stderr, "found=%d\nexpected=%d\n",
			   round(a.ends[0].entries[0].score),
			   score);
			   AlignedReadPrintText(&a, stderr);
			   PrintError(FnName, "numScoreLessThan", "The alignment score should not be less than expected", Exit, OutOfRange);
			   }
			   */
		}
		else if(score < round(a.ends[0].entries[0].score)) {
			numScoreGreaterThan++;
			/*
			   PrintError(FnName, "numScoreGreaterThan", "The alignment score was greater than expected", Exit, OutOfRange);
			   */
		}
		else {
			numScoreEqual++;
		}
		/* Free */
		AlignedReadFree(&a);
	}
	fprintf(stderr, "%s", BREAK_LINE);

	/* Close matches file */
	CloseTmpGZFile(&matchesFP, &matchesFileName, 1);
	CloseTmpGZFile(&alignFP, &alignFileName, 1);
	CloseTmpGZFile(&notAlignedFP, &notAlignedFileName, 1);

	fprintf(stdout, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
			numReads,
			numScoreLessThan,
			numScoreEqual,
			numScoreGreaterThan,
			numReads - numScoreLessThan - numScoreEqual - numScoreGreaterThan,
			numSNPs,
			numErrors,
			deletionLength,
			insertionLength,
			totalAlignTime
		   );
}
