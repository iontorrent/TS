/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
// Mapper - a simple edit distance approach to mapping reads against one or more templates
// inputs: FASTQ/FASTA read file, template list file
// outputs include % reads above some quality threshold, and counts/percentages against each template

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "fstrcmp.h"
#include "../Histogram.h"
#include "../file-io/ion_util.h"

struct TestSeq {
	char	seq[256];
	int	len;
	int	count;
};

/*
TestSeq testSeq[] = {
	{"CATTGTCTTCCGTAGTCGCTGATTG", 25, 0},
	{"TGCCTGCGGCGGCACGCCCAGCCCT", 25, 0},
	{"ACCCGTGTGGCATTCGCAGGTCTGA", 25, 0},
// 	{"CCGGGTTGCGGACAGTGTCTGGTGG", 25, 0}, // bad template 'D' is one base (1st) off from 'E'!
	{"TCCGGGTTGCGGACAGTGTCTGGTG", 25, 0},
	{"ATGAGTCACATCACAACGGAAGATC", 25, 0},
	{"CCTGGATGATGTTGGTAGGCCAGGC", 25, 0},
};
*/

/*
TestSeq testSeq[] = {
        {"CATTGTCTTCCGTAGTCGCTGATTGTTTACCGCCTGATGGGCGAAGAGAAAGAACGAGTAAAGGTCG", 67, 0},
        {"TGCCTGCGGCGGCACGCCCAGCCCTGCACAGGCGGCGAGAAAGATTTCCGGGTCCGGTTTCGAGTTT", 67, 0},
        {"ACCCGTGTGGCATTCGCAGGTCTGAAATTCCAGGATGTGGGTTCTTTCGACTACGGTCGTAACTACG", 67, 0},
        {"TCCGGGTTGCGGACAGTGTCTGGTGGGTAGTTTGACTGGGCGGTCTCCTCCTAAAGAGTAACGGAGG", 67, 0},
        {"ATGAGTCACATCACAACGGAAGATCCAGCAACTTTACGCCTGCCCTTTAAAGAGAAACTCTCTTACG", 67, 0},
};
*/

/*
TestSeq testSeq[] = {
	{"GTGAGTCGAGCGAGTGTGTGAGGAGTGTGAGTGAGTCGTGAGTCCGGGCGGGAGCGACGTGAGAGTGGAGGTGTCGAGAGAGTGA", 85, 0},
	{"AACAGCTATGTCAGCATTGTCTTCCGTAGTCGCTGATTGTTTACCGCCTGATGGGCGAAGAGAAAGAACGAGTAAAGGTCG", 81, 0},
};
*/

// rand100
/*
TestSeq testSeq[] = {
	{"GTGAGTCGAGCGAGTGTGTGAGGAGTGTGAGTGAGTCGTGAGTCCGGGCGGGAGCGACGTGA", 62, 0},
};
*/

/*
TestSeq testSeq[] = {
	{"GTGAGTCGAGCGAGTGTGTGAGGAGTGTGAGTGAGTCGTGAGTCCGGGCGGGAGCGAGTGACTGAGCGGGCTGGCAAGGC", 80, 0},
};
*/

// rand100
TestSeq testSeq[] = {
	{"GTGAGTCGAGCGAGTGTGTGAGGAGTGTGAGTGAGTCGTGAGTCCGGGCGGGAGCGACGTGAGAGTGGAGGTGTCGAGAGAGTGA", 85, 0},
};

//rand1005bk
//TF1.1
//TF1.1 subset
/*
TestSeq testSeq[] = {
	{"TGTGAGTCGAGCGAGTGTGTGAGGAGTGTGAGTGAGTCGTGAGTCCGGGCGGGAGC", 57, 0},
//	{"CTAGTTTTAGGGTCCCCGGGGTTAAAAGGTTCGAACCCGGAAACCACCGTCAAGGGCGAATTCGTTTAAACCTGCAGGACTAGTCCCTTTAGTGAGGGTTAATTCTGAGCTTGGCGTAATCATGGTCATAGCTGTTTCCTGTGTGAAATTGTTATCCGCTCACAATTCCACACAACATACGAGCCGGAAGCATAAAGTGTAAAGC",208,0},
//	{"CTAGTTTTAGGGTCCCCGGGGTTAAAAGGTTCGAAC",35,0},
};
*/

int numTestSeq = sizeof(testSeq) / sizeof(TestSeq);

// GetReadLen - algorithm to compute read length at some given quality
// approach is to convert the ref seq & test seq into flow-space calls (so includes 0's)
// then compare to each other, stop when read errors exceed thresh
int GetReadLen(char *refSeq, int refLen, char *testSeq, int q, bool best, bool details, int *perFlowErrors)
{
// printf("Calculating Q%d score for %s aligned to %s\n", q, testSeq, refSeq);
	int numErrors = (int)(pow(10.0, q/10.0));
	double errRatio = 1.0/numErrors;
	int errCount = 0;
	int testLen = strlen(testSeq);
	if (testLen > refLen)
		testLen = refLen;

	// convert each into array of flow-space hits
	char *flowOrder = "TACG";
	int refArray[1000];
	int testArray[1000];
	int i;
	int refBases = 0;
	int testBases = 0;
	i = 0;
	while (refBases < refLen && testBases < testLen) {
		refArray[i] = 0;
		while (flowOrder[i%4] == refSeq[refBases] && refBases < refLen) {
			refArray[i]++;
			refBases++;
		}

		testArray[i] = 0;
		while (flowOrder[i%4] == testSeq[testBases] && testBases < testLen) {
			testArray[i]++;
			testBases++;
		}

		i++;
	}
	int flowLen = i;
	int readLen = 0;

if (details) {
	printf("Flow-space ref:\n");
	for(i=0;i<flowLen;i++)
		printf("%d", refArray[i]);
	printf("\n");
	printf("Flow-space tf:\n");
	for(i=0;i<flowLen;i++)
		printf("%d", testArray[i]);
	printf("\n");

	// generate the alignment strings
	char refBuf[256];
	char testBuf[256];
	int k = 0;
	int j;
	int iref = 0;
	int itest = 0;
	for(i=0;i<flowLen;i++) {
		if (refArray[i] > 0 || testArray[i] > 0) {
			int max = (refArray[i] > testArray[i] ? refArray[i] : testArray[i]);
			int refCount = 0;
			int testCount = 0;
			for(j=0;j<max;j++) {
				if (refCount < refArray[i]) {
					refBuf[k] = refSeq[iref];
					iref++;
				} else {
					refBuf[k] = '-';
				}
				refCount++;

				if (testCount < testArray[i]) {
					testBuf[k] = testSeq[itest];
					itest++;
				} else {
					testBuf[k] = '-';
				}
				testCount++;

				k++;
			}
		}
	}
	refBuf[k] = 0;
	testBuf[k] = 0;
	printf("%s\n%s\n", refBuf, testBuf);
}

// printf("Using %d flows to test with\n", flowLen);

	int bestReadlen = 0;
	for(i=0;i<flowLen;i++) {
		if (refArray[i] == 0 && testArray[i] == 0) { // match but no read length boost
		} else if (refArray[i] == testArray[i]) { // match & read length boost
			readLen += testArray[i];
			double ratio = (double)errCount/(double)readLen;
			if (ratio <= errRatio)
				bestReadlen = readLen;
		} else { // miss-match
			readLen += testArray[i];
			errCount += (abs(refArray[i] - testArray[i]));
			double ratio = 1.0;
			if (readLen > 0)
				ratio = (double)errCount/(double)readLen;
			if (best) {
				if (ratio <= errRatio)
					bestReadlen = readLen;
			} else {
				if (ratio > errRatio) {
					bestReadlen = readLen;
					break;
				}
			}
			perFlowErrors[i] += errCount;
		}
	}

// printf("Got readlength: %d\n", bestReadlen);
	return bestReadlen;
}

int main(int argc, char *argv[])
{
	FILE *fp_fnq = fopen(argv[1], "r");
	double scoreThresh = 0.6;
	if (argc == 3)
		sscanf(argv[2], "%lf", &scoreThresh);
	char line[1024];
	int i;
	int numReads = 0;
	int mappedReads = 0;
	Histogram *hist[numTestSeq];
	for(i=0;i<numTestSeq;i++)
		hist[i] = new Histogram(testSeq[0].len+1, 0, testSeq[0].len);

	int perFlowErrors[200];
	memset(perFlowErrors, 0, sizeof(perFlowErrors));
	int cumulativeErrors[200];
	memset(cumulativeErrors, 0, sizeof(cumulativeErrors));

	// keep track of where the 'good' reads are
	int readquality[1152*1348];
	int x, y;
	memset(readquality, 0, sizeof(readquality));
	int bestQ10ReadLen = 0;

	while (fgets(line, sizeof(line), fp_fnq)) {
		if (line[0] == '@') {
			int len = strlen(line);
			// remove return line / line feeds
			while (len > 1 && (line[len-1] == '\n' || line[len-1] == '\r'))
				len--;
			line[len] = 0;
				
			// R&D pipe format is "@XXXXX"
			// Commercial pipe format is "@IONPGM_XXXXX"
			if ((line[1] == 'r') && strchr(line, '|')) { // in r|c format
				sscanf(line, "@r%d|c%d", &y, &x);
			} else {
				switch (strlen (line))
				{
					case 13:
                                          ion_id_to_xy(line, &x, &y);
					break;
				
					case 6:
                                          ion_id_to_xy(line, &x, &y);
					break;
				
					default:
						fprintf (stderr, "Unknown read name type: '%s'\n", line);
                        fclose(fp_fnq);
						return (1);
					break;
				}
			}
//fprintf (stdout, "Row %d Col %d\n", y, x);
			if (fgets(line, sizeof(line), fp_fnq)) {
				int len = strlen(line);
				// remove return line / line feeds
				while (len > 1 && (line[len-1] == '\n' || line[len-1] == '\r'))
					len--;
				line[len] = 0;

				numReads++;
				// score the read against each template, picking the best (lowest edit dist) score
				double bestScore = 0.0;
				int edit1, edit2;
				int bestTemplate = -1;
				double score;
				for(i=0;i<numTestSeq;i++) {
					score = fstrcmp (line, len, testSeq[i].seq, testSeq[i].len, 0.0, &edit1, &edit2);
// printf("Score: %.2lf for <%s> to <%s>\n", score, testSeq[i].seq, line);
					if (score > bestScore) {
						bestScore = score;
						bestTemplate = i;
					}
				}

				// if its good enough, count it
				if (bestScore >= scoreThresh) {
					// testSeq[bestTemplate].count++;
					// mappedReads++;

					// and see what it's Q10 readlength is
					int q10readlen = GetReadLen(testSeq[bestTemplate].seq, testSeq[bestTemplate].len, line, 10, true, false, perFlowErrors);
					if (q10readlen >= 8) { // only care if it passes some min length quality
						hist[bestTemplate]->Add(q10readlen);
						testSeq[bestTemplate].count++;
						mappedReads++;

						if (q10readlen >= 17)
							readquality[x+y*1348] = 1;

						if (q10readlen > bestQ10ReadLen)
							bestQ10ReadLen = q10readlen;
					}
/*
if (q10readlen == bestQ10ReadLen) {
	printf("Golden at %d,%d: %s\n", x, y, line);
	GetReadLen(testSeq[bestTemplate].seq, testSeq[bestTemplate].len, line, 10, true, true, perFlowErrors);
}
*/
				}
			}
		}
	}
	fclose(fp_fnq);

	printf("Processed %d reads and %d mapped for %.2lf%%\n", numReads, mappedReads, 100.0*mappedReads/numReads);
	char histName[256];
	for(i=0;i<numTestSeq;i++) {
		printf("Seq: %s had %d hits for %.2lf%%  mean readlen: %.1lf\n",
			testSeq[i].seq, testSeq[i].count, 100.0*testSeq[i].count/numReads, hist[i]->Mean());
		sprintf(histName, "Template_%d_Q10.txt", i+1);
		hist[i]->Dump(histName);//hist[i]->Dump(histName, 1);
	}
	printf("Longest Q10 readlen: %d\n", bestQ10ReadLen);

	// dump the read quality array - formatted like a beadfind mask so our density plotting tools work
	FILE *fp = fopen("readquality.txt", "w");
	if (fp) {
		fprintf(fp, "0, 0, 1348, 1152\n");
		for(y=0;y<1152;y++) {
			for(x=0;x<1348;x++) {
				fprintf(fp, "%d ", readquality[x+y*1348]);
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
		fclose(fp);
	}

        // convert the per flow errors into a cumulative error array
        for(i=0;i<200;i++) {
                if (i > 0)
                        cumulativeErrors[i] = cumulativeErrors[i-1] + perFlowErrors[i];
                else
                        cumulativeErrors[i] = perFlowErrors[i];
        }

	// dump both error plots
	fp = fopen("errorplot.txt", "w");
	if (fp) {
		fprintf(fp, "Per-flow errors:\n");
		for(i=0;i<200;i++) {
			fprintf(fp, "%d\n", perFlowErrors[i]);
		}
		fprintf(fp, "\n\n");

		fprintf(fp, "Cumulative errors:\n");
		for(i=0;i<200;i++) {
			fprintf(fp, "%d\n", cumulativeErrors[i]);
		}
		fprintf(fp, "\n\n");
		fclose(fp);
	}

}

