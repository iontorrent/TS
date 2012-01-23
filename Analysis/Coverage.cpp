/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include "Histogram.h"

#define genomeSize 33476

bool isSeqChar(char c)
{
	if ((c == 'g') || (c == 't') || (c == 'a') || (c == 'c') || (c == '-'))
		return true;
	return false;
}

int main(int argc, char *argv[])
{
	int minLen = 21;
	char *blastFile = "blast.output";
	bool verbose = false;
	bool oldBlast = false;
	Histogram hist(101, 0, 100);
	bool wantRowCol = true;

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'l': // min length we accept
				argcc++;
				minLen = atoi(argv[argcc]);
			break;

			case 'b': // blast file
				argcc++;
				blastFile = argv[argcc];
			break;

			case 'v':
				verbose = true;
			break;

			case 'o': // use old blast format
				oldBlast = true;
			break;

			case 'r':
				wantRowCol = false;
			break;
		}
		argcc++;
	}

	unsigned char *coverage;
	coverage = (unsigned char *)malloc(genomeSize);
	memset(coverage, 0, genomeSize);

	FILE *fp = fopen(blastFile, "r");
	char line[256];
	char qtext[256];
	char stext[256];
	int qstart, qend;
	int sstart, send;
	double score = 0.0;
	int readCount = 0;
	int blastReads = 0;
	char rowcol[256];
	int matchMismatch = 0;
	while (fgets(line, sizeof(line), fp)) {
		// capture the score for the read we are about to process
		if (strncmp(line, " Score =", 8) == 0) {
			sscanf(line, " Score = %lf", &score);
		}

		// capture the row/col query
		if (strncmp(line, "Query= ", 7) == 0) {
			sscanf(line, "Query= %s", rowcol);
		}

		// capture the match/mismatch info
		if (strncmp(line, " Identities =", 12) == 0) {
			int matchScore, totalScore;
			sscanf(line, " Identities = %d/%d", &matchScore, &totalScore);
			matchMismatch = matchScore - (totalScore-matchScore);
		}

		// look for a query line, get start & end, if long enough and start near 0 then count
		if (strncmp(line, "Query:", 6) == 0) {
			blastReads++;
			// get query line info
			sscanf(line, "Query: %d %s %d", &qstart, qtext, &qend);
			// printf("Query start: %d  end: %d  text: %s\n", qstart, qend, qtext);
			if (qstart > qend) {
				int temp = qstart;
				qstart = qend;
				qend = temp;
			}

			// test to see if this read starts near the beginning of our sequence, and is long enough
			if ((qstart < 4) && (qend-qstart+1 >= minLen)) {
				// skip the graphical alignment bars
				assert(fgets(line, sizeof(line), fp));

				// get the subject line info
				assert(fgets(line, sizeof(line), fp));
				sscanf(line, "Sbjct: %d %s %d", &sstart, stext, &send);

				// see if this read continues
				assert(fgets(line, sizeof(line), fp));
				assert(fgets(line, sizeof(line), fp));
				assert(fgets(line, sizeof(line), fp));
				if (strncmp(line, "Query: ", 7) == 0) {
					char text[256];
					int dummy;
					sscanf(line, "Query: %d %s", &dummy, text);
					strcat(qtext, text); // append
					// skip the alignment bars
					assert(fgets(line, sizeof(line), fp));
					// get the subject line
					assert(fgets(line, sizeof(line), fp));
					sscanf(line, "Sbjct: %d %s", &dummy, text);
					strcat(stext, text); // append
				}

				// printf("Sbjct start: %d  end: %d  text: %s\n", sstart, send, stext);

				// see what direction the read aligned to
				int len = strlen(qtext);
				int start;
				int dir = 1;
				int i;
				start = sstart;
				if (sstart > send) {
					dir = -1;
					start = send;
				}

				// give ourselves credit for each base that matched in both the subject & query
				for(i=0;i<len;i++) {
					if (qtext[i] == stext[i]) {
						coverage[start]++;
					}
					start += dir;
					if (start < 0)
						break;
				}

				if (verbose) {
					int len = strlen(qtext);
					int i;
					for(i=0;i<len;i++) qtext[i] = toupper(qtext[i]);
					len = strlen(stext);
					for(i=0;i<len;i++) stext[i] = toupper(stext[i]);
					// we are writing the output all on one line, separated with ':', so linux sort can work for us
					// later, 'tr' is used to convert the colons into line feeds
					if (wantRowCol)
						printf("Score= %.2lf:Row/Col= %s:%s:", score, rowcol, qtext);
					else
						printf("Score= %.2lf:%s:", score, qtext);
					len = strlen(qtext);
					// we re-generate the alignment bars, and only print when subject & query are the same
					for(i=0;i<len;i++)
						if (qtext[i] == stext[i])
							printf("|");
						else
							printf(" ");
					printf(":%s:\n", stext);
				}

				// here, we are keeping track of our own match-mismatch histogram, and we only count filtered reads
				hist.Add(matchMismatch);
				readCount++;
			}
		}
	}
	fclose(fp);

	int i;
	int count = 0;
	fp = fopen("coverage.txt", "w");
	for(i=0;i<genomeSize;i++) {
		if (coverage[i] > 0)
			count++;
		fprintf(fp, "%d\n", coverage[i]);
	}
	fclose(fp);
	printf("Coverage = %.2lf%%::", 100.0 * (double)count/(double)genomeSize);
	printf("Reads used = %d out of %d::", readCount, blastReads);
	// hist.Dump("MatchMismatch.hist", 2);
	hist.Dump("MatchMismatch.hist");

	free(coverage);
}

