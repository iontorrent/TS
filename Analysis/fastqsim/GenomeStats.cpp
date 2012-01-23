/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../fstrcmp.h"
#include <math.h>
#include <sys/time.h>

#include "../Histogram.h"

#include "../LinuxCompat.h"

void GenerateFragments(char *sourceGenomeName, int numFragments, int fragmentLen, char ***fragmentList)
{
	char *key = "TCAG";

	(*fragmentList) = (char **)malloc(sizeof(char *) * numFragments);
	int i;
	for(i=0;i<numFragments;i++)
		(*fragmentList)[i] = (char *)malloc(fragmentLen+1);

	FILE *fp = NULL;
	fopen_s(&fp, sourceGenomeName, "r");
	if (fp) {
		char line[512]; // limits us to about 500bp fragments right now
		fgets(line, sizeof(line), fp);
		int start = ftell(fp);
		fseek(fp, 0, SEEK_END);
		int end = ftell(fp);
		end -= fragmentLen;
		int genomeSize = end - start - 10; // technically this isn't really the genome size since it includes the return line chars, for long fragments, they are broken up over many lines, hence the 10

		for(i=0;i<numFragments;i++) {
			// load up a random part of the genome
			int fragmentStart = start + rand()%genomeSize;
			fseek(fp, fragmentStart, SEEK_SET);
			fread(line, 1, fragmentLen+10, fp);

			// copy the key, then the random fragment into our next fragment
			(*fragmentList)[i][0] = 0;
			strcpy((*fragmentList)[i], key);

			// copy valid chars into our fragment
			int k = 4;
			int j = 0;
			while (k < fragmentLen && j < (fragmentLen+10)) {
				if (line[j] >= 'A' && line[j] < 'Z') {
					(*fragmentList)[i][k] = line[j];
					k++;
				}
				j++;
			}
			(*fragmentList)[i][k] = 0; // terminate the string
			// printf("Fragment %03d: %s\n", i, (*fragmentList)[i]);
		}

		fclose(fp);
	}
}

void FreeFragments(char **fragmentList, int numFragments)
{
	int i;
	for(i=0;i<numFragments;i++)
		free(fragmentList[i]);
	free(fragmentList);
}

double random(double val)
{
	double r = rand() / (double)RAND_MAX;
	return r * val;
}

// gauss_random - gaussian distributed random number generator
// sigma - the standard deviation, use 1.0 for gauss
double gauss_random(double sigma)
{
	double x, y, r2;
	do {
		x = -1.0 + random(2.0);
		y = -1.0 + random(2.0);
		r2 = x*x + y*y;
	} while (r2 > 1.0 || r2 == 0);
	return sigma * y * sqrt(-2.0 * log(r2)/r2);
}

int GetNumFlows(char *seq, int target, char *flowOrder)
{
	int bases = 0;
	int flows = 0;
	while(seq[bases] != 0 && (bases < target)) {
		while(seq[bases] == flowOrder[flows%4]) {
			bases++;
		}
		flows++;
	}

	return flows;
}

int main(int argc, char *argv[])
{
	int numFragments = 10000;
	int fragmentLen = 165;
	char **fragmentList;
	int targetBases = 100;
	double percentPass = 0.85;

	int seed = 1134;

	char *flowOrder = "TACG";

	Histogram	hist(201, 0, 200);

	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'n':
				argcc++;
				sscanf(argv[argcc], "%d", &numFragments);
			break;

			case 'l':
				argcc++;
				sscanf(argv[argcc], "%d", &fragmentLen);	
			break;

			case 't':
				argcc++;
				sscanf(argv[argcc], "%d", &targetBases);
			break;

			case 'p':
				argcc++;
				sscanf(argv[argcc], "%lf", &percentPass);
			break;
		}
		argcc++;
	}

	srand(seed);

	GenerateFragments("CP000948.fna", numFragments, fragmentLen, &fragmentList);

	// generate histogram of flows to reach target
	int i;
	for(i=0;i<numFragments;i++) {
		int numFlows = GetNumFlows(fragmentList[i], targetBases+4, flowOrder); // notice we also need to get through the key
		hist.Add(numFlows);
	}

	// evaluate histogram at the 85% level - see how many flows to get 85% of reads above target bases level
	int numReads = (int)(numFragments * percentPass + 0.5);
	int reads = 0;
	for(i=0;i<201;i++) {
		reads += hist.GetCount(i);
		if (reads >= numReads)
			break;
	}
	printf("Need %d flows to get %.0lf%% of the reads past %d bases\n", i, percentPass*100.0, targetBases);
	hist.Dump("FlowsTo100bp.txt", 0);

	FreeFragments(fragmentList, numFragments);

	return 0;
}

