/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// Wells stats tool
// (c) 2009
// $Rev: $
//      $Date: $
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>

#include "RawWells.h"
#include "Histogram.h"

struct LookupInfo {
	int r;
	int c;
};

int numLookups = 0;
LookupInfo *lookupInfo = NULL;
int lookupLimit = 1000;

void LookupInit(char *fileName)
{
	FILE *fp = fopen(fileName, "r");
	char buf[256];
	int r, c;
	while (fgets(buf, sizeof(buf), fp)) {
		if (buf[0] == 'r' || buf[0] == '#')
			continue; // skip any non-valid entries or comments
		sscanf(buf, "%d %d", &r, &c);
		if (lookupInfo == NULL)
			lookupInfo = (LookupInfo *)malloc(sizeof(LookupInfo));
		else
			lookupInfo = (LookupInfo *)realloc(lookupInfo, (numLookups+1) * sizeof(LookupInfo));
		lookupInfo[numLookups].r = r;
		lookupInfo[numLookups].c = c;
		numLookups++;
		if (numLookups == lookupLimit)
			break;
	}
	fclose(fp);
}

bool LookupFind(int r, int c)
{
	int i;
	for(i=0;i<numLookups;i++) {
		if (lookupInfo[i].r == r && lookupInfo[i].c == c)
			return true;
	}
	return false;
}

int main(int argc, char *argv[])
{
	// set some defaults
	char *expDir = ".";
	bool allreads = false;
	bool unmappedReads = false;
	bool lookup = false;

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'e': // set exp dir
				argcc++;
				expDir = argv[argcc];
			break;

			case 'a':
				allreads = true;
			break;

			case 'u':
				unmappedReads = true;
			break;

			case 'l':
				argcc++;
				lookup = true;
				LookupInit(argv[argcc]);
			break;

			case 'L':
				argcc++;
				sscanf(argv[argcc], "%d", &lookupLimit);
			break;
		}
		argcc++;
	}

	// crazy, but only way to get rows/cols right now is from mask.
        Mask mask(1,1);
	char maskPath[MAX_PATH_LENGTH];
	sprintf(maskPath, "%s/bfmask.bin", expDir);
	mask.SetMask(maskPath);

	Histogram meanHist(1001, -1.0, 3.0);
	Histogram stdevZeromerHist(1001, 0.0, 3.0);
	Histogram stdevOnemerHist(1001, 0.0, 3.0);
	Histogram avgZeromer(1001, -2.0, 1.0);

	int w = mask.W();
	int h = mask.H();
	int validReads[w][h];
	memset(validReads, 0, sizeof(validReads));
	char blastFile[MAX_PATH_LENGTH];
	sprintf(blastFile, "%s/keypass.rpt", expDir);
	FILE *fp = fopen(blastFile, "r");
	if (fp) {
		char line[256];
		while (fgets(line, sizeof(line), fp)) {
			int row, col;
			sscanf(line, "r%d|c%d", &row, &col);
			char *ptr = strrchr(line, '|');
			ptr++;
			double qual;
			int len;
			sscanf(ptr, "%lf %d", &qual, &len);
			validReads[col][row] = 0;
			if (len > 50 && qual > 0.9) { // look at high quality reads
				validReads[col][row] |= 1;
			}
			if (len > 30 && qual > 0.8) { // look at medium quality reads
				validReads[col][row] |= 2;
			}
			if (len > 20 && !(validReads[col][row] & 4))
				validReads[col][row] |= 4;
		}
		fclose(fp);
	}

	RawWells wells(expDir, "1.wells", mask.H(), mask.W());
	wells.OpenForRead();
	const WellData *data = NULL;
	int numFlows = wells.NumFlows();
	double measured[numFlows];
	int keypassCount = 0;
	int keypassFailCount = 0;
	const int numFlowsInKey = 7;
	int keypassLib[numFlowsInKey] = {1, 0, 1, 0, 0, 1, 0};
	// int keypassLib[numFlowsInKey] = {0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1};
	// const int numFlowsInKey = 11;
	int i;
	int zeromerCount = 0;
	int onemerCount = 0;

	double runningAvg0mer = 0.0;
	double runningAvg1mer = 0.0;
	double runningAvgN0mer = 0.0;
	double runningAvgN1mer = 0.0;
	int runningAvgCount = 0;

	for(i=0;i<numFlowsInKey;i++) {
		if (keypassLib[i] == 0)
			zeromerCount++;
		if (keypassLib[i] == 1)
			onemerCount++;
	}
	while ((data = wells.ReadNextRegionData()) != NULL) {
		// look for all live library reads
		// 1 0 1 0 0 1 0 N - library TCAG key with TACG flow order
		if (mask.Match(data->x, data->y, MaskLib)) {
			bool useit = false;
			if (unmappedReads && validReads[data->x][data->y] == 0)
				useit = true;
			else if (!unmappedReads && (allreads || (validReads[data->x][data->y] > 0))) // look at any (4), medium (2), or high(1) quality reads
				useit = true;

			if (lookup) {
				useit = false;
				if (LookupFind(data->y, data->x))
					useit = true;
			}
			if (useit) { 
			// do simple keypass on raw well data:
			//   1 - subtract avg 0-mer
			//   2 - normalize to avg 1-mer
			//   3 - threshold to generate vector, compare to 1st 7 flows of known
			double avg0mer = 0.0;
			for(i=0;i<numFlowsInKey;i++) {
				if (keypassLib[i] == 0)
					avg0mer += data->flowValues[i];
			}
			avg0mer /= zeromerCount;

			double avg1mer = 0.0;
			for(i=0;i<numFlowsInKey;i++) {
				if (keypassLib[i] == 1)
					avg1mer += data->flowValues[i];
			}
			avg1mer /= onemerCount;

			// keep a running avg on 0-mer & 1-mer raw key signal
			runningAvg0mer += avg0mer;
			runningAvg1mer += avg1mer;
			runningAvgCount++;

			// key normalization...
			avg0mer = 0.0; // force our algorithm to assume weka was right, and 0-mer on avg is already 0 !!! (need to think on this)
			int flow;
			for(flow=0;flow<numFlows;flow++) {
				measured[flow] = data->flowValues[flow] - avg0mer;
			}

			double mult = 1.0/avg1mer;
			for(flow=0;flow<numFlows;flow++) {
				measured[flow] *= mult;
			}

			// calc avg normalized 0-mers & 1-mers
			double avgN0mer = 0.0;
			double avgN1mer = 0.0;
			for(i=0;i<numFlowsInKey;i++) {
				if (keypassLib[i] == 0)
					avgN0mer += measured[i];
				if (keypassLib[i] == 1)
					avgN1mer += measured[i];
			}
			avgN0mer /= zeromerCount;
			avgN1mer /= onemerCount;

			runningAvgN0mer += avgN0mer;
			runningAvgN1mer += avgN1mer;

			// keypass...
			int keypassVec[numFlowsInKey];
			bool keypass = true;
			for(flow=0;flow<numFlowsInKey;flow++) {
				keypassVec[flow] = (int)(measured[flow]+0.5);
				if (keypassVec[flow] != keypassLib[flow])
					keypass = false;
			}

			if (keypass) {
				keypassCount++;
			} else {
				keypassFailCount++;
			}

			// now, lets generate a few metrics and see how they correlate to mapped reads, interest is in mixed fragments
			// metric1 - the dist between the avg 0-mer and the avg 1-mer - for this, we can usually call the read without cafie corrections for around 40 flows, so we go ahead and do that, then avg the 0-mer and 1-mer signals, then report the mean dist, and the stdev on the 0-mers and 1-mers
			int numTestFlows = 12;
			double onemerSig[40];
			double zeromerSig[40];
			int onemerCount = 0;
			int zeromerCount = 0;
			int base;
			for(flow=numFlowsInKey+1;flow<numTestFlows+numFlowsInKey+1;flow++) { // note we ignore the key
				base = (int)(measured[flow]+0.5);
				if (base == 0) {
					zeromerSig[zeromerCount] = measured[flow];
					zeromerCount++;

					avgZeromer.Add(measured[flow]);
				}
				if (base == 1) {
					onemerSig[onemerCount] = measured[flow];
					onemerCount++;
				}
			}

			// if we have sane counts, calc metrics for this read
			double avgZeroMer = 0.0;
			double avgOneMer = 0.0;
			double onemerStdev = 0.0;
			double zeromerStdev = 0.0;
			if (zeromerCount > 2 && onemerCount > 2) {
				int k;
				for(k=0;k<zeromerCount;k++) {
					avgZeroMer += zeromerSig[k];
				}
				avgZeroMer /= (double)zeromerCount;
				for(k=0;k<onemerCount;k++) {
					avgOneMer += onemerSig[k];
				}
				avgOneMer /= (double)onemerCount;

				double delta = 0.0;
				for(k=0;k<zeromerCount;k++) {
					delta = avgZeroMer - zeromerSig[k];
					zeromerStdev += delta*delta;
				}
				zeromerStdev = sqrt(zeromerStdev);
				for(k=0;k<onemerCount;k++) {
					delta = avgOneMer - onemerSig[k];
					onemerStdev += delta*delta;
				}
				onemerStdev = sqrt(onemerStdev);

				meanHist.Add(avgOneMer - avgZeroMer);
				stdevZeromerHist.Add(zeromerStdev);
				stdevOnemerHist.Add(onemerStdev);
			}
			}
		}
	}

	wells.Close();

	// dump some stats
	printf("Reads: pass/fail/all %d/%d/%d\n", keypassCount, keypassFailCount, keypassCount + keypassFailCount);
	printf("Avg signals in key:  Raw:  0-mer: %.4lf  1-mer: %.4lf  Norm:  0-mer: %.4lf  1-mer: %.4lf\n",
		runningAvg0mer/runningAvgCount, runningAvg1mer/runningAvgCount, runningAvgN0mer/runningAvgCount, runningAvgN1mer/runningAvgCount);

	meanHist.Dump("AvgDist.txt", 1);
	stdevZeromerHist.Dump("ZeromerStdev.txt", 1);
	stdevOnemerHist.Dump("OnemerStdev.txt", 1);
	avgZeromer.Dump("Avg0mer.txt", 1);
}

