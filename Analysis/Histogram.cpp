/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "Histogram.h"
#include "LinuxCompat.h"

#include "dbgmem.h"

Histogram::Histogram()
{
	Init(99, -500.0, 3000.0);
}

Histogram::Histogram(int _numBins, double _minVal, double _maxVal)
{
	Init(_numBins, _minVal, _maxVal);
}

void Histogram::Init(int _numBins, double _minVal, double _maxVal)
{
	numBins = _numBins;
	bin = new int[numBins];
	histValues = new double[numBins];
	minVal = _minVal;
	maxVal = _maxVal;
	count = 0;
	mean = 0.0;

	int i;
	for(i=0;i<numBins;i++) {
		bin[i] = 0;
		histValues[i] = minVal + i * (maxVal-minVal)/(double)(numBins-1);
	}
}

void Histogram::ChangeParams(int _numBins, double _minVal, double _maxVal)
{
        delete [] bin;
        delete [] histValues;
        Init(_numBins, _minVal, _maxVal);
}

Histogram::~Histogram()
{
	delete [] bin;
	delete [] histValues;
}

int Histogram::GetCount(int binIndex)
{
	if (binIndex < 0)
		return 0;
	else if (binIndex >= numBins)
		return 0;
	else
		return bin[binIndex];
}

int Histogram::GetBin(double val)
{
    int binIndex;
    double t;
    t = (val-minVal)/(maxVal-minVal);
    binIndex = (int)(numBins * t);
    if (binIndex >= numBins) {
        binIndex = numBins-1;
    } else if (binIndex < 0) {
        binIndex = 0;
    }
    return binIndex;
}

void Histogram::Add(double val, bool ignoreEnds)
{
	int bidId = GetBin(val);
	if (ignoreEnds && ((bidId == 0) || (bidId == (numBins-1))))
		return;
	bin[bidId]++;
	count++;

	mean += val;
}

double Histogram::Mean()
{
	if (count)
		return mean/count;
	else
		return 0.0;
}

int Histogram::ModeBin()
{
	int modeBin = 0;
	int modeBinVal = bin[0];
	int i;
	for(i=1;i<numBins;i++) {
		if (bin[i] >= modeBinVal) {
			modeBin = i;
			modeBinVal = bin[i];
		}
	}
	return modeBin;
}

double Histogram::StdDev()
{
	// this is an approximation based on the number of bins we have
	// stddev is the sqrt of (the sum of the deltas squared over # samples)

	if (count == 0)
		return 0.0;

	int numSamples = 0;
	int i;
	double meanVal = Mean();
	double binVal;
	double diff;
	double totalVal = 0.0;
	for(i=1;i<numBins-1;i++) { // don't count the outliers
		binVal = minVal + ((double)i/(numBins-1.0))*(maxVal-minVal);
		diff = binVal-meanVal;
		diff = diff*diff;
		numSamples += bin[i];
		totalVal += bin[i] * diff;
	}
	if (numSamples == 0)
		return 0.0;
	totalVal /= numSamples;
	double stddev = sqrt(totalVal);
	return stddev;
}

double Histogram::SNR()
{
	double noise = StdDev();
	if (noise > 0.0)
		return Mean() / noise;
	else
		return 0.0;
}

void Histogram::Dump(char *name, int style)
{
	FILE *fp = NULL;
	if (strcmp(name, "stdout") == 0)
		fp = stdout;
	else
		fopen_s(&fp, name, "w");
	if (fp) {
		if (style != 3)
			fprintf(fp, "Range: %.4lf to %.4lf  Mean: %.4lf  Stdev: %.4lf\n", minVal, maxVal, mean/count, StdDev());
		int i;
		int mode = ModeBin();
		double mult = 1.0/bin[mode];
		for(i=0;i<numBins;i++) {
			if (style == 0)
				fprintf(fp, "%.4lf\n", (double)bin[i]/(double)count);
			else if (style == 1)
				fprintf(fp, "%d\n", bin[i]);
			else if (style == 2)
				fprintf(fp, "%.4lf\n", (double)bin[i]*mult);
			else if (style == 3) {
				int k;
				int numToList = int((double)numBins * bin[i]/ (double)count + 0.5);
				for(k=0;k<numToList;k++) {
					fprintf(fp, "%.4lf ", GetVal(i));
				}
			}
		}
		if (fp != stdout)
			fclose(fp);
	}
}

void Histogram::Dump(FILE *fp)
{
	int i;
	for(i=0;i<numBins;i++)
		fprintf(fp, "%d ", bin[i]);
}

//
// evalSplit - evaluates the error value for the proposed sub-set clustering, lower returned values are better
//             I'm sure the methods used to eval error of a proposed cluster are well documented on the web
//             but I just made them all up here as it was easy enough.
// inputs:
// data:      input values (sorted low to high)
// mult:      list of multipliers for each histogram value
// numData:   number of data items
// split:     array of split point indexes, think of each value as the last index of a set,
//            so {0, 3} would mean items 0-0, 1-3, 4-9 are the 0-based set indices, given a set of 10 items
// numSplits: number of split points in the split array, the number of clusters to evaluate is one more than this value
// method:    which error eval method to use, see comments below on methods
//

double Histogram::evalSplit(int *split, int numSplits, int method, bool includeEnds)
{
	// method 1
	// 1.1 calc mean of each subset
	// 1.2 sum dist*dist from each point in subset to mean
	// 1.3 add sums
	// return value

	// method 2
	// same as method 1 except that the error function (1.2) is defined to be
	// the sum of the abs dist of each point to the mean

	const int maxSplits = 10;
	double mean[maxSplits];
	double sum[maxSplits];
	int subsetCount[maxSplits];
	int i;

	memset(mean, 0, sizeof(mean));
	memset(sum, 0, sizeof(sum));
	memset(subsetCount, 0, sizeof(subsetCount));


	// step 1.1 - calc the mean of each sub-set
	int set = 0;
	int minBin, maxBin;
	if (includeEnds) {
		minBin = 0;
		maxBin = numBins;
	} else {
		minBin = 1;
		maxBin = numBins-1;
	}
	for(i=minBin;i<maxBin;i++) {
		// printf("Set: %d  Val: %.3lf\n", set, data[i]);
		mean[set] += histValues[i]*bin[i];
		subsetCount[set] += bin[i];

		if (i == split[set]) {
			set++;
		}
	}

	for(i=0;i<=numSplits;i++) {
		mean[i] /= subsetCount[i];
	}

	// step 1.2 - calculate the error in each set as the sum of dist squared to mean
	double val;
	set = 0;
	switch (method) {
		case 1:
			for(i=minBin;i<maxBin;i++) {
				val = mean[set] - histValues[i];
				sum[set] += val*val*bin[i];
				if (i == split[set])
					set++;
			}
		break;

		case 2:
			for(i=minBin;i<maxBin;i++) {
				val = mean[set] - histValues[i];
				if (val < 0.0) val = -val;
				sum[set] += val*bin[i];
				if (i == split[set])
					set++;
			}
		break;
	}

	// step 1.3 - sum the errors per set, thats our metric
	val = 0;
	for(i=0;i<=numSplits;i++)
		val += sum[i];

	return val;
}

void Histogram::Cluster()
{
    printf("Looking for best clustering...\n");

	int i, j;
	int split[3];
	split[2] = 0;
	int method = 1; // choices are 1 or 2, 1 is sum of squares, 2 is sum of abs deltas
	double best2ClusterVal = 9999999999999.0;
	double best3ClusterVal = 9999999999999.0;
	double val = 0.0;
	for(i=0;i<numBins-1;i++) {
		// printf("."); fflush(stdout);
		split[0] = i;
		for(j=i+1;j<numBins;j++) {
			split[1] = j;
			val = evalSplit(split, 2, method, true);
			// printf("Split 0-%d, %d-%d, %d-%d val=%.5lf\n",
				// split[0], split[0]+1, split[1], split[1]+1, numData-1, val);

			if (val < best3ClusterVal) {
				best3ClusterVal = val;
				best3cluster[0] = split[0];
				best3cluster[1] = split[1];
			}
		}
		if (val < best2ClusterVal) {
			best2ClusterVal = val;
			best2cluster[0] = split[0];
		}
	}

	printf("\n");
	printf("Best 2-cluster split is %d-%d, %d-%d with val: %.5lf\n",
		0, best2cluster[0], best2cluster[0]+1, numBins-1, best2ClusterVal);
	printf("Best 3-cluster split is %d-%d, %d-%d, %d-%d with val: %.5lf\n",
		0, best3cluster[0], best3cluster[0]+1, best3cluster[1], best3cluster[1]+1, numBins-1, best3ClusterVal);
}
