/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <stdio.h>
#include <inttypes.h>

class Histogram {
	public:
		Histogram();
		Histogram(int numBins, double minVal, double maxVal);
		virtual ~Histogram();

		void ChangeParams(int _numBins, double _minVal, double _maxVal);
		void Add(double val, bool ignoreEnds = false);
		void Dump(char *name, int style = 0);
		void Dump(FILE *fp);
		double	Mean();
		int	ModeBin();
		double	StdDev();
		double SNR();
		int		Count() {return count;}
		void	Range(double min, double max) {minVal = min; maxVal = max;}
		int		GetBin(double val);
		int		GetCount(int binIndex);
		double	evalSplit(int *split, int numSplits, int method, bool includeEnds);
		void	Cluster();
		int		GetSplit() {return best3cluster[1];}
		double	GetVal(int binIndex) {return histValues[binIndex];}
		int best2cluster[1];
		int best3cluster[2];
	protected:
		void Init(int numBins, double minVal, double maxVal);
		int numBins;
		double minVal;
		double maxVal;
		int	*bin;
		double *histValues; // value each bin represents
		int count;
		double mean;
};

#endif // HISTOGRAM_H
