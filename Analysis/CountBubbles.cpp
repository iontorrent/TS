/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include "BubbleFilter.h"
#include "OptArgs.h"
#include "Utils.h"

int main(int argc, const char *argv[]) {
  OptArgs opts;  
	string resultsDir;
	string maskFile;
	string outPrefix;
	string flowOrder;
	int startFlow;
	bool doWells;
	int numFlows;
	double iqrThresh;
	double flowStep;
	double wellStep;
	opts.ParseCmdLine(argc, argv);	
	opts.GetOption(resultsDir, "", '-', "results-dir");
  opts.GetOption(maskFile, "", '-', "mask-file");
	opts.GetOption(outPrefix, "", '-', "out-prefix");
	opts.GetOption(doWells, "false", '-', "save-wells");
	opts.GetOption(numFlows, "100", '-', "num-flows");
	opts.GetOption(startFlow, "0", '-', "start-flow");
	opts.GetOption(flowOrder, "", '-', "flow-order");
	opts.GetOption(iqrThresh, "5", '-', "iqr-thresh");
	opts.GetOption(flowStep, "1", '-', "flow-step");
	opts.GetOption(wellStep, "1", '-', "well-step");
	string resultsRoot = resultsDir + "/acq_";
	string resultsSuffix = ".dat";	
	char buff[resultsRoot.size() + resultsSuffix.size() + 20];	
	Mask mask;
	if (maskFile.empty()) {
		snprintf(buff, sizeof(buff), "%s%.4d%s", resultsRoot.c_str(), (int)0, resultsSuffix.c_str());
		Image img;
		img.LoadRaw(buff);
		const RawImage *raw = img.GetImage();
		int cols = raw->cols;
		int rows = raw->rows;
		mask.Init(cols,rows,MaskEmpty);
	}
	else {
		mask.SetMask(maskFile.c_str());
	}
	BubbleFilter filter(numFlows, mask.H() * mask.W(), iqrThresh);
	if (doWells) {
		string fullPath = outPrefix + ".sd-bubble.wells";
		string dir, file;
		FillInDirName(fullPath, dir, file);
		filter.SetSdWellsFile(dir, file, flowOrder, mask.H(), mask.W());
	}

	filter.SetMeshOut(outPrefix + ".image-mesh.txt");
	for (int flowIx = startFlow; flowIx < startFlow + numFlows; flowIx+=flowStep) {
		GridMesh<std::pair<float,int> > badPercent;
		badPercent.Init(mask.H(), mask.W(), max(1.0,ceil(mask.H()/150)), max(1.0,ceil(mask.W()/150)));
		snprintf(buff, sizeof(buff), "%s%.4d%s", resultsRoot.c_str(), (int)flowIx, resultsSuffix.c_str());
		Image img;
		img.LoadRaw(buff);
		filter.FilterBubbleWells(&img, flowIx, &mask, &badPercent);
	}
	
	ofstream statsOut;
	string outFile = outPrefix + ".bubble-counts.txt";
	statsOut.open(outFile.c_str());
	filter.WriteResults(statsOut);
	statsOut.close();

	ofstream cumulativeOut;
	string cumulativeFile = outPrefix + ".bubble-cumulative.txt";
	cumulativeOut.open(cumulativeFile.c_str());
	filter.WriteCumulative(cumulativeOut);
	cumulativeOut.close();


	ofstream summaryOut;
	string summaryFile = outPrefix + ".bubble-summary.txt";
	summaryOut.open(summaryFile.c_str());
	int covered, available;
	filter.GetSdCovered(1, covered, available);
	summaryOut << "TotalBubble=" << covered << endl;
	summaryOut << "PercentBubble=" << (double)covered/available << endl;
	summaryOut.close();
	cout << covered << " filt bubble covered wells." << endl;

	return 0;
}
