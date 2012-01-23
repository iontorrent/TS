/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BUBBLEFILTER_H
#define BUBBLEFILTER_H

#include <vector>
#include <string>
#include <fstream>
#include "SampleQuantiles.h"
#include "Mask.h"
#include "Image.h"
#include "GridMesh.h"
#include "RawWells.h"

class BubbleFilter {

 public:

	BubbleFilter(int numFlows, int numWells, float iqrThresh=5);

	~BubbleFilter();

	void SetMeshOut(const std::string &meshOut) {
		mMeshOut.open(meshOut.c_str());
	}

	void SetSdWellsFile(const std::string &experimentName, const std::string &wellsName,
											const std::string &flowOrder, int rows, int cols);

	void SetSdWellsFile(const std::string &experimentName, const std::string &wellsName);
	
	void FilterBubbleWells(Image *img, int flowIx, Mask *mask, GridMesh<std::pair<float,int> > *mesh);

	void WriteResults(std::ostream &out);

	void GetSdCovered(int threshold, int &underBubble, int &totalAvailable) {
		int count = 0;
		int available = 0;
		for (size_t i = 0; i < mSdWellBubbleCount.size(); i++) {
			if (mSdWellBubbleCount[i] >= 0) {
				available++;
			}
			if (mSdWellBubbleCount[i] >= threshold) {
				count++;
			}
		}
		underBubble = count;
		totalAvailable = available;
	}

	void WriteCumulative(std::ofstream &out) {
		out << "well\tflowsCovered" << endl;
		for (size_t i = 0; i < mSdWellBubbleCount.size(); i++) {
			out << i << "\t" << mSdWellBubbleCount[i] << endl;
		}
	}

 private:
	std::vector<int> mMeanBubbleCount;
	std::vector<int> mSdBubbleCount;

	std::vector<int> mSdWellBubbleCount;
	float mIqrThresh;
	std::vector<SampleQuantiles<float> > mMeanStats;
	std::vector<SampleQuantiles<float> > mSdStats;
	std::ofstream mMeshOut;
	RawWells *mBubbleSdWells;
};

#endif // BUBBLEFILTER_H
