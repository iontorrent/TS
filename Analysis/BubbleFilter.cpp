/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include "BubbleFilter.h"

BubbleFilter::BubbleFilter(int numFlows, int numWells, float iqrThresh) {
	int sampleSize = 5000;
	mIqrThresh = iqrThresh;
	mMeanBubbleCount.resize(numFlows, 0);
	mSdBubbleCount.resize(numFlows, 0);
	mMeanStats.resize(numFlows);
	mSdStats.resize(numFlows);
	for (size_t i = 0; i < mSdStats.size(); i++) {
		mMeanStats[i].Init(sampleSize);
		mSdStats[i].Init(sampleSize);
	}
	mSdWellBubbleCount.resize(numWells, 0);
	mBubbleSdWells = NULL;
}

BubbleFilter::~BubbleFilter() {

	if (mBubbleSdWells != NULL) {
		mBubbleSdWells->Close();
		delete mBubbleSdWells;
	}
}

void BubbleFilter::SetSdWellsFile(const std::string &experimentName, const std::string &wellsName,
																		const std::string &flowOrder, int rows, int cols) {
	mBubbleSdWells = new RawWells(experimentName.c_str(), wellsName.c_str());
	mBubbleSdWells->CreateEmpty(mSdBubbleCount.size(), flowOrder.c_str(), rows, cols);
	mBubbleSdWells->OpenForWrite();
}

void BubbleFilter::FilterBubbleWells(Image *img, int flowIx, Mask *mask, GridMesh<std::pair<float,int> > *mesh) {
	assert((unsigned int)flowIx < mMeanBubbleCount.size());
	const RawImage *raw = img->GetImage();
	int rows = raw->rows;
	int cols = raw->cols;
	if (rows <= 0 || cols <= 0) {
		cout << "Why bad row/cols for flow: " << flowIx << " rows: " << rows << " cols: " << cols << endl;
		exit(EXIT_FAILURE);
	}

	vector<float> wellMean(raw->rows * raw->cols, 0);
	vector<float> wellSd(raw->rows * raw->cols, 0);
	SampleStats<float> wellStat;
	for (int rowIx = 0; rowIx < raw->rows; rowIx++) {
		for (int colIx = 0; colIx < raw->cols; colIx++) {
			int idx = rowIx * raw->cols + colIx;
			if ((*mask)[idx] & MaskExclude || (*mask)[idx] & MaskPinned) {
				wellSd[idx] = -1;
				wellMean[idx] = -1;
				continue;
			}
			assert((size_t)idx < wellMean.size());
			wellStat.Clear();
			for (int frameIx = 0; frameIx < raw->frames; frameIx++) {
				float val = raw->image[frameIx * raw->frameStride + colIx + rowIx * raw->cols] - raw->image[ colIx + rowIx * raw->cols];
				wellStat.AddValue(val);
			}
			wellMean[idx] = wellStat.GetMean();
			wellSd[idx] = wellStat.GetSD();
			mMeanStats[flowIx].AddValue(wellMean[idx]);
			mSdStats[flowIx].AddValue(wellSd[idx]);
		}
	}
	double meanThreshold = 0;
	double sdThreshold = 0;
	if (mSdStats[flowIx].GetNumSeen() > 0) {
	  sdThreshold = mSdStats[flowIx].GetMedian() - (mIqrThresh * (mSdStats[flowIx].GetQuantile(.5) - mSdStats[flowIx].GetQuantile(.25)));
		sdThreshold = max(20.0, sdThreshold);
	}
	for (size_t binIx = 0; binIx < mesh->GetNumBin(); binIx++) {
		int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
		mesh->GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
		std::pair<float,int> &stats = mesh->GetItem(binIx);
		stats.first = 0.0f;
		stats.second = 0;
		for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
			for (int colIx = colStart; colIx < colEnd; colIx++) {
				int i = rowIx * raw->cols + colIx;
				if ((*mask)[i] & MaskExclude || (*mask)[i] & MaskPinned) {
					continue;
				}
				if (wellMean[i] < meanThreshold || wellSd[i] < sdThreshold) {
					stats.first += 1;
				}
				stats.second++;
				if (wellSd[i] < sdThreshold) {
					mSdBubbleCount[flowIx]++;
					mSdWellBubbleCount[i]++;
				}
				if (mBubbleSdWells != NULL) {
					mBubbleSdWells->WriteFlowgram(flowIx, i  % cols, i / cols, wellSd[i]);
				}
			}
		}
		if (stats.second > 0) {
			stats.first = stats.first / stats.second;
		}
	}
	if (mMeshOut.is_open()) {
		for (size_t bColIx = 0; bColIx < mesh->GetColBin(); bColIx++) {
			mMeshOut << flowIx;
			for (size_t bRowIx = 0; bRowIx < mesh->GetColBin(); bRowIx++) {
				std::pair<float,int> &stats = mesh->GetItem(bRowIx, bColIx);
				mMeshOut << "\t";
				if (stats.second == 0) {
					mMeshOut << "nan";
				}
				else {
					mMeshOut << stats.first;
				}
			}
			mMeshOut << endl;
		}
	}
	cout << "Flow : " << flowIx << " Flagged sd: " << mSdBubbleCount[flowIx] << endl; 
}

void BubbleFilter::WriteResults(std::ostream &out) {
	out << "flow\ttotalFilt\tmeanFilt\tsdFilt\tsdThresh\tsdQ25\tsdQ50\tsdQ75\tmeanQ25\tmeanQ50\tmeanQ75" << endl;
	char s = '\t';
	for (size_t flowIx = 0; flowIx < mMeanBubbleCount.size(); flowIx++) {
		double sdThreshold = 0;
		if (mSdStats[flowIx].GetNumSeen() > 0) {
			sdThreshold = mSdStats[flowIx].GetMedian() - (mIqrThresh * (mSdStats[flowIx].GetQuantile(.5) - mSdStats[flowIx].GetQuantile(.25)));
			sdThreshold = max(20.0, sdThreshold);
		}
		out << flowIx << s 
				<< (mMeanBubbleCount[flowIx] + mSdBubbleCount[flowIx]) << s 
				<< mMeanBubbleCount[flowIx] << s
				<< mSdBubbleCount[flowIx] << s
				<< sdThreshold << s
				<< (mSdStats[flowIx].GetNumSeen() > 0 ? mSdStats[flowIx].GetQuantile(.25) : 0) << s
				<< (mSdStats[flowIx].GetNumSeen() > 0 ? mSdStats[flowIx].GetQuantile(.5) : 0) << s
				<< (mSdStats[flowIx].GetNumSeen() > 0 ? mSdStats[flowIx].GetQuantile(.75) : 0) << s
				<< (mMeanStats[flowIx].GetNumSeen() > 0 ? mMeanStats[flowIx].GetQuantile(.25) : 0) << s
				<< (mMeanStats[flowIx].GetNumSeen() > 0 ? mMeanStats[flowIx].GetQuantile(.5) : 0) << s
				<< (mMeanStats[flowIx].GetNumSeen() > 0 ? mMeanStats[flowIx].GetQuantile(.75) : 0) << s
				<< endl;
	}													 
}
