/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SEPARATOR_H
#define SEPARATOR_H

#include <vector>
#include <fstream>
#include "Image.h"
#include "Region.h"
#include "Mask.h"
#include "Utils.h"
#include "DualGaussMixModel.h"
#include "AvgKeyIncorporation.h"
#include "SpecialDataTypes.h"

enum SeparatorSignalType {
	SeparatorSignal_None = 0,
	SeparatorSignal_Integral = 1,
};

class SepModel {
 public:
  int row;
  int col;
  MixModel model;
};

class Separator : public AvgKeyIncorporation {
	public:
		Separator(bool _bfTest = false);
		virtual ~Separator();

		void	SetSize(int w, int h, int numKeyFlows, int numGroups = 0);
		void	SetFlowOrder(char *flowOrder);

		// assumes that CalcBeadfindMetric_1 was already called in the Image, as its much faster that way, could have had this
		// called on a per-region basis to make this clearer to read, but the code is designed for whole-chip analysis anyway,
		// so it made more sense to do this in one fast pass earlier.
		// designed to be called from threads, so each object can exist on its own,
		// writes to the section of the mask specified by the region.
		void	FindBeads(Image *image, Region *region, Mask *mask, char *prepend);
		void	CalcSignal(Image *image, Region *region, Mask *mask, MaskType these, int flow, SeparatorSignalType signalType);
		void	SetIntegralBounds(int _start, int _uncomp_start, int _end)
        {
            start = _start;
            uncomp_start = _uncomp_start;
            end = _end;
            
            // it seems like this should also move, but I'm not sure I completely understand it.....TMR
            //timeStart = _start;
            //timeEnd = _end;
        }
		void	SetMinPeak(int _threshold) {minPeakThreshold = _threshold;}
		void	Categorize(SequenceItem *seqList, int numSeqListItems, Mask *mask);
		double	*GetWork(){return work;}
		void	SetDir (char *_experimentDir);
		double	GetAvgTFSig () {return (avgTFSig);}
		double	GetAvgLibSig () {return (avgLibSig);}
		int	GetGroup(int x, int y) {if (groups && numGroups > 1) return groups[x+y*w]; else return 0;}
		int	NumGroups() {return numGroups;}
		float  *GetAvgKeySig (int region_num, int rStart, int rEnd, int cStart, int cEnd) {if (region_num < num_regions) return (&avgRegionKeySignal[region_num*(end-start)]); else return NULL;}
        double  GetAvgKeySigLen () { return(end-start); }
        void    SetRegions (int _num_regions,Region *_region_list) { num_regions = _num_regions; region_list = _region_list; }
        int     GetStart(void) {return uncomp_start;}
        int     GetStart(int region_num, int rStart, int rEnd, int cStart, int cEnd) {return GetStart();}
        int     GetEnd(void) {return end;}
		void	SetFracEmpty (float _multiplier){emptyFraction = _multiplier;}
		const std::vector<SepModel> & GetRegionModels() { return mModels; }
	protected:
		double	GetRQS(double *signal, int *zeromers, int *onemers);
		double	GetRQS(double *signal, int *zeromers, int *onemers, double *signalSum);
		double	GetCat(double *signal, int *zeromers, int *onemers);
		void	CleanupBeads ();
		int GetRegionBin(int index, int nCol, double theta, const std::vector<double> &breaks);
		void AddGlobalSample(int nRow, int nCol, int size, Region *region, 
				     const std::vector<double> &breaks, 
				     double theta,
				     std::vector<double> &sampleData,
				     const double *results);
		double GetDistOnDiag(int index, int nCol, double theta);
		void FindCoords(double dist, double theta, int &row, int &col);
		void GetChipSample(Image *image, Mask *mask, double fraction);
		bool isInRegion(int index, int nCol, Region *region);
		
	private:
		int w, h;
		int numKeyFlows;
        int timeStart;
        int timeEnd;
		double	*work;
		int16_t *bead;
		uint64_t beadIdx1_Len;
		uint64_t beadIdx2_Len;
        float *avgRegionKeySignal;
		int start, uncomp_start, end;
		char *flowOrder;
		int numFlowsPerCycle;
		int frameStride;
		int minPeakThreshold;
		char experimentDir[MAX_PATH_LENGTH];
		double avgTFSig;
		double avgLibSig;
		int numGroups;
		unsigned char *groups; // index for the group this well belongs to
		bool regenerateGroups;
        Region *region_list;
        int num_regions;
		bool bfTest;
		double percentForSample;
//		double regionMult;
		float emptyFraction;


		/* Distance from the lower (0,0) left hand of chip (outlet on beta machine) of
		   the areas for global sampling. */
		std::vector<double> chipSampleRegions;
		/*
		  Sampled data from wells cut into areas perpendicular to 
		  the y = nRow/nCol * x diagonal of the chip. Approach is to sample
		  from the rest of the chip for data points in addition to the usual local 
		  region for beadfind clustering. Sampling is done from each area weighted
		  by the inverse distance to the current local region. Sampling is done
		  in this manner as there was concern that the flow would cause differences
		  based on distance from the inlet so we want to sample more from areas that 
		  are similar distance from the inlet.
		*/
		std::vector< std::vector<int> > chipSampleIdx;
        std::ofstream statsOut;
		int rCnt;
		int beadFindIx;
	    pthread_mutex_t lock;
	    std::vector<SepModel> mModels;
};

#endif // SEPARATOR_H

