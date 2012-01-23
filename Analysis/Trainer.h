/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRAINER_H
#define TRAINER_H

#include "Image.h"
#include "Region.h"
#include "Mask.h"
#include "Separator.h"

class Trainer {
	public:
		Trainer(int maxFrames, int numRegions, int numKeyFlows, char *_keySequence, int _numGroups = 0);
		virtual ~Trainer();

		void AddTraces(Image *img, Region *region, int r, Mask *mask, MaskType these, Separator *sep, int trainingVal, const double *trace);
		void AddTraces(Image *img, Region *region, int r, Mask *mask, MaskType these, Separator *sep, int trainingVal);
		void TrainAll();
		const double *GetVector(int region);
		void SetDir (char *directory);
		double *FluxIntegrator (Image *img, int r, int x, int y, const double *zeroTrace, MaskType these);
		void ApplyVector(Image *img, int r, Region *region, Mask *mask, MaskType these, const double *zeroTrace);
		const double *GetResults () {return (results);}
		void SetFlux(bool _use_flux_integration = true) {use_flux_integration = _use_flux_integration;}
		// WEKA specific
		int list_to_arff (char *fileName, int r);
		int runWeka (char *, int r);
		void runWeka ();
		void SetNNData (int flag) {using_NN_data = flag;}

	protected:
		double	**vec;
		double	***rawTraces;

	private:
		Trainer(); // don't call, not implemented
		void extractWeights (char *buffer, int r);
		void CleanupRawTraces ();
		void CleanupFluxTraces ();
		
		int	numRegions;
		int	maxFrames;
		int numKeyFlows;
		int *maxTraces;
		int imageCnt;
		int *wellCnt;
		int numTraces;
		char *experimentName;
		char *keySequence;
		double *results;
		bool	use_flux_integration;
		int maxTracesPerFlow; // defaults to 1000 per flow per region, clustering to sub-population train would reduce this value
		int numGroups; // number of sub-population groups per region
		int	using_NN_data;
		int lastFrame;
};

#endif // TRAINER_H

