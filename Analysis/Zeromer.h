/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ZEROMER_H
#define ZEROMER_H

#include <complex.h>
#include "fftw3.h"

#include "Image.h"
#include "Region.h"
#include "Mask.h"
#include "Utils.h"
class Zeromer {
	public:
		Zeromer (int _numRegions, int _numKeyFlows, int _numFrames);
        virtual 		~Zeromer ();
		double 			*GetVirtualZeromer (Image *img, Region *region, Mask *mask, int r, int flow);
		double 			*GetAverageZeromer (int region, int flow);
		void 			CalcAverageTrace (Image *img, Region *region, int r, Mask *mask, MaskType these, int flow);
		void 			InitHVector (Image *img, Region *region, int r, Mask *mask, MaskType these, int flow);
		fftw_complex	*HVector (double *avgBead, double *avgEmpty, int cnt);
		void			SetDir (char *_experimentDir);
	protected:
	private:
        int numRegions;
        int numKeyFlows;
        int numFrames;
        double ***avgTrace;
        fftw_complex ***H_vectors;
		char experimentName[MAX_PATH_LENGTH];
};

#endif // ZEROMER_H
