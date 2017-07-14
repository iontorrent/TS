/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef PCACOMPRESSION_H
#define PCACOMPRESSION_H

#include "AdvCompr.h"

class PCACompr{
public:
	PCACompr(int _nRvect, int _nFvect, int _npts, int _ntrcs, int _ntrcsL,
			int _t0est, float *_trcs, float *_trcs_coeffs, float *_basis_vectors)
	{
		nRvect=_nRvect;
		nFvect=_nFvect;
		npts=_npts;
		ntrcs=_ntrcs;
		ntrcsL=_ntrcsL;
		t0est=_t0est;
		trcs=_trcs;
		trcs_coeffs=_trcs_coeffs;
		basis_vectors=_basis_vectors;
	}
	int Compress(void);


private:
	float SubtractVector(int nvect, int skip);
	int   AccumulateDotProducts(float *p, float *t, int skip);
	void  ComputeOrderVect(int nvect, int order);
	void  EnsureOrthogonal(int nvect, int check);
	int   ComputeNextVector(int nvect, int skip);
	void  ComputeEmphasisVector(float *gv, float mult, float adder, float width);
	void  smoothNextVector(int nvect, int blur);


	int nRvect;
	int nFvect;
	int npts;
	int ntrcs;
	int ntrcsL;
	int t0est;
	float *trcs;             // raw image block loaded into floats, mean subtracted and zeroed
	float *trcs_coeffs;  // pca/spline coefficients per trace
	float *basis_vectors;    // pca/spline vectors
};


#endif // PCACOMPRESSION_H
