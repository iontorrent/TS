/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CFIEDRFIT_H
#define CFIEDRFIT_H

#include "CafieSolver.h"

// MGD - comment out line below to get new Levmar V2 (should be faster but not yet?) more better fitting
#define CFIEDR_FIT_V1

#ifdef CFIEDR_FIT_V1
#include "LevMarFitter.h"
#else
#include "LevMarFitterV2.h"
#endif /* CFIEDR_FIT_V1 */

struct CfiedrParams {
	float	cf;
	float	ie;
	float	dr;
};

#ifdef CFIEDR_FIT_V1
class CfiedrFit : public LevMarFitter
#else
class CfiedrFit : public LevMarFitterV2
#endif /* CFIEDR_FIT_V1 */
{
    // evaluate the fitted function w/ the specified parameters
public:

    // constructor
    // inputs are:
    //   _numMeasuredFlows - number of measured values
    //   _numUsefulFlows - number of flows where we are confident in the predicted cf/ie/dr values
    //                     must be equal to or less than _numMeasuredFlows (typically 10% less)
    //   _cafieSolver - the Cafie Solver class instance to use
    CfiedrFit(int _numMeasuredFlows, int _numUsefulFlows, CafieSolver *_cafieSolver, bool _fixedDroop)
    {
	fixedDroop = _fixedDroop;
	const double *measured = _cafieSolver->GetMeasured();
        yvals = new float[_numUsefulFlows];
        predicted = new float[_numUsefulFlows];
	int i;
	for(i=0;i<_numUsefulFlows;i++) {
		yvals[i] = (float)measured[i];
		predicted[i] = 0;
	}

	int numFitParams = 3;
	if (fixedDroop)
		numFitParams = 2;
        Initialize(numFitParams, _numUsefulFlows, NULL);

	// set up reasonable step size for derivatives
	dp[0] = (0.05 / 100.0); // CF is in the 0 to 5% range, step is 1% of that
	dp[1] = (0.05 / 100.0); // IE is in the 0 to 5% range, step is 1% of that
	if (numFitParams == 3)
		dp[2] = (0.005 / 100.0); // droop is in the 0 to 0.5% range, step is 1% of that

	// generate ideal vector from input flows
	numIdealFlows = _numMeasuredFlows;
	ideal = new double[numIdealFlows];
	for(i=0;i<numIdealFlows;i++) {
		ideal[i] = (int)(measured[i] + 0.5);
	}

	cafieSolver = _cafieSolver;
	ignoreHPs = false;
	ignoreLowQual = false;
    }

    void Init()
    {
	const double *measured = cafieSolver->GetMeasured();
	int i;
	for(i=0;i<len;i++) {
		yvals[i] = (float)measured[i];
	}

	for(i=0;i<numIdealFlows;i++) {
		ideal[i] = (int)(measured[i] + 0.5);
		if (i < len) {
			if (ignoreHPs) {
				if (ideal[i] > 1)
					residualWeight[i] = 0.0f;
				else
					residualWeight[i] = 1.0f;
			} else if (ignoreLowQual) {
				double delta = ideal[i] - measured[i];
				if (delta > 0.2 || delta < -0.2)
					residualWeight[i] = 0.0f;
				else
					residualWeight[i] = 1.0f;
			} else {
				residualWeight[i] = 1.0f;
			}
		}
	}
    }

    void Init(double *_ideal)
    {
	const double *measured = cafieSolver->GetMeasured();
	int i;
	for(i=0;i<len;i++) {
		yvals[i] = (float)measured[i];
	}

	for(i=0;i<numIdealFlows;i++) {
		ideal[i] = (int)(_ideal[i] + 0.5);
		if (i < len) {
			if (ignoreHPs) {
				if (ideal[i] > 1)
					residualWeight[i] = 0.0f;
				else
					residualWeight[i] = 1.0f;
			} else if (ignoreLowQual) {
				double delta = ideal[i] - measured[i];
				if (delta > 0.2 || delta < -0.2)
					residualWeight[i] = 0.0f;
				else
					residualWeight[i] = 1.0f;
			} else {
				residualWeight[i] = 1.0f;
			}
		}
	}
    }

    void MaskFlows(int startFlow, int endFlow) {
        int i;
        for(i=startFlow;i<=endFlow;i++) {
		if (i < len)
                	residualWeight[i] = 0.0;
        }
    }

    /* MaskHPs - for each ideal vector value matching an input n-mer, we ignore that flow */
    /* lets us do stuff like ignore all 0-mers, or all 4, 5, 6, ... mers */
    void MaskHPs(int *hpList, int numInList) {
        int i, hp;
	for(i=0;i<len;i++) {
		for(hp=0;hp<numInList;hp++) {
			if (ideal[i] == hpList[hp])
				residualWeight[i] = 0.0;
		}
	}
    }

    // optionally set maximum value for parameters
    void SetParamMax(CfiedrParams _max_params)
    {
        max_params = _max_params;
#ifdef CFIEDR_FIT_V1
        LevMarFitter::SetParamMax((float *)&max_params);
#else
        LevMarFitterV2::SetParamMax((float *)&max_params);
#endif /* CFIEDR_FIT_V1 */
    }

    // optionally set minimum value for parameters
    void SetParamMin(CfiedrParams _min_params)
    {
        min_params = _min_params;
#ifdef CFIEDR_FIT_V1
        LevMarFitter::SetParamMin((float *)&min_params);
#else
        LevMarFitterV2::SetParamMin((float *)&min_params);
#endif /* CFIEDR_FIT_V1 */
    }

    // SetIgnoreHP - set to true to fit data by looking at 0-mers & 1-mers only
    void SetIgnoreHP(bool ignore) {ignoreHPs = ignore;}

    // SetIgnoreLowQual - set to true to cause residule weights for questionable base calls to be ignored
    void SetIgnoreLowQual(bool ignore) {ignoreLowQual = ignore;}

    // entry point for grid search
    void GridSearch(int steps)
    {
#ifdef CFIEDR_FIT_V1
        LevMarFitter::GridSearch(steps,yvals,(float *)(&params));
#else
        LevMarFitterV2::GridSearch(steps,yvals,(float *)(&params));
#endif /* CFIEDR_FIT_V1 */
    }

    // entry point for fitting
    int Fit(int max_iter)
    {
#ifdef CFIEDR_FIT_V1
        return(LevMarFitter::Fit(max_iter,yvals,(float *)(&params)));
#else
        return(LevMarFitterV2::Fit(max_iter,yvals,(float *)(&params)));
#endif /* CFIEDR_FIT_V1 */
    }

    // entry point for fitting
    int Fit(int max_iter,float *y)
    {
#ifdef CFIEDR_FIT_V1
        return(LevMarFitter::Fit(max_iter,y,(float *)(&params)));
#else
        return(LevMarFitterV2::Fit(max_iter,y,(float *)(&params)));
#endif /* CFIEDR_FIT_V1 */
    }

    // evaluates the function using the values in params
    void Evaluate(float *y) {
        Evaluate(y,(float *)(&params));
    }

    // get the percentage of positive flows
    float GetPercentPositiveFlows(int minFlow, int maxFlow) {

      int nPos=0;
      for(int i=minFlow; i<maxFlow; i++)
        if(ideal[i] > FLT_EPSILON)
          nPos++;

      int nFlow = maxFlow-minFlow;
      return((float)(nPos/(float)nFlow));
    }

    // get the mean squared error after the fit
    float GetResidual(void) {return residual;}

    float *GetPredicted(CfiedrParams *localparams) {
      Evaluate(predicted, (float *)localparams);
      return(predicted);
    }

    float *GetMeasured(void) { return(yvals); }

    // the starting point and end point of the fit
    CfiedrParams params;

    ~CfiedrFit()
    {
        delete [] yvals;
        delete [] predicted;
	delete [] ideal;
    }

protected:
    virtual void Evaluate(float *y, float *localparams) {
        
        CfiedrParams *p = (CfiedrParams *)localparams;
	if (!fixedDroop)
		cafieSolver->Model(ideal, numIdealFlows, /* predicted */ y, /* numPredictedFlows */ len, (double)p->cf, (double)p->ie, (double)p->dr);
	else // droop is fixed, so take from our param list
		cafieSolver->Model(ideal, numIdealFlows, /* predicted */ y, /* numPredictedFlows */ len, (double)p->cf, (double)p->ie, (double)params.dr);
    }

private:
    float *yvals;
    CfiedrParams min_params,max_params;
    CafieSolver *cafieSolver;
    double *ideal;
    int numIdealFlows;
    float *predicted;
    bool ignoreHPs;
    bool ignoreLowQual;
    bool fixedDroop;
};


#endif // CFIEDRFIT_H

