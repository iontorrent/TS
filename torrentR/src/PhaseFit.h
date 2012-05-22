/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PHASEFIT_H
#define PHASEFIT_H

#include "PhaseSim.h"
#include "LevMarFitter.h"

using namespace std;
using namespace arma;

enum ResidualType    { SQUARED, ABSOLUTE, GEMAN_MCCLURE };             // The residual transformation
enum ResidualSummary { MEAN, MEDIAN, MEAN_OF_MEDIAN, MEDIAN_OF_MEAN }; // How should residuals be summarize

class PhaseFit : public LevMarFitter
{
  public:
    PhaseFit();
    void InitializeFlowSpecs(
      string                  _flowString,
      vector<weight_vec_t>    _concentration,
      weight_vec_t            _cf,
      weight_vec_t            _ie,
      weight_vec_t            _dr,
      unsigned int            _nFlow,
      DroopType               _droopType,
      unsigned int            _maxAdv
    );
    void AddRead(string seq, weight_vec_t &signal);
    void AddRead(weight_vec_t &seqFlow, weight_vec_t &sig, weight_t maxErr=1);
    int LevMarFit(int max_iter, int nParam, float *params);
    vector<weight_vec_t> GetRawResiduals(void) { return(residual_raw_vec); };
    vector<weight_vec_t> GetWeightedResiduals(void) { return(residual_weighted_vec); };
    float GetSummarizedResidual(void) { return(residual); };
    void SetFlowWeight(weight_vec_t &_flowWeight) { flowWeight = _flowWeight; };
    void SetIgnoreHPs(bool _ignoreHPs) { ignoreHPs = _ignoreHPs; };
    void updateResidualWeight(bool ignoreHPs, weight_vec_t &flowWeight, vector<weight_vec_t> &hpWeight);
    void SetResidualType(ResidualType r) { residualType = r; };
    void SetResidualSummary(ResidualSummary r) { residualSummary = r; };
    void SetExtraTaps(unsigned int _extraTaps) { extraTaps = _extraTaps; };
    float CalcResidual(float *refVals, float *testVals, int numVals, double *err_vec = NULL);
    void PrintResidual(float *params);

  protected:
    vector<PhaseSim>        read;
    vector<weight_vec_t>    signal;
    vector<weight_vec_t>    residual_raw_vec;
    vector<weight_vec_t>    residual_weighted_vec;
    weight_vec_t            err;
    string                  flowString;
    nuc_vec_t               flowCycle;
    vector<weight_vec_t>    concentration;
    weight_vec_t            cf;
    weight_vec_t            ie;
    weight_vec_t            dr;
    unsigned int            nFlow;
    DroopType               droopType;
    unsigned int            maxAdv;
    advancer_t              droopAdvancerFirst;
    advancer_t              extendAdvancerFirst;
    advancer_t              droopAdvancer;
    advancer_t              extendAdvancer;
    vector<weight_vec_t>    hpWeight;             // optional vector of vector of multipliers in the range [0,1] to do per-read per-flow weighting
    weight_vec_t            flowWeight;           // optional vector of multipliers in the range [0,1] to weight some flows more than others.
    bool                    ignoreHPs;            // if true all HPs > 1 will be ignored in fit
    unsigned int            extraTaps;            // sets extraTaps in PhaseSim
    ResidualType            residualType;
    ResidualSummary         residualSummary;
};

#endif // PHASEFIT_H
