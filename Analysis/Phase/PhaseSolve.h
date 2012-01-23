/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PHASESOLVE_H
#define PHASESOLVE_H

#include <stdint.h>
#include <float.h>
#include "PhaseSim.h"

using namespace std;

#define MAX_HOMOPOLYMER_LENGTH 12
#define MIN_NORMALIZED_SIGNAL_VALUE 0.01

class PhaseSolve : public PhaseSim
{
  public:
    PhaseSolve();
    unsigned int GreedyBaseCall(
      weight_vec_t &          _signal,
      unsigned int            nIterations,
      bool                    debug
    );
    unsigned int GreedyBaseCall (
      weight_vec_t & _signal,
      unsigned int   nIterations,
      advancer_t   & thisExtendAdvancerFirst,
      advancer_t   & thisDroopAdvancerFirst,
      advancer_t   & thisExtendAdvancer,
      advancer_t   & thisDroopAdvancer,
      bool           debug
    );
    weight_vec_t & GetSignal(void) { return(signal); };
    weight_vec_t & GetPredictedSignal(void) { return(predictedSignal); };
    weight_vec_t & GetResidualSignal(void)  { return(residualSignal ); };
    weight_vec_t & GetSecondResidualSignal(void)  { return(secondResidualSignal ); };
    hpLen_vec_t &  GetPredictedHpFlow(void) { return(hpFlow         ); };
    weight_vec_t & GetMultiplier(void) { return(multiplier); };
    void GetCorrectedSignal(weight_vec_t & correctedSignal);
    void GetBaseFlowIndex(std::vector<uint16_t> &baseFlowIndex);
    void SetResidualScale(bool b) { residualScale = b; };
    void SetResidualScaleMinFlow(unsigned int m) { residualScaleMinFlow = m; };
    void SetResidualScaleMaxFlow(unsigned int m) { residualScaleMaxFlow = m; };
    void SetResidualScaleHpWeight(weight_vec_t w) { residualScaleHpWeight = w; };
    weight_t rescale(void);

  protected:
    weight_vec_t     signal;
    weight_vec_t     predictedSignal;
    weight_vec_t     residualSignal;
    weight_vec_t     secondResidualSignal;
    weight_vec_t     correctedSignal;
    hpLen_vec_t      hpFlow;

    // variables related to signal rescaling
    bool             residualScale;
    unsigned int     residualScaleMinFlow;
    unsigned int     residualScaleMaxFlow;
    weight_t         residualScaleMinSignal;
    weight_vec_t     residualScaleHpWeight;
    weight_vec_t     multiplier;

  private:
    void updateSeq(
      unsigned int iHP,
      hpLen_t newHpLen,
      nuc_t newNuc,
      hpLen_t oldHpLen,
      hpLen_t maxAdvances
    );
    void redoFlow(
      unsigned int             iHP,
      unsigned int             iFlow,
      unsigned int             editFlow,
      hpLen_t                  testHpLen,
      vector <readLen_vec_t> & vPrevPosWeight,
      vector <weight_vec_t>  & vPrevHpWeight,
      weight_vec_t           & vPrevDroopedWeight,
      weight_vec_t           & vPrevIgnoredWeight,
      vector <hpLen_vec_t>   & vPrevHpLen,
      vector <nuc_vec_t>     & vPrevHpNuc,
      advancer_t             & thisExtendAdvancerFirst,
      advancer_t             & thisDroopAdvancerFirst,
      advancer_t             & thisExtendAdvancer,
      advancer_t             & thisDroopAdvancer,
      vector <bool>          & edit,
      vector <unsigned int>  & editHpIndex,
      vector <hpLen_t>       & editHpLen,
      vector <unsigned int>  & editNextHpIndex,
      hpLen_t                  maxAdvances,
      bool                     storeSignal,
      bool                     debug,
      weight_vec_t           & newHpWeight,
      readLen_vec_t          & newPosWeight
    );
};

#endif // PHASESOLVE_H
