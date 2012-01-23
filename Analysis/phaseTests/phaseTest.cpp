/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PhaseSim.h"

int main()
{
  try {
    string flowCycle = "TACG";
    string seq = "TCAGGATTTGGCCCAAAGGGAACTA";
    weight_t cfVal = 0.012;
    weight_t ieVal = 0.008;
    weight_t drVal = 0.0005;
    DroopType droopType = ONLY_WHEN_INCORPORATING;
    //DroopType droopType = EVERY_FLOW;

    vector<weight_vec_t> concentration;
    // Set up pure nuc concentrations
    concentration.resize(flowCycle.size());
    for(unsigned int iFlow=0; iFlow < flowCycle.size(); iFlow++)
      concentration[iFlow].assign(N_NUCLEOTIDES,0);
    concentration[0][3] = 1;
    concentration[1][0] = 1;
    concentration[2][1] = 1;
    concentration[3][2] = 1;
    // Add some contamination
    concentration[0][1] = 0.02;
    concentration[1][2] = 0.02;
    concentration[2][3] = 0.02;
    concentration[3][0] = 0.02;

    weight_vec_t cf(N_NUCLEOTIDES,cfVal);
    weight_vec_t ie(N_NUCLEOTIDES,ieVal);
    weight_vec_t dr(N_NUCLEOTIDES,drVal);

    //PhaseSim p;
    //p.setDroopType(droopType);
    //p.setFlowCycle(flowCycle);
    //p.setSeq(seq);
    //p.setAdvancerContexts(2);
    //advancer_t extendAdvancerFirst, droopAdvancerFirst;
    //p.setAdvancerWeights(concentration, cf, ie, dr, extendAdvancerFirst, droopAdvancerFirst, true );
    //if(!p.advancerWeightOK(extendAdvancerFirst, droopAdvancerFirst))
    //  cerr << "Advancer weights in first cycle do not all sum to 1.\n";
    //advancer_t extendAdvancer, droopAdvancer;
    //p.setAdvancerWeights(concentration, cf, ie, dr, extendAdvancer,      droopAdvancer,      false);
    ////p.printAdvancerWeights(extendAdvancerFirst,droopAdvancerFirst);
    //if(!p.advancerWeightOK(extendAdvancer, droopAdvancer))
    //  cerr << "Advancer weights after first cycle do not all sum to 1.\n";
    ////p.printAdvancerWeights(extendAdvancer,droopAdvancer);
    //p.setSeqWeights();
    //for(unsigned int iFlow=0; iFlow < 4; iFlow++) {
    //  weight_t sig = p.applyFlow(iFlow,extendAdvancerFirst,droopAdvancerFirst);
    //  cout << "Flow" << iFlow << "\t" << setiosflags(ios::fixed) << setprecision(10) << sig << "\n";
    //  if(!p.hpWeightOK())
    //    cerr << "HP weights after flow " << iFlow << " do not all sum to 1.\n";
    //}

    weight_vec_t signal;
    vector<weight_vec_t> hpWeight;
    vector<weight_vec_t> droopWeight;
    bool returnIntermediates=true;
    unsigned int nFlow = 10;
    unsigned int maxAdvances = 2;
    PhaseSim p;
    p.simulate(
      flowCycle,
      seq,
      concentration,
      cf,
      ie,
      dr,
      nFlow,
      signal,
      hpWeight,
      droopWeight,
      returnIntermediates,
      droopType,
      maxAdvances
    );
    for(unsigned int i=0; i<nFlow; i++) {
      cout << "Flow" << i << "\t" << setiosflags(ios::fixed) << setprecision(10) << signal[i] << "\n";
      for(unsigned int j=0; j<hpWeight[i].size(); j++)
        cout << " " << setiosflags(ios::fixed) << setprecision(10) << hpWeight[i][j];
      cout << "\n";
      for(unsigned int j=0; j<droopWeight[i].size(); j++)
        cout << " " << setiosflags(ios::fixed) << setprecision(10) << droopWeight[i][j];
      cout << "\n";
    }
  } catch(char const *s) {
    cerr << "Caught exception:\n" << s << "\n";
  }
}
