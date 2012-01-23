/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "PhaseSim.h"

using namespace std;

/* Class with a SetUp() function for test fixtures, could also have a TearDown() */
class PhaseSimTest : public ::testing::Test {
protected :
 
  virtual void SetUp() {
    droopType   = ONLY_WHEN_INCORPORATING;
    maxAdvances = 2;
    flowCycle   = "TACG";
    nFlow       = 10;
    seq         = "TCAGGATTTGGCCCAAAGGGAACTAGCAT";

    // CAFIE parameters - same per nuc
    cf.assign(N_NUCLEOTIDES,0.012);
    ie.assign(N_NUCLEOTIDES,0.008);
    dr.assign(N_NUCLEOTIDES,0.0005);

    // Set up pure nuc concentrations
    concentration.resize(N_NUCLEOTIDES);
    concentration.resize(flowCycle.size());
    for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++) {
      concentration[iNuc].assign(N_NUCLEOTIDES,0);
      concentration[iNuc][iNuc] = 1;
    }
    // Add some contamination
    concentration[0][2] = 0.02;
    concentration[1][3] = 0.02;
    concentration[2][0] = 0.02;
    concentration[3][1] = 0.02;

    // The expected results
    weight_t s[] = {
      1.0111656036,
      0.0208234747,
      0.9752343291,
      0.1185218282,
      0.0101367377,
      1.0130196461,
      0.0188181752,
      1.8239818051,
      0.1984354465,
      0.9506541947
    };
    unsigned int slen = sizeof(s)/sizeof(weight_t);
    assert(slen == nFlow);
    expectedSignal.resize(nFlow);
    std::copy(&s[0],&s[0]+nFlow,expectedSignal.begin());
  }
 
  DroopType droopType;
  unsigned int maxAdvances;
  string flowCycle;
  string seq;
  weight_vec_t cf;
  weight_vec_t ie;
  weight_vec_t dr;
  vector<weight_vec_t> concentration;
  unsigned int nFlow;
  weight_vec_t expectedSignal;
};
 
/* Test using test fixture PhaseSimTest with SetUp() function called by framework. */
TEST_F(PhaseSimTest, Simulate) {

  PhaseSim p;
  weight_vec_t signal;
  vector<weight_vec_t> hpWeight;
  weight_vec_t droopWeight;
  bool returnIntermediates=true;
  p.simulate(flowCycle, seq, concentration, cf, ie, dr, nFlow, signal, hpWeight, droopWeight, returnIntermediates, droopType, maxAdvances);
  for(unsigned int i=0; i<nFlow; i++) {
    // Confirm signal is what we expect
    ASSERT_NEAR(signal[i],expectedSignal[i],1e-6);
    // Confirm weight accounting - should sum to 1 for every flow.
    ASSERT_NEAR(p.getWeightSum(),1,1e-6);
    // Uncomment this if you want to see and update the reference signal
    // cout << "Signal " << i << "\t" << setiosflags(ios::fixed) << setprecision(10) << signal[i] << "\n";
  }

}
