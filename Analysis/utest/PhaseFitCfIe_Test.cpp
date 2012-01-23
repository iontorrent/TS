/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include "PhaseFitCfIe.h"
 
/* Class with a SetUp() function for test fixtures, could also have a TearDown() */
class PhaseFitTest : public ::testing::Test {
protected :
 
  virtual void SetUp() {
    flowCycle = "TACG";

    // Set up pure nuc concentrations
    concentration.resize(N_NUCLEOTIDES);
    for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++) {
      concentration[iNuc].assign(N_NUCLEOTIDES,0);
      concentration[iNuc][iNuc] = 1;
    }

    cf.assign(1,0.012);
    ie.assign(1,0.008);
    dr.assign(1,0.005);

    nFlow = 30;
    droopType = ONLY_WHEN_INCORPORATING;
    maxAdv = 2;

    seq.push_back("CTTGTAACAGGTCAGTTACCGTCCGTCCACGCCGCCGCGATGCTGTTATTGGTCGCTTTTATCGTCATCAGTCTATTACTCAAAAAGGCCTTTTGCTCAT");
    seq.push_back("GCAATCAACTGGCGAAACTGGAACCGATTGTTTCGGTATTGCAGAGCGACCCGGAACAGTTCGAACAGTTAAAAGAAGATTACGCGTACTCTCAGCAGAT");
    seq.push_back("GCAAATAAAACATTATTTTCTCGATCTTCACTATAAATAGCGCCAGGTATTATATCCCATCCTCCGGGGCTAAGCTGTGTTCCAGCATGGATATTGTGTG");
    seq.push_back("GCGCTTACTGTTACAGAGGTTATCAAGAATGATGACATCATGACCGTTTTGCAGTAATTGCACACAGGTATGACTTCCAATGTAACCGCTACCACCGGTA");
    seq.push_back("TTCCGGGGAGAGGTCGACCGCGATGAGAATGTGTTTATAAGCCATAGTGTTACTCCTTCCATAAAGTTGTCGATGACTGGCCAGCTAGCGTTTCTTGTGC");
    seq.push_back("AAAGTGTTATTTTTATCATCGAAATCAGAATGCTTTTATGCTGGCAGAGCGATACAAGCTGGTGCTAACGGTTTTGTCAGTAAATGCAATGATCAGAATG");
    seq.push_back("GCACCGAGCGTTCTTTGTCATCCACCACCAGCCATTCAAAGCCGTACGGGTCAAAATCCAGTTCATGCATTGCTTTATGGTGGCGGTAGGTGAGGTTCAG");
    seq.push_back("CATATGTCGTCTCCGTTACACCTTTTCCACATTCACAAGGAAGGACTTAAACTCCGGCGTCTGCGTGTTCGCATCACCAACGAATGGCGTTAAAGTATTG");
    seq.push_back("GATGCAGGATCTGGAGGAAGAAGGTTATCTGGTTGGCCTGGAGAAAGCGAAGTTCGTCGAGCGGCTGGCGCATTACTATTGTGAAATCAACGTGCTGCAT");
    seq.push_back("GGGCCGAGCGTCGAGTTAGAGGAAGAGTCAACGCTGGAGACTTCCTGCTCTATTTCGAGCAACACCGCGTCGCCTTCATTGACCTGCGGAGTAACTTTGA");

    // Simulate some data to fit
    unsigned int nRead = seq.size();
    signal.resize(nRead);
    PhaseSim pSim;
    for(unsigned int iRead=0; iRead < nRead; iRead++) {
      vector<weight_vec_t> hpWeight;
      weight_vec_t droopWeight;
      bool returnIntermediates=false;
      pSim.simulate(
        flowCycle,
        seq[iRead],
        concentration,
        cf,
        ie,
        dr,
        nFlow,
        signal[iRead],
        hpWeight,
        droopWeight,
        returnIntermediates,
        droopType,
        maxAdv
      );
    }
  }
 
  string flowCycle;
  vector<weight_vec_t> concentration;
  weight_vec_t cf;
  weight_vec_t ie;
  weight_vec_t dr;
  unsigned int nFlow;
  DroopType droopType;
  unsigned int maxAdv;
  vector<string> seq;
  vector<weight_vec_t> signal;
};
 

// Confirm that we can recover cf and ie parameters from noiseless simulated data
TEST_F(PhaseFitTest, CfIeLevMar) {
  PhaseFitCfIe pFit;
  pFit.InitializeFlowSpecs(flowCycle,concentration,cf,ie,dr,nFlow,droopType,maxAdv);
  for(unsigned int iRead=0; iRead < seq.size(); iRead++)
    pFit.AddRead(seq[iRead],signal[iRead]);
  CfIeParam param_in;
  param_in.cf = (float) 0.01;
  param_in.ie = (float) 0.01;
  pFit.SetParam(param_in);
  int maxIter = 30;
  pFit.LevMarFit(maxIter);
  CfIeParam param_out = pFit.GetParam();
  ASSERT_NEAR(param_out.cf,cf[0],1e-6);
  ASSERT_NEAR(param_out.ie,ie[0],1e-6);
  // cout << "Signal " << i << "\t" << setiosflags(ios::fixed) << setprecision(10) << signal[i] << "\n";

}
