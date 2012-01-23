/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PhaseFitCfIe.h"

int main()
{
  try {
    string flowCycle = "TACG";

    // Set up pure nuc concentrations
    vector<weight_vec_t> concentration;
    concentration.resize(N_NUCLEOTIDES);
    for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++) {
      concentration[iNuc].assign(N_NUCLEOTIDES,0);
      concentration[iNuc][iNuc] = 1;
    }

    weight_t cfVal = 0.012;
    weight_t ieVal = 0.008;
    weight_t drVal = 0.0005;
    weight_vec_t cf(1,cfVal);
    weight_vec_t ie(1,ieVal);
    weight_vec_t dr(1,drVal);

    unsigned int nFlow = 30;
    DroopType droopType = ONLY_WHEN_INCORPORATING;
    unsigned int maxAdv = 2;

    vector<string> seq;
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
    vector<weight_vec_t> signal(nRead);
    PhaseSim pSim;
    for(unsigned int iRead=0; iRead < nRead; iRead++) {
      vector<weight_vec_t> hpWeight;
      vector<weight_vec_t> droopWeight;
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

    PhaseFitCfIe pFit;
    pFit.InitializeFlowSpecs(flowCycle,concentration,cf,ie,dr,nFlow,droopType,maxAdv);
    for(unsigned int iRead=0; iRead < nRead; iRead++) {
      pFit.AddRead(seq[iRead],signal[iRead]);
    }

    CfIeParam param_in;
    param_in.cf = (float) 0.01;
    param_in.ie = (float) 0.01;
    cout << "Input (cf,ie) =  (" << setprecision(5) << param_in.cf << ", " << param_in.ie << ")\n";
    pFit.SetParam(param_in);
    int maxIter = 30;
    int nIter = pFit.LevMarFit(maxIter);
    CfIeParam param_out = pFit.GetParam();
    cout << "Output (cf,ie) = (" << setprecision(5) << param_out.cf << ", " << param_out.ie << ")\n";
    cout << "  nIter = " << nIter << "\n";
  } catch(char const *s) {
    cerr << "Caught exception:\n" << s << "\n";
  }

}
