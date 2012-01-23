/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PhaseSolve.h"
#include "CafieSolver.h"
#include "OptArgs.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

string randomSeq(unsigned int len);
void usage(void);
void sim(string &simSeq, string &keySeq, string &adapterSeq, weight_vec_t &signal, const unsigned int nBases, const unsigned int nFlows, const string flowString, const vector <weight_vec_t> &concentration, const weight_vec_t &cf, const weight_vec_t &ie, const weight_vec_t &dr, const weight_vec_t &hpScale, const unsigned int maxPopulationsSim, const unsigned int maxAdv);
double rnorm(double mu, double sigma);
void addNoise(weight_vec_t &signal,double snr);

int main(int argc, const char *argv[])
{
  unsigned int nReads;
  double seqLenMean;
  double seqLenSD;
  double seqLenMin;
  double seqLenMax;
  unsigned int nFlows;
  string flowString;
  string keySeq;
  string adapterSeq;
  double cfVal;
  double ieVal;
  double drVal;
  double snr;
  string outFile = "";
  bool help;

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  opts.GetOption(nReads,                "10",                              '-', "reads");
  opts.GetOption(seqLenMean,            "100",                             '-', "len-mean");
  opts.GetOption(seqLenSD,              "25",                              '-', "len-sd");
  opts.GetOption(seqLenMin,             "0",                               '-', "len-min");
  opts.GetOption(seqLenMax,             "200",                             '-', "len-max");
  opts.GetOption(nFlows,                "240",                             '-', "flows");
  opts.GetOption(flowString,            "TACG",                            '-', "flow-order");
  opts.GetOption(keySeq,                "TCAG",                            '-', "key-seq");
  opts.GetOption(adapterSeq,            "CTGAGACTGCCAAGGCACACAGGGGATAGG",  '-', "adapter-seq");
  opts.GetOption(cfVal,                 "0",                               '-', "cf");
  opts.GetOption(ieVal,                 "0",                               '-', "ie");
  opts.GetOption(drVal,                 "0",                               '-', "dr");
  opts.GetOption(snr,                   "10",                              '-', "snr");
  opts.GetOption(outFile,               "",                                '-', "out");
  opts.GetOption(help,                  "false",                           'o', "help");

  opts.CheckNoLeftovers();

  if(help) {
    usage();
    return(0);
  }
  if(outFile == "") {
    cerr << "Must specify file for output with --out option" << endl;
    usage();
    return(1);
  }

  if(nFlows == 0)
    nFlows = 2*seqLenMean;

  ofstream outFlows;
  if(outFile != "") {
    outFlows.open(outFile.c_str());
    if(outFlows.fail()) {
      cerr << "Unable to open file " << outFile << " for write\n";
      exit(EXIT_FAILURE);
    }
  }

  try {
    // Initialize cafie model params
    vector<weight_vec_t> concentration;
    concentration.resize(N_NUCLEOTIDES);
    for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++) {
      concentration[iNuc].assign(N_NUCLEOTIDES,0);
      concentration[iNuc][iNuc] = 1;
    }
    weight_vec_t cf(1,(weight_t) cfVal);
    weight_vec_t ie(1,(weight_t) ieVal);
    weight_vec_t dr(1,(weight_t) drVal);
    weight_vec_t hpScale(1,1.0);
    unsigned int maxPopulations = 100;
    unsigned int maxAdv = 2;

    outFlows << "simSeq\tadapterStart\tflowPos";
    for(unsigned int iFlow=0; iFlow<nFlows; iFlow++)
      outFlows << "\tflow" << iFlow;
    outFlows << endl;

    srand(0);
    for(unsigned int iRead=0; iRead < nReads; iRead++) {
     
      weight_vec_t signal;
      string simSeq;

      // Simulate sequence and flow data from a file
      double seqLen = rnorm(seqLenMean, seqLenSD);
      seqLen = std::min(seqLen,seqLenMax);
      seqLen = std::max(seqLen,seqLenMin);
      sim(simSeq, keySeq, adapterSeq, signal, (unsigned int) seqLen, nFlows, flowString, concentration, cf, ie, dr, hpScale, maxPopulations, maxAdv);
      if(snr > 0)
        addNoise(signal,snr);

      // Write out seqeunce and flow data
      long basePos = (keySeq.length() + (unsigned int) seqLen);
      long bestFlow =  computeSeqFlow(simSeq, flowString, basePos);

      outFlows << simSeq << "\t" << basePos << "\t" << bestFlow;
      for(unsigned int iFlow=0; iFlow<nFlows; iFlow++)
        outFlows << "\t" << setiosflags(ios::fixed) << setprecision(7) << signal[iFlow];
      outFlows << endl;
    }
  } catch(char const *s) {
    cerr << "Caught exception:\n" << s << "\n";
  }

  if(outFile != "")
    outFlows.close();
}

string randomSeq(unsigned int len) {
  vector<char> bases(0);
  bases.push_back('A');
  bases.push_back('C');
  bases.push_back('G');
  bases.push_back('T');

  string dna;
  while(len-- > 0) {
    unsigned int baseIndex = rand() % 4;
    dna.push_back(bases[baseIndex]);
  }

  return(dna);
}

void sim(string &simSeq, string &keySeq, string &adapterSeq, weight_vec_t &signal, const unsigned int nBases, const unsigned int nFlows, const string flowString, const vector <weight_vec_t> &concentration, const weight_vec_t &cf, const weight_vec_t &ie, const weight_vec_t &dr, const weight_vec_t &hpScale, const unsigned int maxPopulationsSim, const unsigned int maxAdv) {

  simSeq = keySeq;
  simSeq += randomSeq(nBases);
  simSeq += adapterSeq;

  DroopType droopType = ONLY_WHEN_INCORPORATING;
  vector<weight_vec_t> hpWeight;
  vector<weight_vec_t> droopWeight;
  bool returnIntermediates=false;

  PhaseSim pSim;
  pSim.setHpScale(hpScale);
  pSim.setMaxPops(maxPopulationsSim);
  pSim.simulate(
    flowString,
    simSeq,
    concentration,
    cf,
    ie,
    dr,
    nFlows,
    signal,
    hpWeight,
    droopWeight,
    returnIntermediates,
    droopType,
    maxAdv
  );

  return;
}

void usage() {
  cout << "" << endl;
  cout << "flowSim - simulate flow data. " << endl;
  cout << "  Produces a tab-delimited text file of simulated flow data." << endl;
  cout << "  First line is a header, after that one line per simulated read." << endl;
  cout << "  First column is the sequence, second is the 0-indexed adapater start." << endl;
  cout << "  Subsequent columns are the flow values." << endl;
  cout << endl;
  cout << "Usage: flowSim [options] --out out.txt" << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  reads=s        Number of reads to simulate (default 10)" << endl;
  cout << "  len-mean=s     Mean length of simulated reads (default 100)" << endl;
  cout << "  len-sd=s       Standard deviation for simulated read length (default 25)" << endl;
  cout << "  len-min=s      Min length of simulated reads (default 0)" << endl;
  cout << "  len-max=s      Max length of simulated reads (default 200)" << endl;
  cout << "  flows=s        Number of flow to simulate (default 240)" << endl;
  cout << "  flow-order=s   Flow cycle (default TACG)" << endl;
  cout << "  key-seq=s      Sequence prefix for each read (default TCAG)" << endl;
  cout << "  adapter-seq=s  Suffix sequence for each read (default CTGAGACTGCCAAGGCACACAGGGGATAGG)" << endl;
  cout << "  cf=s           Carry-forward value (default 0)" << endl;
  cout << "  ie=s           Incomplete extension value (default 0)" << endl;
  cout << "  dr=s           Droop value (default 0)" << endl;
  cout << "  snr=s          Signal-to-noise ratio (default 10)" << endl;
  cout << "  out=s          Output file" << endl;
  cout << "  help           This message" << endl;
  cout << "" << endl;
}

double rnorm(double mu, double sigma) {
  static bool first=true;
  static const gsl_rng_type * T;
  static gsl_rng * r;
     
  if(first) {
    first = false;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
  }
     
  return(mu + gsl_ran_gaussian(r, sigma));
}

void addNoise(weight_vec_t &signal,double snr) {
  if(snr <= 0)
    return;

  double sigma = 1/snr;

  for(unsigned int iFlow=0; iFlow < signal.size(); iFlow++) {
    signal[iFlow] += rnorm(0,sigma);
    if(signal[iFlow] < 0)
      signal[iFlow] = 0;
  }
}
