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
bool readNextSequence(ifstream &inFlows, string &simSeq, weight_vec_t &signal, const unsigned int nFlows);
void sim(string &simSeq, string &keySeq, weight_vec_t &signal, const unsigned int nBases, const unsigned int nFlows, const string flowString, const vector <weight_vec_t> &concentration, const weight_vec_t &cf, const weight_vec_t &ie, const weight_vec_t &dr, const weight_vec_t &hpScale, const unsigned int maxPopulationsSim, const unsigned int maxAdv);
void setErrVector(hpLen_vec_t &estimatedFlow, hpLen_vec_t &idealFlow, unsigned int nEstimatedFlows, unsigned int maxBases, vector <uint16_t> &errProfile);
void writeErrSummary(vector< vector <uint16_t> > &errProfile, unsigned int minBases, unsigned int maxBases, unsigned int stepSize, unsigned int nReads, unsigned int nKeyPass);
double rnorm(double mu, double sigma);
void addNoise(weight_vec_t &signal,double snr);

int main(int argc, const char *argv[])
{
  unsigned int nReads;
  unsigned int nBases;
  unsigned int nFlows;
  unsigned int maxIterations;
  unsigned int maxPopulationsSim;
  unsigned int maxPopulationsSolve;
  string solverType;
  string flowString;
  string keySeq;
  double cfVal;
  double ieVal;
  double drVal;
  double snr;
  double weightPrecisionDouble = 1e-6;
  double hpScaleVal = 1;
  string inFlowsFile = "";
  string outFlowsFile = "";
  bool printFlows;
  bool help;

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  opts.GetOption(nReads,                "10",       '-', "reads");
  opts.GetOption(nBases,                "100",      '-', "bases");
  opts.GetOption(nFlows,                "0",        '-', "flows");
  opts.GetOption(maxIterations,         "3",        '-', "iterations");
  opts.GetOption(maxPopulationsSim,     "0",        '-', "populations-sim");
  opts.GetOption(maxPopulationsSolve,   "0",        '-', "populations-solve");
  opts.GetOption(inFlowsFile,           "",         '-', "input-flowvals");
  opts.GetOption(outFlowsFile,          "",         '-', "output-flowvals");
  opts.GetOption(solverType,            "PhaseSim", '-', "solver-type");
  opts.GetOption(flowString,            "TACG",     '-', "flow-order");
  opts.GetOption(keySeq,                "TCAG",     '-', "key-seq");
  opts.GetOption(weightPrecisionDouble, "0.000001", '-', "precision");
  opts.GetOption(cfVal,                 "0.012",    '-', "cf");
  opts.GetOption(ieVal,                 "0.008",    '-', "ie");
  opts.GetOption(drVal,                 "0.0005",   '-', "dr");
  opts.GetOption(snr,                   "0",        '-', "snr");
  opts.GetOption(printFlows,            "false",    '-', "print-flows");
  opts.GetOption(help,                  "false",    '-', "help");

  opts.CheckNoLeftovers();

  if(help)
    usage();

  weight_t weightPrecision = (weight_t) weightPrecisionDouble;

  if(nFlows == 0)
    nFlows = 2*nBases;

  ofstream outFlows;
  if(outFlowsFile != "") {
    outFlows.open(outFlowsFile.c_str());
    if(outFlows.fail()) {
      cerr << "Unable to open file " << outFlowsFile << " for write\n";
      exit(EXIT_FAILURE);
    }
  }

  ifstream inFlows;
  if(inFlowsFile != "") {
    inFlows.open(inFlowsFile.c_str());
    if(inFlows.fail()) {
      cerr << "Unable to open file " << inFlowsFile << " for read\n";
      exit(EXIT_FAILURE);
    }
  }

  try {
    // Set up pure nuc concentrations
    vector<weight_vec_t> concentration;
    concentration.resize(N_NUCLEOTIDES);
    for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++) {
      concentration[iNuc].assign(N_NUCLEOTIDES,0);
      concentration[iNuc][iNuc] = 1;
    }
    //concentration[0][1] = 0.01;
    //concentration[0][2] = 0.02;
    //concentration[2][3] = 0.03;

    weight_vec_t cf(1,(weight_t) cfVal);
    weight_vec_t ie(1,(weight_t) ieVal);
    weight_vec_t dr(1,(weight_t) drVal);
    weight_vec_t hpScale(1,(weight_t) hpScaleVal);

    unsigned int maxAdv = 3;
    unsigned int maxBases = 1.5 * nBases;

    srand(0);
    vector< vector <uint16_t> > errProfile;
    errProfile.reserve(nReads);
    unsigned int nKeyPass = 0;
    for(unsigned int iRead=0; iRead < nReads; iRead++) {
     
      weight_vec_t signal;
      string simSeq;
      if(inFlowsFile != "") {
        // Read sequence and flow data from a file
        bool done = readNextSequence(inFlows, simSeq, signal, nFlows);
        if(done)
          break;
      } else {
        // Simulate sequence and flow data from a file
        sim(simSeq, keySeq, signal, nBases, nFlows, flowString, concentration, cf, ie, dr, hpScale, maxPopulationsSim, maxAdv);
        if(snr > 0)
          addNoise(signal,snr);
      }

      if(outFlowsFile != "") {
        // Write out seqeunce and flow data
        outFlows << simSeq;
        for(unsigned int iFlow=0; iFlow<nFlows; iFlow++)
          outFlows << "\t" << setiosflags(ios::fixed) << setprecision(7) << signal[iFlow];
        outFlows << endl;
      } else {
        // Run flow data through cafie solver
        string result = "";
        hpLen_vec_t estimatedFlow;
        unsigned int nIterations = 0;
        if(solverType == "PhaseSim") {
          bool debug = false;
          PhaseSolve pSolve;
          pSolve.setHpScale(hpScale);
          pSolve.setWeightPrecision(weightPrecision);
          pSolve.setMaxPops(maxPopulationsSolve);
          pSolve.setPhaseParam(flowString,maxAdv,concentration,cf,ie,dr,ONLY_WHEN_INCORPORATING);
          nIterations = pSolve.GreedyBaseCall(signal,maxIterations,debug);
          pSolve.getSeq(result);
          estimatedFlow = pSolve.GetPredictedHpFlow();
        } else if(solverType == "CafieSolver") {
          CafieSolver solver;
          solver.SetFlowOrder((char *)flowString.c_str());
          solver.SetCAFIE((double) cfVal, (double) ieVal);
          solver.SetDroop((double) drVal);
          double *signal_array = new double[nFlows];
          for(unsigned int iFlow=0; iFlow<nFlows; iFlow++)
            signal_array[iFlow] = (double) signal[iFlow];
          solver.SetMeasured((int)nFlows, signal_array);
          bool doDotFixes = true;
          solver.Solve((int)maxIterations, doDotFixes);
          estimatedFlow.resize(nFlows);
          for(unsigned int iFlow=0; iFlow < nFlows; iFlow++) {
            unsigned int hpLen = solver.GetPredictedExtension((int)iFlow);
            estimatedFlow[iFlow] = hpLen;
            char thisNuc = flowString[iFlow % flowString.length()];
            while(hpLen > 0) {
              hpLen--;
              result += thisNuc;
            }
          }
          delete [] signal_array;
        }

        // Check if we keypass
        if((result.length() < keySeq.length()) || (result.substr(0,keySeq.length()) != keySeq)) {
          continue;
        } else {
          nKeyPass++;
        }

        // Figure out the last positive flow in the estimate
        unsigned int nEstimatedFlows = 0;
        for(unsigned int iFlow=0; iFlow < nFlows; iFlow++)
          if(estimatedFlow[iFlow] > 0)
            nEstimatedFlows = iFlow;
        nEstimatedFlows++;

        hpLen_vec_t idealFlow;
        computeSeqFlow(simSeq,flowString,idealFlow);
        errProfile.resize(1+errProfile.size());
        setErrVector(estimatedFlow,idealFlow,nEstimatedFlows,maxBases,errProfile.back());

        if(printFlows) {
          cout << "sim: ";
          for(unsigned int iFlow=0; iFlow < idealFlow.size(); iFlow++)
            cout << (int) idealFlow[iFlow];
          cout << endl;
          cout << "est: ";
          for(unsigned int iFlow=0; iFlow < nEstimatedFlows; iFlow++)
            cout << (int) estimatedFlow[iFlow];
          cout << endl;
        }
      }
    }

    if(outFlowsFile == "")
      writeErrSummary(errProfile, 1, nBases, 1, nReads, nKeyPass);
  } catch(char const *s) {
    cerr << "Caught exception:\n" << s << "\n";
  }

  if(outFlowsFile != "")
    outFlows.close();

  if(inFlowsFile != "")
    inFlows.close();
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

bool readNextSequence(ifstream &inFlows, string &simSeq, weight_vec_t &signal, const unsigned int nFlows) {
  bool done=false;
  if(inFlows.good()) {
    // Read the line
    string line;
    char delim = '\t';
    getline(inFlows,line);

    // Parse the line
    size_t current = 0;
    size_t next = 0;
    bool first=true;
    while(current < line.length()) {
      next = line.find(delim, current);
      if (next == string::npos) {
        next = line.length();
      }
      string entry = line.substr(current, next-current);
      if(first) {
        first = false;
        simSeq = entry;
      } else {
        std::istringstream i(entry);
        weight_t value;
        char c;
        if(!(i >> value) || (i.get(c))) {
          cerr << "Problem converting entry \"" << entry << "\" to numeric value" << endl;
          exit(EXIT_FAILURE);
        } else {
          signal.push_back(value);
        }
      }
      current = next + 1;
    }

    // Ensure we have the requested number of flows
    if(signal.size() < nFlows) {
      cerr << "Too few flow values read in - expected " << nFlows << " but read " << signal.size() << endl;
    } else if(signal.size() > nFlows) {
      signal.resize(nFlows);
    }
  } else {
    done = true;
  }
  return(done);
}

void sim(string &simSeq, string &keySeq, weight_vec_t &signal, const unsigned int nBases, const unsigned int nFlows, const string flowString, const vector <weight_vec_t> &concentration, const weight_vec_t &cf, const weight_vec_t &ie, const weight_vec_t &dr, const weight_vec_t &hpScale, const unsigned int maxPopulationsSim, const unsigned int maxAdv) {
  unsigned int nKeyBases = keySeq.length();
  simSeq = keySeq + randomSeq(nBases+nKeyBases);

  DroopType droopType = ONLY_WHEN_INCORPORATING;
  vector<weight_vec_t> hpWeight;
  weight_vec_t droopWeight;
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

void setErrVector(hpLen_vec_t &estimatedFlow, hpLen_vec_t &idealFlow, unsigned int nEstimatedFlows, unsigned int maxBases, vector <uint16_t> &errProfile) {
  unsigned int nBases=0;
  uint16_t nErr=0;
  unsigned int maxFlow = min((unsigned int)estimatedFlow.size(),nEstimatedFlows);
  for(unsigned int iFlow=0; iFlow < maxFlow; ++iFlow) {
    unsigned int newBases  = estimatedFlow[iFlow];
    unsigned int trueBases = (iFlow < idealFlow.size()) ? idealFlow[iFlow] : 0;
    unsigned int newErr    = abs(((int) newBases) - ((int) trueBases));
    if(newBases > 0) {
      errProfile.insert(errProfile.end(),min(newBases,trueBases),nErr);
      for(unsigned int iErr=0; iErr < newErr; ++iErr) {
        ++nErr;
        errProfile.push_back(nErr);
      }
    } else if(trueBases > 0) {
      nErr += trueBases;
    }

    nBases += newBases;
    if(nBases >= maxBases)
      break;
  }

  if(errProfile.size() > maxBases)
    errProfile.resize(maxBases);
}

void writeErrSummary(vector< vector <uint16_t> > &errProfile, unsigned int minBases, unsigned int maxBases, unsigned int stepSize, unsigned int nReads, unsigned int nKeyPass) {
  cout << "#KeyPass" << "\t" << setiosflags(ios::fixed) << setprecision(4) << (100.0 * (double)nKeyPass/(double)nReads) << endl;
  for(unsigned int readLength=minBases; readLength <= maxBases; readLength += stepSize) {
    unsigned int nKeyPass = errProfile.size();
    unsigned int nBases = 0;
    unsigned int nErrs  = 0;
    for(unsigned int iRead=0; iRead < nKeyPass; iRead++) {
      unsigned int thisLen = min(readLength,(unsigned int)errProfile[iRead].size());
      nBases += thisLen;
      nErrs  += errProfile[iRead][thisLen-1];
    }
    cout << readLength << "\t" << setiosflags(ios::fixed) << setprecision(4) << (100.0 * (double)nErrs/(double)nBases) << "\t" << nBases << endl;
  }
}

void usage() {
  cout << "" << endl;
  cout << "phaseSolveTest - test harness for phase correction and basecalling. " << endl;
  cout << "  There are two modes of analysis - writing simulated data to a file" << endl;
  cout << "  or solving phase on simulated or input data." << endl;
  cout << "  --output-flowvals <file>   : write simulated data to a file" << endl;
  cout << "  --input-flowvals <file>    : read simulated data from a file" << endl;
  cout << "  --reads <nReads>           : number of reads to generate/process" << endl;
  cout << "  --bases <nBases>           : number of bases to generate/process per read" << endl;
  cout << "  --flows <nFlows>           : number of flows to generage/process per read" << endl;
  cout << "  --iterations <nIt>         : max number of iterations of cafie solving to perform" << endl;
  cout << "  --populations-sim <nPop>   : max number of populations to track when simulating" << endl;
  cout << "  --populations-solve <nPop> : max number of populations to track when solving" << endl;
  cout << "  --solver-type <solver>     : cafie model to use - CafieSolver or PhaseSim" << endl;
  cout << "  --precision <precision>    : the weight precision used in the cafie model" << endl;
  cout << "  --cf <cfVal>               : the carry-forward value" << endl;
  cout << "  --ie <ieVal>               : the incomplete extension value" << endl;
  cout << "  --dr <drVal>               : the droop value" << endl;
  cout << "  --snr <snrVal>             : the signal-to-noise.  If zero (default) then no noise added." << endl;
  cout << "  --print-flows              : print simulated and estimated flow data" << endl;
  cout << "  --help                     : this help message" << endl;
  cout << "" << endl;
  exit(0);
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
