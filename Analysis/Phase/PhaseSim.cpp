/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
#include "PhaseSim.h"
#include "min_interval.h"
#include "IonErr.h"

using namespace std;

#define PHASE_DEBUG 0


PhaseSim::PhaseSim()
: droopType(ONLY_WHEN_INCORPORATING)
, extraTaps(0)
, maxAdvances(0)
, droopedWeight(0)
, ignoredWeight(0)
, weightPrecision(1e-6)
, maxPops(0)
{
  nucHpScale.push_back(1);
}

PhaseSim::~PhaseSim()
{
}
//flowstring flow order, seq = read in bases
long computeSeqFlow(const string &seq, const string &flowString, long basePos) {
  size_t nFlowPerCycle = flowString.length();
  size_t nBases = seq.length();
  //seqFlow.resize(0);
  
  unsigned int iFlow=0;
  unsigned int iNuc=0;
  while(iNuc < nBases) {
    char flowBase = flowString[iFlow % nFlowPerCycle];
    hpLen_t hp=0;
    while((iNuc < nBases) && (seq[iNuc]==flowBase)) {
      if (iNuc == basePos) {
        return iFlow;
      }
      iNuc++;
      hp++;
    }
    //seqFlow.push_back(hp);
    iFlow++;
  }
  return -1;//must not have been found..
}

//flowstring flow order, seq = read in bases
void computeSeqFlow(const string &seq, const string &flowString, hpLen_vec_t &seqFlow) {
  size_t nFlowPerCycle = flowString.length();
  size_t nBases = seq.length();
  seqFlow.resize(0);
  
  unsigned int iFlow=0;
  unsigned int iNuc=0;
  while(iNuc < nBases) {
    char flowBase = flowString[iFlow % nFlowPerCycle];
    hpLen_t hp=0;
    while((iNuc < nBases) && (seq[iNuc]==flowBase)) {
      iNuc++;
      hp++;
    }
    seqFlow.push_back(hp);
    iFlow++;
  }
}

void computeSeqFlow(const hpLen_vec_t &hpLen, const nuc_vec_t &hpNuc, const nuc_vec_t &flowCycle, hpLen_vec_t &seqFlow) {
  
  unsigned int nFlowsPerCycle = flowCycle.size();
  if(nFlowsPerCycle > 0) {
    unsigned int nHP = hpLen.size();
    seqFlow.resize(0);
    unsigned int iFlow=0;
    for(unsigned int iHP=0; iHP < nHP; iHP++) {
      while(flowCycle[iFlow]!=hpNuc[iHP]) {
        seqFlow.push_back(0);
        iFlow = (iFlow+1) % nFlowsPerCycle;
      }
      seqFlow.push_back(hpLen[iHP]);
      iFlow = (iFlow+1) % nFlowsPerCycle;
    }
  }

}

void PhaseSim::setFlowCycle(string flowString) {
  size_t nFlowPerCycle = flowString.length();
  flowCycle.resize(nFlowPerCycle);
  for(unsigned int i=0; i<nFlowPerCycle; i++)
    flowCycle[i] = charToNuc(flowString[i]);

  // If we already have a sequence loaded, update seqFlow for the new flow order
  if(hpLen.size() > 0)
    computeSeqFlow(hpLen,hpNuc,flowCycle,seqFlow);
}

// Overloaded call to setSeq accepting a string as input
void PhaseSim::setSeq(string seqString) {
  size_t seqLen = seqString.length();
  nuc_vec_t seq(seqLen);
  for(unsigned int i=0; i<seqLen; i++) {
    seq[i] = charToNuc(seqString[i]);
  }

  setSeq(seq);
}

void PhaseSim::getSeq(string &seqString) {

  unsigned int nBases=0;
  for(hpLen_vec_t::const_iterator iHpLen=hpLen.begin(); iHpLen!=hpLen.end(); iHpLen++)
    nBases += (unsigned int) *iHpLen;

  seqString.reserve(nBases);
  seqString.assign("");
  hpLen_vec_t::const_iterator iHpLen=hpLen.begin();
  nuc_vec_t::const_iterator   iHpNuc=hpNuc.begin();
  while(iHpLen != hpLen.end() && iHpNuc != hpNuc.end()) {
    char b = nucToChar(*iHpNuc);
    for(hpLen_t iBase=0; iBase < *iHpLen; iBase++)
      seqString += b;
    iHpLen++;
    iHpNuc++;
  }
}

// Given a sequence, sets the hpLen and hpNuc
void PhaseSim::setSeq(nuc_vec_t &seq) {

  size_t nBase = seq.size();
  hpLen.resize(nBase);
  hpNuc.resize(nBase);
  if(nBase > 0) {
    hpLen_vec_t::iterator  iHpLen = hpLen.begin();
    nuc_vec_t::iterator    iHpNuc = hpNuc.begin();
    nuc_t prevHpNuc = seq[0];
    *iHpLen = 0;
    *iHpNuc = seq[0];
    unsigned int nHp = 1;
    for(nuc_vec_t::const_iterator iSeq=seq.begin(); iSeq!=seq.end(); iSeq++) {
      if(*iSeq == prevHpNuc) {
        // Continuation of an existing HP stretch
        (*iHpLen)++;
      } else {
        // Start of a new HP stretch
        iHpLen++;
        *iHpLen = 1;
        iHpNuc++;
        *iHpNuc = *iSeq;
        prevHpNuc = *iHpNuc;
        nHp++;
      }
    }
    hpLen.resize(nHp);
    hpNuc.resize(nHp);
  }

  // Set seq flow from the hpLen and hpNuc vectors
  computeSeqFlow(hpLen,hpNuc,flowCycle,seqFlow);
}

void PhaseSim::setSeq(hpLen_vec_t &_seqFlow) {

  seqFlow = _seqFlow;

  if(flowCycle.size() == 0)
    throw("Cannot set the seq flow because the flow order has not been set");

  unsigned int nFlowsPerCycle = flowCycle.size();
  unsigned int nFlow = seqFlow.size();
  hpLen.reserve(nFlow);
  hpLen.resize(0);
  hpNuc.reserve(nFlow);
  hpNuc.resize(0);
  unsigned int iFlow=0;
  for(hpLen_vec_t::const_iterator iSeqFlow=seqFlow.begin(); iSeqFlow != seqFlow.end(); iSeqFlow++, iFlow++) {
    if(*iSeqFlow > 0) {
      hpLen.push_back(*iSeqFlow);
      hpNuc.push_back(flowCycle[iFlow % nFlowsPerCycle]);
    }
  }
  hpLen.resize(hpLen.size());
  hpNuc.resize(hpNuc.size());

}

// TODO: make this funciton faster by providing an interface to allow for
// updating a limited range when only a local region of the sequence has changed.
void PhaseSim::setAdvancerContexts(unsigned int maxAdv) {
  maxAdvances = maxAdv;

  // Make sure the advancer vectors have enough capacity for the current sequeunce.
  // If not then reserve with some headroom to help reduce future realloc
  if(advancerLen.capacity() < hpNuc.size())
    advancerLen.reserve((unsigned int)(1.2*hpNuc.size()));
  if(advancerContext.capacity() < hpNuc.size())
    advancerContext.reserve((unsigned int)(1.2*hpNuc.size()));

  // Fill in advancerContext and advancerLen vectors
  hpNucToIndex(hpNuc,maxAdvances,advancerContext,advancerLen);
}

void PhaseSim::setAdvancerWeights(vector<weight_vec_t> &concentration, weight_vec_t &cf, weight_vec_t &ie, weight_vec_t &dr, advancer_t &extendAdvancer, advancer_t &droopAdvancer, bool firstCycle) {

  unsigned int nFlowPerCycle = flowCycle.size();

  // Make sure a valid set of concentrations was supplied
  if(concentration.size() == 0) {
    concentration.resize(N_NUCLEOTIDES);
    for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++) {
      concentration[iNuc].assign(N_NUCLEOTIDES,0);
      concentration[iNuc][iNuc] = 1;
    }
  } else if(concentration.size() != N_NUCLEOTIDES) {
    throw("Vector of conentrations must be either empty or of length four");
  } else {
    for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++)
      if(concentration[iNuc].size() != N_NUCLEOTIDES)
        throw("Each entry in the vector of concentrations should be of length four");
  }

  // Make sure valid CF, IE and DR parameters were supplied
  if(cf.size() == 1) {
    cf.assign(nFlowPerCycle,cf.front());
  } else if(cf.size() != nFlowPerCycle) {
    throw("Vector of cf values must be of length 1 or of the same length as the number of flows");
  }
  if(ie.size() == 1) {
    ie.assign(nFlowPerCycle,ie.front());
  } else if(ie.size() != nFlowPerCycle) {
    throw("Vector of ie values must be of length 1 or of the same length as the number of flows");
  }
  if(dr.size() == 1) {
    dr.assign(nFlowPerCycle,dr.front());
  } else if(dr.size() != nFlowPerCycle) {
    throw("Vector of dr values must be of length 1 or of the same length as the number of flows");
  }

  if(droopAdvancer.size() != nFlowPerCycle)
    droopAdvancer.resize(nFlowPerCycle);
  if(extendAdvancer.size() != nFlowPerCycle)
    extendAdvancer.resize(nFlowPerCycle);

  unsigned int nLengths = maxAdvances+1;
  // Iterate over each flow in the flowCycle
  for(unsigned int iFlow=0; iFlow < flowCycle.size(); iFlow++) {

    // Determine the effect of carry-forward on the nuc concentrations
    weight_vec_t cumulative_cf = cf;
    weight_vec_t conc_with_cf = concentration[flowCycle[iFlow]];
    for(unsigned int stepsBack=1; stepsBack < nFlowPerCycle; stepsBack++) {
      if(firstCycle && stepsBack > iFlow)
        break;
      unsigned int prevFlow = (iFlow - stepsBack + nFlowPerCycle) % nFlowPerCycle;
      for(unsigned int i=0; i < conc_with_cf.size(); i++) {
        conc_with_cf[i] += cumulative_cf[i] * concentration[flowCycle[prevFlow]][i];
        cumulative_cf[i] *= cf[i];
      }
    }
    // Sweep the concentrations to ensure nothing is larger than 1
    for(unsigned int i=0; i < conc_with_cf.size(); i++) {
      if(conc_with_cf[i] > 1)
        conc_with_cf[i] = 1;
    }

    // Debug code - print out the nuc concentrations reflecting both contamination and carry-forward
    //cout << "flow " << iFlow << " nuc " << nucToChar(flowCycle[iFlow]) << ", concs = ";
    //for(unsigned int i=0; i < conc_with_cf.size(); i++)
    //  cout << " " << setiosflags(ios::fixed) << setprecision(8) << conc_with_cf[i];
    //cout << "\n";

    if(droopAdvancer[iFlow].size() != nLengths)
      droopAdvancer[iFlow].resize(nLengths);
    if(extendAdvancer[iFlow].size() != nLengths)
      extendAdvancer[iFlow].resize(nLengths);

    // Iterate over each limit to extension length
    unsigned int nContexts = 1;
    for(unsigned int iLength=0; iLength <= maxAdvances; iLength++) {

      if(iLength > 0)
        nContexts *= N_NUCLEOTIDES;
      if(droopAdvancer[iFlow][iLength].size() != nContexts)
        droopAdvancer[iFlow][iLength].resize(nContexts);
      if(extendAdvancer[iFlow][iLength].size() != nContexts)
        extendAdvancer[iFlow][iLength].resize(nContexts);

      // Iterate over each sequence context
      for(unsigned int iContext=0; iContext < nContexts; iContext++) {
        setAdvancerWeightsByContext(conc_with_cf,iLength,iContext,ie,dr,extendAdvancer[iFlow][iLength][iContext],droopAdvancer[iFlow][iLength][iContext]);
      }
    }
  }
}

void PhaseSim::printAdvancerWeights(advancer_t &extendAdvancer, advancer_t &droopAdvancer) {
  unsigned int nHP = hpLen.size();
  for(unsigned int iFlow=0; iFlow < flowCycle.size(); iFlow++) {
    cout << "working on flow " << iFlow << " which is nuc " << nucToChar(flowCycle[iFlow]) << "\n";
    for(unsigned int iHP=0; iHP < nHP; iHP++) {

    //cout << "templates of length " << iHP << " can advance " << (int) advancerLen[iHP] << " states\n";

    //cout << "  seq context = " << advancerContext[iHP] << " = ";
    nuc_vec_t seqContext;
    indexToSeqContext(advancerContext[iHP],seqContext,advancerLen[iHP]);
    string context;
    for(unsigned int iSeq=0; iSeq < seqContext.size(); iSeq++)
      context += nucToChar(seqContext[iSeq]);
    //cout << context << "\n";

      cout << "flow" << iFlow << "  " << context;
      cout << "\tex:";
      for(unsigned int i=0; i <= extendAdvancer[iFlow][advancerLen[iHP]][advancerContext[iHP]].size(); i++)
        cout << "  " << setiosflags(ios::fixed) << setprecision(6) << extendAdvancer[iFlow][advancerLen[iHP]][advancerContext[iHP]][i];
      cout << "\tdr:";
      for(unsigned int i=0; i <= droopAdvancer[iFlow][advancerLen[iHP]][advancerContext[iHP]].size(); i++)
        cout << "  " << setiosflags(ios::fixed) << setprecision(6) << droopAdvancer[iFlow][advancerLen[iHP]][advancerContext[iHP]][i];
      cout << "\n";
    }

  }
}

void PhaseSim::setAdvancerWeightsByContext(weight_vec_t &nuc_conc, unsigned int len, unsigned int seqContextIndex, weight_vec_t &nuc_ie, weight_vec_t &nuc_dr, weight_vec_t &extender, weight_vec_t &drooper) {

  nuc_vec_t seqContext;
  indexToSeqContext(seqContextIndex, seqContext, len);

  weight_vec_t cc(len);
  weight_vec_t dr(len);
  weight_vec_t iePrime(len);
  weight_vec_t drPrime(len);
  for(unsigned int i=0; i<len; i++) {
    cc[i]      = nuc_conc[seqContext[i]];
    dr[i]      = nuc_dr[seqContext[i]];
    iePrime[i] = 1-nuc_ie[seqContext[i]];
    drPrime[i] = 1-dr[i];
  }
  
  // Initialization
  drooper.assign(len+1,0);
  extender.assign(len+1,0);
  extender.front() = 1;
  for(unsigned int step=0; step<len; step++) {
    // Welcome to the core of the CAFIE model
    weight_t toReAllocate = extender[step];  // The template mass that will be reallocated
      weight_t droops, extends;
      switch(droopType) {
        case(EVERY_FLOW):
          // In this model strands droop regardless of whether or not there will be any extension.
          droops  = toReAllocate * dr[step];
          extends = toReAllocate * (drPrime[step] * cc[step] * iePrime[step]);
          drooper[step] = droops;
          extender[step+1] = extends;
          extender[step] -= (droops+extends);
        break;

        case(ONLY_WHEN_INCORPORATING):
          // In this model droop only happens when incorporation happens, and will be proportional to it.
          extends = toReAllocate * (cc[step] * iePrime[step]);
          droops  = extends * dr[step];
          drooper[step] = droops;
          extender[step+1] = extends-droops;
          extender[step] -= extends;
        break;

        default:
          throw("Invalid droop model");
    }
  }

  // Trim back negligible weights
  trimTail(extender,weightPrecision);
  trimTail(drooper,weightPrecision);

#if PHASE_DEBUG
  // Sanity check - confirm that everything sums to one
  double drSum = 0;
  for(unsigned int i=0; i<drooper.size(); i++)
    drSum += drooper[i];
  double exSum = 0;
  for(unsigned int i=0; i<extender.size(); i++)
    exSum += extender[i];
  if(abs(1-drSum-exSum) > weightPrecision) {
    cout << "diff is " << 1-drSum-exSum << "\n";
    cout << "  seq context = " << seqContextIndex << " = ";
    for(unsigned int iSeq=0; iSeq < seqContext.size(); iSeq++)
      cout << nucToChar(seqContext[iSeq]);
    cout << "\n";
    cout << "  " << "cc:";
    for(unsigned int i=0; i < cc.size(); i++)
      cout << "  " << setiosflags(ios::fixed) << setprecision(4) << cc[i];
    cout << "\n";
    cout << "  " << "dr:";
    for(unsigned int i=0; i < dr.size(); i++)
      cout << "  " << setiosflags(ios::fixed) << setprecision(4) << dr[i];
    cout << "\n";
    cout << "  " << "ie:";
    for(unsigned int i=0; i < iePrime.size(); i++)
      cout << "  " << setiosflags(ios::fixed) << setprecision(4) << (1-iePrime[i]);
    cout << "\n";
    cout << "  " << "extender:";
    for(unsigned int i=0; i < extender.size(); i++)
      cout << "  " << setiosflags(ios::fixed) << setprecision(4) << extender[i];
    cout << "\n";
    cout << "  " << "drooper:";
    for(unsigned int i=0; i < drooper.size(); i++)
      cout << "  " << setiosflags(ios::fixed) << setprecision(4) << drooper[i];
    cout << "\n";
  }
#endif
}

void PhaseSim::setHpScale(const weight_vec_t &_nucHpScale) {

  unsigned int n = _nucHpScale.size();
  if(n != 1 && n != N_NUCLEOTIDES)
    throw("nucHpScale vector must be of length 1 or 4\n");

  nucHpScale = _nucHpScale;
}

void PhaseSim::saveTemplateState(weight_vec_t &_hpWeight, weight_t &_droopedWeight, weight_t &_ignoredWeight, readLen_vec_t &_posWeight, hpLen_vec_t &_hpLen, nuc_vec_t &_hpNuc) {
  _hpWeight      = hpWeight;
  _droopedWeight = droopedWeight;
  _ignoredWeight = ignoredWeight;
  _posWeight     = posWeight;
  _hpLen         = hpLen;
  _hpNuc         = hpNuc;
}

void PhaseSim::resetTemplate(weight_vec_t &_hpWeight, weight_t &_droopedWeight, weight_t &_ignoredWeight, readLen_vec_t &_posWeight, hpLen_vec_t &_hpLen, nuc_vec_t &_hpNuc) {
#if PHASE_DEBUG
  unsigned int _nWeight = _hpWeight.size();
  assert(_hpLen.size() == _hpNuc.size());
  assert(_hpLen.size() == (_nWeight-1));
#endif

  hpLen         = _hpLen;
  hpNuc         = _hpNuc;
  resetTemplate(_hpWeight, _droopedWeight, _ignoredWeight, _posWeight);
}

void PhaseSim::resetTemplate(weight_vec_t &_hpWeight, weight_t &_droopedWeight, weight_t &_ignoredWeight, readLen_vec_t &_posWeight) {
#if PHASE_DEBUG
  unsigned int _nWeight = _hpWeight.size();
  assert(_posWeight.front() <= _posWeight.back());
  assert(_posWeight.back() < _nWeight);
#endif

  // Copy in the new values
  hpWeight      = _hpWeight;
  droopedWeight = _droopedWeight;
  ignoredWeight = _ignoredWeight;
  posWeight     = _posWeight;

  // Make sure template weights sum to one
#if PHASE_DEBUG
  assert(hpWeightOK());
#endif
}

void PhaseSim::resetTemplate(void) {
  unsigned int nWeight = hpLen.size()+1;

  weight_vec_t _hpWeight(nWeight,0);
  _hpWeight[0] = 1;
  weight_t _droopedWeight = 0;
  weight_t _ignoredWeight = 0;
  readLen_vec_t _posWeight(1,0);

  resetTemplate(_hpWeight, _droopedWeight, _ignoredWeight, _posWeight);
}

weight_t PhaseSim::applyFlow(unsigned int iFlow, advancer_t &extendAdvancer, advancer_t &droopAdvancer, bool testOnly) {
  weight_vec_t newHpWeight;
  readLen_vec_t newPosWeight;
  return(applyFlow(iFlow, extendAdvancer, droopAdvancer, testOnly, newHpWeight, newPosWeight));
}

weight_t PhaseSim::applyFlow(unsigned int iFlow, advancer_t &extendAdvancer, advancer_t &droopAdvancer, bool testOnly, weight_vec_t &newHpWeight, readLen_vec_t &newPosWeight) {
#if PHASE_DEBUG
  assert(advancerLen.size() == hpLen.size());
#endif
  if(iFlow >= flowCycle.size())
    iFlow = iFlow % flowCycle.size();

  unsigned int nHP = hpNuc.size();
  unsigned int nWeight = hpWeight.size();
#if PHASE_DEBUG
  assert(nHP+1 == nWeight);
#endif

  // Initialize scratch space to which new configuration will be written
  newHpWeight.assign(nWeight,0);
  newHpWeight[nHP] = hpWeight[nHP];

  weight_t incorporationSignal = 0;
  weight_t newDroopedWeight = 0;
  for(size_t iPosWeight=0; iPosWeight < posWeight.size(); iPosWeight++) {
    readLen_t iHP = posWeight[iPosWeight];

    if(iHP==advancerContext.size())
      break;

    weight_t startingTemplates = hpWeight[iHP];
    advancer_len_t nAdvance = advancerLen[iHP];
    weight_vec_t &ex = extendAdvancer[iFlow][nAdvance][advancerContext[iHP]];
    weight_vec_t &dr =  droopAdvancer[iFlow][nAdvance][advancerContext[iHP]];
    newHpWeight[iHP] += startingTemplates * ex[0];
    newDroopedWeight += startingTemplates * dr[0];
    advancer_len_t maxAdvance = max(ex.size(),dr.size());
    for(unsigned int iAdvance=1; iAdvance < maxAdvance; iAdvance++) {
      // hpLenSum is the sum of the HP incorporation signals for HPs through which we advance
      weight_t hpLenSum = 0;
      for(unsigned int extra=0; extra < iAdvance; extra++) {
        hpLen_t thisHpLen = hpLen[iHP+extra];
        //weight_t scale = nucHpScale[hpNuc[iHP+extra] % nucHpScale.size()];
        //hpLenSum += thisHpLen * pow(scale,(weight_t)thisHpLen);
        hpLenSum += thisHpLen;
      }
      // update live templates
      weight_t extended;
      if(iAdvance < ex.size()) {
        extended = startingTemplates * ex[iAdvance];
        newHpWeight[iHP+iAdvance] += extended;
      } else {
        extended = 0;
      }
      // update drooped templates
      weight_t drooped;
      if(iAdvance < dr.size()) {
        drooped = startingTemplates * dr[iAdvance];
        newDroopedWeight += drooped;
      } else {
        drooped = 0;
      }
      // update incorporation signal
      incorporationSignal += (extended+drooped) * hpLenSum;
    }
  }

#if PHASE_DEBUG
  assert(hpWeightOK());
#endif

  if(!testOnly) {
    // Copy the new configuration into hpWeight
    // TODO: possible fast approximation by limiting to tracking no more than N positive populations here
    // TODO: instead of iterating over all positive positions, jump only over ones with positive weight.
    newPosWeight.clear();
    unsigned int max_possible_hp_len = min((nWeight-1),(unsigned int) (posWeight.back() + maxAdvances));
    for(unsigned int iHP=posWeight.front(); iHP <= max_possible_hp_len; iHP++) {
      weight_t thisWeight = newHpWeight[iHP];
      if(thisWeight > weightPrecision) {
        hpWeight[iHP] = thisWeight;
        newPosWeight.push_back(iHP);
      } else {
        hpWeight[iHP] = 0;
        ignoredWeight += thisWeight;
      }
    }
    droopedWeight += newDroopedWeight;
    posWeight = newPosWeight;
#if PHASE_DEBUG
    assert(hpWeightOK());
#endif
  }

#if PHASE_DEBUG
  assert(hpWeightOK());
#endif

  // Recursive calls to add multi-tap flows if requested.
  // Currently the signal from these extra flows is just lost.
  // It could easily be tracked here if necessary.
  if(!testOnly && (extraTaps > 0)) {
    extraTaps--;
    applyFlow(iFlow, extendAdvancer, droopAdvancer, testOnly, newHpWeight, newPosWeight);
#if PHASE_DEBUG
    assert(hpWeightOK());
#endif
    extraTaps++;
  }

  return(incorporationSignal);
}

weight_t PhaseSim::getWeightSum(void) {
  unsigned int nWeight = hpWeight.size();
  weight_t total = 0;
  for(unsigned int iHP=0; iHP < nWeight; iHP++) {
    total += hpWeight[iHP];
  }
  total += droopedWeight;
  total += ignoredWeight;
  return(total);
}

bool PhaseSim::hpWeightOK(void) {
  weight_t total = getWeightSum();
  weight_t eps   = max(numeric_limits<weight_t>::epsilon(),10*weightPrecision);
#if PHASE_DEBUG
  weight_t activ = accumulate(hpWeight.begin(),      hpWeight.end(),      0.0);
#endif
  weight_t activTail = accumulate(hpWeight.begin()      + posWeight.back() + 1, hpWeight.end(),      0.0);

  bool okTotal = abs(total-1) < eps;
  bool okActivTail = activTail < eps;

  bool ok = okTotal && okActivTail;

#if PHASE_DEBUG
  if(not ok){
    cout << "OK?"
      << setw(16) << scientific << activ
      << setw(16) << scientific << droopedWeight
      << setw(16) << scientific << ignoredWeight
      << setw(16) << scientific << total
      << endl;
    cout << setw(16) << scientific << total     << endl;
    cout << setw(16) << scientific << 1.0-total << endl;
    unsigned int nWeight = hpWeight.size();
    cout << setw(6) << hpWeight.size() << endl;
    for(unsigned int iHP=0; iHP < nWeight; iHP++) {
      cout << setw( 6) << iHP << setw(16) << scientific << hpWeight[iHP] << endl;
    }
  }
#endif
  return(ok);
}

weight_t PhaseSim::vectorSum(const weight_vec_t &v) {
  weight_t sum = 0;
  for(weight_vec_t::const_iterator i=v.begin(); i != v.end(); i++)
    sum += *i;
  return(sum);
}

bool PhaseSim::advancerWeightOK(advancer_t &extendAdvancer, advancer_t &droopAdvancer) {
  bool weightOK = true;

  for(unsigned int iFlow=0; iFlow < flowCycle.size(); iFlow++) {
    unsigned int nContexts = 1;
    for(unsigned int iLength=0; iLength <= maxAdvances; iLength++) {
      if(iLength > 0)
        nContexts *= N_NUCLEOTIDES;
      for(unsigned int iContext=0; iContext < nContexts; iContext++) {
        weight_t s = vectorSum(extendAdvancer[iFlow][iLength][iContext]) + vectorSum(droopAdvancer[iFlow][iLength][iContext]);
        if(abs(1-s) > 1e-8) {
          weightOK = false;
          break;
        }
      }
    }
  }

  return(weightOK);
}


nuc_t charToNuc(char base) {

  switch(base) {
    case 'A':
    case 'a':
      return(0);
      break;

    case 'C':
    case 'c':
      return(1);
      break;

    case 'G':
    case 'g':
      return(2);
      break;

    case 'T':
    case 't':
      return(3);
      break;

    default:
      throw("Invalid nucleotide");
  }

}

char nucToChar(nuc_t nuc) {
  switch(nuc) {
    case 0:
      return('A');
      break;

    case 1:
      return('C');
      break;

    case 2:
      return('G');
      break;

    case 3:
      return('T');
      break;

    default:
      cerr << "Invalid nuc " << (int)nuc << "\n";
      assert(false);
      throw("Invalid nuc value");
  }

}

void PhaseSim::simulate(
  string                  flowString,
  string                  seq,
  vector<weight_vec_t>    concentration,
  weight_vec_t            cf,
  weight_vec_t            ie,
  weight_vec_t            dr,
  unsigned int            nFlow,
  weight_vec_t &          signal,
  vector<weight_vec_t> &  hpWeightReturn,
  weight_vec_t         &  droopWeightReturn,
  bool                    returnIntermediates,
  DroopType               droopType,
  unsigned int            maxAdv
) {

  setDroopType(droopType);
  setFlowCycle(flowString);
  setSeq(seq);
  setAdvancerContexts(maxAdv);
  setAdvancerWeights(concentration, cf, ie, dr, extendAdvancerFirst, droopAdvancerFirst, true );
  setAdvancerWeights(concentration, cf, ie, dr, extendAdvancer,      droopAdvancer,      false);
  resetTemplate();
  
  signal.resize(nFlow);
  if(returnIntermediates) {
    hpWeightReturn.resize(nFlow);
    droopWeightReturn.resize(nFlow);
  }

  bool testOnly = false;
  bool firstCycle = true;
  unsigned int nFlowPerCycle = flowCycle.size();
  for(unsigned int iFlow=0; iFlow<nFlow; iFlow++) {
    if(iFlow >= nFlowPerCycle)
      firstCycle = false;

    if(firstCycle)
      signal[iFlow] = applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst, testOnly);
    else
      signal[iFlow] = applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer, testOnly);

    if(returnIntermediates) {
      hpWeightReturn[iFlow]    = getHpWeights();
      droopWeightReturn[iFlow] = getDroopWeight();
    }
#if PHASE_DEBUG
    if(!hpWeightOK())
      cerr << "Problem after applying flow " << iFlow << ", weight sum is " << setiosflags(ios::fixed) << setprecision(10)<< getWeightSum() << "\n";
#endif
  }
  
}

// TODO: redo this with bit shifting for speed
void PhaseSim::indexToSeqContext(unsigned int index, nuc_vec_t &seqContext, unsigned int len) {

  seqContext.resize(len);

  if(len > 0) {
    unsigned int divisor = 1;
    for(unsigned int iNuc=1; iNuc < len; iNuc++) {
      divisor *= N_NUCLEOTIDES;
    }
    for(int iNuc=len-1; iNuc >= 0; iNuc--) {
      nuc_t nuc = index/divisor;
      index -= nuc*divisor;
      seqContext[iNuc] = nuc;
      divisor /= N_NUCLEOTIDES;
    }
  }

}

// Key normalization
void keyNormalize(weight_vec_t &signal, string key, string flowOrder) {
  hpLen_vec_t keyFlow;
  computeSeqFlow(key, flowOrder, keyFlow);
  keyNormalize(signal, keyFlow);
}

// Key normalization when key flows have already been computed
void keyNormalize(weight_vec_t &signal, hpLen_vec_t &keyFlow) {

  // Determine scaling factor
  unsigned int nOneMer=0;
  weight_t oneMerSum=0;
  unsigned int maxKeyFlow = min(keyFlow.size()-1,signal.size());
  for(unsigned int iFlow=0; iFlow < maxKeyFlow; iFlow++) {
    if(keyFlow[iFlow] == 1) {
      nOneMer++;
      oneMerSum += signal[iFlow];
    }
  }
  if(nOneMer==0)
    ION_ABORT("Cannot normalize to key, it has no 1-mers");
  oneMerSum /= (weight_t) nOneMer;

  // Scale all flow values
  for(unsigned int iFlow=0; iFlow < signal.size(); iFlow++) {
    signal[iFlow] /= oneMerSum;
  }
}

void trimTail(weight_vec_t &v, weight_t precision) {
  // keep track of weight that is taken away
  weight_t trimmedWeight = 0;

  // trim trailing entries
  weight_vec_t::reverse_iterator rit;
  for(rit=v.rbegin(); rit < v.rend(); ++rit) {
    if(*rit < precision) {
      trimmedWeight += *rit;
      v.pop_back();
    } else {
      break;
    }
  }

  // zero intervening entries
  unsigned int nonzeroEntries=0;
  for(; rit < v.rend(); ++rit) {
    if(*rit < precision) {
      trimmedWeight += *rit;
      *rit = 0;
    } else {
      nonzeroEntries++;
    }
  }

  // add weight back in to the nonzero entries
  if(nonzeroEntries > 0) {
    trimmedWeight /= (weight_t) nonzeroEntries;
    weight_vec_t::iterator it;
    for(it=v.begin(); it != v.end(); ++it) {
      if(*it > 0) {
        *it += trimmedWeight;
      }
    }
  } else {
    if(v.size()==0) {
      v.push_back(trimmedWeight);
    } else {
      v[0] = trimmedWeight;
    }
  }
}

unsigned int makeMask(unsigned char n) {
  unsigned int mask=0;
  for(unsigned int i=0; i < n; i++) {
    mask <<= 1;
    mask |= 1;
  }
  return(mask);
}

void hpNucToIndex(nuc_vec_t &hpNuc, advancer_len_t maxAdvances, vector<unsigned int> &contextIndex, vector<advancer_len_t> &contextLen) {

  // Make a bitmask with lowest 2*maxAdvances bits set to 1.
  unsigned int contextMask = makeMask(2*maxAdvances);

  unsigned int nHP = hpNuc.size();
  contextIndex.resize(nHP);
  contextLen.resize(nHP);
  unsigned int thisContextIndex=0;
  advancer_len_t thisContextLen=0;
  nuc_vec_t::reverse_iterator hpNucRit=hpNuc.rbegin();
  vector<unsigned int>::reverse_iterator contextIndexRit=contextIndex.rbegin();
  vector<advancer_len_t>::reverse_iterator contextLenRit=contextLen.rbegin();
  for( ; hpNucRit != hpNuc.rend(); hpNucRit++, contextIndexRit++, contextLenRit++) {
    thisContextIndex <<= 2;
    thisContextIndex &= contextMask;
    thisContextIndex |= (unsigned int) *hpNucRit;
    if(thisContextLen < maxAdvances)
      thisContextLen++;
    *contextIndexRit = thisContextIndex;
    *contextLenRit = thisContextLen;
  }
}


void PhaseSim::setPhaseParam(
  string               & _flowString,
  hpLen_t                _maxAdvances,
  vector<weight_vec_t> & _concentration,
  weight_vec_t         & _cf,
  weight_vec_t         & _ie,
  weight_vec_t         & _dr,
  DroopType              _droopType
) {
  setFlowCycle(_flowString);
  setMaxAdvances(_maxAdvances);
  setDroopType(_droopType);
  setAdvancerWeights(_concentration, _cf, _ie, _dr, extendAdvancerFirst, droopAdvancerFirst, true );
  setAdvancerWeights(_concentration, _cf, _ie, _dr, extendAdvancer,      droopAdvancer,      false);
}
