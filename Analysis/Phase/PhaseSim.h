/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PHASESIM_H
#define PHASESIM_H

#include <string>
#include <vector>
#include <stdint.h>
#include "SpecialDataTypes.h"



typedef uint16_t                readLen_t;      // for holding lengths of HP sequences
typedef std::vector<readLen_t>  readLen_vec_t;
typedef unsigned char           nuc_t;
typedef std::vector<nuc_t>      nuc_vec_t;
typedef unsigned char           advancer_len_t;

typedef std::vector<std::vector<std::vector<weight_vec_t> > > advancer_t;

const unsigned int N_NUCLEOTIDES = 4;

enum DroopType { EVERY_FLOW, ONLY_WHEN_INCORPORATING };

nuc_t charToNuc(char base);
char  nucToChar(nuc_t nuc);
void computeSeqFlow(const hpLen_vec_t &hpLen, const nuc_vec_t &hpNuc, const nuc_vec_t &flowCycle, hpLen_vec_t &seqFlow);
void computeSeqFlow(const std::string &seq, const std::string &flowString, hpLen_vec_t &seqFlow);
//returns flow position of basePos
long computeSeqFlow(const std::string &seq, const std::string &flowString, long basePos);
void keyNormalize(weight_vec_t &signal, std::string key, std::string flowOrder);
void keyNormalize(weight_vec_t &signal, hpLen_vec_t &keyFlow);
void trimTail(weight_vec_t &v, weight_t precision);
unsigned int makeMask(unsigned char n);
void hpNucToIndex(nuc_vec_t &hpNuc, advancer_len_t maxAdvances, std::vector<unsigned int> &contextIndex, std::vector<advancer_len_t> &contextLen);

class PhaseSim {
  public:
    PhaseSim();
    virtual ~PhaseSim();
    void setFlowCycle(std::string flowString);
    void setMaxAdvances(hpLen_t _maxAdvances) { maxAdvances = _maxAdvances; };
    void setWeightPrecision(weight_t _weightPrecision) { weightPrecision = _weightPrecision; };
    void setExtraTaps(unsigned int _extraTaps) { extraTaps = _extraTaps; };
    void setSeq(std::string seq);
    void setSeq(nuc_vec_t &seq);
    void setSeq(hpLen_vec_t &_seqFlow);
    void setMaxPops(unsigned int _maxPops) { maxPops = _maxPops; };
    void setDroopType(DroopType d) { droopType = d; };
    void setAdvancerContexts(unsigned int maxAdv=2);
    void setAdvancerWeights(std::vector<weight_vec_t> &concentration, weight_vec_t &cf, weight_vec_t &ie, weight_vec_t &dr, advancer_t &extendAdvancer, advancer_t &droopAdvancer, bool firstCycle);
    void setAdvancerWeightsByContext(weight_vec_t &nuc_conc, unsigned int len, unsigned int seqContextIndex, weight_vec_t &nuc_ie, weight_vec_t &nuc_dr, weight_vec_t &extender, weight_vec_t &drooper);
    void setPhaseParam(
      std::string               & _flowString,
      hpLen_t                     _maxAdvances,
      std::vector<weight_vec_t> & _concentration,
      weight_vec_t              & _cf,
      weight_vec_t              & _ie,
      weight_vec_t              & _dr,
      DroopType                   _droopType
    );
    void setHpScale(const weight_vec_t &_nucHpScale);
    void resetTemplate(weight_vec_t &_hpWeight, weight_t &_droopedWeight, weight_t &_ignoredWeight, readLen_vec_t &_posWeight, hpLen_vec_t &_hpLen, nuc_vec_t &_hpNuc);
    void resetTemplate(weight_vec_t &_hpWeight, weight_t &_droopedWeight, weight_t &_ignoredWeight, readLen_vec_t &_posWeight);
    void resetTemplate(void);
    void saveTemplateState(weight_vec_t &_hpWeight, weight_t &_droopedWeight, weight_t &_ignoredWeight, readLen_vec_t &_posWeight, hpLen_vec_t &_hpLen, nuc_vec_t &_hpNuc);
    weight_t applyFlow(unsigned int iFlow, advancer_t &extendAdvancer, advancer_t &droopAdvancer, bool testOnly=false);
    weight_t applyFlow(unsigned int iFlow, advancer_t &extendAdvancer, advancer_t &droopAdvancer, bool testOnly, weight_vec_t &newHpWeight, readLen_vec_t &newPosWeight);
    weight_t getWeightSum(void);
    bool hpWeightOK(void);
    bool advancerWeightOK(advancer_t &extendAdvancer, advancer_t &droopAdvancer);
    DroopType getDroopType(void) { return(droopType); };
    hpLen_vec_t &getSeqFlow(void) { return(seqFlow); };
    void getSeq(std::string &seqString);
    advancer_t &getExtendAdvancerFirst(void) { return(extendAdvancerFirst); };
    advancer_t &getDroopAdvancerFirst(void) { return(droopAdvancerFirst); };
    advancer_t &getExtendAdvancer(void) { return(extendAdvancer); };
    advancer_t &getDroopAdvancer(void) { return(droopAdvancer); };
    const weight_vec_t &getHpWeights(void) { return(hpWeight); };
    const weight_t &getDroopWeight(void) { return(droopedWeight); };
    void printAdvancerWeights(advancer_t &extendAdvancer, advancer_t &droopAdvancer);
    void simulate(
      std::string                flowCycle,
      std::string                seq,
      std::vector<weight_vec_t>  concentration,
      weight_vec_t               cf,
      weight_vec_t               ie,
      weight_vec_t               dr,
      unsigned int               nFlow,
      weight_vec_t             & signal,
      std::vector<weight_vec_t>& hpWeight,
      weight_vec_t             & droopWeight,
      bool                       returnIntermediates,
      DroopType                  droopType,
      unsigned int               maxAdv
    );

  protected:
    void indexToSeqContext(unsigned int index, nuc_vec_t &seqContext, unsigned int len);
    weight_t vectorSum(const weight_vec_t &v);

    DroopType                   droopType;           // The model we are using for droop
    nuc_vec_t                   flowCycle;           // For example: (3,0,1,2) for "TACG"
    unsigned int                extraTaps;           // The number of follow-up 'taps' or repeat flows to do for every flow
    hpLen_vec_t                 hpLen;               // Length of each HP for true sequence
    nuc_vec_t                   hpNuc;               // Nucs in true HP sequence
    hpLen_vec_t                 seqFlow;             // The ideal (phase-error-free) sequence in flow space
    advancer_len_t              maxAdvances;         // The max number of states any population can advance in one flow
    std::vector<advancer_len_t> advancerLen;         // Max number of advances at each position
    std::vector<unsigned int>   advancerContext;     // Encoding of sequence context in the next maxAdvandes nucs at each position
    advancer_t                  droopAdvancerFirst;  // Models how droop operates in the first cycle
    advancer_t                  extendAdvancerFirst; // Models how extension operates in the first cycle
    advancer_t                  droopAdvancer;       // Models how droop operates in cycles after the first
    advancer_t                  extendAdvancer;      // Models how extension operates in cycles after the first
    weight_vec_t                hpWeight;            // The fraction of templates currently in each state
    weight_t                    droopedWeight;       // The fraction of templates of each length that have drooped
    weight_t                    ignoredWeight;       // The fraction of templates in small ignored populations
    readLen_vec_t               posWeight;           // Stores the lengths (indices) of templates of positive weight, for fast computation
    weight_t                    weightPrecision;     // Precision to use when deciding if a population is too small to track
    weight_vec_t                nucHpScale;          // Incorporation signal for HP of length h is h*nucHpDrop[nuc]^(h-1)
    unsigned int                maxPops;             // Maximum number of populations to track, or zero for no limit
};

#endif // PHASESIM_H
