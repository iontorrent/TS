/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     FlowDist.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#ifndef FLOWDIST_H
#define FLOWDIST_H

#include <iostream>
#include "LocalContext.h"
#include "VariantAssist.h"
#include "peakestimator.h"  // for definition of maxsigdev only
//@TODO: break this up into "bookkeeping" (i.e. filterable fields)
// and "data processing" - i.e. likelihoods, etc.


class FlowDist {
  public:
    uint32_t refPosition;
    int *homPolyDist;
    int *corrFlowDist;

    VariantBook summary_stats;
    VariantOutputInfo summary_info;

    uint16_t indelCount;
    uint16_t flowLength;

    vector<float>* refLikelihoods;
    vector<float>* varLikelihoods;

    LocalReferenceContext variant_context;

    bool DEBUG;

    FlowDist():refPosition(0),homPolyDist(NULL), corrFlowDist(NULL),
        indelCount(0), flowLength(0),
        refLikelihoods(NULL), varLikelihoods(NULL),
        DEBUG(false) {
    };
    FlowDist(uint32_t refPos, LocalReferenceContext &my_context, uint16_t flowLen, bool debug) {
      refPosition = refPos;

      flowLength = flowLen;
      variant_context = my_context;

      indelCount = 0;

      homPolyDist = new int[MAXSIGDEV];
      corrFlowDist = new int[MAXSIGDEV];
      for (int i = 0; i < MAXSIGDEV; i++) {
        homPolyDist[i] = 0;
        corrFlowDist[i] = 0;
      }
      refLikelihoods = new vector<float>();
      varLikelihoods = new vector<float>();

      DEBUG = debug;
      summary_stats.DEBUG = debug;
    };

    ~FlowDist() {
      if (homPolyDist != NULL)
        delete[] homPolyDist;


      if (corrFlowDist != NULL)
        delete[] corrFlowDist;

      if (refLikelihoods != NULL)
        delete refLikelihoods;

      if (varLikelihoods != NULL)
        delete varLikelihoods;
    };



    int * getHomPolyDist() {
      return homPolyDist;
    }

    int * getCorrPolyDist() {
      return corrFlowDist;
    }

    uint32_t getRefPosition() {
      return refPosition;
    };

    uint8_t getHomLength() {
      return variant_context.my_hp_length.at(0);
    };

    void incrementIndelCount() {
      indelCount++;
    };

    uint16_t getIndelCount() {
      return indelCount;
    };

    vector<float>* getReferenceLikelihoods() {
      return refLikelihoods;
    };

    vector<float>* getVariantLikelihoods() {
      return varLikelihoods;
    };

    string getGenotype(int alleleIndex) {
      //TODO total hack for now
      float altAlleleFreq = summary_stats.getAltAlleleFreq();
      stringstream genotypeStream;
      if (altAlleleFreq < 0.75)
        genotypeStream << 0 << "/" << alleleIndex;

      else
        genotypeStream << alleleIndex << "/" << alleleIndex;

      return genotypeStream.str();
    };

    bool isHet() {
      float altAlleleFreq = summary_stats.getAltAlleleFreq();
      if (altAlleleFreq < 0.75)
        return true;
      else
        return false;

    }
};

#endif // FLOWDIST_H
