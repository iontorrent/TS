/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SPECIALDATATYPES_H
#define SPECIALDATATYPES_H

// for 'convenience' data types that get shared broadly, but possibly not requiring any functions
#include <string>
#include <vector>
#ifndef ALIGNSTATS_IGNORE
#include "Mask.h"
#endif

typedef char                  hpLen_t;
typedef std::vector<hpLen_t>  hpLen_vec_t;
typedef float                 weight_t;     // use this to toggle between double/float
typedef std::vector<weight_t> weight_vec_t;

struct SequenceItem {
  #ifndef ALIGNSTATS_IGNORE
  MaskType    type;
  #endif
  std::string seq;

  int numKeyFlows;    // number of flows needed to sequence through the entire key
  int usableKeyFlows; // number of flows where we know the base is really a 1-mer (last base not always good to use)
  int Ionogram[64];   // vector of per-flow expected incorporations
  int zeromers[64];   // list of what flows int the key have 0-mers, and in nuc flow order
  int onemers[64];    // list of what flows in the key have 1-mers, and in nuc flow order

  // Boost Serialization support:
  template <class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // fprintf(stdout, "Serialize SequenceItem ... ");
    #ifndef ALIGNSTATS_IGNORE
    ar & type;
    #endif
    ar & seq;
    ar & numKeyFlows;
    ar & usableKeyFlows;
    ar & Ionogram;
    ar & zeromers;
    ar & onemers;
    // fprintf(stdout, "done with SequenceItem\n");
  }
};

enum {
  MASK_TF = 0, MASK_LIB = 1
};

enum {
  imageLoadAllE, 
  bkgWorkE, 
  SeparatorWorkE, 
  imageInitBkgModel, 
  MULTI_FLOW_REGIONAL_FIT, 
  INITIAL_FLOW_BLOCK_ALLBEAD_FIT, 
  INITIAL_FLOW_BLOCK_REMAIN_REGIONAL_FIT, 
  SINGLE_FLOW_FIT, 
  POST_FIT_STEPS
  //imageLoadAllE, bkgWorkE, MULTIFLOWFIT, SeparatorWorkE, imageInitBkgModel, SINGLEFLOWFIT, POSTPROCESS_SINGLE_FLOW
};



#endif // SPECIALDATATYPES_H
