/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BEADTRACKER_H
#define BEADTRACKER_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <set>
#include <ostream>
#include <vector>
#include "BeadParams.h"
#include "DiffEqModel.h"
#include "Mask.h"
#include "PinnedInFlow.h"
#include "Region.h"
#include "SpecialDataTypes.h"
#include "GlobalDefaultsForBkgModel.h"  //to get the flow order 
#include "MaskSample.h"

// @TODO unfortunately this structure cannot be used at this time
// CUDA code is incompatible with this structure and difficult to change.
struct bead_box{
    bead_params nn;
    bead_params low;
    bead_params hi;
};

float ComputeNormalizerKeyFlows(const float *observed, const int *keyval, int len);

// wrap a bunch of bead tracking into one place
class BeadTracker{
public:
    // bead parameters
    int  numLBeads;
    int  numLBadKey;
    bound_params *params_high;
    bound_params *params_low;
    bead_params  *params_nn;

    SequenceItem *seqList;
    int numSeqListItems;

    int DEBUG_BEAD;
    int max_emphasis; // current maximum emphasis vector (?)
    
    // a property of the collection of beads
    // spatially oriented bead data
    // not ideally located
    // only used for xtalk
    int *ndx_map;
    int *key_id; // index into key map for beads

    // track quality for use in regional parameter fitting
    // aggregate 'clonal', 'corrupt', 'high copy number' states
    bool *high_quality;
    float my_mean_copy_count;

    int regionindex; // for debugging
    
    BeadTracker();
    ~BeadTracker();
    void Delete();

    void  InitBeadList(Mask *bfMask, Region *region, SequenceItem *_seqList, int _numSeq, const std::set<int>& sample);
    void  InitBeadParams();
    void  InitBeadParamR(Mask *bfMask, Region *region, std::vector<float> *tauB,std::vector<float> *tauE);
    void  InitQualityTracker();
    void  InitRandomSample(Mask& bf, Region& region, const std::set<int>& sample);
    void  InitModelBeadParams();

    void  InitLowBeadParams();
    void  InitHighBeadParams();
    void  SelectDebugBead();
    void  DefineKeySequence(SequenceItem *seq_list, int number_Keys);
    // utilities for optimization
    void  AssignEmphasisForAllBeads(int max_emphasis);
    void  LimitBeadEvolution(bool first_block, float R_change_max, float copy_change_max);
    float KeyNormalizeReads(bool overwrite_key);
    float KeyNormalizeOneBead(int ibd, bool overwrite_key);
    float ComputeSSQRatioOneBead(int ibd);
    void  SelectKeyFlowValuesFromBeadIdentity(int *keyval, float *observed, int my_key_id, int &keyLen);
    void  SelectKeyFlowValues(int *keyval,float *observed, int keyLen);
    void  ResetFlowParams(int bufNum);
    void  ResetLocalBeadParams();

    void  UpdateClonalFilter(int flow, const std::vector<float>& copy_multiplier);
    void  FinishClonalFilter();
    int   NumHighPPF() const;
    int   NumPolyclonal() const;
    int   NumBadKey() const;
    float FindMeanDmult(bool skip_beads);
    void  RescaleDmult(float scale);
    float CenterDmult(bool skip_beads);
    void  RescaleRatio(float scale);

    // likely not the right place for this, but does iterate over beads
    int  FindKeyID(Mask *bfMask, int ax, int ay);
    void AssignKeyID(Region *region, Mask *bfmask);
    void BuildBeadMap(Region *region, Mask *bfmask,MaskType &process_mask);
    void WriteCorruptedToMask(Region* region, Mask* bfmask);
    void ZeroOutPins(Region *region, Mask *bfmask, PinnedInFlow &pinnedInFlow, int flow, int iFlowBuffer);
    void DumpBeads(FILE *my_fp, bool debug_only, int offset_col, int offset_row);
    void DumpAllBeadsCSV(FILE *my_fp);

    void LowCopyBeadsAreLowQuality ( float mean_copy_count);
    void LowSSQRatioBeadsAreLowQuality ( float snr_threshold);
    void CorruptedBeadsAreLowQuality ();
    void TypicalBeadParams(bead_params *p);

    void CompensateAmplitudeForEmptyWellNormalization(float *my_scale_buffer);
    //void DumpHits(int offset_col, int offset_row, int flow);

private:
    void CheckKey(const std::vector<float>& copy_multiplier);
    void ComputeKeyNorm(const std::vector<int>& keyIonogram, const std::vector<float>& copy_multiplier);
    void AdjustForCopyNumber(std::vector<float>& ampl, const bead_params& p, const std::vector<float>& copy_multiplier);
    void UpdatePPFSSQ(int flow, const std::vector<float>& copy_multiplier);
};

#endif // BEADTRACKER_H
