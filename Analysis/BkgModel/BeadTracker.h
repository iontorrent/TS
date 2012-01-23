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
#include <vector>
#include "BeadParams.h"
#include "DiffEqModel.h"
#include "Mask.h"
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
    bead_params *params_high;
    bead_params *params_low;
    bead_params *params_nn;

    std::vector<float> ppf;
    std::vector<float> ssq;

    SequenceItem *seqList;
    int numSeqListItems;

    int DEBUG_BEAD;
    
    // a property of the collection of beads
    // spatially oriented bead data
    // not ideally located
    // only used for xtalk
    int     *ndx_map;
    int *key_id; // index into key map for beads
    
    
    BeadTracker();
    ~BeadTracker();
    void Delete();
    void InitBeadList(Mask *bfMask, Region *region, SequenceItem *_seqList, int _numSeq, const std::set<int>& sample);
    void InitBeadParams();
    void InitRandomSample(Mask& bf, Region& region, const std::set<int>& sample);
    void InitModelBeadParams();
    void InitLowBeadParams();
    void InitHighBeadParams();
    void SelectDebugBead();
    void DefineKeySequence(SequenceItem *seq_list, int number_Keys);
    // utilities for optimization
    void AssignEmphasisForAllBeads(int max_emphasis);
    void LimitBeadEvolution(bool first_block, float R_change_max, float copy_change_max);
    float KeyNormalizeReads();
    float KeyNormalizeOneBead(int ibd);
    void SelectKeyFlowValuesFromBeadIdentity(int *keyval, float *observed, int my_key_id, int keyLen);
    void SelectKeyFlowValues(int *keyval,float *observed, int keyLen);
    void ResetFlowParams(int bufNum, int flow);
    void LockKeyFlows(int ibd);
    void UnLockKeyFlows(int ibd);
    void LockAllBeadKeys();
    void UnLockAllBeadKeys();
    void AllBeadAverageError();
    int UpdateClonalFilter();
    int FindClonalReads();
    float FindMeanDmult();
    void RescaleDmult(float scale);
    float CenterDmult();
    void RescaleRatio(float scale);
    void MeanResidualByFlow(float *flow_res_mean);
    void RescaleResidualByMeanPerFlow(float *flow_res_mean);
    void DetectCorrupted();
    // likely not the right place for this, but does iterate over beads
    int FindKeyID(Mask *bfMask, int ax, int ay);
    void AssignKeyID(Region *region, Mask *bfmask);
    void BuildBeadMap(Region *region, Mask *bfmask,MaskType &process_mask);
    void  WriteCorruptedToMask(Region* region, Mask* bfmask);
   void DumpBeads(FILE *my_fp, bool debug_only,int offset_col, int offset_row);
   void DumpAllBeadsCSV(FILE *my_fp);
};

#endif // BEADTRACKER_H
