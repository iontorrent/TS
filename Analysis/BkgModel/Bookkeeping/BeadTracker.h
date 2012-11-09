/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BEADTRACKER_H
#define BEADTRACKER_H

#include <set>
#include <vector>
#include "BkgMagicDefines.h"
#include "BeadParams.h"
#include "Mask.h"
#include "Region.h"
#include "SpecialDataTypes.h"
#include "GlobalDefaultsForBkgModel.h"  //to get the flow order 
#include "PinnedInFlow.h"

class LevMarBeadAssistant;

float ComputeNormalizerKeyFlows(const float *observed, const int *keyval, int len);

// wrap a bunch of bead tracking into one place
class BeadTracker{
public:
    // bead parameters
    int  numLBeads;
    int  numLBadKey;
    bound_params params_high;
    bound_params params_low;
    std::vector<bead_params>  params_nn;
    std::vector<SequenceItem> seqList;
    int numSeqListItems;

    int DEBUG_BEAD;
    int max_emphasis; // current maximum emphasis vector (?)
    
    // a property of the collection of beads
    // spatially oriented bead data
    // not ideally located
    // only used for xtalk
    std::vector<int> ndx_map;
    std::vector<int> key_id; // index into key map for beads

    // track quality for use in regional parameter fitting
    // aggregate 'clonal', 'corrupt', 'high copy number' states
    float                   my_mean_copy_count;
    std::vector<bool>       high_quality;
    std::vector<bead_state> all_status;

    // track beads sampled for use in regional parameter fitting
    std::vector<bool>  sampled;
    bool isSampled;

    int regionindex; // for debugging
    
    BeadTracker();

    void  InitBeadList(Mask *bfMask, Region *region, SequenceItem *_seqList, int _numSeq, const std::set<int>& sample, float AmplLowerLimit);
    void  InitBeadParams(float AmplLowerLimit);
    void  InitBeadParamR(Mask *bfMask, Region *region, std::vector<float> *tauB,std::vector<float> *tauE);
    void  InitRandomSample(Mask& bf, Region& region, const std::set<int>& sample);
    // Two use cases in the code when sampling is set
    // use only beads that are marked as Sampled
    // use only beads that are marked as StillSampled
    // which use in the same code is determined by skip_beads
    void  SetSampled();
    void  SystematicSampling();
    void  UpdateSampled( bool *well_completed);  
    void  ExcludeFromSampled( int ibd );
    bool  StillSampled(int ibd) { return( sampled[ibd] && high_quality[ibd] ); }
    bool  Sampled(int ibd) { return( sampled[ibd] ); }
    bool  BeadIncluded (int ibd, bool skip_beads) { return ( high_quality[ibd] || !skip_beads );} // if not skipping beads, ignore the high_quality flag

    void  InitModelBeadParams();

    void  InitLowBeadParams(float AmplLowerLimit);
    void  InitHighBeadParams();
    void  SelectDebugBead(int seed);
    void  DefineKeySequence(SequenceItem *seq_list, int number_Keys);
    // utilities for optimization
    void  AssignEmphasisForAllBeads(int max_emphasis);

    float KeyNormalizeReads(bool overwrite_key, bool sampled_only=false);
    float KeyNormalizeSampledReads(bool overwrite_key);
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
    float FindMeanDmultFromSample(bool skip_beads);
    float CenterDmultFromSample(bool skip_beads);

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

 private:
    // Boost serialization support:
    friend class boost::serialization::access;
    template<class Archive>
      void serialize(Archive& ar, const unsigned int version)
      {
	// fprintf(stdout, "Serialize BeadTracker...");
	ar & numLBeads;
	ar & numLBadKey;
	ar & params_high;
	ar & params_low;
	ar & all_status; // serialize all_status before params_nn
	ar & params_nn;  // params_nn points at bead_state objects in all_status
	ar & seqList;
	ar & numSeqListItems;
	ar & DEBUG_BEAD;
	ar & max_emphasis;
	ar & ndx_map;
	ar & key_id;
	ar & high_quality;
	ar & sampled;
	ar & isSampled;
	ar & my_mean_copy_count;
	ar & regionindex;
	// fprintf(stdout, "done with BeadTracker\n");
      }

};


#endif // BEADTRACKER_H
