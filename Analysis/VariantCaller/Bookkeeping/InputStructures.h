/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     InputStructures.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef INPUTSTRUCTURES_H
#define INPUTSTRUCTURES_H

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <pthread.h>
#include <Variant.h>
#include <VariantCaller/OrderedBAMWriter.h>

#include "LinearCalibrationModel.h"
#include "../Splice/ErrorMotifs.h"
#include "ExtendParameters.h"
#ifdef __SSE3__
#include "TreephaserSSE.h"
#endif
#include "DPTreephaser.h"
#include "Realigner.h"

using namespace std;
using namespace BamTools;
using namespace ion;


class InputStructures;
class ReferenceReader;
class TargetsManager;
class BAMWalkerEngine;
class AlleleParser;
class OrderedVCFWriter;
class SampleManager;
class MetricsManager;
class IndelAssembly;
class MolecularTagManager;
// ==============================================================================

struct VariantCallerContext {

  ExtendParameters *    parameters;                   //! Raw parameters, retrieved from command line and json
  InputStructures *     global_context;               //! Miscellaneous data structures
  ReferenceReader *     ref_reader;                   //! Memory-mapped reference reader
  TargetsManager *      targets_manager;              //! Manages target regions from BED file
  BAMWalkerEngine *     bam_walker;                   //! Manages traversing reference and retrieving reads covering each position
  AlleleParser *        candidate_generator;          //! Candidate variant generator
  OrderedBAMWriter *    bam_writer;                   //! Container for reads ensuring ordering
  OrderedVCFWriter *    vcf_writer;                   //! Sorting, threading friendly VCF writer
  MetricsManager *      metrics_manager;              //! Keeps track of metrics to output in tvc_metrics.json
  SampleManager*        sample_manager;               //! Tracks the sample used in multi-sample analysis
  IndelAssembly*        indel_assembly;               //! Outputs the indel_assembly.vcf for long indels
  MolecularTagManager*  mol_tag_manager;              //! Manager for molecular tags

  pthread_mutex_t       bam_walker_mutex;             //! Mutex for state-altering bam_walker operations
  pthread_mutex_t       read_loading_mutex;           //! Mutex for raw read retrieval
  pthread_mutex_t       candidate_generation_mutex;   //! Mutex for candidate generation
  pthread_mutex_t       read_removal_mutex;           //! Mutex for removal/write of used reads
  pthread_cond_t        memory_contention_cond;       //! Conditional variable for preventing memory contention
  pthread_cond_t        alignment_tail_cond;          //! Conditional variable for blocking until the final alignments are processed

  int                   candidate_counter;            //! Number of candidates generated so far
  //int                 candidate_dot;                //! Number of candidates that will trigger printing next "."
  time_t                dot_time;                     //! Time that will trigger printing next '.'
};

// ==============================================================================

struct VariantSpecificParams {
  VariantSpecificParams() :
      min_allele_freq_override(false), min_allele_freq(0),
      strand_bias_override(false), strand_bias(0),
      strand_bias_pval_override(false), strand_bias_pval(0),

      min_coverage_override(false), min_coverage(0),
      min_coverage_each_strand_override(false), min_coverage_each_strand(0),
	  min_var_coverage_override(false), min_var_coverage(0),
      min_variant_score_override(false), min_variant_score(0),
      data_quality_stringency_override(false), data_quality_stringency(0),

      position_bias_override(false), position_bias(0),
      position_bias_pval_override(false), position_bias_pval(0),

      hp_max_length_override(false), hp_max_length(0),
      filter_unusual_predictions_override(false), filter_unusual_predictions(0),
      filter_insertion_predictions_override(false), filter_insertion_predictions(0),
      filter_deletion_predictions_override(false), filter_deletion_predictions(0),
      sse_prob_threshold_override(false), sse_prob_threshold(0),
	  min_tag_fam_size_override(false), min_tag_fam_size(0),
      black_strand('.') {}

  bool  min_allele_freq_override;
  float min_allele_freq;
  bool  strand_bias_override;
  float strand_bias;
  bool  strand_bias_pval_override;
  float strand_bias_pval;
  bool  min_coverage_override;
  int   min_coverage;
  bool  min_coverage_each_strand_override;
  int   min_coverage_each_strand;
  bool  min_var_coverage_override;
  int   min_var_coverage;
  bool  min_variant_score_override;
  float min_variant_score;
  bool  data_quality_stringency_override;
  float data_quality_stringency;

  bool  position_bias_override;
  float position_bias;
  bool position_bias_pval_override;
  float position_bias_pval;

  bool  hp_max_length_override;
  int   hp_max_length;
  bool  filter_unusual_predictions_override;
  float filter_unusual_predictions;
  bool  filter_insertion_predictions_override;
  float filter_insertion_predictions;
  bool  filter_deletion_predictions_override;
  float filter_deletion_predictions;
  bool  sse_prob_threshold_override;
  float sse_prob_threshold;
  bool min_tag_fam_size_override;
  int  min_tag_fam_size;
  char  black_strand;
};

// ==============================================================================

struct VariantCandidate {
  VariantCandidate(vcf::VariantCallFile& initializer) : variant(initializer), position_upper_bound(0) {}
  vcf::Variant variant;
  vector<VariantSpecificParams> variant_specific_params;
  int position_upper_bound;   // Prevents SNP healing from accidentally reordering vcf records
};


// ==============================================================================


class RecalibrationHandler{
  public:
    bool use_recal_model_only;
    bool is_live;
    LinearCalibrationModel recalModel;
    
    map<string, LinearCalibrationModel> bam_header_recalibration; // look up the proper recalibration handler by run id + block coordinates
    multimap<string,pair<int,int> > block_hash;  // from run id, find appropriate block coordinates available
    
 void ReadRecalibrationFromComments(const SamHeader &samHeader, const map<string, int> &max_flows_by_run_id);
 
//  vector<vector<vector<float> > > * getAs(string &found_key, int x, int y){return(recalModel.getAs(x,y));};
//  vector<vector<vector<float> > > * getBs(string &found_key, int x, int y){return(recalModel.getBs(x,y));};
  void getAB(MultiAB &multi_ab, const string &found_key, int x, int y) const;
  
  bool recal_is_live() const { return(is_live); };
  string FindKey(const string &runid, int x, int y) const;
  
  RecalibrationHandler(){use_recal_model_only = false; is_live = false; };
};

// ==============================================================================
//Input Structures

class InputStructures {
public:


  // Key and barcode sequence are read group specific
  // whereas flow order and recalibration are chip specific
  vector<ion::FlowOrder >   flow_order_vector;
  map<string, int>          flow_order_index_by_run_id;
  map<string, int>          num_flows_by_run_id;
  map<string, string>       key_by_read_group;

  bool                      use_SSE_basecaller;
  bool                      resolve_clipped_bases;
  int                       DEBUG;

  // Reusable objects
  TIonMotifSet              ErrorMotifs;
  RecalibrationHandler      do_recal;


  InputStructures();
  void Initialize(ExtendParameters &parameters, const ReferenceReader& ref_reader,
      const SamHeader &bam_header);
  void read_error_motifs(string & fname){ErrorMotifs.load_from_file(fname.c_str());};
  void DetectFlowOrderzAndKeyFromBam(const SamHeader &samHeader);
};

// ==============================================================================

// A collections of objects that are shared and reused throughout the execution of one tread
class PersistingThreadObjects {
public:

    PersistingThreadObjects(const InputStructures &global_context);
    ~PersistingThreadObjects() {
//#ifdef __SSE3__
//        for (vector<TreephaserSSE*>::iterator iter = treephaserSSE_vector.begin(); (iter != treephaserSSE_vector.end()); ++iter) {
//            delete *iter;
//            *iter = NULL;
//        } 
//#endif
    };

    //@brief Interface for setting the phasing model parameters
    void SetModelParameters(const int & flow_order_index, const vector<float> & phase_params) {
#ifdef __SSE3__
      if (use_SSE_basecaller)
        treephaserSSE_vector.at(flow_order_index).SetModelParameters(phase_params.at(0), phase_params.at(1));
      else
#endif
        dpTreephaser_vector.at(flow_order_index).SetModelParameters(phase_params.at(0), phase_params.at(1), phase_params.at(2));
    };

    //@brief Interface for setting the phasing model parameters
    void DisableRecalibration(const int & flow_order_index)
    {
#ifdef __SSE3__
       if (use_SSE_basecaller)
         treephaserSSE_vector.at(flow_order_index).DisableRecalibration();
       else
#endif
         dpTreephaser_vector.at(flow_order_index).DisableRecalibration();
    };

    //@brief Interface set the calibration coefficients
    bool SetAsBs(const int & flow_order_index, const vector<vector< vector<float> > > *As, const vector<vector< vector<float> > > *Bs)
    {
#ifdef __SSE3__
      if (use_SSE_basecaller)
        return (treephaserSSE_vector.at(flow_order_index).SetAsBs(As, Bs));
      else
#endif
        return (dpTreephaser_vector.at(flow_order_index).SetAsBs(As, Bs));
    };

    //@brief Interface for simulating and solving read bases
    void SolveRead(const int & flow_order_index, BasecallerRead& read, const int & begin_flow, const int & end_flow){
#ifdef __SSE3__
      if (use_SSE_basecaller)
        treephaserSSE_vector.at(flow_order_index).SolveRead(read, begin_flow, end_flow);
      else
#endif
        dpTreephaser_vector.at(flow_order_index).Solve(read, end_flow, begin_flow);
    };


    bool                   use_SSE_basecaller;    // Switch that tells us which basecaller is active

    Realigner              realigner;             // realignment tool
    vector<DPTreephaser >  dpTreephaser_vector;   // c++ treephaser
#ifdef __SSE3__
    vector<TreephaserSSE>  treephaserSSE_vector;  // vectorized treephaser
#endif






};





#endif //INPUTSTRUCTURES_H
