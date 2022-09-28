/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BarcodeClassifier.h
//! @ingroup  BaseCaller
//! @brief    BarcodeClassifier. Barcode detection and trimming for BaseCaller

#ifndef BARCODECLASSIFIER_H
#define BARCODECLASSIFIER_H


#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "json/json.h"

#include "BarcodeDatasets.h"
#include "BaseCallerUtils.h"
#include "OptArgs.h"
#include "OrderedDatasetWriter.h"
#include "Mask.h"
#include "DPTreephaser.h"

using namespace std;

// ------------------------------------------------------------------------

struct Handle {
  string base_seq;
  vector<int> flow_seq;
};


// ------------------------------------------------------------------------

struct Barcode {
  int           mask_index;
  int           read_group_index;
  vector<int>   flow_seq;     // flow-space vector representation for the barcode
  int           num_flows;    // number of flows for the flow-space representation, includes 5' adapter
  int           end_flow;     // inclusive end flow (closed interval)
  int           adapter_start_flow;
  string        full_barcode;
  vector<float> predicted_signal;
  int           last_homopolymer;
};


class BarcodeClassifier {
public:

  BarcodeClassifier(OptArgs& opts, BarcodeDatasets& datasets, const ion::FlowOrder& flow_order,
        const vector<KeySequence>& keys, const string& output_directory,
        int chip_size_x, int chip_size_y, const Json::Value& structure);

  ~BarcodeClassifier();

  void BuildPredictedSignals(const ion::FlowOrder& flow_order, float cf, float ie, float dr);

  static void PrintHelp();

  bool has_barcodes() const { return num_barcodes_ > 0; }
  int  num_barcodes() const { return num_barcodes_; }

  void SetClassificationParams(int mode, double cutoff, double separation);

  int  SimpleBaseSpaceClassification(const BasecallerRead& basecaller_read);

  int  FlowAlignClassification(const ProcessedRead &processed_read, const vector<int>& base_to_flow, int& best_errors);

  int  SignalSpaceClassification(const BasecallerRead& basecaller_read, float& best_distance, int& best_errors,
                                 vector<float>& best_bias, bool& filtered_zero_errors, int& org_read_group);

  int  ProportionalSignalClassification(const BasecallerRead& basecaller_read, float& best_distance, int& best_errors,
                                 vector<float>& best_bias, bool& filtered_zero_errors, int& org_read_group);

  bool AdapterValidation(const BasecallerRead& basecaller_read, int best_barcode, bool& filtered_read);

  void ClassifyAndTrimBarcode(int read_index, ProcessedRead &processed_read, const BasecallerRead& basecaller_read, const vector<int>& base_to_flow);

  bool MatchesBarcodeSignal(const BasecallerRead& basecaller_read);

  int  NoBarcodeReadGroup() const { return no_barcode_read_group_; };

  bool TrimBarcodes() const { return trim_barcodes_; };

  Mask* GetBarcodeMaskPointer() { return &barcode_mask_; };

  void Close(BarcodeDatasets& datasets, int num_end_barcodes);

protected:

  //void PhysicalWriteRegion(int region);

  // Computes the barcode set's Hamming distance in flow space
  void ComputeHammingDistance();

  // Transfer barcode information from dataset structure to class structure
  void LoadBarcodesFromDataset(BarcodeDatasets& datasets, const vector<KeySequence>& keys, const ion::FlowOrder& flow_order);

  void LoadHandlesFromArgs(OptArgs& opts, const ion::FlowOrder& flow_order, const Json::Value& structure);

  void ClassifyAndTrimHandle(int read_index,
                             int best_barcode,
                             ProcessedRead &processed_read,
                             const BasecallerRead& basecaller_read,
                             const vector<int>& base_to_flow);

  int  HandleBaseSpaceClassification(const BasecallerRead& basecaller_read, const ProcessedRead &processed_read);

  int  HandleFlowAlign(int best_barcode, ProcessedRead &processed_read, const vector<int>& base_to_flow);


  template <class T>
  bool CheckParameterLowerUpperBound(string identifier ,T &parameter, T lower_limit, int use_lower, T upper_limit, int use_upper, T default_val) {

    if (not check_limits_)
      return true;

	bool lower_ok = true;
	bool upper_ok = true;
	std::stringstream strstream;

	// Use of use_lower and use_upper: 0: no limit given; 1: inclusive limit, 2: exclusive limit
	if (use_lower > 0)
      lower_ok = (use_lower==1) ? (parameter >= lower_limit) : (parameter > lower_limit);
	if (use_upper > 0)
      upper_ok = (use_upper==1) ? (parameter <= upper_limit) : (parameter < upper_limit);
    bool is_ok = lower_ok and upper_ok;

    if (not is_ok) {
      string LL_eq = "";
      if (use_lower > 0)
        LL_eq = (use_lower==1) ? " <= " : " < ";
      string UL_eq = "";
      if (use_upper > 0)
        UL_eq = (use_upper==1) ? " <= " : " < ";

      strstream << "   WARNING in BarcodeClassifier: Parameter " << identifier << "(" << parameter << ") not within limits: ";
      if (use_lower > 0)
    	  strstream << lower_limit << LL_eq;
      strstream << identifier;
      if (use_upper > 0)
    	  strstream << UL_eq << upper_limit;
      strstream << ". Using value of " << default_val;
      parameter = default_val;
      // Write to both cout and cerr if we change the value of a parameter
      cout << strstream.str() << endl;
      cerr << strstream.str() << endl;
    }
    return is_ok;
  }

  // --- Variables

  Mask                      barcode_mask_;
  int                       num_barcodes_;

  string                    barcode_list_file_;
  string                    barcode_mask_filename_;
  int                       barcode_min_start_flow_;     // Minimum start flow over all barcodes
  int                       barcode_max_flows_;          // The maximum number of barcode flows over all barcodes
  int                       barcode_full_flows_;         // The number of overall barcode flows, if synchronized
  bool                      end_flow_synch_;             // Are end barcode end flows synchronized?
  int                       barcode_max_hp_;             // The largest homopolymer in the barcodes
  double                    barcode_filter_;             // Barcode frequency cutoff filter
  int                       barcode_filter_minreads_;    // Minimum number of reads per barcode group
  double                    barcode_error_filter_;       // Filter barcodes basedon the average number of errors
  string                    barcode_filter_filename_;    // Output summary filename
  int                       barcode_filter_postpone_;    // Switch to not filter / weigh filtering down per block 0: filter 1: don't filter 2: pre-filter
  double                    barcode_filter_weight_;      // Weighting factor for per-block-stringency of frequncy filter
  bool                      barcode_filter_named_;       // Exclude barcodes with an associated sample name from filtering
  bool                      barcode_ignore_flows_;       // Switch telling the classifier to exclude certain flows from classification
  bool                      trim_barcodes_;              // Switch to trim or leave barcodes alone
  bool                      have_ambiguity_codes_;       // Do we have non-ACGT characters in key, barcode?
  bool                      barcode_clip_measurements_;   // Restrict measurements to the expected interval?
  vector<int>               classifier_ignore_flows_;    // Specifying an interval of flows to exclude from classification

  int                       hamming_dmin_;
  int                       score_mode_;
  double                    score_cutoff_;
  double                    score_separation_;
  bool                      score_auto_config_;
  float                     adapter_cutoff_;              // Maximum allowed per-flow squared residual for the adapter

  vector<Barcode>           barcode_;
  int                       no_barcode_read_group_;       // Index of the non-barcoded read group
  bool                      dataset_in_use_;              // Indicated whether any action should be performed for this dataset
  bool                      is_control_dataset_;           // Indicates a set of control barcodes

  int                       windowSize_;                  // Normalization window size for Treephaser
  bool                      skip_droop_;                  // Switch to let basecaller skip droop
  bool						barcode_bam_tag_;             // Add the barcode tag to output bam
  bool                      check_limits_;                // Check whether command line arguments are within reasonable bounds

  // Variables associated with handle identification & trimming

  bool                      have_handles_;
  vector<string>            handle_sequence_;
  vector<Handle>            handle_;
  int                       handle_mode_;
  int                       handle_cutoff_;

};

// ------------------------------------------------------------------------

class EndBarcodeClassifier {

  struct EndBarcode {
    string        barcode_name;
    string        barcode_sequence;
    string        barcode_adapter;
    int           bc_start_hp;
    int           adapter_start_hp;
    bool          analyze_as_single;

    EndBarcode();
  };

  bool                      enable_barcodes_;
  bool                      have_end_barcodes_;
  bool                      trim_barcodes_;

  int                       nomatch_read_group_;
  vector<EndBarcode>        read_group_;
  vector<EndBarcode>        end_barcodes_;
  vector<string>            end_barcode_names_;
  vector<int>               bead_adapter_start_hp;

  unsigned int              num_end_barcodes_;
  bool                      demux_barcode_list_;

  const ion::FlowOrder*     flow_order_p_;
  Mask*                     barcode_mask_pointer_;

  int                       score_mode_;
  int                       score_cutoff_;
  int                       score_separation_;
  int                       adapter_cutoff_;

  bool                      have_handles_;
  vector<string>            handle_sequence_;
  int                       handle_mode_;
  int                       handle_cutoff_;
  bool                      handle_filter_;

public:

  EndBarcodeClassifier(
           OptArgs&           opts,
           BarcodeDatasets&   datasets,
           ion::FlowOrder*    flow_order,
           Mask*              MaskPointer,
           const Json::Value& structure);

  static void PrintHelp();

  void AddBarcode(EndBarcode& barcode,
                  const string& bc_sequence,
                  const string& bc_adapter,
                  bool          as_single);


  bool LoadBarcodesFromCSV(string filename);

  bool CreateBarcodeListFromDatasets(BarcodeDatasets& datasets, bool enable_rna_barcodes);

  void LoadHandlesFromArgs(OptArgs& opts, const Json::Value& structure);

  int  GetStartHP(const string& base_str);

  void ClassifyAndTrimBarcode(
           int read_index,
           ProcessedRead&        processed_read,
           const BasecallerRead& basecaller_read,
           const vector<int>&    base_to_flow);

  void UpdateReadTrimming(
           ProcessedRead &processed_read,
           int trim_n_bases,
           const string& YK_tag);

  void PushReadToNomatch(
           int read_index,
           ProcessedRead &processed_read);

  int BaseSpaceClassification(
           ProcessedRead &processed_read,
           int& n_bases_barcode,
           int& bc_start_flow,
           const BasecallerRead& basecaller_read,
           const vector<int>& base_to_flow);

  int FlowAlignClassification(
           ProcessedRead &processed_read,
           int& n_bases_barcode,
           int& best_bc_start_flow,
           int  adapter_hp,
           const BasecallerRead& basecaller_read,
           const vector<int>& base_to_flow);

  bool TrimHandle(
           int  adapter_flow,
           int  adapter_hp,
           int  temp_end_base,
           const BasecallerRead& basecaller_read,
           const vector<int>& base_to_flow,
           ProcessedRead &processed_read,
           int& n_bases_handle,
           int& handle_start_flow);

  bool BaseSpaceMatch(
           int                   n_bases_prefix,
           int                   n_bases_filtered,
           const BasecallerRead& basecaller_read,
           const string &        query);

  vector<int> BaseToFlow(
           const string &query,
           int   flow);

  void ReverseFlowAlignDistance(
           const vector<int>& base_to_flow,
           const string& my_query,
           int   prefix_flow,
           int   adapter_flow,
           int   adapter_hp,
           int&  best_start_flow,
           int&  best_distance,
           int&  best_read_bases);

  int NumEndBarcodes() { return num_end_barcodes_; };
  int NoMatchReadGroup() { return nomatch_read_group_; }

  const vector<string>& EndBarcodeNames() { return end_barcode_names_; }

};


#endif // BARCODECLASSIFIER_H
