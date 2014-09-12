/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BarcodeClassifier.h
//! @ingroup  BaseCaller
//! @brief    BarcodeClassifier. Barcode detection and trimming for BaseCaller

#ifndef BARCODECLASSIFIER_H
#define BARCODECLASSIFIER_H


#include <string>
#include <vector>
#include <map>
#include "json/json.h"

#include "BarcodeDatasets.h"
#include "BaseCallerUtils.h"
#include "OptArgs.h"
#include "OrderedDatasetWriter.h"
#include "Mask.h"
#include "DPTreephaser.h"

using namespace std;

struct Barcode {
  int           mask_index;
  int           read_group_index;
  vector<int>   flow_seq;     // flow-space vector representation for the barcode
  int           num_flows;    // number of flows for the flow-space representation, includes 5' adapter
  //int           start_flow;   // calculated from the start base & end base, used for scoring/matching
  int           end_flow;     // inclusive end flow (closed interval)
  int           adapter_start_flow;
  string        full_barcode;
  vector<float> predicted_signal;
  int           last_homopolymer;
};



class BarcodeClassifier {
public:

  BarcodeClassifier(OptArgs& opts, BarcodeDatasets& datasets, const ion::FlowOrder& flow_order,
      const vector<KeySequence>& keys, const string& output_directory, int chip_size_x, int chip_size_y);

  ~BarcodeClassifier();

  void BuildPredictedSignals(float cf, float ie, float dr);

  static void PrintHelp();

  bool has_barcodes() const { return num_barcodes_ > 0; }
  int  num_barcodes() const { return num_barcodes_; }

  void ClassifyAndTrimBarcode(int read_index, ProcessedRead &processed_read, const BasecallerRead& basecaller_read, const vector<int>& base_to_flow);

  void Close(BarcodeDatasets& datasets);

protected:

  void PhysicalWriteRegion(int region);

  template <class T>
  bool CheckParameterLowerUpperBound(string identifier ,T &parameter, T lower_limit, int use_lower, T upper_limit, int use_upper, T default_val) {
	bool lower_ok = true;
	bool upper_ok = true;
	// Use of use_lower and use_upper: 0: no limit given; 1: inclusive limit, 2: exclusive limit
	if (use_lower > 0)
      lower_ok = (use_lower==1) ? (parameter >= lower_limit) : (parameter > lower_limit);
	if (use_upper > 0)
      upper_ok = (use_upper==1) ? (parameter <= upper_limit) : (parameter < upper_limit);
    bool is_ok = lower_ok and upper_ok;

    if (is_ok)
      return is_ok;
    else{
      string LL_eq = "";
      if (use_lower > 0)
        LL_eq = (use_lower==1) ? " <= " : " < ";
      string UL_eq = "";
      if (use_upper > 0)
        UL_eq = (use_upper==1) ? " <= " : " < ";

      cerr << " WARNING in BarcodeClassifyer: Parameter " << identifier << "(" << parameter << ") not within limits: ";
      if (use_lower > 0)
        cerr << lower_limit << LL_eq;
      cerr << identifier;
      if (use_upper > 0)
        cerr << UL_eq << upper_limit;
      cerr << ". Using default value of " << default_val;
      cerr << endl;
      parameter = default_val;
    }
    return (is_ok);
  }


  ion::FlowOrder            flow_order_;
  int                       num_barcodes_;
  Mask                      barcode_mask_;

  string                    barcode_list_file_;
  string                    barcode_mask_filename_;
  int                       barcode_min_start_flow_;     // Minimum start flow over all barcodes
  int                       barcode_max_flows_;          // The maximum number of barcode flows over all barcodes
  int                       barcode_max_hp_;             // The largest homopolymer in the barcodes
  double                    barcode_filter_;             // Barcode frequency cutoff filter
  int                       barcode_filter_minreads_;    // Minimum number of reads per barcode group
  double                    barcode_error_filter_;       // Filter barcodes basedon the average number of errors
  string                    barcode_filter_filename_;    // Output summary filename
  int                       barcode_filter_postpone_;    // Switch to not filter / weigh filtering down per block 0: filter 1: don't filter 2: pre-filter
  double                    barcode_filter_weight_;      // Weighting factor for per-block-stringency of frequncy filter

  int                       score_mode_;
  double                    score_cutoff_;
  double                    score_separation_;

  vector<Barcode>           barcode_;

  int                       windowSize_;                  // Normalization window size for Treephaser
  bool                      skip_droop_;                   // Switch to let basecaller skip droop
  bool						barcode_bam_tag_;             // Add the barcode tag to output bam

  // Dummy variables for debugging
  int num_prints_;


public:
  int                       no_barcode_read_group_;

};


#endif // BARCODECLASSIFIER_H
