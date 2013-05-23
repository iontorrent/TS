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
  int           start_flow;   // calculated from the start base & end base, used for scoring/matching
  int           end_flow;
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

  ion::FlowOrder            flow_order_;
  int                       num_barcodes_;
  Mask                      barcode_mask_;

  string                    barcode_list_file_;
  string                    barcode_directory_;
  string                    barcode_mask_filename_;
  double                    barcode_filter_;
  string                    barcode_filter_filename_;

  int                       score_mode_;
  double                    score_cutoff_;
  double                    score_separation_;

  vector<Barcode>           barcode_;

  int                       windowSize_;

public:
  int                       no_barcode_read_group_;

};


#endif // BARCODECLASSIFIER_H
