/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BARCODETRACKER_H
#define BARCODETRACKER_H

#include <set>
#include <vector>
#include "BkgMagicDefines.h"
#include "SeqList.h"
#include "CommandLineOpts.h"
#include "json/json.h"
#include <boost/algorithm/string/predicate.hpp>
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


class BarcodeTracker{
public:
  string barcode_file_name;
  bool defer_processing;
  vector<string> my_codes;

  std::vector<SequenceItem> barcode_data;
  int consistentFlows;
  int maxFlows;
  std::vector<int> barcode_id; // index into barcode data per bead

  // track some useful elements
  std::vector< vector <float> > barcode_bias;
  std::vector<int> barcode_count;
  void ResetCounts();

  BarcodeTracker();
  void SetupEightKeyNoT(const char *letter_flowOrder);
  void ReadFromFile(string &barcode_file);
  void SetupBarcodes(vector<string> &barcode_plus_key, const char *letter_flowOrder);
  void SetupLoadedBarcodes(const char *letter_flowOrder);
  int ClassifyOneBead(float *Ampl, double basic_threshold, double second_threshold, int flow_block_size, int flow_block_start);
  float ComputeNormalizerBarcode (const float *observed, int ib, int flow_block_size, int flow_block_start);
  int SetBarcodeFlows(float *observed, int ib);
  void ReportClassificationTable(int dummy_tag);

private:
   // Boost serialization support:
   friend class boost::serialization::access;
   template<class Archive>
     void serialize(Archive& ar, const unsigned int version)
     {
       ar &barcode_file_name
        &consistentFlows
           & maxFlows
       & barcode_data
        & barcode_id
            & my_codes;
     }
};


#endif // BARCODETRACKER_H
