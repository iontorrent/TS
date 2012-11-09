/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BarcodeDatasets.h
//! @ingroup  BaseCaller
//! @brief    BarcodeDatasets. Manage datasets and associated barcode information.

#ifndef BARCODEDATASETS_H
#define BARCODEDATASETS_H

#include "OptArgs.h"

#include <string>
#include <vector>
#include <map>
#include "json/json.h"

using namespace std;


class BarcodeDatasets {
public:
  BarcodeDatasets(OptArgs& opts, const string& run_id);
  BarcodeDatasets(const string& run_id);
  ~BarcodeDatasets();

  void InitializeNonbarcoded(const string& run_id);
  void InitializeFromBarcodeList(const string& run_id, string barcode_list_filename);
  void LoadJson(const string& filename_json);
  void SaveJson(const string& filename_json);

  void GenerateFilenames(const string& file_type, const string& file_extension);

  static void PrintHelp();

  bool has_barcodes() const { return num_barcodes_ > 0; }
  int  num_barcodes() const { return num_barcodes_; }

  int num_datasets() const { return num_datasets_; }

  Json::Value&    dataset(int idx) { return datasets_json_["datasets"][idx]; }

  Json::Value&    read_groups() { return datasets_json_["read_groups"]; }
  vector<string>  read_group_names() { return datasets_json_["read_groups"].getMemberNames(); }
  Json::Value&    read_group(const string& read_group_name) { return datasets_json_["read_groups"][read_group_name]; }

  int num_read_groups() const { return num_read_groups_; }
  Json::Value&    read_group(int idx) { return datasets_json_["read_groups"][read_group_id_to_name_[idx]]; }
  int read_group_name_to_id(const string& rg_name) { return read_group_name_to_id_[rg_name]; }

  // map read_group_name -> dataset_idx
  // map read_group_idx -> dataset_idx


  Json::Value  barcode_config() const { return datasets_json_.get("barcode_config",Json::objectValue); }
  Json::Value&  json() { return datasets_json_; }

protected:

  void  EnumerateReadGroups();

  Json::Value               datasets_json_;
  int                       num_barcodes_;
  int                       num_datasets_;
  int                       num_read_groups_;

  vector<string>            read_group_id_to_name_;
  map<string,int>           read_group_name_to_id_;

};


#endif // BARCODEDATASETS_H
