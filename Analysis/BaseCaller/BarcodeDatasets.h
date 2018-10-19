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
  BarcodeDatasets(const string& run_id, const string input_datasets_file="");
  //BarcodeDatasets(const string& run_id);
  ~BarcodeDatasets();

  void InitializeNonbarcoded(const string& run_id);
  void LoadJsonFile(const string& filename_json);
  void LoadJson(const Json::Value& datasets_json, string data_source);
  void SaveJson(const string& filename_json);

  void GenerateFilenames(const string& bead_type, const string& file_type, const string& file_extension, const string& output_directory);

  // Remove duplicate Barcodes between dataset & control dataset
  void RemoveControlBarcodes(const Json::Value& control_datasets_json);

  void SetTF(bool process_tfs);
  void SetIonControl(const string& run_id);
  static void PrintHelp();

  bool has_barcodes() const { return num_barcodes_ > 0; }
  int  num_barcodes() const { return num_barcodes_; }

  int num_datasets() const { return num_datasets_; }

  Json::Value&    dataset(int idx) { return datasets_json_["datasets"][idx]; }
  Json::Value&    barcode_filters() { return datasets_json_["barcode_filters"]; };

  Json::Value&    read_groups() { return datasets_json_["read_groups"]; }
  vector<string>  read_group_names() { return datasets_json_["read_groups"].getMemberNames(); }
  Json::Value&    read_group(const string& read_group_name) { return datasets_json_["read_groups"][read_group_name]; }

  int num_read_groups() const { return num_read_groups_; }
  Json::Value&    read_group(int idx) { return datasets_json_["read_groups"][read_group_id_to_name_[idx]]; }
  int read_group_name_to_id(const string& rg_name) { return read_group_name_to_id_[rg_name]; }
  const string& read_group_name(int idx) { return read_group_id_to_name_[idx]; }
  const string& start_barcode_name(int idx) { return start_barcode_names_.at(idx); };

  int  GetBCmaxFlows()    const     { return barcode_max_flows_; }
  void SetBCmaxFlows(int max_flows) { barcode_max_flows_ = max_flows; }
  bool DatasetInUse()     const     { return dataset_in_use_; }
  bool IsControlDataset() const     { return control_dataset_; }
  bool IsLibraryDataset() const     { return (dataset_in_use_ and not(control_dataset_ or tf_dataset_)); }

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
  int                       barcode_max_flows_;
  bool                      dataset_in_use_;         //!< Flag indicating whether any output is written for this dataset
  bool                      control_dataset_;        //!< Flag indicating whether this is a control dataset
  bool                      tf_dataset_;             //!< Flag that indicates whetehr this is a TF dataset

  vector<string>            read_group_id_to_name_;
  map<string,int>           read_group_name_to_id_;
  vector<string>            start_barcode_names_;

};


#endif // BARCODEDATASETS_H
