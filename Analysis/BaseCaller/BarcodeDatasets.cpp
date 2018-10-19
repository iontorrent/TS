/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BarcodeDatasets.cpp
//! @ingroup  BaseCaller
//! @brief    BarcodeDatasets. Manage datasets and associated barcode information.

#include "BarcodeDatasets.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <errno.h>
#include <time.h>

#include "Utils.h"
#include "IonErr.h"


// -----------------------------------------------------------------------------------------

BarcodeDatasets::BarcodeDatasets(const string& run_id, const string input_datasets_file)
  : datasets_json_(Json::objectValue), num_barcodes_(0), num_datasets_(0), num_read_groups_(0),
    barcode_max_flows_(0), dataset_in_use_(false),  control_dataset_(false), tf_dataset_(false)
{
  if (not input_datasets_file.empty())
    LoadJsonFile(input_datasets_file);
  else
    InitializeNonbarcoded(run_id);
}


BarcodeDatasets::~BarcodeDatasets()
{
}

// -----------------------------------------------------------------------------------------

void BarcodeDatasets::LoadJsonFile(const string& filename_json)
{
  ifstream in(filename_json.c_str(), ifstream::in);
  Json::Value temp_datasets;

  if (!in.good()) {
    printf("Opening dataset file %s unsuccessful. Aborting\n", filename_json.c_str());
    exit(EXIT_FAILURE);
  }
  in >> temp_datasets;
  LoadJson(temp_datasets, filename_json);
}

void BarcodeDatasets::LoadJson(const Json::Value& datasets_json, string data_source)
{
  datasets_json_ = datasets_json;

  datasets_json_["meta"]["format_name"] = "Dataset Map";
  datasets_json_["meta"]["format_version"] = "1.0";
  datasets_json_["meta"]["generated_by"] = "BaseCaller";
  datasets_json_["meta"]["datasets_source"] = data_source;
  time_t now;
  time(&now);
  datasets_json_["meta"]["creation_date"] = get_time_iso_string(now);

  if(not datasets_json_.isMember("barcode_filters"))
    datasets_json_["barcode_filters"] = Json::objectValue;

  num_datasets_ = datasets_json_["datasets"].size();
  num_barcodes_ = 0;
  if (num_datasets_ == 0) {
    cerr << "BarcodeDatasets ERROR: dataset " << data_source << " does not contain any datasets." << endl;
    exit(EXIT_FAILURE);
  }
  EnumerateReadGroups();
  dataset_in_use_ = true;
}

// -----------------------------------------------------------------------------------------

// This initializer makes the class usable when no additional information is available.
// Essentially emulates current non-barcoded behavior

void BarcodeDatasets::InitializeNonbarcoded(const string& run_id)
{

  datasets_json_ = Json::objectValue;

  // Prepare metadata
  datasets_json_["meta"]["format_name"] = "Dataset Map";
  datasets_json_["meta"]["format_version"] = "1.0";
  datasets_json_["meta"]["generated_by"] = "BaseCaller";
  time_t now;
  time(&now);
  datasets_json_["meta"]["creation_date"] = get_time_iso_string(now);


  // Populate datasets table with one dataset: "Nonbarcoded library"

  datasets_json_["barcode_filters"] = Json::objectValue;

  datasets_json_["datasets"] = Json::arrayValue;
  datasets_json_["datasets"][0]["dataset_name"] = "Nonbarcoded library";
  datasets_json_["datasets"][0]["file_prefix"] = "rawlib";
  datasets_json_["datasets"][0]["read_groups"][0] = run_id;

  datasets_json_["read_groups"][run_id]["index"] = 0; // Used only for barcodeMask
  datasets_json_["read_groups"][run_id]["description"] = "Nonbarcoded library";

  num_barcodes_ = 0;
  num_datasets_ = 1;
  EnumerateReadGroups();
  dataset_in_use_ = true;
}

// -----------------------------------------------------------------------------------------
// Need to make sure that control & template barcodes are not duplicates of each other

void BarcodeDatasets::RemoveControlBarcodes(const Json::Value& control_datasets_json)
{
  Json::Value new_dataset_json;
  new_dataset_json = datasets_json_;
  new_dataset_json["datasets"]    = Json::arrayValue;
  new_dataset_json["read_groups"] = Json::objectValue;
  int new_ds_idx = 0;

  for (int ds = 0; ds < num_datasets_; ++ds) {
    int num_read_groups = 0;
    Json::Value new_read_groups(Json::arrayValue);

    for (Json::Value::iterator rg = dataset(ds)["read_groups"].begin(); rg != dataset(ds)["read_groups"].end(); ++rg) {
      string read_group_name = (*rg).asString();
      string barcode_sequence;

      if (read_groups()[read_group_name].isMember("barcode"))
        barcode_sequence = read_groups()[read_group_name]["barcode"].get("barcode_sequence", "").asString();
      else
        barcode_sequence = read_groups()[read_group_name].get("barcode_sequence", "").asString();

      if (not barcode_sequence.empty()){

        // Check if the barcode matches any of the control barcodes
    	bool found_match = false;
    	for (Json::Value::iterator control_rg = control_datasets_json["read_groups"].begin(); control_rg != control_datasets_json["read_groups"].end(); ++control_rg) {
          if (barcode_sequence == (*control_rg).get("barcode_sequence", "").asString()){
            cerr << "BarcodeDatasets WARNING: read group " << read_group_name << " has the same barcode as a control read group - it will be removed." << endl;
            cout << "BarcodeDatasets WARNING: read group " << read_group_name << " has the same barcode as a control read group - it will be removed." << endl;
            found_match = true;
            break;
          }
    	}

    	if(not found_match) {
          new_read_groups.append(read_group_name);
          new_dataset_json["read_groups"][read_group_name] = read_groups()[read_group_name];
          num_read_groups++;
    	}
      }
      else {
    	// Non-barcoded read group
    	new_read_groups.append(read_group_name);
        new_dataset_json["read_groups"][read_group_name] = read_groups()[read_group_name];
        num_read_groups++;
      }

    }

    if (num_read_groups > 0) {
    	new_dataset_json["datasets"][new_ds_idx] = dataset(ds);
    	new_dataset_json["datasets"][new_ds_idx]["read_groups"] = new_read_groups;
    	new_ds_idx++;
    }
  }

  LoadJson(new_dataset_json, datasets_json_["meta"]["datasets_source"].asString());
}

// -----------------------------------------------------------------------------------------


void BarcodeDatasets::EnumerateReadGroups()
{
  num_barcodes_ = 0;
  read_group_id_to_name_ = datasets_json_["read_groups"].getMemberNames();
  num_read_groups_ = read_group_id_to_name_.size();
  read_group_name_to_id_.clear();
  start_barcode_names_.resize(num_read_groups_);

  for (int rg = 0; rg < num_read_groups_; ++rg) {
    read_group_name_to_id_[read_group_id_to_name_[rg]] = rg;

    bool have_barcode = datasets_json_["read_groups"][read_group_id_to_name_[rg]].isMember("barcode_sequence") or
               datasets_json_["read_groups"][read_group_id_to_name_[rg]].isMember("barcode");

    if (datasets_json_["read_groups"][read_group_id_to_name_[rg]].isMember("barcode_sequence")){
      num_barcodes_++;
      start_barcode_names_[rg] = datasets_json_["read_groups"][read_group_id_to_name_[rg]].get("barcode_name", "NoName").asString();
    }
    else if (datasets_json_["read_groups"][read_group_id_to_name_[rg]].isMember("barcode")) {
      num_barcodes_++;
      start_barcode_names_[rg] = datasets_json_["read_groups"][read_group_id_to_name_[rg]]["barcode"].get("barcode_name", "NoName").asString();
    }
    else
      start_barcode_names_[rg] = "NoMatch";
  }
  if (num_read_groups_ == 0) {
    cerr << "BarcodeDatasets ERROR: no read groups found." << endl;
    exit(EXIT_FAILURE);
  }

}


// -----------------------------------------------------------------------------------------

void BarcodeDatasets::SaveJson(const string& filename_json)
{
  ofstream out(filename_json.c_str(), ios::out);
  if (out.good())
    out << datasets_json_.toStyledString();
  else
    ION_WARN("Unable to write JSON file " + filename_json);

}

// -----------------------------------------------------------------------------------------

void BarcodeDatasets::SetTF(bool process_tfs)
{
  tf_dataset_ = true;
  dataset_in_use_ = process_tfs;
	if (dataset_in_use_) {
	  dataset(0)["file_prefix"] = "rawtf";
      dataset(0)["dataset_name"] = "Test Fragments";
      read_group(0)["description"] = "Test Fragments";
	}
}

// -----------------------------------------------------------------------------------------
// Create control read group names including the run id from default names

void BarcodeDatasets::SetIonControl(const string& run_id)
{
  control_dataset_ = true;
  dataset_in_use_ = (num_barcodes_ > 0);
  if (not dataset_in_use_)
    return;

  if (num_read_groups_ > num_barcodes_) {
    cerr << "BarcodeDatasets ERROR: IonControl datasets must all have a barcode_sequence entry." << endl;
    exit(EXIT_FAILURE);
  }

  Json::Value control_read_groups;
  for (int ds = 0; ds < num_datasets_; ++ds) {
    for (int rg = 0; rg < (int)dataset(ds)["read_groups"].size(); ++rg) {
      string old_rg_name = dataset(ds)["read_groups"][rg].asString();
      string new_rg_name = run_id + '.' + old_rg_name;
      // Do not append run id if read group name already starts with it.
      if ((old_rg_name.length() > run_id.length()) and (old_rg_name.compare(0, run_id.length(), run_id) == 0))
        new_rg_name = old_rg_name;
      Json::Value temp_rg(new_rg_name);
      dataset(ds)["read_groups"][rg] = temp_rg;
      control_read_groups[new_rg_name] = read_groups()[old_rg_name];
    }
  }
  datasets_json_["read_groups"] = control_read_groups;
  EnumerateReadGroups();
};

// -----------------------------------------------------------------------------------------


void BarcodeDatasets::GenerateFilenames(const string& bead_type, const string& file_type, const string& file_extension, const string& output_directory)
{
  if (not dataset_in_use_) {
    printf("%s dataset disabled.\n", bead_type.c_str());
    return;
  }

  printf("Datasets summary (%s):\n", bead_type.c_str());
  printf("   datasets.json %s/datasets_basecaller.json\n", output_directory.c_str());
  for (int ds = 0; ds < num_datasets_; ++ds) {
    string base = dataset(ds)["file_prefix"].asString();
    dataset(ds)[file_type] = base + file_extension;
    printf("            %3d: %s   Contains read groups: ", ds+1, dataset(ds)["basecaller_bam"].asCString());
    for (int bc = 0; bc < (int)dataset(ds)["read_groups"].size(); ++bc)
      printf("%s ", dataset(ds)["read_groups"][bc].asCString());
    printf("\n");
  }
  printf("\n");
}





