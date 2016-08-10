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

      if (read_groups()[read_group_name].isMember("barcode_sequence")){

        // Check if the barcode matches any of the control barcodes
    	bool found_match = false;
    	for (Json::Value::iterator control_rg = control_datasets_json["read_groups"].begin(); control_rg != control_datasets_json["read_groups"].end(); ++control_rg) {
          if (read_groups()[read_group_name]["barcode_sequence"] == (*control_rg).get("barcode_sequence", "").asString()){
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

// This function is not used any more
void BarcodeDatasets::InitializeFromBarcodeList(const string& run_id, string barcode_list_filename)
{

  datasets_json_ = Json::objectValue;

  // Prepare metadata
  datasets_json_["meta"]["format_name"] = "Dataset Map";
  datasets_json_["meta"]["format_version"] = "1.0";
  datasets_json_["meta"]["generated_by"] = "BaseCaller";
  time_t now;
  time(&now);
  datasets_json_["meta"]["creation_date"] = get_time_iso_string(now);

  // Actually populate the json structure

  datasets_json_["barcode_filters"] = Json::objectValue;

  datasets_json_["datasets"] = Json::arrayValue;
  datasets_json_["datasets"][0]["dataset_name"] = "Reads not matched to any barcode";
  datasets_json_["datasets"][0]["file_prefix"] = "nomatch_rawlib";
  datasets_json_["datasets"][0]["read_groups"][0] = run_id + ".nomatch";
  num_barcodes_ = 0;
  num_datasets_ = 1;

  datasets_json_["read_groups"][run_id+".nomatch"]["index"] = 0; // Used only for barcodeMask
  datasets_json_["read_groups"][run_id+".nomatch"]["description"] = "Reads not matched to any barcode";


  datasets_json_["barcode_config"]["barcode_id"] = "";
  datasets_json_["barcode_config"]["score_mode"] = 1;
  datasets_json_["barcode_config"]["score_cutoff"] = 2.0;

  //
  // Step 1. First phase of initialization: parse barcode list file
  //

  //ValidateAndCanonicalizePath(barcode_list_filename);
  // Open file
  FILE *fd = fopen(barcode_list_filename.c_str(), "rb");
  if (fd == NULL) {
    fprintf (stderr, "ERROR Cannot open barcode list file: %s: %s\n", barcode_list_filename.c_str(), strerror(errno));
    exit (EXIT_FAILURE);
  }

  // default token indexes to V0 file formats, these will adjust as necessary depending on discovered keywords
  // MGD - in the future, we will parse up a json file and not need to worry about these annoying version changes
  int token_idString = 0;
  int token_barcodeSequence = 1;
  int token_fimeprimeAdapter = 2;

  //Read in barcodes
  char line[1024];
  char *key = NULL;
  char *arg = NULL;
  while ( fgets ( line, sizeof ( line ), fd ) )
  {
    //tokenize at first space
    static const char whitespace[] = " \t\r\n";
    key = strtok ( line, whitespace );
    arg = strtok ( NULL, whitespace );
    //match token
    if (strcmp(key, "file_id") == 0) {
      datasets_json_["barcode_config"]["barcode_id"] = arg;

    } else if (strcmp(key, "score_mode") == 0) {
      int score_mode = 0;
      if (arg) {
        int ret = sscanf(arg, "%d", &score_mode);
        if (ret != 1)
          score_mode = 0;
      }
      datasets_json_["barcode_config"]["score_mode"] = score_mode;

      // hack - looks like a V1 file
      token_idString = 1;
      token_barcodeSequence = 2;
      token_fimeprimeAdapter = 3;

    } else if (strcmp(key, "score_cutoff") == 0) {
      double score_cutoff = 0;
      if (arg) {
        int ret = sscanf(arg, "%lf", &score_cutoff);
        if (ret != 1)
          score_cutoff = 0;
      }
      datasets_json_["barcode_config"]["score_cutoff"] = score_cutoff;

      // hack - looks like a V1 file
      token_idString = 1;
      token_barcodeSequence = 2;
      token_fimeprimeAdapter = 3;

    } else if (strcmp(key, "barcode") == 0) {
      //tokenize arg by comma
      char *ptr = arg; // ptr points to our current token
      int tokenCount = 0;
      char *token[20]; // list of tokens (will actually all just point to locations in the line)
      while (ptr) {
        token[tokenCount] = ptr; // point to the start of the token
        tokenCount++;
        // find the next delimeter
        ptr = strchr(ptr, ',');
        // if its not NULL (end of string), make it NULL and advance one char (to start of next token)
        if (ptr != NULL)
        {
          *ptr = 0;
          ptr++;
        }
      }

      // tokens are:
      //   0 - index
      //   1 - ID string [only with V1 and newer formats]
      //   2 - barcode sequence
      //   3 - fiveprimeadapter
      //   4 - annotation
      //   5 - type
      //   6 - length
      //   7 - floworder

      ToUpper ( token[token_barcodeSequence] );
      ToUpper ( token[token_fimeprimeAdapter] );

      // input validation
      char c = token[token_fimeprimeAdapter][0];
      if ( ( c != 'A' ) && ( c != 'C' ) && ( c != 'G' ) && ( c != 'T' ) )
        token[token_fimeprimeAdapter][0] = 0; // not valid input, so just clear it to NULL

      //index, and make sure it isn't 0 and/or invalid.
      int bcIndex = atoi ( token[0] );
      if ( bcIndex<=0 )
      {
        fprintf (stderr, "BarcodeDatasets: Invalid index in barcode list file: %s\n", token[0] );
        continue;
      }

      string barcode_name = token[token_idString];
      string barcode_sequence = token[token_barcodeSequence];
      string barcode_adapter = token[token_fimeprimeAdapter];

      string read_group = run_id + "." + barcode_name;

      datasets_json_["read_groups"][read_group]["barcode_adapter"] = barcode_adapter;
      datasets_json_["read_groups"][read_group]["barcode_sequence"] = barcode_sequence;
      datasets_json_["read_groups"][read_group]["barcode_name"] = barcode_name;
      datasets_json_["read_groups"][read_group]["index"] = bcIndex; // Only needed for barcodeMask
      datasets_json_["read_groups"][read_group]["library"] = barcode_name;
      datasets_json_["read_groups"][read_group]["description"] = barcode_name+"-barcoded library";

      num_barcodes_++;

      datasets_json_["datasets"][num_datasets_]["dataset_name"] = string("Barcode ")+barcode_name;
      datasets_json_["datasets"][num_datasets_]["file_prefix"] = barcode_name+string("_rawlib");
      datasets_json_["datasets"][num_datasets_]["read_groups"][0] = read_group;
      num_datasets_++;

    } else {
      fprintf (stderr, "BarcodeDatasets: Unknown entry in barcode list file: %s\n", key);
    }
  }

  fclose ( fd );

}

// -----------------------------------------------------------------------------------------

void BarcodeDatasets::EnumerateReadGroups()
{
  num_barcodes_ = 0;
  read_group_id_to_name_ = datasets_json_["read_groups"].getMemberNames();
  num_read_groups_ = read_group_id_to_name_.size();
  read_group_name_to_id_.clear();
  for (int rg = 0; rg < num_read_groups_; ++rg) {
    read_group_name_to_id_[read_group_id_to_name_[rg]] = rg;
    if (datasets_json_["read_groups"][read_group_id_to_name_[rg]].isMember("barcode_sequence"))
      num_barcodes_++;
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





