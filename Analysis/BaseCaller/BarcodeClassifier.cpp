/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BarcodeClassifier.cpp
//! @ingroup  BaseCaller
//! @brief    BarcodeClassifier. Barcode detection and trimming for BaseCaller

#include "BarcodeClassifier.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "Utils.h"

void ValidateAndCanonicalizePath(string &path);   // Borrowed from BaseCaller.cpp

void BarcodeClassifier::PrintHelp()
{
  printf ("Barcode classification options:\n");
  printf ("  -b,--barcodes              FILE/off   detect barcodes listed in provided file [off]\n");
  printf ("     --barcode-mode          INT        selects barcode classification algorithm [1]\n" );
  printf ("     --barcode-cutoff        FLOAT      minimum score to call a barcode [2\n" );
  printf ("     --barcode-separation    FLOAT      minimum difference between best and second best scores [0.5]\n" );
  printf ("     --barcode-filter        FLOAT      barcode freq. threshold, if >0 writes output-dir/barcodeFilter.txt [0.0 = off]\n" );
  printf ("     --barcode-filter-minreads FLOAT      barcode reads threshold, if >0 writes output-dir/barcodeFilter.txt [0 = off]\n" );
  printf ("\n");
}



BarcodeClassifier::BarcodeClassifier(OptArgs& opts, BarcodeDatasets& datasets,
    const ion::FlowOrder& flow_order, const vector<KeySequence>& keys, const string& output_directory, int chip_size_x, int chip_size_y)
  : barcode_mask_(chip_size_x, chip_size_y)
{
  flow_order_ = flow_order;
  num_barcodes_ = 0;
  no_barcode_read_group_ = 0;

  // Retrieve command line options

  barcode_directory_              = opts.GetFirstString ('-', "barcode-directory", output_directory+"/bc_files");
  barcode_filter_                 = opts.GetFirstDouble ('-', "barcode-filter", 0.0);
  barcode_filter_minreads_        = opts.GetFirstInt 	('-', "barcode-filter-minreads", 0);
  windowSize_                     = opts.GetFirstInt    ('-', "window-size", DPTreephaser::kWindowSizeDefault_);

  bc_adjust_				      = opts.GetFirstBoolean('-', "bc-adjust", false);

  if (barcode_filter_ > 0.0 || barcode_filter_minreads_ > 0)
    barcode_filter_filename_ = output_directory+"/barcodeFilter.txt";


  //
  // Step 1. First phase of initialization: parse barcode list file
  //

  string file_id  = datasets.barcode_config().get("barcode_id","").asString();
  //score_mode_     = datasets.barcode_config().get("score_mode",1).asInt();
  //score_cutoff_   = datasets.barcode_config().get("score_cutoff",2.0).asDouble();

  score_mode_                     = opts.GetFirstInt    ('-', "barcode-mode", 1);
  score_cutoff_                   = opts.GetFirstDouble ('-', "barcode-cutoff", 2);
  score_separation_               = opts.GetFirstDouble ('-', "barcode-separation", 0.5);

  barcode_.reserve(datasets.num_read_groups());

  for (int rg_idx = 0; rg_idx < datasets.num_read_groups(); ++rg_idx) {

    Json::Value& read_group = datasets.read_group(rg_idx);

    // Group for non-barcoded reads
    if (!read_group.isMember("barcode_sequence")) {
      no_barcode_read_group_ = rg_idx;
      continue;
    }

    num_barcodes_++;
    barcode_.push_back(Barcode());
    barcode_.back().mask_index = read_group.get("index",0).asInt(); // Only used for barcodeMask
    barcode_.back().read_group_index = rg_idx;
    barcode_.back().flow_seq.assign(flow_order_.num_flows(), 0);
    barcode_.back().start_flow = -1;
    barcode_.back().end_flow = -1;
	barcode_.back().adapter_end_flow = -1;

    barcode_.back().full_barcode = keys[0].bases();
    int key_length = barcode_.back().full_barcode.length();

    barcode_.back().full_barcode += read_group["barcode_sequence"].asString();
    int key_barcode_length = barcode_.back().full_barcode.length();

    barcode_.back().full_barcode += read_group.get("barcode_adapter","").asString();
    int key_barcode_adapter_length = barcode_.back().full_barcode.length();


    int flow = 0;
    int curBase = 0;

    while (curBase < key_barcode_adapter_length and flow < flow_order_.num_flows()) {

      while (barcode_.back().full_barcode[curBase] == flow_order_[flow] and curBase < key_barcode_adapter_length) {
        barcode_.back().flow_seq[flow]++;
        curBase++;
      }
      // grab the next flow after we sequence through the key, this will be the first flow we will want to count towards barcode matching/scoring, even if its a 0-mer flow
      if (curBase >= key_length and barcode_.back().start_flow == -1)
        barcode_.back().start_flow = flow+1;
      // grab the last positive incorporating flow for the barcode, any 0-mer flows after this and before the insert or adapter would not be counted in the barcode matching/scoring
      if (curBase >= key_barcode_length and barcode_.back().end_flow == -1)
        barcode_.back().end_flow = flow;

      if (curBase >= key_barcode_adapter_length and barcode_.back().adapter_end_flow == -1)
        barcode_.back().adapter_end_flow = flow;

      flow++;
    }
    if (barcode_.back().end_flow == -1)
      barcode_.back().end_flow = flow - 1;
    barcode_.back().num_flows = flow;
    barcode_.back().last_homopolymer = barcode_.back().flow_seq[flow-1];
  }

  // Prepare directory structure and output files

  CreateResultsFolder ((char*)barcode_directory_.c_str());
  ValidateAndCanonicalizePath(barcode_directory_);


  barcode_mask_filename_ = output_directory + "/barcodeMask.bin";
  barcode_mask_.Init(chip_size_x, chip_size_y, MaskAll);
  if (0 != barcode_mask_.WriteRaw (barcode_mask_filename_.c_str()))
    fprintf (stderr, "BarcodeClassifier: Cannot create mask file file: %s\n", barcode_mask_filename_.c_str());
  else
    ValidateAndCanonicalizePath(barcode_mask_filename_);



  // Pretty print barcode processing settings

  printf("Barcode settings:\n");
  printf("   Barcode output directory : %s\n", barcode_directory_.c_str());
  printf("   Barcode mask file        : %s\n", barcode_mask_filename_.c_str());

  printf("   Barcode set name         : %s\n", file_id.c_str());
  printf("   Number of barcodes       : %d\n", num_barcodes_);
  printf("   Scoring mode             : %d\n", score_mode_);
  printf("   Scoring threshold        : %1.1lf\n", score_cutoff_);
  printf("   Barcode filter threshold : %1.4f (0.0 = disabled)\n", barcode_filter_);
  printf("   Barcode filter minreads :  %d (0 = disabled)\n", barcode_filter_minreads_);
  printf("   Barcode filter filename  : %s\n", barcode_filter_filename_.c_str());
  printf("\n");

}


BarcodeClassifier::~BarcodeClassifier()
{
}


void BarcodeClassifier::BuildPredictedSignals(float cf, float ie, float dr)
{
  if (num_barcodes_ == 0)
    return;

  DPTreephaser treephaser(flow_order_, windowSize_);
  BasecallerRead basecaller_read;
  treephaser.SetModelParameters(cf, ie, dr);

  for (int bc = 0; bc < num_barcodes_; ++bc) {
    basecaller_read.sequence.assign(barcode_[bc].full_barcode.begin(),barcode_[bc].full_barcode.end());
    basecaller_read.prediction.assign(flow_order_.num_flows(), 0);

    treephaser.Simulate(basecaller_read, barcode_[bc].num_flows);

    barcode_[bc].predicted_signal.swap(basecaller_read.prediction);
  }
}





/*
 * flowSpaceTrim - finds the closest barcode in flowspace to the sequence passed in,
 * and then trims to exactly the expected flows so it can be error tolerant in base space
 */
void BarcodeClassifier::ClassifyAndTrimBarcode(int read_index, ProcessedRead &processed_read, const BasecallerRead& basecaller_read,
    const vector<int>& base_to_flow)
{
  int best_barcode = -1;
  int best_errors = 1 + (int)score_cutoff_; // allows match with this many errors minimum when in bc_score_mode 1

  float best_distance = 1 + score_cutoff_;
  float second_best_distance = 1e20;

  int baseAdapter = 0;
  int flowAdapter = 0;

  if (score_mode_ == 1) { // looks at flow-space absolute error counts, not ratios

    for (int bc = 0; bc < num_barcodes_; ++bc) {

      int num_errors = 0;
	  int flow = 0;
	  int base = 0;
      for (; flow <= barcode_[bc].end_flow; ++flow) {
        int hp_length = 0;
        while (base < processed_read.filter.n_bases and base_to_flow[base] == flow) {
          base++;
          hp_length++;
        }
        if (flow >= barcode_[bc].start_flow)
          num_errors += abs(barcode_[bc].flow_seq[flow] - hp_length);
      }

      if (num_errors < best_errors) {
        best_errors = num_errors;
        best_barcode = bc;
		baseAdapter = base;
		flowAdapter = flow;
      }
    }
  }

  else if (score_mode_ == 3) { // Minimize square-distance to barcode predicted signal

    for (int bc = 0; bc < num_barcodes_; ++bc) {

      float xy = 0;
      float yy = 0;
      for (int flow = barcode_[bc].start_flow; flow <= barcode_[bc].end_flow; ++flow) {
        xy += barcode_[bc].predicted_signal[flow] * basecaller_read.normalized_measurements[flow];
        yy += basecaller_read.normalized_measurements[flow] * basecaller_read.normalized_measurements[flow];
      }
      if (yy == 0)
        continue;

      float distance = 0;
      for (int flow = barcode_[bc].start_flow; flow <= barcode_[bc].end_flow; ++flow)
        distance += fabs(barcode_[bc].predicted_signal[flow] - basecaller_read.normalized_measurements[flow] * xy / yy);

      if (distance < best_distance) {
        best_errors = (int)distance;
        second_best_distance = best_distance;
        best_distance = distance;
        best_barcode = bc;
      }
      else if (distance < second_best_distance)
        second_best_distance = distance;
    }

    if (second_best_distance - best_distance  < score_separation_)
      best_barcode = -1;
  }


  if (best_barcode == -1) {
    int x, y;
    barcode_mask_.IndexToRowCol (read_index, y, x);
    barcode_mask_.SetBarcodeId(x, y, 0);
    processed_read.read_group_index = no_barcode_read_group_;
    return;
  }

  const Barcode& bce = barcode_[best_barcode];

  int x, y;
  barcode_mask_.IndexToRowCol (read_index, y, x);
  barcode_mask_.SetBarcodeId(x, y, (uint16_t)bce.mask_index);
  processed_read.read_group_index = bce.read_group_index;

  processed_read.barcode_n_errors = best_errors;
  if(bc_adjust_)
  {
	  processed_read.barcode_adjust_errors = best_errors;
	        
	  /*for (int flow = flowAdapter, base = baseAdapter; flow <= barcode_[best_barcode].adapter_end_flow; ++flow)
	  {
        int hp_length = 0;
        while (base < processed_read.filter.n_bases and base_to_flow[base] == flow) 
		{
          base++;
          hp_length++;
        }

        processed_read.barcode_adjust_errors += abs(barcode_[best_barcode].flow_seq[flow] - hp_length);
      }*/
  }

  processed_read.filter.n_bases_prefix = 0;
  while (processed_read.filter.n_bases_prefix < processed_read.filter.n_bases and base_to_flow[processed_read.filter.n_bases_prefix] < bce.num_flows-1)
    processed_read.filter.n_bases_prefix++;

  int last_homopolymer = bce.last_homopolymer;
  while (processed_read.filter.n_bases_prefix < processed_read.filter.n_bases and base_to_flow[processed_read.filter.n_bases_prefix] < bce.num_flows and last_homopolymer > 0) {
    processed_read.filter.n_bases_prefix++;
    last_homopolymer--;
  }
}



void BarcodeClassifier::Close(BarcodeDatasets& datasets)
{
  if (0 != barcode_mask_.WriteRaw (barcode_mask_filename_.c_str()))
    fprintf (stderr, "BarcodeClassifier: Cannot write mask file file: %s\n", barcode_mask_filename_.c_str());

  // Generate barcodeFilter.txt file

  if (barcode_filter_ > 0.0 || barcode_filter_minreads_ > 0) {

    vector<int> read_counts;
    for (Json::Value::iterator rg = datasets.read_groups().begin(); rg != datasets.read_groups().end(); ++rg)
      if ((*rg).isMember("read_count") and (*rg).isMember("barcode_sequence"))
        read_counts.push_back((*rg)["read_count"].asInt());
    sort (read_counts.begin(), read_counts.end(), std::greater<int>());

    int read_threshold = (barcode_filter_minreads_ > 0) ? barcode_filter_minreads_ : 20;
	
	if (barcode_filter_ > 0.0) {
	  for (int i = 1; i < (int)read_counts.size(); ++i) {
		if (read_counts[i] / (read_counts[i-1] + 0.001) < barcode_filter_) {
		  read_threshold = max(read_threshold, read_counts[i-1]);
		  break;
		}
	  }
	}

    //datasets.barcode_config()["filter_threshold"] = read_threshold;

    FILE *ffile = fopen(barcode_filter_filename_.c_str(),"wt");
    if (ffile)
      fprintf(ffile, "BarcodeId,BarcodeName,NumReads,Include\n");
    for (Json::Value::iterator rg = datasets.read_groups().begin(); rg != datasets.read_groups().end(); ++rg) {
      if ((*rg).isMember("read_count") and (*rg).isMember("barcode_sequence")) {
        int read_count = (*rg)["read_count"].asInt();
        (*rg)["filtered"] = (read_count >= read_threshold) ? false : true;
        if (ffile)
          fprintf(ffile, "%s,%s,%i,%i\n", (*rg)["barcode_name"].asCString(), (*rg)["barcode_sequence"].asCString(),
            read_count, (int)(read_count >= read_threshold));
      }
    }
    if (ffile)
      fclose ( ffile );
  }
}

