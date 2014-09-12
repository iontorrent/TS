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
  printf ("  -b,--barcodes                    FILE/off   detect barcodes listed in provided file [off]\n");
  printf ("     --barcode-mode                INT        selects barcode classification algorithm [2] Range: {1,2,3}\n" );
  printf ("     --barcode-cutoff              FLOAT      minimum score to call a barcode [1.0] Range: [x>0]\n" );
  printf ("     --barcode-separation          FLOAT      minimum difference between best and second best scores [2.5] Range: [x>0]\n" );
  printf ("     --barcode-filter              FLOAT      barcode freq. threshold, if >0 writes output-dir/barcodeFilter.txt [0.0 = off] Range: [0<=x<1]\n" );
  printf ("     --barcode-filter-minreads     INT        barcode reads threshold, if >0 writes output-dir/barcodeFilter.txt [0 = off] Range: {x>=0}\n" );
  printf ("     --barcode-error-filter        FLOAT      filter barcodes based on average number of errors [1.0] Range: [0<=x<2]\n" );
  printf ("     --barcode-filter-postpone     INT        switch to disable / turn down barcode filters in basecaller Range: {0,1,2}\n" );
  printf ("\n");
}



BarcodeClassifier::BarcodeClassifier(OptArgs& opts, BarcodeDatasets& datasets,
    const ion::FlowOrder& flow_order, const vector<KeySequence>& keys, const string& output_directory, int chip_size_x, int chip_size_y)
  : barcode_mask_(chip_size_x, chip_size_y)
{
  flow_order_ = flow_order;
  num_barcodes_ = 0;
  no_barcode_read_group_ = 0;
  barcode_max_flows_ = 0;
  barcode_max_hp_ = 0;
  barcode_min_start_flow_ = -1;

  num_prints_ = 0;

  // Retrieve command line options

  barcode_filter_                 = opts.GetFirstDouble ('-', "barcode-filter", 0.01);
  barcode_filter_weight_          = opts.GetFirstDouble ('-', "barcode-filter-weight", 0.1);
  barcode_filter_minreads_        = opts.GetFirstInt    ('-', "barcode-filter-minreads", 20);
  barcode_error_filter_           = opts.GetFirstDouble ('-', "barcode-error-filter", 0.0);
  barcode_filter_postpone_        = opts.GetFirstInt    ('-', "barcode-filter-postpone", 0);

  //CheckParameterLowerUpperBound(string identifier ,T &parameter, T lower_limit, int use_lower, T upper_limit, int use_upper, T default_val)
  CheckParameterLowerUpperBound("barcode-filter",          barcode_filter_,          0.0, 1, 1.0, 2, 0.01);
  CheckParameterLowerUpperBound("barcode-filter-weight",   barcode_filter_weight_,   0.0, 1, 1.0, 1, 0.1 );
  CheckParameterLowerUpperBound("barcode-filter-minreads", barcode_filter_minreads_, 0,   1, 0,   0, 20  );
  CheckParameterLowerUpperBound("barcode-error-filter",    barcode_error_filter_,    0.0, 1, 2.0, 2, 0.0 );
  CheckParameterLowerUpperBound("barcode-filter-postpone", barcode_filter_postpone_, 0,   1, 2,   1, 0   );


  windowSize_                     = opts.GetFirstInt    ('-', "window-size", DPTreephaser::kWindowSizeDefault_);
  barcode_bam_tag_		          = opts.GetFirstBoolean('-', "barcode-bam-tag", false);
  skip_droop_                     = opts.GetFirstBoolean('-', "skipDroop", true);

  // We always write this file
  barcode_filter_filename_ = output_directory+"/barcodeFilter.txt";
  if (barcode_filter_postpone_ != 1)
    barcode_filter_weight_ = 1.0;

  //
  // Step 1. First phase of initialization: parse barcode list file
  //

  string file_id  = datasets.barcode_config().get("barcode_id","").asString();
  //score_mode_     = datasets.barcode_config().get("score_mode",1).asInt();
  //score_cutoff_   = datasets.barcode_config().get("score_cutoff",2.0).asDouble();

  score_mode_                     = opts.GetFirstInt    ('-', "barcode-mode", 2);
  score_cutoff_                   = opts.GetFirstDouble ('-', "barcode-cutoff", 1.0);
  score_separation_               = opts.GetFirstDouble ('-', "barcode-separation", 2.5);

  CheckParameterLowerUpperBound("barcode-mode",       score_mode_,       1,   1, 3,   1, 2  );
  CheckParameterLowerUpperBound("barcode-cutoff",     score_cutoff_,     0.0, 1, 0.0, 0, 1.0);
  CheckParameterLowerUpperBound("barcode-separation", score_separation_, 0.0, 1, 0.0, 0, 2.5);

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
    barcode_.back().end_flow = -1;
	barcode_.back().adapter_start_flow = -1;

	// All barcodes share the same key in front of them
    barcode_.back().full_barcode = keys[0].bases();
    int key_length = barcode_.back().full_barcode.length();

    barcode_.back().full_barcode += read_group["barcode_sequence"].asString();
    int key_barcode_length = barcode_.back().full_barcode.length();

    barcode_.back().full_barcode += read_group.get("barcode_adapter","").asString();
    int key_barcode_adapter_length = barcode_.back().full_barcode.length();


    int flow = 0;
    int curBase = 0;
    int end_length = -1;

    while (curBase < key_barcode_adapter_length and flow < flow_order_.num_flows()) {

      // Increment bases for this flow
      while (curBase < key_barcode_adapter_length and barcode_.back().full_barcode[curBase] == flow_order_[flow]) {
        barcode_.back().flow_seq[flow]++;
        barcode_max_hp_ = max(barcode_max_hp_, barcode_.back().flow_seq[flow]);
        curBase++;
      }

      // determine start flow for barcode analysis -- all barcodes share same library key
      if (barcode_min_start_flow_ == -1 or flow < barcode_min_start_flow_) {
        // grab - the next flow after we sequence through the key if there is no overlap between key and barcode
    	//      - or the last the key flow if there is overlap
        if (curBase == key_length)
          barcode_min_start_flow_ = flow +1;
        else if (curBase > key_length)
          barcode_min_start_flow_ = flow;
      }

      // grab the last positive incorporating flow for the barcode.
      if (curBase >= key_barcode_length and barcode_.back().end_flow == -1) {
        barcode_.back().end_flow = flow;
        end_length = curBase;
      }
      // First unique adapter incorporation flow past last barcode HP.
      if (end_length != -1 and curBase > end_length and barcode_.back().adapter_start_flow == -1)
        barcode_.back().adapter_start_flow = flow;

      flow++;
    }

    if (barcode_.back().end_flow == -1)
      barcode_.back().end_flow = flow - 1;
    if (barcode_.back().adapter_start_flow == -1)
      barcode_.back().adapter_start_flow = flow;
    barcode_.back().num_flows = flow;
    barcode_.back().last_homopolymer = barcode_.back().flow_seq[flow-1];
  }

  // And loop through barcodes again to determine maximum amount of flows after start flow has been determined
  for (unsigned int bc=0; bc<barcode_.size(); bc++) {
    barcode_max_flows_ = max(barcode_max_flows_, ((barcode_.at(bc).adapter_start_flow - barcode_min_start_flow_)));
  }

  // Prepare directory structure and output files

  barcode_mask_filename_ = output_directory + "/barcodeMask.bin";
  barcode_mask_.Init(chip_size_x, chip_size_y, MaskAll);
  if (0 != barcode_mask_.WriteRaw (barcode_mask_filename_.c_str()))
    fprintf (stderr, "BarcodeClassifier: Cannot create mask file file: %s\n", barcode_mask_filename_.c_str());
  else
    ValidateAndCanonicalizePath(barcode_mask_filename_);

  // Export barcode_max_flows_
  datasets.SetBCmaxFlows(barcode_max_flows_);

  // Pretty print barcode processing settings

  printf("Barcode settings:\n");
  printf("   Barcode mask file        : %s\n", barcode_mask_filename_.c_str());

  printf("   Barcode set name         : %s\n", file_id.c_str());
  printf("   Number of barcodes       : %d\n", num_barcodes_);
  printf("   Scoring mode             : %d\n", score_mode_);
  printf("   Scoring threshold        : %1.1lf\n", score_cutoff_);
  printf("   Separation threshold     : %1.2lf\n", score_separation_);
  printf("   Barcode filter threshold : %1.6f (0.0 = disabled)\n", barcode_filter_);
  printf("   Barcode error filter     : %1.3f\n", barcode_error_filter_);
  printf("   Barcode filter minreads  : %d (0 = disabled)\n", barcode_filter_minreads_);
  printf("   Barcode filter filename  : %s\n", barcode_filter_filename_.c_str());
  printf("   Generation of XB bam-tag : %s (number of base errors during barcode classification)\n", (barcode_bam_tag_ ? "on" : "off"));
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
  if (skip_droop_)
	treephaser.SetModelParameters(cf, ie);
  else
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
  processed_read.barcode_filt_zero_error = -1;
  processed_read.barcode_distance = 0.0;
  processed_read.barcode_bias.assign(barcode_max_flows_,0);

  float best_distance = score_cutoff_ + score_separation_;
  float second_best_distance = 1e20;
  vector<float> best_bias(barcode_max_flows_,0);

  // looks at flow-space absolute error counts, not ratios
  if (score_mode_ == 1) {

    for (int bc = 0; bc < num_barcodes_; ++bc) {

      int num_errors = 0;
	  int flow = 0;
	  int base = 0;
      for (; flow < barcode_[bc].adapter_start_flow; ++flow) {
        int hp_length = 0;
        while (base < processed_read.filter.n_bases and base_to_flow[base] == flow) {
          base++;
          hp_length++;
        }
        if (flow >= barcode_min_start_flow_) {
          if (flow < barcode_[bc].num_flows-1)
            num_errors += abs(barcode_[bc].flow_seq[flow] - hp_length);
          else
        	num_errors += max(0, (barcode_[bc].flow_seq[flow] - hp_length));
        }
      }

      if (num_errors < best_errors) {
        best_errors = num_errors;
        best_barcode = bc;
        best_distance = 0.0;
      }
    }
  } // ----------

  // Minimize distance to barcode predicted signal
  else if (score_mode_ == 2 or score_mode_ == 3) {
    // Minimize L2 distance for score_mode_ == 2
    // Minimize L1 distance for score_mode_ == 3

    for (int bc = 0; bc < num_barcodes_; ++bc) {

      int num_errors = 0;
      float distance = 0.0;
      vector<float> bias(barcode_max_flows_,0);

      for (int flow = barcode_min_start_flow_; flow < barcode_[bc].adapter_start_flow; ++flow) {

    	// Compute Bias
    	bias.at(flow-barcode_min_start_flow_) = basecaller_read.normalized_measurements.at(flow) - barcode_[bc].predicted_signal.at(flow);
    	// Thresholding of measurements to a range of [0,2]
    	double acting_measurement = basecaller_read.normalized_measurements.at(flow);
    	acting_measurement = max(min(acting_measurement, (double)barcode_max_hp_),0.0);
    	// Compute distance
    	double residual = barcode_[bc].predicted_signal[flow] - acting_measurement;
    	if (flow == barcode_[bc].num_flows-1)
          residual = max(residual, 0.0);
        if (score_mode_ == 2)
          distance += residual * residual;
        else
          distance += fabs(residual);
        // Compute hard decision errors - approximation from predicted values
        if (flow < barcode_[bc].num_flows-1)
          num_errors += round(fabs(barcode_[bc].predicted_signal[flow] - basecaller_read.prediction[flow]));
        else
          num_errors += round(max(barcode_[bc].predicted_signal[flow] - basecaller_read.prediction[flow], (float)0.0));
      }

      if (distance < best_distance) {
        best_errors = num_errors;
        second_best_distance = best_distance;
        best_distance = distance;
        best_barcode = bc;
        best_bias = bias;
      }
      else if (distance < second_best_distance)
        second_best_distance = distance;
    }

    if (second_best_distance - best_distance  < score_separation_) {
      if (best_errors == 0) {
        processed_read.barcode_filt_zero_error = best_barcode;
      }
      best_barcode = -1;
    }
  }

  // -------- Classification done, now accounting ----------

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
  if(barcode_bam_tag_)
	processed_read.bam.AddTag("XB","i", processed_read.barcode_n_errors);
  processed_read.barcode_bias = best_bias;
  processed_read.barcode_distance = best_distance;

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

  // Write barcode filter information into datasets json for python filters
  datasets.barcode_filters()["filter_frequency"] = barcode_filter_;
  datasets.barcode_filters()["filter_minreads"]  = (Json::Int64)barcode_filter_minreads_;
  datasets.barcode_filters()["filter_errors_hist"] = barcode_error_filter_;
  datasets.barcode_filters()["filter_postpone"]   = (Json::Int64)barcode_filter_postpone_;
  // not used by python filters but useful information if we are looking at the json file:
  datasets.barcode_filters()["filter_weight"]     = barcode_filter_weight_;
  datasets.barcode_filters()["classifier_mode"]   = (Json::Int64)score_mode_;
  datasets.barcode_filters()["classifier_cutoff"] = score_cutoff_;
  datasets.barcode_filters()["classifier_separation"] = score_separation_;

  // Generate Filter Thresholds for barcode filtering (minreads filter only active if filtering done in basecaller)
  int read_threshold = 0;
  if(barcode_filter_postpone_ == 0)
	read_threshold = barcode_filter_minreads_;

  // Adjust filter threshold based on barcode frequencies if desired
  if (barcode_filter_postpone_ < 2 and barcode_filter_ > 0.0) {
    //vector<int> read_counts;
    //for (Json::Value::iterator rg = datasets.read_groups().begin(); rg != datasets.read_groups().end(); ++rg){
    //  if ((*rg).isMember("read_count") and (*rg).isMember("barcode_sequence"))
    //    read_counts.push_back((*rg)["read_count"].asInt());
    //}
    //sort (read_counts.begin(), read_counts.end(), std::greater<int>());
	int max_read_count = 0;
	for (Json::Value::iterator rg = datasets.read_groups().begin(); rg != datasets.read_groups().end(); ++rg){
      if ((*rg).isMember("read_count") and (*rg).isMember("barcode_sequence"))
        max_read_count = max(max_read_count, (*rg)["read_count"].asInt());
	}
	read_threshold = max(read_threshold, (int)((double)max_read_count*barcode_filter_*barcode_filter_weight_));
  }

  // Below is the actual filtering of barcodes and file writing
  // We still want to create barcode statistics, even if all filters are turned off.
  // We always filter barcode groups with zero reads in them

  FILE *ffile = fopen(barcode_filter_filename_.c_str(),"wt");
  if (ffile)
    fprintf(ffile, "BarcodeId,BarcodeName,NumReads,Include\n");

  for (Json::Value::iterator rg = datasets.read_groups().begin(); rg != datasets.read_groups().end(); ++rg) {
    if ((*rg).isMember("read_count") and (*rg).isMember("barcode_sequence")) {
      int read_count = (*rg)["read_count"].asInt();

      // Initial filtering based on number of reads in read group
      bool i_am_filtered = (read_count > read_threshold) ? false : true;

      // Further filtering based on average number of errors
      if((not barcode_filter_postpone_) and (not i_am_filtered)
          and (barcode_error_filter_ > 0) and (*rg).isMember("barcode_errors_hist")) {
        double one_error  = (*rg)["barcode_errors_hist"][1].asDouble();
        double two_errors = (*rg)["barcode_errors_hist"][2].asDouble();
        i_am_filtered = ((one_error + 2.0*two_errors) / (double)read_count) > barcode_error_filter_;
      }

      // Set status in datasets
      (*rg)["filtered"] = i_am_filtered;
      if (ffile)
        fprintf(ffile, "%s,%s,%i,%i\n", (*rg)["barcode_name"].asCString(),
        		                        (*rg)["barcode_sequence"].asCString(),
                                         read_count, (int)(not i_am_filtered));
      }
    }
    if (ffile)
      fclose ( ffile );
}

