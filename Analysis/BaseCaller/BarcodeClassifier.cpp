/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BarcodeClassifier.cpp
//! @ingroup  BaseCaller
//! @brief    BarcodeClassifier. Barcode detection and trimming for BaseCaller

#include "BarcodeClassifier.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <algorithm>

#include "Utils.h"
#include "MiscUtil.h"

// Definition of default vlaues
#define DEF_HANDLE_MODE   1
#define DEF_HANDLE_CUTOFF 3

//void ValidateAndCanonicalizePath(string &path);   // Borrowed from BaseCaller.cpp

void BarcodeClassifier::PrintHelp()
{
  printf ("Barcode classification options:\n");

  //printf ("  -b,--barcodes                    FILE/off   detect barcodes listed in provided file [off]\n");
  printf ("     --barcode-mode                INT        selects barcode classification algorithm [2] Range: {1,2,3}\n" );
  printf ("     --barcode-cutoff              FLOAT      minimum score to call a barcode [1.0] Range: [x>0]\n" );
  printf ("     --barcode-separation          FLOAT      minimum difference between best and second best scores [2.0] Range: [x>0]\n" );
  printf ("     --barcode-filter              FLOAT      barcode freq. threshold, if >0 writes output-dir/barcodeFilter.txt [0.0 = off] Range: [0<=x<1]\n" );
  printf ("     --barcode-filter-minreads     INT        barcode reads threshold, if >0 writes output-dir/barcodeFilter.txt [0 = off] Range: {x>=0}\n" );
  printf ("     --barcode-error-filter        FLOAT      filter barcodes based on average number of errors [1.0] Range: [0<=x<2]\n" );
  printf ("     --barcode-filter-postpone     INT        switch to disable{2} / reduce barcode filters{1} in basecaller. Range: {0,1,2}\n" );
  printf ("     --barcode-filter-named        BOOL       If true, include barcodes with a specified sample name in filtering [false]\n");
  printf ("     --barcode-ignore-flows        INT,INT    Specify a (open ended) interval of flows to ignore for barcode classification.\n");
  printf ("     --barcode-compute-dmin        BOOL       If true, computes minimum Hamming distance of barcode set [true]\n");
  printf ("     --barcode-auto-config         BOOL       If true, automatically selects barcode cutoff and separation parameters. [false]\n");
  printf ("     --barcode-check-limits        BOOL       If true, performs a basic sanity check on input options. [true]\n");
  printf ("     --barcode-adapter-check       FLOAT      Maximum allowed squared residual per adapter flow (off=0) [0.15]\n");
  printf ("\n");
}

// ------------------------------------------------------------------------

BarcodeClassifier::~BarcodeClassifier()
{
}

// ------------------------------------------------------------------------

BarcodeClassifier::BarcodeClassifier(OptArgs& opts, BarcodeDatasets& datasets, const ion::FlowOrder& flow_order,
                                     const vector<KeySequence>& keys, const string& output_directory,
                                     int chip_size_x, int chip_size_y, const Json::Value& structure)
  : barcode_mask_(chip_size_x, chip_size_y), have_handles_(false), handle_mode_(DEF_HANDLE_MODE), handle_cutoff_(DEF_HANDLE_CUTOFF)
{
  num_barcodes_      = 0;
  barcode_max_flows_ = 0;
  barcode_full_flows_= -1;
  end_flow_synch_    = false;
  barcode_max_hp_    = 0;
  hamming_dmin_      = -1;
  barcode_min_start_flow_ = -1;
  no_barcode_read_group_  = -1;
  have_ambiguity_codes_   = false;

  dataset_in_use_       = datasets.DatasetInUse();
  is_control_dataset_   = datasets.IsControlDataset();


  // --- Prepare directory structure and output files

  // Header for all subsequent warning / info messages - printed only if there are barcode sin the dataset
  if (dataset_in_use_)
    cout << "Barcode classification settings (" << (is_control_dataset_ ? "IonControl" : "Template")   << "):" << endl;

  string barcode_id  = datasets.barcode_config().get("barcode_id","noID").asString();
  // Only write barcode mask for template barcodes - not for control barcodes
  if (is_control_dataset_ or (not dataset_in_use_)) {
    barcode_mask_filename_   = "disabled";
    barcode_filter_filename_ = "disabled";
  } else {
	barcode_filter_filename_ = output_directory+"/barcodeFilter.txt";
    barcode_mask_filename_ = output_directory + "/barcodeMask.bin";
    barcode_mask_.Init(chip_size_x, chip_size_y, MaskAll);
    if (0 != barcode_mask_.WriteRaw (barcode_mask_filename_.c_str()))
      fprintf (stderr, "BarcodeClassifier WARNING: Cannot create mask file file: %s\n", barcode_mask_filename_.c_str());
  }


  // --- Retrieve command line options

  barcode_filter_                 = opts.GetFirstDouble ('-', "barcode-filter", 0.01);
  barcode_filter_weight_          = opts.GetFirstDouble ('-', "barcode-filter-weight", 0.1);
  barcode_filter_minreads_        = opts.GetFirstInt    ('-', "barcode-filter-minreads", 10);
  barcode_error_filter_           = opts.GetFirstDouble ('-', "barcode-error-filter", 0.0);
  barcode_filter_postpone_        = opts.GetFirstInt    ('-', "barcode-filter-postpone", 0);
  barcode_filter_named_           = opts.GetFirstBoolean('-', "barcode-filter-named", false);
  check_limits_                   = opts.GetFirstBoolean('-', "barcode-check-limits", true);
  score_auto_config_              = opts.GetFirstBoolean('-', "barcode-auto-config", false);
  bool compute_dmin               = opts.GetFirstBoolean('-', "barcode-compute-dmin", true);
  adapter_cutoff_                 = opts.GetFirstDouble ('-', "barcode-adapter-check", 0.15);

  trim_barcodes_                  = opts.GetFirstBoolean('-', "trim-barcodes", true);
  int calibration_training        = opts.GetFirstInt    ('-', "calibration-training", -1);
  if (not trim_barcodes_ and calibration_training >= 0) {
    cerr << "BarcodeClassifier WARNING: Ignoring option 'trim-barcode=off' during calibration training phase." << endl;
    trim_barcodes_=true;
  }

  // Check parameters that have absolute limits
  // CheckParameterLowerUpperBound(string identifier ,T &parameter, T lower_limit, int use_lower, T upper_limit, int use_upper, T default_val)
  CheckParameterLowerUpperBound("barcode-filter",          barcode_filter_,          0.0, 1, 1.0, 2, 0.01);
  CheckParameterLowerUpperBound("barcode-filter-weight",   barcode_filter_weight_,   0.0, 1, 1.0, 1, 0.1 );
  CheckParameterLowerUpperBound("barcode-filter-minreads", barcode_filter_minreads_, 0,   1, 0,   0, 20  );
  CheckParameterLowerUpperBound("barcode-error-filter",    barcode_error_filter_,    0.0, 1, 2.0, 2, 0.0 );
  CheckParameterLowerUpperBound("barcode-filter-postpone", barcode_filter_postpone_, 0,   1, 2,   1, 0   );
  if (barcode_filter_postpone_ != 1) { barcode_filter_weight_ = 1.0; }

  opts.GetOption(classifier_ignore_flows_, "0,0", '-', "barcode-ignore-flows");
  barcode_ignore_flows_           = (classifier_ignore_flows_.size() == 2) and (classifier_ignore_flows_[0]>=0)
		                            and (classifier_ignore_flows_[0]<classifier_ignore_flows_[1]);
  windowSize_                     = opts.GetFirstInt    ('-', "window-size", DPTreephaser::kWindowSizeDefault_);
  barcode_bam_tag_		          = opts.GetFirstBoolean('-', "barcode-bam-tag", false);
  skip_droop_                     = opts.GetFirstBoolean('-', "skipDroop", true);


  // --- First phase of initialization: parse barcode list file
  LoadBarcodesFromDataset(datasets, keys, flow_order);
  LoadHandlesFromArgs(opts, flow_order, structure);
  if (have_handles_){
    datasets.barcode_filters()["handle_mode"] = handle_mode_;
    datasets.barcode_filters()["handle_cutoff"] = handle_cutoff_;
    datasets.barcode_filters()["handles"] = (int)handle_sequence_.size();
  }

  // For now only option to get the properties of the barcode set
  if ((compute_dmin or score_auto_config_) and not have_ambiguity_codes_)
    ComputeHammingDistance();

  // Set options with limits that potentially depend on barcode set Hamming distance
  SetClassificationParams(opts.GetFirstInt    ('-', "barcode-mode", 2),
                          opts.GetFirstDouble ('-', "barcode-cutoff", 1.0),
                          opts.GetFirstDouble ('-', "barcode-separation", 2.0));


  // --- Pretty print barcode processing settings

  //printf("Barcode classification settings:\n"); // is at top
  // Only print classification settings if we're using the classifier.
  if (dataset_in_use_) {
    printf("   Barcode mask file        : %s\n", barcode_mask_filename_.c_str());
    printf("   Barcode set name         : %s\n", barcode_id.c_str());
    printf("   Number of barcodes       : %d\n", num_barcodes_);
    printf("   Scoring mode             : %d\n", score_mode_);
    printf("   Cutoff threshold         : %1.1lf\n", score_cutoff_);
    printf("   Separation threshold     : %1.2lf\n", score_separation_);
    printf("   Barcode filter threshold : %1.6f (0.0 = disabled)\n", barcode_filter_);
    printf("   Barcode error filter     : %1.3f\n", barcode_error_filter_);
    printf("   Barcode filter minreads  : %d (0 = disabled)\n", barcode_filter_minreads_);
    printf("   Barcode filter filename  : %s\n", barcode_filter_filename_.c_str());
    printf("   Generation of XB bam-tag : %s (number of base errors during barcode classification)\n", (barcode_bam_tag_ ? "on" : "off"));
    if (barcode_ignore_flows_)
      printf("   Ignoring flows %d to %d for barcode classification.\n",classifier_ignore_flows_[0],classifier_ignore_flows_[1]);
    printf("\n");
  }
}


// ------------------------------------------------------------------------
// Extract barcode information from datasets

void BarcodeClassifier::LoadBarcodesFromDataset(BarcodeDatasets& datasets, const vector<KeySequence>& keys, const ion::FlowOrder& flow_order)
{
  string barcode_id  = datasets.barcode_config().get("barcode_id","noID").asString();
  barcode_.reserve(datasets.num_read_groups());

  // --- Loop loading individual barcode informations
  for (int rg_idx = 0; rg_idx < datasets.num_read_groups(); ++rg_idx) {
    Json::Value& read_group = datasets.read_group(rg_idx);

    // Accommodate for change in json format
    string barcode_sequence, barcode_adapter;
    if (read_group.isMember("barcode_sequence")){
      barcode_sequence = read_group.get("barcode_sequence", "").asString();
      barcode_adapter  = read_group.get("barcode_adapter", "").asString();
    }
    else if (read_group.isMember("barcode")){
      barcode_sequence = read_group["barcode"].get("barcode_sequence", "").asString();
      barcode_adapter  = read_group["barcode"].get("barcode_adapter", "").asString();
    }

    // Group for non-barcoded reads
    if (barcode_sequence.empty()) {
      if (no_barcode_read_group_ >= 0) {
    	cerr << "BarcodeClassifier WARNING: Dataset " << barcode_id << " has more than one non-barcoded read group." << endl;
    	cout << "WARNING: Dataset " << barcode_id << " has more than one non-barcoded read group." << endl;
      }
      else
        no_barcode_read_group_ = rg_idx;
      continue;
    }

    num_barcodes_++;
    barcode_.push_back(Barcode());
    barcode_.back().mask_index = read_group.get("index",0).asInt(); // Only used for barcodeMask
    barcode_.back().read_group_index = rg_idx;
    barcode_.back().flow_seq.assign(flow_order.num_flows(), 0);
    barcode_.back().end_flow = -1;
	barcode_.back().adapter_start_flow = -1;

	// All barcodes share the same key in front of them
    barcode_.back().full_barcode = keys[0].bases();
    int key_length = barcode_.back().full_barcode.length();
    barcode_.back().full_barcode += barcode_sequence;
    int key_barcode_length = barcode_.back().full_barcode.length();
    barcode_.back().full_barcode += barcode_adapter;
    int key_barcode_adapter_length = barcode_.back().full_barcode.length();

    // Check for non-ACGT characters
    if (std::string::npos != barcode_.back().full_barcode.find_first_not_of("ACGT")){
      have_ambiguity_codes_ = true;
      continue;
    }

    int flow = 0;
    int curBase = 0;
    int end_length = -1;

    while (curBase < key_barcode_adapter_length and flow < flow_order.num_flows()) {

      // Increment bases for this flow
      while (curBase < key_barcode_adapter_length and barcode_.back().full_barcode[curBase] == flow_order[flow]) {
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
  } // --- End Loop loading individual barcode informations

  // Only print classification settings if we're using the classifier.
  if (dataset_in_use_ and no_barcode_read_group_ < 0)
    cout << "   INFO: Dataset " << barcode_id << " does not have a non-barcoded read group." << endl;

  if (not have_ambiguity_codes_) {
    // And loop through barcodes again to determine maximum amount of flows after start flow has been determined
    end_flow_synch_ = true;
    int barcode_flows, barcode_min_flows = -1;
    for (unsigned int bc=0; bc<barcode_.size(); bc++) {
      barcode_flows = barcode_.at(bc).adapter_start_flow - barcode_min_start_flow_;
      barcode_max_flows_ = max( barcode_max_flows_, barcode_flows);
      if (barcode_min_flows < 0 or barcode_flows < barcode_min_flows)
        barcode_min_flows = barcode_flows;
      if (barcode_full_flows_ < 0)
        barcode_full_flows_ = barcode_.at(bc).num_flows;
      else if (barcode_full_flows_ != barcode_.at(bc).num_flows)
        end_flow_synch_ = false;
    }
    if (dataset_in_use_ and barcode_min_flows >= 0 and barcode_min_flows != barcode_max_flows_)
      cout << "   WARNING: Barcode set is not flow space synchronized. Barcodes range from "
           << barcode_min_flows << " to " << barcode_max_flows_ << " flows." << endl;
  }

  // Export barcode_max_flows_ to datasets structure
  datasets.SetBCmaxFlows(barcode_max_flows_);
};


// ------------------------------------------------------------------------

void BarcodeClassifier::SetClassificationParams(int mode, double cutoff, double separation)
{
  score_mode_                     = mode;
  score_cutoff_                   = cutoff;
  score_separation_               = separation;

  CheckParameterLowerUpperBound("barcode-mode",          score_mode_,      0,1, 5,1, 2);
  if (have_ambiguity_codes_ and score_mode_ != 0){
    cout << "   WARNING: Non-ACGT characters detected. Setting score-mode=0." << endl;
    score_mode_ =0;
    return;
  }

  // Do we have minimum distance information available?
  if (hamming_dmin_ > 0) {
    if (dataset_in_use_ and score_auto_config_)
      cout << "   Auto-config of barcode scoring parameters enabled." << endl;

    // Possible settings vary by score mode
    if (score_mode_ == 1){
      double max_cutoff = (double)((hamming_dmin_-1) / 2);
      if (score_auto_config_)
        score_cutoff_ = max_cutoff;
      else
        CheckParameterLowerUpperBound("barcode-cutoff",    score_cutoff_,    0.0,1, max_cutoff,1, max_cutoff);
    }
    else {
      // Thought: Make cutoff barcode length dependent?
      CheckParameterLowerUpperBound("barcode-cutoff",    score_cutoff_,    0.5,1, 0.0,0, 1.0);

      double def_separation = (double)(hamming_dmin_) * 0.5;
      if (score_auto_config_)
        score_separation_ = def_separation;
      else {
        double min_separation = (double)(hamming_dmin_) * 0.25;
        double max_separation = (double)(hamming_dmin_) * 0.75;

        CheckParameterLowerUpperBound("barcode-separation",score_separation_,min_separation,1, max_separation, 1, def_separation);
      }

    }
  }
  else if (dataset_in_use_) {
    // No distance information available - simple sanity checks
    cout << "   No Hamming distance information available." << endl;
    if (score_auto_config_) {
      cout << "   WARNING: Auto-config of barcode scoring parameters not possible." << endl;
      cerr << " BarcodeClassifier WARNING: Auto-config of barcode scoring parameters not possible: No Hamming distance available." << endl;
    }
    CheckParameterLowerUpperBound("barcode-cutoff",     score_cutoff_,     0.0,1, 0.0,0, 1.0);
    CheckParameterLowerUpperBound("barcode-separation", score_separation_, 0.0,1, 0.0,0, 2.5);
  }
};


// ------------------------------------------------------------------------

void BarcodeClassifier::ComputeHammingDistance()
{
  if (not dataset_in_use_)
    return;

  int max_compare_flows = 0;
  hamming_dmin_ = -1;

  for (int bc_a = 0; bc_a < num_barcodes_-1; ++bc_a) {
    for (int bc_b = bc_a+1; bc_b < num_barcodes_; ++bc_b) {

      max_compare_flows = min(barcode_[bc_a].adapter_start_flow, barcode_[bc_b].adapter_start_flow);
      int my_distance = 0;

      for (int flow = barcode_min_start_flow_; flow < max_compare_flows; ++flow) {
        if (barcode_ignore_flows_ and  (flow >= classifier_ignore_flows_[0]) and (flow < classifier_ignore_flows_[1]))
          continue;
        my_distance += abs(barcode_[bc_a].flow_seq[flow] - barcode_[bc_b].flow_seq[flow]);
      }

      if (hamming_dmin_ < 0 or my_distance < hamming_dmin_)
        hamming_dmin_ = my_distance;

    }
  }
  // Zero distance means there is a duplicate entry in the barcode set that needs to be removed!
  if (hamming_dmin_ == 0){
    cout << "ERROR: Barcode set contains duplicate barcode entries. Please fix the barcode set before proceeding." << endl;
    exit(EXIT_FAILURE);
  }
  cout << "   Computed minimum Hamming distance of barcode set as d_min = " << hamming_dmin_ << endl;

};


// --------------------------------------------------------------------------

void BarcodeClassifier::BuildPredictedSignals(const ion::FlowOrder& flow_order, float cf, float ie, float dr)
{
  if (num_barcodes_ == 0)
    return;

  DPTreephaser treephaser(flow_order, windowSize_);
  BasecallerRead basecaller_read;
  if (skip_droop_)
	treephaser.SetModelParameters(cf, ie);
  else
    treephaser.SetModelParameters(cf, ie, dr);

  for (int bc = 0; bc < num_barcodes_; ++bc) {
    basecaller_read.sequence.assign(barcode_[bc].full_barcode.begin(),barcode_[bc].full_barcode.end());
    basecaller_read.prediction.assign(flow_order.num_flows(), 0);

    treephaser.Simulate(basecaller_read, barcode_[bc].num_flows);

    barcode_[bc].predicted_signal.swap(basecaller_read.prediction);
  }
}


// ------------------------------------------------------------------------
// trivially simple base space matching

int  BarcodeClassifier::SimpleBaseSpaceClassification(const BasecallerRead& basecaller_read)
{
  int bc_len;
  for (int bc = 0; bc < num_barcodes_; ++bc) {
    bc_len = barcode_[bc].full_barcode.length();
    if (basecaller_read.sequence.size() < barcode_[bc].full_barcode.length())
      continue;

    int i=0;
    while (i<bc_len and isBaseMatch(basecaller_read.sequence[i],barcode_[bc].full_barcode[i]))
      ++i;
    if (i==bc_len)
      return bc; // perfect match found
  }
  return -1;
}

// ------------------------------------------------------------------------

int  BarcodeClassifier::FlowAlignClassification(const ProcessedRead &processed_read, const vector<int>& base_to_flow, int& best_errors)
{
    int best_barcode = -1;
	best_errors = 1 + (int)score_cutoff_; // allows match with fewer than this many errors when in bc_score_mode 1

    for (int bc = 0; bc < num_barcodes_; ++bc) {

      int num_errors = 0;
	  int flow = 0;
	  int base = 0;
	  bool evaluate_me = true;
      for (; flow < barcode_[bc].adapter_start_flow; ++flow) {
        int hp_length = 0;
        while (base < processed_read.filter.n_bases and base_to_flow[base] == flow) {
          base++;
          hp_length++;
        }
        if (barcode_ignore_flows_)
          evaluate_me = flow < classifier_ignore_flows_[0] or flow >= classifier_ignore_flows_[1];
        if (flow >= barcode_min_start_flow_ and evaluate_me) {
          if (flow < barcode_[bc].num_flows-1)
            num_errors += abs(barcode_[bc].flow_seq[flow] - hp_length);
          else
        	num_errors += max(0, (barcode_[bc].flow_seq[flow] - hp_length));
        }
      }

      if (num_errors < best_errors) {
        best_errors = num_errors;
        best_barcode = bc;
      } else if (num_errors == best_errors){
        // We only assign a barcode when there is a unique best barcode
        best_barcode = -1;
      }
    }
    return best_barcode;
};

// ------------------------------------------------------------------------

int  BarcodeClassifier::SignalSpaceClassification(const BasecallerRead& basecaller_read, float& best_distance, int& best_errors,
                                                  vector<float>& best_bias, int& filtered_zero_errors)
{
    int best_barcode     = -1;
    best_errors          =  0;
    filtered_zero_errors = -1;
    best_distance        = score_cutoff_ + score_separation_;
    float second_best_distance = 1e20;

    for (int bc = 0; bc < num_barcodes_; ++bc) {

      int num_errors = 0;
      float distance = 0.0;
      vector<float> bias(barcode_max_flows_,0);

      for (int flow = barcode_min_start_flow_; flow < barcode_[bc].adapter_start_flow; ++flow) {

        if (barcode_ignore_flows_ and  (flow >= classifier_ignore_flows_[0]) and (flow < classifier_ignore_flows_[1]))
          continue;

    	// Compute Bias
    	bias.at(flow-barcode_min_start_flow_) = basecaller_read.normalized_measurements.at(flow) - barcode_[bc].predicted_signal.at(flow);
    	// Thresholding of measurements to a range of [0,2]
    	double acting_measurement = basecaller_read.normalized_measurements.at(flow);
    	acting_measurement = max(min(acting_measurement, (double)barcode_max_hp_),0.0);
    	// Compute distance
    	double residual = barcode_[bc].predicted_signal.at(flow) - acting_measurement;
    	if (flow == barcode_[bc].num_flows-1)
          residual = max(residual, 0.0);

    	(score_mode_ == 2) ? (distance += residual * residual) : (distance += fabs(residual));

        // Compute hard decision errors - approximation from predicted values
        if (flow < barcode_[bc].num_flows-1)
          num_errors += round(fabs(barcode_[bc].predicted_signal.at(flow) - basecaller_read.prediction[flow]));
        else
          num_errors += round(max(barcode_[bc].predicted_signal.at(flow) - basecaller_read.prediction[flow], (float)0.0));
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
      if (best_errors == 0)
        filtered_zero_errors = best_barcode;
      best_barcode = -1;
    }
    return best_barcode;
};


// ------------------------------------------------------------------------

bool BarcodeClassifier::MatchesBarcodeSignal(const BasecallerRead& basecaller_read)
{
  if ((not dataset_in_use_) or (num_barcodes_ == 0))
    return false;

  int   best_barcode         = -1;
  int   best_errors          =  0;
  int   filtered_zero_errors = -1;
  float best_distance        = 0.0;
  vector<float> best_bias;

  best_barcode = SignalSpaceClassification(basecaller_read, best_distance, best_errors, best_bias, filtered_zero_errors);

  return (best_barcode >= 0);
};

// ------------------------------------------------------------------------

int  BarcodeClassifier::ProportionalSignalClassification(const BasecallerRead& basecaller_read, float& best_distance, int& best_errors,
                                                  vector<float>& best_bias, int& filtered_zero_errors)
{
    int best_barcode     = -1;
    best_errors          =  0;
    filtered_zero_errors = -1;
    best_distance        = score_cutoff_ + score_separation_;
    float second_best_distance = 1e20;

    for (int bc = 0; bc < num_barcodes_; ++bc) {

      int num_errors = 0;
      float distance = 0.0;
      vector<float> bias(barcode_max_flows_,0);

      for (int flow = barcode_min_start_flow_; flow < barcode_[bc].adapter_start_flow; ++flow) {

        if (barcode_ignore_flows_ and  (flow >= classifier_ignore_flows_[0]) and (flow < classifier_ignore_flows_[1]))
          continue;
        double proportional_signal = (basecaller_read.normalized_measurements.at(flow)+0.5)/(barcode_[bc].predicted_signal.at(flow)+0.5);

      // Compute Bias
      bias.at(flow-barcode_min_start_flow_) = 1.0-proportional_signal;

      // Compute distance
      double residual = 1.0-proportional_signal;
      // overcall allowed here but not undercall
      if (flow == barcode_[bc].num_flows-1)
          residual = max(residual, 0.0);

      (score_mode_ == 4) ? (distance += residual * residual) : (distance += fabs(residual));

        // Compute hard decision errors - approximation from predicted values
        if (flow < barcode_[bc].num_flows-1)
          num_errors += round(fabs(barcode_[bc].predicted_signal.at(flow) - basecaller_read.prediction[flow]));
        else
          num_errors += round(max(barcode_[bc].predicted_signal.at(flow) - basecaller_read.prediction[flow], (float)0.0));
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
      if (best_errors == 0)
        filtered_zero_errors = barcode_[best_barcode].read_group_index;
      best_barcode = -1;
    }
    return best_barcode;
};

// ------------------------------------------------------------------------


bool BarcodeClassifier::AdapterValidation(const BasecallerRead& basecaller_read, int& best_barcode, int& filtered_read_group)
{
  filtered_read_group = -1;

  // zero means disabled
  if (adapter_cutoff_ == 0)
    return true;

  if (best_barcode < 0)
    return false;

  int num_adapter_flows = barcode_[best_barcode].num_flows - barcode_[best_barcode].adapter_start_flow;
  if (num_adapter_flows == 0)
    return true;

  // Sum up barcode adapter residual
  float residual, residual_2 = 0.0;

  int flow = barcode_[best_barcode].adapter_start_flow;
  for (; flow < barcode_[best_barcode].num_flows-1; ++flow) {
    residual = basecaller_read.normalized_measurements.at(flow) - barcode_[best_barcode].predicted_signal.at(flow);
    residual_2 += residual * residual;
  }
  // There might be additional template bases in the last adapter flow
  if (barcode_[best_barcode].num_flows > barcode_[best_barcode].adapter_start_flow){
    residual = std::min(0.0f, (basecaller_read.normalized_measurements.at(flow) - barcode_[best_barcode].predicted_signal.at(flow)));
    residual_2 += residual * residual;
  }

  if (residual_2 > num_adapter_flows * adapter_cutoff_) {
    filtered_read_group = barcode_[best_barcode].read_group_index;
    best_barcode = -1;
    return false;
  }
  else
    return true;
}


// ------------------------------------------------------------------------

/*
 * flowSpaceTrim - finds the closest barcode in flowspace to the sequence passed in,
 * and then trims to exactly the expected flows so it can be error tolerant in base space
 */

void BarcodeClassifier::ClassifyAndTrimBarcode(int read_index, ProcessedRead &processed_read, const BasecallerRead& basecaller_read,
    const vector<int>& base_to_flow)
{
  if (not dataset_in_use_)
    return;

  int   best_barcode  = -1;
  switch (score_mode_){
    case 1: // looks at flow-space absolute error counts, not ratios
      best_barcode = FlowAlignClassification(processed_read, base_to_flow, processed_read.barcode_n_errors);
      break;
    case 2: // Minimize L2 distance for score_mode_ == 2
    case 3: // Minimize L1 distance for score_mode_ == 3
      best_barcode = SignalSpaceClassification(basecaller_read, processed_read.barcode_distance, processed_read.barcode_n_errors,
                                                 processed_read.barcode_bias, processed_read.barcode_filt_zero_error);
      break;
    case 4: //L2 for score mode 4
    case 5: //L1 for score mode 5
      best_barcode = ProportionalSignalClassification(basecaller_read, processed_read.barcode_distance, processed_read.barcode_n_errors,
                                                   processed_read.barcode_bias, processed_read.barcode_filt_zero_error);
      break;
    default: // Trivial base space comparison
      best_barcode = SimpleBaseSpaceClassification(basecaller_read);
  }

  // Optionally verify barcode adapter
  if (score_mode_ > 0)
    AdapterValidation(basecaller_read, best_barcode, processed_read.barcode_adapter_filtered);


  // -------- Classification done, now accounting ----------

  if (best_barcode == -1) {
    int x, y;
    barcode_mask_.IndexToRowCol (read_index, y, x);
    barcode_mask_.SetBarcodeId(x, y, 0);
    if (no_barcode_read_group_ >= 0)
      processed_read.read_group_index = no_barcode_read_group_;
    return;
  }

  const Barcode& bce = barcode_[best_barcode];

  int x, y;
  barcode_mask_.IndexToRowCol (read_index, y, x);
  barcode_mask_.SetBarcodeId(x, y, (uint16_t)bce.mask_index);
  processed_read.read_group_index = bce.read_group_index;

  if(barcode_bam_tag_)
	processed_read.bam.AddTag("XB","i", processed_read.barcode_n_errors);

  if (not trim_barcodes_)
    return;

  // Account for barcode + barcode adapter bases
  if (score_mode_ == 0){
    processed_read.filter.n_bases_barcode = min(processed_read.filter.n_bases, (int)barcode_[best_barcode].full_barcode.length());
  }
  else {
    processed_read.filter.n_bases_barcode = 0;
    while (processed_read.filter.n_bases_barcode < processed_read.filter.n_bases and base_to_flow[processed_read.filter.n_bases_barcode] < bce.num_flows-1)
      processed_read.filter.n_bases_barcode++;

    int last_homopolymer = bce.last_homopolymer;
    while (processed_read.filter.n_bases_barcode < processed_read.filter.n_bases and base_to_flow[processed_read.filter.n_bases_barcode] < bce.num_flows and last_homopolymer > 0) {
      processed_read.filter.n_bases_barcode++;
      last_homopolymer--;
    }
  }

  // Propagate current 5' trimming point
  processed_read.filter.n_bases_prefix = processed_read.filter.n_bases_tag = processed_read.filter.n_bases_barcode;
  // And now do handle classification and trimming
  ClassifyAndTrimHandle(read_index, best_barcode, processed_read, basecaller_read, base_to_flow);

}

// ------------------------------------------------------------------------

void BarcodeClassifier::LoadHandlesFromArgs(OptArgs& opts,
       const ion::FlowOrder& flow_order, const Json::Value& structure)
{
  handle_sequence_  = opts.GetFirstStringVector ('-', "barcode-handles", "");
  if (handle_sequence_.empty()) {
    if (structure.isMember("handle")){
      vector<string> keys = structure["handle"].getMemberNames();
      for (unsigned int k=0; k<keys.size(); ++k)
        handle_sequence_.push_back(structure["handle"][keys[k]].asString());
    }
    else{
      have_handles_ = false;
      return;
    }
  }

  cout << "   Have " << handle_sequence_.size() << " handles. ";
  //for (unsigned int iHandle=0; iHandle<handle_sequence_.size(); ++iHandle){
  //  cout << handle_sequence_[iHandle] << ',';
  //}
  cout << endl;

  handle_mode_   =   opts.GetFirstInt    ('-', "handle-mode", DEF_HANDLE_MODE);
  handle_cutoff_ =   opts.GetFirstInt    ('-', "handle-cutoff", DEF_HANDLE_CUTOFF);
  cout << "   handle-mode              : " << handle_mode_ << endl;
  cout << "   handle-cutoff            : " << handle_cutoff_ << endl;

  if (not end_flow_synch_ and handle_mode_ !=0){
    cerr << "WARNING: Handles do not have a distinct start flow. Using base space matching handle-mode=0!" << endl;
    handle_mode_ = 0;
  }
  if (handle_mode_ == 0){
    have_handles_ = true;
    return;
  }

  // Build handle flow information for mode 1
  handle_.resize(handle_sequence_.size());
  int have_ambiguity_codes = false;

  for (unsigned int iHandle=0; iHandle<handle_sequence_.size(); ++iHandle){

    // Check for non-ACGT characters
    if (std::string::npos != handle_sequence_[iHandle].find_first_not_of("ACGT")){
      have_ambiguity_codes = true;
      break;
    } else if (handle_sequence_[iHandle].length() == 0){
      cerr << "ERROR: Barcode handle " << (iHandle+1) << " has no sequence specified" << endl;
      exit(EXIT_FAILURE);
    }

    // Build flow signal
    handle_[iHandle].base_seq = handle_sequence_[iHandle];
    handle_[iHandle].flow_seq.clear();

    // Start at last barcode flow to account for potential overlap
    int flow = max(0, barcode_full_flows_ -1);
    unsigned int curBase = 0;

    while (curBase < handle_[iHandle].base_seq.length() and flow < flow_order.num_flows()){
      int myHP = 0;
      while (curBase < handle_[iHandle].base_seq.length() and handle_[iHandle].base_seq[curBase] == flow_order[flow]) {
        myHP++;
        curBase++;
      }
      handle_[iHandle].flow_seq.push_back(myHP);
      flow++;
    }
  }

  if (have_ambiguity_codes){
    cerr << "WARNING: Non-ACTG characters detected in handle sequences. Using base space matching handle-mode=0!" << endl;
    handle_mode_ = 0;
    handle_.clear();
  }
  have_handles_ = true;
}

// ------------------------------------------------------------------------

void BarcodeClassifier::ClassifyAndTrimHandle(int read_index, int best_barcode,
    ProcessedRead &processed_read, const BasecallerRead& basecaller_read,
    const vector<int>& base_to_flow)
{
  if  (is_control_dataset_ or (not dataset_in_use_) or (not have_handles_))
      return;

  if (processed_read.read_group_index == no_barcode_read_group_)
    return;

  int   best_handle  = -1;
  switch (handle_mode_){
    case 1: // looks at flow-space absolute error counts
      best_handle = HandleFlowAlign(best_barcode, processed_read, base_to_flow);
      break;

    default: // base space comparison
      best_handle = HandleBaseSpaceClassification(basecaller_read, processed_read);
  }

  // After classification we push reads where we did not find a valid handle into no-match
  if (best_handle < 0){
    int x, y;
    barcode_mask_.IndexToRowCol (read_index, y, x);
    barcode_mask_.SetBarcodeId(x, y, 0);
    if (no_barcode_read_group_ >= 0){
      processed_read.barcode_handle_filtered = processed_read.read_group_index;
      processed_read.read_group_index = no_barcode_read_group_;
      processed_read.handle_n_errors = 1+handle_cutoff_;
    }
    return;
  }

  // Trimming & accounting of handle bases

  processed_read.handle_index = best_handle;
  if (not trim_barcodes_)
    return;

  if (handle_mode_ ==1){ // Flow alignment

    int n_bases = 0;
    int last_hp = - handle_.at(best_handle).flow_seq.at(handle_.at(best_handle).flow_seq.size()-1);
    int max_flow = barcode_full_flows_-1 + handle_.at(best_handle).flow_seq.size();

    while (n_bases < processed_read.filter.n_bases and base_to_flow[n_bases] < max_flow){
      if (base_to_flow[n_bases] == max_flow-1)
        ++last_hp;
      ++n_bases;
    }
    processed_read.filter.n_bases_barcode = n_bases - max (0, last_hp);
  }
  else { // default string matching
    processed_read.filter.n_bases_barcode += handle_sequence_[best_handle].length();
    processed_read.filter.n_bases_barcode = min(processed_read.filter.n_bases_barcode, processed_read.filter.n_bases);
  }

  // Save tag & Propagate current 5' trimming point
  processed_read.bam.AddTag("ZK", "Z", handle_sequence_[best_handle]);
  processed_read.filter.n_bases_prefix = processed_read.filter.n_bases_tag = processed_read.filter.n_bases_barcode;
}

// ------------------------------------------------------------------------
// Looks for perfect matches (with ambiguity symbols)

int BarcodeClassifier::HandleBaseSpaceClassification(const BasecallerRead& basecaller_read, const ProcessedRead &processed_read)
{

  int best_handle = -1;
  for (unsigned int iHandle=0; iHandle<handle_sequence_.size(); ++iHandle){

    // Test available length
    if (processed_read.filter.n_bases - processed_read.filter.n_bases_prefix < (int)handle_sequence_[iHandle].length())
      continue;

    unsigned int base = 0;
    for (; base<handle_sequence_[iHandle].length(); ++base){
      if (not isBaseMatch(handle_sequence_[iHandle].at(base),
            basecaller_read.sequence.at(base+processed_read.filter.n_bases_prefix) ))
        break;
    }
    if (base == handle_sequence_[iHandle].length()){
      best_handle = iHandle;
      break;
    }
  }
  return best_handle;
}

// ------------------------------------------------------------------------

int  BarcodeClassifier::HandleFlowAlign(int best_barcode, ProcessedRead &processed_read, const vector<int>& base_to_flow)
{
    int best_handle = -1;
    int best_errors = 1 + handle_cutoff_; // allows match with this many errors minimum when in bc_score_mode 1

    // Increment bases beyond barcode
    int start_base = 0;
    int last_bc_read_hp = 0;
    int last_bc_hp = barcode_[best_barcode].flow_seq.at(barcode_[best_barcode].num_flows-1);

    while (start_base < (int)base_to_flow.size() and base_to_flow[start_base]<barcode_full_flows_-1)
      ++start_base;
    // Get last barcode flow HP size
    while (start_base < (int)base_to_flow.size() and base_to_flow[start_base]==barcode_full_flows_-1){
      ++start_base;
      ++last_bc_read_hp;
    }

    for (unsigned int hd = 0; hd < handle_.size(); ++hd) {

      int num_errors = 0;
      int base = start_base;

      // Handle flow 0 (last barcode flow) separately
      if (handle_[hd].flow_seq.at(0)>0){
        num_errors += abs(handle_[hd].flow_seq.at(0) +last_bc_hp -last_bc_hp);
      }

      for (int flow=1; flow < (int)handle_[hd].flow_seq.size(); ++flow) {
        int hp_length = 0;
        while (base < processed_read.filter.n_bases and base_to_flow[base] == (flow + barcode_full_flows_-1)) {
          base++;
          hp_length++;
        }
        if (flow < (int)handle_[hd].flow_seq.size()-1)
          num_errors += abs(handle_[hd].flow_seq.at(flow) - hp_length);
        else
          num_errors += max(0, (handle_[hd].flow_seq.at(flow) - hp_length));
      }

      if (num_errors < best_errors) {
        best_errors = num_errors;
        best_handle = hd;
      } 
    }
    processed_read.handle_n_errors = best_errors;
    return best_handle;
};


// ------------------------------------------------------------------------

void BarcodeClassifier::Close(BarcodeDatasets& datasets)
{
  // Nothing to do if dataset is not in use
  // Do not filter control dataset and do not generate output files
  if  (is_control_dataset_ or (not dataset_in_use_))
    return;


  if (0 != barcode_mask_.WriteRaw (barcode_mask_filename_.c_str()))
    fprintf (stderr, "BarcodeClassifier: Cannot write mask file file: %s\n", barcode_mask_filename_.c_str());

  // Write barcode filter information into datasets json for python filters
  datasets.barcode_filters()["filter_frequency"]  = barcode_filter_;
  datasets.barcode_filters()["filter_minreads"]   = (Json::Int64)barcode_filter_minreads_;
  datasets.barcode_filters()["filter_errors_hist"]= barcode_error_filter_;
  datasets.barcode_filters()["filter_postpone"]   = (Json::Int64)barcode_filter_postpone_;
  // not used by python filters but useful information if we are looking at the json file:
  datasets.barcode_filters()["filter_weight"]     = barcode_filter_weight_;
  datasets.barcode_filters()["classifier_mode"]   = (Json::Int64)score_mode_;
  datasets.barcode_filters()["classifier_cutoff"] = score_cutoff_;
  datasets.barcode_filters()["classifier_separation"] = score_separation_;
  if (hamming_dmin_ >=0)
    datasets.barcode_filters()["hamming_distance"] = hamming_dmin_;

  // Generate Filter Thresholds for barcode filtering (minreads filter only active if filtering done in basecaller)
  unsigned int read_threshold = 0;
  if(barcode_filter_postpone_ == 0)
	read_threshold = barcode_filter_minreads_;

  // Adjust filter threshold based on barcode frequencies if desired
  if (barcode_filter_postpone_ < 2 and barcode_filter_ > 0.0) {
	unsigned int max_read_count = 0;
	for (Json::Value::iterator rg = datasets.read_groups().begin(); rg != datasets.read_groups().end(); ++rg){
	  bool has_barcode = (*rg).isMember("barcode_sequence") or (*rg).isMember("barcode");
      if ((*rg).isMember("read_count") and has_barcode)
        max_read_count = max(max_read_count, (*rg)["read_count"].asUInt());
	}
	read_threshold = max(read_threshold, (unsigned int)((double)max_read_count*barcode_filter_*barcode_filter_weight_));
  }

  // Below is the actual filtering of barcodes and file writing
  // We still want to create barcode statistics, even if all filters are turned off.
  // We always filter barcode groups with zero reads in them

  FILE *ffile = fopen(barcode_filter_filename_.c_str(),"wt");
  if (ffile)
    fprintf(ffile, "BarcodeId,BarcodeName,NumReads,Include\n");

  for (Json::Value::iterator rg = datasets.read_groups().begin(); rg != datasets.read_groups().end(); ++rg) {

    string barcode_sequence, barcode_name;
    double one_error = 0.0, two_errors = 0.0;
    unsigned int adapter_filtered = 0;

    // We assume we consume a file written by the most up to date BaserCaller executable
    // but may hve an older pipeline version

    if ((*rg).isMember("barcode")) {
      adapter_filtered = (*rg)["barcode"]["barcode_adapter_filtered"].asUInt();
      one_error        = (*rg)["barcode"]["barcode_errors_hist"][1].asDouble();
      two_errors       = (*rg)["barcode"]["barcode_errors_hist"][2].asDouble();
      if ((*rg).isMember("barcode_sequence")){
        barcode_sequence = (*rg).get("barcode_sequence", "").asString();
        barcode_name     = (*rg).get("barcode_name", "").asString();
      }
      else{
        barcode_sequence = (*rg)["barcode"].get("barcode_sequence", "").asString();
        barcode_name     = (*rg)["barcode"].get("barcode_name", "").asString();
      }
    }

    if ((*rg).isMember("read_count") and not barcode_sequence.empty()) {

      unsigned int read_count = (*rg)["read_count"].asUInt();
      bool i_am_filtered = false;
      bool filter_this_bc = barcode_filter_named_ or (*rg)["sample"].asString() == "none";

      // Initial filtering based on number of reads in read group
      if (filter_this_bc) {
        i_am_filtered = (read_count > read_threshold) ? false : true;
      }

      // Further filtering based on average number of errors
      if((barcode_error_filter_ > 0) and ((*rg).isMember("barcode") or (*rg).isMember("barcode_errors_hist"))
         and filter_this_bc and (not i_am_filtered) and (not barcode_filter_postpone_)) {
        i_am_filtered = ((one_error + 2.0*two_errors) / (double)read_count) > barcode_error_filter_;
      }

      // Filter read groups where a too large proportion of reads failed adapter verification
      // Likely to be a highly contaminated sample and should not be analyzed
      if (not i_am_filtered)
      {
        //unsigned int adapter_filtered = (*rg)["barcode_adapter_filtered"].asUInt();
        i_am_filtered = (5*adapter_filtered > read_count) ? true : false;
        if (i_am_filtered)
          cerr << "WARNING: Read group " << barcode_name << " is likely to be contaminated and is being filtered." << endl;
      }

      // Set status in datasets
      (*rg)["filtered"] = i_am_filtered;
      if (ffile)
        fprintf(ffile, "%s,%s,%i,%i\n", barcode_name.c_str(),
                                        barcode_sequence.c_str(),
                                        read_count, (int)(not i_am_filtered));
      }
    }
    if (ffile)
      fclose ( ffile );
}

// =======================================================================
// End Barcode Classifier Class XXX


EndBarcodeClassifier::EndBarcode::EndBarcode()
  : end_barcode(false), bc_start_hp(0), adapter_start_hp(0)
{
};


void EndBarcodeClassifier::PrintHelp()
{
  cout << "End Barcode classification options:" << endl;
  cout << "     --end-barcodes                BOOL       Activates/Deactivates end barcde classification [on]" << endl;
  cout << "     --end-barcode-mode            INT        Selects barcode classification algorithm [1]" << endl;
  cout << "     --end-barcode-cutoff          INT        Maximum score to call a barcode [3]" << endl;
  cout << "     --end-bc-adapter-cutoff       INT        Maximum score to accept barcode adapter [3]" << endl;
  // Handle options are listed in the start barcode classifier
  cout << endl;
}


// ----------------------------------------------------------------------------

EndBarcodeClassifier::EndBarcodeClassifier(OptArgs& opts, BarcodeDatasets& datasets,
        ion::FlowOrder* flow_order, Mask* MaskPointer, const Json::Value& structure)
  :  have_end_barcodes_(false), nomatch_read_group_(-1),
     num_end_barcodes_(0), demux_barcode_list_(false),
     flow_order_p_(flow_order), barcode_mask_pointer_(MaskPointer),
     have_handles_(false), handle_filter_(false)
{

  enable_barcodes_            = opts.GetFirstBoolean('-', "end-barcodes", true);
  score_mode_                 = opts.GetFirstInt    ('-', "end-barcode-mode", 1);
  score_cutoff_               = opts.GetFirstInt    ('-', "end-barcode-cutoff", 2);
  score_separation_           = opts.GetFirstInt    ('-', "end-barcode-separation", 2);
  adapter_cutoff_             = opts.GetFirstInt    ('-', "end-bc-adapter-cutoff", 2);
  trim_barcodes_              = opts.GetFirstBoolean('-', "trim-barcodes", true);

  if (not datasets.DatasetInUse() or datasets.IsControlDataset())
    return;

  cout << "End Barcode Classifier:" << endl;
  if (not enable_barcodes_){
    demux_barcode_list_ = false;
    cout << "  Disabled." << endl << endl;
    return;
  }

  // Development option to demultiplex barcodes read from a csv list
  string end_barcode_list     = opts.GetFirstString('-', "end-barcode-list", "");
  demux_barcode_list_         = LoadBarcodesFromCSV(end_barcode_list);

  // Verbose classifier options
  cout << "  end-barcode-mode      : " << score_mode_     << endl;
  cout << "  end-barcode-cutoff    : " << score_cutoff_   << endl;
  cout << "  end-bc-adapter-cutoff : " << adapter_cutoff_ << endl;
  if (demux_barcode_list_){
    cout << "  end-barcode-list      : " << end_barcode_list << endl;
  }

  // Load pcr handle information
  LoadHandlesFromArgs(opts, structure);

  // --- Loop loading individual barcode informations from json
  read_group_.resize(datasets.num_read_groups());

  for (int rg_idx = 0; rg_idx < datasets.num_read_groups(); ++rg_idx) {
    Json::Value& read_group = datasets.read_group(rg_idx);

    if (not read_group.isMember("barcode_sequence") and
        not read_group.isMember("barcode") and
        datasets.read_group_name(rg_idx).find("nomatch") != std::string::npos)
    {
      nomatch_read_group_ = rg_idx;
      continue;
    }

    if (read_group.isMember("end_barcode")){

      if (demux_barcode_list_){
        cerr << "EndBarcodeClassifyer ERROR: Can only use end-barcode-list when there are no end-barcodes in the run plan." << endl;
        exit(EXIT_FAILURE);
      }

      AddBarcode(read_group_[rg_idx],
          read_group["end_barcode"].get("barcode_sequence", "").asString(),
          read_group["end_barcode"].get("barcode_adapter", "").asString());

      read_group_[rg_idx].barcode_name = read_group["end_barcode"].get("barcode_name", "").asString();
      // The command line argument to de-multiplex a barcode list takes precedence
      if (demux_barcode_list_){
        read_group_[rg_idx].end_barcode = false;
      }
    }

  }

  // Verbose
  cout << "  Found " << num_end_barcodes_ << " read groups with end barcodes." << endl << endl;

  // Add classifier options to datasets meta information

  datasets.barcode_filters()["end_classifier_mode"]       = (Json::Int64)score_mode_;
  datasets.barcode_filters()["end_classifier_cutoff"]     = (Json::Int64)score_cutoff_;
  datasets.barcode_filters()["end_classifier_separation"] = (Json::Int64)score_separation_;

  // Load and log the starting HP(s) of the bead adapter(s)

  vector<string> trim_adapter = opts.GetFirstStringVector ('-', "trim-adapter", "ATCACCGACTGCCCATAGAGAGGCTGAGAC");
  if (trim_adapter.size() > 0 and trim_adapter.at(0) == "off")
    trim_adapter.clear();
  bead_adapter_start_hp.resize(trim_adapter.size());
  for (unsigned int iA=0; iA < trim_adapter.size(); ++iA){
    bead_adapter_start_hp.at(iA) = GetStartHP(trim_adapter.at(iA));
  }

};

// ----------------------------------------------------------------------------

void EndBarcodeClassifier::AddBarcode(EndBarcode& barcode, const string& bc_sequence, const string& bc_adapter)
{
  barcode.barcode_sequence = bc_sequence;
  barcode.barcode_adapter  = bc_adapter;

  if (bc_sequence.length() >0) {

    barcode.end_barcode = true;
    have_end_barcodes_ = true;
    ++num_end_barcodes_;

    // Check for non-ACGT characters
    if (score_mode_ !=0 and std::string::npos != bc_sequence.find_first_not_of("ACGT")){
      cout << "   WARNING: Non-ACGT characters detected in end-barcode. Setting end-barcode-mode=0." << endl;
      score_mode_ = 0;
    }

    // Add start HPs
    barcode.bc_start_hp      = GetStartHP(bc_sequence);
    barcode.adapter_start_hp = GetStartHP(bc_adapter);

  }
};

int  EndBarcodeClassifier::GetStartHP(const string& base_str)
{
  if (base_str.empty())
    return 0;
  int base_len = (int)base_str.length();
  int hp_len = 1;
  while (hp_len < base_len and base_str.at(hp_len) == base_str.at(0))
    ++hp_len;
  return hp_len;
}

// ----------------------------------------------------------------------------

bool EndBarcodeClassifier::LoadBarcodesFromCSV(string filename)
{
  if (filename.length() ==0)
    return false;

  ifstream bcfile(filename.c_str());
  if (not bcfile.is_open()){
    cerr << "ERROR: Unable to open end barcode list file " << filename << endl;
    exit(1);
  }

  string line, csvfield;
  vector<string> csv_line;
  int line_number = 0;
  int name_idx = -1, seq_idx = -1, adapter_idx = -1;
  end_barcodes_.clear();
  end_barcode_names_.clear();

  while(getline(bcfile, line)){

    ++line_number;
    csv_line.clear();
    stringstream ss(line);
    int my_idx = 0;

    while (getline(ss, csvfield, ',')){
      csv_line.push_back(csvfield);

      // Read & process header line
      if (line_number == 1){
        if (csvfield == "id_str")
          name_idx = my_idx;
        else if (csvfield == "sequence")
          seq_idx = my_idx;
        else if (csvfield == "adapter")
          adapter_idx = my_idx;
        ++my_idx;
      }
    }

    if (line_number == 1){
      if (name_idx < 0 or seq_idx <0 or adapter_idx < 0)
        cerr << "ERROR: The end barcode csv file needs to have the columns id_str, sequence, adapter. " << endl;
      continue;
    }

    // Add barcode information
    end_barcodes_.push_back(EndBarcode());
    AddBarcode(end_barcodes_.back(), csv_line.at(seq_idx), csv_line.at(adapter_idx));
    end_barcodes_.back().barcode_name = csv_line.at(name_idx);
    end_barcode_names_.push_back(end_barcodes_.back().barcode_name);
  }
  return true;
};

// ----------------------------------------------------------------------------
// Classify end barcode and handles
// Reads that do not display the correct end barcode or don't have an adapter are filtered
// Reads where we don't find an adapter


void EndBarcodeClassifier::ClassifyAndTrimBarcode(int read_index, ProcessedRead &processed_read,
               const BasecallerRead& basecaller_read, const vector<int>& base_to_flow)
{
  if (processed_read.filter.is_filtered or processed_read.is_control_barcode)
    return;
  if (processed_read.read_group_index == nomatch_read_group_)
    return;

  // *** No end barcode to trim but we need to investigate handles & propagate a valid 3' trimming point

  bool have_adapter = processed_read.filter.n_bases_after_adapter_trim > 0 and
        processed_read.filter.n_bases_after_adapter_trim < processed_read.filter.n_bases;
  int best_handle, handle_start_flow, n_bases_handle = 0;

  if (not read_group_.at(processed_read.read_group_index).end_barcode and not demux_barcode_list_) {

    if (have_handles_){
      if (have_adapter and TrimHandle(
              base_to_flow.at(processed_read.filter.n_bases_filtered),
              bead_adapter_start_hp.at(processed_read.filter.adapter_type),
              processed_read.filter.n_bases_filtered, basecaller_read, base_to_flow,
              processed_read, n_bases_handle, handle_start_flow))
      {
        UpdateReadTrimming(processed_read, n_bases_handle,
                    handle_sequence_.at(processed_read.end_handle_index));
        return;
      }
      else {
        processed_read.end_handle_filtered = true;
        FilterRead(read_index, processed_read);
        return;
      }
    }
    else
    { // no handles - propagate trimming point
      processed_read.filter.n_bases_after_barcode_trim = processed_read.filter.n_bases_after_adapter_trim;
      return;
    }
  }

  // *** Barcode classification and trimming

  int best_barcode = -1;
  int n_bases_trimmed = 0, n_bases_barcode = 0, n_bases_adapter = 0;
  int best_bc_start_flow, best_bc_distance;
  int best_adapter_flow, best_adapter_distance;
  string my_YK_tag;

  if (have_adapter) {

    // -- End barcode classification

    switch (score_mode_){
      case 1: // In/Del Error tolerant flow alignment
        best_barcode = FlowAlignClassification(processed_read, n_bases_barcode, best_bc_start_flow,
                           bead_adapter_start_hp.at(processed_read.filter.adapter_type),
                           basecaller_read, base_to_flow);
        break;

      default: // Exact Base Space Match
        best_barcode = BaseSpaceClassification(processed_read, n_bases_barcode, best_bc_start_flow,
                                 basecaller_read, base_to_flow);
    }

    // -- Barcode adapter trimming

    int adapter_start_hp = read_group_.at(processed_read.read_group_index).adapter_start_hp;

    if (best_barcode >= 0){

      n_bases_trimmed = n_bases_barcode; // Log barcode trimming
      int start_hp = read_group_.at(processed_read.read_group_index).bc_start_hp;
      string trim_adapter(read_group_.at(processed_read.read_group_index).barcode_adapter);
      if (demux_barcode_list_) {
        trim_adapter = end_barcodes_[best_barcode].barcode_adapter;
        start_hp = end_barcodes_[best_barcode].bc_start_hp;
        adapter_start_hp = end_barcodes_[best_barcode].adapter_start_hp;
      }

      if (trim_adapter.length()>0){

        ReverseFlowAlignDistance(base_to_flow, trim_adapter,
               base_to_flow.at(processed_read.filter.n_bases_prefix),
               best_bc_start_flow, start_hp, best_adapter_flow,
               best_adapter_distance, n_bases_adapter);

        if (best_adapter_distance > adapter_cutoff_){ // Filter
          processed_read.end_adapter_filtered = true;
          best_barcode = -1;
        }
        else {
          n_bases_trimmed = n_bases_barcode + n_bases_adapter;
        }
      }
      else{ // No adapter to trim - reset to reflect barcode end
        best_adapter_flow = best_bc_start_flow;
        adapter_start_hp = start_hp;
      }

    }

    // -- Barcode handle trimming

    if (have_handles_ and best_barcode >= 0){

      if (TrimHandle(best_adapter_flow, adapter_start_hp, processed_read.filter.n_bases_filtered-n_bases_trimmed,
              basecaller_read, base_to_flow, processed_read, n_bases_handle, handle_start_flow))
      {
        // Filter reads where the same handle was found on the front and back end
        if (handle_filter_ and processed_read.end_handle_index==processed_read.handle_index){
          processed_read.end_handle_filtered = true;
          best_barcode = -1;
        }
        else if (trim_barcodes_){
          my_YK_tag = handle_sequence_.at(processed_read.end_handle_index);
          n_bases_trimmed += n_bases_handle;
        }
      }
      else {
        processed_read.end_handle_filtered = true;
        best_barcode = -1;
      }
    }
  }

  // Push to nomatch if no barcode assignment occurred
  if (best_barcode < 0) {
    FilterRead(read_index, processed_read);
    return;
  }
  processed_read.end_barcode_index = best_barcode;

  if (not trim_barcodes_) {
    processed_read.filter.n_bases_after_barcode_trim = -1;
    return;
  }

  // If we demultiplex, add the end barcode sequence to YK field.
  if (demux_barcode_list_){
    my_YK_tag += end_barcodes_.at(best_barcode).barcode_adapter;
    my_YK_tag += end_barcodes_.at(best_barcode).barcode_sequence;
  }
  UpdateReadTrimming(processed_read, n_bases_trimmed, my_YK_tag);
};

// ----------------------------------------------------------------------------

void EndBarcodeClassifier::UpdateReadTrimming(ProcessedRead &processed_read,
         int trim_n_bases, const string& YK_tag)
{
  // Trim bases from read
  processed_read.filter.n_bases_after_barcode_trim = max(0, processed_read.filter.n_bases_filtered-trim_n_bases);
  processed_read.filter.n_bases_filtered = processed_read.filter.n_bases_after_barcode_trim;

  // Edit ZA tag
  processed_read.bam.EditTag("ZA", "i", max(processed_read.filter.n_bases_filtered - processed_read.filter.n_bases_prefix, 0));

  if (not YK_tag.empty()){
    processed_read.bam.AddTag("YK", "Z", YK_tag);
  }
}

// ----------------------------------------------------------------------------

void EndBarcodeClassifier::FilterRead(int read_index, ProcessedRead &processed_read)
{
  int x, y;
  barcode_mask_pointer_->IndexToRowCol (read_index, y, x);
  if (nomatch_read_group_ >= 0){
    processed_read.end_barcode_filtered = processed_read.read_group_index;
    processed_read.read_group_index = nomatch_read_group_;
    barcode_mask_pointer_->SetBarcodeId(x, y, 0);
  }
}

// ----------------------------------------------------------------------------

int EndBarcodeClassifier::BaseSpaceClassification(ProcessedRead &processed_read,
           int& n_bases_barcode, int& bc_start_flow,
           const BasecallerRead& basecaller_read,
           const vector<int>& base_to_flow)
{
  int best_barcode = -1;

  // Matching the single end barcode specified in the read group
  if (end_barcodes_.size()==0){

    if (BaseSpaceMatch(processed_read.filter.n_bases_prefix, processed_read.filter.n_bases_filtered,
            basecaller_read, read_group_.at(processed_read.read_group_index).barcode_sequence)){
      best_barcode = processed_read.read_group_index;
      processed_read.end_bc_n_errors = 0;
      n_bases_barcode    = read_group_.at(processed_read.read_group_index).barcode_sequence.length();
      bc_start_flow = base_to_flow.at(processed_read.filter.n_bases_filtered-n_bases_barcode);
    }

  }
  else{
    // Matching the sequences in the end barcode vector
    unsigned int bci = 0;
    while (bci < end_barcodes_.size() and not BaseSpaceMatch(
        processed_read.filter.n_bases_prefix, processed_read.filter.n_bases_filtered,
           basecaller_read, end_barcodes_[bci].barcode_sequence))
      ++bci;
    if (bci < end_barcodes_.size()){
      best_barcode = bci;
      processed_read.end_bc_n_errors = 0;
      n_bases_barcode    = end_barcodes_[bci].barcode_sequence.length();
      bc_start_flow = base_to_flow.at(processed_read.filter.n_bases_filtered-n_bases_barcode);
    }
  }
  return best_barcode;
}

// ----------------------------------------------------------------------------

int EndBarcodeClassifier::FlowAlignClassification(ProcessedRead &processed_read,
           int& n_bases_barcode, int& best_bc_start_flow, int adapter_hp,
           const BasecallerRead& basecaller_read,
           const vector<int>& base_to_flow)
{
  int best_barcode = -1;
  int my_distance, my_start_flow, my_bases;
  int best_bc_distance  = score_cutoff_ + 1;
  int second_best_distance = 1000;

  // Matching the sequences in the end barcode vector
  if (demux_barcode_list_){

    for (unsigned int bci=0; bci < end_barcodes_.size(); ++bci){
      ReverseFlowAlignDistance(base_to_flow,
             end_barcodes_[bci].barcode_sequence,
             base_to_flow.at(processed_read.filter.n_bases_prefix),
             base_to_flow.at(processed_read.filter.n_bases_filtered),
             adapter_hp,
             my_start_flow, my_distance, my_bases);
      if (my_distance < best_bc_distance ){
        second_best_distance = best_bc_distance;
        best_bc_distance = my_distance;
        best_barcode = bci;
        n_bases_barcode = my_bases;
        best_bc_start_flow = my_start_flow;
      }
      else if (my_distance < second_best_distance){
        second_best_distance = my_distance;
      }
    }

    if (second_best_distance - best_bc_distance < score_separation_){
      best_barcode = -1;
    }
    else if (best_barcode >= 0){
      processed_read.end_bc_n_errors = best_bc_distance;
    }

  }
  // Matching the single end barcode specified in the read group
  else{

    ReverseFlowAlignDistance(base_to_flow,
           read_group_[processed_read.read_group_index].barcode_sequence,
           base_to_flow.at(processed_read.filter.n_bases_prefix),
           base_to_flow.at(processed_read.filter.n_bases_filtered),
           1,
           best_bc_start_flow, best_bc_distance, n_bases_barcode);
    if (best_bc_distance <= score_cutoff_){
      processed_read.end_bc_n_errors = best_bc_distance;
      best_barcode = processed_read.read_group_index;
    }
  }
  return best_barcode;
}

// ----------------------------------------------------------------------------
// Trim handle sequence

bool EndBarcodeClassifier::TrimHandle(int adapter_flow, int adapter_hp, int temp_end_base,
         const BasecallerRead& basecaller_read, const vector<int>& base_to_flow,
         ProcessedRead &processed_read, int& n_bases_handle, int& handle_start_flow)
{
  n_bases_handle = 0;
  handle_start_flow = -1;


  int best_handle  = -1;
  int best_distance  = handle_cutoff_+1;
  int my_distance, my_start_flow, my_bases;
  unsigned int bci = 0;

  switch (handle_mode_){

    case 1: // looks at flow-space absolute error counts

      for (; bci < handle_sequence_.size(); ++bci){
        ReverseFlowAlignDistance(base_to_flow, handle_sequence_[bci],
               base_to_flow.at(processed_read.filter.n_bases_prefix),
               adapter_flow, adapter_hp,
               my_start_flow, my_distance, my_bases);
        if (my_distance < best_distance){
          best_handle = bci;
          best_distance = my_distance;
          n_bases_handle = my_bases;
          handle_start_flow = my_start_flow;
        }
      }
      break;

    default: // base space comparison

      while (bci < handle_sequence_.size() and not BaseSpaceMatch(
                 processed_read.filter.n_bases_prefix, temp_end_base,
                 basecaller_read, handle_sequence_[bci])) {
        ++bci;
      }
      if (bci < handle_sequence_.size()){
        best_handle = bci;
        //processed_read.end_handle_n_errors = 0;
        best_distance = 0;
        n_bases_handle = handle_sequence_[bci].length();
        handle_start_flow = base_to_flow.at(temp_end_base-n_bases_handle);
      }
  }

  if (best_handle >= 0){
    processed_read.end_handle_n_errors = best_distance;
    processed_read.end_handle_index = best_handle;
    return true;
  }
  else
    return false;
}

// ----------------------------------------------------------------------------
// Matches a sequence of bases at the read end

bool  EndBarcodeClassifier::BaseSpaceMatch(int n_bases_prefix, int n_bases_filtered,
        const BasecallerRead& basecaller_read, const string & query)
{

  int bc_len = (int)query.length();
  int bc_base = n_bases_filtered - bc_len;

  if (bc_base < n_bases_prefix)
    return -1;

  int i=0;
  while (i<bc_len and isBaseMatch(basecaller_read.sequence.at(bc_base+i),query[i]))
    ++i;
  if (i==bc_len)
    return true; // perfect match found
  else
    return false;
}

// ----------------------------------------------------------------------------

vector<int> EndBarcodeClassifier::BaseToFlow(const string &query, int flow)
{
  vector<int> base_to_flow(query.length());
  for (unsigned int base=0; base<query.length(); ++base){
    while(flow < flow_order_p_->num_flows() and query[base] != flow_order_p_->nuc_at(flow))
      ++flow;
    base_to_flow.at(base) = flow;
  }

  return base_to_flow;
}

// ----------------------------------------------------------------------------
// This function currently assumes that there is no


void EndBarcodeClassifier::ReverseFlowAlignDistance(
         const vector<int>& base_to_flow,
         const string& my_query,
         int   prefix_flow,
         int   adapter_flow,
         int   adapter_hp,
         int&  best_start_flow,
         int&  best_distance,
         int&  best_nBases)
{
  best_start_flow = -1;
  best_distance   = 1000;
  best_nBases     = 0;

  // To account for possible flow overlap of sequences
  int start_flow = adapter_flow;
  int end_flow   = adapter_flow;
  bool score_end = false;
  if (adapter_hp>0){
    ++end_flow;
    score_end = true;
  }

  char abase = flow_order_p_->nuc_at(start_flow);
  string query(my_query);
  query += abase; //We need at least one adapter at the end.
  for (int ihp=1; ihp<adapter_hp; ++ihp)
    query += abase;

  // Step 1 Get latest possible start flow for flow alignment

  for (int base=query.length()-1; base>=0; --base){
    while (start_flow >= prefix_flow and query.at(base) != flow_order_p_->nuc_at(start_flow))
      --start_flow;
  }
  if (start_flow < prefix_flow)
    return;
  start_flow++;

  // Step 2 calculate distance by iterating over possible flow alignments

  vector<int> q_to_flow;
  int local_distance, qHP, rHP;
  unsigned int rbase, qbase, n_bases;

  while(true){
    --start_flow;
    while (start_flow >= prefix_flow and query[0] != flow_order_p_->nuc_at(start_flow))
      --start_flow;
    // Protection against ending at the last base of multi-taps
    while (start_flow >0 and flow_order_p_->nuc_at(start_flow) == flow_order_p_->nuc_at(start_flow-1))
      --start_flow;

    if (start_flow < prefix_flow)
      return;

    q_to_flow = BaseToFlow(query, start_flow);

    if (q_to_flow.at(query.length()-1)!=adapter_flow){
      return;
    }

    // Now calculate distance
    rbase = 0, qbase=0; local_distance = 0;
    qHP=0; rHP=0, n_bases=0;

    while (rbase < base_to_flow.size() and base_to_flow.at(rbase) < q_to_flow[0])
      ++rbase;

    for (int flow=q_to_flow[0]; flow < end_flow; ++flow){
      qHP=0; rHP=0;
      while (rbase < base_to_flow.size() and base_to_flow.at(rbase) == flow){
        ++rHP;
        ++rbase;
        if (score_end and flow < end_flow-1)
          ++n_bases;
      }
      while (qbase < q_to_flow.size() and q_to_flow[qbase] == flow){
        ++qHP;
        ++qbase;
      }
      if (flow==q_to_flow[0]){
        local_distance += max(0, qHP-rHP);
        n_bases = min(rHP, qHP);
      }
      else if (score_end and flow == end_flow-1){
        // Counting errors associated with adapter
        local_distance += abs(qHP-rHP)- max(0, adapter_hp-rHP);;
        n_bases += max(0, rHP-adapter_hp);
      }
      else
        local_distance += abs(qHP-rHP);
    }

    if (local_distance < best_distance){
      best_distance   = local_distance;
      best_start_flow = start_flow;
      best_nBases     = n_bases;
    }
  }
}

// ----------------------------------------------------------------------------

void EndBarcodeClassifier::LoadHandlesFromArgs(OptArgs& opts, const Json::Value& structure)
{
  handle_sequence_  = opts.GetFirstStringVector ('-', "barcode-handles", "");
  if (handle_sequence_.empty()) {
    if (structure.isMember("handle")){
      vector<string> keys = structure["handle"].getMemberNames();
      for (unsigned int k=0; k<keys.size(); ++k)
        handle_sequence_.push_back(structure["handle"][keys[k]].asString());
    }
    else{
      have_handles_ = false;
      return;
    }
  }

  have_handles_ = true;
  for (unsigned int iH=0; iH<handle_sequence_.size(); ++iH){
    RevComplementInPlace(handle_sequence_[iH]);
  }

  cout << "  Have " << handle_sequence_.size() << " reverse handles. ";
  //for (unsigned int iHandle=0; iHandle<handle_sequence_.size(); ++iHandle){
  //  cout << handle_sequence_[iHandle] << ',';
  //}
  cout << endl;

  handle_mode_   =   opts.GetFirstInt    ('-', "handle-mode", DEF_HANDLE_MODE);
  handle_cutoff_ =   opts.GetFirstInt    ('-', "handle-cutoff", DEF_HANDLE_CUTOFF);
  handle_filter_ =   opts.GetFirstBoolean('-', "handle-filter", false);

  // Check for non-ACGT characters
  if (handle_mode_ != 0){
    for (unsigned int iH=0; iH<handle_sequence_.size(); ++iH){
      if (std::string::npos != handle_sequence_.at(iH).find_first_not_of("ACGT")){
        cout << "   WARNING: Non-ACGT handle characters detected. Setting handle-mode=0." << endl;
        handle_mode_ = 0;
        break;
      }
    }
  }

  cout << "  handle-mode              : " << handle_mode_ << endl;
  cout << "  handle-cutoff            : " << handle_cutoff_ << endl;
}

// ----------------------------------------------------------------------------



