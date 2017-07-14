/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BarcodeClassifier.cpp
//! @ingroup  BaseCaller
//! @brief    BarcodeClassifier. Barcode detection and trimming for BaseCaller

#include "BarcodeClassifier.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>

#include "Utils.h"

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
  printf ("     --barcode-compute-dmin        BOOL       If true, computes minimum Hamming distance of barcode set [false]\n");
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
                const vector<KeySequence>& keys, const string& output_directory, int chip_size_x, int chip_size_y)
  : barcode_mask_(chip_size_x, chip_size_y)
{
  flow_order_        = flow_order;
  num_barcodes_      = 0;
  barcode_max_flows_ = 0;
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
  barcode_filter_minreads_        = opts.GetFirstInt    ('-', "barcode-filter-minreads", 20);
  barcode_error_filter_           = opts.GetFirstDouble ('-', "barcode-error-filter", 0.0);
  barcode_filter_postpone_        = opts.GetFirstInt    ('-', "barcode-filter-postpone", 0);
  barcode_filter_named_           = opts.GetFirstBoolean('-', "barcode-filter-named", false);
  check_limits_                   = opts.GetFirstBoolean('-', "barcode-check-limits", true);
  score_auto_config_              = opts.GetFirstBoolean('-', "barcode-auto-config", false);
  bool compute_dmin               = opts.GetFirstBoolean('-', "barcode-compute-dmin", false);
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

  LoadBarcodesFromDataset(datasets, keys);

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

void BarcodeClassifier::LoadBarcodesFromDataset(BarcodeDatasets& datasets, const vector<KeySequence>& keys)
{
  string barcode_id  = datasets.barcode_config().get("barcode_id","noID").asString();
  barcode_.reserve(datasets.num_read_groups());

  // --- Loop loading individual barcode informations
  for (int rg_idx = 0; rg_idx < datasets.num_read_groups(); ++rg_idx) {
    Json::Value& read_group = datasets.read_group(rg_idx);

    // Group for non-barcoded reads
    if (!read_group.isMember("barcode_sequence")) {
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

    // Check for non-ACGT characters
    if (std::string::npos != barcode_.back().full_barcode.find_first_not_of("ACGTN")){
      have_ambiguity_codes_ = true;
      continue;
    }

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
  } // --- End Loop loading individual barcode informations

  // Only print classification settings if we're using the classifier.
  if (dataset_in_use_ and no_barcode_read_group_ < 0)
    cout << "   INFO: Dataset " << barcode_id << " does not have a non-barcoded read group." << endl;

  if (not have_ambiguity_codes_) {
    // And loop through barcodes again to determine maximum amount of flows after start flow has been determined
    int barcode_flows, barcode_min_flows = -1;
    for (unsigned int bc=0; bc<barcode_.size(); bc++) {
      barcode_flows = barcode_.at(bc).adapter_start_flow - barcode_min_start_flow_;
      barcode_max_flows_ = max( barcode_max_flows_, barcode_flows);
      if (barcode_min_flows < 0 or barcode_flows < barcode_min_flows)
        barcode_min_flows = barcode_flows;
    }
    if (dataset_in_use_ and barcode_min_flows >= 0 and barcode_min_flows != barcode_max_flows_)
      cout << "   WARNING: Barcode set is not flow space synchronized. Barcodes range from "
           << barcode_min_flows << " to " << barcode_max_flows_ << " flows." << endl;
  }
  /*cout << "  Barcode dataset info: min_start_flow=" << barcode_min_start_flow_ << endl <<
          "                     barcode_max_flows=" << barcode_max_flows_ << endl <<
          "                             num_flows=" << barcode_.back().num_flows << endl;
   */

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
  cout << "   Computed minimum Hamming distance of barcode set as d_min = " << hamming_dmin_ << endl;

};


// --------------------------------------------------------------------------

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
	best_errors = 1 + (int)score_cutoff_; // allows match with this many errors minimum when in bc_score_mode 1

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

}

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
  datasets.barcode_filters()["filter_frequency"] = barcode_filter_;
  datasets.barcode_filters()["filter_minreads"]  = (Json::Int64)barcode_filter_minreads_;
  datasets.barcode_filters()["filter_errors_hist"] = barcode_error_filter_;
  datasets.barcode_filters()["filter_postpone"]   = (Json::Int64)barcode_filter_postpone_;
  // not used by python filters but useful information if we are looking at the json file:
  datasets.barcode_filters()["filter_weight"]     = barcode_filter_weight_;
  datasets.barcode_filters()["classifier_mode"]   = (Json::Int64)score_mode_;
  datasets.barcode_filters()["classifier_cutoff"] = score_cutoff_;
  datasets.barcode_filters()["classifier_separation"] = score_separation_;
  if (hamming_dmin_ >=0)
    datasets.barcode_filters()["hamming_distance"] = hamming_dmin_;

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
      bool i_am_filtered = false;
      bool filter_this_bc = barcode_filter_named_ or (*rg)["sample"].asString() == "none";

      // Initial filtering based on number of reads in read group
      if (filter_this_bc) {
        i_am_filtered = (read_count > read_threshold) ? false : true;
      }

      // Further filtering based on average number of errors
      if((barcode_error_filter_ > 0) and (*rg).isMember("barcode_errors_hist")
         and filter_this_bc and (not i_am_filtered) and (not barcode_filter_postpone_)) {
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

