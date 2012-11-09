/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BarcodeClassifier.cpp
//! @ingroup  BaseCaller
//! @brief    BarcodeClassifier. Barcode detection and trimming for BaseCaller

#include "BarcodeClassifier.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "Utils.h"

#define BASES(val) ((int)((val)/100.0+0.5))

void ValidateAndCanonicalizePath(string &path);   // Borrowed from BaseCaller.cpp


void BarcodeClassifier::PrintHelp()
{
  printf ("Barcode classification options:\n");
  printf ("  -b,--barcodes              FILE/off   detect barcodes listed in provided file [off]\n");
//  printf ("     --barcode-directory     DIRECTORY  output directory for barcode-specific sff files [output-dir/bc_files]\n");
//  printf ("     --score-mode            STRING     Set the score mode and threshold in XvY format [0v0.9]\n");
//  printf ("     --barcode-mask          on/off     write barcode ids to output-dir/barcodeMask.bin [on]\n" );
  printf ("     --barcode-filter        FLOAT      barcode freq. threshold, if >0 writes output-dir/barcodeFilter.txt [0.0 = off]\n" );
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
  // Deprecated option. Barcode mask generated always!
  opts.GetFirstBoolean('-', "barcode-mask", true);
  barcode_filter_                 = opts.GetFirstDouble ('-', "barcode-filter", 0.0);

  if (barcode_filter_ > 0.0)
    barcode_filter_filename_ = output_directory+"/barcodeFilter.txt";


  //
  // Step 1. First phase of initialization: parse barcode list file
  //

  string file_id  = datasets.barcode_config().get("barcode_id","").asString();
  score_mode_     = datasets.barcode_config().get("score_mode",1).asInt();
  score_cutoff_   = datasets.barcode_config().get("score_cutoff",2.0).asDouble();

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

    string full_barcode = keys[0].bases();
    int key_length = full_barcode.length();

    full_barcode += read_group["barcode_sequence"].asString();
    int key_barcode_length = full_barcode.length();

    full_barcode += read_group.get("barcode_adapter","").asString();
    int key_barcode_adapter_length = full_barcode.length();


    int flow = 0;
    int curBase = 0;

    while (curBase < key_barcode_adapter_length and flow < flow_order_.num_flows()) {

      while (full_barcode[curBase] == flow_order_[flow] and curBase < key_barcode_adapter_length) {
        barcode_.back().flow_seq[flow]++;
        curBase++;
      }
      // grab the next flow after we sequence through the key, this will be the first flow we will want to count towards barcode matching/scoring, even if its a 0-mer flow
      if (curBase >= key_length and barcode_.back().start_flow == -1)
        barcode_.back().start_flow = flow+1;
      // grab the last positive incorporating flow for the barcode, any 0-mer flows after this and before the insert or adapter would not be counted in the barcode matching/scoring
      if (curBase >= key_barcode_length and barcode_.back().end_flow == -1)
        barcode_.back().end_flow = flow;

      flow++;
    }
    if (barcode_.back().end_flow == -1)
      barcode_.back().end_flow = flow - 1;
    barcode_.back().num_flows = flow;
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
  printf("   Barcode filter filename  : %s\n", barcode_filter_filename_.c_str());
  printf("\n");

}


BarcodeClassifier::~BarcodeClassifier()
{
}



/*
 * flowSpaceTrim - finds the closest barcode in flowspace to the sequence passed in,
 * and then trims to exactly the expected flows so it can be error tolerant in base space
 */
void BarcodeClassifier::ClassifyAndTrimBarcode(int read_index, SFFEntry &sffentry)
{

  int minErrors = (int)score_cutoff_; // allows match with this many errors minimum when in bc_score_mode 1

  double bestScore = score_cutoff_;

  int totalFlows;

  // find best barcode match
  int bestBarcodeIndex = -1;

  std::string scoreHistLineBest;

  double old_weightedscore=0.0;
  double old_weightedErr = 0;

  // barcode loop
  for (int i = 0; i < num_barcodes_; ++i) {

    const Barcode& bce = barcode_[i];

    // calculate a score for the barcode comparison
    double totalErrors = 0;
    double weightedErr = 0;
    int flow;

    // lets not try and look at more flows than we were provided!
    int endFlow = bce.end_flow;
    if (endFlow >= flow_order_.num_flows())
      endFlow = flow_order_.num_flows()-1;

    totalFlows = endFlow - bce.start_flow + 1;

    for (flow = bce.start_flow; flow <= endFlow; ++flow) {

      double delta = bce.flow_seq[flow] - BASES(sffentry.flowgram[flow]);
      // AS: for weighted scoring may want to give more weight to accurate 1-mers
      double delta_res = 2*fabs(sffentry.flowgram[flow]/100.0 - BASES(sffentry.flowgram[flow])); //used for weighted scoring

      if (delta < 0)
        delta = -delta;
      totalErrors += delta;
      weightedErr += delta * (1-delta_res);
    }

    double score = 0.0;
    double weightedscore = 0.0;
    if (totalFlows > 0) {
      score = 1.0 - ( double ) totalErrors/ ( double ) totalFlows;
      weightedscore = 1.0 - ( double ) weightedErr/ ( double ) totalFlows;
    }


    // see if this score is the best (best starts at minimum so might never hit)
    if (score_mode_ == 1 or score_mode_ == 2) // looks at flow-space absolute error counts, not ratios
    {
      if (totalErrors <= minErrors)
      {
        minErrors = totalErrors;
        bestBarcodeIndex = i;
        old_weightedErr = weightedErr;
      }
      // use weighted error to resolve conflicts
      else if ( ( score_mode_ == 2 ) && ( totalErrors == minErrors ) && ( bestBarcodeIndex > -1 ) && ( weightedErr < old_weightedErr ) )
      {
        bestBarcodeIndex = i;
        old_weightedErr = weightedErr;
      }
    }
    else   //default score mode
    {
      if ( score > bestScore )
      {
        bestScore = score;
        bestBarcodeIndex = i;
        old_weightedscore = weightedscore;
      }
      // use weighted score to resolve conflicts
      else if ( ( fabs ( bestScore - score ) < 0.000001 ) && ( bestBarcodeIndex > -1 ) && ( weightedscore < old_weightedscore ) )
        bestBarcodeIndex = i;
    }

  }
  // end barcode loop

  // generate the barcode match struct and return to user
  // MGD note - we might just want to have the user pass in a pre-allocated struct or something so we don't thrash mem so bad
  if (bestBarcodeIndex == -1) {
    int x, y;
    barcode_mask_.IndexToRowCol (read_index, y, x);
    barcode_mask_.SetBarcodeId(x, y, 0);
    sffentry.barcode_id = no_barcode_read_group_;
    return;
  }

  const Barcode& bce = barcode_[bestBarcodeIndex];

  // since the match done in flowspace allows for errors, we need to see where in the input read we really want to clip in terms of bases
  // count the number of bases called based on the input flowVals but using the barcode's expected number of flows
  int bases = 0;
  for (int flow = 0; flow < bce.num_flows-1; ++flow)
    bases += BASES ( sffentry.flowgram[flow] );

  // special-case for the last flow since bases may match library insert as well
  // additional is how many we found minus how many we expected for the barcode end (or 5'-adpater end)
  if ( BASES ( sffentry.flowgram[bce.num_flows-1] ) > 0 )
  { // if we called at least one base on the last flow, need to make sure we clip correctly
    int additional = BASES ( sffentry.flowgram[bce.num_flows-1] ) - bce.flow_seq[bce.num_flows-1];
    bases += bce.flow_seq[bce.num_flows-1];
    if ( additional < 0 )
      bases += additional; // remove bases that spilled over into real read
  }

  // keep track of the average residual error in the fit for each barcode type

  sffentry.clip_adapter_left = bases+1; // Trim this many bases

  int x, y;
  barcode_mask_.IndexToRowCol (read_index, y, x);
  barcode_mask_.SetBarcodeId(x, y, (uint16_t)bce.mask_index);
  sffentry.barcode_id = bce.read_group_index;
  sffentry.barcode_n_errors = minErrors;
}



void BarcodeClassifier::Close(BarcodeDatasets& datasets)
{
  if (0 != barcode_mask_.WriteRaw (barcode_mask_filename_.c_str()))
    fprintf (stderr, "BarcodeClassifier: Cannot write mask file file: %s\n", barcode_mask_filename_.c_str());

  // Generate barcodeFilter.txt file

  if (barcode_filter_ > 0.0) {

    vector<int> read_counts;
    for (Json::Value::iterator rg = datasets.read_groups().begin(); rg != datasets.read_groups().end(); ++rg)
      if ((*rg).isMember("read_count") and (*rg).isMember("barcode_sequence"))
        read_counts.push_back((*rg)["read_count"].asInt());
    sort (read_counts.begin(), read_counts.end(), std::greater<int>());

    int read_threshold = 20;
    for (int i = 1; i < (int)read_counts.size(); ++i) {
      if (read_counts[i] / (read_counts[i-1] + 0.001) < barcode_filter_) {
        read_threshold = max(read_threshold, read_counts[i-1]);
        break;
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

