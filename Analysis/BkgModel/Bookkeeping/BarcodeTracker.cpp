/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BarcodeTracker.h"


BarcodeTracker::BarcodeTracker(){
  defer_processing=false;
}

void BarcodeTracker::SetupEightKeyNoT(const char *letter_flowOrder){

  // should be read from file, ignore for now
  my_codes.push_back("TCAGTCCAAAAGGGGGG");
  my_codes.push_back("TCAGTTCCCCAAAAAAG");
  my_codes.push_back("TCAGTTTTCCCCCCAGG");
  my_codes.push_back("TCAGTTTTTTCAAGGGG");
  my_codes.push_back("TCAGAGGTTTTCCCCCC");
  my_codes.push_back("TCAGAAGGGGTTTTTTC");
 my_codes.push_back("TCAGAAAAGGGGGGTCC");
  my_codes.push_back("TCAGAAAAAAGTTCCCC");
  // eight keys set up, now turn into seqlist items
  SetupBarcodes(my_codes, letter_flowOrder);
}

void BarcodeTracker::SetupLoadedBarcodes(const char *letter_flowOrder){
  SetupBarcodes(my_codes,letter_flowOrder);
}

// JSON file containing
// extendedkey+barcode
// to be used for training regional parameters
void BarcodeTracker::ReadFromFile(string &barcode_file){
  barcode_file_name = barcode_file;
  cout << "READING BARCODES:\t" <<barcode_file_name << endl;
  //check file format based on suffix
  if (boost::algorithm::ends_with(barcode_file_name, ".json"))
  {

  Json::Value all_params;
   std::ifstream in(barcode_file_name, std::ios::in);

   if (!in.good()) {
     printf("Opening barcodekey file %s unsuccessful. Aborting\n", barcode_file.c_str());
     exit(1);
   }
   in >> all_params;

   if (all_params.isMember("barcodekey")){
     // strip down to the correct subset of the file

     const Json::Value barcode_key = all_params["barcodekey"];
     for ( int index = 0; index < (int) barcode_key.size(); ++index )
       my_codes.push_back(barcode_key[index].asCString());
     for (unsigned int xi=0; xi<my_codes.size(); xi++)
       cout << "BARCODES:\t" << my_codes[xi] << endl;

   }else{
     std::cout << "ABORT: barcodekey file contains no barcodekey " << barcode_file.c_str() << "\n";
     exit(1);
   }
   // echo as test
    //std::cout << gopt_params.toStyledString();
   in.close();

  } else{

    printf("Abort: %s not a json file",barcode_file.c_str());
    exit(1);

  }

// wait to process until we know the flow order and can upgrade
}

void BarcodeTracker::SetupBarcodes(vector<string> &barcode_plus_key, const char *letter_flowOrder){
  int flowOrderLength = strlen(letter_flowOrder);
  char * dummy_flowOrder = strdup(letter_flowOrder);

  barcode_data.resize(barcode_plus_key.size()); // how many we need
  consistentFlows = 64;
  maxFlows = 0;
  for (unsigned int ib=0; ib<barcode_plus_key.size(); ib++){
    // do one barcode here
    NullSeqItem(barcode_data[ib]); // null ionogram, etc
    barcode_data[ib].type = MaskLib;
    barcode_data[ib].seq.assign(barcode_plus_key[ib]);
    barcode_data[ib].numKeyFlows = seqToFlow(barcode_data[ib].seq.c_str(), barcode_data[ib].seq.length(),
                                       barcode_data[ib].Ionogram, 64, dummy_flowOrder, flowOrderLength);
    // guarantee these flows have the value provided
    barcode_data[ib].usableKeyFlows = barcode_data[ib].numKeyFlows - 1;
    // don't care about 0, 1-mer counts, so not filling them in
    if (barcode_data[ib].usableKeyFlows<consistentFlows){
      consistentFlows = barcode_data[ib].usableKeyFlows; // omit last flow for classification
    }
    if (barcode_data[ib].numKeyFlows>maxFlows)
      maxFlows = barcode_data[ib].numKeyFlows; // individual barcodes out to here
  }
}

void BarcodeTracker::ReportClassificationTable(int dummy_tag){
  // data dump
  vector<int> histogram;
  histogram.resize(barcode_data.size()+1,0);
  for (unsigned int ibd=0; ibd<barcode_id.size(); ibd++){
    histogram[barcode_id[ibd]+1]+=1;
  }
  // dump to standard out for logging
  cout << "TAG:\t" << dummy_tag << "\t";
  for (unsigned int ib=0; ib<histogram.size(); ib++)
    cout << histogram[ib] << "\t";
  cout << "\n";
  for (unsigned int ib=0; ib<barcode_data.size(); ib++){
    cout << "BAR:\t" << dummy_tag << "\t" << barcode_data[ib].seq << "\t";
    for (int iflow=0; iflow<maxFlows; iflow++){
      cout << (barcode_bias[ib][iflow])/(barcode_count[ib+1]+0.01) << "\t";
    }
    cout << "\n";
    // expected value as well
    cout << "BRF:\t" << dummy_tag << "\t" << barcode_data[ib].seq << "\t";
    for (int iflow=0; iflow<maxFlows; iflow++){
      cout << barcode_data[ib].Ionogram[iflow] << "\t";
    }
    cout << "\n";
  }
}

int BarcodeTracker::ClassifyOneBead(float *Ampl, double basic_threshold, double second_threshold, int flow_block_size, int flow_block_start){
  // return an ID of a bead
  double best_fit = 10000;
  double second_fit = best_fit;
  int bar_state = -1; // no class
  // note that last flow can be used, but needs to be thresholded to allow for additional bases
  for (unsigned int ib=0; ib<barcode_data.size(); ib++){
     // proportional signal
    double proportional_fit = 0.0;
    for (int iflow=0; (iflow<consistentFlows)& (iflow<flow_block_size); iflow++){
      double ratio_signal = 1.0-(Ampl[iflow]+0.5)/(barcode_data[ib].Ionogram[iflow]+0.5); // 0-centered, fit proportional to signal, "cheap log"
      proportional_fit += ratio_signal*ratio_signal;
    }
    if (proportional_fit<best_fit){
       bar_state = ib;
       if (best_fit<second_fit)
         second_fit = best_fit;
       best_fit = proportional_fit;
    } else {
      if (proportional_fit<second_fit){
         second_fit = proportional_fit;
      }
    }
  }
  if (best_fit>basic_threshold){
    bar_state = -1; // no match because outside of radius
  }
  if (second_fit-best_fit<second_threshold){
    bar_state = -1; // no match because too ambiguous
  }
  // send consistent data downstream
  barcode_count[bar_state+1]+=1;
  if (bar_state>-1){
    for (int iflow=0; (iflow<maxFlows) & (iflow<flow_block_size); iflow++){
      barcode_bias[bar_state][iflow] += Ampl[iflow];
     }
  }

  return(bar_state);
}

void BarcodeTracker::ResetCounts(){
  barcode_count.resize(barcode_data.size()+1);
  barcode_count.assign(barcode_data.size()+1,0);
  barcode_bias.resize(barcode_data.size());
  for (unsigned int ib=0; ib<barcode_data.size(); ib++){
    barcode_bias[ib].assign(maxFlows, 0);
  }
}

float BarcodeTracker::ComputeNormalizerBarcode (const float *observed, int ib, int flow_block_size, int flow_block_start)
{
  float wsum = 0.0001f;
  float wdenom = 0.0001f; // avoid dividing by zero in the bad case
  if (ib>-1){
    for (int i=0; (i<barcode_data[ib].usableKeyFlows) & (i<flow_block_size); i++)
  {
    int kval = barcode_data[ib].Ionogram[i];
    float weight = 1.0/(kval+0.5);  // relative weight decreases by increased HP due to expected variance
    wsum += weight*observed[i]*kval;  // works for keyval = 0
    wdenom += weight*kval*kval;  // works for keyval >1
  }
  }
  return (wsum/wdenom);
}

int BarcodeTracker::SetBarcodeFlows(float *observed, int ib){
  int ref_span = 0;
  if (ib>-1){
    // these flows are guaranteed correct
    for (int i=0; i<barcode_data[ib].usableKeyFlows; i++)
    {
      observed[i] = barcode_data[ib].Ionogram[i];
    }
    // possibly use last flow as a minimum?
    ref_span = barcode_data[ib].usableKeyFlows;
  }
  return(ref_span);
}
