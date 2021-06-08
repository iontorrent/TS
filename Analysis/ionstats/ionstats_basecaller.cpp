/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ionstats.h"

#include <string>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"

#include "OptArgs.h"
#include "Utils.h"
#include "IonVersion.h"


using namespace std;
using namespace BamTools;



void IonstatsBasecallerHelp()
{
  cerr << endl;
  cerr << "ionstats " << IonVersion::GetVersion() << "-" << IonVersion::GetRelease()
       << " (" <<IonVersion::GetGitHash() << ") - Generate performance metrics and statistics for Ion sequences."
       << endl << endl;
  cerr << "Usage:   ionstats basecaller [options]"
       << endl << endl;
  cerr << "General options:" << endl;
  cerr << "  -i,--input                 FILE                  input BAM (unmapped or mapped) [required option]" << endl;
  cerr << "  -o,--output                FILE                  output json file [ionstats_basecaller.json]" << endl;
  cerr << "  -h,--histogram-length      INT                   read length histogram cutoff [400]" << endl;
  cerr << "  -k,--key                   STRING                seq key - used for calculating system_snr [" << DEFAULT_SEQ_KEY << "]" << endl;
  cerr << "  -b,--barcodes              STRING,STRING,...     select only for this set of barcodes [" << DEFAULT_BARCODE << "]" << endl;
  cerr << "  -d,--output-barcodes       STRING,STRING,...     output barcoded json files [" << DEFAULT_BARCODE_OUTPUT<< "]"
       << endl << endl;
}



int IonstatsBasecaller(OptArgs &opts)
{
  string input_bam_filename   = opts.GetFirstString ('i', "input", "");
  string output_json_filename = opts.GetFirstString ('o', "output","ionstats_basecaller.json");
  int    histogram_length     = opts.GetFirstInt    ('h', "histogram-length", 400);
  string seq_key              = opts.GetFirstString ('k', "key", DEFAULT_SEQ_KEY);
  vector<string> barcodes;
  opts.GetOption(barcodes, DEFAULT_BARCODE, 'b', "barcodes");
  vector<string> output_json_filenames_barcoded;
  
  bool use_barcodes = !barcodes.empty();
  if (use_barcodes){
    opts.GetOption(output_json_filenames_barcoded, DEFAULT_BARCODE_OUTPUT, 'd', "output-barcodes");
    if(barcodes.size() != output_json_filenames_barcoded.size()) {
      IonstatsBasecallerHelp();
      return 1;
    }
    cout << "Barcoded option enabled. Generating stats for barcodes :" << endl;
    for (int i = 0; i < (int)barcodes.size(); i++) 
    {
        cout << barcodes[i] << " to ";
        cout << output_json_filenames_barcoded[i] << endl;
    }
    cout << "Pooled stats for all barcodes : " <<  output_json_filename << endl;
  }

  if(input_bam_filename.empty()) {
    IonstatsBasecallerHelp();
    return 1;
  }


  BamReader input_bam;
  if (!input_bam.Open(input_bam_filename)) {
    fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_bam_filename.c_str());
    return 1;
  }

  SamHeader sam_header = input_bam.GetHeader();
  if(!sam_header.HasReadGroups()) {
    fprintf(stderr, "[ionstats] ERROR: no read groups in %s\n", input_bam_filename.c_str());
    return 1;
  }


  ReadLengthHistogram total_full_histo;
  ReadLengthHistogram total_insert_histo;
  ReadLengthHistogram total_Q17_histo;
  ReadLengthHistogram total_Q20_histo;

  total_full_histo.Initialize(histogram_length);
  total_insert_histo.Initialize(histogram_length);
  total_Q17_histo.Initialize(histogram_length);
  total_Q20_histo.Initialize(histogram_length);

  MetricGeneratorSNR system_snr;
  BaseQVHistogram qv_histogram;

  
    const unsigned int numBarcodes = barcodes.size();
    ReadLengthHistogram bc_full_histo[numBarcodes];
    ReadLengthHistogram bc_insert_histo[numBarcodes];
    ReadLengthHistogram bc_Q17_histo[numBarcodes];
    ReadLengthHistogram bc_Q20_histo[numBarcodes];

    MetricGeneratorSNR bc_system_snr[numBarcodes];
    BaseQVHistogram bc_qv_histogram[numBarcodes];

    for (unsigned int i = 0; i < numBarcodes; i++){
      bc_full_histo[i].Initialize(histogram_length);
      bc_insert_histo[i].Initialize(histogram_length);
      bc_Q17_histo[i].Initialize(histogram_length);
      bc_Q20_histo[i].Initialize(histogram_length);
    }
    


  // We assume all read groups share the same flow order, so point iterating over all
  string flow_order;
  for (SamReadGroupIterator rg = sam_header.ReadGroups.Begin(); rg != sam_header.ReadGroups.End(); ++rg) {
    if(rg->HasFlowOrder())
      flow_order = rg->FlowOrder;
    if (not flow_order.empty())
      break;
  }

  double qv_to_error_rate[256];
  for (int qv = 0; qv < 256; qv++)
    qv_to_error_rate[qv] =  pow(10.0,-0.1*(double)qv);

  BamAlignment alignment;
  
  vector<uint16_t> flow_signal_fz(flow_order.length());
  vector<int16_t> flow_signal_zm(flow_order.length());
  
  while(input_bam.GetNextAlignment(alignment)) {
    
    int idx_barcode = -1;
    if (use_barcodes){
      // get the readgroup from the read
      string read_group;
      alignment.GetTag("RG",read_group);
      
      for (unsigned int i = 0; i < barcodes.size(); i++){
        string barcode = barcodes[i];
        if (read_group.find(barcode) == string::npos){
            continue;
        }
        else{
          idx_barcode = (int) i;
        }
      }
    }    
    
    // Record read length
    unsigned int full_length = alignment.Length;
    total_full_histo.Add(full_length);

    // Record insert length
    int insert_length = 0;
    if (alignment.GetTag("ZA",insert_length))
      total_insert_histo.Add(insert_length);

    // Compute and record Q17 and Q20
    int Q17_length = 0;
    int Q20_length = 0;
    double num_accumulated_errors = 0.0;
    for(int pos = 0; pos < alignment.Length; ++pos) {
      num_accumulated_errors += qv_to_error_rate[(int)alignment.Qualities[pos] - 33];
      if (num_accumulated_errors / (pos + 1) <= 0.02)
        Q17_length = pos + 1;
      if (num_accumulated_errors / (pos + 1) <= 0.01)
        Q20_length = pos + 1;
    }
    total_Q17_histo.Add(Q17_length);
    total_Q20_histo.Add(Q20_length);

    // Record data for system snr
    if(alignment.GetTag("ZM", flow_signal_zm)){
      system_snr.Add(flow_signal_zm, seq_key, flow_order);
      if(use_barcodes){
        bc_system_snr[idx_barcode].Add(flow_signal_zm, seq_key, flow_order);
      }
    }
    else if(alignment.GetTag("FZ", flow_signal_fz)){
      system_snr.Add(flow_signal_fz, seq_key, flow_order);
      if(use_barcodes){
        bc_system_snr[idx_barcode].Add(flow_signal_fz, seq_key, flow_order);
      }
    }

    // Record qv histogram
    qv_histogram.Add(alignment.Qualities);

    // Record data for individual barcode
    if(use_barcodes){
      bc_full_histo[idx_barcode].Add(full_length);
      bc_insert_histo[idx_barcode].Add(insert_length);
      bc_Q17_histo[idx_barcode].Add(Q17_length);
      bc_Q20_histo[idx_barcode].Add(Q20_length);
      
      bc_qv_histogram[idx_barcode].Add(alignment.Qualities);
    }
  } //end of while

  input_bam.Close();


  // Full data to json
  Json::Value output_json(Json::objectValue);
  //output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
  output_json["meta"]["format_name"] = "ionstats_basecaller";
  output_json["meta"]["format_version"] = "1.0";

  system_snr.SaveToJson(output_json);
  qv_histogram.SaveToJson(output_json);
  total_full_histo.SaveToJson(output_json["full"]);
  total_insert_histo.SaveToJson(output_json["insert"]);
  total_Q17_histo.SaveToJson(output_json["Q17"]);
  total_Q20_histo.SaveToJson(output_json["Q20"]);

  int status = 0;
  ofstream out(output_json_filename.c_str(), ios::out);
  if (out.good()) {
    out << output_json.toStyledString();
  } else {
    fprintf(stderr, "ERROR: unable to write to '%s'\n", output_json_filename.c_str());
    status = 1;
  }

  // output data for individual barcode
  if (use_barcodes){
    for (unsigned int i = 0; i < barcodes.size(); i++){ 
      Json::Value output_json_bc(Json::objectValue);
      //output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
      output_json_bc["meta"]["format_name"] = "ionstats_basecaller";
      output_json_bc["meta"]["format_version"] = "1.0";
      output_json_bc["meta"]["barcode"] = barcodes[i];
      // cout << barcodes[i] <<endl;
      bc_system_snr[i].SaveToJson(output_json_bc);
      bc_qv_histogram[i].SaveToJson(output_json_bc);
      bc_full_histo[i].SaveToJson(output_json_bc["full"]);
      bc_insert_histo[i].SaveToJson(output_json_bc["insert"]);
      bc_Q17_histo[i].SaveToJson(output_json_bc["Q17"]);
      bc_Q20_histo[i].SaveToJson(output_json_bc["Q20"]);


      ofstream out_bc(output_json_filenames_barcoded[i].c_str(), ios::out);
      // delete output_json;
      if (out_bc.good()) {
        out_bc << output_json_bc.toStyledString();
      } else {
        fprintf(stderr, "ERROR: unable to write to '%s'\n", output_json_filenames_barcoded[i].c_str());
        status = 1;
      }
    }
 }
 return status;
}



int IonstatsBasecallerReduce(const string& output_json_filename, const vector<string>& input_jsons)
{
  BaseQVHistogram qv_histogram;
  MetricGeneratorSNR system_snr;
  ReadLengthHistogram total_full_histo;
  ReadLengthHistogram total_insert_histo;
  ReadLengthHistogram total_Q17_histo;
  ReadLengthHistogram total_Q20_histo;

  for (unsigned int input_idx = 0; input_idx < input_jsons.size(); ++input_idx) {

    ifstream in(input_jsons[input_idx].c_str(), ifstream::in);
    if (!in.good()) {
      fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_jsons[0].c_str());
      return 1;
    }
    Json::Value current_input_json;
    in >> current_input_json;
    in.close();

    BaseQVHistogram current_qv_histogram;
    current_qv_histogram.LoadFromJson(current_input_json);
    qv_histogram.MergeFrom(current_qv_histogram);

    MetricGeneratorSNR current_system_snr;
    current_system_snr.LoadFromJson(current_input_json);
    system_snr.MergeFrom(current_system_snr);

    ReadLengthHistogram current_total_full_histo;
    current_total_full_histo.LoadFromJson(current_input_json["full"]);
    total_full_histo.MergeFrom(current_total_full_histo);

    ReadLengthHistogram current_total_insert_histo;
    current_total_insert_histo.LoadFromJson(current_input_json["insert"]);
    total_insert_histo.MergeFrom(current_total_insert_histo);

    ReadLengthHistogram current_total_Q17_histo;
    current_total_Q17_histo.LoadFromJson(current_input_json["Q17"]);
    total_Q17_histo.MergeFrom(current_total_Q17_histo);

    ReadLengthHistogram current_total_Q20_histo;
    current_total_Q20_histo.LoadFromJson(current_input_json["Q20"]);
    total_Q20_histo.MergeFrom(current_total_Q20_histo);
  }

  Json::Value output_json(Json::objectValue);
  //output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
  output_json["meta"]["format_name"] = "ionstats_basecaller";
  output_json["meta"]["format_version"] = "1.0";

  system_snr.SaveToJson(output_json);
  qv_histogram.SaveToJson(output_json);
  total_full_histo.SaveToJson(output_json["full"]);
  total_insert_histo.SaveToJson(output_json["insert"]);
  total_Q17_histo.SaveToJson(output_json["Q17"]);
  total_Q20_histo.SaveToJson(output_json["Q20"]);

  ofstream out(output_json_filename.c_str(), ios::out);
  if (out.good()) {
    out << output_json.toStyledString();
    return 0;
  } else {
    fprintf(stderr, "ERROR: unable to write to '%s'\n", output_json_filename.c_str());
    return 1;
  }
}
