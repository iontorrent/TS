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



void IonstatsTestFragmentsHelp()
{
  printf ("\n");
  printf ("ionstats %s-%s (%s) - Generate performance metrics and statistics for Ion sequences.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetSvnRev().c_str());
  printf ("\n");
  printf ("Usage:   ionstats tf [options]\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -i,--input                 FILE       input test fragments BAM (mapped) [required option]\n");
  printf ("  -r,--ref                   FILE       FASTA file containing TF sequences [required option]\n");
  printf ("  -o,--output                FILE       output json file [ionstats_tf.json]\n");
  printf ("  -h,--histogram-length      INT        read length histogram cutoff [400]\n");
  printf ("\n");
}




// Matches "chromosome" names from BAM to sequences in the reference.fasta file
void PopulateReferenceSequences(map<string,string> &tf_sequences, const string &fasta_filename)
{
  // Iterate through the fasta file. Check each sequence name for a match to BAM.

  ifstream fasta;
  fasta.open(fasta_filename.c_str());
  if (!fasta.is_open()) {
    printf ("Failed to open reference %s\n", fasta_filename.c_str());
    exit(1);
  }

  char line[4096] = "";

  while (!fasta.eof()) {

    if (strlen(line) <= 1 or line[0] != '>') {
      fasta.getline(line,4096);
      continue;
    }

    string name = line+1;
    string sequence;

    while (!fasta.eof()) {
      fasta.getline(line,4096);
      if (strlen(line) > 1 and line[0] == '>')
        break;
      sequence += line;
    }

    tf_sequences[name] = sequence;
  }

  fasta.close();
}




int IonstatsTestFragments(int argc, const char *argv[])
{
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string input_bam_filename   = opts.GetFirstString('i', "input", "");
  string fasta_filename       = opts.GetFirstString('r', "ref", "");
  string output_json_filename = opts.GetFirstString('o', "output", "ionstats_tf.json");
  int histogram_length        = opts.GetFirstInt   ('h', "histogram-length", 400);

  if(argc < 2 or input_bam_filename.empty() or fasta_filename.empty()) {
    IonstatsTestFragmentsHelp();
    return 1;
  }

  //
  // Prepare for metric calculation
  //

  map<string,string> tf_sequences;
  PopulateReferenceSequences(tf_sequences, fasta_filename);


  BamReader input_bam;
  if (!input_bam.Open(input_bam_filename)) {
    fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_bam_filename.c_str());
    return 1;
  }

  int num_tfs = input_bam.GetReferenceCount();


  SamHeader sam_header = input_bam.GetHeader();
  if(!sam_header.HasReadGroups()) {
    fprintf(stderr, "[ionstats] ERROR: no read groups in %s\n", input_bam_filename.c_str());
    return 1;
  }

  string flow_order;
  string key;
  for (SamReadGroupIterator rg = sam_header.ReadGroups.Begin(); rg != sam_header.ReadGroups.End(); ++rg) {
    if(rg->HasFlowOrder())
      flow_order = rg->FlowOrder;
    if(rg->HasKeySequence())
      key = rg->KeySequence;
  }


  // Need these metrics stratified by TF.

  vector<ReadLengthHistogram> called_histogram(num_tfs);
  vector<ReadLengthHistogram> aligned_histogram(num_tfs);
  vector<ReadLengthHistogram> AQ10_histogram(num_tfs);
  vector<ReadLengthHistogram> AQ17_histogram(num_tfs);
  vector<SimpleHistogram> error_by_position(num_tfs);
  vector<MetricGeneratorSNR> system_snr(num_tfs);
  vector<MetricGeneratorHPAccuracy> hp_accuracy(num_tfs);

  for (int tf = 0; tf < num_tfs; ++tf) {
    called_histogram[tf].Initialize(histogram_length);
    aligned_histogram[tf].Initialize(histogram_length);
    AQ10_histogram[tf].Initialize(histogram_length);
    AQ17_histogram[tf].Initialize(histogram_length);
    error_by_position[tf].Initialize(histogram_length);
  }

  vector<uint16_t> flow_signal_fz(flow_order.length());
  vector<int16_t> flow_signal_zm(flow_order.length());

  const RefVector& refs = input_bam.GetReferenceData();

  // Missing:
  //  - hp accuracy - tough, copy verbatim from TFMapper?


  BamAlignment alignment;
  vector<char>  MD_op;
  vector<int>   MD_len;
  MD_op.reserve(1024);
  MD_len.reserve(1024);
  string MD_tag;

  //
  // Main loop over mapped reads in the input BAM
  //

  while(input_bam.GetNextAlignment(alignment)) {


    if (!alignment.IsMapped() or !alignment.GetTag("MD",MD_tag))
      continue;

    int current_tf = alignment.RefID;

    //
    // Step 1. Parse MD tag
    //

    MD_op.clear();
    MD_len.clear();

    for (const char *MD_ptr = MD_tag.c_str(); *MD_ptr;) {

      int item_length = 0;
      if (*MD_ptr >= '0' and *MD_ptr <= '9') {    // Its a match
        MD_op.push_back('M');
        for (; *MD_ptr and *MD_ptr >= '0' and *MD_ptr <= '9'; ++MD_ptr)
          item_length = 10*item_length + *MD_ptr - '0';
      } else {
        if (*MD_ptr == '^') {                     // Its a deletion
          MD_ptr++;
          MD_op.push_back('D');
        } else                                    // Its a substitution
          MD_op.push_back('X');
        for (; *MD_ptr and *MD_ptr >= 'A' and *MD_ptr <= 'Z'; ++MD_ptr)
          item_length++;
      }
      MD_len.push_back(item_length);
    }

    //
    // Step 2. Synchronously scan through Cigar and MD, doing error accounting
    //

    int MD_idx = alignment.IsReverseStrand() ? MD_op.size()-1 : 0;
    int cigar_idx = alignment.IsReverseStrand() ? alignment.CigarData.size()-1 : 0;
    int increment = alignment.IsReverseStrand() ? -1 : 1;

    int AQ10_bases = 0;
    int AQ17_bases = 0;
    int num_bases = 0;
    int num_errors = 0;

    while (cigar_idx < (int)alignment.CigarData.size() and MD_idx < (int) MD_op.size() and cigar_idx >= 0 and MD_idx >= 0) {

      if (alignment.CigarData[cigar_idx].Length == 0) { // Try advancing cigar
        cigar_idx += increment;
        continue;
      }
      if (MD_len[MD_idx] == 0) { // Try advancing MD
        MD_idx += increment;
        continue;
      }

      // Match
      if (alignment.CigarData[cigar_idx].Type == 'M' and MD_op[MD_idx] == 'M') {
        int advance = min((int)alignment.CigarData[cigar_idx].Length, MD_len[MD_idx]);
        num_bases += advance;
        alignment.CigarData[cigar_idx].Length -= advance;
        MD_len[MD_idx] -= advance;

      // Insertion (read has a base, reference doesn't)
      } else if (alignment.CigarData[cigar_idx].Type == 'I') {
        int advance = alignment.CigarData[cigar_idx].Length;
        for (int cnt = 0; cnt < advance; ++cnt) {
          error_by_position[current_tf].Add(num_bases);
          num_bases++;
          num_errors++;
        }
        alignment.CigarData[cigar_idx].Length -= advance;

      // Deletion (reference has a base, read doesn't)
      } else if (alignment.CigarData[cigar_idx].Type == 'D' and MD_op[MD_idx] == 'D') {
        int advance = min((int)alignment.CigarData[cigar_idx].Length, MD_len[MD_idx]);
        for (int cnt = 0; cnt < advance; ++cnt) {
          error_by_position[current_tf].Add(num_bases);
          num_errors++;
        }
        alignment.CigarData[cigar_idx].Length -= advance;
        MD_len[MD_idx] -= advance;

      // Substitution
      } else if (MD_op[MD_idx] == 'X') {
        int advance = min((int)alignment.CigarData[cigar_idx].Length, MD_len[MD_idx]);
        for (int cnt = 0; cnt < advance; ++cnt) {
          error_by_position[current_tf].Add(num_bases);
          num_bases++;
          num_errors++;
        }
        alignment.CigarData[cigar_idx].Length -= advance;
        MD_len[MD_idx] -= advance;

      } else {
        printf("ionstats tf: Unexpected OP combination: %s Cigar=%c, MD=%c !\n",
            alignment.Name.c_str(), alignment.CigarData[cigar_idx].Type, MD_op[MD_idx]);
        break;
      }

      if (num_errors*10 <= num_bases)   AQ10_bases = num_bases;
      if (num_errors*50 <= num_bases)   AQ17_bases = num_bases;
    }

    //
    // Step 3. Profit
    //

    called_histogram[current_tf].Add(alignment.Length);
    aligned_histogram[current_tf].Add(num_bases);
    AQ10_histogram[current_tf].Add(AQ10_bases);
    AQ17_histogram[current_tf].Add(AQ17_bases);

    if(alignment.GetTag("ZM", flow_signal_zm))
      system_snr[current_tf].Add(flow_signal_zm, key.c_str(), flow_order);
    else if(alignment.GetTag("FZ", flow_signal_fz))
      system_snr[current_tf].Add(flow_signal_fz, key.c_str(), flow_order);


    // HP accuracy - keeping it simple

    if (!alignment.IsReverseStrand()) {

      string genome = key + tf_sequences[refs[current_tf].RefName];
      string calls = key + alignment.QueryBases;
      const char *genome_ptr = genome.c_str();
      const char *calls_ptr = calls.c_str();

      for (int flow = 0; flow < (int)flow_order.length() and *genome_ptr and *calls_ptr; ++flow) {
        int genome_hp = 0;
        int calls_hp = 0;
        while (*genome_ptr == flow_order[flow]) {
          genome_hp++;
          genome_ptr++;
        }
        while (*calls_ptr == flow_order[flow]) {
          calls_hp++;
          calls_ptr++;
        }
        hp_accuracy[current_tf].Add(genome_hp, calls_hp);
      }
    }
  }



  //
  // Processing complete, generate ionstats_tf.json
  //

  Json::Value output_json(Json::objectValue);
  output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
  output_json["meta"]["format_name"] = "ionstats_tf";
  output_json["meta"]["format_version"] = "1.0";

  output_json["results_by_tf"] = Json::objectValue;

  for (int tf = 0; tf < num_tfs; ++tf) {

    if (aligned_histogram[tf].num_reads() < 1000)
      continue;

    called_histogram[tf].SaveToJson(output_json["results_by_tf"][refs[tf].RefName]["full"]);
    aligned_histogram[tf].SaveToJson(output_json["results_by_tf"][refs[tf].RefName]["aligned"]);
    AQ10_histogram[tf].SaveToJson(output_json["results_by_tf"][refs[tf].RefName]["AQ10"]);
    AQ17_histogram[tf].SaveToJson(output_json["results_by_tf"][refs[tf].RefName]["AQ17"]);
    error_by_position[tf].SaveToJson(output_json["results_by_tf"][refs[tf].RefName]["error_by_position"]);
    system_snr[tf].SaveToJson(output_json["results_by_tf"][refs[tf].RefName]);
    hp_accuracy[tf].SaveToJson(output_json["results_by_tf"][refs[tf].RefName]);

    output_json["results_by_tf"][refs[tf].RefName]["sequence"] = tf_sequences[refs[tf].RefName];
  }

  input_bam.Close();

  ofstream out(output_json_filename.c_str(), ios::out);
  if (out.good()) {
    out << output_json.toStyledString();
    return 0;
  } else {
    fprintf(stderr, "ERROR: unable to write to '%s'\n", output_json_filename.c_str());
    return 1;
  }
}



int IonstatsTestFragmentsReduce(const string& output_json_filename, const vector<string>& input_jsons)
{

  map<string,int> tf_name_lookup;
  int num_tfs = 0;
  deque<ReadLengthHistogram>        called_histogram;
  deque<ReadLengthHistogram>        aligned_histogram;
  deque<ReadLengthHistogram>        AQ10_histogram;
  deque<ReadLengthHistogram>        AQ17_histogram;
  deque<SimpleHistogram>            error_by_position;
  deque<MetricGeneratorSNR>         system_snr;
  deque<MetricGeneratorHPAccuracy>  hp_accuracy;
  deque<string>                     tf_name;
  deque<string>                     tf_sequences;

  for (unsigned int input_idx = 0; input_idx < input_jsons.size(); ++input_idx) {

    ifstream in(input_jsons[input_idx].c_str(), ifstream::in);
    if (!in.good()) {
      fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_jsons[0].c_str());
      return 1;
    }
    Json::Value current_input_json;
    in >> current_input_json;
    in.close();

    vector<string> tf_list = current_input_json["results_by_tf"].getMemberNames();

    for (int idx = 0; idx < (int)tf_list.size(); ++idx) {
      int current_tf = -1;
      if (tf_name_lookup.count(tf_list[idx]) == 0) {
        current_tf = num_tfs++;
        tf_name_lookup[tf_list[idx]] = current_tf;
        called_histogram.push_back(ReadLengthHistogram());
        aligned_histogram.push_back(ReadLengthHistogram());
        AQ10_histogram.push_back(ReadLengthHistogram());
        AQ17_histogram.push_back(ReadLengthHistogram());
        error_by_position.push_back(SimpleHistogram());
        system_snr.push_back(MetricGeneratorSNR());
        hp_accuracy.push_back(MetricGeneratorHPAccuracy());
        tf_name.push_back(tf_list[idx]);
        tf_sequences.push_back(current_input_json["results_by_tf"][tf_list[idx]]["sequence"].asString());

      } else // TF already encountered
        current_tf = tf_name_lookup[tf_list[idx]];

      ReadLengthHistogram current_called_histogram;
      current_called_histogram.LoadFromJson(current_input_json["results_by_tf"][tf_list[idx]]["full"]);
      called_histogram[current_tf].MergeFrom(current_called_histogram);

      ReadLengthHistogram current_aligned_histogram;
      current_aligned_histogram.LoadFromJson(current_input_json["results_by_tf"][tf_list[idx]]["aligned"]);
      aligned_histogram[current_tf].MergeFrom(current_aligned_histogram);

      ReadLengthHistogram current_AQ10_histogram;
      current_AQ10_histogram.LoadFromJson(current_input_json["results_by_tf"][tf_list[idx]]["AQ10"]);
      AQ10_histogram[current_tf].MergeFrom(current_AQ10_histogram);

      ReadLengthHistogram current_AQ17_histogram;
      current_AQ17_histogram.LoadFromJson(current_input_json["results_by_tf"][tf_list[idx]]["AQ17"]);
      AQ17_histogram[current_tf].MergeFrom(current_AQ17_histogram);

      MetricGeneratorSNR current_system_snr;
      current_system_snr.LoadFromJson(current_input_json["results_by_tf"][tf_list[idx]]);
      system_snr[current_tf].MergeFrom(current_system_snr);

      SimpleHistogram current_error_by_position;
      current_error_by_position.LoadFromJson(current_input_json["results_by_tf"][tf_list[idx]]["error_by_position"]);
      error_by_position[current_tf].MergeFrom(current_error_by_position);

      MetricGeneratorHPAccuracy current_hp_accuracy;
      current_hp_accuracy.LoadFromJson(current_input_json["results_by_tf"][tf_list[idx]]);
      hp_accuracy[current_tf].MergeFrom(current_hp_accuracy);
    }
  }


  Json::Value output_json(Json::objectValue);
  output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
  output_json["meta"]["format_name"] = "ionstats_tf";
  output_json["meta"]["format_version"] = "1.0";

  output_json["results_by_tf"] = Json::objectValue;

  for (int tf = 0; tf < num_tfs; ++tf) {
    called_histogram[tf].SaveToJson(output_json["results_by_tf"][tf_name[tf]]["full"]);
    aligned_histogram[tf].SaveToJson(output_json["results_by_tf"][tf_name[tf]]["aligned"]);
    AQ10_histogram[tf].SaveToJson(output_json["results_by_tf"][tf_name[tf]]["AQ10"]);
    AQ17_histogram[tf].SaveToJson(output_json["results_by_tf"][tf_name[tf]]["AQ17"]);
    error_by_position[tf].SaveToJson(output_json["results_by_tf"][tf_name[tf]]["error_by_position"]);
    system_snr[tf].SaveToJson(output_json["results_by_tf"][tf_name[tf]]);
    hp_accuracy[tf].SaveToJson(output_json["results_by_tf"][tf_name[tf]]);
    output_json["results_by_tf"][tf_name[tf]]["sequence"] = tf_sequences[tf];
  }

  ofstream out(output_json_filename.c_str(), ios::out);
  if (out.good()) {
    out << output_json.toStyledString();
    return 0;
  } else {
    fprintf(stderr, "ERROR: unable to write to '%s'\n", output_json_filename.c_str());
    return 1;
  }
}



