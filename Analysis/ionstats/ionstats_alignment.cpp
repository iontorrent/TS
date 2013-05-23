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



void IonstatsAlignmentHelp()
{
  printf ("\n");
  printf ("ionstats %s-%s (%s) - Generate performance metrics and statistics for Ion sequences.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetSvnRev().c_str());
  printf ("\n");
  printf ("Usage:   ionstats alignment [options]\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -i,--input                 FILE       input BAM (mapped) [required option]\n");
  printf ("  -o,--output                FILE       output json file [ionstats_alignment.json]\n");
  printf ("  -h,--histogram-length      INT        read length histogram cutoff [400]\n");
  printf ("  -m,--minimum-aq-length     INT        minimum AQ read length [21]\n");
  printf ("\n");
}



// Metrics in ionstats_alignment.json should carry the following data:
//
// - Histogram of read lengths (copy of basecaller's "full" histogram) - done
// - Histogram of aligned lengths - done
// - Histogram of AQ17 and AQ20 and AQ47 aligned lengths - done
// - Genome name, genome version, mapper version, genome size - ???
// - Error rate by position: numerator and denominator - ??? (actually might be very easy)


int IonstatsAlignment(int argc, const char *argv[])
{
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string input_bam_filename   = opts.GetFirstString('i', "input", "");
  string output_json_filename = opts.GetFirstString('o', "output", "ionstats_alignment.json");
  int histogram_length        = opts.GetFirstInt   ('h', "histogram-length", 400);
  int minimum_aq_length       = opts.GetFirstInt   ('m', "minimum-aq-length", 21);

  if(argc < 2 or input_bam_filename.empty()) {
    IonstatsAlignmentHelp();
    return 1;
  }

  //
  // Prepare for metric calculation
  //

  BamReader input_bam;
  if (!input_bam.Open(input_bam_filename)) {
    fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_bam_filename.c_str());
    return 1;
  }

  ReadLengthHistogram called_histogram;
  ReadLengthHistogram aligned_histogram;
  ReadLengthHistogram AQ7_histogram;
  ReadLengthHistogram AQ10_histogram;
  ReadLengthHistogram AQ17_histogram;
  ReadLengthHistogram AQ20_histogram;
  ReadLengthHistogram AQ30_histogram;
  ReadLengthHistogram AQ47_histogram;
  SimpleHistogram error_by_position;

  called_histogram.Initialize(histogram_length);
  aligned_histogram.Initialize(histogram_length);
  AQ7_histogram.Initialize(histogram_length);
  AQ10_histogram.Initialize(histogram_length);
  AQ17_histogram.Initialize(histogram_length);
  AQ20_histogram.Initialize(histogram_length);
  AQ30_histogram.Initialize(histogram_length);
  AQ47_histogram.Initialize(histogram_length);
  error_by_position.Initialize(histogram_length);

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

    // Record read length
    called_histogram.Add(alignment.Length);

    if (!alignment.IsMapped() or !alignment.GetTag("MD",MD_tag))
      continue;

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

    int AQ7_bases = 0;
    int AQ10_bases = 0;
    int AQ17_bases = 0;
    int AQ20_bases = 0;
    int AQ30_bases = 0;
    int AQ47_bases = 0;
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
          error_by_position.Add(num_bases);
          num_bases++;
          num_errors++;
        }
        alignment.CigarData[cigar_idx].Length -= advance;

      // Deletion (reference has a base, read doesn't)
      } else if (alignment.CigarData[cigar_idx].Type == 'D' and MD_op[MD_idx] == 'D') {
        int advance = min((int)alignment.CigarData[cigar_idx].Length, MD_len[MD_idx]);
        for (int cnt = 0; cnt < advance; ++cnt) {
          error_by_position.Add(num_bases);
          num_errors++;
        }
        alignment.CigarData[cigar_idx].Length -= advance;
        MD_len[MD_idx] -= advance;

      // Substitution
      } else if (MD_op[MD_idx] == 'X') {
        int advance = min((int)alignment.CigarData[cigar_idx].Length, MD_len[MD_idx]);
        for (int cnt = 0; cnt < advance; ++cnt) {
          error_by_position.Add(num_bases);
          num_bases++;
          num_errors++;
        }
        alignment.CigarData[cigar_idx].Length -= advance;
        MD_len[MD_idx] -= advance;

      } else {
        printf("ionstats alignment: Unexpected OP combination: %s Cigar=%c, MD=%c !\n",
            alignment.Name.c_str(), alignment.CigarData[cigar_idx].Type, MD_op[MD_idx]);
        break;
      }

      if (num_errors*5 <= num_bases)    AQ7_bases = num_bases;
      if (num_errors*10 <= num_bases)   AQ10_bases = num_bases;
      if (num_errors*50 <= num_bases)   AQ17_bases = num_bases;
      if (num_errors*100 <= num_bases)  AQ20_bases = num_bases;
      if (num_errors*1000 <= num_bases) AQ30_bases = num_bases;
      if (num_errors == 0)              AQ47_bases = num_bases;
    }

    //
    // Step 3. Profit
    //

    aligned_histogram.Add(num_bases);
    if (AQ7_bases >= minimum_aq_length)     AQ7_histogram.Add(AQ7_bases);
    if (AQ10_bases >= minimum_aq_length)    AQ10_histogram.Add(AQ10_bases);
    if (AQ17_bases >= minimum_aq_length)    AQ17_histogram.Add(AQ17_bases);
    if (AQ20_bases >= minimum_aq_length)    AQ20_histogram.Add(AQ20_bases);
    if (AQ30_bases >= minimum_aq_length)    AQ30_histogram.Add(AQ30_bases);
    if (AQ47_bases >= minimum_aq_length)    AQ47_histogram.Add(AQ47_bases);
  }

  input_bam.Close();


  //
  // Processing complete, generate ionstats_alignment.json
  //

  Json::Value output_json(Json::objectValue);
  output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
  output_json["meta"]["format_name"] = "ionstats_alignment";
  output_json["meta"]["format_version"] = "1.0";

  called_histogram.SaveToJson(output_json["full"]);
  aligned_histogram.SaveToJson(output_json["aligned"]);
  AQ7_histogram.SaveToJson(output_json["AQ7"]);
  AQ10_histogram.SaveToJson(output_json["AQ10"]);
  AQ17_histogram.SaveToJson(output_json["AQ17"]);
  AQ20_histogram.SaveToJson(output_json["AQ20"]);
  AQ30_histogram.SaveToJson(output_json["AQ30"]);
  AQ47_histogram.SaveToJson(output_json["AQ47"]);
  error_by_position.SaveToJson(output_json["error_by_position"]);

  ofstream out(output_json_filename.c_str(), ios::out);
  if (out.good()) {
    out << output_json.toStyledString();
    return 0;
  } else {
    fprintf(stderr, "ERROR: unable to write to '%s'\n", output_json_filename.c_str());
    return 1;
  }

  return 0;
}







int IonstatsAlignmentReduce(const string& output_json_filename, const vector<string>& input_jsons)
{

  ReadLengthHistogram called_histogram;
  ReadLengthHistogram aligned_histogram;
  ReadLengthHistogram AQ7_histogram;
  ReadLengthHistogram AQ10_histogram;
  ReadLengthHistogram AQ17_histogram;
  ReadLengthHistogram AQ20_histogram;
  ReadLengthHistogram AQ30_histogram;
  ReadLengthHistogram AQ47_histogram;
  SimpleHistogram error_by_position;

  for (unsigned int input_idx = 0; input_idx < input_jsons.size(); ++input_idx) {

    ifstream in(input_jsons[input_idx].c_str(), ifstream::in);
    if (!in.good()) {
      fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_jsons[0].c_str());
      return 1;
    }
    Json::Value current_input_json;
    in >> current_input_json;
    in.close();

    ReadLengthHistogram current_called_histogram;
    current_called_histogram.LoadFromJson(current_input_json["full"]);
    called_histogram.MergeFrom(current_called_histogram);

    ReadLengthHistogram current_aligned_histogram;
    current_aligned_histogram.LoadFromJson(current_input_json["aligned"]);
    aligned_histogram.MergeFrom(current_aligned_histogram);

    ReadLengthHistogram current_AQ7_histogram;
    current_AQ7_histogram.LoadFromJson(current_input_json["AQ7"]);
    AQ7_histogram.MergeFrom(current_AQ7_histogram);

    ReadLengthHistogram current_AQ10_histogram;
    current_AQ10_histogram.LoadFromJson(current_input_json["AQ10"]);
    AQ10_histogram.MergeFrom(current_AQ10_histogram);

    ReadLengthHistogram current_AQ17_histogram;
    current_AQ17_histogram.LoadFromJson(current_input_json["AQ17"]);
    AQ17_histogram.MergeFrom(current_AQ17_histogram);

    ReadLengthHistogram current_AQ20_histogram;
    current_AQ20_histogram.LoadFromJson(current_input_json["AQ20"]);
    AQ20_histogram.MergeFrom(current_AQ20_histogram);

    ReadLengthHistogram current_AQ30_histogram;
    current_AQ30_histogram.LoadFromJson(current_input_json["AQ30"]);
    AQ30_histogram.MergeFrom(current_AQ30_histogram);

    ReadLengthHistogram current_AQ47_histogram;
    current_AQ47_histogram.LoadFromJson(current_input_json["AQ47"]);
    AQ47_histogram.MergeFrom(current_AQ47_histogram);

    SimpleHistogram current_error_by_position;
    current_error_by_position.LoadFromJson(current_input_json["error_by_position"]);
    error_by_position.MergeFrom(current_error_by_position);

  }

  Json::Value output_json(Json::objectValue);
  output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
  output_json["meta"]["format_name"] = "ionstats_alignment";
  output_json["meta"]["format_version"] = "1.0";

  called_histogram.SaveToJson(output_json["full"]);
  aligned_histogram.SaveToJson(output_json["aligned"]);
  AQ7_histogram.SaveToJson(output_json["AQ7"]);
  AQ10_histogram.SaveToJson(output_json["AQ10"]);
  AQ17_histogram.SaveToJson(output_json["AQ17"]);
  AQ20_histogram.SaveToJson(output_json["AQ20"]);
  AQ30_histogram.SaveToJson(output_json["AQ30"]);
  AQ47_histogram.SaveToJson(output_json["AQ47"]);
  error_by_position.SaveToJson(output_json["error_by_position"]);

  ofstream out(output_json_filename.c_str(), ios::out);
  if (out.good()) {
    out << output_json.toStyledString();
    return 0;
  } else {
    fprintf(stderr, "ERROR: unable to write to '%s'\n", output_json_filename.c_str());
    return 1;
  }
}




