/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ionstats.h"

#include <string>
#include <fstream>
#include <map>
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

struct BarcodeInfo {
  vector<int>   flow_seq;     // flow-space vector representation for the barcode
  int           start_flow;   // calculated from the start base & end base, used for scoring/matching
  int           end_flow;
};

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
  printf ("  -b,--bc-adjust				BOOL       adjust barcode alignment result with key sequence\n");
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
  bool bc_adjust			  = opts.GetFirstBoolean('-', "bc-adjust", false);

  if(argc < 2 or input_bam_filename.empty()) {
    IonstatsAlignmentHelp();
    return 1;
  }

  //
  // Prepare for metric calculation
  //

  map<string, string> flow_orders;
  map<string, string> keys;
  map<string, int> keyBases;
  map<string, BarcodeInfo> bcInfos;
  BamReader input_bam;
  if (!input_bam.Open(input_bam_filename)) {
    fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_bam_filename.c_str());
    return 1;
  }

  if(bc_adjust)
  {
	  bool hasBc = false;
	  SamHeader samHeader = input_bam.GetHeader();
	  if(!samHeader.HasReadGroups())
	  {
		input_bam.Close();
		fprintf(stderr, "[ionstats] ERROR: there is no read group in %s\n", input_bam_filename.c_str());
		return 1;
	  }

	  for (SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr )
	  {
		if(!itr->HasKeySequence())
		{
			keys[itr->ID] = "";
			keyBases[itr->ID] = 0;
		}
		else
		{
			keys[itr->ID] = itr->KeySequence;
			int len = (itr->KeySequence).length() - 4;
			if(len < 1)
			{
				len = 0;
			}
			else
			{
				hasBc = true;
			}
			keyBases[itr->ID] = len;
		}

		flow_orders[itr->ID] = itr->FlowOrder;
	    
		int flow = 0;
		int curBase = 0;
		BarcodeInfo bcInfo;
		bcInfo.start_flow = -1;
		bcInfo.end_flow = -1;
		bcInfo.flow_seq.assign((itr->FlowOrder).length(), 0);

		while(curBase < (int)(itr->KeySequence).length() && flow < (int)(itr->FlowOrder).length())
		{
			while(curBase < (int)(itr->KeySequence).length() && itr->KeySequence[curBase] == itr->FlowOrder[flow])
			{
				bcInfo.flow_seq[flow]++;
				++curBase;
			}
			// grab the next flow after we sequence through the key, this will be the first flow we will want to count towards barcode matching/scoring, even if its a 0-mer flow
			if(bcInfo.start_flow == -1 && curBase >= 4)
			{
				bcInfo.start_flow = flow + 1;
			}
			// grab the last positive incorporating flow for the barcode, any 0-mer flows after this and before the insert or adapter would not be counted in the barcode matching/scoring
			if(bcInfo.end_flow == -1 && curBase >= (int)(itr->KeySequence).length())
			{
				bcInfo.end_flow = flow - 3;
			}

			++flow;
		}
		if(bcInfo.end_flow == -1)
		{
			bcInfo.end_flow = flow - 1;
		}

		bcInfos[itr->ID] = bcInfo;
	  }
	  
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

  ReadLengthHistogram called_histogram_bc;
  ReadLengthHistogram aligned_histogram_bc;
  ReadLengthHistogram AQ7_histogram_bc;
  ReadLengthHistogram AQ10_histogram_bc;
  ReadLengthHistogram AQ17_histogram_bc;
  ReadLengthHistogram AQ20_histogram_bc;
  ReadLengthHistogram AQ30_histogram_bc;
  ReadLengthHistogram AQ47_histogram_bc;
  if(bc_adjust)
  {
	  called_histogram_bc.Initialize(histogram_length);
	  aligned_histogram_bc.Initialize(histogram_length);
	  AQ7_histogram_bc.Initialize(histogram_length);
	  AQ10_histogram_bc.Initialize(histogram_length);
	  AQ17_histogram_bc.Initialize(histogram_length);
	  AQ20_histogram_bc.Initialize(histogram_length);
	  AQ30_histogram_bc.Initialize(histogram_length);
	  AQ47_histogram_bc.Initialize(histogram_length);
  }

  BamAlignment alignment;
  vector<char>  MD_op;
  vector<int>   MD_len;
  MD_op.reserve(1024);
  MD_len.reserve(1024);
  string MD_tag;

  long sumReads = 0;
  long sumBcErrReads = 0;

  //
  // Main loop over mapped reads in the input BAM
  //

  while(input_bam.GetNextAlignment(alignment)) 
  {
	++sumReads;

    int bcBases = 0;
	int bcErrors = 0;
	if(bc_adjust)
	{
		int rgind = alignment.Name.find(":");
		string rgname = alignment.Name;
		if(rgind > 0)
		{
		  rgname = alignment.Name.substr(0, rgind);
		}
		if(alignment.HasTag("RG"))
		{
		  string rgname2;
		  if(alignment.GetTag("RG", rgname2))
		  {
			rgname = rgname2;
		  }
		}

		if(keyBases.find(rgname) != keyBases.end())
		{
			bcBases = keyBases[rgname];
		}

		if(alignment.HasTag("XB"))
		{
			alignment.GetTag("XB", bcErrors);
		}
		/*else
		{
			vector<int16_t> new_flow_signal;
			vector<int> base_to_flow;
			base_to_flow.reserve(flow_orders[rgname].length());

			if(alignment.GetTag("ZM", new_flow_signal))
			{
				for(int pos = 0; pos < (int)new_flow_signal.size(); ++pos)
				{
					int n = max(0,(int)new_flow_signal[pos]);
					float v = (float)n / 256.0 + 0.5;
					n = (int)v;
					while(n > 0)
					{
						base_to_flow.push_back(pos);
						--n;
					}
				}
		  
				for(int flow = 0, base = 0; flow <= bcInfos[rgname].end_flow; ++flow)
				{
					int hp_length = 0;
					while(base < (int) base_to_flow.size() && base_to_flow[base] == flow) 
					{
						++base;
						++hp_length;
					}
					if(flow >= bcInfos[rgname].start_flow)
					{
						bcErrors += abs(bcInfos[rgname].flow_seq[flow] - hp_length);
					}
				}
			}
		}*/

		if(bcErrors > bcBases)
		{
			bcErrors = bcBases;
		}
		if(bcErrors > 0)
		{
			++sumBcErrReads;
		}
	}

    // Record read length
    called_histogram.Add(alignment.Length);
	if(bc_adjust)
	{
		called_histogram_bc.Add(alignment.Length + bcBases);
	}

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

    int AQ7_bases_bc = 0;
    int AQ10_bases_bc = 0;
    int AQ17_bases_bc = 0;
    int AQ20_bases_bc = 0;
    int AQ30_bases_bc = 0;
    int AQ47_bases_bc = 0;
    int num_bases_bc = 0;
	int num_errors_bc = 0;   

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

	  if(bc_adjust)
	  {
		  num_bases_bc = num_bases + bcBases;
		  num_errors_bc = num_errors + bcErrors;

		  if (num_errors_bc*5 <= num_bases_bc)    AQ7_bases_bc = num_bases_bc;
		  if (num_errors_bc*10 <= num_bases_bc)   AQ10_bases_bc = num_bases_bc;
		  if (num_errors_bc*50 <= num_bases_bc)   AQ17_bases_bc = num_bases_bc;
		  if (num_errors_bc*100 <= num_bases_bc)  AQ20_bases_bc = num_bases_bc;
		  if (num_errors_bc*1000 <= num_bases_bc) AQ30_bases_bc = num_bases_bc;
		  if (num_errors_bc == 0)				  AQ47_bases_bc = num_bases_bc;
	  }
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

	if(bc_adjust)
	{
		aligned_histogram_bc.Add(num_bases_bc);
		if (AQ7_bases_bc >= minimum_aq_length)     AQ7_histogram_bc.Add(AQ7_bases_bc);
		if (AQ10_bases_bc >= minimum_aq_length)    AQ10_histogram_bc.Add(AQ10_bases_bc);
		if (AQ17_bases_bc >= minimum_aq_length)    AQ17_histogram_bc.Add(AQ17_bases_bc);
		if (AQ20_bases_bc >= minimum_aq_length)    AQ20_histogram_bc.Add(AQ20_bases_bc);
		if (AQ30_bases_bc >= minimum_aq_length)    AQ30_histogram_bc.Add(AQ30_bases_bc);
		if (AQ47_bases_bc >= minimum_aq_length)    AQ47_histogram_bc.Add(AQ47_bases_bc);
	}
  }

  input_bam.Close();

  cout << endl << input_bam_filename << " has " << sumReads << " reads and " << sumBcErrReads << " reads have bc errors." << endl;

  //
  // Processing complete, generate ionstats_alignment.json
  //

  Json::Value output_json(Json::objectValue);
  output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
  output_json["meta"]["format_name"] = "ionstats_alignment";
  output_json["meta"]["format_version"] = "1.0";

  if(bc_adjust)
  {
	  called_histogram_bc.SaveToJson(output_json["full"]);
	  aligned_histogram_bc.SaveToJson(output_json["aligned"]);
	  AQ7_histogram_bc.SaveToJson(output_json["AQ7"]);
	  AQ10_histogram_bc.SaveToJson(output_json["AQ10"]);
	  AQ17_histogram_bc.SaveToJson(output_json["AQ17"]);
	  AQ20_histogram_bc.SaveToJson(output_json["AQ20"]);
	  AQ30_histogram_bc.SaveToJson(output_json["AQ30"]);
	  AQ47_histogram_bc.SaveToJson(output_json["AQ47"]);

	  called_histogram.SaveToJson(output_json["Original"]["full"]);
	  aligned_histogram.SaveToJson(output_json["Original"]["aligned"]);
	  AQ7_histogram.SaveToJson(output_json["Original"]["AQ7"]);
	  AQ10_histogram.SaveToJson(output_json["Original"]["AQ10"]);
	  AQ17_histogram.SaveToJson(output_json["Original"]["AQ17"]);
	  AQ20_histogram.SaveToJson(output_json["Original"]["AQ20"]);
	  AQ30_histogram.SaveToJson(output_json["Original"]["AQ30"]);
	  AQ47_histogram.SaveToJson(output_json["Original"]["AQ47"]);  
	  error_by_position.SaveToJson(output_json["Original"]["error_by_position"]);
  }
  else
  {
	  called_histogram.SaveToJson(output_json["full"]);
	  aligned_histogram.SaveToJson(output_json["aligned"]);
	  AQ7_histogram.SaveToJson(output_json["AQ7"]);
	  AQ10_histogram.SaveToJson(output_json["AQ10"]);
	  AQ17_histogram.SaveToJson(output_json["AQ17"]);
	  AQ20_histogram.SaveToJson(output_json["AQ20"]);
	  AQ30_histogram.SaveToJson(output_json["AQ30"]);
	  AQ47_histogram.SaveToJson(output_json["AQ47"]);  
  }
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




