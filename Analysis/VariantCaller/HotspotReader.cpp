/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HotspotReader.cpp
//! @ingroup  VariantCaller
//! @brief    Customized hotspot VCF parser

#include "HotspotReader.h"

#include <stdlib.h>
#include <errno.h>


HotspotReader::HotspotReader()
{
  ref_reader_ = NULL;
  has_more_variants_ = false;
  line_number_ = 0;
  next_chr_ = 0;
  next_pos_ = 0;
  hint_header_ = 0;
  hint_cur_ = 0;
}



HotspotReader::~HotspotReader()
{
}

void HotspotReader::Initialize(const ReferenceReader &ref_reader, const string& hotspot_vcf_filename)
{
  if (hotspot_vcf_filename.empty())
    return;
  ref_reader_ = &ref_reader;
  line_number_ = 0;

  hotspot_vcf_.parseSamples = false;
  string tmp = hotspot_vcf_filename;
  hotspot_vcf_.open(tmp);
  if (not hotspot_vcf_.is_open()) {
    cerr << "ERROR: Could not open hotspot file : " << hotspot_vcf_filename << " : " << strerror(errno) << endl;
    exit(1);
  }

  has_more_variants_ = true;
  FetchNextVariant();

  MakeHintQueue(hotspot_vcf_filename);
  // while (!blacklist_queue.empty())
  // {
  //   std::cout << "BL: " << blacklist_queue.front().first << ", " << blacklist_queue.front().second << endl;
  //   blacklist_queue.pop();
  // }
}

void HotspotReader::Initialize(const ReferenceReader &ref_reader)
{
  ref_reader_ = &ref_reader;
  line_number_ = 0;

  hotspot_vcf_.parseSamples = false;
  has_more_variants_ = false;
}

void HotspotReader::MakeHintQueue(const string& hotspot_vcf_filename)
{
  // go through the entire vcf to generate the blacklist
  ifstream hotspot_vcf;
  hotspot_vcf.open(hotspot_vcf_filename.c_str(), ifstream::in);

  string bstrand = "BSTRAND";
  while (!hotspot_vcf.eof()) {
    string line;
    getline(hotspot_vcf, line);
    if (line[0] != '#') {
      // look for the BSTRAND tag
      size_t found = line.find(bstrand);
      if (found != string::npos) {
	// found BSTRAND, look for semicolon delimiter or end of line
	long int hint = NO_HINT;
	size_t semicolon_pos = line.find(";", found+bstrand.size());
	if (semicolon_pos == string::npos)
	  semicolon_pos = hotspot_vcf.gcount()-1;  // impute a semicolon at the end of the line
	// look for the code B
	size_t b_pos = line.find("B", found+bstrand.size());
	// look for the code R
	size_t r_pos = line.find("R", found+bstrand.size());
	// look for the code F
	size_t f_pos = line.find("F", found+bstrand.size());
	size_t s_pos = line.find("S", found+bstrand.size());
	bool blacklist = false;
	if (f_pos < semicolon_pos) {
	  hint = FWD_BAD_HINT;
	}
	if (r_pos < semicolon_pos) {
	  hint = REV_BAD_HINT;
	}
	if (b_pos < semicolon_pos){
	  hint = BOTH_BAD_HINT;
	}
	if ( (f_pos < semicolon_pos) &&  (r_pos < semicolon_pos)){
	  hint = BOTH_BAD_HINT;	}
	
	// blacklist this position
	if (hint != NO_HINT) {
	  // #CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT [SAMPLE1 .. SAMPLEN]
	  vector<string> fields = split(line, '\t');

	  string sequenceName = fields.at(0);
	  char* end; // dummy variable for strtoll
	  long int pos = strtoll(fields.at(1).c_str(), &end, 10);
	  long int chrom_idx = ref_reader_->chr_idx(sequenceName.c_str());
	  hint_item hint_entry;
	  hint_entry.chr_ind = chrom_idx;
	  hint_entry.pos = pos-1;
	  hint_entry.value = hint;
	  hint_entry.rlen = 0;
	  hint_vec.push_back(hint_entry);
	} else if (s_pos < semicolon_pos) {
          vector<string> fields = split(line, '\t');
          string sequenceName = fields.at(0);
          char* end; // dummy variable for strtoll
          long int pos = strtoll(fields.at(1).c_str(), &end, 10);
          long int chrom_idx = ref_reader_->chr_idx(sequenceName.c_str());
	  vector<string> alts = split(fields.at(4), ',');
	  vector<string> bstr = split(line.substr(found+bstrand.size()+1, semicolon_pos-(found+bstrand.size()+1)), ',');
	  int len = fields.at(3).size();
	  for (unsigned int i = 0; i < alts.size(); i++) {
	    if (bstr.at(i)[0] == 'S') {
	      hint = BOTH_BAD_HINT;
	      switch(bstr.at(i)[1]) {
		case 'b':  
		  hint = BOTH_BAD_HINT;
		  break;
 		case 'f':
		  hint = FWD_BAD_HINT;
		  break;
		case 'r':
		  hint = REV_BAD_HINT;
		  break;
	      }
		hint_item hint_entry;
          	hint_entry.chr_ind = chrom_idx;
          	hint_entry.pos = pos-1;
          	hint_entry.value = hint;
		hint_entry.rlen = len;
		hint_entry.alt = alts.at(i);
		hint_vec.push_back(hint_entry);
	    }
	  }
	}
      }
    }
  }
   //std::cout << "BL size: " << hint_vec.size() << endl; 
}
		  
  
void HotspotReader::FetchNextVariant()
{
  if (not has_more_variants_)
    return;

  next_.clear();

  vcf::Variant current_hotspot(hotspot_vcf_);

  while (has_more_variants_) {
    try {
        has_more_variants_ = hotspot_vcf_.getNextVariant(current_hotspot);
    }
    catch (...) {
        cerr << "FATAL ERROR: invalid variant in hotspot file. Check the file format " << current_hotspot.sequenceName << endl;
        exit(1);
    }
    if (not has_more_variants_)
      return;

    next_chr_ = ref_reader_->chr_idx(current_hotspot.sequenceName.c_str());
    next_pos_ = current_hotspot.position - 1;

    if (next_chr_ < 0) {
      cerr << "ERROR: invalid chromosome name in hotspot file " << current_hotspot.sequenceName << endl;
      exit(1);
    }


    vector<string>& min_allele_freq = current_hotspot.info["min_allele_freq"];
    vector<string>& strand_bias = current_hotspot.info["strand_bias"];
    vector<string>& min_coverage = current_hotspot.info["min_coverage"];
    vector<string>& min_coverage_each_strand = current_hotspot.info["min_coverage_each_strand"];
    vector<string>& min_var_coverage = current_hotspot.info["min_var_coverage"];
    vector<string>& min_variant_score = current_hotspot.info["min_variant_score"];
    vector<string>& data_quality_stringency = current_hotspot.info["data_quality_stringency"];
    vector<string>& hp_max_length = current_hotspot.info["hp_max_length"];

    vector<string>& filter_unusual_predictions = current_hotspot.info["filter_unusual_predictions"];
    vector<string>& filter_insertion_predictions = current_hotspot.info["filter_insertion_predictions"];
    vector<string>& filter_deletion_predictions = current_hotspot.info["filter_deletion_predictions"];
    vector<string>& min_tag_fam_size = current_hotspot.info["min_tag_fam_size"];
    vector<string>& sse_prob_threshold = current_hotspot.info["sse_prob_threshold"];

    // collect bad-strand info
    vector<string>& black_list_strand = current_hotspot.info["BSTRAND"];

    next_.reserve(current_hotspot.alt.size());
    for (unsigned int alt_idx = 0; alt_idx < current_hotspot.alt.size(); ++alt_idx) {
      if (current_hotspot.ref == current_hotspot.alt[alt_idx])
        continue;
      
      next_.push_back(HotspotAllele());
      HotspotAllele& hotspot = next_.back();

      hotspot.chr = next_chr_;
      hotspot.pos = next_pos_;
      hotspot.ref_length = current_hotspot.ref.length();
      hotspot.alt = current_hotspot.alt[alt_idx];

      int altlen = hotspot.alt.size();

      if (hotspot.ref_length == 1 and altlen == 1) {
        hotspot.type = ALLELE_SNP;
        hotspot.length = altlen;
      } else if (hotspot.ref_length == altlen) {
        hotspot.type = ALLELE_MNP;
        hotspot.length = altlen;
      } else if (hotspot.ref_length > altlen and altlen == 1) {
        hotspot.type = ALLELE_DELETION;
        hotspot.length = hotspot.ref_length-1;
      } else if (hotspot.ref_length < altlen and hotspot.ref_length == 1) {
        hotspot.type = ALLELE_INSERTION;
        hotspot.length = altlen-1;
      } else {
        hotspot.type = ALLELE_COMPLEX;
        hotspot.length = altlen;
      }



      if (alt_idx < min_allele_freq.size() and min_allele_freq[alt_idx] != ".") {
        hotspot.params.min_allele_freq_override = true;
        hotspot.params.min_allele_freq = atof(min_allele_freq[alt_idx].c_str());
      }

      if (alt_idx < strand_bias.size() and strand_bias[alt_idx] != ".") {
        hotspot.params.strand_bias_override = true;
        hotspot.params.strand_bias = atof(strand_bias[alt_idx].c_str());
      }

      if (alt_idx < min_coverage.size() and min_coverage[alt_idx] != ".") {
        hotspot.params.min_coverage_override = true;
        hotspot.params.min_coverage = atoi(min_coverage[alt_idx].c_str());
      }

      if (alt_idx < min_coverage_each_strand.size() and min_coverage_each_strand[alt_idx] != ".") {
        hotspot.params.min_coverage_each_strand_override = true;
        hotspot.params.min_coverage_each_strand = atoi(min_coverage_each_strand[alt_idx].c_str());
      }

      if (alt_idx < min_var_coverage.size() and min_var_coverage[alt_idx] != ".") {
        hotspot.params.min_var_coverage_override = true;
        hotspot.params.min_var_coverage = atoi(min_var_coverage[alt_idx].c_str());
      }

      if (alt_idx < min_variant_score.size() and min_variant_score[alt_idx] != ".") {
        hotspot.params.min_variant_score_override = true;
        hotspot.params.min_variant_score = atof(min_variant_score[alt_idx].c_str());
      }

      if (alt_idx < data_quality_stringency.size() and data_quality_stringency[alt_idx] != ".") {
        hotspot.params.data_quality_stringency_override = true;
        hotspot.params.data_quality_stringency = atof(data_quality_stringency[alt_idx].c_str());
      }
      
      if (alt_idx < hp_max_length.size() and hp_max_length[alt_idx] != ".") {
        hotspot.params.hp_max_length_override = true;
        hotspot.params.hp_max_length = atoi(hp_max_length[alt_idx].c_str());
      }

      if (alt_idx < filter_unusual_predictions.size() and filter_unusual_predictions[alt_idx] != ".") {
        hotspot.params.filter_unusual_predictions_override = true;
        hotspot.params.filter_unusual_predictions = atof(filter_unusual_predictions[alt_idx].c_str());
      }

      if (alt_idx < filter_insertion_predictions.size() and filter_insertion_predictions[alt_idx] != ".") {
        hotspot.params.filter_insertion_predictions_override = true;
        hotspot.params.filter_insertion_predictions = atof(filter_insertion_predictions[alt_idx].c_str());
      }

      if (alt_idx < filter_deletion_predictions.size() and filter_deletion_predictions[alt_idx] != ".") {
        hotspot.params.filter_deletion_predictions_override = true;
        hotspot.params.filter_deletion_predictions = atof(filter_deletion_predictions[alt_idx].c_str());
      }

      if (alt_idx < min_tag_fam_size.size() and min_tag_fam_size[alt_idx] != ".") {
        hotspot.params.min_tag_fam_size_override = true;
        hotspot.params.min_tag_fam_size = atof(min_tag_fam_size[alt_idx].c_str());
      }

      if (alt_idx < sse_prob_threshold.size() and sse_prob_threshold[alt_idx] != ".") {
        hotspot.params.sse_prob_threshold_override = true;
        hotspot.params.sse_prob_threshold = atof(sse_prob_threshold[alt_idx].c_str());
      }

      // record bad-strand info
      hotspot.params.black_strand = alt_idx < black_list_strand.size() ? black_list_strand[alt_idx][0] : '.';

    }

    if (next_.empty())
      continue;

    break;

  }

}


/*
void HotspotReader::Initialize(const ReferenceReader &ref_reader, const string& hotspot_vcf_filename)
{
  if (hotspot_vcf_filename.empty())
    return;
  ref_reader_ = &ref_reader;
  line_number_ = 0;
  hotspot_vcf_.open(hotspot_vcf_filename.c_str(), ifstream::in);
  if (not hotspot_vcf_.is_open()) {
    cerr << "ERROR: Could not open hotspot file : " << hotspot_vcf_filename << " : " << strerror(errno) << endl;
    exit(1);
  }

  FetchNextVariant();
}


void HotspotReader::FetchNextVariant()
{

  has_more_variants_ = false;
  next_.clear();

  string line;
  vector<string>  vcf_columns;

  while (getline(hotspot_vcf_, line).good()) {
    ++line_number_;

    if (line.empty() or line[0] == '#')
      continue;

    split (line, '\t', vcf_columns);

    if (vcf_columns.size() == 0) {
      cerr << "WARNING: Failed to parse hotspot file line " << line_number_ << endl;
      continue;
    }

    if (vcf_columns.size() < 8) {
      cerr << "WARNING: Failed to parse hotspot file line " << line_number_ << endl;
      continue;
    }

    has_more_variants_ = true;
    break;
  }

  if (not has_more_variants_)
    return;

  string &chr_name  = vcf_columns[0];
  string &pos_str   = vcf_columns[1];
  string &ref       = vcf_columns[3];
  string &alt_list  = vcf_columns[4];
  string &info_list = vcf_columns[7];

  next_chr_ = ref_reader_->chr_idx(chr_name.c_str());
  next_pos_ = atoi(pos_str.c_str()) - 1;

  if (next_chr_ < 0) {
    cerr << "ERROR: invalid chromosome name in hotspot file line " << line_number_ << endl;
    exit(1);
  }

  vector<string>  alts;
  split(alt_list, ',', alts);

  vector<string>  info_entries;
  split(info_list, ';', info_entries);

  vector<string>  min_allele_freq;
  vector<string>  strand_bias;
  vector<string>  min_coverage;
  vector<string>  min_coverage_each_strand;
  vector<string>  min_variant_score;
  vector<string>  data_quality_stringency;

  for (vector<string>::iterator info_entry = info_entries.begin(); info_entry != info_entries.end(); ++info_entry) {
    if (info_entry->compare(0, 16, "min_allele_freq=") == 0)
      split(info_entry->substr(16), ',', min_allele_freq);

    if (info_entry->compare(0, 12, "strand_bias=") == 0)
      split(info_entry->substr(12), ',', strand_bias);

    if (info_entry->compare(0, 13, "min_coverage=") == 0)
      split(info_entry->substr(13), ',', min_coverage);

    if (info_entry->compare(0, 25, "min_coverage_each_strand=") == 0)
      split(info_entry->substr(25), ',', min_coverage_each_strand);

    if (info_entry->compare(0, 18, "min_variant_score=") == 0)
      split(info_entry->substr(18), ',', min_variant_score);

    if (info_entry->compare(0, 24, "data_quality_stringency=") == 0)
      split(info_entry->substr(24), ',', data_quality_stringency);
  }

  next_.reserve(alts.size());

  for (unsigned int alt_idx = 0; alt_idx < alts.size(); ++alt_idx) {
    next_.push_back(HotspotAllele());
    HotspotAllele& hotspot = next_.back();

    hotspot.chr = next_chr_;
    hotspot.pos = next_pos_;
    hotspot.ref = ref;
    hotspot.alt = alts[alt_idx];

    if (alt_idx < min_allele_freq.size() and min_allele_freq[alt_idx] != ".") {
      hotspot.custom_min_allele_freq = true;
      hotspot.min_allele_freq = atof(min_allele_freq[alt_idx].c_str());
    } else
      hotspot.custom_min_allele_freq = false;

    if (alt_idx < strand_bias.size() and strand_bias[alt_idx] != ".") {
      hotspot.custom_strand_bias = true;
      hotspot.strand_bias = atof(strand_bias[alt_idx].c_str());
    } else
      hotspot.custom_strand_bias = false;

    if (alt_idx < min_coverage.size() and min_coverage[alt_idx] != ".") {
      hotspot.custom_min_coverage = true;
      hotspot.min_coverage = atof(min_coverage[alt_idx].c_str());
    } else
      hotspot.custom_min_coverage = false;

    if (alt_idx < min_coverage_each_strand.size() and min_coverage_each_strand[alt_idx] != ".") {
      hotspot.custom_min_coverage_each_strand = true;
      hotspot.min_coverage_each_strand = atof(min_coverage_each_strand[alt_idx].c_str());
    } else
      hotspot.custom_min_coverage_each_strand = false;

    if (alt_idx < min_variant_score.size() and min_variant_score[alt_idx] != ".") {
      hotspot.custom_min_variant_score = true;
      hotspot.min_variant_score = atof(min_variant_score[alt_idx].c_str());
    } else
      hotspot.custom_min_variant_score = false;

    if (alt_idx < data_quality_stringency.size() and data_quality_stringency[alt_idx] != ".") {
      hotspot.custom_data_quality_stringency = true;
      hotspot.data_quality_stringency = atof(data_quality_stringency[alt_idx].c_str());
    } else
      hotspot.custom_data_quality_stringency = false;

  }

}
*/

