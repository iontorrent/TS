/* Copyright (C) 2015 Thermo Fisher Scientific, All Rights Reserved */

//! @file     MolecularTagTrimmer.cpp
//! @ingroup  BaseCaller
//! @brief    MolecularTagTrimmer. Trimming and structural accounting for molecular tags


#include "MolecularTagTrimmer.h"
#include <locale>
#include <algorithm>

enum TagFilterOptions {
  kneed_only_prefix_tag,
  kneed_only_suffix_tag,
  kneed_all_tags
};

enum TagTrimMethod {
  kStrictTrim,
  kSloppyTrim
};

// -------------------------------------------------------------------------

MolecularTagTrimmer::MolecularTagTrimmer()
    : num_read_groups_with_tags_(0), suppress_mol_tags_(false),
      command_line_tags_(false), tag_trim_method_(0), tag_filter_method_(0),
	  heal_tag_hp_indel_(true)
{}

// -------------------------------------------------------------------------
//TODO write help function

void MolecularTagTrimmer::PrintHelp(bool tvc_call)
{
  string space1_for_tvc = tvc_call? "           " : "";
  string space2_for_tvc = tvc_call? " " : "";

  cout << "Molecular tagging options:" << endl;
  // Both
  cout << "     --suppress-mol-tags     " << space1_for_tvc << "BOOL"   << space2_for_tvc << "       Ignore tag information [false]" << endl;
  cout << "     --tag-trim-method       " << space1_for_tvc << "STRING" << space2_for_tvc << "     Method to trim tags. Options: {strict-trim, sloppy-trim} [sloppy-trim]" << endl;
  // TVC only options
  if (tvc_call)
    cout << "     --min-tag-fam-size      " << space1_for_tvc << "INT" << space2_for_tvc << "        Minimum required size of molecular tag family [3]" << endl;
  // BaseCaller only options
  else{
    cout << "     --prefix-mol-tag        STRING     Structure of prefix molecular tag {ACGTN bases}" << endl;
    cout << "     --suffix-mol-tag        STRING     Structure of suffix molecular tag {ACGTN bases}" << endl;
    cout << "     --heal-tag-hp-indel     BOOL       Heal hp indel on tags [true]" << endl;
    cout << "     --tag-filter-method     STRING     Filter reads based on tags. Options: {need-prefix, need-suffix, need-all} [need-all]" << endl;
    cout << endl;
  }
}


// -------------------------------------------------------------------------
// Command line parsing

TagTrimmerParameters MolecularTagTrimmer::ReadOpts(OptArgs& opts)
{
  // Reading command line options to set tag structures
  TagTrimmerParameters my_params;

  my_params.min_family_size            = opts.GetFirstInt     ('-', "min-tag-fam-size", 3);
  my_params.suppress_mol_tags          = opts.GetFirstBoolean ('-', "suppress-mol-tags", false);
  //my_params.cl_a_handle                = opts.GetFirstString  ('-', "tag-handle", "");
  //my_params.handle_cutoff              = opts.GetFirstInt     ('-', "handle-cutoff", 2);

  my_params.master_tags.prefix_mol_tag = opts.GetFirstString  ('-', "prefix-mol-tag", "");
  my_params.master_tags.suffix_mol_tag = opts.GetFirstString  ('-', "suffix-mol-tag", "");

  my_params.heal_tag_hp_indel          = opts.GetFirstBoolean ('-', "heal-tag-hp-indel", true);

  ValidateTagString(my_params.master_tags.prefix_mol_tag);
  ValidateTagString(my_params.master_tags.suffix_mol_tag);

  // Overload to disable molecular tagging
  if (my_params.min_family_size == 0)
    my_params.suppress_mol_tags = true;
  else if (my_params.min_family_size < 1) {
    cerr << "MolecularTagTrimmer Error: min-tag-fam-size must be at least 1. " << endl;
    exit(EXIT_FAILURE);
  }

  my_params.command_line_tags = my_params.master_tags.HasTags();

  // Options for read filtering & and trimming method selection
  string trim_method          = opts.GetFirstString  ('-', "tag-trim-method", "sloppy-trim");
  if (trim_method == "sloppy-trim")
    my_params.tag_trim_method = kSloppyTrim;
  else if (trim_method == "strict-trim")
    my_params.tag_trim_method = kStrictTrim;
  else {
    cerr << "MolecularTagTrimmer Error: Unknown tag trimming option " << trim_method << endl;
    exit(EXIT_FAILURE);
  }

  string filter_method        = opts.GetFirstString  ('-', "tag-filter-method", "need-all");
  if (filter_method == "need-all")
    my_params.tag_filter_method = kneed_all_tags;
  else if (filter_method == "need-prefix")
    my_params.tag_filter_method = kneed_only_prefix_tag;
  else if (filter_method == "need-suffix")
    my_params.tag_filter_method = kneed_only_suffix_tag;
  else {
    cerr << "MolecularTagTrimmer Error: Unknown tag filtering option " << filter_method << endl;
    exit(EXIT_FAILURE);
  }
  return my_params;
}

// -------------------------------------------------------------------------

void MolecularTagTrimmer::InitializeFromJson(const TagTrimmerParameters params, Json::Value& read_groups_json, bool trim_barcodes)
{
  // Initialize object with command line parameters
  suppress_mol_tags_ = params.suppress_mol_tags;
  command_line_tags_ = params.command_line_tags;
  tag_trim_method_   = params.tag_trim_method;
  tag_filter_method_ = params.tag_filter_method;
  master_tags_       = params.master_tags;
  heal_tag_hp_indel_ = params.heal_tag_hp_indel;

  if (not trim_barcodes and not suppress_mol_tags_){
    cerr << "MolecularTagTrimmer WARNING: Cannot trim tags because barcode trimming is disabled." << endl;
    suppress_mol_tags_ = true;
  }

  // Initialize read groups from json
  read_group_index_to_name_ = read_groups_json.getMemberNames();
  read_group_name_to_index_.clear();
  num_read_groups_with_tags_ = 0;
  string read_group_name;

  read_group_has_tags_.assign(read_group_index_to_name_.size(), false);
  tag_structure_.resize(read_group_index_to_name_.size());
  heal_prefix_.assign(read_group_index_to_name_.size(), HealTagHpIndel(false));
  heal_suffix_.assign(read_group_index_to_name_.size(), HealTagHpIndel(true));

  for (int rg = 0; rg < (int)read_group_index_to_name_.size(); ++rg) {

    read_group_name = read_group_index_to_name_[rg];
    read_group_name_to_index_[read_group_name] = rg;
    //cout << "Checking read group " << read_group_name << " for tags: " << endl; // XXX

    // Don't load any tag info if we need to supress tags
    if (suppress_mol_tags_){
      tag_structure_.at(rg).Clear();
      continue;
    }

    // If we had tags explicitly defined by command line, they overwrite auto loaded information
    if (command_line_tags_){
      // check if this is the no-match read group and make sure the no-match group never has tags associated with it
      if (read_group_index_to_name_[rg].find("nomatch") == std::string::npos)
        tag_structure_.at(rg) = master_tags_;
      else
        tag_structure_.at(rg).Clear();

      // Update read groups json so that datasets_basecaller.json reflects the run parameters
      read_groups_json[read_group_name]["mol_tag_prefix"] =  tag_structure_.at(rg).prefix_mol_tag;
      read_groups_json[read_group_name]["mol_tag_suffix"] =  tag_structure_.at(rg).suffix_mol_tag;
    }
    else{
      // Read tag info from datasets json
      tag_structure_.at(rg).prefix_mol_tag = read_groups_json[read_group_name].get("mol_tag_prefix","").asString();
      tag_structure_.at(rg).suffix_mol_tag = read_groups_json[read_group_name].get("mol_tag_suffix","").asString();
    }

    // Validate input at end
    ValidateTagString(tag_structure_.at(rg).prefix_mol_tag);
    ValidateTagString(tag_structure_.at(rg).suffix_mol_tag);

    // Set heal_prefix_ and heal_sufffix_
    heal_prefix_.at(rg).SetBlockStructure(tag_structure_.at(rg).prefix_mol_tag);
    heal_suffix_.at(rg).SetBlockStructure(tag_structure_.at(rg).suffix_mol_tag);

    read_group_has_tags_.at(rg) = tag_structure_.at(rg).HasTags();
    if (read_group_has_tags_.at(rg))
      ++num_read_groups_with_tags_;
  }

  PrintOptionValues(false);
}

// -------------------------------------------------------------------------

void    MolecularTagTrimmer::InitializeFromSamHeader(const TagTrimmerParameters params, const BamTools::SamHeader &samHeader)
{
  // Initialize object with command line parameters
  suppress_mol_tags_ = params.suppress_mol_tags;
  command_line_tags_ = params.command_line_tags;
  tag_trim_method_   = params.tag_trim_method;
  tag_filter_method_ = params.tag_filter_method;
  master_tags_       = params.master_tags;

  int num_read_groups = 0;
  num_read_groups_with_tags_ = 0;
  read_group_index_to_name_.clear();
  read_group_name_to_index_.clear();
  read_group_has_tags_.clear();
  tag_structure_.clear();
  heal_prefix_.clear();
  heal_suffix_.clear();

  for (BamTools::SamReadGroupConstIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr) {

    if (itr->ID.empty()){
      cerr << "MolecularTagTrimmer ERROR: BAM file has a read group without ID string." << endl;
      exit(EXIT_FAILURE);
    }
    read_group_index_to_name_.push_back(itr->ID);
    read_group_name_to_index_.insert(pair<string,int>(itr->ID, num_read_groups));

    // Iterate through custom read group tags to find tag information
    MolTag my_tags;
    for (vector<BamTools::CustomHeaderTag>::const_iterator tag_itr = itr->CustomTags.begin(); tag_itr != itr->CustomTags.end(); ++tag_itr){
      if (tag_itr->TagName == "zt")
        my_tags.prefix_mol_tag = tag_itr->TagValue;
      else if (tag_itr->TagName == "yt")
        my_tags.suffix_mol_tag = tag_itr->TagValue;
    }

    // Treat sample as untagged if desired
    if (suppress_mol_tags_)
      my_tags.Clear();
    // Only allow command line tag trimming if there aren't already trimmed tags in the BAM
    else if (command_line_tags_){
      if (my_tags.HasTags()){
        cerr << "MolecularTagTrimmer ERROR: Cannot apply command line tags - BAM file already contains trimmed tags for read group " << itr->ID << endl;
        exit(EXIT_FAILURE);
      }
      else
        my_tags = master_tags_;
    }

    tag_structure_.push_back(my_tags);
    read_group_has_tags_.push_back(my_tags.HasTags());
    heal_prefix_.push_back(HealTagHpIndel(false));
    heal_suffix_.push_back(HealTagHpIndel(true));
    heal_prefix_.back().SetBlockStructure(my_tags.prefix_mol_tag);
    heal_suffix_.back().SetBlockStructure(my_tags.suffix_mol_tag);

    if (my_tags.HasTags())
      ++num_read_groups_with_tags_;

    ++num_read_groups;
  }

  PrintOptionValues(true);
}

// -------------------------------------------------------------------------

void MolecularTagTrimmer::PrintOptionValues(bool tvc_call)
{
  // Verbose output XXX do not say anything if there are no tags
  if (num_read_groups_with_tags_ > 0) {
    cout << "MolecularTagTrimmer settings:" << endl;
    cout << "    found " << num_read_groups_with_tags_ << " read groups with tags." << endl;
    cout << "    suppress-mol-tags : " << (suppress_mol_tags_ ? "on" : "off") << endl;

    if (not tvc_call) {
      cout << "    heal-tag-hp-indel : " << (heal_tag_hp_indel_? "on": "off") << endl;
      cout << "      tag-trim-method : ";
      switch (tag_trim_method_){
        case kSloppyTrim : cout << "sloppy-trim" << endl; break;
        case kStrictTrim : cout << "strict-trim" << endl; break;
        default : cout << "unknown option" << endl; break;
      }
    }

    cout << "    tag-filter-method : ";
    switch (tag_filter_method_){
      case kneed_all_tags : cout << "need-all" << endl; break;
      case kneed_only_prefix_tag : cout << "need-prefix" << endl; break;
      case kneed_only_suffix_tag : cout << "need-suffix" << endl; break;
      default : cout << "unknown option" << endl; break;
    }

    // Only give info about command line tags if they were specified
    if (command_line_tags_){
      cout << "     prefix-mol-tag : " << master_tags_.prefix_mol_tag << endl;
      cout << "     suffix-mol-tag : " << master_tags_.suffix_mol_tag << endl;
    }
    cout << endl;
  }
}

// -------------------------------------------------------------------------
// Check for non-ACGTN characters

void MolecularTagTrimmer::ValidateTagString(string& tag_str)
{
  if (tag_str.empty())
    return;

  // Make sure tag is in all upper case letters and count number of N's
  unsigned int n_count = 0;
  for (std::string::iterator it=tag_str.begin(); it!=tag_str.end(); ++it) {
     *it = toupper(*it);
     if (*it == 'N')
       ++n_count;
  }

  // Throw an error if the tag does not contain at least one N
  if (n_count==0){
    cerr << "MolecularTagtrimmer ERROR: tag " << tag_str << " does not contain any N! " << endl;
    exit(EXIT_FAILURE);
  }

  // And throw an error if there are non-ACGTN characters in the string
  if (std::string::npos != tag_str.find_first_not_of("ACGTN")){
    cerr << "MolecularTagtrimmer ERROR: tag " << tag_str << " contains non-ACGTN characters! " << endl;
    exit(EXIT_FAILURE);
  }
}

// -------------------------------------------------------------------------
// Trimming function for prefix tag

int  MolecularTagTrimmer::TrimPrefixTag(const int read_group_idx,
                                        const char* tag_start,
                                        int seq_length,
                                        MolTag& Tag) const
{
  Tag.Clear();
  // Check if this read group has a prefix tag to potentially trim
  if (tag_structure_.at(read_group_idx).prefix_mol_tag.empty())
    return 0;

  // Distribute read to trimming methods
  int n_bases_trimmed = 0;
  if (tag_start == NULL) {
    n_bases_trimmed = -1;
  }
  else if (tag_trim_method_ == kStrictTrim){
	if(heal_tag_hp_indel_){
		string tag_identified;
		bool is_heal_or_match = heal_prefix_.at(read_group_idx).TagTrimming(tag_start, seq_length, tag_identified, n_bases_trimmed);
	    if(is_heal_or_match){
	    	Tag.prefix_mol_tag += tag_identified; // I follow the code in MolecularTagTrimmer::StrictTagTrimming where it does nothing on Tag.prefix_mol_tag if not is_heal_or_match
	    }
	    else{
	    	n_bases_trimmed = -1;
	    }
	}
	else{
      n_bases_trimmed = StrictTagTrimming(tag_structure_.at(read_group_idx).prefix_mol_tag,
                                          tag_start, seq_length, Tag.prefix_mol_tag);
	}
  }
  else // if (tag_trim_method_ == kSloppyTrim)
  {
	if(heal_tag_hp_indel_){
		string tag_identified;
		heal_prefix_.at(read_group_idx).TagTrimming(tag_start, seq_length, tag_identified, n_bases_trimmed);
		Tag.prefix_mol_tag += tag_identified;
	}
	else{
      n_bases_trimmed = SloppyTagTrimming(tag_structure_.at(read_group_idx).prefix_mol_tag,
                                        tag_start, seq_length, Tag.prefix_mol_tag);
	}
  }
  // TODO add a more clever method

  if (n_bases_trimmed < 0 and (tag_filter_method_ == kneed_only_suffix_tag))
    n_bases_trimmed = 0;

  return n_bases_trimmed;
}

// -------------------------------------------------------------------------
// Trimming function for prefix tag

int  MolecularTagTrimmer::TrimSuffixTag(const int read_group_idx,
                                        const char* tag_end,
                                        int template_length,
                                        MolTag& Tag) const
{
  // Check if this read group has a prefix tag to potentially trim
  if (tag_structure_.at(read_group_idx).suffix_mol_tag.empty())
    return 0;

  // Distribute read to trimming methods
  int n_bases_trimmed = 0;
  int tag_length = tag_structure_.at(read_group_idx).suffix_mol_tag.length();

  if (tag_end == NULL or template_length < tag_length) {
    n_bases_trimmed = -1;
  }
  else if (tag_trim_method_ == kStrictTrim){
	if (heal_tag_hp_indel_){
		string tag_identified;
		bool is_heal_or_match = heal_suffix_.at(read_group_idx).TagTrimming(tag_end - template_length, template_length, tag_identified, n_bases_trimmed);
	    if(is_heal_or_match){
	    	Tag.suffix_mol_tag += tag_identified;  // I follow the code in MolecularTagTrimmer::StrictTagTrimming where it does nothing on Tag.prefix_mol_tag if not is_heal_or_match
	    }
	    else{
	    	n_bases_trimmed = -1;
	    }
	}
	else{
      n_bases_trimmed = StrictTagTrimming(tag_structure_.at(read_group_idx).suffix_mol_tag,
                                          tag_end-tag_length, tag_length, Tag.suffix_mol_tag);
	}
  }
  else // if (tag_trim_method_ == kSloppyTrim)
  {
	if (heal_tag_hp_indel_){
		string tag_identified;
		heal_suffix_.at(read_group_idx).TagTrimming(tag_end - template_length, template_length, tag_identified, n_bases_trimmed);
		Tag.suffix_mol_tag += tag_identified;
	}
	else{
      n_bases_trimmed = SloppyTagTrimming(tag_structure_.at(read_group_idx).suffix_mol_tag,
                                          tag_end-tag_length, tag_length, Tag.suffix_mol_tag);
	}
  }
  // TODO add a more clever method

  if (n_bases_trimmed < 0 and (tag_filter_method_ == kneed_only_prefix_tag))
    n_bases_trimmed = 0;

  return n_bases_trimmed;
}


// -------------------------------------------------------------------------
// Enforces a strict match between the structure of the tag and the called sequence

int MolecularTagTrimmer::StrictTagTrimming(const string tag_structure,
                       const char* base,
                       int seq_length,
                       string& trimmed_tag) const
{
  bool match = true;
  int tag_length = tag_structure.length();

  if (base == NULL or seq_length < tag_length)
    return -1;

  const char* comp_base = base;
  for (string::const_iterator tag_it=tag_structure.begin(); match and tag_it!=tag_structure.end(); ++tag_it){
    if (*tag_it != 'N' and *tag_it != *comp_base)
      match = false;
    ++comp_base;
  }

  if (match){
    trimmed_tag.append(base, tag_length);
    return tag_length;
  }
  else
    return -1;
}

// -------------------------------------------------------------------------
// The super-sloppy method of taking any tag_length bases as tag

int MolecularTagTrimmer::SloppyTagTrimming(const string tag_structure,
                       const char* base,
                       int seq_length,
                       string& trimmed_tag) const
{
  int tag_length = tag_structure.length();

  if (base == NULL or seq_length < tag_length)
    return -1;

  trimmed_tag.append(base, tag_length);
  return tag_length;
}


// -------------------------------------------------------------------------
// This function for now only reads tags from BAM files and does not trim anything

bool  MolecularTagTrimmer::GetTagsFromBamAlignment(const BamTools::BamAlignment& alignment, MolTag& Tags)
{
  // Don't bother if there is nothing to look at
  if (num_read_groups_with_tags_ == 0){
    Tags.Clear();
    return true;
  }

  // Load Tags from Bam Alignment
  if (not alignment.GetTag("ZT", Tags.prefix_mol_tag))
    Tags.prefix_mol_tag.clear();

  if (not alignment.GetTag("YT", Tags.suffix_mol_tag))
    Tags.suffix_mol_tag.clear();

  // Check if this read should have tags associated with it
  string read_group_name;
  if (not alignment.GetTag("RG",read_group_name))
    return false;

  std::map<string,int>::const_iterator idx_it = read_group_name_to_index_.find(read_group_name);
  if (idx_it == read_group_name_to_index_.end())
    return false;

  if (NeedPrefixTag(idx_it->second)) {
    if (Tags.prefix_mol_tag.empty())
      return false;
  }
  else
    Tags.prefix_mol_tag.clear();

  if (NeedSuffixTag(idx_it->second)) {
    if (Tags.suffix_mol_tag.empty())
      return false;
  }
  else
    Tags.suffix_mol_tag.clear();

  // We don't allow the joint analysis of tagged and untagged samples at the same time
  if (not Tags.HasTags())
    return false;

  return true;
}

// -------------------------------------------------------------------------

bool MolecularTagTrimmer::NeedPrefixTag(int read_group_index) const
{
  return (not tag_structure_.at(read_group_index).prefix_mol_tag.empty() and tag_filter_method_ != kneed_only_suffix_tag);
}

// -------------------------------------------------------------------------

bool  MolecularTagTrimmer::NeedSuffixTag(int read_group_index) const
{
  return (not tag_structure_.at(read_group_index).suffix_mol_tag.empty() and tag_filter_method_ != kneed_only_prefix_tag);
}

// -------------------------------------------------------------------------

bool  MolecularTagTrimmer::NeedAdapter(string read_group_name) const
{
  std::map<string,int>::const_iterator idx_it = read_group_name_to_index_.find(read_group_name);
  if (idx_it != read_group_name_to_index_.end())
    return NeedSuffixTag(idx_it->second);
  else
    return false;
}

// -------------------------------------------------------------------------

MolTag MolecularTagTrimmer::GetReadGroupTags (string read_group_name) const
{
  MolTag tag;
  std::map<string,int>::const_iterator idx_it = read_group_name_to_index_.find(read_group_name);
  if (idx_it != read_group_name_to_index_.end()){
    tag = tag_structure_.at(idx_it->second);
  }
  return tag;
}

// -------------------------------------------------------------------------

string MolecularTagTrimmer::GetPrefixTag (string read_group_name) const
{
  string tag;
  std::map<string,int>::const_iterator idx_it = read_group_name_to_index_.find(read_group_name);
  if (idx_it != read_group_name_to_index_.end()){
    tag = tag_structure_.at(idx_it->second).prefix_mol_tag;
  }
  return tag;
}

// -------------------------------------------------------------------------

string MolecularTagTrimmer::GetPrefixTag (int read_group_idx) const
{
  string tag = tag_structure_.at(read_group_idx).prefix_mol_tag;
  return tag;
}

// -------------------------------------------------------------------------

string MolecularTagTrimmer::GetSuffixTag (string read_group_name) const
{
  string tag;
  std::map<string,int>::const_iterator idx_it = read_group_name_to_index_.find(read_group_name);
  if (idx_it != read_group_name_to_index_.end()){
    tag = tag_structure_.at(idx_it->second).suffix_mol_tag;
  }
  return tag;
}

// -------------------------------------------------------------------------

string MolecularTagTrimmer::GetSuffixTag (int read_group_idx) const
{
  string tag = tag_structure_.at(read_group_idx).suffix_mol_tag;
  return tag;
}

// -------------------------------------------------------------------------

bool MolecularTagTrimmer::HasTags(string read_group_name) const
{
  bool has_tags = false;
  std::map<string,int>::const_iterator idx_it = read_group_name_to_index_.find(read_group_name);
  if (idx_it != read_group_name_to_index_.end()){
    has_tags = read_group_has_tags_.at(idx_it->second);
  }
  return has_tags;
}

// -------------------------------------------------------------------------

// I split tag_format into several blocks
// A block is assumed to start with random nucs and then follow by the flag nucs.
// Example 1: (prefix) tag_structure = "NNNACTNNNTGN", block_structure_ = { ("NNN", "ACT"), ("NNN", "TG"), ("N", "") }
// Example 2: (prefix) tag_structure = "TACNNN", block_structure_ = { ("", "TAC"), ("NNN", "") }
void HealTagHpIndel::SetBlockStructure(string tag_structure)
{
    if (is_heal_suffix_){
        reverse(tag_structure.begin(), tag_structure.end());
    }
    block_structure_.clear();
    block_size_.clear();

    for (string::iterator tag_it = tag_structure.begin(); tag_it != tag_structure.end(); ++tag_it){
        bool is_random_nuc = (*tag_it == 'N' or *tag_it == 'n');
    	if (block_structure_.empty()){
            block_structure_.push_back(pair<string, string>(string(""), string("")));
        }
        else if (is_random_nuc and block_structure_.back().second.size() > 0){
            block_structure_.push_back(pair<string, string>(string(""), string("")));
        }

        if (is_random_nuc){
            block_structure_.back().first += *tag_it;
        }
        else{
            block_structure_.back().second += *tag_it;
        }
    }
    perfect_tag_len_ = (unsigned int) tag_structure.size();
    max_tag_len_ = perfect_tag_len_ + block_structure_.size() * kMaxInsAllowed_; // allow max_ins_allowed_ per block

    block_size_.assign(block_structure_.size(), 0);
    for (unsigned int i_block = 0; i_block < block_size_.size(); ++i_block){
        block_size_[i_block] = block_structure_[i_block].first.size() + block_structure_[i_block].second.size();
    }
}

// -------------------------------------------------------------------------

// Find the maximum hp len of base_seq, also take anchor_bases into account.
// Example:
// Inputs: base_seq = "TTTAGGGG", anchor_bases = "CCGTT"
// Outputs: max_hp_len_in_base_seq = 3, max_hp_len_from_anchor = 2, max_hp_base_idx = 0, is_max_hp_unique = true
bool FindMaxHp(const string& base_seq, unsigned int& max_hp_len_in_base_seq, unsigned int& max_hp_len_from_anchor, unsigned int& max_hp_base_idx, const string& anchor_bases = "")
{
    bool is_max_hp_unique = true;
    unsigned int current_hp_len = 1; // base_seq[0] has hp len at least 1
    int max_hp_base_idx_tmp = 0;
    max_hp_base_idx = 0;
    max_hp_len_in_base_seq = 0;
    max_hp_len_from_anchor = 0;

    if (base_seq.empty()){
        is_max_hp_unique = false;
        return is_max_hp_unique;
    }

    // Count the hp len of base_seq[0] contributed from anchor_bases
    for (string::const_iterator anchor_it = anchor_bases.end() - 1; anchor_it != anchor_bases.begin() - 1; --anchor_it){
        if (*anchor_it == base_seq[0]){
            ++max_hp_len_from_anchor;
        }
        else{
            break;
        }
    }

    // hp len for base_seq[0]
    current_hp_len += max_hp_len_from_anchor;
    max_hp_len_in_base_seq = current_hp_len;

    if (base_seq.size() == 1){
        return is_max_hp_unique;
    }

    for (unsigned int i = 1; i < base_seq.size(); ++i){
        if (base_seq[i] == base_seq[i-1]){
            ++current_hp_len;
        }
        else{
            if (current_hp_len > max_hp_len_in_base_seq){
            	max_hp_len_in_base_seq = current_hp_len;
            	// max_hp_base_idx_tmp is the starting index of the max_hp_len bases in base_seq
            	// max_hp_base_idx_tmp could be negative due to anchor_bases
            	max_hp_base_idx_tmp = i - current_hp_len;
                is_max_hp_unique = true;
            }
            else if (current_hp_len == max_hp_len_in_base_seq){
                is_max_hp_unique = false;
            }
            current_hp_len = 1;
        }
    }

    // Check the hp len for the last base of base_seq
    if (current_hp_len > max_hp_len_in_base_seq){
    	max_hp_len_in_base_seq = current_hp_len;
    	max_hp_base_idx_tmp = base_seq.size()  - current_hp_len;
        is_max_hp_unique = true;
    }
    else if (current_hp_len == max_hp_len_in_base_seq){
        is_max_hp_unique = false;
    }

    if (max_hp_base_idx_tmp <= 0){
    	// The max_hp_len occurs at base_seq[0]
    	max_hp_base_idx = 0;
    	max_hp_len_in_base_seq -= max_hp_len_from_anchor;
    }
    else{
    	max_hp_base_idx = (unsigned int) max_hp_base_idx_tmp;
    	// Set max_hp_len_from_anchor = 0 because anchor_bases does not contribute any hp length to max_hp_len
    	max_hp_len_from_anchor = 0;
    }
    return is_max_hp_unique;
}

// -------------------------------------------------------------------------

// Now I try to heal the hp indel for base_sub_seq
// base_sub_seq is the subsequence of the original bases, and base_sub_seq[0] is the starting base of the block.
// tag_in_block is the output identified tag in the block
// indel_len is the hp length healed in the block.
// anchor_bases is just for finding the hp len of base_sub_seq[0]
// Example 1: block_structure[block_index] = ("NNN", "TAC"), base_sub_seq = "CCCCCTAC", then I output tag_in_block = "CCCTAC", indel_len = 2 (healed 2-mer insertion)
// Example 2: block_structure[block_index] = ("NNN", "TAC"), base_sub_seq = "TTTAC", then I output tag_in_block = "TTTTAC", indel_len = -1 (healed 1-mer deletion)
//@TODO: Ugly code because of using too many string operations. Should be optimized!
bool HealTagHpIndel::HealHpIndelOneBlock_(const string& base_sub_seq, unsigned int block_index, string& tag_in_block, int& indel_len, const string &anchor_bases = "") const
{
    unsigned int max_hp_base_idx = 0;
    unsigned int max_hp_len = 0;
    unsigned int max_hp_len_from_anchor = 0;
    bool is_match_or_healed = false;
    string try_flag;

    // Default output
	indel_len = 0;
    if (base_sub_seq.size() < block_size_[block_index]){
    	// There is definitely ins. Maybe we will heal the ins.
    	tag_in_block.clear();
    }
    else{
        tag_in_block = base_sub_seq.substr(0, block_size_[block_index]);  // Actually I am doing sloppy tag trimming here

        // First try the case where there is no indel
        try_flag = tag_in_block.substr(block_structure_[block_index].first.size(), block_structure_[block_index].second.size()); // tag w/o indel offset
        if (try_flag == block_structure_[block_index].second){
        	// base_sub_seq matches the block structure. No heal needed.
        	is_match_or_healed = true;
        	return is_match_or_healed;
        }
    }

    // Here I primarily focus on healing the hp indel in the random nucs.
    // Find the max hp len of the bases in the block. I will adjust the len of the max hp to see if the flag can match the tag structure.
    bool is_max_hp_unique = FindMaxHp(tag_in_block, max_hp_len, max_hp_len_from_anchor, max_hp_base_idx, anchor_bases);
    char max_hp_nuc = base_sub_seq[max_hp_base_idx];

    if ((not is_max_hp_unique) or (max_hp_len + max_hp_len_from_anchor< kMinHpLenForHeal_)){
        return is_match_or_healed;
    }

    int min_try_hp_len = max(0, (int) kMinHealedHpLen_ - (int) max_hp_len_from_anchor);
    string pre_max_hp_bases = base_sub_seq.substr(0, max_hp_base_idx); // bases before the max hp flow
    string post_max_hp_bases = base_sub_seq.substr(max_hp_base_idx + max_hp_len); // bases after the max hp flow

    // Adjust the length of the maximum hp to see if the flag can match the tag structure
    for (int try_indel_len = - (int) kMaxDelAllowed_; try_indel_len <= (int) kMaxInsAllowed_; ++try_indel_len){
        int try_hp_len = (int) max_hp_len - try_indel_len;
    	if ((try_indel_len == 0)
    			or (try_hp_len < min_try_hp_len)
				or (base_sub_seq.size() < block_size_[block_index] + (unsigned int) try_indel_len)){
            continue;
        }
        string try_hp(try_hp_len, max_hp_nuc);
        string try_bases = pre_max_hp_bases + try_hp + post_max_hp_bases;

        try_flag = try_bases.substr(block_structure_[block_index].first.size(), block_structure_[block_index].second.size());
        if (try_flag == block_structure_[block_index].second){
            indel_len = try_indel_len;
            tag_in_block = try_bases.substr(0, block_size_[block_index]);
            is_match_or_healed = true;
            break;
        }
    }

    return is_match_or_healed;
}

// -------------------------------------------------------------------------

bool HealTagHpIndel::TagTrimming(const char* base, int seq_length, string& trimmed_tag, int& tag_len) const
{
    int safe_max_tag_len = min((int) max_tag_len_, seq_length);
    bool is_healed_or_match = false;

    if (base == NULL or (unsigned int) seq_length < perfect_tag_len_){
        // The base sequence is shorter than the format_len. I'm not going to do anything.
        tag_len = -1;
        is_healed_or_match = false;
        return is_healed_or_match;
    }
    string safe_max_tag_seq;  // I will trim the tag from safe_max_tag_seq.
    if (is_heal_suffix_){
    	safe_max_tag_seq = string(base + (seq_length - safe_max_tag_len), safe_max_tag_len);
    }
    else{
    	safe_max_tag_seq = string(string(base, safe_max_tag_len));
    }
	is_healed_or_match = TagTrimming(safe_max_tag_seq, trimmed_tag, tag_len);
    return is_healed_or_match;
}

// -------------------------------------------------------------------------
// Input:
// base_seq is the base sequence to be trimned.
// Outputs:
// trimmed_tag is the tag that I identified where hp indels are healed if applicable.
// tag_len is the length of the base that I need to trim from base_seq.
// is_healed_or_match indicates whether trimmed_tag matches the tag structure or not.
// If is_healed_or_match = false, the block of trimmed_tag that doesn't match the tag structure is obtained by the same approach as sloppy tag trimmer.
// Example for prefix tag trimming:
// block_structure_ = {("NNN", "TAC"), }. If base_seq = "TTTTTAC", then trimmed_tag = "TTT", tag_len = 4, return true.
bool HealTagHpIndel::TagTrimming(string base_seq, string& trimmed_tag, int& tag_len) const
{
    bool is_healed_or_match = true;

    if (base_seq.size() < perfect_tag_len_){
        // The base sequence is shorter than the ideal_tag_len_. I'm not going to do anything.
        is_healed_or_match = false;
        tag_len = -1;
        return is_healed_or_match;
    }

    if (is_heal_suffix_){
        reverse(base_seq.begin(), base_seq.end());
    }

    trimmed_tag.clear();
    tag_len = 0;

    for (unsigned int i_block = 0; i_block < block_structure_.size(); ++i_block){
        unsigned int block_start = tag_len;
        unsigned int expand_block_size = block_size_[i_block] + kMaxInsAllowed_;
        int indel_len = 0;
        string tag_in_block;

        if (block_start + expand_block_size > base_seq.size()){
            expand_block_size = base_seq.size() - block_start;
        }

        is_healed_or_match &= HealHpIndelOneBlock_(base_seq.substr(block_start, expand_block_size), i_block, tag_in_block, indel_len, trimmed_tag);

        // Should I heal the rest of the blocks if one block is out of sync?
        // Example: block_format_ = {("NNN", "ACT"), ("NNN", "TGA")}, base_seq = "CCTA" + "ACT" + "ATT" + "TGA"
        // It fails to heal the first block since the max hp is not unique.
        // If I try to heal the second block, trimmed_tag would be "CCTA" + "ACT" + "AT" + "TGA".
        // Pons: better trimming result (trim 13 bases)
        // Cons: wrong tag identification
        // If not, trimmed_tag would be "CCTA" + "ACT" + "ATT" + "TG", which is still a wrong tag, but we only trim 12 bases.
        // Since both tags are incorrect, I choose to continue to heal hp in the second block.
        // A sophisticated but more complex approach to solve the problem is probably doing alignment.

        trimmed_tag += tag_in_block;
        tag_len += ((int) tag_in_block.size() + indel_len);
    }

    if (tag_len < (int) perfect_tag_len_){
    	trimmed_tag = base_seq.substr(0, perfect_tag_len_);
    	tag_len = perfect_tag_len_;
    	is_healed_or_match = false;
    }

    if (is_heal_suffix_){
        reverse(trimmed_tag.begin(), trimmed_tag.end());
    }

    return is_healed_or_match;
}


