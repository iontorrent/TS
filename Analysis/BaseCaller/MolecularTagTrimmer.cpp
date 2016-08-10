/* Copyright (C) 2015 Thermo Fisher Scientific, All Rights Reserved */

//! @file     MolecularTagTrimmer.cpp
//! @ingroup  BaseCaller
//! @brief    MolecularTagTrimmer. Trimming and structural accounting for molecular tags


#include "MolecularTagTrimmer.h"
#include <locale>

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
      command_line_tags_(false), tag_trim_method_(0), tag_filter_method_(0)
{}

// -------------------------------------------------------------------------
//TODO write help function

void MolecularTagTrimmer::PrintHelp(bool tvc_call)
{
  cout << "Molecular tagging options:" << endl;
  // TVC only options
  if (tvc_call)
    cout << "     --min-tag-fam-size      INT        Minimum required size of molecular tag family [3]" << endl;

  // BaseCaller only options
  else{
    cout << "     --prefix-mol-tag        STRING     Structure of prefix molecular tag {ACGTN bases}" << endl;
    cout << "     --suffix-mol-tag        STRING     Structure of suffix molecular tag {ACGTN bases}" << endl;
    cout << "     --tag-trim-method       STRING     Method to trim tags. Options: {strict-trim, sloppy-trim} [sloppy-trim]" << endl;
  }
  // Both
  cout << "     --suppress-mol-tags     BOOL       Ignore tag information [false]" << endl;
  cout << "     --tag-filter-method     STRING     Filter reads based on tags. Options: {need-prefix, need-suffix, need-all} [need-all]" << endl;
  cout << endl;
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
    cout << "  found " << num_read_groups_with_tags_ << " read groups with tags." << endl;
    cout << "  suppress-mol-tags : " << (suppress_mol_tags_ ? "on" : "off") << endl;

    if (not tvc_call) {
      cout << "    tag-trim-method : ";
      switch (tag_trim_method_){
        case kSloppyTrim : cout << "sloppy-trim" << endl; break;
        case kStrictTrim : cout << "strict-trim" << endl; break;
        default : cout << "unknown option" << endl; break;
      }
    }

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
    n_bases_trimmed = StrictTagTrimming(tag_structure_.at(read_group_idx).prefix_mol_tag,
                                        tag_start, seq_length, Tag.prefix_mol_tag);
  }
  else // if (tag_trim_method_ == kSloppyTrim)
  {
    n_bases_trimmed = SloppyTagTrimming(tag_structure_.at(read_group_idx).prefix_mol_tag,
                                        tag_start, seq_length, Tag.prefix_mol_tag);
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
    n_bases_trimmed = StrictTagTrimming(tag_structure_.at(read_group_idx).suffix_mol_tag,
                                        tag_end-tag_length, tag_length, Tag.suffix_mol_tag);
  }
  else // if (tag_trim_method_ == kSloppyTrim)
  {
    n_bases_trimmed = SloppyTagTrimming(tag_structure_.at(read_group_idx).suffix_mol_tag,
                                        tag_end-tag_length, tag_length, Tag.suffix_mol_tag);
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

bool MolecularTagTrimmer::HasTags(string read_group_name) const
{
  bool has_tags = false;
  std::map<string,int>::const_iterator idx_it = read_group_name_to_index_.find(read_group_name);
  if (idx_it != read_group_name_to_index_.end()){
    has_tags = read_group_has_tags_.at(idx_it->second);
  }
  return has_tags;
}





