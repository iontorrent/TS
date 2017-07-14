/* Copyright (C) 2015 Thermo Fisher Scientific, All Rights Reserved */

//! @file     MolecularTagTrimmer.h
//! @ingroup  BaseCaller
//! @brief    MolecularTagTrimmer. Trimming and structural accounting for molecular tags

#ifndef MOLECULARTAGTRIMMER_H
#define MOLECULARTAGTRIMMER_H

#include "OptArgs.h"

#include <string>
#include <vector>
#include <map>
#include "json/json.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"

using namespace std;

// ---------------------------------------------------------------------
// The absolute bare-bone minimum structure to make tags work

struct MolTag
{
  string    prefix_mol_tag;
  string    suffix_mol_tag;
  //bool      tags_are_trimmed;

  //MolTag() : tags_are_trimmed(false) {};

  void Clear()
  {
    prefix_mol_tag.clear();
    suffix_mol_tag.clear();
  }

  bool HasTags()
  {
    return (prefix_mol_tag.length()>0 or
            suffix_mol_tag.length()>0);
  }
};

// ---------------------------------------------------------------------
// To make the tag trimmer work with BaseCaller as well as TVC setup

struct TagTrimmerParameters
{
  bool    suppress_mol_tags;    //! Ignore mol tag familiy information
  bool    command_line_tags;    //! Tas tag structures been specified vis command line?
  int     tag_trim_method;      //! Tag trimming method
  int     tag_filter_method;    //! Tag filtering method
  MolTag  master_tags;          //! Tags provided via command line
  int     min_family_size;      //! Minimum size of a functional familiy
  string  cl_a_handle;          //! Command line specified a-handle
  int     handle_cutoff;        //! Cutoff to match an a-handle in flow alignment
  bool    heal_tag_hp_indel;    //! Heal the hp indel when doing tag trimming
};

// ---------------------------------------------------------------------

// For healing the homopolymer indel on tags
class HealTagHpIndel
{
private:
	// tag_structure related variables
    vector< pair<string, string> > block_structure_; // block_structure_[i].first = random nucs, block_structure_[i].second = flag nucs
    vector<unsigned int> block_size_;                // number of (random nucs + flag nucs) in each block
    bool is_heal_suffix_;                            // Am I trying to heal a suffix tag?
    unsigned int perfect_tag_len_ = 0;               // length of the tag in the prefect case (i.e. not indel)
    unsigned int max_tag_len_ = 0;                   // maximum tag length to be trimmed

    // Parameters for HealTagHpIndel
    const static unsigned int kMinHpLenForHeal_ = 2;  // I attempt to heal a hp only if its length >= this value.
    const static unsigned int kMinHealedHpLen_ = 3;   // All healed hp should have length >= this value.
    const static unsigned int kMaxDelAllowed_ = 2;    // Allow the base sequence to have at most max_del_allowed-mer hp del per block. Declare as unsigned int to avoid confusion.
    const static unsigned int kMaxInsAllowed_ = 2;    // Allow the base sequence to have at most max_ins_allowed-mer hp ins per block. Declare as unsigned int to avoid confusion.

    bool HealHpIndelOneBlock_(const string& base_in_block, unsigned int block_index, string& tag_in_block, int& indel_offset, const string &anchor_base) const;

public:
    HealTagHpIndel(bool trim_suffix) {is_heal_suffix_ = trim_suffix; };
    void SetBlockStructure(string tag_format);
    bool TagTrimming(string base_seq, string& trimmed_tag, int& tag_len) const; // Trim the tag with a string input
    bool TagTrimming(const char* base, int seq_length, string& trimmed_tag, int& tag_len) const;  // Trim the tag with a char* input
};

// ---------------------------------------------------------------------
// Trimming and structural accounting for molecular tags

class MolecularTagTrimmer
{
private:

  // Only trims tags that perfectly match the expected structure
  int  StrictTagTrimming(const string tag_structure,
                         const char* base,
                         int seq_length,
                         string& trimmed_tag) const;

  // Trims the expected length of the tag, regardless of content
  int  SloppyTagTrimming(const string tag_structure,
                         const char* base,
                         int seq_length,
                         string& trimmed_tag) const;

  // Converts a tag string to all upper case and check for non ACGTN-characters
  static void ValidateTagString(string& tag_str);

  void PrintOptionValues(bool tvc_call);

  map<string,int>           read_group_name_to_index_;
  vector<string>            read_group_index_to_name_;
  vector<string>            tag_a_handle_;
  vector<MolTag>            tag_structure_;
  vector<bool>              read_group_has_tags_;
  int                       num_read_groups_with_tags_;

  bool                      suppress_mol_tags_;          // Do not search for por trim tags
  bool                      command_line_tags_;          // Were tags specified per command line?
  MolTag                    master_tags_;                // Command line overwrite for auto load

  int                       tag_trim_method_;            // method to do tag trimming
  int                       tag_filter_method_;          // method to do tag filtering

  bool                      heal_tag_hp_indel_;          // Do I want to heal hp indel on tags?
  vector<HealTagHpIndel>    heal_prefix_;                // use me to heal the prefix tag for the read groups
  vector<HealTagHpIndel>    heal_suffix_;                // use me to heal the suffix tag for the read groups


public:

  MolecularTagTrimmer();

  static void PrintHelp(bool tvc_call);

  static TagTrimmerParameters    ReadOpts(OptArgs& opts);

  void    InitializeFromJson (const TagTrimmerParameters params, Json::Value& read_groups_json, bool trim_barcodes);

  void    InitializeFromSamHeader(const TagTrimmerParameters params, const BamTools::SamHeader &samHeader);

  bool    GetTagsFromBamAlignment(const BamTools::BamAlignment& alignment, MolTag& Tags);

  // Prefix tag trimming function
  int     TrimPrefixTag(const int read_group_idx,
                        const char* tag_start,
                        int seq_length,
                        MolTag& Tag) const;

  // Suffix trimming function
  int     TrimSuffixTag(const int read_group_idx,
                        const char* tag_end,
                        int template_length,
                        MolTag& Tag) const;

  bool    NeedPrefixTag(int read_group_index) const;
  bool    NeedSuffixTag(int read_group_index) const;
  bool    NeedAdapter(string read_group_name) const;

  int     NumReadGroups() const { return read_group_index_to_name_.size(); };
  int     NumTaggedReadGroups() const { return num_read_groups_with_tags_; };

  MolTag  GetReadGroupTags (string read_group_name) const;
  string  GetPrefixTag (string read_group_name) const;
  string  GetPrefixTag (int read_group_idx) const;
  string  GetSuffixTag (string read_group_name) const;
  string  GetSuffixTag (int read_group_idx) const;

  bool    HasTags(int read_group_idx) const { return read_group_has_tags_.at(read_group_idx); };
  bool    HasTags(string read_group_name) const;
  bool    HaveTags() const { return (num_read_groups_with_tags_>0); };

  int     GetTagTrimMethod() const {return tag_trim_method_; };
};


// ---------------------------------------------------------------------

#endif // MOLECULARTAGTRIMMER_H
