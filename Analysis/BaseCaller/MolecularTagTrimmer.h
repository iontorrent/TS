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
  string  GetSuffixTag (string read_group_name) const;

  bool    HasTags(int read_group_idx) const { return read_group_has_tags_.at(read_group_idx); };
  bool    HasTags(string read_group_name) const;
  bool    HaveTags() const { return (num_read_groups_with_tags_>0); };


};



// ---------------------------------------------------------------------
#endif // MOLECULARTAGTRIMMER_H
