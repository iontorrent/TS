/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HotspotReader.h
//! @ingroup  VariantCaller
//! @brief    Customized hotspot VCF parser


#ifndef HOTSPOTREADER_H
#define HOTSPOTREADER_H

#include <string>
#include <vector>
#include <queue>
#include <utility>
#include <fstream>
#include <Variant.h>
#include "ReferenceReader.h"
#include "InputStructures.h"
#include "BAMWalkerEngine.h"

using namespace std;

struct HotspotAllele {

  int chr;
  int pos;
  int ref_length;
  string alt;
  AlleleType type;
  int length;

  VariantSpecificParams params;
};

enum AlleleHint {
  NO_HINT = 0,
  FWD_BAD_HINT = 1,
  REV_BAD_HINT = 2,
  BOTH_BAD_HINT = 3
};

class hint_item {
 public:
  int chr_ind;
  long int pos;
  long int value;
  long int rlen;
  string alt;
}; 
 
class HotspotReader {
public:
  HotspotReader();
  ~HotspotReader();

  void Initialize(const ReferenceReader &ref_reader, const string& hotspot_vcf_filename);
  void Initialize(const ReferenceReader &ref_reader);

  bool HasMoreVariants() const { return has_more_variants_; }
  void FetchNextVariant();

  const vector<HotspotAllele>& next() const { return next_; }
  int next_chr() const { return next_chr_; }
  int next_pos() const { return next_pos_; }

  vector< hint_item >     hint_vec;
  int hint_chr_index() const { return (int)hint_vec[hint_cur_].chr_ind; }
  long int hint_position() const { return hint_vec[hint_cur_].pos; }
  long int hint_value() const { return hint_vec[hint_cur_].value; }
  long int  hint_rlen() const { return hint_vec[hint_cur_].rlen; }
  string hint_alt() const { return hint_vec[hint_cur_].alt; }
  bool hint_empty() const { return hint_vec.empty() || hint_header_ >=  hint_vec.size();}
  void hint_pop() { hint_header_++;}
  void hint_next() { hint_cur_++;}
  void hint_start() { hint_cur_ = hint_header_;}
  bool hint_more() { return hint_cur_ < hint_vec.size();}

  void MakeHintQueue(const string& hotspot_vcf_filename);

private:
  const ReferenceReader * ref_reader_;
  vector<HotspotAllele>   next_;
  int                     next_chr_;
  int                     next_pos_;
  bool                    has_more_variants_;
  unsigned int 			  hint_header_;
  unsigned int			  hint_cur_;


  vcf::VariantCallFile    hotspot_vcf_;
  //ifstream                hotspot_vcf_;

  int                     line_number_;

};



#endif //HOTSPOTREADER_H


