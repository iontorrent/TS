/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HotspotReader.h
//! @ingroup  VariantCaller
//! @brief    Customized hotspot VCF parser


#ifndef HOTSPOTREADER_H
#define HOTSPOTREADER_H

#include <string>
#include <vector>
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


class HotspotReader {
public:
  HotspotReader();
  ~HotspotReader();

  void Initialize(const ReferenceReader &ref_reader, const string& hotspot_vcf_filename);

  bool HasMoreVariants() const { return has_more_variants_; }
  void FetchNextVariant();

  const vector<HotspotAllele>& next() const { return next_; }
  int next_chr() const { return next_chr_; }
  int next_pos() const { return next_pos_; }

private:
  const ReferenceReader * ref_reader_;
  vector<HotspotAllele>   next_;
  int                     next_chr_;
  int                     next_pos_;
  bool                    has_more_variants_;


  vcf::VariantCallFile    hotspot_vcf_;
  //ifstream                hotspot_vcf_;

  int                     line_number_;

};



#endif //HOTSPOTREADER_H


