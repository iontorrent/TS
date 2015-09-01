/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerUtils.h
//! @ingroup  BaseCaller
//! @brief    Small helper classes and functions for BaseCaller

#ifndef BASECALLERUTILS_H
#define BASECALLERUTILS_H

#include <string>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "OptArgs.h"


namespace ion {

// =====================================================================
class FlowOrder {
public:
  FlowOrder() : cycle_nucs_("TACG"), num_flows_(0) {}
  FlowOrder(const std::string& cycle_nucs, int num_flows) : cycle_nucs_(cycle_nucs), num_flows_(num_flows) {
    BuildFullArrays();
  }

  // ------------------------------------------------------------------
  void SetFlowOrder(const std::string& cycle_nucs, int num_flows) {
    cycle_nucs_ = cycle_nucs;
    num_flows_ = num_flows;
    BuildFullArrays();
  }

  // ------------------------------------------------------------------
  void SetFlowOrder(const std::string& cycle_nucs) {
    cycle_nucs_ = cycle_nucs;
    BuildFullArrays();
  }

  // ------------------------------------------------------------------
  void SetNumFlows(int num_flows) {
    num_flows_ = num_flows;
    BuildFullArrays();
  }

  // ------------------------------------------------------------------
  char operator[](int flow) const { return full_nucs_.at(flow); }

  const std::string& str() const { return cycle_nucs_; }
  const char *c_str() const { return cycle_nucs_.c_str(); }
  bool is_ok() const { return num_flows_ > 0 and not cycle_nucs_.empty(); }

  int int_at(int flow) const { return full_ints_.at(flow); }
  char nuc_at(int flow) const { return full_nucs_.at(flow); }

  int num_flows() const { return num_flows_; }
  const char *full_nucs() const { return full_nucs_.c_str(); }

  // ------------------------------------------------------------------
  int BasesToFlows (const std::string& seq, int *ionogram, int ionogram_space) const
  {
    int flows = 0;
    unsigned int bases = 0;
    while (flows < ionogram_space and flows < num_flows_ and bases < seq.length()) {
      ionogram[flows] = 0;
      while (bases < seq.length() and full_nucs_[flows] == seq[bases]) {
        ionogram[flows]++;
        bases++;
      }
      flows++;
    }
    return flows;
  }

  // ------------------------------------------------------------------
  int FlowsToBases(const std::vector<char>& ionogram, std::string &bases) const
  {
    int num_bases = 0;
    for(int flow = 0; flow < (int)ionogram.size() and flow < num_flows_; ++flow)
      num_bases += ionogram[flow];
    bases.clear();
    bases.reserve(num_bases+1);
    for(int flow = 0; flow < (int)ionogram.size() and flow < num_flows_; ++flow)
      for (int hp = 0; hp < ionogram[flow]; ++hp)
        bases.push_back(nuc_at(flow));
    return num_bases;
  }

  // ------------------------------------------------------------------
  static int NucToInt (char nuc) {
      if (nuc=='a' or nuc=='A') return 0;
      if (nuc=='c' or nuc=='C') return 1;
      if (nuc=='g' or nuc=='G') return 2;
      if (nuc=='t' or nuc=='T') return 3;
      return -1;
  }

  // ------------------------------------------------------------------
  static char IntToNuc (int idx) {
      if (idx == 0) return 'A';
      if (idx == 1) return 'C';
      if (idx == 2) return 'G';
      if (idx == 3) return 'T';
      return 'N';
  }

  // -----------------------------------------------------------------
  static std::string IntToNucStr (int idx){
    char nuc = IntToNuc(idx);
    std::string nuc_str(&nuc,1);
    return nuc_str;
  }

  // ------------------------------------------------------------------
private:
  void BuildFullArrays() {
    if (cycle_nucs_.empty())
      return;
    full_nucs_.resize(num_flows_);
    full_ints_.resize(num_flows_);
    for (int flow = 0; flow < num_flows_; ++flow) {
      full_nucs_[flow] = cycle_nucs_[flow % cycle_nucs_.size()];
      full_ints_[flow] = NucToInt(full_nucs_[flow]);
    }
  }

  std::string       cycle_nucs_;
  std::string       full_nucs_;
  std::vector<int>  full_ints_;
  int               num_flows_;
};

// =============================================================================
// Facilitate handling of chip subsets

class ChipSubset {
public:
  ChipSubset() {
    chip_size_x_    = 0;  chip_size_y_    = 0;
    region_size_x_  = 0;  region_size_y_  = 0;
    num_regions_x_  = 0;  num_regions_y_  = 0;
    num_regions_    = 0;
    block_offset_x_ = 0;  block_offset_y_ = 0;
    subset_begin_x_ = 0;  subset_begin_y_ = 0;
    subset_end_x_   = 0;  subset_end_y_   = 0;
    next_region_    = 0;
    next_begin_x_   = 0;  next_begin_y_   = 0;
    num_wells_      = 0;
  }

  // ------------------------------------------------------------------
  // Chip subset initialization for BaseCaller

  bool InitializeChipSubsetFromOptArgs(OptArgs &opts, const unsigned int chipSize_x, const unsigned int chipSize_y,
                                                  const unsigned int regionSize_x, const unsigned int regionSize_y){

    subset_begin_x_ = subset_begin_y_ = 0;
    subset_end_x_   = chip_size_x_ = chipSize_x;
    subset_end_y_   = chip_size_y_ = chipSize_y;

    // Coordinate offset is addressable in row,col and x,y format, with x,y taking precedence
    // In BaseCaller we do not use the offset for anything other than output info.
    // Counters within block assume a zero offset index.
    block_offset_x_ = opts.GetFirstInt    ('-', "block-col-offset", 0);
    block_offset_y_ = opts.GetFirstInt    ('-', "block-row-offset", 0);
    std::stringstream default_opt_val;
    default_opt_val << block_offset_x_ << ',' << block_offset_y_;
    std::vector<int> arg_block_offset  = opts.GetFirstIntVector ('-', "block-offset", default_opt_val.str(), ',');
    if (arg_block_offset.size() != 2) {
      std::cerr << "BaseCaller Option Error: argument 'block-offset' needs to be 2 comma separated values <Int>,<Int>" << std::endl;
      exit (EXIT_FAILURE);
    }
    block_offset_x_ = arg_block_offset.at(0);
    block_offset_y_ = arg_block_offset.at(1);

    // Make default options from variables in function header
    default_opt_val.str("");
    default_opt_val << regionSize_x << ',' << regionSize_y;
    std::vector<int> arg_region_size  = opts.GetFirstIntVector ('-', "region-size", default_opt_val.str(), ',');
    if (arg_region_size.size() != 2) {
      std::cerr << "BaseCaller Option Error: argument 'region-size' needs to be 2 comma separated values <Int>,<Int>" << std::endl;
      exit (EXIT_FAILURE);
    }
    region_size_x_ = arg_region_size.at(0);
    region_size_y_ = arg_region_size.at(1);
    num_regions_x_ = (chip_size_x_ +  region_size_x_ - 1) / region_size_x_;
    num_regions_y_ = (chip_size_y_ +  region_size_y_ - 1) / region_size_y_;
    num_regions_ = num_regions_x_ * num_regions_y_;

    // Dash as string separator protects against someone trying to input negative numbers
    // Subset specified by rows,cols is zero indexed without offset
    std::vector<int> arg_subset_rows  = opts.GetFirstIntVector ('r', "rows", "", '-');
    if (arg_subset_rows.size() == 2) {
      subset_begin_y_ = arg_subset_rows.at(0);
      subset_end_y_   = arg_subset_rows.at(1);
      // Immediate sanity checks
      subset_end_y_   = std::min(subset_end_y_, chip_size_y_);
      subset_begin_y_ = std::min(subset_begin_y_, subset_end_y_);
    }
    else if (not arg_subset_rows.empty()) {
      std::cerr << "BaseCaller Option Error: argument 'rows' needs to be in the format <Int>-<Int>" << std::endl;
      exit (EXIT_FAILURE);
    }

    std::vector<int> arg_subset_cols  = opts.GetFirstIntVector ('c', "cols", "", '-');
    if (arg_subset_cols.size() == 2) {
      subset_begin_x_ = arg_subset_cols.at(0);
      subset_end_x_   = arg_subset_cols.at(1);
      // Immediate sanity checks
      subset_end_x_   = std::min(subset_end_x_, chip_size_x_);
      subset_begin_x_ = std::min(subset_begin_x_, subset_end_x_);
    }
    else if (not arg_subset_cols.empty()) {
      std::cerr << "BaseCaller Option Error: argument 'cols' needs to be in the format <Int>-<Int>" << std::endl;
      exit (EXIT_FAILURE);
    }

    num_wells_ = (subset_end_x_-subset_begin_x_) * (subset_end_y_-subset_begin_y_);

    std::cout << "BaseCaller chip region x: " << subset_begin_x_ << "-" << subset_end_x_
    	      << " y: " << subset_begin_y_ << "-" << subset_end_y_
    	      << " with a total of " << num_wells_ << " wells and regions of size "
    	      << region_size_x_ << "x" << region_size_y_ << std::endl;
    return true;
  };

  // ------------------------------------------------------------------

  int GetChipSizeX()    const { return chip_size_x_; };
  int GetChipSizeY()    const { return chip_size_y_; };
  int GetBeginX()       const { return subset_begin_x_; };
  int GetBeginY()       const { return subset_begin_y_; };
  int GetEndX()         const { return subset_end_x_; };
  int GetEndY()         const { return subset_end_y_; };
  int GetRowOffset()    const { return block_offset_y_; };
  int GetColOffset()    const { return block_offset_x_; };
  int GetOffsetX()      const { return block_offset_x_; };
  int GetOffsetY()      const { return block_offset_y_; };
  int GetRegionSizeX()  const { return region_size_x_; };
  int GetRegionSizeY()  const { return region_size_y_; };
  int GetNumRegionsX()  const { return num_regions_x_; };
  int GetNumRegionsY()  const { return num_regions_y_; };
  int NumRegions()      const { return num_regions_; };
  int NumWells()        const { return num_wells_; };

  // ------------------------------------------------------------------

  bool GetCurrentRegionAndIncrement(int & current_region, int & begin_x, int & end_x, int & begin_y, int & end_y)
  {
    if (next_begin_y_ >= chip_size_y_)
      return false;

	current_region = next_region_;
    begin_x = next_begin_x_;
    begin_y = next_begin_y_;
    end_x   = std::min(begin_x + region_size_x_, chip_size_x_);
    end_y   = std::min(begin_y + region_size_y_, chip_size_y_);

    // Increment region
    next_region_++;
    next_begin_x_ += region_size_x_;
    if (next_begin_x_ >= chip_size_x_) {
      next_begin_x_ = 0;
      next_begin_y_ += region_size_y_;
    }
    return true;
  };

  // ------------------------------------------------------------------
  // Chip subset initialization for Calibration Module

  bool InitializeCalibrationRegionsFromOpts(OptArgs &opts) {

    // Read in block coordinate offset

    block_offset_x_ = opts.GetFirstInt    ('-', "block-col-offset", 0);
    block_offset_y_ = opts.GetFirstInt    ('-', "block-row-offset", 0);
    std::stringstream default_opt_val;
    default_opt_val << block_offset_x_ << ',' << block_offset_y_;
    std::vector<int> arg_block_offset  = opts.GetFirstIntVector ('-', "block-offset", default_opt_val.str(), ',');
    if (arg_block_offset.size() != 2) {
      std::cerr << "BaseCaller Option Error: argument block-offset needs to be 2 comma separated values <Int>,<Int>" << std::endl;
      exit (EXIT_FAILURE);
    }

    // Read in chip/block size (default to 318 for testing)

    chip_size_y_ = opts.GetFirstInt    ('-', "block-row-size", 3792);
    chip_size_x_ = opts.GetFirstInt    ('-', "block-col-size", 3392);
    default_opt_val.str("");
    default_opt_val << chip_size_x_ << ',' << chip_size_y_;
    std::vector<int> arg_chip_size  = opts.GetFirstIntVector ('-', "block-size", default_opt_val.str(), ',');
    if (arg_chip_size.size() != 2) {
      std::cerr << "Calibration Option Error: argument  block-size needs to be 2 comma separated values <Int>,<Int>" << std::endl;
      exit (EXIT_FAILURE);
    }

    // Chip subdivision by regions

    std::vector<int> arg_num_regions  = opts.GetFirstIntVector ('-', "num-calibration-regions", "2,2", ',');
    if (arg_num_regions.size() != 2) {
      std::cerr << "Calibration Option Error:argument num-regions needs to be 2 comma separated values <Int>,<Int>" << std::endl;
      exit (EXIT_FAILURE);
    }

    bool success = InitializeCalibrationRegions(arg_block_offset.at(0), arg_block_offset.at(1),
                                         arg_chip_size.at(0), arg_chip_size.at(1),
                                         arg_num_regions.at(0), arg_num_regions.at(1));

    std::cout << "Calibration chip region x: " << subset_begin_x_ << "-" << subset_end_x_
              << " y: " << subset_begin_y_ << "-" << subset_end_y_ << " divided into "
              << num_regions_x_ << "x" << num_regions_y_ << "=" << num_regions_ << " regions." << std::endl;
    return (success);
  }

  // ------------------------------------------------------------------

 bool InitializeCalibrationRegions(const int block_offset_x, const int block_offset_y,
                                   const int chip_size_x,    const int chip_size_y,
                                   const int num_regions_x,  const int num_regions_y)
  {
    subset_begin_x_ = block_offset_x_ = block_offset_x;
    subset_begin_y_ = block_offset_y_ = block_offset_y;

    chip_size_x_ = chip_size_x;
    chip_size_y_ = chip_size_y;
    subset_end_x_ = subset_begin_x_ + chip_size_x_;
    subset_end_y_ = subset_begin_y_ + chip_size_y_;

    num_regions_x_ = num_regions_x;
    num_regions_y_ = num_regions_y;
    region_size_x_  = (chip_size_x_ + num_regions_x_ -1) / num_regions_x_;
    region_size_y_  = (chip_size_y_ + num_regions_y_ -1) / num_regions_y_;

    num_regions_ = num_regions_x_ * num_regions_y_;
    num_wells_   = (subset_end_x_-subset_begin_x_) * (subset_end_y_-subset_begin_y_);

    return true;
  }

  // ------------------------------------------------------------------

  int CoordinatesToRegionIdx(int x, int y) const
  {
    if (x < subset_begin_x_ or x >= subset_end_x_)
      return(-1);
    if (y < subset_begin_y_ or y >= subset_end_y_)
      return(-1);

    // In line with BaseCaller region numbering, opposite to old recalibration code
    return (((x - subset_begin_x_) / region_size_x_) + ((y - subset_begin_y_) / region_size_y_)*num_regions_x_);
  };

  // ------------------------------------------------------------------

  void GetRegionStart(int region_idx, int& start_x, int& start_y) const
  {
    if (num_regions_x_ == 0) {
      std::cerr << "ERROR in ChipSubset::GetRegionStart : num_regions_x_ is zero!" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (region_idx > -1 and region_idx < num_regions_) {
	  start_x = (region_idx % num_regions_x_) * region_size_x_ + subset_begin_x_;
      start_y = (region_idx / num_regions_x_) * region_size_y_ + subset_begin_y_;
    }
    else {
      start_x = start_y = -1;
    }
  }

  // ------------------------------------------------------------------
private:
    // Generic sizes
    int      chip_size_y_;            //!< Chip/block height in wells file (rows)
    int      chip_size_x_;            //!< Chip/block width in wells file (columns)
    int      region_size_y_;          //!< Processing region (H5 chunk size in wells file)
    int      region_size_x_;          //!< Processing region (H5 chunk size in wells file)

    int      num_regions_x_;          //!< Number of regions along X
    int      num_regions_y_;          //!< Number of regions along Y
    int      num_regions_;            //!< Total number of regions

    // Block offset for Proton Chips
    int      block_offset_x_;         //!< X coordinate offset of block
    int      block_offset_y_;         //!< Y coordinate offset of block

    // Chip subset coordinates selected
    int      subset_begin_x_;         //!< Starting X of chip subset selected
    int      subset_begin_y_;         //!< Starting Y of chip subset selected
    int      subset_end_x_;           //!< Ending X of chip subset selected
    int      subset_end_y_;           //!< Ending Y of chip subset selected

    // Threading block management
    int      next_region_;            //!< Number of next region that needs processing by a worker
    int      next_begin_x_;           //!< Starting X coordinate of next region
    int      next_begin_y_;           //!< Starting Y coordinate of next region

    int      num_wells_;              //!< Total number of wells in the selected chip sub-block
};

}


// =============================================================================

#define MAX_KEY_FLOWS     64

class KeySequence {
public:
  KeySequence() : bases_length_(0), flows_length_(0) { flows_[0]=0; }
  KeySequence(const ion::FlowOrder& flow_order, const std::string& key_string, const std::string& key_name) {
    Set(flow_order, key_string, key_name);
  }

/*  void Set(const std::string& flow_order, const std::string& key_string, const std::string& key_name) {
    name_ = key_name;
    bases_ = key_string;
    transform(bases_.begin(), bases_.end(), bases_.begin(), ::toupper);
    bases_length_ = key_string.length();
    flows_length_ = seqToFlow(bases_.c_str(), bases_length_, &flows_[0], MAX_KEY_FLOWS,
          (char *)flow_order.c_str(), flow_order.length());
  }*/
  void Set(const ion::FlowOrder& flow_order, const std::string& key_string, const std::string& key_name) {
    name_ = key_name;
    bases_ = key_string;
    transform(bases_.begin(), bases_.end(), bases_.begin(), ::toupper);
    bases_length_ = key_string.length();
    flows_length_ = flow_order.BasesToFlows(bases_, flows_, MAX_KEY_FLOWS);
  }

  const std::string &   name() const { return name_; }
  const std::string &   bases() const { return bases_; }
  int                   bases_length() const { return bases_length_; }
  const int *           flows() const { return flows_; }
  int                   flows_length() const { return flows_length_; }
  int                   operator[](int i) const { return flows_[i]; }
  const char *          c_str() const { return bases_.c_str(); }

  // Feature request: keypass a basespace read
  // Feature request: keypass a flowspace read (int or float)

private:
  std::string           name_;
  std::string           bases_;
  int                   bases_length_;
  int                   flows_length_;
  int                   flows_[MAX_KEY_FLOWS];
};




#endif // BASECALLERUTILS_H
