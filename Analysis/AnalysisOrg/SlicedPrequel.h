/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SLICEDPREQUEL_H
#define SLICEDPREQUEL_H

#include <string>
#include <vector>
#include "Region.h"
#include "Serialization.h"

// organize prequel data for a whole chip
class SlicedPrequel{
public:
  // beadfind/bkgmodel need to know how we slice the chip up
  std::vector<Region> region_list;
  int num_regions;

  // bkg model needs to know these things per location/slice
  std::vector<RegionTiming> region_timing;
  std::vector<float> smooth_t0_est;
  // possibly need these in bkg model
  std::vector<float> tauB, tauE;

  // and the files where we're going to save things
  std::string bfFile;
  std::string bfMaskFile;
  std::string bfStatsFile;
   
  SlicedPrequel();
  ~SlicedPrequel();
  void FileLocations(std::string &analysisLocation);
  void Allocate(int _num_regions);
  void SetRegions(int _num_regions, int rows, int cols,
		  int regionXsize, int regionYsize);

  void WriteBeadFindForSignalProcessing();
  void LoadBeadFindForSignalProcessing(bool load);

  void RestrictRegions(std::vector<int>& region_list);

 private:
  void Elide(std::vector<unsigned int>& regions_to_use);

  // Serialization section
  friend class boost::serialization::access;
  template<typename Archive>
    void serialize(Archive& ar, const unsigned version) {
      ar & 
	region_list &
	num_regions &
	region_timing &
  smooth_t0_est &
	tauB &
	tauE;
      // the following are reinitialized every time
	// bfFile &
	//bfMaskFile &
	//bfStatsFile;
  }
 
};

#endif // SLICEDPREQUEL_H
