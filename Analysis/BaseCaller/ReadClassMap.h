/* Copyright (C) 2019 Thermo Fisher Scientific. All Rights Reserved */

#ifndef READCLASSMAP_H
#define READCLASSMAP_H

#include <string>
#include <vector>
#include <assert.h>
#include "Mask.h"
#include "ReservoirSample.h"

using namespace std;

// -------------------------------------------------------------

enum ReadClassType{
  MapNone               = 0,
  MapLibrary            = ( 1<<0 ), // This bead is a library bead
  MapTF                 = ( 1<<1 ), // This bead is a TF
  MapOutputWell         = ( 1<<2 ), // Should this well part of the output bam (downsampling)
  MapCalibration        = ( 1<<3 ), // Random calibration wells
  MapUnfiltered         = ( 1<<4 ), // Random unfiltered wells
  MapFilteredBadKey     = ( 1<<5 ), // MaskFilteredBadKey is set
  MapFilteredHighPPF    = ( 1<<6 ), // MaskFilteredBadResidual is set
  MapFilteredPolyclonal = ( 1<<7 ), // MaskFilteredBadPPF is set

  MapUseWells1          = ( 1<<8 ), // Use signal from wells file 1
  MapUseWells2          = ( 1<<9 ), // etc., ...
  MapUseWells3          = ( 1<<10),
  MapUseWells4          = ( 1<<11),
  MapUseWells5          = ( 1<<12),
  MapUseWells6          = ( 1<<13),
  MapUseWells7          = ( 1<<14),
  MapUseWells8          = ( 1<<15),
  MapAll                = 0xffff,

  MapLiveBead           = MapLibrary | MapTF,
  MapFiltered           = MapFilteredBadKey | MapFilteredHighPPF | MapFilteredPolyclonal
};


// ============================================================================
// Class ReadClassMap
// A mask type class for BaseCaller that contains a summary of all the relevant
// classifcation & sampling information that threads need to be aware of.

class ReadClassMap {

  std::vector<uint16_t>  class_map;
  int                    W;
  int                    H;
  bool                   ignore_washouts_;
  unsigned int           num_library_;
  unsigned int           num_tf_;
  unsigned int           num_valid_lib_;
  unsigned int           num_valid_tf_;

public:

  Mask                   filter_mask;

  ReadClassMap();

  void LoadMaskFiles(const std::vector<std::string> & mask_file_names, bool ignore_washouts=false);

  bool WriteFilterMask(const std::string mask_filename);

  inline void setClassType(int index, ReadClassType type){
    class_map.at(index) |= type;
  };

  int getMaskHeight() const
  {
    return H;
  };

  int getMaskWidth() const
  {
    return W;
  };

  unsigned int getNumWells() const
  {
    return class_map.size();
  }

  inline unsigned int getSignalDiversity(int index) const
  {
    unsigned int diversity = 0;
    for (unsigned int i = 8; i<16; ++i){
      if ((class_map.at(index) & (1<<i)) > 0)
        ++diversity;
    }
    return diversity;
  };

  inline bool IsValidRead(int well_index) const
  {
    return (ClassMatch(well_index, MapFiltered) == false &&
            ClassMatch(well_index, MapLiveBead) &&
            getSignalDiversity(well_index) > 0);
  };

  inline bool IsValidRead(int x, int y) const
  {
    return IsValidRead(y*W+x);
  };

  inline void SetClassUnfiltered(int index)
  {
    class_map.at(index) &= 0xff1f;
  };

  inline bool UseWells(int well_index, unsigned int file_index) const
    {
      assert(file_index<8);
      return (class_map.at(well_index) & (1<<(8+file_index)) ? true : false);
    };

  inline bool UseWells(int x, int y, unsigned int file_index) const
  {
    return UseWells(y*W+x, file_index);
  };

  inline bool MaskMatch(int x, int y, MaskType type) const
  {
    return filter_mask.Match(x,y, type);
  };

  inline bool MaskMatch(int index, MaskType type) const
  {
    return filter_mask.Match(index, type);
  };

  inline bool ClassMatch(int x, int y, ReadClassType type) const
  {
    return ClassMatch(y*W+x, type);
  };

  inline uint16_t getClass(int x, int y) const
  {
    return (class_map.at(y*W+x));
  }
  inline bool ClassMatch(int index, ReadClassType type) const
  {
    if ( index < 0 || index >= ( W*H ) )
      return false;

    return ( ( class_map.at(index) & type ? true : false ) );
  };

  unsigned int NumLibWells(){
    return num_library_;
  };

  unsigned int NumTfWells(){
    return num_tf_;
  };

  unsigned int NumValidWells(){
    return (num_valid_lib_+num_valid_tf_);
  }

  void Close();

};

#endif // READCLASSMAP_H
