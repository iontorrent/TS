/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGESPECCLASS_H
#define IMAGESPECCLASS_H


#include "CommandLineOpts.h"


class ImageSpecClass
{
public:
  int rows, cols;
  int scale_of_chip;
  unsigned int uncompFrames;
  int *timestamps;
  bool vfr_enabled;

  int n_timestamps;

  ImageSpecClass();
  void ReadFirstImage(Image &img, SystemContext &sys_context, ImageControlOpts &img_control, SpatialContext &loc_context);
  void TimeStampsFromImage ( Image &img , ImageControlOpts &img_control);
  void DimensionsFromImage(Image &img, SpatialContext &loc_context);
  void DeriveSpecsFromDat ( SystemContext &sys_context,ImageControlOpts &img_control, SpatialContext &loc_context);
  int LeadTimeForChipSize();
  ~ImageSpecClass();

 private:

  friend class boost::serialization::access;
  
};

namespace boost {  namespace serialization {
template<typename Archive>
void save(Archive& ar, const ImageSpecClass& o, const unsigned version) {
  // is there a better way to handle dynamic arrays??
  // if a pointer to this object had to serialized in a second time
  // timestamps would not point to the same memory...
  assert(o.n_timestamps >=0); // in case somehow we allow <0
  std::vector<int> timestamps_serializer;
  timestamps_serializer.resize(o.n_timestamps);
  for (int i=0; i<o.n_timestamps; i++){
    timestamps_serializer[i] = o.timestamps[i];
  }
  
  ar & 
    o.rows & o.cols &
    o.scale_of_chip &
    o.uncompFrames &
    timestamps_serializer;
  
  timestamps_serializer.clear();
}
template<typename Archive>
void load(Archive& ar, ImageSpecClass &o, const unsigned version) {
  std::vector<int> timestamps_serializer;
  ar & 
    o.rows & o.cols &
    o.scale_of_chip &
    o.uncompFrames &
    timestamps_serializer;
  // acqPrefix handled in the default constructor

  o.n_timestamps = (int)timestamps_serializer.size();
  o.timestamps = new int[o.n_timestamps];
  for (int i=0; i<o.n_timestamps; i++){
    o.timestamps[i] = timestamps_serializer[i];
  }
}
	}
}
BOOST_SERIALIZATION_SPLIT_FREE(ImageSpecClass)

#endif // IMAGESPECCLASS_H
