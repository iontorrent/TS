/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGETRANSFORMER_H
#define IMAGETRANSFORMER_H


#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

#include "Mask.h"
#include "datahdr.h"
#include "ByteSwapUtils.h"
#include "Region.h"
#include "ChipIdDecoder.h"
#include "ChannelXTCorrection.h"
#include "TikhonovSmoother.h"
#include "RawImage.h"

#include <vector>
#include <string>

  #define DEFAULT_VECT_LEN  7
class SynchDat;
  
// Clarify current state of image transformation available globally for all image loading
// this is state that needs to be re-entrant

class ImageCropping{
  public:
    static Region chipSubRegion;
  static  void SetCroppedRegionOrigin(int x,int y)
  {
    cropped_region_offset_x = x;
    cropped_region_offset_y = y;
  }
  static void SetCroppedSubRegion(Region &_subRegion){
    chipSubRegion.row = _subRegion.row;
    chipSubRegion.col = _subRegion.col;
    chipSubRegion.w   = _subRegion.w;
    chipSubRegion.h   = _subRegion.h;
  };
  // allow the user to specify the origin of a cropped data set so that cross-talk correction
  // works properly
  static int cropped_region_offset_x;
  static int cropped_region_offset_y;
};

class ImageTransformer{
public:
  // void    XTChannelCorrect(Mask *mask);
  static void    XTChannelCorrect(RawImage *raw, const char *experimentName);

  static  void    CalibrateChannelXTCorrection(const char *exp_dir,const char *filename, bool wait_for_prerun=true);

  static int dump_XTvects_to_file;  
  static ChipXtVectArrayType default_chip_xt_vect_array[];
  static int chan_xt_column_offset[];
  static ChannelXTCorrectionDescriptor selected_chip_xt_vectors;
  static ChannelXTCorrection *custom_correction_data;

//---------------gain corrections-------------------------------------------------

  static void CalculateGainCorrectionFromBeadfindFlow (char *_datDir, bool gain_debug_output);

  static void GainCalculationFromBeadfind(Mask *mask, RawImage *raw);
  static void GainCorrectImage(RawImage *raw);
  static void GainCorrectImage(SynchDat *sdat);
    static float getPixelGain(int row,int col,int img_cols)
  {
      return(gain_correction[row*img_cols+ col]);
  }
  static void DumpTextGain(int _image_cols, int _image_rows);

  // gain correction that is to be applied to all images
  static float *gain_correction;

  ~ImageTransformer() {
    if (gain_correction != NULL)
      {free(gain_correction);
	gain_correction = NULL;
      }
  }	
};

  inline float CalculatePixelGain(float *my_trc,float *reference_trc,int min_val_frame, int raw_frames);


#endif // IMAGETRANSFORMER_H
