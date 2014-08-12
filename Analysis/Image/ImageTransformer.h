/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGETRANSFORMER_H
#define IMAGETRANSFORMER_H


#include <stdio.h>

#include "Mask.h"
#include "datahdr.h"
#include "ByteSwapUtils.h"
#include "Region.h"
#include "ChannelXTCorrection.h"
#include "ImageNNAvg.h"
//#include "Image.h"
class Image;
struct RawImage;


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
  static void CalculateGainCorrectionFromBeadfindFlow (bool gain_debug_output, Image &bfImg, Mask &mask);
  /* uses a beadfind flow to compute the gain of each pixel this can
     be used as a correction for all future images */
  static void GainCalculationFromBeadfind(Mask *mask, RawImage *raw);
    /* optimized version of above function */
  static void GainCalculationFromBeadfindFaster(Mask *mask, RawImage *raw);
  static void GainCalculationFromBeadfindFasterSave(RawImage *raw, Mask *mask, char *bad_wells, 
                                                    int row_step, int col_step, ImageNNAvg *imageNN);

  static void GainCorrectImage(RawImage *raw);
  static void GainCorrectImage(SynchDat *sdat);
  static float getPixelGain(int row,int col,int img_cols)
  {
    return(gain_correction[row*img_cols+ col]);
  }
  static void DumpTextGain(int _image_cols, int _image_rows);

  // gain correction that is to be applied to all images
  static float *gain_correction;
  static char PCATest[128];

  ~ImageTransformer() {
    if (gain_correction != NULL)
    {free(gain_correction);
      gain_correction = NULL;
    }
  }
};

inline float CalculatePixelGain(float *my_trc,float *reference_trc,int min_val_frame, int raw_frames);


#endif // IMAGETRANSFORMER_H
