/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MERGEACQ_H
#define MERGEACQ_H

#include <stdint.h>
#include <iostream>
#include "Image.h"
#include "IonErr.h"
#include "Acq.h"

/**
 * Class to take to image files and merge them into a single image
 * file based on the the coordinates given. Image files must line up
 * with coordinates given and have same time stamps. Originally
 * intended to merge the thumbnails from the top and bottom of bb.
 */
class MergeAcq {

 public:
  /// Constructor
  MergeAcq() { Init(); }

  /// Register first image that will start at coordinates (0,0);
  void SetFirstImage(Image *img) { mFirst = img; }

  /// Register first image and the bottom left hand side where it will start
  void SetSecondImage(Image *img, int rowStart, int colStart);

  /// Merge the two images set above into a single image
  void Merge(Image &combo);

  void MergeImageStack(RawImage *combo, std::vector<Image *> &imgStack, bool colWise=true) ;

 private:

  /// Fill the buffers based on input chips and size of new chip.
  void FillNewImage(const short *first, size_t firstRows, size_t firstCols,
                    const short *second, size_t secondRows, size_t secondCols,
                    size_t rowStart, size_t colStart,
                    size_t rows, size_t cols, size_t frames, short *dest);

  void FillOneImage(const short *first, size_t firstRows, size_t firstCols,
                    size_t rowStart, size_t colStart,
                    size_t rows, size_t cols, size_t frames, short *dest);

  /// Some sanity checks for images
  void CheckImagesAlign();
  void CheckAlignment(int n, int cols, int rows, int &preCols, int &preRows,bool colWise);
  void stackMetaData(const RawImage *src, RawImage *dst, int nImages, bool colWise);

  void Init();

  Image *mFirst; ///< memory owned elsewhere
  Image *mSecond; ///< memory owned elsewhere
  ///< Offsets into new file where second image goes
  size_t mSecondRowStart, mSecondColStart;

};

#endif // MERGEACQ_H
