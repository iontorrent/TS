/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "IonErr.h"
#include "MergeAcq.h"

void MergeAcq::SetSecondImage(Image *img, int rowStart, int colStart) {
  mSecond = img;
  mSecondRowStart = rowStart;
  mSecondColStart = colStart;
}

void MergeAcq::Merge(Image &combo) {
  CheckImagesAlign(); 
  RawImage *raw = new RawImage();
  const RawImage *first = mFirst->GetImage();
  const RawImage *second = mSecond->GetImage();
  /* do either compressed or uncompressed. */
  if (first->image != NULL) { 
    raw->rows = first->rows + mSecondRowStart;
    raw->cols = first->cols + mSecondColStart;
    raw->frames = first->frames;
    raw->uncompFrames = first->uncompFrames;
    raw->compFrames = first->compFrames;
    raw->channels = first->channels;
    raw->interlaceType = first->interlaceType;
    raw->frameStride = raw->cols * raw->rows;
    raw->baseFrameRate = first->baseFrameRate;
    // for some reason these are null with bb images.
    if (first->interpolatedFrames != NULL) {
      raw->interpolatedFrames = (int *)malloc(sizeof(int)*raw->uncompFrames);
      memcpy(raw->interpolatedFrames, first->interpolatedFrames, sizeof(int)*raw->uncompFrames);
    }
    if (first->interpolatedMult != NULL) {
      raw->interpolatedMult = (float *)malloc(sizeof(float)*raw->uncompFrames);
      memcpy(raw->interpolatedMult, first->interpolatedMult, sizeof(float)*raw->uncompFrames);
    }
    
    // Note = use malloc as that is how
    uint64_t size = (uint64_t)raw->rows * raw->cols * raw->frames * sizeof(float);
    raw->image = (short *)malloc(size);
    memset(raw->image, 0, size);
    raw->timestamps = (int *)malloc(raw->frames * sizeof(int));
    memcpy(raw->timestamps, first->timestamps, raw->frames * sizeof(int));
    FillNewImage(first->image, first->rows, first->cols,
                 second->image, second->rows, second->cols,
                 mSecondRowStart, mSecondColStart, 
                 raw->rows, raw->cols, raw->frames, raw->image);
    combo.SetImage(raw);
  }
  else { 
    delete raw;
    raw = NULL;
    ION_ABORT("Don't support the compImage buffer");
  }
}

void MergeAcq::FillNewImage(const short *first, size_t firstRows, size_t firstCols,
                            const short *second, size_t secondRows, size_t secondCols,
                            size_t rowStart, size_t colStart,
                            size_t rows, size_t cols, size_t frames, short *dest) {
  size_t firstStride = firstRows * firstCols;
  size_t secondStride = secondRows * secondCols;
  size_t finalStride = rows * cols;

  for (size_t f = 0; f < frames; f++) {
    // First image copied in for this frame 
    for (size_t r = 0; r < firstRows; r++) {
      for (size_t c = 0; c < firstCols; c++) {
        dest[r * cols + c + finalStride * f] = first[r * firstCols + c + firstStride * f];
      }
    }
    for (size_t r = 0; r < secondRows; r++) {
      for (size_t c = 0; c < secondCols; c++) {
        dest[(r + rowStart) * cols + (c + colStart) + finalStride * f] = second[r * secondCols + c + secondStride * f];
      }
    }
  }
}

void MergeAcq::CheckImagesAlign() {
  const RawImage *first = mFirst->GetImage();
  const RawImage *second = mSecond->GetImage();
  ION_ASSERT(mFirst != NULL && mSecond != NULL, "Can't merge null images.");
  ION_ASSERT(first->rows == second->rows || first->cols == second->cols, 
             "Images must mactch in dimension with common edge.");
  ION_ASSERT(first->frames == second->frames, "Images must have same number of frames");
  if ( !((mSecondRowStart == (size_t)first->rows && mSecondColStart == 0) ||
         (mSecondColStart == 0 && mSecondColStart == (size_t)first->cols))) {
    ION_ABORT("Image 2 must start where image 1 starts.");
  }
}

void MergeAcq::Init() {
  mFirst = NULL;
  mSecond = NULL;
  mSecondRowStart = mSecondColStart = 0;
}
