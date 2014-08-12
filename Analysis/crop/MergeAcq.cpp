/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <vector>

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


void MergeAcq::FillOneImage(const short *first, size_t firstRows, size_t firstCols,
                            size_t rowStart, size_t colStart,
                            size_t rows, size_t cols, size_t frames, short *dest) {
  size_t firstStride = firstRows * firstCols;
  size_t finalStride = rows * cols;

  for (size_t f = 0; f < frames; f++) {
    size_t f1 =firstStride * f;
    size_t f2 =finalStride * f;
    // First image copied in for this frame
    for (size_t r = 0; r < firstRows; r++) {
        size_t r1 = r * firstCols;
        size_t r2 = (r+rowStart) * cols;
      for (size_t c = 0; c < firstCols; c++) {
        //if (f==0) std::cout << c << "," << r1 << "\t" << (colStart+c) << "," << r2 << std::endl << std::flush;
        dest[f2+r2+colStart+c] = first[f1+r1+c];
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

void MergeAcq::CheckAlignment(int n, int cols, int rows, int &preCols, int &preRows,bool colWise)
{
    if (n>0)
    {
        if (colWise)
        {
            if (cols != preCols)
            {
                std::cerr << "MergeAcq::MergeImageStack image alignment error at n=" << n << ": cols(" << cols << ") != preCols(" << preCols << ")" << std::endl << std::flush;
                exit(1);
            }
        }
        else
        {
            if (rows != preRows)
            {
                std::cerr << "MergeAcq::MergeImageStack image alignment error at n=" << n << ": rows(" << rows << ") != preRows(" << preRows << ")" << std::endl << std::flush;
                exit(1);
            }
        }
    }
    preRows = rows;
    preCols = cols;
}


void MergeAcq::Init() {
  mFirst = NULL;
  mSecond = NULL;
  mSecondRowStart = mSecondColStart = 0;
}


void MergeAcq::stackMetaData(const RawImage *src, RawImage *dst, int nImages, bool colWise)
{
    assert(src!=NULL);
    assert(dst!=NULL);
    dst->rows = src->rows;
    dst->cols = src->cols;
    if (colWise)
        dst->rows *= nImages;
    else
        dst->cols *= nImages;

    //std::cout << "MergeAcq::MergeImageStack rows=" << dst->rows << " cols=" << dst->cols << std::endl << std::flush;
    dst->frameStride = dst->cols * dst->rows;
    dst->frames = src->frames;
    dst->uncompFrames = src->uncompFrames;
    dst->compFrames = src->compFrames;
    dst->channels = src->channels;
    dst->interlaceType = src->interlaceType;
    dst->baseFrameRate = src->baseFrameRate;
    // for some reason these are null with bb images.
    if (src->interpolatedFrames != NULL) {
      dst->interpolatedFrames = (int *)malloc(sizeof(int)*dst->uncompFrames);
      memcpy(dst->interpolatedFrames, src->interpolatedFrames, sizeof(int)*dst->uncompFrames);
    }
    if (src->interpolatedMult != NULL) {
      dst->interpolatedMult = (float *)malloc(sizeof(float)*dst->uncompFrames);
      memcpy(dst->interpolatedMult, src->interpolatedMult, sizeof(float)*dst->uncompFrames);
    }
}


void MergeAcq::MergeImageStack(RawImage *raw, std::vector<Image *> &imgStack, bool colWise) {
    //assert(raw!=NULL);
    int nImages = imgStack.size();
    assert (nImages>0);
    //for (int i=0; i<nImages; i++)
    //    std::cout << "MergeImageStack imgStack[" << i << "]=" << imgStack[i] << " GetImage()=" << imgStack[i]->GetImage() << std::endl << std::flush;

  //CheckImagesAlign_stack();

  int curRowStart = 0;
  int curColStart = 0;
  int preRows = 0;
  int preCols = 0;

  for (int n=0; n<nImages; n++)
  {
      //std::cout << "MergeAcq::MergeImageStack copy image " << n << std::endl << std::flush;
      assert(imgStack[n] != NULL);
      assert(imgStack[n]->GetImage() != NULL);
      const RawImage *first = imgStack[n]->GetImage();
      //SetFirstImage(imgStack[n]);
      //const RawImage *first = mFirst->GetImage();
      /* do either compressed or uncompressed. */
      if (first->image != NULL)
      {
          if (n==0)
          {
              stackMetaData(first,raw,nImages,colWise);
              //std::cout << "MergeImageStack... merging " << first->rows << "x" << first->cols << " into " << raw->rows << "x" << raw->cols << std::endl << std::flush;

              // alloc memory only once!!!
              // Note = use malloc as that is how
              uint64_t size = (uint64_t)raw->rows * raw->cols * raw->frames * sizeof(float);
              if (raw->image)
                  free (raw->image);
              raw->image = (short *)malloc(size);
              memset(raw->image, 0, size);
              if (raw->timestamps)
                  free (raw->timestamps);
              raw->timestamps = (int *)malloc(raw->frames * sizeof(int));
              memcpy(raw->timestamps, first->timestamps, raw->frames * sizeof(int));
          }

        // check image alignment
        CheckAlignment(n,first->cols,first->rows,preCols,preRows,colWise);

        FillOneImage(first->image, first->rows, first->cols,
                     curRowStart, curColStart,
                     raw->rows, raw->cols, raw->frames, raw->image);

        if (colWise)
            curRowStart += first->rows;
        else
            curColStart += first->cols;
      }
      else
      {
          std::cerr << "MergeImageStack error: imgStack[" << n << "]==NULL" << std::endl << std::flush;
          exit(1);
      }

  /*
  else {
    delete raw;
    raw = NULL;
    std::cerr << "imgStack[" << n << "]==NULL" << std::endl << std::flush;
    ION_ABORT("MergeAcq::MergeImageStack aborted...");
    }
  */
  }
}

