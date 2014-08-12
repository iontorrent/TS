/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef DFCCOMPR_H
#define DFCCOMPR_H

#include <stdio.h>
#include <string>
#include <vector>
#include "ParallelDFT.h"

/***
 Data resulting from a compression that would be necessary
 to uncompress the resulting data. 
*/
class DfcCompr {

public:
  DfcCompr() : DFT() {
    n_wells = n_frames = n_basis = 0; n_maxBits = 7; n_minBits = 2;
    /*basis_vectors = coefficients =*/ scratch_pad = NULL;
    keyFrameI = keyFrameQ = scaleVectorI = scaleVectorQ = NULL;
    bitsPerFreq = NULL;
    deltaI = deltaQ = NULL;
    winParamLen = 5; winParamAlpha = 2.15f;
    GenerateWindow();
  }

  ~DfcCompr()
  {
      windowCompress.clear();
      windowExpand.clear();
  }

  /***
   n_frames has had to be hidden away because other things internal to the DFC
   engine now rely on it and also need to be reconfigured when it changes value.
  */
  int SetNumFrames(int frames);

  int n_wells;   ///< number of columns in region
  int n_basis; ///< number of basis vectors (frequency elements), must be less than half the number of frames
  int n_maxBits;  ///< maximum number of bits per frequency element (excluding sign bit)
  int n_minBits;  ///< minimum number of bits per frequency element (excluding sign bit)
  ///*** 
  //  matrix of basis vectors in serialized form. First nframe 
  //  floats are first basis vector, next nframe floats are second
  //  basis vector, etc.
  //*/
  //float *basis_vectors;  // NOT USED BY DFC
  ///*** 
  //  matrix of coefficents per basis vectors. Conceptually a matrix
  //  of number wells (nrow * ncol) by nbasis vectors in column
  //  major format (first nrow *ncol are coefficents for first basis
  //  vector). For historical reasons the wells themselves are
  //  in row major format so the coefficent for the nth basis vector 
  //  (basis_n) for the well at position row_i,col_j would be:
  //  value(row_i,col_j,basis_n) = coefficents[row_i * ncol + col_j + (nrow * ncol * basis_n)];
  //*/
  //float *coefficients;  // NOT USED BY DFC


  /***
   DFC is very different to PCA/Splines and requires a few additional items to 
   be stored for each block of wells that get compressed:
    - keyFrameI     : float * n_basis, vector containing real componens of mean spectrum
    - keyFrameQ     : float * n_basis, vector containing imaginary components of mean spectrum
    - bitsPerFreq   : uchar * n_basis, vector containing number of bits used to encode each frequency element delta
    - scaleVectorI  : float * n_basis, vector containing the common scaling values applied to the real frequency deltas
    - scaleVectorQ  : float * n_basis, vector containing the common scaling values applied to the immaginary frequency deltas
  */
  float         *keyFrameI;
  float         *keyFrameQ;
  unsigned char *bitsPerFreq;
  float         *scaleVectorI;
  float         *scaleVectorQ;

  /***
   DFC bulk data for bit-packing and storage.  Each vector is organized as n_basis
   samples for one well, followed by n_basis samples for the next well, and so on.
   deltaI[n] is the "real" part and deltaQ[n] is the "immaginary" part of the complex
   delta sample "delta[n]".
    - deltaI    : short * n_basis * n_wells
    - deltaQ    : short * n_basis * n_wells
  */
  short         *deltaI;
  short         *deltaQ;
  

  /***
    Decompose the image data into a set of basis vectors and each well's projection
    onto them.
    @param n_wells - number of wells in the image patch
    @param n_frame - number of frames in this image patch.
    @param image - Same order as RawImage structure. Individual wells of data in frame, row, col major order so
                   value(row_i,col_j,frame_k) = image[row_i * ncol + col_j + (nrow * ncol * frame_k)]
    @param n_sample_wells - number of wells in the sample of the patch
    @param image_sample - sample of image above for vectors and such
    @param compressed - output of a lossy compressed patch
  */
  void LossyCompress(float *image);


  /***
    Reconstruct the image data from the compressed frequency domain vectors.
    @param n_wells - number of wells in the image patch
    @param n_frame - number of frames in this image patch.
    @param image - Same order as RawImage structure. Individual wells of data in frame, row, col major order so
                   value(row_i,col_j,frame_k) = image[row_i * ncol + col_j + (nrow * ncol * frame_k)]
  */
  void LossyUncompress(float *image);


  /***
    Given the command line input, return the number of basis vectors
    to store in the file.
    @param _param - Dfc parameter passed on the command line
  */
  static int GetNumCoeff(int _param)
  {
	  //TODO: DFC figure out how many coefficients this input parameter means...
	  return _param;
  }

  /***
   Configure the transition smoothing window parameters.
  */
  int SetWindowLength(int length);
  inline int GetWindowLength() { return winParamLen; }
  float SetWindowAlpha(float alpha);
  inline float GetWindowAlpha() { return winParamAlpha; }


  float *scratch_pad;  // intermediate storage needed for DFT partial results, currently allocated/released in LossyCompress()


protected:

    void GenerateWindow();

    /***
     Calculate the frequency domain correlation statistics needed to generate the
     emphasis vector, which in turn is used to populate the bitsPerFreq vector.
    */
    void Emphasis();

    ParallelDFT         DFT;  // Discrete Fourier Transform processing object
    int                 n_frames; ///< number of frames represented
    int                 winParamLen;
    float               winParamAlpha;
    std::vector<float>  windowCompress;  // ripple reduction window
    std::vector<float>  windowExpand;    // inverse of window for use during reconstruction

};


/***
 This wrapper class maps the basis_vector and coefficients memory blocks to the
 DFC object's output vectors at run-time.  The DFC object is exposed so that all
 of the configuration members and methods can still be set/called.
*/
class DfcComprWrapper
{
public:
    DfcComprWrapper() : dfc() { basis_vectors = coefficients = NULL; }

    void LossyCompress(float *image);
    void LossyUncompress(float *image);

    float *basis_vectors;
    float *coefficients;
    
    DfcCompr dfc;

protected:
    void SetupPointers();
};


#endif // DFCCOMPR_H
