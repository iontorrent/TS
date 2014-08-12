/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef PCASPLINE_H
#define PCASPLINE_H

#include <stdio.h>
#include <string>
#include <vector>

/***
 Data resulting from a compression that would be necessary
 to uncompress the resulting data. 
*/
class PcaSplineRegion {

public:
  PcaSplineRegion() { 
    n_wells = n_frames = n_basis = 0;
    basis_vectors = coefficients = NULL;
  }

  int n_wells;   ///< number of columns in region
  int n_frames; ///< number of frames represented
  int n_basis; ///< number of basis vectors
  /*** 
    matrix of basis vectors in serialized form. First nframe 
    floats are first basis vector, next nframe floats are second
    basis vector, etc.
  */
  float *basis_vectors; 
  /*** 
    matrix of coefficents per basis vectors. Conceptually a matrix
    of number wells (nrow * ncol) by nbasis vectors in column
    major format (first nrow *ncol are coefficents for first basis
    vector). For historical reasons the wells themselves are
    in row major format so the coefficent for the nth basis vector 
    (basis_n) for the well at position row_i,col_j would be:
    value(row_i,col_j,basis_n) = \
        coefficents[row_i * ncol + col_j + (nrow * ncol * basis_n)];
  */
  float *coefficients;
};

/***
  Algorithms to compress a region of data. 
*/
class PcaSpline {

 public:
  PcaSpline() {
    m_num_pca = 0;
    m_knot_strategy = "no-knots";
    m_order = 4;
    m_tikhonov_reg = 0.0;
  }
  
  /*** 
    Constructor
     @param num_pca - Number of principal component vectors to use
     @param num_knots - Number of knots to select for splines
     @param knot_selection - Description of the method for chosing knots.
  */
  PcaSpline(int num_pca, const std::string &knot_selection) : 
    m_num_pca(num_pca), 
    m_knot_strategy(knot_selection) {
      m_order = 4;
      m_tikhonov_reg = 0.0f;
    }

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
  void LossyCompress(int n_wells, int n_frame, float *image, 
		     int n_sample_wells, float *image_sample,
		     PcaSplineRegion &compressed);

  /*** 
    Recreate the original image of data from the pca compression.
    @param nrow - number of rows in this image patch
    @param ncol - number of cols in this image patch
    @param nframe - number of frames in this image patch.
    @param image - Same order as RawImage structure. Individual wells of data in frame, row, col major order so 
                   value(row_i,col_j,frame_k) = image[row_i * ncol + col_j + (nrow * ncol * frame_k)]
  */
  void LossyUncompress(int n_wells, int nframe, float *image, const PcaSplineRegion &compressed);

  /**
    Create a knots vector depending on the strategy supplied. Some options include:
    - "even4" split range into 4 regions and pick middle two as knots
    - "middle" single point at middle of range
    - "explicit:3,6,...,45,52" tell program exactly where to put the knots.
    @param knot_strategy - text description of knots
    @param n_frame - number of frames in this image
    @param knots - vector to be filled in with correct knots.
   */
  void FillInKnots(const std::string &knot_strategy, int n_frame, std::vector<float> &knots);

  /** 
    How many basis vectors will be needed for a particular knot strategy
    @param n_pca - Number of pca vectors desired
    @param n_frames - Number of frames in image
    @param n_order - Order of splines desired.
    @param knot_strategy - How knots are being chosen (see function FillInKnots()
   */
  int  NumBasisVectors(int n_pca, int n_frames, int n_order, const std::string &knot_strategy);

  int  NumBasisVectors(int n_frames) { return NumBasisVectors(m_num_pca, n_frames, m_order, m_knot_strategy); }
  void SampleImageMatrix(int n_wells, int n_frames, float * __restrict orig, 
			 int *__restrict trcs_state, int target_sample,
			 int &n_sample, float ** __restrict sample);

  int GetOrder() { return m_order; }
  void SetOrder(int order) { m_order = order; }

  std::string GetKnotStrategy() { return m_knot_strategy; }
  void SetKnotStrategy(const std::string &knot_strategy) { m_knot_strategy = knot_strategy; }

  int GetNumPca() { return m_num_pca; }
  void SetNumPca(int num_pca) { m_num_pca = num_pca; }

  float GetTikhonov() { return m_tikhonov_reg; }
  void SetTikhonov(float k) { m_tikhonov_reg = k; }

private: 

  int m_num_pca;   ///< Number of pca vectors to use as basis functions
  std::vector<float> m_knots; ///< Knots for splines
  int m_order;  ///< Order of polynomial for splines
  float m_tikhonov_reg; ///< If greater than zero do regularization
  std::string m_knot_strategy; ///< Method to use for choosing splines
  
};

#endif // PCASPLINE_H
