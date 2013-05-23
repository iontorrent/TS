/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef COMPARATORCLEAN_H
#define COMPARATORCLEAN_H

#include <armadillo>
#include <algorithm>
#include "TraceChunk.h"
#include "GridMesh.h"
#include "IonErr.h"

class ComparatorClean {

public:

  void GetNEigenScatter(arma::Mat<float> &YY, arma::Mat<float> &E, int nEigen) {
    try {
      Cov = YY.t() * YY;
      eig_sym(EVal, EVec, Cov);
      E.set_size(YY.n_cols, nEigen);
      // Copy largest N eigen vectors as our basis vectors
      int count = 0;
      for(size_t v = Cov.n_rows - 1; v >= Cov.n_rows - nEigen; v--) {
	std::copy(EVec.begin_col(v), EVec.end_col(v), E.begin_col(count++));
      }
    }
    catch(std::exception &e) {
      const char *w = e.what();
      ION_ABORT(w);
    }
  }

  void GetEigenProjection(arma::Mat<float> &data, arma::Col<unsigned int> &goodRows, size_t nEigen, arma::Mat<float> &proj) {
    ION_ASSERT(nEigen > 0 && nEigen < data.n_cols, "Must specify reasonable selection of eigen values.");
    ION_ASSERT(goodRows.n_rows > 2, "Must have at least a few good columns.");
    Y = data.rows(goodRows);
    GetNEigenScatter(Y,X, nEigen);
    // Calculate our best projection of data onto eigen vectors, as vectors are already orthonomal don't need to solve, just multiply
    try {
      B = data * X;
      proj = B * X.t();
      ION_ASSERT(proj.n_rows == data.n_rows && proj.n_cols == data.n_cols,"Wrong dimensions.");
      // arma::Mat<float> D;
      // D = abs(data - proj);
      // double val = arma::mean(arma::mean(D,0));
      // std::cout << "Mean abs val diff is: " << val << std::endl;
    }
    catch(std::exception &e) {
      const char *w = e.what();
      ION_ABORT(w);
    }
  }

  void GetComparatorCorrection(arma::Mat<float> &data, arma::Col<unsigned int> &goodData, arma::Mat<float> &predicted, 
  			       size_t patchRows, size_t patchCols, arma::Mat<float> &corrections, size_t nMod, size_t minSample) {
    ION_ASSERT(predicted.n_rows == data.n_rows && predicted.n_cols == data.n_cols,"Wrong dimensions.");
    Diff = data - predicted;
    size_t colModSize = nMod * patchCols;
    corrections.set_size(colModSize, data.n_cols);
    corrections.fill(0.0f);
    arma::Mat<float> M[colModSize];
    size_t mSize[colModSize];
    // Figure out number of "good" rows in each comparator and create scratch matrices
    std::fill(mSize, mSize + colModSize, 0);
    for (size_t i = 0; i < goodData.n_rows; i++) {
      int row = goodData(i) / patchCols;
      int col = goodData(i) % patchCols;
      int ix = (row % nMod) * patchCols + col;
      mSize[ix]++;
    }
    for (size_t i = 0; i < colModSize; i++) {
      if (mSize[i] > minSample) {
	M[i].set_size(mSize[i], data.n_cols);
	M[i].fill(0.0f);
      }
    }
    
    // Fill in scratch matrices for different mod number of rows
    size_t mCount[colModSize];
    std::fill(mCount, mCount + colModSize, 0);
    for (size_t i = 0; i < goodData.n_rows; i++) {
      int row = goodData(i) / patchCols;
      int col = goodData(i) % patchCols;
      int ix = (row % nMod) * patchCols + col;
      if (mSize[ix] > minSample) {
	try {
	  M[ix].row(mCount[ix]++) = Diff.row(goodData(i));
	}
	catch (...) {
	  std::cout << "Error at row: " << i << " " << goodData(i) << std::endl;
	}
      }
    }

    // For reach comparator,row mod tuple calculate the median residual
    for (size_t colIx = 0; colIx < patchCols; colIx++) {
      for (size_t modIx = 0; modIx < nMod; modIx++) {
	size_t ix = (modIx * patchCols + colIx);
	if (mCount[ix] > minSample) {
	  try {
	    arma::Col<float> m(M[ix].n_cols);
	    for (size_t i = 0; i < M[ix].n_cols; i++) {
	      m(i) = arma::median(M[ix].col(i));
	    }
	    corrections.row(ix) = m.t();
	    // if (modIx == 3 && colIx == 3) {
	    //   M[ix].print("Median data:");
	    //   m.print("Median:");
	    // }
	  }
	  catch (...) {
	    M[ix].print("sample is:" );
	  }
	}
      }
    }
  }

  bool isOk(size_t rowIx, size_t colIx, Mask &mask) {
    size_t ix = rowIx * mask.W() + colIx;
    return (!(mask[ix] & MaskPinned || mask[ix] & MaskExclude));
  }

  short &ValAt(short *img, size_t rows, size_t cols, size_t row, size_t col, size_t frame) {
    size_t step = rows * cols;
    return img[frame * step + row * cols + col];
  }

  void CorrectChip(short *image, size_t rows, size_t cols, size_t frames,
		   size_t rowStep, size_t colStep,
		   Mask *mask, int nEigen, int nEigenSmooth) {
    GridMesh<int> grid;
    grid.Init(rows, cols, rowStep, colStep);
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    for (size_t binIx = 0; binIx < grid.GetNumBin(); binIx++) {
      grid.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
      CorrectChunk(image, rows, cols, frames, rowStart, rowEnd, colStart, colEnd, mask, nEigen, nEigenSmooth);
    }
  }

  void CorrectChunk(short *image, size_t rows, size_t cols, size_t frames, 
		    Mask *mask, int nEigen, int nEigenSmooth) {
    CorrectChunk(image, rows, cols, frames, 0, rows, 0, cols, mask, nEigen, nEigenSmooth);
  }			       

  void CorrectChunk(short *image, size_t imageRows, size_t imageCols, size_t imageFrames, 
		    size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd,
		    Mask *mask, int nEigen, int nEigenSmooth) {
    size_t count = 0;
    arma::Col<unsigned int> good;
    size_t patchCols = colEnd - colStart;
    size_t patchRows = rowEnd - rowStart;
    try {
      // Initialize the matrix
      //      Copy.set_size(cols * rows, chunk.mDepth);
      Copy.set_size(imageCols * imageRows, imageFrames);
      count = 0;
      for (size_t rowIx = rowStart; rowIx < rowEnd; rowIx++) {
	for (size_t colIx = colStart; colIx < colEnd; colIx++) {
	  for (size_t frameIx = 0; frameIx < imageFrames; frameIx++) {
	    Copy(count, frameIx) = ValAt(image, imageRows, imageCols, rowIx, colIx, frameIx);
	  }
	  count++;
	}
      }
      // Set up the good wells either from all or from unpinned, unexcluded
      if (mask == NULL) {
	good.set_size(Copy.n_rows);
	for (size_t wIx = 0; wIx < Copy.n_rows; wIx++) {
	  good(wIx) = wIx;
	}
      }
      else {
	count = 0;
	for (size_t rowIx = rowStart; rowIx < rowEnd; rowIx++) {
	  for (size_t colIx = colStart; colIx < colEnd; colIx++) {
	    if (isOk(rowIx, colIx, *mask)) {
	      count++;
	    }
	  }
	}
	good.set_size(count);
	count = 0;
	for (size_t rowIx = rowStart; rowIx < rowEnd; rowIx++) {
	  for (size_t colIx = colStart; colIx < colEnd; colIx++) {
	    if (isOk(rowIx, colIx, *mask)) {
	      good(count++) = (rowIx - rowStart) * patchCols  + colIx - colStart;
	    }
	  }
	}
      }
      // Get the adjustments from smoothed line
      GetEigenProjection(Copy, good, 3, Proj);
      GetComparatorCorrection(Copy, good, Proj, rowEnd-rowStart, colEnd-colStart, Corr, 4, 4);
      ION_ASSERT(Proj.n_rows == Copy.n_rows && Proj.n_cols == Copy.n_cols,"Wrong dimensions.");
      ION_ASSERT(Corr.n_cols == Copy.n_cols,"Wrong dimensions.");

      // Note - iterating over the patch here, not the entire chip
      count = 0;

      for (size_t rowIx = 0; rowIx < patchRows; rowIx++) {
	for (size_t colIx = 0; colIx < patchCols; colIx++) {
	  size_t mod = rowIx % 4;
	  size_t ix = mod * patchCols + colIx;
	  ION_ASSERT(ix < Corr.n_rows, "not enough rows.");
	  ION_ASSERT(count < Copy.n_rows, "not enough rows.");
	  arma::Row<float> r = Corr.row(ix);
	  arma::Row<float> v = Copy.row(count);
	  arma::Row<float> x = v - r;
	  Copy.row(count) = x;
	  count++;
	}
      }
      if (nEigenSmooth > 0) {
      	GetEigenProjection(Copy, good, nEigenSmooth, Proj);
      	Copy=Proj;
      }
      // Copy back
      count = 0;
      for (size_t rowIx = rowStart; rowIx < rowEnd; rowIx++) {
	for (size_t colIx = colStart; colIx < colEnd; colIx++) {
	  for (size_t frameIx = 0; frameIx < imageFrames; frameIx++) {
	    ValAt(image, imageRows, imageCols, rowIx, colIx, frameIx) = (short) (Copy(count, frameIx) + .5);
	  }
	  count++;
	}
      }
    }
 
    catch(std::exception &e) {
      const char *w = e.what();
      ION_ABORT(w);
    }
  }			       

private:
  arma::Col<float> EVal;
  arma::Mat<float> Y, X, B, Proj, Cov, EVec, Diff, Copy,  Corr;
};

#endif // COMPARATORCLEAN_H
