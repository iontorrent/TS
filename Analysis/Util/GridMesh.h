/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GRIDMESH_H
#define GRIDMESH_H

#include <vector>
#include <math.h>
#include <assert.h>
#include <iostream>

/** 
 * Container to overlay a grid with regions and provide easy access to
 * items by bucket. Indexing in 0 based and conceptually 0,0 is the lower 
 * left of the grid.
 */
template <class T, class C=int>
class GridMesh {

public:

  void Init(C nRow, C nCol, C nRowStep, C nColStep) {

    assert(nRow > 0);
    assert(nCol > 0);
    assert(nRowStep > 0);
    assert(nColStep > 0);

    mRow = nRow;
    mCol = nCol;
    mRowStep = nRowStep;
    mColStep = nColStep;
    mRowBin = static_cast<C>(ceil(1.0 * mRow / mRowStep));
    mColBin = static_cast<C>(ceil(1.0 * mCol / mColStep));
    mBins.clear();
    mBins.resize(mRowBin*mColBin);
    //    std::cout  << "Created GridMesh with " << mBins.size() << " bins with rowStep: " << mRowStep << " and colStep: " << mColStep << std::endl;
  }

  GridMesh() {
    mRow = 0;
    mCol = 0;
    mRowStep = 0;
    mColStep = 0;
    mRowBin = 0;
    mColBin = 0;
  }
  
  /** 
   * Constructor giving the total number of rows and columns of the items to map
   * and the number of bins in the row and column dimension desired. So for example
   * GridMesh<double> grid(100, 100, 10, 10); creates a 10x10 mesh of bins overlaying
   * 100x100 individual points.
   */
  GridMesh(C nRow, C nCol, C nRowStep, C nColStep) {
    Init(nRow, nCol, nRowStep, nColStep);
  }

  /** 
   * Clean out for new init().
   */
  void Clear() {
    mRow = mCol = mRowBin = mColBin = mRowStep = mColStep = 0;
    mBins.resize(0);
  }

  /** Convert row,col coordinates into an index in the mBins vector. */
  size_t XyToIndex(C rowBin, C colBin) const { return (size_t)(mColBin * rowBin + colBin); }
  
  /** Convert the index of a bin in the mBins vector into a row and a column. */
  void IndexToXY(C index, C &rowBin, C &colBin) {
    colBin = index % mColBin;
    rowBin = index / mColBin;
  }

  /** Return a reference to the item in the bin at rowBin,colBin */
  T& GetItem(C rowBin, C colBin) { return mBins[XyToIndex(rowBin, colBin)]; }

  /** Return a reference to the item in the bin at rowBin,colBin */
  T& GetItemByRowCol(C row, C col) { return mBins[XyToIndex(row/mRowStep, col/mColStep)]; }

  /** Return a reference to the item in the bin at rowBin,colBin */
  const T& GetItemByRowCol(C row, C col) const { return mBins[XyToIndex(row/mRowStep, col/mColStep)]; }

  /** Return a reference to the item in the bin at rowBin,colBin */
  T& GetItem(C binIx) { return mBins[binIx]; }

  /** Return a reference to the item in the bin at rowBin,colBin */
  const T& GetItem(C binIx) const { return mBins[binIx]; }

  /** Get the bin for a well index. */
  size_t GetBin(C index) { return (XyToIndex( (index/mCol)/ mRowStep, (index % mCol)/ mColStep)); }

  /** Get the bin for a well index. */
  size_t GetBin(C row, C col) { return (XyToIndex( (row)/ mRowStep, (col)/ mColStep)); }

  /**
   * Fill in the values vector with the T from the bins surrounding row,col
   * up to bin distance binDist (minimum 1). Each values entry has a corresponding
   * distance entry in the distances vector which is the distance of row,col to the
   * center of the bucket containing the T value in values.
   */
  void GetClosestNeighbors(C row, C col, C binDist,
			   std::vector<double> &distances,
			   std::vector<T *> &values) {
    C rowBucket = row / mRowStep;
    C colBucket = col / mColStep;
    distances.clear();
    values.clear();
    // search out the 6 neigboring 
    C rowIx = rowBucket >= binDist ? rowBucket - binDist : 0;
    for (; rowIx <= rowBucket + binDist && rowIx < mRowBin; rowIx++) {
      C colIx = colBucket >= binDist ? colBucket - binDist : 0;
      for (; colIx <= colBucket + binDist && colIx < mColBin; colIx++) {
	// Ingore items off the edge of the grid
	if (rowIx >= 0 && rowIx < mRowBin && colIx >= 0 && colIx < mColBin) {
	  // Compare from current row to middle of bin
	  double rowDist = static_cast<double>(row) - (rowIx * static_cast<double>(mRowStep) + mRowStep/2);
	  double colDist = static_cast<double>(col) - (colIx * static_cast<double>(mColStep) + mColStep/2);
	  double dist = sqrt(rowDist * rowDist + colDist * colDist);
	  distances.push_back(dist);
	  values.push_back(&GetItem(rowIx, colIx));
	}
      }
    }
  }

  /**
   * Fill in the values vector with the T from the bins surrounding row,col
   * up to bin distance binDist (minimum 1). Each values entry has a corresponding
   * distance entry in the distances vector which is the distance of row,col to the
   * center of the bucket containing the T value in values.
   */
void GetClosestNeighborsWithinGrid(C row, C col, C binDist, C rowMod, C colMod,
                                     std::vector<double> &distances,
                                     std::vector<T *> &values) {
    C rowBucket = row / mRowStep;
    C colBucket = col / mColStep;
    distances.clear();
    values.clear();
    // search out the 6 neigboring 
    C rowIx = rowBucket >= binDist ? rowBucket - binDist : 0;
    for (; rowIx <= rowBucket + binDist && rowIx < mRowBin; rowIx++) {
      C colIx = colBucket >= binDist ? colBucket - binDist : 0;
      for (; colIx <= colBucket + binDist && colIx < mColBin; colIx++) {
	// Ingore items off the edge of the grid
	if (rowIx >= 0 && rowIx < mRowBin && colIx >= 0 && colIx < mColBin) {
          C candRow = (rowIx * mRowStep);
          C candCol = (colIx * mColStep);
          if (row / rowMod == candRow / rowMod &&
              col / colMod == candCol / colMod) {
            // Compare from current row to middle of bin
            double rowDist = static_cast<double>(row) - (rowIx * static_cast<double>(mRowStep) + mRowStep/2);
            double colDist = static_cast<double>(col) - (colIx * static_cast<double>(mColStep) + mColStep/2);
            double dist = sqrt(rowDist * rowDist + colDist * colDist);
            distances.push_back(dist);
            values.push_back(&GetItem(rowIx, colIx));
          }
          /* else { */
          /*   std::cout << "Outside of grid for: " << row << "," << col << " with " << candRow << "," << candCol << std::endl; */
          /* } */
        }
      }
    }
  }


  size_t GetNumBin() const { return (size_t)(mRowBin * mColBin); }

  void GetBinCoords(C regionIdx, C &rowStart, C &rowEnd, C &colStart, C &colEnd) {
    rowStart = (regionIdx / mColBin) * mRowStep;
    rowEnd = std::min((regionIdx / mColBin + 1) * mRowStep, mRow);
    colStart = (regionIdx % mColBin) * mColStep;
    colEnd = std::min(((regionIdx % mColBin) + 1) * mColStep, mCol);
  }

  void SetValue(const T& val, int rowBin, int colBin) {
    mBins[XyToIndex(rowBin, colBin)] = val;
  }

  size_t GetRow() const { return mRow; }
  size_t GetCol() const { return mCol; }
  size_t GetRowStep() const { return mRowStep; } 
  size_t GetColStep() const { return mColStep; }
  size_t GetRowBin() const { return mRowBin; }
  size_t GetColBin() const { return mColBin; }

 public:

  C mRow;       ///< Number of rows
  C mCol;       ///< Number of columns
  C mRowBin;    ///< Number of bins in row dimension
  C mColBin;    ///< Number of bins in column dimension
  C mRowStep;   ///< Number of individual rows in each bin
  C mColStep;   ///< Number of individual columns in each bin
  std::vector<T> mBins; ///< Collection of T associated with each bin mBins.size() == mRowBin *mColBin

};


#endif // GRIDMESH_H
