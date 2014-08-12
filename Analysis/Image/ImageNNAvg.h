/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGENNAVG_H
#define IMAGENNAVG_H

#include <stdint.h>
#include "Mask.h"
#include "Utils.h"
/** 
 * Calculate nearest neighbor averages for wells in chip image using
 * efficient algorithm using cumulative sum to calculate average.
 */
class ImageNNAvg {

 public:

  /** Basic constructor */
  ImageNNAvg() { 
    InitializeClean();
  }

  /**
     Constructor with dimensions
     @param - n_rows number of rows in chip
     @param - n_cols number of columns in chip
     @param - n_frames number of frames in the image
  */
  ImageNNAvg(int n_rows, int n_cols, int n_frames);

  void FreeNNAvg() { FREEZ(&m_nn_avg); }
  void FreeNumGoodWells() { FREEZ(&m_num_good_wells); }

  /** Destructor */
  ~ImageNNAvg() { Cleanup();  }

  /** Initialize buffers */
  void Init(int n_rows, int n_cols, int n_frames) {
    Alloc(n_rows, n_cols, n_frames);
  }

  /**
    Algorithmic trick - Instead of recalculating the average for each well we're going
    to calculate the cumulative sum of every sub matrix and then use the difference
    in the cumulative sums to get sum for a region we want to average over. 
   
    Let M(x,y) be our original frame matrix and C(x,y) be the cumulative sum matrix
    then C(x,y) = M(x,y) + C(x-1,y) + C(x,y-1) - C(x-1,y-1)

    Original matrix  Cumulative Sum matrix
    9  10  11  12    15  33  54  78
    5   6   7   8     6  14  24  36
    1   2   3   4     1   3   6  10

    Then once we want the average for a region, say x = 1..2 and y = 1..2
    Avg({1:2},{1:2}) = C(2,2) - C(0,2) - C(2,0) + C(0,0)
    = 54 - 6 - 15 + 1
    = 34
    Which does equal 10 + 11 + 6 + 7 = 34 from the original matrix without having to iterate
    over and over to get the average for each region.

    Only additional issue is that we need to zero out pinned and bad wells and thus
    keep track of how many wells are actually used in a region for the cumulative sum.
    we'll do the same trick now just keeping track of good wells in a submatrix

     @param image - pointer to the image of data in usual frame, row major order
     @param mask - usual beadfind mask with bad wells marked
     @param bad_wells - array with non-zero entries for bad wells that shouldn't be used
  */
  void CalcCumulativeSum(const short *__restrict image, const Mask *mask, 
                         const char *__restrict bad_wells);

  void CalcCumulativeSum(const int *__restrict image, const Mask *mask, 
                         const char *__restrict bad_wells);
  /**
     Calculate the average of the non-bad nearest neighbors wells +/-
     num_row_neighbors and +/- num_col_neighbors for all wells. Use 0
     if there are no good wells in a region. row_step and col_step
     indicate regions within which to calculate average (e.g 100,100
     for thumbnail). Column of well is not used in average to avoid 
     column flicker issues.

     @param row_step - row modulous within which to calculate average (100 for thumbnail)
     @param col_step - col modulous within which to calculate average (100 for thumbnail)
     @param num_row_neighbors - window above and below well to include in average
     @param num_col_neighbors - window left and and right of well to use
     @param trace_min - array to put the minimum value in for each well (preallocated)
     @param trace_min_fram - array to put the minimum frame in for each well (preallocated)
     @param image - pointer to the image of data
   */
  void CalcNNAvgAndMinFrame(int row_step, int col_step, 
                            int num_row_neighbors, int num_col_neighbors,
                            float *__restrict trace_min, int *__restrict trace_min_frame,
                            const short *__restrict image,
                            bool replace_with_val);

  /**
     Return a pointer to the nn averaged data in same order as usual image
     with frame, row major order. The memory is owned by the object, don't try to free it
   */
  const float *GetNNAvgImagePtr() { return m_nn_avg; }
  const int64_t *GetNNCumSumPtr() { return m_cum_sum; }
  const int * GetNumGoodWellsPtr() { return  m_num_good_wells; }

  /**
     Get avg for individual well and frame. 
     @param - row on chip
     @param - col on chip
     @param - frame in image
     @return - NN avg for well and frame
  */
  inline float GetNNAvg(int row, int col, int frame) { 
    return m_nn_avg[row * m_num_cols + col + (m_num_rows * m_num_cols * frame)]; 
  }

  inline int GetRows() { return m_num_rows; }
  inline int GetCols() { return m_num_cols; }
  inline int GetFrames() { return m_num_frames; }
  inline void SetGainMinMult(int mult) { m_gain_min_mult = mult; }

  /// Cleanup the memory previously allocated
  void Cleanup();

 private :
  ION_DISABLE_COPY_ASSIGN(ImageNNAvg)

  /// Initalize everything to zeros
  void InitializeClean() {
    m_cum_sum = NULL;
    m_num_good_wells = NULL;
    m_nn_avg = NULL; 
    m_num_rows = m_num_cols = m_num_frames = m_cum_sum_size = 0;
  }

  /// Allocate the memory used
  void Alloc(int n_rows, int n_cols, int n_frames);

  /// Dimensions of chip
  int m_num_rows, m_num_cols, m_num_frames;
  /// Size of the cumulative sum array
  size_t m_cum_sum_size;
  int64_t *__restrict m_cum_sum; ///< Cumulative sum of image data frame major order for each frame individually
  int *__restrict m_num_good_wells; ///< Cumulative sum of good wells in region
  float *__restrict m_nn_avg; ///< Average for specified NN region
  int  m_gain_min_mult;
};

#endif // IMAGENNAVG_H
