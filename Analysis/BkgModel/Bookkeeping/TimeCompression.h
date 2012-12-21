/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TIMECOMPRESSION_H
#define TIMECOMPRESSION_H

#include <stdio.h>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "BkgMagicDefines.h"
#include "Serialization.h"
#include "IonErr.h"
#include "VectorMacros.h"
#include "Image.h"
#ifndef __CUDACC__   //cuda compiler is allergic to xmmintrin.h
    #include <xmmintrin.h>
#endif

// avx
/* #define TC_VEC_INC 8 */
/* union simdvec { */
/*   v8sf v; */
/*   float f[TC_VEC_INC]; */
/* }; */

// non-avx
#define TC_VEC_INC 4
union simdvec {
  v4sf v;
  float f[TC_VEC_INC];
};

class TimeCompression
{
  public:

  // time compression information
  std::vector<float> frameNumber;// for each averaged data point, the mean frame number
  std::vector<float> deltaFrame;    // the delta of each data point from the last
  std::vector<float> deltaFrameSeconds; // in seconds
  float frames_per_second;
  std::vector<int> frames_per_point;      // helper table used to construct average of incoming data
  
  float time_start; // when real points exist in the data we take
  float t0;
  int choose_time; // switch between different time compression schema
  
  // list of points and their weight for each vfc compressed point to convert into bg time
  std::vector<std::vector<std::pair<float,int> > > mVfcAverage; 
  // Keep the sum around so don't have to recalculate every time.
  std::vector<float> mVfcTotalWeight; 
  
  std::vector<float> mWeight;
  std::vector<float> mTotalWeight;
  std::vector<int> mVFCFlush;
  std::vector<float> mTimePoints;
  std::vector<int> origTimeStamps;
  TimeCompression();
  ~TimeCompression();
  void Allocate(int imgFrames);
  void CompressionFromFramesPerPoint();
  void DeAllocate();
  inline int Coverage(int s1, int e1, int s2, int e2) { 
    return  std::max(0, std::min(e1,e2) - std::max(s1,s2));
  }
	int npts(int npt);  // setter for npts
	inline int npts() { return (int)_npts; }         // getter for npts

  void SetUpTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg); // interface
  void SetUpOldTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void SetUpStandardTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void StandardFramesPerPoint(int imgFrames,float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void ExponentialFramesPerPoint(int imgFrames, float t_comp_start, int start_detailed_time, float geom_ratio);
  void HyperTime(int imgFrames, float t_comp_start, int start_detailed_time);
  void StandardAgain(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void HalfSpeedSampling(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void SetupConvertVfcTimeSegments(int frames, int *timestamps, int baseFrameRate, int frameStep);
  void WriteLinearTransformation(int frameStep);
  void RecompressTrace (float *fgPtr, float *tmp_shifted)
  {
    int frame = 0;
    // do not shift real bead wells at all
    // compress them from the frames_per_point structure in time-compression
    for (int npt=0;npt < npts();npt++)   // real data
    {
        float avg;
        avg=0.0;
        for (int i=0;i<frames_per_point[npt];i++)
        {
            avg += tmp_shifted[frame+i];
        }

        fgPtr[npt] = (avg/frames_per_point[npt]);
        frame+=frames_per_point[npt];
    }
  }


  /* Preferred vectorized version. */
  template <typename ShortVec> void ConvertVfcSegmentsOpt(size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd, 
                                                          size_t nRow, size_t nCol, size_t nFrame, ShortVec &source, int16_t *output) {
    size_t outFrameStep = (colEnd - colStart) * (rowEnd-rowStart);
    size_t wIx = 0;
    simdvec acc,dat,weight,zero;
    for (size_t i = 0; i < TC_VEC_INC; i++) {
      zero.f[i] = 0.0f;
    }
    size_t frameSizes[_npts];
    for (size_t n = 0; n < _npts; n++) {
      frameSizes[n] =  mVfcAverage[n].size();
    }
    size_t rowOffset = rowStart * nCol;
    for (size_t row = rowStart; row < rowEnd; row++) {
      for (size_t col = colStart; col < colEnd; col += TC_VEC_INC) {
        // in case we're not a multiple of TC_VEC_INC, just do the first n
        size_t cend = std::min(colEnd - col, (size_t) TC_VEC_INC);
        for (size_t frame = 0; frame < _npts; frame++) {
	  acc.v = zero.v;
          for (size_t n = 0; n < frameSizes[frame]; n++) {
            size_t offset = mVfcAverage[frame][n].second + rowOffset + col;
            for (size_t i = 0; i < cend; i++) {
              weight.f[i] = mVfcAverage[frame][n].first;
              dat.f[i] = source[offset++];
            }
	    acc.v = acc.v + (dat.v * weight.v);
          }
          size_t out = frame*outFrameStep + wIx; 
          for (size_t i = 0; i < cend; i++) {
            output[out++] =  acc.f[i];
          }
        }
        wIx+=cend;
      }
      rowOffset += nCol;
    }
  }

  /* Preferred vectorized version. */
  void ConvertVfcSegmentsOpt(size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd, 
			     size_t nRow, size_t nCol, size_t nFrame, int16_t **frameBuff, int16_t *output) {
    size_t outFrameStep = (colEnd - colStart) * (rowEnd-rowStart);
    size_t wIx = 0;
    simdvec acc,dat,weight,zero;
    for (size_t i = 0; i < TC_VEC_INC; i++) {
      zero.f[i] = 0.0f;
    }
    size_t frameSizes[_npts];
    for (size_t n = 0; n < _npts; n++) {
      frameSizes[n] =  mVfcAverage[n].size();
    }
    size_t rowOffset = rowStart * nCol;
    for (size_t row = rowStart; row < rowEnd; row++) {
      for (size_t col = colStart; col < colEnd; col += TC_VEC_INC) {
        // in case we're not a multiple of TC_VEC_INC, just do the first n
        size_t cend = std::min(colEnd - col, (size_t) TC_VEC_INC);
        for (size_t frame = 0; frame < _npts; frame++) {
	  acc.v = zero.v;
          for (size_t n = 0; n < frameSizes[frame]; n++) {
	    //            size_t offset = mVfcAverage[frame][n].second + rowOffset + col;
	    size_t offset = rowOffset + col;
	    int iframe = mVfcAverage[frame][n].second;
            for (size_t i = 0; i < cend; i++) {
              weight.f[i] = mVfcAverage[frame][n].first;
              dat.f[i] = frameBuff[iframe][offset++];
            }
	    acc.v = acc.v + (dat.v * weight.v);
          }
          size_t out = frame*outFrameStep + wIx; 
          for (size_t i = 0; i < cend; i++) {
            output[out++] =  acc.f[i];
          }
        }
        wIx+=cend;
      }
      rowOffset += nCol;
    }
  }


  /* don't use - deprecated experiment moving through memory in different way. */
  void ConvertVfcSegmentsFlat(size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd, 
                              size_t nRow, size_t nCol, size_t nFrame, short *source, uint16_t *output) {
    size_t outFrameStep = (colEnd - colStart) * (rowEnd-rowStart);
    int numWells = (colEnd - colStart) * (rowEnd - rowStart);
    float aligned_acc[numWells] __attribute__ ( (aligned (16) ) );
    memset(aligned_acc, 0, sizeof(float) * numWells);
    size_t wellOffset = rowStart * nCol + colStart;
    size_t frameStep = nRow * nCol;
    size_t wellEnd = (nFrame -1) * frameStep + rowEnd * nCol + colEnd;
    size_t numCols = colEnd - colStart;
    size_t frame = 0;
    size_t wellsSeen = 0;
    size_t row = rowStart;
    for (wellOffset = rowStart * nCol + colStart; wellOffset < wellEnd; wellOffset += nCol) {
      for (size_t col = 0; col < numCols; col++) {
        aligned_acc[wellsSeen] += source[wellOffset + col] * mWeight[frame];
        if (mVFCFlush[frame]) {
          output[wellsSeen] = round(aligned_acc[wellsSeen]/mTotalWeight[frame]);
          aligned_acc[wellsSeen] = source[wellOffset + col] * (1 - mWeight[frame]);
        }
        wellsSeen++;
      }
      if (wellsSeen == outFrameStep) {
        if (mVFCFlush[frame]) { 
          output += outFrameStep;
        }
        frame++;
        wellOffset = frame * frameStep + (rowStart) * nCol + colStart - nCol;
        wellsSeen = 0;
        row = rowStart;
      }
      else {
        row++;
      }
    }
  }
  
  void ReportVfcConversion(int frames, int *timestamps, int baseFrameRate, std::ostream &out);
  size_t GetTimeCompressedFrames() { return _npts; }
  template<typename T>
    static void Interpolate(float *delta1, T *v1, int n1, float *delta2, float *v2, int n2);


 private:
  size_t _npts;           // number of data points after time compression

  // Serialization section
  friend class boost::serialization::access;
  template<typename Archive>
    void serialize(Archive& ar, const unsigned version) {
    ar & 
      frameNumber &
      deltaFrame &
      deltaFrameSeconds &
      frames_per_second &
      frames_per_point &
      _npts &
      time_start &
      t0 &
      choose_time &
      mVfcAverage &
      mVfcTotalWeight &
      mWeight &
      mTotalWeight &
      mVFCFlush &
      mTimePoints;
  }

};

/**
 * Given cumulative times in t1, and values v1 in vectors of length n1
 * 0 < t1[0] < t1[1] < ... < t1[n1]
 * v1[0] is associated with the time as t1[0]/2
 * v1[i] is associated with the time at (t1[i-1] + t1[i])/2, i>0
 * similarly time in t2, a vector of length n2, 
 * return interpolated values as floats in v2.
 * t1 and t2 must be in the same units, e.g. seconds
 **/
template<typename T>
inline void TimeCompression::Interpolate(float *t1_end, T *v1, int n1, float *t2_end, float *v2, int n2){

  // set time to midpoints
  std::vector<float> t1(n1, 0);
  t1[0] = t1_end[0] * 0.5f;
  for (int i=1; i<n1; i++)
    t1[i] = (t1_end[i-1] + t1_end[i]) * 0.5f;

  std::vector<float> t2(n2, 0);
  t2[0] = t2_end[0] * 0.5f;
  for (int i=1; i<n2; i++)
    t2[i] = (t2_end[i-1] + t2_end[i]) * 0.5f;

  float oldTime1 = 0;
  float time1 = t1[0];
  float time2 = t2[0];
  
  int i1 = 0;
  int i2 = 0;

  while ( i2 < n2 )
  {
    if (time2 < time1) {
      if (i1 == 0) {
	// cumulative time2 is less than time1
	// extrapolate values to match v1[0]
	v2[i2] = v1[0];
      }
      else {
	// cumulative time2 is bracketed by [oldTime1 time1)
	// linearly interpolate values based on time
	float f = (time2 - oldTime1)/(time1 - oldTime1);;
	v2[i2] = (v1[i1] - v1[i1-1]) * f + v1[i1-1];
      }
      // increment time for t2 and v2
      ++i2;
      if (i2 == n2) {
	break;
      }
      time2 = t2[i2];
    }
    else {
      if (i1 == (n1-1)){
	// cumulative time2 is >= than max possible time1
	// extrapolate values to be the final v1
	v2[i2] = v1[n1-1];
	// increment time for t2 and v2
	i2++;
	if (i2 == n2) {
	  break;
	}
	time2 = t2[i2];
      }
      else {
	// get the next cumulative time1
	++i1;
	oldTime1 = time1;
	time1 = t1[i1];
      }
    }
  }
}

#endif // TIMECOMPRESSION_H
