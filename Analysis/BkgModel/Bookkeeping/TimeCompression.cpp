/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <algorithm>
#include "TimeCompression.h"
#include "IonErr.h"
#include "VectorMacros.h"
using namespace std;

TimeCompression::TimeCompression()
{
  _npts = 0;
  _stdFrames = 0;
  _etfFrames = 0;
  _uncompressedFrames = 0;
  time_start = 0.0-0.1;  // every data point real by default
  choose_time = 0; // default time
  frames_per_second = 15.0f;
  t0 = -1.0f;
  etf_tail_start_frame = 0;
  _standardCompression = true;
  //  frames_per_second = 16.39344f;
}
 
void TimeCompression::DeAllocate()
{
  frameNumber.clear();
  deltaFrame.clear();
  deltaFrameSeconds.clear();
  frames_per_point.clear();
  mTimePoints.clear();

  // standard timing compression
  std_frameNumber.clear();
  std_deltaFrame.clear();
  std_deltaFrameSeconds.clear();
  std_frames_per_point.clear();
  std_interpolate_mult.clear();
  std_interpolate_frame.clear();
  std_mTimePoints.clear();

  // exponentail tail fit timing compression
  etf_frameNumber.clear();
  etf_deltaFrame.clear();
  etf_deltaFrameSeconds.clear();
  etf_frames_per_point.clear();
  etf_interpolate_mult.clear();
  etf_interpolate_frame.clear();
  etf_mTimePoints.clear();
}

TimeCompression::~TimeCompression()
{
    DeAllocate();
}

void TimeCompression::Allocate(int imgFrames)
{
  // just allocate worst-case arrays
  frameNumber.resize(imgFrames);
  deltaFrame.resize(imgFrames);
  deltaFrameSeconds.resize(imgFrames); // save recalculation on this
  frames_per_point.resize(imgFrames);
  mTimePoints.resize(imgFrames);

  // standard timing compression
  std_frameNumber.resize(imgFrames);
  std_deltaFrame.resize(imgFrames);
  std_deltaFrameSeconds.resize(imgFrames);
  std_frames_per_point.resize(imgFrames);
  std_interpolate_mult.resize(imgFrames);
  std_interpolate_frame.resize(imgFrames);
  std_mTimePoints.resize(imgFrames);

  // exponentail tail fit timing compression
  etf_frameNumber.resize(imgFrames);
  etf_deltaFrame.resize(imgFrames);
  etf_deltaFrameSeconds.resize(imgFrames);
  etf_frames_per_point.resize(imgFrames);
  etf_interpolate_mult.resize(imgFrames);
  etf_interpolate_frame.resize(imgFrames);
  etf_mTimePoints.resize(imgFrames);
}

// placeholder:  do time compression correctly
void TimeCompression::SetUpTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{

  SetUncompressedFrames(imgFrames);

  DeAllocate();
  Allocate(imgFrames);
  switch(choose_time){
    case 2:
       // generate both ETF and standard compression but se the compression to ETF
       SetUpETFCompression(t_comp_start, start_detailed_time,stop_detailed_time, left_avg);
       SetUpStandardCompression(t_comp_start, start_detailed_time,stop_detailed_time, left_avg);
       UseETFCompression();
       break;
    default:
       SetUpStandardCompression(t_comp_start, start_detailed_time,stop_detailed_time, left_avg);
       UseStandardCompression();
  }
}

void TimeCompression::SetUpETFCompression(float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg) 
{
  ETFCompatible(_uncompressedFrames,t_comp_start, start_detailed_time,stop_detailed_time, left_avg);
  SetUpInterpolationVectorsForETF(_uncompressedFrames);
  CompressionFromETFFramesPerPoint();
}

void TimeCompression::SetUpStandardCompression(float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg) 
{
  StandardAgain(_uncompressedFrames,t_comp_start, start_detailed_time,stop_detailed_time, left_avg);
  SetUpInterpolationVectorsForStd(_uncompressedFrames);
  CompressionFromStdFramesPerPoint();
}


// imgFrames, t_compression_start, -5, 16, 5
void TimeCompression::SetUpOldTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  DeAllocate(); // just in case!
  Allocate(imgFrames);
  
  // starting point of non-averaged data
  int il = (int) t_comp_start+start_detailed_time;
  // stopping point of non-averaged data..intermediate average points
  int ir = (int)(t_comp_start+stop_detailed_time);

  if (il < 0) il = 0;
  if (ir > imgFrames) ir = imgFrames;

  int npt=0;
  float fnum_sum = 0.0;
  int fnum_num = 0;
  int irpts = 2;
  float last_fnum = 0.0;
  bool capture = false;

  for (int i=0;i < imgFrames;i++)
  {
    fnum_sum += i;
    fnum_num++;

    capture = false;

    if (i < il)
    {
      if ((fnum_num >= left_avg) || ((i+1) >= il))
        capture = true;
    }
    else if ((i >= il) && (i < ir))
    {
      capture = true;
    }
    else if (i >= ir)
    {
      if (fnum_num >= irpts)
      {
        capture = true;
        irpts += 4;
        if (irpts > (MAX_COMPRESSED_FRAMES-1))
	  irpts = (MAX_COMPRESSED_FRAMES-1);
      }
    }

    if (i == (imgFrames - 1))
      capture = true;

    if (capture == true)
    {
      frameNumber[npt] = (fnum_sum / fnum_num);
      deltaFrame[npt] = frameNumber[npt] - last_fnum;
      deltaFrameSeconds[npt] = deltaFrame[npt]/frames_per_second; // cache for hydrogen generation
      frames_per_point[npt] = fnum_num;
      last_fnum = frameNumber[npt];
      npt++;
      fnum_sum = 0.0;
      fnum_num = 0;
    }
  }

  npts(npt);
}

int CompressOneStep(int &cur_sum, int step)
{
  assert(step > 0);
  int tstep = step;
  cur_sum -= tstep;
  if (cur_sum<0)
    tstep = step+cur_sum;
  return(tstep);
}

void TimeCompression::StandardAgain(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  int i_start = (int) t_comp_start+start_detailed_time;
  int i_done = min((int)(t_comp_start+stop_detailed_time),imgFrames);
  // go to i_start compressing aggressively
  int cur_pt =0;
  int cur_sum=i_start;
  int i=0;
  for (; (i<imgFrames) && (cur_sum>0); i++)
  {
    std_frames_per_point[cur_pt] = CompressOneStep(cur_sum,left_avg);
    cur_pt++;
  }
  // now do the middle time when we are at full detail
  cur_sum = i_done-i_start;
  for (; (i<imgFrames) && (cur_sum>0); i++)
  {
    std_frames_per_point[cur_pt] = CompressOneStep(cur_sum,1);
    cur_pt++;
  }
  // finally compress the tail very heavily indeed
  int try_step = 2;
  cur_sum = imgFrames-i_done;
 for (; (i<imgFrames) && (cur_sum>0); i++)
  {
    std_frames_per_point[cur_pt] = CompressOneStep(cur_sum,try_step);
    cur_pt++;
    try_step += 4;
    //try_step *=2;
  }
 SetStandardFrames(cur_pt);
}

void TimeCompression::ETFCompatible(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  int i_start = (int) t_comp_start+start_detailed_time;
  int i_done = min((int)(t_comp_start+stop_detailed_time),imgFrames);
  // go to i_start compressing aggressively
  int cur_pt =0;
  int cur_sum=i_start;
  int i=0;
  for (; (i<imgFrames) & (cur_sum>0); i++)
  {
    etf_frames_per_point[cur_pt] = CompressOneStep(cur_sum,left_avg);
    cur_pt++;
  }
  // now do the middle time when we are at full detail
  cur_sum = i_done-i_start;
  for (; (i<imgFrames) & (cur_sum>0); i++)
  {
    etf_frames_per_point[cur_pt] = CompressOneStep(cur_sum,1);
    cur_pt++;
  }
  
  // set the start of the tail
  etf_tail_start_frame = cur_pt;
  // finally compress the tail very heavily indeed
  int try_step = 1;
  float try_step_float = 1.0f;
  float try_step_inc = 0.20f;
  cur_sum = imgFrames-i_done;
 for (; (i<imgFrames) & (cur_sum>0); i++)
  {
    etf_frames_per_point[cur_pt] = CompressOneStep(cur_sum,try_step);
    cur_pt++;
    try_step_float += try_step_inc;
    try_step = (int)(try_step_float+0.5f);
    if (try_step > 8)
       try_step = 8;
  }
  SetETFFrames(cur_pt);
}



void TimeCompression::HyperTime(int imgFrames, float t_comp_start, int start_detailed_time)
{
  // super aggressive compression
  int cur_pt =0;
  int cur_sum=imgFrames;
  int i_start = (int) t_comp_start+start_detailed_time;
  if (i_start<1) i_start = 1;
  frames_per_point[cur_pt] = i_start;
  cur_sum -= frames_per_point[cur_pt];
  cur_pt++;
  for (int i=0; (i<3) & (cur_sum>0); i++)
  {
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,1);
    cur_pt++;
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,2);
    cur_pt++;
  }
  for (int i=0; (i<3) & (cur_sum>0); i++)
  {
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,2);
    cur_pt++;
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,3);
    cur_pt++;
  }
  npts(cur_pt);
}

/*void TimeCompression::StandardFramesPerPoint(int imgFrames,float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  // starting point of non-averaged data
  int il = (int) t_comp_start+start_detailed_time;
  // stopping point of non-averaged data..intermediate average points
  int ir = (int)(t_comp_start+stop_detailed_time);

  if (il < 0) il = 0;
  if (ir > imgFrames) ir = imgFrames;

  int npt=0;
  int fnum_num = 0;
  int irpts = 2;

  for (int i=0;i < imgFrames;i++)
  {
    fnum_num++;
    bool capture = false;

    if (i < il)
    {
      if ((fnum_num >= left_avg) || ((i+1) >= il))
        capture = true;
    }
    else if ((i >= il) && (i < ir))
    {
      capture = true;
    }
    else if (i >= ir)
    {
      if (fnum_num >= irpts)
      {
        capture = true;
        irpts += 4;
        if (irpts > (MAX_COMPRESSED_FRAMES-1))
	  irpts = (MAX_COMPRESSED_FRAMES-1);
      }
    }

    if (i == (imgFrames - 1))
      capture = true;

    if (capture == true)
    {
      frames_per_point[npt] = fnum_num;
      npt++;
      fnum_num = 0;
    }
  }
  npts(npt);
}*/

void TimeCompression::CompressionFromStdFramesPerPoint()
{
  // frames_per_point contains the time compression information
  // sum (frames_per_point) = imgFrames
  // npts already defined
  // allocation already assumed
  // apply time compression to 0:(imgFrames-1)
  size_t npt = 0;
  std_mTimePoints.resize(_stdFrames);
  // do not shift real bead wells at all
  int cur_frame=0;
  float  last_fnum = 0.0;

  for (;npt < _stdFrames; npt++)   // real data
  {
    float avg;
    avg=0.0;

    for (int i=0; i<std_frames_per_point[npt]; i++)
    {
      avg += i + cur_frame;
    }
    std_frameNumber[npt] = (avg / std_frames_per_point[npt]);

    std_deltaFrame[npt] = std_frameNumber[npt] - last_fnum;
    std_deltaFrameSeconds[npt] = std_deltaFrame[npt]/frames_per_second; // cache for hydrogen generation

    last_fnum = std_frameNumber[npt];            
    cur_frame+=std_frames_per_point[npt];
    std_mTimePoints[npt] = (float) (cur_frame-1) / frames_per_second;
  }
}

void TimeCompression::CompressionFromETFFramesPerPoint()
{
  // frames_per_point contains the time compression information
  // sum (frames_per_point) = imgFrames
  // npts already defined
  // allocation already assumed
  // apply time compression to 0:(imgFrames-1)
  size_t npt = 0;
  etf_mTimePoints.resize(_etfFrames);
  
  // do not shift real bead wells at all
  int cur_frame=0;
  float  last_fnum = 0.0;

  for (; npt < _etfFrames; npt++)   // real data
  {
    float avg;
    avg=0.0;

    for (int i=0;i<etf_frames_per_point[npt];i++)
    {
      avg += i+cur_frame;
    }
    etf_frameNumber[npt] = (avg / etf_frames_per_point[npt]);

    etf_deltaFrame[npt] = etf_frameNumber[npt] - last_fnum;
    etf_deltaFrameSeconds[npt] = etf_deltaFrame[npt]/frames_per_second; // cache for hydrogen generation

    last_fnum = etf_frameNumber[npt];            
    cur_frame += etf_frames_per_point[npt];
    etf_mTimePoints[npt] = (float) (cur_frame-1) / frames_per_second;
  }
}

void TimeCompression::SetUpInterpolationVectorsForETF(int imgFrames) {

  int j=0; // tracking actual uncompressed frame number
  int frames = static_cast<int>(_etfFrames);
  for (int i=0; i<frames; ++i) {
    for (int pt=1; pt<=etf_frames_per_point[i]; ++pt) {
      etf_interpolate_frame[j] = i;      
      etf_interpolate_mult[j++] = 
        (static_cast<float>(etf_frames_per_point[i]) - static_cast<float>(pt)) / static_cast<float>(etf_frames_per_point[i]);      
    }
  }
  assert(imgFrames == j);
}

void TimeCompression::SetUpInterpolationVectorsForStd(int imgFrames) {

  int j=0; // tracking actual uncompressed frame number
  int frames = static_cast<int>(_stdFrames);
  for (int i=0; i<frames; ++i) {
    for (int pt=1; pt<=std_frames_per_point[i]; ++pt) {
      std_interpolate_frame[j] = i;
      std_interpolate_mult[j++] = 
        (static_cast<float>(std_frames_per_point[i]) - static_cast<float>(pt)) / static_cast<float>(std_frames_per_point[i]);
    }
  }
  assert(imgFrames == j);
}


void TimeCompression::UseStandardCompression() {
  _standardCompression = true;
  npts(_stdFrames);
  frameNumber =  std_frameNumber;
  deltaFrame = std_deltaFrame;
  deltaFrameSeconds = std_deltaFrameSeconds;
  frames_per_point = std_frames_per_point;
  mTimePoints = std_mTimePoints;
}


void TimeCompression::UseETFCompression() {
  _standardCompression = false;
  npts(_etfFrames);
  frameNumber =  etf_frameNumber;
  deltaFrame = etf_deltaFrame;
  deltaFrameSeconds = etf_deltaFrameSeconds;
  frames_per_point = etf_frames_per_point;
  mTimePoints = etf_mTimePoints;
}

void TimeCompression::SetStandardFrames(int npt){
  _stdFrames = (size_t)npt;
  std_frameNumber.resize(_stdFrames);
  std_deltaFrame.resize(_stdFrames);
  std_deltaFrameSeconds.resize(_stdFrames);
  std_frames_per_point.resize(_stdFrames);
  std_mTimePoints.resize(_stdFrames);
}

void TimeCompression::SetETFFrames(int npt){
  _etfFrames = (size_t)npt;
  etf_frameNumber.resize(_etfFrames);
  etf_deltaFrame.resize(_etfFrames);
  etf_deltaFrameSeconds.resize(_etfFrames);
  etf_frames_per_point.resize(_etfFrames);
  etf_mTimePoints.resize(_etfFrames);
}

void TimeCompression::npts(int npt){
  _npts = (size_t)npt;
  frameNumber.resize(_npts);
  deltaFrame.resize(_npts);
  deltaFrameSeconds.resize(_npts);
  frames_per_point.resize(_npts);
  mTimePoints.resize(_npts);
}


//////////////////////////////////////////////////////////////
//
// Synchronized DAT routines
//
//////////////////////////////////////////////////////////////

void TimeCompression::SetupConvertVfcTimeSegments(int frames, int *timestamps, int baseFrameRate, int frameStep) {
  std::vector<int> vfc_time(&timestamps[0], &timestamps[0] + frames);
  // std::vector<int> vfc_time(frames);
  // float last_timestamp = 0.0;
  // for ( int j=0; j < frames; j++) {
  //     vfc_time[j] = ( timestamps[j] + last_timestamp ) /2.0;
  //     last_timestamp = timestamps[j];
  //   }
          origTimeStamps = vfc_time;
        
        /* Cumulative sum of time bg alg wants. */
  std::vector<int> bg_time(_npts, 0);
  for (size_t i = 0;  i < (size_t)_npts; i++) {
    bg_time[i] = mTimePoints[i] * 1000.0f;
  }
  mVfcAverage.resize(_npts);
  mVfcTotalWeight.resize(_npts);
  
  fill(mVfcTotalWeight.begin(), mVfcTotalWeight.end(), 0.0f);
  /* Convert to vector for lower_bound() function. */
  std::vector<int>::iterator tstart;

  /* Loop through each point background algoritm wants and figure out the vfc compressed
   * points coming from datacollect that should be averaged together to best etimate that 
   * point. */
  for (size_t i = 0; i < (size_t)_npts; i++) {
    int bg_start = i == 0 ? 0 : bg_time[i-1];
    tstart = std::upper_bound(vfc_time.begin(), vfc_time.end(), bg_start);
    for (; tstart < vfc_time.end(); ++tstart) {
      int vfc_start = (tstart == vfc_time.begin()) ? 0 : (*(tstart-1));
      int overlap = Coverage(bg_start, bg_time[i], vfc_start, *tstart);
      if(overlap > 0) {
        /* weight is the portion of this segment covered by background point. */
        float weight = (float)overlap / (*tstart - vfc_start);
        std::pair<float,int> p;
        p.first = weight;
        p.second = (tstart - vfc_time.begin());
        mVfcAverage[i].push_back(p);
        mVfcTotalWeight[i] += weight;
      }
      else {
        break;
      }
    }
  }
  for (size_t i = 0; i < mVfcAverage.size(); i++) {
    for (size_t j = 0; j < mVfcAverage[i].size(); j++) {
      mVfcAverage[i][j].first /= mVfcTotalWeight[i];
    }
  }

}


void TimeCompression::WriteLinearTransformation(int frameStep) {
  std::vector<int> bg_time(_npts, 0);
  for (size_t i = 0;  i < (size_t)_npts; i++) {
    bg_time[i] = mTimePoints[i] * 1000.0f;
  }
  cout << "Averaging: " << endl;
  cout << "frame\tbg_time\tsec\tvfc_st\tframes" << endl;
  int numFrames = 0;
  for (size_t i = 0; i < mVfcAverage.size(); i++) {
    double last = i == 0 ? 0 : bg_time[i-1];
    float stime = numFrames / frames_per_second;
    float etime = (frames_per_point[i] + numFrames) / frames_per_second;
    numFrames += frames_per_point[i];
    cout << i << "\t" << last << "," << bg_time[i] << "\t" << stime << "," << etime << "\t" << mVfcAverage[i].front().second/frameStep << "," << mVfcAverage[i].back().second/frameStep << "\t";
    for (size_t j = 0; j < mVfcAverage[i].size(); j++) {
      cout << mVfcAverage[i][j].second / frameStep << ",";
    }
    cout << "\t";
    for (size_t j = 0; j < mVfcAverage[i].size(); j++) {
      cout << origTimeStamps[mVfcAverage[i][j].second / frameStep] << ",";
    }
    cout << "\t";
    for (size_t j = 0; j < mVfcAverage[i].size(); j++) {
      cout << mVfcAverage[i][j].first << ",";
    }
    cout << endl;
  }
  cout << "Done." << endl;
}

void TimeCompression::ReportVfcConversion(int frames, int *timestamps, int baseFrameRate, std::ostream &out) {
  int cur_time = 0;
  for (size_t i = 0; i < mVfcTotalWeight.size(); i++) {
    out << i << "\t" << frames_per_point[i] << "\t" << deltaFrame[i] << "\t" << deltaFrameSeconds[i] << "\t" << cur_time << "\t" << cur_time+deltaFrameSeconds[i]*1000 << "\t" ;
    for (size_t x = 0; x < mVfcAverage[i].size(); x++) {
      int timeStart = 0;
      if (mVfcAverage[i][x].second > 0) {
        timeStart = timestamps[mVfcAverage[i][x].second/-1];
      }
      out <<  "(" << mVfcAverage[i][x].second << "," << mVfcAverage[i][x].first << ",[" << timeStart << "-" << timestamps[mVfcAverage[i][x].second] << "]) ";
    }
    out << std::endl;
    cur_time += deltaFrameSeconds[i] * 1000;
  }
}

size_t TimeCompression::SecondsToIndex(float seconds){
  // given input time "seconds," return an index into time compressed trace
  // nearest the time point of those with a lesser value,
  // or 0 if it is less than all time points
  {
    if ( seconds < mTimePoints[0] )
      return 0;
    std::vector<float>::iterator f = std::upper_bound ( mTimePoints.begin(), mTimePoints.end(), seconds );
    return ( f-mTimePoints.begin() -1 );
  }
}




