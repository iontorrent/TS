/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <cmath>
#include "EmphasisVector.h"
#include <cstddef>

using namespace std;

// do useful tasks involving emphasis vectors

void EmphasisClass::Allocate ( int tsize )
{
  npts = tsize;
  if ( EmphasisVectorByHomopolymer == NULL )
  {
    my_frames_per_point.resize ( npts );
    my_frameNumber.resize ( npts );
    emphasis_vector_storage.resize ( npts*numEv );
    AllocateScratch();
  }
}

void EmphasisClass::AllocateScratch()
{
  EmphasisVectorByHomopolymer = new float *[numEv];
  EmphasisScale.resize ( numEv );
}

void EmphasisClass::Destroy()
{
  if ( EmphasisVectorByHomopolymer !=NULL )
    delete[] EmphasisVectorByHomopolymer;
  EmphasisVectorByHomopolymer=NULL;
}

EmphasisClass::~EmphasisClass()
{
  Destroy();
}

void EmphasisClass::DefaultValues()
{
  float t_emp[]  = {6.86, 1.1575, 2.081, 1.230, 7.2625, 1.91, 0.0425, 19.995};
  for ( int i=0; i<8; i++ )
    emp[i]= t_emp[i];
  // matches the poisson look up table
  numEv = MAX_POISSON_TABLE_COL;
  emphasis_ampl = 7.25f;
  emphasis_width = 2.89f;
}

void EmphasisClass::SetDefaultValues ( float *t_emp, float emphasis_ampl_default,float emphasis_width_default )
{
  emphasis_ampl = emphasis_ampl_default;
  emphasis_width = emphasis_width_default;
  for ( int i=0; i<NUMEMPHASISPARAMETERS; i++ )
    emp[i] = t_emp[i];
}

EmphasisClass::EmphasisClass()
{
  npts = 0;
  EmphasisVectorByHomopolymer=NULL;
  restart = false;
  point_emphasis_by_compression = true;
  DefaultValues();
}

int RescaleVector ( float *vect, int npts )
{
  float scale = 0.0f;
  for ( int i=0; i<npts; i++ )
  {
    if ( vect[i] < 0.0f ) vect[i] = 0.0f;
    scale += vect[i];
  }
  for ( int i=0;i < npts;i++ ) // actual data
    vect[i] *= ( float ) npts/scale;
  return ( npts );
}

// just a gaussian
int GenerateBlankEmphasis ( float *vect, float *emp, int tsize, float t_center, const vector<int>& frames_per_point, const vector<float>& frameNumber )
{
  int npts = tsize;

  for ( int i=0;i < npts;i++ ) // emphasis for actual data
  {
    float deltat = frameNumber[i] - t_center;
    float EmphasisOffsetC = ( deltat-emp[6] ) / emp[7];
    vect[i] = frames_per_point[i];
    vect[i] *= exp ( -EmphasisOffsetC*EmphasisOffsetC );
  }

  RescaleVector ( vect, tsize );
  return ( npts );
}

// unusually fancy "square wave"
int GenerateStratifiedEmphasis ( float *vect, int vn, float *emp, int tsize, float t_center, const vector<int>& frames_per_point, const vector<float>& frameNumber, float width, float ampl )
{
  int npts = tsize;
  float na = emp[0]+vn*emp[1];
  float nb = emp[2]+vn*emp[3];
  float db = emp[4]+vn*emp[5];

  for ( int i=0;i < npts;i++ ) // emphasis for actual data
  {
    float deltat = frameNumber[i]-t_center;

    vect[i] = frames_per_point[i];
    float EmphasisOffsetB = ( deltat-nb ) / ( width*db );
    float tmp_val= ampl * exp ( -EmphasisOffsetB*EmphasisOffsetB );

    float EmphasisOffsetA  = ( deltat-na ) / width;
    if ( ( EmphasisOffsetA < 0.0f ) && ( deltat >= -3.0f ) ) // note: technically incompatible with spline nuc rise
      vect[i] *= tmp_val;
    else if ( EmphasisOffsetA >= 0.0f )
      vect[i] *= tmp_val * exp ( -EmphasisOffsetA*EmphasisOffsetA );
  }

  RescaleVector ( vect,npts );
  return ( npts );
}


// prepare to export the default math as a separate routine
// so that Rcpp can wrap it and we see what we expect to see
int GenerateIndividualEmphasis ( float *vect, int vn, float *emp, int tsize, float t_center, const vector<int>& frames_per_point, const vector<float>& frameNumber, float amult,float width, float ampl )
{
  float a_control = amult * ampl;
  if ( a_control>0.0f )
    return ( GenerateStratifiedEmphasis ( vect, vn, emp, tsize, t_center, frames_per_point, frameNumber, width, a_control ) );
  else
    return ( GenerateBlankEmphasis ( vect, emp, tsize, t_center, frames_per_point, frameNumber ) );
}

void EmphasisClass::SetupEmphasisTiming ( int _npts, int *frames_per_point, float *frameNumber )
{
  Allocate ( _npts );
  for ( int i=0; i<npts; i++ )
  {
    my_frames_per_point[i] = frames_per_point[i];
    my_frameNumber[i] = frameNumber[i];
  }
}

// assume we know global timing and width values already
// note if change time compression,need to resynch this function
void EmphasisClass::BuildCurrentEmphasisTable ( float t_center, float amult )
{
  std::vector<int>   local_frames_per_point;
  if (point_emphasis_by_compression)
    local_frames_per_point = my_frames_per_point;
  else{
    local_frames_per_point.assign(my_frames_per_point.size(),1); // no weighting by compression to get rid of edge artifacts
    int cur_lowest = 1000;
    // annoyingly, it looks like I need to keep the emphasis at the >start< for the clonal penalty 
    //@TODO: fix the clonal penalty in multiflow lev mar to be sensible instead of senseless
    for (unsigned int i=0; i<my_frames_per_point.size(); i++)
    {
      if (my_frames_per_point[i]<cur_lowest)
	cur_lowest = my_frames_per_point[i];
      local_frames_per_point[i] = cur_lowest;
    }
  }
    
  for ( int vn = 0;vn < numEv;vn++ )
  {
    float *vect = EmphasisVectorByHomopolymer[vn] = &emphasis_vector_storage[npts*vn];

    EmphasisScale[vn] = GenerateIndividualEmphasis ( vect, vn, emp, npts, t_center, local_frames_per_point, my_frameNumber, amult,emphasis_width, emphasis_ampl ); // actual data
  }
}

void EmphasisClass::CustomEmphasis ( float *evect, float evSel )
{
  if ( evSel < ( numEv-1 ) )
  {
    int left = ( int ) evSel;
    float f1 = ( left+1.0f-evSel );

    if ( left < 0 )
    {
      left = 0;
      f1 = 1.0f;
    }

    for ( int i=0;i < npts;i++ )
      evect[i] = f1* ( * ( EmphasisVectorByHomopolymer[left]+i ) ) + ( 1.0f-f1 ) * ( * ( EmphasisVectorByHomopolymer[left+1]+i ) );
  }
  else
  {
    for ( int i=0; i<npts; i++ )
      evect[i] = * ( EmphasisVectorByHomopolymer[numEv-1]+i );
  }
}

int EmphasisClass::ReportUnusedPoints ( float threshold, int min_used )
{
  int point_used[npts];

  for ( int i=0; i<npts; i++ )
    point_used[i] = 0;
  for ( int vn=0; vn<numEv; vn++ )
  {
    for ( int i=0; i<npts; i++ )
      if ( EmphasisVectorByHomopolymer[vn][i]>threshold )
        point_used[i]++;
  }
  int workable;
  for ( workable=npts; workable>0; workable-- )
    if ( point_used[workable-1]>=min_used ) // how many lengths need to use a point
      break;
  return ( workable ); // stop after finding a used point
}

void EmphasisClass::SignalToBkgEmphasis ( int hp_val, float *signal, float *background, float basic_noise, float relative_threshold )
{
  // two tuning parameters:  basic_noise, background_scale
  // assume we have a "typical" signal and an estimate of the ph step
  // the idea here is to auto-tune the emphasis based on these characteristics
  float *vect = EmphasisVectorByHomopolymer[hp_val] = &emphasis_vector_storage[npts*hp_val];
  for ( int i=0; i<npts; i++ )
  {
    // background drift scales by background
    // basic_noise always happens
    // signal has to rise above basic noise and bkg in order to strongly weight these points
    // basic noise takes care of "no particular backround" case at the start of the flow and any zero values
    float signal_proportion = ( signal[i]*signal[i] ) / ( basic_noise+signal[i]*signal[i]+background[i]*background[i] );
    // always between zero and 1
    vect[i] = signal_proportion;
  }
  // post process to cut off bad points
  float v_max = 0;
  for ( int i=0; i<npts;i++ )
    if ( v_max<vect[i] )
      v_max = vect[i];

  float v_threshold = v_max*relative_threshold;
  // don't know what I want here
  for ( int i=0; i<npts; i++ )
    if ( vect[i]>v_threshold )
      vect[i] = 1.0f;
    else
      vect[i] = 0.0f;

  RescaleVector ( vect,npts );
  EmphasisScale[hp_val] = npts;
}
