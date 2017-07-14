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
    nonZeroEmphasisFrames.resize(numEv);
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
int GenerateBlankEmphasis ( float *vect, float *emp, int tsize, float t_center,
                            const vector<int>& frames_per_point, const vector<float>& frameNumber )
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

// t-distribution heavier tailed but has natural scale span
int BlankDataWeight(float *vect, int tsize,
                    float t_center, const vector<int>& frames_per_point, const vector<float>& frameNumber,
                    float blank_span){
  int npts = tsize;
  for (int i=0; i<npts; i++){
    float deltat = frameNumber[i]-t_center;
    float blank_weight = deltat/blank_span;
    blank_weight = 1.0f/(blank_weight*blank_weight+1);
    blank_weight *=blank_weight;
    vect[i] = frames_per_point[i];
    vect[i] *=blank_weight;
  }
  RescaleVector(vect,tsize);
  return(npts);
}

// unusually fancy "square wave"
int GenerateStratifiedEmphasis ( float *vect, int vn, float *emp, int tsize,
                                 float t_center, const vector<int>& frames_per_point, const vector<float>& frameNumber,
                                 float width, float ampl )
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

// fix from offsets
int GenerateNmerDataWeights(float *vect, int tsize, int vn,
                            float t_center, const vector<int>& frames_per_point, const vector<float>& frameNumber,
                            DataWeightDefaults &data_weights
                            ){
  // we fall to zero here
  float nmer_zero = t_center+data_weights.zero_span+vn*data_weights.nmer_span_increase;
  // we fall to zero at this rate
  float zero_slope= (data_weights.zero_span*(1.0f-data_weights.zero_fade_start));
  if (zero_slope<1.0f ) zero_slope =1.0f; // 1 frame drop minimum
  zero_slope = -1.0f/zero_slope; // no divide by zero, no weird negatives

  float offset= -1.0f * zero_slope*nmer_zero;
  int npts = tsize;
  for (int i=0; i<npts; i++){

    // manufacture downslope
    float tmp_weight = frameNumber[i]*zero_slope+offset;
    if (tmp_weight<0.0f) tmp_weight = 0.0f;
    if (tmp_weight>1.0f) tmp_weight = 1.0f;

    // now apply constraint for prefix
    if (t_center-frameNumber[i]>=data_weights.prefix_start) tmp_weight=data_weights.prefix_value;
    vect[i] = frames_per_point[i];
    vect[i] *= tmp_weight;

  }
  RescaleVector(vect, npts);
  return(npts);
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
  }

  for ( int vn = 0;vn < numEv;vn++ )
  {
    float *vect = EmphasisVectorByHomopolymer[vn] = &emphasis_vector_storage[npts*vn];

    if (data_weights.use_data_weight){
      if (amult>0.0f){
        EmphasisScale[vn]=GenerateNmerDataWeights(vect, npts, vn,
                                                  t_center, local_frames_per_point, my_frameNumber,
                                                  data_weights);

      } else {
        EmphasisScale[vn] = BlankDataWeight(vect, npts,
                                            t_center, local_frames_per_point, my_frameNumber,
                                            data_weights.blank_span);
      }
/*
      int vm=vn+1;
      if (vm>=numEv) vm=vn;
      float *tect = EmphasisVectorByHomopolymer[vm] = &emphasis_vector_storage[npts*(vm)];
      GenerateIndividualEmphasis ( tect, vn, emp, npts,
                                                             t_center, local_frames_per_point, my_frameNumber,
                                                             amult,emphasis_width, emphasis_ampl ); // actual data
      for(int vp=0; vp<npts; vp++){
        printf("DataWeight: %d %d %f %f %f %f\n",vn,vp,t_center, amult, vect[vp],tect[vp]);
      }*/
    }
    else {
      EmphasisScale[vn] = GenerateIndividualEmphasis ( vect, vn, emp, npts,
                                                       t_center, local_frames_per_point, my_frameNumber,
                                                       amult,emphasis_width, emphasis_ampl ); // actual data
    }
    DetermineNonZeroEmphasisFrames(vn);
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

void EmphasisClass::DetermineNonZeroEmphasisFrames(int hp) {
  int zeroCnt = 0;
  for (int i=npts-1; i>=0 ; i--) {
    if (emphasis_vector_storage[npts*hp + i] <= CENSOR_THRESHOLD)
      zeroCnt++;
    else
      break;
  }
  nonZeroEmphasisFrames[hp] = npts - zeroCnt;
}

void EmphasisClass::SaveEmphasisVector()
{
  printf("dst VFC profile: ");
  for ( int i=0; i<npts; i++ )
  {
    printf(" %d(%.1lf)", my_frames_per_point[i],my_frameNumber[i]);
    //printf(" %.1lf", my_frameNumber[i]);
  }
  printf("\n");
}

void EmphasisClass::SetUpEmphasis(TimeAndEmphasisDefaults &data_control, TimeCompression &time_c){
  SetDefaultValues (
        data_control.emphasis_params.emp,
        data_control.emphasis_params.emphasis_ampl_default,
        data_control.emphasis_params.emphasis_width_default);
  data_weights = data_control.data_weights;
  SetupEmphasisTiming (
        time_c.npts(),
        &(time_c.frames_per_point[0]),
        &(time_c.frameNumber[0]));
  point_emphasis_by_compression = data_control.emphasis_params.point_emphasis_by_compression;
}
