/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "EmphasisVector.h"

// do useful tasks involving emphasis vectors



void EmphasisClass::Allocate(int tsize)
{
  npts = tsize;
  if (emphasis_vector_storage == NULL)
  {
    emphasis_vector_storage = new float[npts*numEv];
    EmphasisVectorByHomopolymer = new float *[numEv];
    EmphasisScale = new float[numEv];
  }
}

void EmphasisClass::Destroy()
{
  if (emphasis_vector_storage!=NULL)
      delete[] emphasis_vector_storage;
  emphasis_vector_storage=NULL;
  if (EmphasisVectorByHomopolymer !=NULL)
    delete[] EmphasisVectorByHomopolymer;
  EmphasisVectorByHomopolymer=NULL;
  if (EmphasisScale!=NULL)
      delete[] EmphasisScale;
  EmphasisScale = NULL;
}

EmphasisClass::~EmphasisClass()
{
    Destroy();
}

void EmphasisClass::DefaultValues()
{
  float t_emp[]  = {6.86, 1.1575, 2.081, 1.230, 7.2625, 1.91, 0.0425, 19.995};
  for (int i=0; i<8; i++)
      emp[i]= t_emp[i];
  numEv = MAX_HPLEN+1;
    emphasis_ampl = 7.25;
    emphasis_width = 2.89;
}

void EmphasisClass::SetDefaultValues(float *t_emp, float emphasis_ampl_default,float emphasis_width_default)
{
  emphasis_ampl = emphasis_ampl_default;
  emphasis_width = emphasis_width_default;
  for (int i=0; i<NUMEMPHASISPARAMETERS; i++)
      emp[i] = t_emp[i];
}

EmphasisClass::EmphasisClass()
{
    emphasis_vector_storage=NULL;
    EmphasisVectorByHomopolymer=NULL;
    EmphasisScale = NULL;
    DefaultValues();
}

// prepare to export the default math as a separate routine
// so that Rcpp can wrap it and we see what we expect to see

int GenerateIndividualEmphasis(float *vect, int vn, float *emp, int tsize, float t_center, int *frames_per_point, float *frameNumber, float amult,float width, float ampl)
{
    int npts = tsize;
    float scale = 0.0;
    float na = emp[0]+vn*emp[1];
    float nb = emp[2]+vn*emp[3];
    float db = emp[4]+vn*emp[5];

    for (int i=0;i < npts;i++)   // emphasis for actual data
    {
      vect[i] = frames_per_point[i];
      float deltat = frameNumber[i]-t_center;
      if (amult*ampl > 0.0)
      {
        float EmphasisOffsetA  = (deltat-na) /width;
        float EmphasisOffsetB = (deltat-nb) / (width*db);
        if ((EmphasisOffsetA < 0.0) && (deltat >= -3.0)) // note: technically incompatible with spline nuc rise
          vect[i] *= ampl*amult*exp(-EmphasisOffsetB*EmphasisOffsetB);
        else if (EmphasisOffsetA >= 0.0)
          vect[i] *= ampl*amult*exp(-EmphasisOffsetA*EmphasisOffsetA) *exp(-EmphasisOffsetB*EmphasisOffsetB);
      }
      else
      {
        float EmphasisOffsetC = (deltat-emp[6]) /emp[7];
        vect[i] *= exp(-EmphasisOffsetC*EmphasisOffsetC);
      }

      if (vect[i] < 0.0) vect[i] = 0.0;

      scale += vect[i];
    }

    for (int i=0;i < npts;i++)   // actual data
      vect[i] *= (float) npts/scale;
    
    return(npts);
}


void EmphasisClass::GenerateEmphasis(int tsize, float t_center, int *frames_per_point, float *frameNumber, float amult,float width, float ampl)
{
  Allocate(tsize);
  // allocate memory if necessary

  for (int vn = 0;vn < numEv;vn++)
  {
    float *vect = EmphasisVectorByHomopolymer[vn] = &emphasis_vector_storage[npts*vn];

    EmphasisScale[vn] = GenerateIndividualEmphasis(vect, vn, emp, tsize,t_center,frames_per_point,frameNumber, amult,width,ampl); // actual data
  }
}

void EmphasisClass::CustomEmphasis(float *evect, float evSel)
{
  if (evSel < (numEv-1))
  {
    int left = (int) evSel;
    float f1 = (left+1.0-evSel);

    if (left < 0)
    {
      left = 0;
      f1 = 1.0;
    }

    for (int i=0;i < npts;i++)
      evect[i] = f1* (* (EmphasisVectorByHomopolymer[left]+i)) + (1-f1) * (* (EmphasisVectorByHomopolymer[left+1]+i));
  }
  else
  {
    for (int i=0; i<npts; i++)
      evect[i] = * (EmphasisVectorByHomopolymer[numEv-1]+i);
  }
}