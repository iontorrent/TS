/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef EMPHASISVECTOR_H
#define EMPHASISVECTOR_H

#include <stdio.h>
#include <vector>
#include <math.h>

#include "BkgMagicDefines.h"
#include "MathOptim.h"

#define NUMEMPHASISPARAMETERS 8

class EmphasisClass
{
  public: // yes, bad form
  // data for applying emphasis to data points during fitting
  float   *emphasis_vector_storage;      // storage for emphasis vectors
  float   **EmphasisVectorByHomopolymer;                   // array of pointers to different vectors
  float   *EmphasisScale;              // scaling factor for each vector
   int numEv;               // number of emphasis vectors allocated
  float emp[NUMEMPHASISPARAMETERS];  // parameters for emphasis vector generation
  float   emphasis_width;   // parameters scaling the emphasis vector
  float   emphasis_ampl;    // parameters scaling the emphasis vector
  
  int npts; // how long the vector should be
  
  void CustomEmphasis(float *evect, float evSel);
  void GenerateEmphasis(int tsize, float t_center, int *frames_per_point, float *frameNumber, float amult,float width, float ampl);
  void Allocate(int tsize);
  void Destroy();
  void DefaultValues();
  void SetDefaultValues(float *, float, float);
  EmphasisClass();
  ~EmphasisClass();
};

int GenerateIndividualEmphasis(float *vect, int vn, float *emp, int tsize, float t_center, int *frames_per_point, float *frameNumber, float amult,float width, float ampl);


#endif // EMPHASISVECTOR_H