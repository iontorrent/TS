/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "PoissonCdf.h"
#include <cstring>




PoissonCDFApproxMemo::PoissonCDFApproxMemo()
{
  poiss_cdf = NULL;
  dpoiss_cdf = NULL;
  ipoiss_cdf = NULL;
  poissLUT = NULL;
  max_events = MAX_POISSON_TABLE_COL;
  max_dim = 0;
  scale = 0.05f;
  t = NULL;

}

void PoissonCDFApproxMemo::Allocate(int _max_events, int _max_dim, float _scale)
{
   max_events = _max_events;
   max_dim = _max_dim;
   scale = _scale;
   poiss_cdf = new float * [max_events];
   for (int i=0; i<max_events; i++){
      poiss_cdf[i] = new float [max_dim];
   }
   dpoiss_cdf = new float *[max_events];
   for (int i=0; i<max_events; i++){
      dpoiss_cdf[i] = new float [max_dim];
   }
   ipoiss_cdf = new float *[max_events];
   for (int i=0; i<max_events; i++){
      ipoiss_cdf[i] = new float [max_dim];
   }
   poissLUT = new __m128 *[max_events+1];
   for (int i=0; i<max_events+1; i++){
      poissLUT[i] = new __m128 [max_dim];
   }

   t = new float [max_dim];
}

void PoissonCDFApproxMemo::GenerateValues()
{
  for (int i=0; i<max_dim; i++)
    t[i] = ((float)i)*scale;
  // set up first value exponential decay
  for (int i=0; i<max_dim; i++)
    poiss_cdf[0][i] = exp(-t[i]);
  // generate basic incremental values
  // t^k*exp(-t)/k!
  for (int ei=1; ei<max_events; ei++)
  {
    int pe = ei-1;
    for (int i=0; i<max_dim; i++)
    {
        poiss_cdf[ei][i] = poiss_cdf[pe][i] * t[i]/((float)ei);
    }
  }
  for (int ei=0; ei<max_events; ei++)
  {
    for (int i=0; i<max_dim; i++)
      dpoiss_cdf[ei][i] = -poiss_cdf[ei][i]; // derivative will be last term added, negative because of cancellations
  }
  // generate cumulative values
  for (int i=0; i<max_dim; i++)
  {
    float tmp_sum = 0;
    for (int ei=0; ei<max_events; ei++)
    {
      tmp_sum += poiss_cdf[ei][i];
      poiss_cdf[ei][i] = tmp_sum;
    }
  }
  // generate integrated values
  for (int i=0; i<max_dim; i++)
  {
    float tmp_sum = 0;
    for (int ei=0; ei<max_events; ei++)
    {
      tmp_sum += 1.0-poiss_cdf[ei][i]; // number that have >finished< at this total intensity
      ipoiss_cdf[ei][i] = tmp_sum; // total generated hydrogens per molecule at this intensity
    }
  }

  //pack poisson values for optimized 2D interpolation in function GetStep
  for( int i=0; i<max_dim-1; ++i ){
      poissLUT[0][i] = _mm_set_ps( poiss_cdf[0][i], poiss_cdf[0][i], poiss_cdf[0][i+1], poiss_cdf[0][i+1] );
  }
  poissLUT[0][max_dim-1] = _mm_set_ps( poiss_cdf[0][max_dim-1], poiss_cdf[0][max_dim-1], poiss_cdf[0][max_dim-1], poiss_cdf[0][max_dim-1] );

  for( int ei=0; ei<max_events-2; ++ei ){
      for( int i=0; i<max_dim-1; ++i ){
          poissLUT[ei+1][i] = _mm_set_ps( poiss_cdf[ei][i], poiss_cdf[ei+1][i], poiss_cdf[ei][i+1], poiss_cdf[ei+1][i+1] );
      }
      poissLUT[ei+1][max_dim-1] = _mm_set_ps( poiss_cdf[ei][max_dim-1], poiss_cdf[ei+1][max_dim-1], poiss_cdf[ei][max_dim-1], poiss_cdf[ei+1][max_dim-1] );
  }

  for( int i=0; i<max_dim-1; ++i ){
      poissLUT[max_events-1][i] = _mm_set_ps( poiss_cdf[max_events-2][i], poiss_cdf[max_events-2][i], poiss_cdf[max_events-2][i+1], poiss_cdf[max_events-2][i+1] );
      poissLUT[max_events][i] = _mm_set_ps( poiss_cdf[max_events-1][i], poiss_cdf[max_events-1][i], poiss_cdf[max_events-1][i+1], poiss_cdf[max_events-1][i+1] );
  }
  poissLUT[max_events-1][max_dim-1] = _mm_set_ps( poiss_cdf[max_events-2][max_dim-1], poiss_cdf[max_events-2][max_dim-1], poiss_cdf[max_events-2][max_dim-1], poiss_cdf[max_events-2][max_dim-1] );
  poissLUT[max_events][max_dim-1] = _mm_set_ps( poiss_cdf[max_events-1][max_dim-1], poiss_cdf[max_events-1][max_dim-1], poiss_cdf[max_events-1][max_dim-1], poiss_cdf[max_events-1][max_dim-1] );
}

void PoissonCDFApproxMemo::DumpValues()
{
  FILE *fp = fopen("poisson.cdf.txt","wt");
  for (int i=0; i<max_dim; i++)
  {
    fprintf(fp,"%1.5f\t",t[i]);
    for (int ei=0; ei<max_events; ei++){
      fprintf(fp,"%1.5f\t",poiss_cdf[ei][i]);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
}

void PoissonCDFApproxMemo::Delete()
{
  delete[] t;
  for (int i=0; i<max_events; i++)
      delete[] poiss_cdf[i];
  delete[] poiss_cdf;
  for (int i=0; i<max_events; i++)
      delete[] dpoiss_cdf[i];
  delete[] dpoiss_cdf;
  for (int i=0; i<max_events; i++)
      delete[] ipoiss_cdf[i];
  delete[] ipoiss_cdf;
  for(int i=0;i<max_events+1;i++)
    delete[] poissLUT[i];
  delete[] poissLUT;
}

PoissonCDFApproxMemo::~PoissonCDFApproxMemo()
{
  Delete();
}
