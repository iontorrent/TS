/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "PoissonCdf.h"




PoissonCDFApproxMemo::PoissonCDFApproxMemo()
{
  poiss_cdf = NULL;
  dpoiss_cdf = NULL;
  ipoiss_cdf = NULL;
  max_events = MAX_HPLEN;
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
  delete[] ipoiss_cdf;}

PoissonCDFApproxMemo::~PoissonCDFApproxMemo()
{
  Delete();
}