/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * MathOptim.cpp
 *
 *  Created on: Jun 7, 2010
 *      Author: Mark Beauchemin
 */

#include <math.h>
#include <stdio.h>
#include "MathOptim.h"

#include "MathTables.h"  // annoying tables that make code unreadable

float Expm2Approx(float x)
{
	int left, right;
//	float sign = 1.0;
	float frac;
	float ret;

	if (x < 0.0)
	{
		x = -x;
//		sign = -1.0;
	}

	left = (int) (x * 100.0); // left-most point in the lookup table
	right = left + 1; // right-most point in the lookup table

	// both left and right points are inside the table...interpolate between them
	if ((left >= 0) && (right < (int)(sizeof(Exp2ApproxArray)/sizeof(Exp2ApproxArray[0]))))
	{
		frac = (x * 100.0 - left);
		ret = (1 - frac) * Exp2ApproxArray[left] + frac * Exp2ApproxArray[right];
	}
	else
	{
		if (left < 0)
			ret = Exp2ApproxArray[0];
		else
			ret = Exp2ApproxArray[(sizeof(Exp2ApproxArray)/sizeof(Exp2ApproxArray[0])) - 1];
	}

	return (ret); // don't multiply by the sign..
}


float ErfApprox(float x)
{
	int left, right;
	float sign = 1.0;
	float frac;
	float ret;

	if (x < 0.0)
	{
		x = -x;
		sign = -1.0;
	}

	left = (int) (x * 100.0); // left-most point in the lookup table
	right = left + 1; // right-most point in the lookup table

	// both left and right points are inside the table...interpolate between them
	if ((left >= 0) && (right < (int)(sizeof(ErfApproxArray)/sizeof(ErfApproxArray[0]))))
	{
		frac = (x * 100.0 - left);
		ret = (1 - frac) * ErfApproxArray[left] + frac * ErfApproxArray[right];
	}
	else
	{
		if (left < 0)
			ret = ErfApproxArray[0];
		else
			ret = 1.0;//ErfApproxArray[(sizeof(ErfApproxArray)/sizeof(ErfApproxArray[0])) - 1];
	}

	return (ret * sign);
}

float ExpApprox(float x)
{
	int left, right;
	float frac;
	float ret;

	if (x > 0)
	{
		printf("got positive number %f\n",x);
		return exp(x);
	}

	x = -x; // make the index positive

	left = (int) (x * 100.0); // left-most point in the lookup table
	right = left + 1; // right-most point in the lookup table

	// both left and right points are inside the table...interpolate between them
	if ((left >= 0) && (right < (int)(sizeof(ExpApproxArray)/sizeof(ExpApproxArray[0]))))
	{
		frac = (x * 100.0 - left);
		ret = (1 - frac) * ExpApproxArray[left] + frac * ExpApproxArray[right];
	}
	else
	{
		if (left < 0)
			ret = ExpApproxArray[0];
		else
			ret = 0.0;//ExpApproxArray[(sizeof(ExpApproxArray)/sizeof(ExpApproxArray[0])) - 1];
	}

	return (ret);
}


    Dual operator+(const Dual &x, const Dual &y){
      return Dual(x.a+y.a,x.da+y.da,x.dk+y.dk);
    };
    Dual operator-(const Dual &x, const Dual &y){
      return Dual(x.a-y.a,x.da-y.da,x.dk-y.dk);
    };
    Dual operator*(const Dual &x, const Dual &y){
      return Dual(x.a*y.a, x.a*y.da+y.a*x.da,x.a*y.dk+y.a*x.dk);
    };
    Dual operator/(const Dual &x, const Dual &y){
      return Dual(x.a/y.a, (y.a*x.da-x.a*y.da)/(y.a*y.a), (y.a*x.dk-x.a*y.dk)/(y.a*y.a));
    };



PoissonCDFApproxMemo::PoissonCDFApproxMemo()
{
  poiss_cdf = NULL;
  dpoiss_cdf = NULL;
  ipoiss_cdf = NULL;
  max_events = MAX_HPLEN;
  max_dim = 0;
  scale = 0.05;
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

MixtureMemo::MixtureMemo()
{
  my_mixL = NULL;
  my_mixR = NULL;
  my_deltaL = NULL;
  my_deltaR = NULL;
  my_totalL = NULL;
  my_totalR = NULL;
  A = 1.0;
  max_dim = 0;
  max_entry = 0;
  inv_scale = 20;
  scale = 0.05;
  occ_r = 1.0;
  occ_l = 0.0;
  dA=0;
  total_live = 0;
}

Dual MixtureMemo::Generate(Dual &_A, PoissonCDFApproxMemo *my_math)
{
  float tA = Generate(_A.a,my_math);
  dA = _A.da;
  return(Dual(tA,_A.da,_A.dk)); 
}

float MixtureMemo::Generate(float _A, PoissonCDFApproxMemo *my_math)
{
  max_dim = my_math->max_dim;
  max_entry = max_dim-1;
  inv_scale = 1/my_math->scale;
  scale = my_math->scale;
  
     int ileft, iright;
    float idelta, ifrac;

    A = _A;
    if (A!=A)
      A=0; // safety check
    if (A<0)
      A = 0; // safety
    // initialize diffusion/reaction simulation for this flow
    int maxA = MAX_HPLEN; // largest value computed in the table
    if (A>maxA)
        A = maxA;
    ileft = (int) A;
    idelta = A-ileft;
    iright = ileft+1;
    ifrac = 1-idelta;
    ileft--;
    iright--;

    occ_l = ifrac; // lower mixture
    occ_r = idelta; // upper mixture
    
    // special case # 1
    if (ileft<0) // A between 0 and 1
    {
      ileft = 0;
      occ_l = 0.0;
    }
    
    if (iright==maxA) // A at upper limit
    {
        // swap so only have one test when executing
        iright=ileft;
        occ_r = occ_l; // 1.0
        occ_l = 0;
    }
    my_mixL = my_math->poiss_cdf[ileft];
    my_mixR = my_math->poiss_cdf[iright];
    my_deltaL = my_math->dpoiss_cdf[ileft];
    my_deltaR = my_math->dpoiss_cdf[iright];
    my_totalL = my_math->ipoiss_cdf[ileft];
    my_totalR = my_math->ipoiss_cdf[iright];
    
    // could combine these two here, but as they're likely longer than the accesses, keep separate for now.
    
    total_live = occ_l + occ_r;

    return(A); // if hit maximum 
}

void MixtureMemo::ScaleMixture(float SP)
{
    dA*=SP;
   total_live *=SP;
    occ_l *=SP;
    occ_r *=SP;
}

float MixtureMemo::GetStep(float x){
  x *= inv_scale;
  int left = (int) x;
  int right = left+1;
  float idelta = x-left;
  float ifrac = 1-idelta;
  if (right>(max_dim-1)) right = max_dim-1;
  if (left>(max_dim-1)) left = max_dim-1;
  // interpolate between levels and between total intensities
  return(ifrac*(occ_l*my_mixL[left]+occ_r*my_mixR[left])+idelta*(occ_l*my_mixL[right]+occ_r*my_mixR[right]));
}

// generate both currently active polymerase and generated hplus
void MixtureMemo::UpdateState(float total_intensity, float &active_polymerase, float &generated_hplus)
{
    total_intensity *= inv_scale;
  int left = (int) total_intensity;
  int right = left+1;
  float idelta = total_intensity-left;
  float ifrac = 1-idelta;
  if (right>(max_dim-1)) right = max_dim-1;
  if (left>(max_dim-1)) left = max_dim-1;
  // interpolate between levels and between total intensities
  float lx = ifrac*occ_l;
  float ly = ifrac*occ_r;
  float rx = idelta*occ_l;
  float ry = idelta*occ_r;
  active_polymerase = lx*my_mixL[left]+ly*my_mixR[left]+rx*my_mixL[right]+ry*my_mixR[right];
  generated_hplus = lx*my_totalL[left]+ly*my_totalR[left]+rx*my_totalL[right]+ry*my_totalR[right];
}

Dual MixtureMemo::GetStep(Dual x){
  float tx = x.a*inv_scale;
  int left = (int) tx;
  int right = left+1;
  float idelta = tx-left;
  float ifrac = 1-idelta;
  if (right>(max_dim-1)) right = max_dim-1;
  if (left>(max_dim-1)) left = max_dim-1;
  // interpolate between levels and between total intensities
  float retval = (ifrac*(occ_l*my_mixL[left]+occ_r*my_mixR[left])+idelta*(occ_l*my_mixL[right]+occ_r*my_mixR[right]));
  float fa = (ifrac*(occ_l*my_deltaL[left]+occ_r*my_deltaR[left])+idelta*(occ_l*my_deltaL[right]+occ_r*my_deltaR[right])); //handle instantaneous changes
  float xs = -(ifrac*my_deltaR[left]+idelta*my_deltaR[right]); // handle dA influence on pact from initial allocation
  return Dual(retval,x.da*fa+dA*xs,x.dk*fa);
}

float MixtureMemo::GetDStep(float x){
    x *= inv_scale;
    int left = (int) x;
    if (left>max_entry) left = max_entry;
    float idelta = (x-left)*scale; // should this be >after< the test?
    // use derivative here, not interpolation
    // this may imply the need to transpose this matrix
    return(occ_l*(my_mixL[left]+idelta*my_deltaL[left])+occ_r*(my_mixR[left]+idelta*my_deltaR[left]));
}

float MixtureMemo::GetIStep(float x){
  x *= inv_scale;
  int left = (int) x;
  if (left>max_entry) left = max_entry;
  return(occ_l*my_mixL[left]+occ_r*my_mixR[left]);
}

void MixtureMemo::Delete()
{
  // nothing allocated, just dereference
  my_mixL = NULL;
  my_mixR = NULL;
  my_deltaL = NULL;
  my_deltaR = NULL;
}

MixtureMemo::~MixtureMemo()
{
  Delete();
}