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

float Expm2Approx ( float x )
{
  int left, right;
// float sign = 1.0;
  float frac;
  float ret;

  if ( x < 0.0 )
  {
    x = -x;
//  sign = -1.0;
  }

  left = ( int ) ( x * 100.0 ); // left-most point in the lookup table
  right = left + 1; // right-most point in the lookup table

  // both left and right points are inside the table...interpolate between them
  if ( ( left >= 0 ) && ( right < ( int ) ( sizeof ( Exp2ApproxArray ) /sizeof ( Exp2ApproxArray[0] ) ) ) )
  {
    frac = ( x * 100.0 - left );
    ret = ( 1 - frac ) * Exp2ApproxArray[left] + frac * Exp2ApproxArray[right];
  }
  else
  {
    if ( left < 0 )
      ret = Exp2ApproxArray[0];
    else
      ret = Exp2ApproxArray[ ( sizeof ( Exp2ApproxArray ) /sizeof ( Exp2ApproxArray[0] ) ) - 1];
  }

  return ( ret ); // don't multiply by the sign..
}


float ErfApprox ( float x )
{
  int left, right;
  float sign = 1.0;
  float frac;
  float ret;

  if ( x < 0.0 )
  {
    x = -x;
    sign = -1.0;
  }

  left = ( int ) ( x * 100.0 ); // left-most point in the lookup table
  right = left + 1; // right-most point in the lookup table

  // both left and right points are inside the table...interpolate between them
  if ( ( left >= 0 ) && ( right < ( int ) ( sizeof ( ErfApproxArray ) /sizeof ( ErfApproxArray[0] ) ) ) )
  {
    frac = ( x * 100.0 - left );
    ret = ( 1 - frac ) * ErfApproxArray[left] + frac * ErfApproxArray[right];
  }
  else
  {
    if ( left < 0 )
      ret = ErfApproxArray[0];
    else
      ret = 1.0;//ErfApproxArray[(sizeof(ErfApproxArray)/sizeof(ErfApproxArray[0])) - 1];
  }

  return ( ret * sign );
}


#if 0
float ExpApprox ( float x )
{
  int left, right;
  float frac;
  float ret;

  if ( x > 0 )
  {
    printf ( "got positive number %f\n",x );
    return exp ( x );
  }

  x = -x; // make the index positive

  left = ( int ) ( x * 100.0 ); // left-most point in the lookup table
  right = left + 1; // right-most point in the lookup table

  // both left and right points are inside the table...interpolate between them
  if ( ( left >= 0 ) && ( right < ( int ) ( sizeof ( ExpApproxArray ) /sizeof ( ExpApproxArray[0] ) ) ) )
  {
    frac = ( x * 100.0 - left );
    ret = ( 1 - frac ) * ExpApproxArray[left] + frac * ExpApproxArray[right];
  }
  else
  {
    if ( left < 0 )
      ret = ExpApproxArray[0];
    else
      ret = 0.0;//ExpApproxArray[(sizeof(ExpApproxArray)/sizeof(ExpApproxArray[0])) - 1];
  }

  return ( ret );
}
#endif




MixtureMemo::MixtureMemo()
{
  my_mixL = NULL;
  my_mixR = NULL;
  my_deltaL = NULL;
  my_deltaR = NULL;
  my_totalL = NULL;
  my_totalR = NULL;
  mixLUT = NULL;
  A = 1.0;
  max_dim = 0;
  max_dim_minus_one = 0;
  max_entry = 0;
  inv_scale = 20.0f;
  //_inv_scale = _mm_set_ss( inv_scale );

  scale = 0.05f;
  occ_r = 1.0f;
  occ_l = 0.0f;
  lx = ly = rx = ry = 0.0f; // interpolation coefficients, cached
  left = right = 0; // cached interpolation state
  dA=0.0f;
  total_live = 0.0f;
}



float MixtureMemo::Generate ( float _A, PoissonCDFApproxMemo *my_math )
{
  max_dim = my_math->max_dim;
  max_dim_minus_one = max_dim-1;
  max_entry = max_dim-1;
  inv_scale = 1/my_math->scale;
  //_inv_scale = _mm_set_ss( inv_scale );
  scale = my_math->scale;

  int ileft, iright;
  float idelta, ifrac;

  A = _A;
  if ( A!=A )
    A=0.0001f; // safety check
  if ( A<0.0001f )
    A = 0.0001f; // safety
  // initialize diffusion/reaction simulation for this flow
  int maxA = LAST_POISSON_TABLE_COL; // largest value computed in the table
  if ( A>maxA )
    A = maxA;
  ileft = ( int ) A;
  idelta = A-ileft;
  iright = ileft+1;
  ifrac = 1-idelta;
  ileft--;
  iright--;

  occ_l = ifrac; // lower mixture
  occ_r = idelta; // upper mixture

  // special case # 1
  if ( ileft<0 ) // A between 0 and 1
  {
    ileft = 0;
    occ_l = 0.0f;
  }

  if ( iright==maxA ) // A at upper limit
  {
    // swap so only have one test when executing
    iright=ileft;
    occ_r = occ_l; // 1.0
    occ_l = 0.0f;
  }
  load_occ_vec( occ_r, occ_l );

  my_mixL = my_math->poiss_cdf[ileft];
  my_mixR = my_math->poiss_cdf[iright];

  if( ileft == 0 && iright == 0 )
      mixLUT = my_math->poissLUT[0]; //special case for the packed case for 0 < A < 1
  else
      mixLUT = my_math->poissLUT[ileft+1]; //layout: poiss_cdf[ei][i], poiss_cdf[ei+1][i], poiss_cdf[ei][i+1], poiss_cdf[ei+1][i+1]

  my_deltaL = my_math->dpoiss_cdf[ileft];
  my_deltaR = my_math->dpoiss_cdf[iright];
  my_totalL = my_math->ipoiss_cdf[ileft];
  my_totalR = my_math->ipoiss_cdf[iright];

  // could combine these two here, but as they're likely longer than the accesses, keep separate for now.

  total_live = occ_l + occ_r;
  left = right = 0;

  return ( A ); // if hit maximum
}

void MixtureMemo::ScaleMixture ( float SP )
{
  dA*=SP;
  total_live *=SP;
  occ_l *=SP;
  occ_r *=SP;
  load_occ_vec( occ_r, occ_l );
}


// generate both currently active polymerase and generated hplus
void MixtureMemo::UpdateState ( float total_intensity, float &active_polymerase, float &generated_hplus )
{
  total_intensity *= inv_scale;
  left = ( int ) total_intensity;
  right = left+1;
  float idelta = total_intensity-left;
  float ifrac = 1.0f-idelta;
  if ( right> ( max_dim-1 ) ) right = max_dim-1;
  if ( left> ( max_dim-1 ) ) left = max_dim-1;
  // interpolate between levels and between total intensities
  lx = ifrac*occ_l;
  ly = ifrac*occ_r;
  rx = idelta*occ_l;
  ry = idelta*occ_r;
  active_polymerase = lx*my_mixL[left]+ly*my_mixR[left]+rx*my_mixL[right]+ry*my_mixR[right];
  generated_hplus = lx*my_totalL[left]+ly*my_totalR[left]+rx*my_totalL[right]+ry*my_totalR[right];
}

// generate both currently active polymerase and generated hplus
void MixtureMemo::UpdateActivePolymeraseState ( float total_intensity, float &active_polymerase )
{
  total_intensity *= inv_scale;
  left = ( int ) total_intensity;
  right = left+1;
  float idelta = total_intensity-left;
  float ifrac = 1.0f-idelta;
  if ( right> ( max_dim_minus_one ) ) right = left = max_dim_minus_one;

  // interpolate between levels and between total intensities
  lx = ifrac*occ_l;
  ly = ifrac*occ_r;
  rx = idelta*occ_l;
  ry = idelta*occ_r;
  active_polymerase = lx*my_mixL[left]+ly*my_mixR[left]+rx*my_mixL[right]+ry*my_mixR[right];
}

void MixtureMemo::UpdateGeneratedHplus ( float &generated_hplus )
{
  // exploit the fact that we previously cached the interpolation field
  generated_hplus = lx*my_totalL[left]+ly*my_totalR[left]+rx*my_totalL[right]+ry*my_totalR[right];
}


float MixtureMemo::GetDStep ( float x )
{
  x *= inv_scale;
  int left = ( int ) x;
  if ( left>max_entry ) left = max_entry;
  float idelta = ( x-left ) *scale; // should this be >after< the test?
  // use derivative here, not interpolation
  // this may imply the need to transpose this matrix
  return ( occ_l* ( my_mixL[left]+idelta*my_deltaL[left] ) +occ_r* ( my_mixR[left]+idelta*my_deltaR[left] ) );
}

float MixtureMemo::GetIStep ( float x )
{
  x *= inv_scale;
  int left = ( int ) x;
  if ( left>max_entry ) left = max_entry;
  return ( occ_l*my_mixL[left]+occ_r*my_mixR[left] );
}

void MixtureMemo::Delete()
{
  // nothing allocated, just dereference
  my_mixL = NULL;
  my_mixR = NULL;
  my_deltaL = NULL;
  my_deltaR = NULL;
  mixLUT = NULL;
}

MixtureMemo::~MixtureMemo()
{
  Delete();
}

