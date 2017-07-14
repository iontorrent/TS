/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * MathOptim.h
 *
 *  Created on: Jun 7, 2010
 *      Author: Mark Beauchemin
 */

#ifndef MATHOPTIM_H
#define MATHOPTIM_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <stdlib.h>

#include "PoissonCdf.h"


#if defined( __SSE__ ) && !defined( __CUDACC__ )
    #include <x86intrin.h>
#else
    typedef float __m128 __attribute__ ((vector_size (16)));
#endif


union Vec4
{
  float v __attribute__ ( ( vector_size ( 16 ) ) );
  float e[4];
};



// wrap the poisson lookup table
// we do a mixture of two adjacent homopolymer levels
class MixtureMemo
{
  public:
    float A;
    float dA; // initial derivative for pact
    float *my_mixL; // custom interpolation
    float *my_mixR; // custom interpolation
    float *my_deltaL;
    float *my_deltaR;
    float *my_totalL; // total hydrogens generated
    float *my_totalR;
    __m128 * mixLUT;

    int left;
    int right;
    float lx;
    float ly;
    float rx;
    float ry;
    int max_dim;
    int max_dim_minus_one;
    int max_entry;
    float inv_scale;
    float scale;
    float total_live;
    float occ_l, occ_r; // historical
    __m128 occ_vec;
    __m128 _inv_scale;

    MixtureMemo();
    ~MixtureMemo();
    float Generate ( float A, PoissonCDFApproxMemo  *my_math );

    void ScaleMixture ( float SP );

#if !defined( __SSE3__ ) || defined( __CUDACC__ )
    //slow version that is accelerated below
    inline float GetStep ( float x )
    {
      x *= inv_scale;
      left = ( int ) x;
      right = left+1;
      float idelta = x-left;
      float ifrac = 1.0f-idelta;
      if ( right > max_dim_minus_one )  right = left = max_dim_minus_one;
      // interpolate between levels and between total intensities
      return ( ifrac* ( occ_l*my_mixL[left]+occ_r*my_mixR[left] ) +idelta* ( occ_l*my_mixL[right]+occ_r*my_mixR[right] ) );
      // astonishingly, the below is noticeably slower
      //return ( occ_l * ( my_mixL[left] + idelta* ( my_mixL[right]-my_mixL[left] ) ) +occ_r* ( my_mixR[left]+idelta* ( my_mixR[right]-my_mixR[left] ) ) );
    }
#else
    inline int min( int x, int y ){ return (x<y)? x:y; }
    inline float GetStep ( float x )
    {
        float ret;
        x *= inv_scale;
        left = (int) x;

        float idelta = x-left;
        float ifrac = 1.0f-idelta;

        left = min( left, max_dim_minus_one );

        // interpolate between levels and between total intensities

        __m128 frac_vec = _mm_set_ps(ifrac, ifrac, idelta, idelta);
        __m128 frac_m_occ = _mm_mul_ps( frac_vec, occ_vec );

        //__m128 frac_m_occ_m_mix = _mm_mul_ps( frac_m_occ, mixLUT[left] );

        //For SSE4 we can replace the next three lines with this:
        //__m128 ret_vec = _mm_dp_ps( frac_m_occ, mixLUT[left],  0xF1 );

        __m128 frac_m_occ_m_mix = _mm_mul_ps( frac_m_occ, mixLUT[left] );
        __m128 ret_vec = _mm_hadd_ps( frac_m_occ_m_mix, frac_m_occ_m_mix );
        ret_vec = _mm_hadd_ps( ret_vec, ret_vec );

        _mm_store_ss( &ret, ret_vec);

        return ret;
    }
#endif

    float GetDStep ( float x );
    float GetIStep ( float x );
    void UpdateState ( float total_intensity, float &active_polymerase, float &generated_hplus );
    void UpdateActivePolymeraseState ( float total_intensity, float &active_polymerase );
    void UpdateGeneratedHplus ( float &generated_hplus ); // uses current active intensity & interpolation
    void Delete();
private:
    inline void load_occ_vec( const float& occ_r, const float& occ_l ){
#if defined( __SSE3__ ) && !defined( __CUDACC__ )
        occ_vec = _mm_set_ps( occ_l, occ_r, occ_l, occ_r );
#else
        (void) occ_l; (void) occ_r; //stub for cuda
#endif
    }

};


float ErfApprox ( float x );
float Expm2Approx ( float x );
inline float ExpApprox ( float x )
{
	  x = 1.0 + x / 256.0;
	  x *= x; x *= x; x *= x; x *= x;
	  x *= x; x *= x; x *= x; x *= x;
	  return x;
}

#endif // MATHOPTIM_H
