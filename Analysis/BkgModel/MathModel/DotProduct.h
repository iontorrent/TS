/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DOTPRODUCT_H
#define DOTPRODUCT_H

#include <xmmintrin.h>
#include <pmmintrin.h>

#ifndef __SSE__
//"SSE not available. Using slow dot product. If your processor supports SSE, use -march=native compiler flag."
inline float DotProduct( int N, float* X, float* Y )
{
    float dot0=0.f, dot1=0.f, dot2=0.f, dot3=0.f;
    float *stX, *stX0=X+N;

    stX = X + ((N>>2)<<2);
    if (X != stX)
    {
        do
        {
            //ATL_pfl1R(X+16);
            dot0 += *X * *Y;
            dot1 += X[1] * Y[1];
            dot2 += X[2] * Y[2];
            dot3 += X[3] * Y[3];
            X += 4;
            Y += 4;
        }
        while (X != stX);
        dot0 += dot1;
        dot2 += dot3;
        dot0 += dot2;
    }
    while (X != stX0) dot0 += *X++ * *Y++;
    return(dot0);
}
#else

#ifndef __SSE4_1__
//"SSE4 not available. Using slower dot product. If your processor supports SSE4.1, use -march=native compiler flag."

inline float DotProduct( int N, float* X, float* Y )
{
    float dot = 0.f;

    __m128 dot0, dot1, dot2, dot3;

    float *stX, *stX0=X+N;

    dot3 = _mm_setzero_ps();

    stX = X + ((N>>2)<<2);
    if (X != stX)
    {
        do
        {
            dot0 = _mm_loadu_ps( X );
            dot1 = _mm_loadu_ps( Y );

            dot2 = _mm_mul_ps( dot0, dot1 );
            dot2 = _mm_hadd_ps( dot2, dot2 );

            dot3 = _mm_add_ps(dot3, dot2 );

            X += 4;
            Y += 4;
        }
        while (X != stX);

        dot3 = _mm_hadd_ps( dot3, dot3 );
        _mm_store_ss( &dot, dot3 );
    }
    while (X != stX0) dot += *X++ * *Y++;
    return(dot);
}
#else
#include <smmintrin.h>
inline float DotProduct( int N, float* X, float* Y )
{
    float dot=0.f;
    float *stX0=X+N;

    int nIter = N>>2;

    __m128 X_sse, Y_sse, XY_sse;
    __m128 dot_sse = _mm_setzero_ps();


    while (nIter--)
    {
        X_sse = _mm_loadu_ps( X );
        Y_sse = _mm_loadu_ps( Y );

        XY_sse = _mm_dp_ps( X_sse, Y_sse,  0xF1 );
        dot_sse = _mm_add_ss( dot_sse, XY_sse );

        X += 4;
        Y += 4;
    }
    _mm_store_ss( &dot, dot_sse );

    while (X != stX0) dot += *X++ * *Y++;

    return(dot);
}
#endif //#ifndef __SSE4_1__
#endif //#ifndef __SSE__

#endif // DOTPRODUCT_H
