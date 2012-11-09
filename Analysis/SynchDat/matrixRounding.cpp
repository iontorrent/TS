/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * matrixRounding.cpp
 * Exported: roundToI16/roundToI32/convToValidU16/convToFloat
 * Armadillo does not provide any fast rounding routines (yet?).
 * Also contains a matrix conversion specific to this task that
 * clamps the resulting matrix to be in [0, 16383].
 *  
 * @author Magnus Jedvert
 * @version 1.1 april 2012
*/

#include <utility>
#include <cmath>
#include <functional>
#include <omp.h>
#include "matrixRounding.h"
#include "xmmintrin.h"

using namespace std;

typedef uint16_t u16;
typedef int16_t i16;
typedef int32_t i32;

#define rep(i, n) for (size_t i = 0; i < size_t(n); ++i)
#define ALWAYS_INLINE __attribute__((always_inline))

typedef float v4sf __attribute__((vector_size(16), aligned(16)));
typedef int32_t v4si __attribute__((vector_size(16), aligned(16)));
typedef int16_t v8hi __attribute__((vector_size(16), aligned(16)));

union f4vector {
    v4sf v;
    float f[4];
    float ALWAYS_INLINE sum() const {
        return (f[0] + f[1]) + (f[2] + f[3]);
    }   
};

static ALWAYS_INLINE v4sf fillSame(float f)
    { return (v4sf){f, f, f, f}; }

static v8hi ALWAYS_INLINE roundAndPack(const v4sf &f1, const v4sf &f2) {
  return (v8hi)_mm_packs_epi32(                      /* pack 8 16-bit integers from... */
			 _mm_cvtps_epi32(f1),  /* ...4 rounded floats */
			 _mm_cvtps_epi32(f2)); /* ...4 rounded floats */
}

static v4sf ALWAYS_INLINE bound(const v4sf &a, const v4sf &b, const v4sf &f) {
    return _mm_max_ps(a, _mm_min_ps(b, f)); // bound f to [a, b]
}

template<typename T>
static bool is128bitAligned(const T* t) {
    return uint64_t(t) % 16 == 0;
}

template<typename T, typename F>
static ALWAYS_INLINE Mat<T> matrixConversion(const Mat<F> &M, void (*fun)(const F*, T*, const size_t)) {
    const size_t H = M.n_rows;
    const size_t W = M.n_cols;
    Mat<T> res(H, W);
    fun( M.colptr(0), res.colptr(0), H*W );  
    return res;
}

// will round N floats from src to N i32 integers in dst
static void roundToI32(const float src[], i32 dst[], const size_t N) {
    const size_t E = is128bitAligned(src) && is128bitAligned(dst) ? N/4: 0;
    //    #pragma omp parallel for
    rep(j, E) ((v4si*)dst)[j] = (v4si)_mm_cvtps_epi32( ((v4sf*)src)[j] ); // fast sse
    //    #pragma omp parallel for
    for (size_t i = E*4; i < N; ++i) dst[i] = round(src[i]); // slow
}

// will round N floats from src to N i16 integers in dst
static void ALWAYS_INLINE roundToI16(const float src[], i16 dst[], const size_t N) {
    const v4sf *const src4 = (v4sf*)src;
    // if memory aligned, make fast sse:
    const size_t E = is128bitAligned(src) && is128bitAligned(dst) ? N/8: 0;
    //    #pragma omp parallel for
    rep(j, E) ((v8hi*)dst)[j] = roundAndPack(src4[j*2+0], src4[j*2+1]); // fast sse
    //    #pragma omp parallel for
    for (size_t i = E*8; i < N; ++i) dst[i] = round(src[i]); // slow
}

static void ALWAYS_INLINE convToFloat(const i16 src[], float dst[], const size_t N) {
    const v8hi VZERO = {0, 0, 0, 0, 0, 0, 0, 0};
    v4sf *const dst4 = (v4sf*)dst;

    // if memory aligned, make fast sse:
    const size_t E = is128bitAligned(src) && is128bitAligned(dst) ? N/8: 0;
    //    #pragma omp parallel for
    rep(j, E) {  // fast sse
        const v8hi srcJ = ((v8hi*)src)[j];
        dst4[j*2+0] = (v4sf)_mm_cvtepi32_ps( _mm_unpacklo_epi16( (__m128i)srcJ, (__m128i)VZERO ) );
        dst4[j*2+1] = (v4sf)_mm_cvtepi32_ps( _mm_unpackhi_epi16( (__m128i)srcJ, (__m128i)VZERO ) );    
    }    
    //    #pragma omp parallel for
    for (size_t i = E*8; i < N; ++i) dst[i] = float( src[i] ); // slow
}

// will round and bound N floats from src to N u16 integers in dst
static void convToValidU16(const float src[], u16 dst[], const size_t N) {
    const v4sf VZERO = fillSame(0.0f);
    const float MAXF = 16383.0f;
    const v4sf MAX4F = fillSame(MAXF);

    const v4sf *const src4 = (v4sf*)src;
 
    // if memory aligned, make fast sse:
    const size_t E = is128bitAligned(src) && is128bitAligned(dst) ? N/8: 0;
    //    #pragma omp parallel for
    rep(j, E) ((v8hi*)dst)[j] = roundAndPack(
                bound(VZERO, MAX4F, src4[j*2+0]), 
                bound(VZERO, MAX4F, src4[j*2+1]) );     // fast sse
    // slow reference: (used when sse is not possible)
    //    #pragma omp parallel for
    for (size_t i = E*8; i < N; ++i) dst[i] = round(max(0.0f, min(MAXF, src[i])));
}

/**
 * roundToI32 - Converts Mat<float> to Mat<i32> using rounding instead of 
 * truncation. Will use fast sse-instructions if M is 128-bits aligned.
 */
Mat<i32> roundToI32(const Mat<float> &M) {
    return matrixConversion<i32, float>(M, &roundToI32);
}

/**
 * roundToI16 - Converts Mat<float> to Mat<i16> using rounding instead of 
 * truncation. Will use fast sse-instructions if M is 128-bits aligned.
 */
Mat<i16> roundToI16(const Mat<float> &M) {
    return matrixConversion<i16, float>(M, &roundToI16);
}

/**
 * convToValidU16 - Converts Mat<float> to Mat<u16>. Uses rounding instead of 
 * truncation and clamps the resulting values to [0, 16383]. Will use 
 * fast sse-instructions if M is 128-bits aligned.
 */
Mat<u16> convToValidU16(const Mat<float> &M) {
    return matrixConversion<u16, float>(M, &convToValidU16);
}

/**
 * convToFloat - Exactly like conv_to<Mat<float>>::from, but faster
 */
Mat<float> convToFloat(const Mat<i16> &M) {
    return matrixConversion<float, i16>(M, &convToFloat);
}

