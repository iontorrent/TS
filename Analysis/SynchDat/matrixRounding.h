/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * matrixRounding.h
 * Exported: Exported: roundToI16/roundToI32/convToValidU16/convToFloat
 * Armadillo does not provide any fast rounding routines (yet?).
 * Also contains a matrix conversion specific to this task that
 * clamps the resulting matrix to be in [0, 16383].
 *  
 * @author Magnus Jedvert
 * @version 1.1 april 2012
*/
#ifndef MATRIXROUNDING_H
#define MATRIXROUNDING_H
#include <armadillo>
#include <stdint.h>
using arma::Mat;

/**
 * roundToI32 - Converts fmat to Mat<i32> using rounding instead of 
 * truncation. Will use fast sse-instructions if M is 128-bits aligned.
 */
Mat<int32_t> roundToI32(const Mat<float> &M);

/**
 * roundToI16 - Converts fmat to Mat<i16> using rounding instead of 
 * truncation. Will use fast sse-instructions if M is 128-bits aligned.
 */
Mat<int16_t> roundToI16(const Mat<float> &M);

/**
 * convToValidU16 - Converts fmat to Mat<u16>. Uses rounding instead of 
 * truncation and clamps the resulting values to [0, 16383]. Will use 
 * fast sse-instructions if M is 128-bits aligned.
 */
Mat<uint16_t> convToValidU16(const Mat<float> &M);

/**
 * convToFloat - Exactly like conv_to<fmat>::from, but faster
 */
Mat<float> convToFloat(const Mat<int16_t> &M);

#endif // MATRIXROUNDING_H
