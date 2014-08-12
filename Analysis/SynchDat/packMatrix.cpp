/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * packMatrix.cpp
 * Exported interface: packMatrix
 * This is the file that contains all linear algebra on the compression side
 * (finding a good basis, partioning the columns into good/bad, calculating scores etc)
 * 
 * @author Magnus Jedvert
 * jedvert@gmail.com
 * @version 1.1 april 2012
*/

#include <vector>
#include <armadillo>
#include <stdint.h>
#include <cassert>
#include <cmath>
#include "secantFind.h"
#include "matrixRounding.h"
#include "packMatrix.h"
#include "BitHandler.h"
//#include "ByteHandler.h"
#include "HuffmanEncode.h"
#include "compression.h"
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include <x86intrin.h>

using namespace std;
using namespace arma;

#define rep(i, n) for (size_t i = 0; i < size_t(n); ++i)
#define unlikely(x)     __builtin_expect((x),0)
#define likely(x)       __builtin_expect((x),1)
#define all(c) (c).begin(), (c).end()
#define ALWAYS_INLINE __attribute__((always_inline))

typedef int16_t i16;
typedef int32_t i32;

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

template<typename T>
static bool is128bitAligned(const T* t) {
    return uint64_t(t) % 16 == 0;
}

// ========================= linear algebra part ===============================
// these parameters adjust the error
//static const float ERR_AIM = 49.0;  // aiming at this Frobenius norm error
static const float ERR_AIM = 64.0;  // aiming at this Frobenius norm error
static const int RANK_GOOD = 8;    // number of eigenvectors used for ordinary columns

// shouldnt need to change these:
static const size_t N_SAMPLE = 2000;  // number of columns sampled when creating basis
static const size_t MIN_NBAD = 10; // min numbers of columns represented with full rank

/*  Reference implementation, slower
fmat sampleGoodBasis(const fmat &data, const int nSample) {
    const fmat sampleData = data.cols( linspace<uvec>(0, data.n_cols-1, nSample) );
    fmat U, V;
    fvec S;
    svd_econ(U, S, V, sampleData, 'l');
    return U;
}
*/

/**
 * sampleGoodBasis - Creates and returns a good basis for input argument data.
 * Uses the left eigenvectors of a random subset of nSample columns from data.
 */
static fmat sampleGoodBasis(const Mat<u16> &data, const int nSample) {
    const int L = data.n_rows;

    // random subset of columns:
    fmat sampleData = conv_to<fmat>::from( data.cols( linspace<uvec>(0, data.n_cols-1, nSample) ) );  

    // make a transformation matrix that represents taking adjacent differences
    // between each row (except the first row):
    mat P = eye(L, L);
    P.diag(-1) = -ones(L-1);

    // same as sampleData = P * sampleData (adjacent differences between rows)
    sampleData.rows(1, L-1) -= sampleData.rows(0, L-2);     

    // this is the time consuming step 
    // numerical stable with floats because values are much smaller now, 
    // convert resulting small matrix to double when done:
    mat M = conv_to<mat>::from(sampleData * sampleData.t());  

    // redo the effect of P before calculating eigenvectors 
    M = P.i() * M * P.t().i(); // M is now original sampleData * sampleData.t()

    vec eigenvalues;    
    mat eigenvectors;
    eig_sym(eigenvalues, eigenvectors, M); 

    // flip because eigenvectors are returned in ascending order:
    return conv_to<fmat>::from( fliplr(eigenvectors) ); 
}
/**
 * getPivot - Calculates and returns a pivot value based on input vector err.
 * The pivot value is used for dividing the columns in good and bad.
 */
static float getPivot(const fvec &err, size_t L, double errCon) {
    const size_t N = err.n_elem;    
    static const size_t MIN_NBAD = 1;  // minimal number of bad columns
    //static const double HIGH_LIM = ERR_AIM * 4.0;  // 4 times as high seems reasonable
    static const double HIGH_LIM = 4 * errCon;  // 4 times as high seems reasonable
    //static const double MAX_AVG_ERR = ERR_AIM - 5.0; // leave some margin for later rounding
    static const double MAX_AVG_ERR = errCon - 5.0; // leave some margin for later rounding
    
    float sumGood = 0.0;        
    vector<float> V;
    V.reserve(N);
    const float sumLim = MAX_AVG_ERR * N * L;
    const float midLim = MAX_AVG_ERR * L;
    const float highLim = HIGH_LIM * L;

    size_t highCnt = 0;
    rep(i, N) {
        const float e = err(i);
            
        if ( likely(e < midLim) ) sumGood += e; // never consider this column as bad
        else if ( likely(e < highLim ) ) V.push_back( e ); // this is a candidate
        else ++highCnt;  // always consider this column as bad
    }
    if ( highCnt == 0 && V.size() == 0 ) return highLim;

    sort( all(V) );
    const int E = (int)V.size() + min( int(highCnt - MIN_NBAD), 0);
    rep(i, E) {
        sumGood += V[i];
        if ( unlikely(sumGood > sumLim ) ) {
            return V[i];  // error is growing too large, pivot here
        }
    }
    // error still small, default with MIN_NBAD rows:
    return E == (int)V.size() ? highLim: V[E];
}

/*
Reference implementation, slower
static double roundingError(const fmat &M, double factor_) {
    const float rcFactor = 1.0 / factor_;
    const float factor = factor_;
    // sum of squared error when using scale factor:
    return accu( square( M - 
        conv_to<fmat>::from(roundToI32(M * rcFactor)) * factor 
    ) );
}
*/

/**
 * roundingError - Returns the squared error of M when converting to integer
 * from float using scale factor factor_
 */
static double roundingError(const fmat &M, double factor_) {
    const float *const ptr = M.colptr(0);
    const v4sf *const ptrV = (v4sf*)ptr;    
    const size_t N =  M.n_elem;

    const v4sf factor = fillSame(factor_);
    const v4sf rcFactor = fillSame(1.0 / factor_);

    size_t E = is128bitAligned(ptr) ? N / 4: 0;
    f4vector sumV = {fillSame(0.0f)};
    rep(k, E) { // fast sse
        const v4sf s = ptrV[k]; 
        const v4sf err = s - factor * (v4sf)_mm_cvtepi32_ps ( _mm_cvtps_epi32 (s * rcFactor) );
        sumV.v += err * err;
    }

    float sum = sumV.sum();
    for (size_t i = E * 4; i < N; ++i) { // slow
        const float diff = round(ptr[i] / factor_) * factor_ - ptr[i];
        sum += diff * diff;
    }
    return sum;
}

/*
static double modRoundErr(const fmat &basis, const fmat &score, double factor)
{
  float rFactor = 1.0/factor;
  const Mat<i16> rScore = roundToI16(score.t() * rFactor );
  fmat tmp = (basis * factor ) * rScore.t();
  frowvec roundErr = sum(square(basis * score - tmp));
  return accu(roundErr);
}
*/

static double modRoundErrV2(const Mat<uint16_t> &data,const fmat &basis,const fmat &score, double factor)
{
  float rFactor = 1.0/factor;
  const Mat<i16> rScore = roundToI16(score.t() * rFactor );
  fmat tmp = (basis * factor) * rScore.t();
  frowvec roundErr = sum(square(data - tmp));
  return accu(roundErr);
}

static size_t iceil(size_t p, size_t q) {
    return (p + q - 1) / q;
}

/*
Reference implementation, slower
static void mulAndOrthError(fmat &score, frowvec &orthError, const fmat &basisGood, const Mat<u16> &data16) {
    const fmat data = conv_to<fmat>::from(data16);
    score = basisGood.t() * data;  // compute the scores for the small basis: (time consuming)
    orthError = sum( square(data - basisGood * score) );   // calculate the error for each row with this basis: (time consuming)
}
*/

/*
static void mulAndOrthErrorV2(fmat &score,fvec &orthError, const fmat &basisGood, const Mat<u16> data16) {
  score = basisGood.t() * data16;
  orthError = sum( square(data16 - basisGood * score ) );
}

*/
#ifdef __INTEL_COMPILER
#pragma optimization_level 2
#endif
// hand coded version:
static void mulAndOrthError(fmat &score, fvec &orthError, const fmat &basisGood, const Mat<u16> &data16) {
    const v4si VZERO = {0,0,0,0};
    const int L = data16.n_rows;
    const size_t N = data16.n_cols;

    const int L4 = iceil(L, 4);
    const int L8 = iceil(L, 8);

    // alloc memory for output:
    //score.set_size(RANK_GOOD, N);
    score.set_size(basisGood.n_cols, N);
    orthError.set_size(N);

    // ------ we need to have the basis aligned and in another format ------
    //f4vector base[L4][RANK_GOOD];
    f4vector base[L4][basisGood.n_cols];
    rep(i, L4) {
        rep(j, basisGood.n_cols) {
            rep(k, 4) {
                const int x = i * 4 + k;
                base[i][j].f[k] = x >= L ? 0.0f: basisGood(x, j); // pad with zeroes
            }
        }
    }
    // ---------------------------------------------------------------------

    const u16 *dataPtr = data16.colptr(0);    
    //#pragma omp parallel for
    rep(i, N) {		
        // ---- unpack and convert one row of data to float into local buffer -----
        v4sf rowV[L8*2];        
        rep(k, L8) {
            const v8hi rowK = (v8hi)_mm_loadu_ps( (float*)(dataPtr + i * L + k * 8) );
            rowV[k*2+0] = _mm_cvtepi32_ps( _mm_unpacklo_epi16( (__m128i)rowK, (__m128i)VZERO ) );
            rowV[k*2+1] = _mm_cvtepi32_ps( _mm_unpackhi_epi16( (__m128i)rowK, (__m128i)VZERO ) );               
        }
        for (int k = L; k % 8 != 0; ++k) ((float*)rowV)[k] = 0.0f;
        // -------------------------------------------------------------------------

#ifndef __INTEL_COMPILER
// ICC has issues with tmpScore[ii].v lines (variable length arrays)
        // -------------------------- calculate score ------------------------------
        //f4vector tmpScore[ RANK_GOOD ];
        f4vector tmpScore[ basisGood.n_cols ];
        //rep(k, RANK_GOOD) tmpScore[k].v = (v4sf)VZERO; 
        rep(k, basisGood.n_cols) tmpScore[k].v = (v4sf)VZERO; 
        rep(k, L4) {
            const v4sf rowK = rowV[k];
            const f4vector *const B = base[k];
            rep(ii, basisGood.n_cols) tmpScore[ii].v += rowK * B[ii].v;
        }

        rep(j, basisGood.n_cols) {
            const float f = tmpScore[j].sum();
            score(j, i) = f;  // save to output matrix
            tmpScore[j].v = fillSame(f);
        }
        // --------------------------------------------------------------------------         

        // ------------------------ calculate orthError -----------------------------
        f4vector lengthV = { (v4sf)VZERO }; 
        rep(k, L4) {
            v4sf diff = rowV[k];
            const f4vector *const B = base[k];
            rep(ii, basisGood.n_cols) diff -= tmpScore[ii].v * B[ii].v;            
            lengthV.v += diff * diff;
        }
        orthError(i) = lengthV.sum(); // store to output vector
        // --------------------------------------------------------------------------
#else
#pragma message "WARNING, code not implemented for ICC"
#endif
    }
}

struct TMPFunction {
    const fmat &scoreGood, &scoreBad;
    TMPFunction(const fmat &scoreGood, const fmat &scoreBad): scoreGood(scoreGood), scoreBad(scoreBad) {}
    double operator()(double factor) const {
        return roundingError(scoreGood, factor) + roundingError(scoreBad, factor); 
    }
};


struct TMPGoodFunction {
  const fmat &scoreGood;
  TMPGoodFunction(const fmat &scoreGood): scoreGood(scoreGood) {}
    double operator()(double factor) const {
      return roundingError(scoreGood, factor);
      }
};

struct TMPGoodFunctionV2 {
  const fmat &basisGood;
  const fmat &scoreGood;
  const Mat<uint16_t> &data;
  TMPGoodFunctionV2(const fmat &basisGood, const fmat &scoreGood,const Mat<uint16_t> &data): basisGood(basisGood),scoreGood(scoreGood),data(data){}
  double operator()(double factor) const {
    return modRoundErrV2(data,basisGood,scoreGood, factor);
  }
};

/**
 * packMatrix - takes as input a pointer to a packed array of uint16_t and 
 * interprets this as a LxN matrix (column-major order). The data for the 
 * resulting matrix approximation is pushed onto input argument dst.
 */
void packMatrix(BytePacker &dst, const u16 *input_data, size_t N, size_t L) {
  // interpret the input pointer as a matrix:
  const Mat<u16> data16((u16*)input_data, L, N, false, true);
  // initialize bases:
  fmat basis = sampleGoodBasis( data16, min(N, N_SAMPLE) );
  //    basis = basis.cols(0, ((RANK_GOOD * 2) -1));
  size_t rankBad = L;
  const fmat basisGood = basis.cols(0, RANK_GOOD - 1);

  fmat score;  // will hold the scores for the small basis
  fvec orthError; // will hold the error using the above score
  fvec badErr; // hold the err for the bad traces using full-rank
  fvec goodErr; // hold the err for the bad traces using full-rank
  mulAndOrthError(score, orthError, basisGood, data16);  // calculate them
  // calculate pivot from orthError:
  const float pivot = getPivot(orthError, L, ERR_AIM);
  // partion the columns based on the pivot:
  const uvec badIdx = find( orthError >= pivot );
  const uvec goodIdx = find( orthError < pivot );
  const int nGood = goodIdx.n_elem;
  const int nBad = badIdx.n_elem;

  // extract the bad columns:
  const fmat dataBad = conv_to<fmat>::from( data16.cols(badIdx) );

  const fmat scoreGood = score.cols(goodIdx);  // copy existing scores
  const fmat scoreBad = basis.t() * dataBad;  // calculate with full basis

  const float sumErr = accu( orthError.elem(goodIdx) );

  // remaining error minus margin (0.1) for final rounding:
  const float aim = (ERR_AIM - 0.1) * N * L - sumErr;
    
  // make an initial linear guess. 0.05 was found empirically   
  const float initialGuess = 0.05 * aim / N; 

  // wrapper to sum error for both good and bad:
  /*
    function< double(double) > errFun = [&](double factor) { 
    return roundingError(scoreGood, factor) + roundingError(scoreBad, factor); 
    }; 
  */
  TMPFunction errFun(scoreGood, scoreBad);
  // find the value of factor that gives almost exactly 36.0 as error:
  float factor = secantFind( errFun, 3, aim, 0.0, 0.0, initialGuess );
  //    float factor = errFun( secantFind, 3, aim, 0.0, 0.0, initialGuess );
  const bool uses16Bit = factor > sqrt(float(L)) / 2.0f + 1.0f; // if scores will fit into 16 bit

  // store small header with key numbers:
  vector<size_t> buffHeader(5);
  buffHeader[0] = nGood;
  buffHeader[1] = nBad;
  buffHeader[2] = RANK_GOOD;
  //    buffHeader[3] = L;  // = rankBad
  buffHeader[3] = rankBad;
  buffHeader[4] = uses16Bit; 
  dst.push( buffHeader );

  dst.push<float>( basis * factor ); // store adjusted basis:
  // store partion of columns:
  Col<u8> partion = zeros< Col<u8> >(N);
  rep(i, nBad) partion( badIdx(i) ) = 1;
  dst.push( partion );

  // store scores:
  const float rFactor = 1.0f / factor;
  if ( uses16Bit ) {
    dst.push< i16 >( roundToI16(scoreGood.t() * rFactor) ); 
    dst.push< i16 >( roundToI16(scoreBad.t() * rFactor) );
  }
  else {
    dst.push< i32 >( roundToI32(scoreGood.t() * rFactor) ); 
    dst.push< i32 >( roundToI32(scoreBad.t() * rFactor) );
  }
  cout << "Good: " << nGood << " Bad: " << nBad << " Factor: " << factor << " Pivot: " << pivot <<endl;
}

void packMatrixPlus(BytePacker &dst, const u16 *input_data, size_t N, size_t L, double errCon, int rankGood, float piv) {
  // interpret the input pointer as a matrix:
  const Mat<u16> data16((u16*)input_data, L, N, false, true);
  // initialize bases:
  fmat basis = sampleGoodBasis( data16, min(N, N_SAMPLE) );
  size_t rankBad = L;
  const fmat basisGood = basis.cols(0, rankGood - 1);

  fmat score;  // will hold the scores for the small basis
  fvec orthError; // will hold the error using the above score
  fvec badErr;
  fvec goodErr;
  fmat score2;
  mulAndOrthError(score, orthError, basisGood, data16);  // calculate them
  mulAndOrthError(score2, badErr, basis, data16);

  // calculate pivot from orthError:
  float pivot;
  if (piv > 0)
    pivot = piv * L;
  else
   pivot = getPivot(orthError, L, errCon);
  //partion the columns based on the pivot:
  fmat diff = data16 - basisGood * score;
  fvec abs(N);
  for(size_t i=0;i<diff.n_cols;i++){
    fvec tmp = diff.col(i);
    abs[i] = getMAB(tmp);
  }
  
  //const uvec badIdx = find( abs >= sqrt(pivot/L) );  
  const uvec badIdx = find( orthError >= pivot );
  //const uvec goodIdx = find( abs < sqrt(pivot/L) );
  const uvec goodIdx = find( orthError < pivot );
  const int nGood = goodIdx.n_elem;
  const int nBad = badIdx.n_elem;
    
  // extract the bad columns and do delta Compression:
  const Mat<uint16_t> dataBadMat = data16.cols(badIdx);
  const vector<uint16_t> dataBadVec(dataBadMat.begin(),dataBadMat.end());
  const Mat<uint16_t> dataGoodMat = data16.cols(goodIdx);
  const fmat scoreGood = score.cols(goodIdx);  // copy existing scores
  // this is the total error so far: (will increase even more with rounding)
  //const float sumErr = accu( orthError.elem(goodIdx) ) + accu( square(dataBad - basis * scoreBad) );
  //const float sumErr = accu( orthError.elem(goodIdx) );

  // remaining error minus margin (0.1) for final rounding:
  //const float aim = fabs((ERR_AIM - 0.1) * nGood * L - sumErr);
  //const float aim = (ERR_AIM - 0.1) * nGood * L - sumErr;
  //const float aim = (ERR_AIM - 0.1) * N * L - sumErr;
  //const float aim = (errCon - 0.1) * nGood * L - sumErr;
  //const float aim = (errCon - .1) * nGood * L;
  const float aim = (errCon /2 ) * nGood * L;
  if ( aim < 0) {
    cout <<"aim error is negative ..." << endl;
    //exit (1);
  }
  // make an initial linear guess. 0.05 was found empirically   
  const float initialGuess = 0.05 * aim / nGood; 
  //const float initialGuess = 0.05 * aim / N; 

  //TMPFunction errFun(scoreGood, scoreBad);
  //TMPGoodFunction errFun(scoreGood);
  TMPGoodFunctionV2 errFun(basisGood,scoreGood,dataGoodMat);
  // find the value of factor that gives almost exactly 36.0 as error:
  //float factor = secantFind( errFun, 3, aim, 0.0, 0.0, initialGuess );
  float factor = secantFind( errFun, 6, aim, 0.0, 0.0, initialGuess );

  cout <<"aim = " << aim << "true err = " << modRoundErrV2(dataGoodMat,basisGood,scoreGood,factor) << endl;
  const bool uses16Bit = factor > sqrt(float(L)) / 2.0f + 1.0f; // if scores will fit into 16 bit

  // store small header with key numbers:
  vector<size_t> buffHeader(5);
  buffHeader[0] = nGood;
  buffHeader[1] = nBad;
  buffHeader[2] = rankGood;
  buffHeader[3] = rankBad;
  buffHeader[4] = uses16Bit; 
  dst.push( buffHeader );
        
  // store partion of columns:
  Col<u8> partion = zeros< Col<u8> >(N);
  rep(i, nBad) partion( badIdx(i) ) = 1;
  dst.push( partion );
    
  //b4 Compression - good traces
  if ( nGood > 0 ) {
    vector<uint8_t> tmpGood; 
    BytePacker gbp(tmpGood);
    // Only need to store basisGood
    gbp.push<float> (basisGood * factor);
    // store scores:
    const float rFactor = 1.0f / factor;
    if ( uses16Bit ) {
      const Mat<int16_t> tmp = roundToI16(scoreGood.t() * rFactor );
      gbp.push<int16_t>(tmp);
    }
    else {
      const Mat<int32_t> tmp = roundToI32(scoreGood.t() * rFactor);
      gbp.push<int32_t>(tmp);
    }
    gbp.finalize();
   
    vector<uint8_t> gHuffOut;
    BitPacker gBitPack;//(gHuffOut);
    gBitPack.put_compressed(&tmpGood[0],tmpGood.size());
    gBitPack.flush();
    gHuffOut = gBitPack.get_data();
    vector<size_t> gSize(2);
    gSize[0] = gHuffOut.size();
    gSize[1] = tmpGood.size();
    dst.push<size_t>(gSize);
    dst.push<uint8_t>(gHuffOut);
  }

  // Do deltaCompression here for bad data.
  if ( nBad > 0 ) {
    vector<uint8_t> deltaCompressed;
    Compressor cp;
    size_t nRows = 1;
    cp.compress(dataBadVec,nRows,nBad,L,deltaCompressed);
    cout <<"Bad traces::Delta + HME Com.Ratio: " << 2.0 * dataBadVec.size()/deltaCompressed.size() << endl;
    vector<size_t> bSize(1);
    bSize[0] = deltaCompressed.size();
    dst.push<size_t>(bSize);
    dst.push<uint8_t> (deltaCompressed);

  }
  cout << "Good: " << nGood << " Bad: " << nBad << " Factor: " << factor << " Pivot: " << pivot << endl;

}

/* packMatrixPlusV2 - using byte version of delta compression */
void packMatrixPlusV2(BytePacker &dst, const u16 *input_data, size_t N, size_t L, double errCon, int rankGood, float piv) {
  // interpret the input pointer as a matrix:
  const Mat<u16> data16((u16*)input_data, L, N, false, true);
  // initialize bases:
  fmat basis = sampleGoodBasis( data16, min(N, N_SAMPLE) );
  size_t rankBad = L;
  const fmat basisGood = basis.cols(0, rankGood - 1);

  fmat score;  // will hold the scores for the small basis
  fvec orthError; // will hold the error using the above score
  fvec badErr;
  fvec goodErr;
  fmat score2;
  mulAndOrthError(score, orthError, basisGood, data16);  // calculate them
  mulAndOrthError(score2, badErr, basis, data16);

  float pivot;
  if (piv > 0)
    pivot = piv;
  else
   pivot = getPivot(orthError, L, errCon);

  //partion the columns based on the pivot:
  const uvec badIdx = find( orthError >= pivot );
  const uvec goodIdx = find( orthError < pivot );
  const int nGood = goodIdx.n_elem;
  const int nBad = badIdx.n_elem;
    
  fvec badErr2 = badErr.elem(badIdx);
  goodErr = orthError.elem(goodIdx);
  SampleQuantiles<float> bErr(5000);
  SampleQuantiles<float> gErr(5000);
  SampleStats<float> bMean;
  SampleStats<float> gMean;
  for(size_t i = 0; i< badErr2.size();i++)
    {
      bErr.AddValue(badErr2.at(i));
      bMean.AddValue(badErr2.at(i));
    }
  cout <<"Median bad err: " << bErr.GetMedian() << "Mean bad Err: " << bMean.GetMean() << endl;
  for(size_t i = 0;i < goodErr.size();i++)
    {
      gErr.AddValue(goodErr[i]);
      gMean.AddValue(goodErr[i]);
    }
  cout <<"Median good err: " << gErr.GetMedian() << "Mean good Err: "<< gMean.GetMean() << endl;
  
  // extract the bad columns and do delta Compression:
  const Mat<uint16_t> dataBadMat = data16.cols(badIdx);
  const vector<uint16_t> dataBadVec(dataBadMat.begin(),dataBadMat.end());
  
  const fmat scoreGood = score.cols(goodIdx);  // copy existing scores
    
  // this is the total error so far: (will increase even more with rounding)
  const float sumErr = accu( orthError.elem(goodIdx) );

  // remaining error minus margin (0.1) for final rounding:
  //const float aim = (errCon - 0.1) * nGood * L - sumErr;
  const float aim = (errCon - 1) * nGood * L - sumErr;
  if ( aim < 0) {
    cout <<"aim error is negative ..." << endl;
    //exit (1);
  }
  // make an initial linear guess. 0.05 was found empirically   

  const float initialGuess = 0.05 * aim / N; 

  TMPGoodFunction errFun(scoreGood);
  // find the value of factor that gives almost exactly 36.0 as error:
  float factor = secantFind( errFun, 3, aim, 0.0, 0.0, initialGuess );
  const bool uses16Bit = factor > sqrt(float(L)) / 2.0f + 1.0f; // if scores will fit into 16 bit

  // store small header with key numbers:
  vector<size_t> buffHeader(5);
  buffHeader[0] = nGood;
  buffHeader[1] = nBad;
  buffHeader[2] = rankGood;
  buffHeader[3] = rankBad;
  buffHeader[4] = uses16Bit; 
  dst.push( buffHeader );
        
  // store partion of columns:
  Col<u8> partion = zeros< Col<u8> >(N);
  rep(i, nBad) partion( badIdx(i) ) = 1;
  dst.push( partion );
    
  //b4 Compression - good traces
  //int nnGood = 0;
  if ( nGood > 0 ) {
    dst.push<float> (basisGood * factor);
    const float rFactor = 1.0f / factor;
    if ( uses16Bit ) 
      dst.push< i16 >( roundToI16(scoreGood.t() * rFactor) ); 
    else 
      dst.push< i32 >( roundToI32(scoreGood.t() * rFactor) ); 
  }

  // Do deltaCompression here for bad data.
  //int nnBad = 0;
  if ( nBad > 0 ) {
    vector<uint8_t> deltaCompressed;
    Compressor cp;
    size_t nRows = 1;
    cp.byteCompress(dataBadVec,nRows,nBad,L,deltaCompressed);
    cout <<"Bad traces::Delta + HME Com.Ratio: " << 2.0 * dataBadVec.size()/deltaCompressed.size() << endl;
    vector<size_t> bSize(1);
    bSize[0] = deltaCompressed.size();
    dst.push<size_t>(bSize);
    dst.push<uint8_t> (deltaCompressed);
  }
  cout << "Good: " << nGood << " Bad: " << nBad << " Factor: " << factor << " Pivot: " << pivot << endl;

}

// get MAD of the difference

float getMAB(fvec &diff){
  SampleQuantiles<float> tmp(diff.size());
  for(size_t i=0;i<diff.size();i++)
    tmp.AddValue(fabs(diff[i]));
  return tmp.GetMedian();
}

void packMatrixPlusV3(BytePacker &dst, const u16 *input_data, size_t N, size_t L, double errCon, int rankGood, float piv) {
  // interpret the input pointer as a matrix:
  const Mat<u16> data16((u16*)input_data, L, N, false, true);
  // initialize bases:
  fmat basis = sampleGoodBasis( data16, min(N, N_SAMPLE) );
  size_t rankBad = L;
  const fmat basisGood = basis.cols(0, rankGood - 1);

  fmat score;  // will hold the scores for the small basis
  fvec orthError; // will hold the error using the above score
  fvec badErr;
  fvec goodErr;
  fmat score2;
  mulAndOrthError(score, orthError, basisGood, data16);  // calculate them
  mulAndOrthError(score2, badErr, basis, data16);
  // calculate pivot from orthError:
  float pivot;
  if (piv > 0)
    pivot = piv * L;
  else
   pivot = getPivot(orthError, L, errCon);
  //partion the columns based on the pivot:
  fmat diff = data16 - basisGood * score;
  fvec abs(N);
  for(size_t i=0;i<diff.n_cols;i++){
    fvec tmp = diff.col(i);
    abs[i] = getMAB(tmp);
  }
  
  //const uvec badIdx = find( abs >= sqrt(pivot/L) );  
  const uvec badIdx = find( orthError >= pivot );
  //const uvec goodIdx = find( abs < sqrt(pivot/L) );
  const uvec goodIdx = find( orthError < pivot );
  const int nGood = goodIdx.n_elem;
  const int nBad = badIdx.n_elem;
    
  // extract the bad columns and do delta Compression:
  const Mat<uint16_t> dataBadMat = data16.cols(badIdx);
  const vector<uint16_t> dataBadVec(dataBadMat.begin(),dataBadMat.end());
  const Mat<uint16_t> dataGoodMat = data16.cols(goodIdx);
  const fmat scoreGood = score.cols(goodIdx);  // copy existing scores
  // this is the total error so far: (will increase even more with rounding)

  // remaining error minus margin (0.1) for final rounding:
  const float aim = (errCon - 5) * nGood * L;
  //const float aim = (errCon /2 ) * nGood * L;
  if ( aim < 0) {
    cout <<"aim error is negative ..." << endl;
    //exit (1);
  }
  // make an initial linear guess. 0.05 was found empirically   
  const float initialGuess = 0.05 * aim / nGood; 

  TMPGoodFunctionV2 errFun(basisGood,scoreGood,dataGoodMat);
  // find the value of factor that gives almost exactly 36.0 as error:
  //float factor = secantFind( errFun, 3, aim, 0.0, 0.0, initialGuess );
  float factor = secantFind( errFun, 6, aim, 0.0, 0.0, initialGuess );

  cout <<"aim = " << aim << "true err = " << modRoundErrV2(dataGoodMat,basisGood,scoreGood,factor) << endl;
  const bool uses16Bit = factor > sqrt(float(L)) / 2.0f + 1.0f; // if scores will fit into 16 bit

  // store small header with key numbers:
  vector<size_t> buffHeader(5);
  buffHeader[0] = nGood;
  buffHeader[1] = nBad;
  buffHeader[2] = rankGood;
  buffHeader[3] = rankBad;
  buffHeader[4] = uses16Bit; 
  dst.push( buffHeader );

  dst.push<float> (basisGood * factor);        
  // store partion of columns:
  Col<u8> partion = zeros< Col<u8> >(N);
  rep(i, nBad) partion( badIdx(i) ) = 1;
  dst.push( partion );
    
  //b4 Compression - good traces
  if ( nGood > 0 ) {
    // Only need to store basisGood

    // store scores:
    const float rFactor = 1.0f / factor;
    if ( uses16Bit ) {
      dst.push<i16>( roundToI16(scoreGood.t() * rFactor ) );
    }
    else {
      dst.push<i32>( roundToI32(scoreGood.t() * rFactor ) );
    }
  }
  // Do deltaCompression here for bad data.
  if ( nBad > 0 ) {
    vector<uint8_t> deltaCompressed;
    Compressor cp;
    size_t nRows = 1;
    cp.compress(dataBadVec,nRows,nBad,L,deltaCompressed);
    cout <<"Bad traces::Delta + HME Com.Ratio: " << 2.0 * dataBadVec.size()/deltaCompressed.size() << endl;
    vector<size_t> bSize(1);
    bSize[0] = deltaCompressed.size();
    dst.push<size_t>(bSize);
    dst.push<uint8_t>(deltaCompressed);
  }
  cout << "Good: " << nGood << " Bad: " << nBad << " Factor: " << factor << " Pivot: " << pivot << endl;

}
