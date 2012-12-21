/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "TreephaserSSE.h"

#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
#include <pmmintrin.h>


using namespace std;

#define SHUF_PS(reg, mode) _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(reg), mode))

#define AD_STATE_OFS (0*MAX_VALS*4*sizeof(float)+16)
#define AD_PRED_OFS (1*MAX_VALS*4*sizeof(float)+16)
#define AD_NRES_OFS (2*MAX_VALS*4*sizeof(float)+16)
#define AD_PRES_OFS (3*MAX_VALS*4*sizeof(float)+16)

namespace {

ALWAYS_INLINE float Sqr(float val) {
  return val*val;
}

inline void setZeroSSE(void *dst, int size) {
  __m128i r0 = _mm_setzero_si128();
  while((size & 31) != 0) {
    --size;
    ((char RESTRICT_PTR)dst)[size] = char(0);
  }
  while(size > 0) {
    _mm_store_si128((__m128i RESTRICT_PTR)((char RESTRICT_PTR)dst+size-16), r0);
    _mm_store_si128((__m128i RESTRICT_PTR)((char RESTRICT_PTR)dst+size-32), r0);
    size -= 32;
  }
}

inline void copySSE(void *dst, void *src, int size) {
  while((size & 31) != 0) {
    --size;
    ((char RESTRICT_PTR)dst)[size] = ((char RESTRICT_PTR)src)[size];
  }
  while(size > 0) {
    __m128i r0 = _mm_load_si128((__m128i RESTRICT_PTR)((char RESTRICT_PTR)src+size-16));
    __m128i r1 = _mm_load_si128((__m128i RESTRICT_PTR)((char RESTRICT_PTR)src+size-32));
    _mm_store_si128((__m128i RESTRICT_PTR)((char RESTRICT_PTR)dst+size-16), r0);
    _mm_store_si128((__m128i RESTRICT_PTR)((char RESTRICT_PTR)dst+size-32), r1);
    size -= 32;
  }
}

inline float sumOfSquaredDiffsFloatSSE(float RESTRICT_PTR src1, float RESTRICT_PTR src2, int count) {
  float sum = 0.0f;
  while((count & 3) != 0) {
    --count;
    sum += Sqr(src1[count]-src2[count]);
  }
  __m128 r0 = _mm_load_ss(&sum);
  while(count > 0) {
    __m128 r1 = _mm_load_ps(&src1[count-4]);
    r1 = _mm_sub_ps(r1, *((__m128 RESTRICT_PTR)(&src2[count-4])));
    count -= 4;
    r1 = _mm_mul_ps(r1, r1);
    r0 = _mm_add_ps(r0, r1);
  }
  __m128 r2 = r0;
  r0 = _mm_movehl_ps(r0, r0);
  r0 = _mm_add_ps(r0, r2);
  r0 = _mm_unpacklo_ps(r0, r0);
  r2 = r0;
  r0 = _mm_movehl_ps(r0, r0);
  r0 = _mm_add_ps(r0, r2);
  _mm_store_ss(&sum, r0);
  return sum;
}

inline void sumVectFloatSSE(float RESTRICT_PTR dst, float RESTRICT_PTR src, int count) {
  while((count & 3) != 0) {
    --count;
    dst[count] += src[count];
  }
  while(count > 0) {
    __m128 r0 = _mm_load_ps(&dst[count-4]);
    r0 = _mm_add_ps(r0, *((__m128 RESTRICT_PTR)(&src[count-4])));
    _mm_store_ps(&dst[count-4], r0);
    count -= 4;
  }
}

};


// ---------

TreephaserSSE::TreephaserSSE(const ion::FlowOrder& flow_order)
  : flow_order_(flow_order)
{
  num_flows_ = flow_order.num_flows();

  // For some perverse reason cppcheck does not like this loop
  //for (int path = 0; path <= MAX_PATHS; ++path)
  //  sv_PathPtr[path] = &(sv_pathBuf[path]);
  sv_PathPtr[0] = &(sv_pathBuf[0]);
  sv_PathPtr[1] = &(sv_pathBuf[1]);
  sv_PathPtr[2] = &(sv_pathBuf[2]);
  sv_PathPtr[3] = &(sv_pathBuf[3]);
  sv_PathPtr[4] = &(sv_pathBuf[4]);
  sv_PathPtr[5] = &(sv_pathBuf[5]);
  sv_PathPtr[6] = &(sv_pathBuf[6]);
  sv_PathPtr[7] = &(sv_pathBuf[7]);
  sv_PathPtr[8] = &(sv_pathBuf[8]);

  ad_MinFrac[0] = ad_MinFrac[1] = ad_MinFrac[2] = ad_MinFrac[3] = 1e-6f;

  int nextIdx[4];
  nextIdx[3] = nextIdx[2] = nextIdx[1] = nextIdx[0] = short(num_flows_);
  for(int flow = num_flows_-1; flow >= 0; --flow) {
    nextIdx[flow_order_.int_at(flow)] = flow;
    ts_NextNuc[0][flow] = (short)(ts_NextNuc4[flow][0] = nextIdx[0]);
    ts_NextNuc[1][flow] = (short)(ts_NextNuc4[flow][1] = nextIdx[1]);
    ts_NextNuc[2][flow] = (short)(ts_NextNuc4[flow][2] = nextIdx[2]);
    ts_NextNuc[3][flow] = (short)(ts_NextNuc4[flow][3] = nextIdx[3]);
  }

  ts_StepCnt = 0;
  for(int i = STEP_SIZE << 1; i < num_flows_; i += STEP_SIZE) {
    ts_StepBeg[ts_StepCnt] = (ts_StepEnd[ts_StepCnt] = i)-(STEP_SIZE << 1);
    ts_StepCnt++;
  }
  ts_StepBeg[ts_StepCnt] = (ts_StepEnd[ts_StepCnt] = num_flows_)-(STEP_SIZE << 1);
  ts_StepEnd[++ts_StepCnt] = num_flows_;
  ts_StepBeg[ts_StepCnt] = 0;

}


void TreephaserSSE::SetModelParameters(double cf, double ie)
{
  double dist[4] = { 0.0, 0.0, 0.0, 0.0 };

  for(int flow = 0; flow < num_flows_; ++flow) {
    dist[flow_order_.int_at(flow)] = 1.0;
    ts_Transition4[flow][0] = ts_Transition[0][flow] = float(dist[0]*(1-ie));
    dist[0] *= cf;
    ts_Transition4[flow][1] = ts_Transition[1][flow] = float(dist[1]*(1-ie));
    dist[1] *= cf;
    ts_Transition4[flow][2] = ts_Transition[2][flow] = float(dist[2]*(1-ie));
    dist[2] *= cf;
    ts_Transition4[flow][3] = ts_Transition[3][flow] = float(dist[3]*(1-ie));
    dist[3] *= cf;
  }
}



void TreephaserSSE::NormalizeAndSolve(BasecallerRead& read)
{
  copySSE(rd_NormMeasure, &read.raw_measurements[0], num_flows_*sizeof(float));

  for(int step = 0; step < ts_StepCnt; ++step) {
    bool is_final = Solve(ts_StepBeg[step], ts_StepEnd[step]);
    WindowedNormalize(read, step);
    if (is_final)
      break;
  }
  Solve(ts_StepBeg[ts_StepCnt], ts_StepEnd[ts_StepCnt]);

  read.sequence.resize(sv_PathPtr[MAX_PATHS]->sequence_length);
  copySSE(&read.sequence[0], sv_PathPtr[MAX_PATHS]->sequence, sv_PathPtr[MAX_PATHS]->sequence_length*sizeof(char));
  copySSE(&read.normalized_measurements[0], rd_NormMeasure, num_flows_*sizeof(float));
  setZeroSSE(&read.prediction[0], num_flows_*sizeof(float));
  copySSE(&read.prediction[0], sv_PathPtr[MAX_PATHS]->pred, sv_PathPtr[MAX_PATHS]->window_end*sizeof(float));

}



// nextState is only used for the simulation step.

void TreephaserSSE::nextState(PathRec RESTRICT_PTR path, int nuc, int end) {
  int idx = ts_NextNuc[nuc][path->flow];
  if(idx > end)
    idx = end;
  if(path->flow != idx) {
    path->flow = idx;
    idx = path->window_end;
    float alive = 0.0f;
    float RESTRICT_PTR trans = ts_Transition[nuc];
    const float minFrac = 1e-6f;
    int b = path->window_start;
    int e = idx--;
    int i = b;
    while(i < idx) {
      alive += path->state[i];
      float s = alive * trans[i];
      path->state[i] = s;
      alive -= s;
      ++i;
      if(!(s < minFrac))
        break;
      b++;
    }
    if(i > b) {
      while(i < idx) {
        alive += path->state[i];
        float s = alive * trans[i];
        path->state[i] = s;
        alive -= s;
        ++i;
      }
      alive += path->state[i];
      while(i < end) {
        float s = alive * trans[i];
        path->state[i] = s;
        alive -= s;
        if(!(alive > minFrac))
          break;
        path->pred[++i] = 0.0f;
        e++;
      }
    } else {
      alive += path->state[i];
      while(i < e) {
        float s = alive * trans[i];
        path->state[i] = s;
        alive -= s;
        if(i++ == b)
          if((i < e) && (s < minFrac))
            b++;
        if((i == e) && (e < end) && (alive > minFrac))
          path->pred[e++] = 0.0f;
      }
    }
    path->window_start = b;
    path->window_end = e;
  }
}


void TreephaserSSE::advanceState4(PathRec RESTRICT_PTR parent, int end)
{
  // This computation was formerly an ALWAYS_INLINE fct. getParentCopyNucMask
//  {
    int idx = parent->flow;
    __m128i rFlowEnd = _mm_cvtsi32_si128(end);
    __m128i rNucCpy = _mm_cvtsi32_si128(idx);
    __m128i rNucIdx = _mm_load_si128((__m128i RESTRICT_PTR)(ts_NextNuc4[idx]));
    rFlowEnd = _mm_shuffle_epi32(rFlowEnd, _MM_SHUFFLE(0, 0, 0, 0));
    rNucCpy = _mm_shuffle_epi32(rNucCpy, _MM_SHUFFLE(0, 0, 0, 0));
    rNucIdx = _mm_min_epi16(rNucIdx, rFlowEnd);
    rNucCpy = _mm_cmpeq_epi32(rNucCpy, rNucIdx);
    _mm_store_si128((__m128i RESTRICT_PTR)ad_FlowEnd, rFlowEnd);
    _mm_store_si128((__m128i RESTRICT_PTR)ad_Idx, rNucIdx);
//    return rNucCpy;

//  }

//  __m128 rParNuc = _mm_castsi128_ps(getParentCopyNucMask(parent, end));
  __m128 rParNuc = _mm_castsi128_ps(rNucCpy);
  __m128 rAlive = _mm_setzero_ps();
  __m128 rPenNeg = rAlive;
  __m128 rPenPos = rAlive;

  int parLast = parent->window_end;
  __m128i rEnd = _mm_cvtsi32_si128(parLast--);
  __m128i rBeg = _mm_cvtsi32_si128(parent->window_start);
  rEnd = _mm_shuffle_epi32(rEnd, _MM_SHUFFLE(0, 0, 0, 0));
  rBeg = _mm_shuffle_epi32(rBeg, _MM_SHUFFLE(0, 0, 0, 0));

  int i = parent->window_start;
  int j = 0;
  ad_Adv = 1;

  while(i < parLast) {

    __m128 rS = _mm_load_ss(&parent->state[i]);
    __m128i rI = _mm_cvtsi32_si128(i);
    rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));
    rI = _mm_shuffle_epi32(rI, _MM_SHUFFLE(0, 0, 0, 0));

    rAlive = _mm_add_ps(rAlive, rS);

    __m128 rTemp1s = rParNuc;
    rS = _mm_and_ps(rS, rTemp1s);
    rTemp1s = _mm_andnot_ps(rTemp1s, *((__m128 RESTRICT_PTR)(ts_Transition4[i])));
    rTemp1s = _mm_mul_ps(rTemp1s, rAlive);
    rS = _mm_add_ps(rS, rTemp1s);

    _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_STATE_OFS])), rS);

    rAlive = _mm_sub_ps(rAlive, rS);

    __m128i rTemp1i = rBeg;
    rTemp1i = _mm_or_si128(rTemp1i, _mm_castps_si128(rParNuc));
    rTemp1i = _mm_cmpeq_epi32(rTemp1i, rI);
    rTemp1s = _mm_and_ps(_mm_castsi128_ps(rTemp1i), *((__m128 RESTRICT_PTR)ad_MinFrac));
    rTemp1s = _mm_cmpnle_ps(rTemp1s, rS);
    rBeg = _mm_sub_epi32(rBeg, _mm_castps_si128(rTemp1s));
    ad_Adv = _mm_movemask_ps(rTemp1s);

    rTemp1s = _mm_load_ss(&parent->pred[i]);
    rTemp1s = SHUF_PS(rTemp1s, _MM_SHUFFLE(0, 0, 0, 0));
    rTemp1s = _mm_add_ps(rTemp1s, rS);

    _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRED_OFS])), rTemp1s);

    rS = _mm_load_ss(&rd_NormMeasure[i]);
    rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));
    rS = _mm_sub_ps(rS, rTemp1s);

    rTemp1s = rS;
    rS = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(rS),31));
    rTemp1s = _mm_mul_ps(rTemp1s, rTemp1s);
    rS = _mm_and_ps(rS, rTemp1s);
    rTemp1s = _mm_xor_ps(rTemp1s, rS);
    rPenNeg = _mm_add_ps(rPenNeg, rS);
    rPenPos = _mm_add_ps(rPenPos, rTemp1s);

    _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_NRES_OFS])), rPenNeg);
    _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRES_OFS])), rPenPos);

    ++i;
    j += 4;
    if(ad_Adv == 0)
      break;
  }

  if(EXPECTED(ad_Adv == 0)) {

    _mm_store_si128((__m128i RESTRICT_PTR)ad_Beg, rBeg);

    while(i < parLast) {

      __m128 rS = _mm_load_ss(&parent->state[i]);
      rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));

      rAlive = _mm_add_ps(rAlive, rS);

      __m128 rTemp1s = rParNuc;
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_andnot_ps(rTemp1s, *((__m128 RESTRICT_PTR)(ts_Transition4[i])));
      rTemp1s = _mm_mul_ps(rTemp1s, rAlive);
      rS = _mm_add_ps(rS, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_STATE_OFS])), rS);

      rAlive = _mm_sub_ps(rAlive, rS);

      rTemp1s = _mm_load_ss(&parent->pred[i]);
      rTemp1s = SHUF_PS(rTemp1s, _MM_SHUFFLE(0, 0, 0, 0));
      rTemp1s = _mm_add_ps(rTemp1s, rS);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRED_OFS])), rTemp1s);

      rS = _mm_load_ss(&rd_NormMeasure[i]);
      rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));
      rS = _mm_sub_ps(rS, rTemp1s);

      rTemp1s = rS;
      rS = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(rS),31));
      rTemp1s = _mm_mul_ps(rTemp1s, rTemp1s);
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_xor_ps(rTemp1s, rS);
      rPenNeg = _mm_add_ps(rPenNeg, rS);
      rPenPos = _mm_add_ps(rPenPos, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_NRES_OFS])), rPenNeg);
      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRES_OFS])), rPenPos);

      ++i;
      j += 4;
    }

    {
      __m128 rS = _mm_load_ss(&parent->state[i]);
      __m128i rI = _mm_cvtsi32_si128(i);
      rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));
      rI = _mm_shuffle_epi32(rI, _MM_SHUFFLE(0, 0, 0, 0));

      rAlive = _mm_add_ps(rAlive, rS);

      __m128 rTemp1s = rParNuc;
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_andnot_ps(rTemp1s, *((__m128 RESTRICT_PTR)(ts_Transition4[i])));
      rTemp1s = _mm_mul_ps(rTemp1s, rAlive);
      rS = _mm_add_ps(rS, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_STATE_OFS])), rS);

      rAlive = _mm_sub_ps(rAlive, rS);

      rTemp1s = _mm_load_ss(&parent->pred[i]);
      rTemp1s = SHUF_PS(rTemp1s, _MM_SHUFFLE(0, 0, 0, 0));
      rTemp1s = _mm_add_ps(rTemp1s, rS);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRED_OFS])), rTemp1s);

      rS = _mm_load_ss(&rd_NormMeasure[i]);
      rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));
      rS = _mm_sub_ps(rS, rTemp1s);

      rTemp1s = rS;
      rS = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(rS),31));
      rTemp1s = _mm_mul_ps(rTemp1s, rTemp1s);
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_xor_ps(rTemp1s, rS);
      rPenNeg = _mm_add_ps(rPenNeg, rS);
      rPenPos = _mm_add_ps(rPenPos, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_NRES_OFS])), rPenNeg);
      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRES_OFS])), rPenPos);

      rTemp1s = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(rTemp1s), _mm_castps_si128(rTemp1s)));
      rTemp1s = _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(rTemp1s), rEnd));
      rTemp1s = _mm_or_ps(rTemp1s, rParNuc);
      rTemp1s = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(rTemp1s), rI));
      rTemp1s = _mm_and_ps(rTemp1s, rAlive);
      rTemp1s = _mm_cmpnle_ps(rTemp1s, *((__m128 RESTRICT_PTR)ad_MinFrac));
      ad_Adv = _mm_movemask_ps(rTemp1s);
      rEnd = _mm_sub_epi32(rEnd, _mm_castps_si128(rTemp1s));

      ++i;
      j += 4;
    }

    while((i < end) && (ad_Adv != 0)) {

      __m128 rS = _mm_load_ss(&parent->state[i]);
      __m128i rI = _mm_cvtsi32_si128(i);
      rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));
      rI = _mm_shuffle_epi32(rI, _MM_SHUFFLE(0, 0, 0, 0));

      __m128 rTemp1s = rParNuc;
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_andnot_ps(rTemp1s, *((__m128 RESTRICT_PTR)(ts_Transition4[i])));
      rTemp1s = _mm_mul_ps(rTemp1s, rAlive);
      rS = _mm_add_ps(rS, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_STATE_OFS])), rS);

      rAlive = _mm_sub_ps(rAlive, rS);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRED_OFS])), rS);

      rTemp1s = _mm_load_ss(&rd_NormMeasure[i]);
      rTemp1s = SHUF_PS(rTemp1s, _MM_SHUFFLE(0, 0, 0, 0));
      rTemp1s = _mm_sub_ps(rTemp1s, rS);
      rS = rTemp1s;

      rS = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(rS),31));
      rTemp1s = _mm_mul_ps(rTemp1s, rTemp1s);
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_xor_ps(rTemp1s, rS);
      rPenNeg = _mm_add_ps(rPenNeg, rS);
      rPenPos = _mm_add_ps(rPenPos, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_NRES_OFS])), rPenNeg);
      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRES_OFS])), rPenPos);

      rTemp1s = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(rTemp1s), _mm_castps_si128(rTemp1s)));
      rTemp1s = _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(rTemp1s), rEnd));
      rTemp1s = _mm_or_ps(rTemp1s, rParNuc);
      rTemp1s = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(rTemp1s), rI));
      rTemp1s = _mm_and_ps(rTemp1s, rAlive);
      rTemp1s = _mm_cmpnle_ps(rTemp1s, *((__m128 RESTRICT_PTR)ad_MinFrac));
      ad_Adv = _mm_movemask_ps(rTemp1s);
      rEnd = _mm_sub_epi32(rEnd, _mm_castps_si128(rTemp1s));

      ++i;
      j += 4;
    }

    rEnd = _mm_min_epi16(rEnd, *((__m128i RESTRICT_PTR)ad_FlowEnd));
    _mm_store_si128((__m128i RESTRICT_PTR)ad_End, rEnd);

  } else {

    {
      __m128 rS = _mm_load_ss(&parent->state[i]);
      __m128i rI = _mm_cvtsi32_si128(i);
      rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));
      rI = _mm_shuffle_epi32(rI, _MM_SHUFFLE(0, 0, 0, 0));

      rAlive = _mm_add_ps(rAlive, rS);

      __m128 rTemp1s = rParNuc;
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_andnot_ps(rTemp1s, *((__m128 RESTRICT_PTR)(ts_Transition4[i])));
      rTemp1s = _mm_mul_ps(rTemp1s, rAlive);
      rS = _mm_add_ps(rS, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_STATE_OFS])), rS);

      rAlive = _mm_sub_ps(rAlive, rS);

      __m128i rTemp1i = rBeg;
      rTemp1i = _mm_or_si128(rTemp1i, _mm_castps_si128(rParNuc));
      rTemp1i = _mm_cmpeq_epi32(rTemp1i, rI);
      rTemp1s = _mm_and_ps(_mm_castsi128_ps(rTemp1i), *((__m128 RESTRICT_PTR)ad_MinFrac));
      rTemp1s = _mm_cmpnle_ps(rTemp1s, rS);
      rBeg = _mm_sub_epi32(rBeg, _mm_castps_si128(rTemp1s));
      rTemp1i = rBeg;
      rTemp1i = _mm_cmpeq_epi32(rTemp1i, rEnd);
      rBeg = _mm_add_epi32(rBeg, rTemp1i);

      rTemp1s = _mm_load_ss(&parent->pred[i]);
      rTemp1s = SHUF_PS(rTemp1s, _MM_SHUFFLE(0, 0, 0, 0));
      rTemp1s = _mm_add_ps(rTemp1s, rS);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRED_OFS])), rTemp1s);

      rS = _mm_load_ss(&rd_NormMeasure[i]);
      rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));
      rS = _mm_sub_ps(rS, rTemp1s);

      rTemp1s = rS;
      rS = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(rS),31));
      rTemp1s = _mm_mul_ps(rTemp1s, rTemp1s);
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_xor_ps(rTemp1s, rS);
      rPenNeg = _mm_add_ps(rPenNeg, rS);
      rPenPos = _mm_add_ps(rPenPos, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_NRES_OFS])), rPenNeg);
      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRES_OFS])), rPenPos);

      rTemp1i = _mm_cmpeq_epi32(rTemp1i, rTemp1i);
      rTemp1i = _mm_add_epi32(rTemp1i, rEnd);
      rTemp1i = _mm_or_si128(rTemp1i, _mm_castps_si128(rParNuc));
      rTemp1i = _mm_cmpeq_epi32(rTemp1i, rI);
      rTemp1s = _mm_and_ps(_mm_castsi128_ps(rTemp1i), rAlive);
      rTemp1s = _mm_cmpnle_ps(rTemp1s, *((__m128 RESTRICT_PTR)ad_MinFrac));
      ad_Adv = _mm_movemask_ps(rTemp1s);
      rEnd = _mm_sub_epi32(rEnd, _mm_castps_si128(rTemp1s));

      ++i;
      j += 4;
    }

    while((i < end) && (ad_Adv != 0)) {

      __m128 rS = _mm_load_ss(&parent->state[i]);
      __m128i rI = _mm_cvtsi32_si128(i);
      rS = SHUF_PS(rS, _MM_SHUFFLE(0, 0, 0, 0));
      rI = _mm_shuffle_epi32(rI, _MM_SHUFFLE(0, 0, 0, 0));

      __m128 rTemp1s = rParNuc;
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_andnot_ps(rTemp1s, *((__m128 RESTRICT_PTR)(ts_Transition4[i])));
      rTemp1s = _mm_mul_ps(rTemp1s, rAlive);
      rS = _mm_add_ps(rS, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_STATE_OFS])), rS);

      rAlive = _mm_sub_ps(rAlive, rS);

      __m128i rTemp1i = rBeg;
      rTemp1i = _mm_or_si128(rTemp1i, _mm_castps_si128(rParNuc));
      rTemp1i = _mm_cmpeq_epi32(rTemp1i, rI);
      rTemp1s = _mm_and_ps(_mm_castsi128_ps(rTemp1i), *((__m128 RESTRICT_PTR)ad_MinFrac));
      rTemp1s = _mm_cmpnle_ps(rTemp1s, rS);
      rBeg = _mm_sub_epi32(rBeg, _mm_castps_si128(rTemp1s));
      rTemp1i = rBeg;
      rTemp1i = _mm_cmpeq_epi32(rTemp1i, rEnd);
      rBeg = _mm_add_epi32(rBeg, rTemp1i);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRED_OFS])), rS);

      rTemp1s = _mm_load_ss(&rd_NormMeasure[i]);
      rTemp1s = SHUF_PS(rTemp1s, _MM_SHUFFLE(0, 0, 0, 0));
      rTemp1s = _mm_sub_ps(rTemp1s, rS);
      rS = rTemp1s;

      rS = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(rS),31));
      rTemp1s = _mm_mul_ps(rTemp1s, rTemp1s);
      rS = _mm_and_ps(rS, rTemp1s);
      rTemp1s = _mm_xor_ps(rTemp1s, rS);
      rPenNeg = _mm_add_ps(rPenNeg, rS);
      rPenPos = _mm_add_ps(rPenPos, rTemp1s);

      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_NRES_OFS])), rPenNeg);
      _mm_store_ps((float RESTRICT_PTR)(&(ad_Buf[j*4+AD_PRES_OFS])), rPenPos);

      rTemp1i = _mm_cmpeq_epi32(rTemp1i, rTemp1i);
      rTemp1i = _mm_add_epi32(rTemp1i, rEnd);
      rTemp1i = _mm_or_si128(rTemp1i, _mm_castps_si128(rParNuc));
      rTemp1i = _mm_cmpeq_epi32(rTemp1i, rI);
      rTemp1s = _mm_and_ps(_mm_castsi128_ps(rTemp1i), rAlive);
      rTemp1s = _mm_cmpnle_ps(rTemp1s, *((__m128 RESTRICT_PTR)ad_MinFrac));
      ad_Adv = _mm_movemask_ps(rTemp1s);
      rEnd = _mm_sub_epi32(rEnd, _mm_castps_si128(rTemp1s));

      ++i;
      j += 4;
    }

    rEnd = _mm_min_epi16(rEnd, *((__m128i RESTRICT_PTR)ad_FlowEnd));
    _mm_store_si128((__m128i RESTRICT_PTR)ad_Beg, rBeg);
    _mm_store_si128((__m128i RESTRICT_PTR)ad_End, rEnd);

  }
}



void TreephaserSSE::sumNormMeasures() {
  int i = num_flows_;
  float sum = 0.0f;
  rd_SqNormMeasureSum[i] = 0.0f;
  while(--i >= 0)
    rd_SqNormMeasureSum[i] = (sum += rd_NormMeasure[i]*rd_NormMeasure[i]);
}


bool TreephaserSSE::Solve(int begin_flow, int end_flow)
{
  sumNormMeasures();

  PathRec RESTRICT_PTR parent = sv_PathPtr[0];
  PathRec RESTRICT_PTR best = sv_PathPtr[MAX_PATHS];

  parent->flow = 0;
  parent->window_start = 0;
  parent->window_end = 1;
  parent->res = 0.0f;
  parent->metr = 0.0f;
  parent->flowMetr = 0.0f;
  parent->dotCnt = 0;
  parent->state[0] = 1.0f;
  parent->sequence_length = 0;
  parent->last_hp = 0;
  parent->pred[0] = 0.0f;

  int pathCnt = 1;
  float bestDist = 1e20; //float(endFlow);

  if(begin_flow > 0) {
    static const int char_to_nuc[8] = {-1, 0, -1, 1, 3, -1, -1, 2};
    for (int base = 0; base < best->sequence_length; ++base) {
      parent->sequence_length++;
      parent->sequence[base] = best->sequence[base];
      if (base and parent->sequence[base] != parent->sequence[base-1])
        parent->last_hp = 0;
      parent->last_hp++;
      nextState(parent, char_to_nuc[best->sequence[base]&7], end_flow);
      for(int k = parent->window_start; k < parent->window_end; ++k) {
        if((k & 3) == 0) {
          sumVectFloatSSE(&parent->pred[k], &parent->state[k], parent->window_end-k);
          break;
        }
        parent->pred[k] += parent->state[k];
      }
      if (parent->flow >= begin_flow)
        break;
    }
    if(parent->window_end < begin_flow) {
      sv_PathPtr[MAX_PATHS] = parent;
      sv_PathPtr[0] = best;
      return true;
    }
    parent->res = sumOfSquaredDiffsFloatSSE(
      (float RESTRICT_PTR)rd_NormMeasure, (float RESTRICT_PTR)parent->pred, parent->window_start);
  }

  best->window_end = 0;
  best->sequence_length = 0;

  do {

    if(pathCnt > 3) {
      int m = sv_PathPtr[0]->flow;
      int i = 1;
      do {
        int n = sv_PathPtr[i]->flow;
        if(m < n)
          m = n;
      } while(++i < pathCnt);
      if((m -= MAX_PATH_DELAY) > 0) {
        do {
          if(sv_PathPtr[--i]->flow < m)
            swap(sv_PathPtr[i], sv_PathPtr[--pathCnt]);
        } while(i > 0);
      }
    }

    while(pathCnt > MAX_PATHS-4) {
      float m = sv_PathPtr[0]->flowMetr;
      int i = 1;
      int j = 0;
      do {
        float n = sv_PathPtr[i]->flowMetr;
        if(m < n) {
          m = n;
          j = i;
        }
      } while(++i < pathCnt);
      swap(sv_PathPtr[j], sv_PathPtr[--pathCnt]);
    }

    parent = sv_PathPtr[0];
    int parentPathIdx = 0;
    for(int i = 1; i < pathCnt; ++i)
      if(parent->metr > sv_PathPtr[i]->metr) {
        parent = sv_PathPtr[i];
        parentPathIdx = i;
      }
    if(parent->metr >= 1000.0f)
      break;

    advanceState4(parent, end_flow);

    int n = pathCnt;
    double bestpen = 25.0;
    for(int nuc = 0; nuc < 4; ++nuc) {
      PathRec RESTRICT_PTR child = sv_PathPtr[n];

      child->flow = ad_Idx[nuc];
      child->window_start = ad_Beg[nuc];
      child->window_end = ad_End[nuc];

      if(child->flow >= end_flow or parent->last_hp >= MAX_HPXLEN or parent->sequence_length >= 2*MAX_VALS-10)
        continue;

      char RESTRICT_PTR pn = ad_Buf+nuc*4+(AD_NRES_OFS-16)-parent->window_start*16;
      float metr = parent->res + *((float RESTRICT_PTR)(pn+child->window_start*16+(AD_PRES_OFS-AD_NRES_OFS)));
      float penPar = *((float RESTRICT_PTR)(pn+child->flow*16+(AD_PRES_OFS-AD_NRES_OFS)));
      float penNeg = *((float RESTRICT_PTR)(pn+child->window_end*16));
      child->res = metr + *((float RESTRICT_PTR)(pn+child->window_start*16));
      metr += penNeg;

      penPar += penNeg;
      penNeg += penPar;
      if(penNeg >= 20.0)
        continue;
      if(bestpen > penNeg)
        bestpen = penNeg;
      else if(penNeg-bestpen >= 0.2)
        continue;
      if(metr > bestDist)
        continue;
      float newSignal = rd_NormMeasure[child->flow];
      if(child->flow < parent->window_end)
        newSignal -= parent->pred[child->flow];
      newSignal /= *((float RESTRICT_PTR)(pn+child->flow*16+(AD_STATE_OFS-AD_NRES_OFS+16)));
      child->dotCnt = 0;
      if(newSignal < 0.3f) {
        if(parent->dotCnt > 0)
          continue;
        child->dotCnt = 1;
      }
      child->metr = float(metr);
      child->flowMetr = float(penPar);
      child->penalty = float(penNeg);
      child->nuc = nuc;
      ++n;
    }

    float dist = parent->res+(rd_SqNormMeasureSum[parent->window_end]-rd_SqNormMeasureSum[end_flow]);
    for(int i = parent->window_start; i < parent->window_end; ++i) {
      if((i & 3) == 0) {
        dist += sumOfSquaredDiffsFloatSSE((float RESTRICT_PTR)(&(rd_NormMeasure[i])),
          (float RESTRICT_PTR)(&(parent->pred[i])), parent->window_end-i);
        break;
      }
      dist += Sqr(rd_NormMeasure[i]-parent->pred[i]);
    }
    int bestPathIdx = -1;
    if(bestDist > dist) {
      bestPathIdx = parentPathIdx;
      parentPathIdx = -1;
    }

    int childPathIdx = -1;
    while(pathCnt < n) {
      PathRec RESTRICT_PTR child = sv_PathPtr[pathCnt];
      if(child->penalty-bestpen >= 0.2f) {
        sv_PathPtr[pathCnt] = sv_PathPtr[--n];
        sv_PathPtr[n] = child;
      } else if((childPathIdx < 0) && (parentPathIdx >= 0)) {
        sv_PathPtr[pathCnt] = sv_PathPtr[--n];
        sv_PathPtr[n] = child;
        childPathIdx = n;
      } else {
        child->flowMetr = (child->metr + 0.5f*child->flowMetr) / child->flow; // ??
        char RESTRICT_PTR p = ad_Buf+child->nuc*4+AD_STATE_OFS;
        for(int i = parent->window_start, j = 0, e = child->window_end; i < e; ++i, j += 16) {
          child->state[i] = *((float*)(p+j));
          child->pred[i] = *((float*)(p+j+(AD_PRED_OFS-AD_STATE_OFS)));
        }
        copySSE(child->pred, parent->pred, parent->window_start << 2);

        copySSE(child->sequence, parent->sequence, parent->sequence_length);
        child->sequence_length = parent->sequence_length + 1;
        child->sequence[parent->sequence_length] = flow_order_[child->flow];
        if (parent->sequence_length and child->sequence[parent->sequence_length] != child->sequence[parent->sequence_length-1])
          child->last_hp = 0;
        else
          child->last_hp = parent->last_hp;
        child->last_hp++;

        ++pathCnt;
      }
    }

    if(childPathIdx >= 0) {
      PathRec RESTRICT_PTR child = sv_PathPtr[childPathIdx];
      parent->flow = child->flow;
      parent->window_end = child->window_end;
      parent->res = child->res;
      parent->metr = child->metr;
      parent->flowMetr = (child->metr + 0.5f*child->flowMetr) / child->flow; // ??
      parent->dotCnt = child->dotCnt;
      char RESTRICT_PTR p = ad_Buf+child->nuc*4+AD_STATE_OFS;
      for(int i = parent->window_start, j = 0, e = child->window_end; i < e; ++i, j += 16) {
        parent->state[i] = *((float*)(p+j));
        parent->pred[i] = *((float*)(p+j+(AD_PRED_OFS-AD_STATE_OFS)));
      }

      parent->sequence[parent->sequence_length] = flow_order_[parent->flow];
      if (parent->sequence_length and parent->sequence[parent->sequence_length] != parent->sequence[parent->sequence_length-1])
        parent->last_hp = 0;
      parent->last_hp++;
      parent->sequence_length++;

      parent->window_start = child->window_start;
      parentPathIdx = -1;
    }

    if(bestPathIdx >= 0) {
      bestDist = dist;
      sv_PathPtr[bestPathIdx] = sv_PathPtr[--pathCnt];
      sv_PathPtr[pathCnt] = sv_PathPtr[MAX_PATHS];
      sv_PathPtr[MAX_PATHS] = parent;
    } else if(parentPathIdx >= 0) {
      sv_PathPtr[parentPathIdx] = sv_PathPtr[--pathCnt];
      sv_PathPtr[pathCnt] = parent;
    }

  } while(pathCnt > 0);

  return false;
}





void TreephaserSSE::WindowedNormalize(BasecallerRead& read, int num_steps)
{
//  int num_flows = read.raw_measurements.size();
  float median_set[STEP_SIZE];

  // Estimate and correct for additive offset

  float next_normalizer = 0;
  int estim_flow = 0;
  int apply_flow = 0;

  for (int step = 0; step <= num_steps; ++step) {

    int window_end = estim_flow + STEP_SIZE;
    int window_middle = estim_flow + STEP_SIZE / 2;
    if (window_middle > num_flows_)
      break;

    float normalizer = next_normalizer;

    int median_set_size = 0;
    for (; estim_flow < window_end and estim_flow < num_flows_ and estim_flow < sv_PathPtr[MAX_PATHS]->window_end; ++estim_flow)
      if (sv_PathPtr[MAX_PATHS]->pred[estim_flow] < 0.3)
        median_set[median_set_size++] = read.raw_measurements[estim_flow] - sv_PathPtr[MAX_PATHS]->pred[estim_flow];

    if (median_set_size > 5) {
      std::nth_element(median_set, median_set + median_set_size/2, median_set + median_set_size);
      next_normalizer = median_set[median_set_size / 2];
      if (step == 0)
        normalizer = next_normalizer;
    }

    float delta = (next_normalizer - normalizer) / STEP_SIZE;

    for (; apply_flow < window_middle and apply_flow < num_flows_; ++apply_flow) {
      rd_NormMeasure[apply_flow] = read.raw_measurements[apply_flow] - normalizer;
      read.additive_correction[apply_flow] = normalizer;
      normalizer += delta;
    }
  }

  for (; apply_flow < num_flows_; ++apply_flow) {
    rd_NormMeasure[apply_flow] = read.raw_measurements[apply_flow] - next_normalizer;
    read.additive_correction[apply_flow] = next_normalizer;
  }

  // Estimate and correct for multiplicative scaling

  next_normalizer = 1;
  estim_flow = 0;
  apply_flow = 0;

  for (int step = 0; step <= num_steps; ++step) {

    int window_end = estim_flow + STEP_SIZE;
    int window_middle = estim_flow + STEP_SIZE / 2;
    if (window_middle > num_flows_)
      break;

    float normalizer = next_normalizer;

    int median_set_size = 0;
    for (; estim_flow < window_end and estim_flow < num_flows_ and estim_flow < sv_PathPtr[MAX_PATHS]->window_end; ++estim_flow)
      if (sv_PathPtr[MAX_PATHS]->pred[estim_flow] > 0.5 and rd_NormMeasure[estim_flow] > 0)
        median_set[median_set_size++] = rd_NormMeasure[estim_flow] / sv_PathPtr[MAX_PATHS]->pred[estim_flow];

    if (median_set_size > 5) {
      std::nth_element(median_set, median_set + median_set_size/2, median_set + median_set_size);
      next_normalizer = median_set[median_set_size / 2];
      if (step == 0)
        normalizer = next_normalizer;
    }

    float delta = (next_normalizer - normalizer) / STEP_SIZE;

    for (; apply_flow < window_middle and apply_flow < num_flows_; ++apply_flow) {
      rd_NormMeasure[apply_flow] /= normalizer;
      read.multiplicative_correction[apply_flow] = normalizer;
      normalizer += delta;
    }
  }

  for (; apply_flow < num_flows_; ++apply_flow) {
    rd_NormMeasure[apply_flow] /= next_normalizer;
    read.multiplicative_correction[apply_flow] = next_normalizer;
  }
}




