/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "TreephaserVEC.h"
#include <x86intrin.h>

#include <vector>
#include <algorithm>
#include <math.h>
#include <cstring>
#include <cassert>
#include <stdint.h>

#include "BaseCallerUtils.h"
#include "DPTreephaser.h"

#define SHUF_PS(reg, mode) _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(reg), mode))


using namespace std;

ALWAYS_INLINE float Sqr(float val) {
  return val*val;
}

inline void setZeroSSE(void *dst, int size) {
  v4i r0 = LD_VEC4I(0);
  v4i *dstV=(v4i *)dst;
  int lsize=size/16;

  while((size & 15) != 0) {
    --size;
    ((char*)dst)[size] = char(0);
  }
  while(lsize) {
	  lsize--;
	dstV[lsize]=r0;
  }
}


inline void setValueSSE(float *buf, float val, int size) {
  v4f valV=LD_VEC4F(val);
  v4f *bufV=(v4f *)buf;
  int lsize=size/4;
  int i=0;
  for(;i<lsize;i++){
    bufV[i] = valV;
  }
  i*=4;
  // fill the rest of the buffer
  while (i<size) {
    buf[i] = val;
    i++;
  }  
}

inline void copySSE(void *dst, void *src, int size) {

	while(size && ((uint64_t)dst & 15)){
		--size;
	    ((char RESTRICT_PTR)dst)[0] = ((char RESTRICT_PTR)src)[0];
	    src=(char *)src+1;
	    dst=(char *)dst+1;
	}
  while((size & 15) != 0) {
    --size;
    ((char*)dst)[size] = ((char*)src)[size];
  }
  v4i *dstV=(v4i *)dst;
  v4i *srcV=(v4i *)src;
  int lsize=size/16;
  while(lsize) {
	lsize--;
    dstV[lsize]=srcV[lsize];
  }
}

inline void copySSE_4th(float *dst, float *src, int size){

	  while(size--) {
	    dst[size] = src[4*size];
	  }

//	  while(((uint64_t)dst & 15) && size) {
//		  size--;
//	    (*dst++) = src[0];
//	    src +=4;
//	  }
//	  while(size & 3) {
//		  size--;
//	    dst[size] = src[4*size];
//	  }
//	  v4f *dstV=(v4f*)dst;
//
//	  while(size) {
//		v4f tmp=(v4f){src[4*(size+0)],src[4*(size+1)],src[4*(size+2)],src[4*(size+3)]};
//	    dstV[size/4] = tmp;
//	  }
}

inline float sumOfSquaredDiffsFloatSSE(float RESTRICT_PTR src1, float RESTRICT_PTR src2, int count) {
  float sum = 0.0f;

  // align the pointers before beginning
  while(((uint64_t)src1 & 31) && count){
	    float r1 = (*src1++) - (*src2++);
	    sum += r1*r1;
	    count--;
  }

  while((count & 3) != 0) {
	--count;
	float r1=src1[count]-src2[count];
	sum += r1*r1;
  }
  count /=4;

  v4f_u sumV;
  sumV.V = (v4f){sum,0,0,0};
  v4f *src1V=(v4f *)src1;
  v4f *src2V=(v4f *)src2;
  while(count--) {
    v4f r1 = src1V[count] - src2V[count];
    sumV.V += r1*r1;
  }

  sum = (sumV.A[3]+sumV.A[1])+(sumV.A[2]+sumV.A[0]);
  return sum;
}

inline float  sumOfSquaredDiffsFloatSSE_recal(float RESTRICT_PTR src1, float RESTRICT_PTR src2, float RESTRICT_PTR A, float RESTRICT_PTR B, int count) {
	  float sum = 0.0f;

	  // align the pointers before beginning
	  while(((uint64_t)src1 & 31) && count){
			float r1=(*src1++) - (*src2++) * (*A++) - (*B++);
		    sum += r1*r1;
		    count--;
	  }

	  while((count & 3) != 0) {
		--count;
		float r1=src1[count] - src2[count]*A[count] - B[count];
		sum += r1*r1;
	  }
	  count /=4;

	  v4f_u sumV;
	  sumV.V= (v4f){sum,0,0,0};
	  v4f *src1V=(v4f *)src1;
	  v4f *src2V=(v4f *)src2;
	  v4f *AV=(v4f *)A;
	  v4f *BV=(v4f *)B;
	  while(count--) {
	    v4f r1 = src1V[count] - src2V[count]*AV[count] - BV[count];
	    sumV.V += r1*r1;
	  }

	  sum = (sumV.A[3]+sumV.A[1])+(sumV.A[2]+sumV.A[0]);
	  return sum;
}

inline void sumVectFloatSSE(float RESTRICT_PTR dst, float RESTRICT_PTR src, int count) {

	while(((uint64_t)dst & 15) && count){
		count--;
		(*dst) += (*src++);
		dst++;
	}
  while((count & 3) != 0) {
    --count;
    dst[count] += src[count];
  }
  count /=4;
  v4f sumV = LD_VEC4F(0);
  v4f *srcV=(v4f *)src;
  v4f *dstV=(v4f *)dst;

  while(count > 0) {
	count--;
	dstV[count] += srcV[count];
  }
}



// ----------------------------------------------------------------------------

// Constructor used in variant caller
TreephaserSSE::TreephaserSSE()
  : flow_order_("TACG", 4), my_cf_(-1.0), my_ie_(-1.0), As_(NULL), Bs_(NULL)
{
  SetNormalizationWindowSize(38);
  SetFlowOrder(flow_order_);
}

// Constructor used in Basecaller
TreephaserSSE::TreephaserSSE(const ion::FlowOrder& flow_order, const int windowSize)
  : my_cf_(-1.0), my_ie_(-1.0), As_(NULL), Bs_(NULL)
{
  SetNormalizationWindowSize(windowSize);
  SetFlowOrder(flow_order);
}

// ----------------------------------------------------------------
// Initilizes all float variables to NAN so that they cause mayhem if we read out of bounds
// and so that valgrind does not complain about uninitialized variables
void TreephaserSSE::InitializeVariables(float init_val) {
  
  // Initializing the elements of the paths
  for (unsigned int path = 0; path <= MAX_PATHS; ++path) {
    sv_PathPtr[path]->flow            = 0;
    sv_PathPtr[path]->window_start    = 0;
    sv_PathPtr[path]->window_end      = 0;
    sv_PathPtr[path]->dotCnt          = 0;
    sv_PathPtr[path]->sequence_length = 0;
    sv_PathPtr[path]->last_hp         = 0;
    sv_PathPtr[path]->nuc             = 0;
    
    sv_PathPtr[path]->res      = init_val;
    sv_PathPtr[path]->metr     = init_val;
    sv_PathPtr[path]->flowMetr = init_val;
    sv_PathPtr[path]->penalty  = init_val;
    for (int val=0; val<MAX_VALS; val++) {
      sv_PathPtr[path]->state[val]         = init_val;
      sv_PathPtr[path]->pred[val]          = init_val;
      sv_PathPtr[path]->state_inphase[val] = init_val;
    }
    for (int val=0; val<(2*MAX_VALS + 12); val++)
      sv_PathPtr[path]->sequence[val] = 0;
  }
  
  // Initializing the other variables of the object
  for (unsigned int idx=0; idx<4; idx++) {
    ad_FlowEnd.A[idx] = 0;
    ad_Idx.A[idx] = 0;
    ad_End.A[idx] = 0;
    ad_Beg.A[idx] = 0;
  }
  for (unsigned int val=0; val<MAX_VALS; val++) {
    rd_NormMeasure[val]      = init_val;
    rd_SqNormMeasureSum[val] = init_val;
  }
  for (unsigned int idx=0; idx<=(MAX_VALS); idx++) {
	    state_Buf[idx].V = LD_VEC4F(0);
	    pred_Buf[idx].V = LD_VEC4F(0);
	    nres_Buf[idx].V = LD_VEC4F(0);
	    pres_Buf[idx].V = LD_VEC4F(0);
  }
  ad_Adv = 0;
}


// ----------------------------------------------------------------

// Initialize Object
void TreephaserSSE::SetFlowOrder(const ion::FlowOrder& flow_order)
{
  flow_order_ = flow_order;
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
  // -- For valgrind and debugging & to make cppcheck happy
  InitializeVariables(0.0);
  // --

  ad_MinFrac.V=LD_VEC4F(1e-6f);

  int nextIdx[4];
  nextIdx[3] = nextIdx[2] = nextIdx[1] = nextIdx[0] = short(num_flows_);
  for(int flow = num_flows_-1; flow >= 0; --flow) {
    nextIdx[flow_order_.int_at(flow)] = flow;
    ts_NextNuc[0][flow] = (short)(ts_NextNuc4[flow].A[0] = nextIdx[0]);
    ts_NextNuc[1][flow] = (short)(ts_NextNuc4[flow].A[1] = nextIdx[1]);
    ts_NextNuc[2][flow] = (short)(ts_NextNuc4[flow].A[2] = nextIdx[2]);
    ts_NextNuc[3][flow] = (short)(ts_NextNuc4[flow].A[3] = nextIdx[3]);
  }

  ts_StepCnt = 0;
  for(int i = windowSize_ << 1; i < num_flows_; i += windowSize_) {
    ts_StepBeg[ts_StepCnt] = (ts_StepEnd[ts_StepCnt] = i)-(windowSize_ << 1);
    ts_StepCnt++;
  }
  ts_StepBeg[ts_StepCnt] = (ts_StepEnd[ts_StepCnt] = num_flows_)-(windowSize_ << 1);
  ts_StepEnd[++ts_StepCnt] = num_flows_;
  ts_StepBeg[ts_StepCnt] = 0;
  
  // The initialization of the recalibration fields for all paths is necessary since we are
  // over-running memory in the use of the recalibration parameters
  ResetRecalibrationStructures(MAX_VALS);

  pm_model_available_              = false;
  recalibrate_predictions_         = false;
  state_inphase_enabled_           = false;
  skip_recal_during_normalization_ = false;
}

// ----------------------------------------------------------------

void TreephaserSSE::SetModelParameters(double cf, double ie)
{
  if (cf == my_cf_ and ie == my_ie_)
    return;
  
  double dist[4] = { 0.0, 0.0, 0.0, 0.0 };

  for(int flow = 0; flow < num_flows_; ++flow) {
    dist[flow_order_.int_at(flow)] = 1.0;
    ts_Transition4[flow].A[0] = ts_Transition[0][flow] = float(dist[0]*(1-ie));
    dist[0] *= cf;
    ts_Transition4[flow].A[1] = ts_Transition[1][flow] = float(dist[1]*(1-ie));
    dist[1] *= cf;
    ts_Transition4[flow].A[2] = ts_Transition[2][flow] = float(dist[2]*(1-ie));
    dist[2] *= cf;
    ts_Transition4[flow].A[3] = ts_Transition[3][flow] = float(dist[3]*(1-ie));
    dist[3] *= cf;
  }
  my_cf_ = cf;
  my_ie_ = ie;
}

// ----------------------------------------------------------------

void TreephaserSSE::NormalizeAndSolve(BasecallerRead& read)
{
  copySSE(rd_NormMeasure, &read.raw_measurements[0], num_flows_*sizeof(float));
  // Disable recalibration during normalization stage if requested
  if (skip_recal_during_normalization_)
    recalibrate_predictions_ = false;

  for(int step = 0; step < ts_StepCnt; ++step) {
    bool is_final = Solve(ts_StepBeg[step], ts_StepEnd[step]);
    WindowedNormalize(read, step);
    if (is_final)
      break;
  }

  //final stage of solve and calculate the state_inphase for QV prediction
  state_inphase_enabled_ = true;
  // And turn recalibration back on (if available) for the final solving part
  EnableRecalibration();

  Solve(ts_StepBeg[ts_StepCnt], ts_StepEnd[ts_StepCnt]);

  int to_flow = min(sv_PathPtr[MAX_PATHS]->window_end, num_flows_); // Apparently window_end can be larger than num_flows_
  read.sequence.resize(sv_PathPtr[MAX_PATHS]->sequence_length);
  copySSE(&read.sequence[0], sv_PathPtr[MAX_PATHS]->sequence, sv_PathPtr[MAX_PATHS]->sequence_length*sizeof(char));
  copySSE(&read.normalized_measurements[0], rd_NormMeasure, num_flows_*sizeof(float));
  setZeroSSE(&read.prediction[0], num_flows_*sizeof(float));
  copySSE(&read.prediction[0], sv_PathPtr[MAX_PATHS]->pred, to_flow*sizeof(float));
  setZeroSSE(&read.state_inphase[0], num_flows_*sizeof(float));
  copySSE(&read.state_inphase[0], sv_PathPtr[MAX_PATHS]->state_inphase, to_flow*sizeof(float));

  // copy inphase population and reset state_inphase flag
  if(state_inphase_enabled_){
    for (int p = 0; p <= 8; ++p) {
      setZeroSSE(&(sv_PathPtr[p]->state_inphase[0]), num_flows_*sizeof(float));
    }
  }
  state_inphase_enabled_ = false;
}

// ----------------------------------------------------------------------

// advanceStateInPlace is only used for the simulation step.

void TreephaserSSE::advanceStateInPlace(PathRec RESTRICT_PTR path, int nuc, int end) {
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
    int em1 = e-1;
    int i = b;
    v4i maskL=(v4i){1,2,3,0};
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

    while(/*(i&3) && */(i < em1)) {
      alive += path->state[i];
      float s = alive * trans[i];
      path->state[i] = s;
      alive -= s;
      ++i;
    }
#if 0
    int llen=(em1-i)/4;
    v4f *psp=(v4f *)&path->state[i];
    v4f *trp=(v4f *)&trans[i];
    for(int j=0;j < llen;j++) {
	      v4f_u ps,s,t;
	      ps.V =psp[j];
	      t.V = trp[j];


	      for(int li=0;li<4;li++){
			  alive += ps.A[0];
			  s.A[0] = alive * t.A[0];
			  alive -= s.A[0];
			  // shift ps, s, and t to the left
			  ps.V = __builtin_shuffle (ps, maskL);
			  s.V = __builtin_shuffle (s, maskL);
			  t.V = __builtin_shuffle (t, maskL);
	      }
	      psp[j] = s.V;
    }
    i+=4*llen;
#endif
    // flow > window start
    if(i > b) {
      // flow < window end - 1
      while(i < idx) {
        alive += path->state[i];
        float s = alive * trans[i];
        path->state[i] = s;
        alive -= s;
        ++i;
      }
      alive += path->state[i];
    
      // flow >= window end - 1
      while(i < e) {
        float s = alive * trans[i];
        path->state[i] = s;
        alive -= s;
        if((i == (e-1)) && (e < end) && (alive > minFrac))
          path->pred[e++] = 0.0f;
        i++;
      }
    } 
    // flow = window start(or window end - 1)
    else {
      alive += path->state[i];
      while(i < em1) {
        float s = alive * trans[i];
        path->state[i] = s;
        alive -= s;
        if((i == b)&& (s < minFrac))
            b++;
        else
        	break;
        i++;
      }
      while(i < em1) {
        float s = alive * trans[i];
        path->state[i] = s;
        alive -= s;
        i++;
      }
      while(i < e) { 
        float s = alive * trans[i];
        path->state[i] = s;
        alive -= s;
        if((i == b)&& (s < minFrac))
            b++;
        if((i == (e-1)) && (e < end) && (alive > minFrac))
          path->pred[e++] = 0.0f;
        i++;
      }
    }
    path->window_start = b;
    path->window_end = e;
  }
}

#define AD_MINFRAC LD_VEC4F(1e-6f)
//// pseudo code:
//	for(int li=0;li<4;li++){
//	      rAlive[li] += parentState;
//
//		  if (parent->flow != child->flow or parent->flow == 0) {
//				// This nuc begins a new homopolymer
//			  rS[li] = parent->state[flow];
//		  }
//		  else{
//			  rS[li] = transition_base_[nuc][flow] * rAlive[li]
//		  }
//		  childState[j] = rS[li];
//		  rAlive[li] -= rS[li];
//
//	}
//// pseudo code:
#define advanceI( parentState, j, rParNuc, ts_tr, rAlive, rS) {\
\
	/*rS = (v4f)((v4i)LD_VEC4F(parentState) & (v4i) rParNuc);*/  \
	/*v4f rTemp1s; = rParNuc;*/ \
 \
	/* add parent state at this flow */ \
	rAlive += LD_VEC4F(parentState); \
 \
	/* one of the entries is 0xFFFF.. where the homopolymer is extended, rest are 0 */ \
	/* keep the parent state for child  where parent homopolymer is extended, rest are 0 */ \
	/*rS = (v4f) ((v4i) rS & (v4i) rParNuc); */\
 \
	/* select transitions where this nuc begins a new homopolymer */ \
	/*rTemp1s = rAlive * (v4f) (~(v4i)rParNuc & (v4i)(ts_tr)); */ \
 \
	/* multiply transition probabilities with alive */ \
	/*rTemp1s *= rAlive;*/ \
 \
	/* child state for this flow */ \
	rS = (v4f)((v4i)LD_VEC4F(parentState) & (v4i)rParNuc) + rAlive * (v4f) (~(v4i)rParNuc & (v4i)(ts_tr)); \
 \
	/* storing child states to the buffer */ \
	state_Buf[j].V = rS; \
 \
	/* alive *= transition_flow[nuc&7][flow] from DpTreephaser.cpp */ \
 \
	rAlive -= rS; \
}

// this one is all about calculating the residual
inline void TreephaserSSE::CalcResidualI(PathRec RESTRICT_PTR parent,
		int flow, int j, v4f &rS, v4f rTemp1s_, v4f &rPenNeg, v4f &rPenPos)
{

    v4f rTemp1S=(rTemp1s_);
	/* storing child predictions */
	pred_Buf[j].V = rTemp1S;

	/* apply recalibration model paramters to predicted signal if model is available */
	if (recalibrate_predictions_) {
		rTemp1S *= LD_VEC4F(parent->calib_A[flow]);
		rTemp1S += LD_VEC4F(parent->calib_B[flow]);
	}

	/* load normalized measurement for the parent */
	rS = LD_VEC4F(rd_NormMeasure[flow]);

	/* residual from normalized and predicted values for this flow */
	rS -= rTemp1S;

	rTemp1S = rS;
	/* find out the negative residual. The number which are -ve have highest bit one and therefore gives */
	/* four ints with 0's in the ones which are not negative */
	rS = (v4f)(_mm_srai_epi32((__m128i)rS,31));
	/*rS = (v4f) (((v4i) rS) >> 31);  get the signed bit... */

	/* squared residual */
	rTemp1S = rTemp1S * rTemp1S;

	/* select negative residuals */
	rS = (v4f) ((v4i) rS & (v4i) rTemp1S);

	/* select positive residuals */
	rTemp1S = (v4f) ((v4i) rTemp1S ^ (v4i) rS);

	/* add to negative penalty the square of negative residuals */
	rPenNeg += rS;

	/* add squared residuals to postive penalty */
	rPenPos += rTemp1S;

	/* running sum of negative penalties */
	nres_Buf[j].V = rPenNeg;
	/* running sum of positive penalties */
	pres_Buf[j].V = rPenPos;
}

#define CheckWindowStartBase()  \
/* tracking flow from parent->window_start */ \
v4i rTemp1i = (v4i)rBeg; \
 \
/* obtain window start for child which doesn't extend parent homopolymer. The one that extends */ \
/* has all bits for its word as 1 */ \
rTemp1i = (rTemp1i | (v4i)rParNuc); \
 \
/* compare parent window start to current flow i. All match except one where parent last hp extends */ \
rTemp1i = (v4i)_mm_cmpeq_epi32((__m128i)rTemp1i,(__m128i)LD_VEC4I(flow)); \
 \
/* filter min frac for nuc homopolymer child paths */ \
v4f rTemp1s = (v4f)(rTemp1i &  (v4i)AD_MINFRAC); \
 \
/* compares not less than equal to for two _m128i words. Entries will be 0xFFFF... for words where */ \
/* (kStateWindowCutoff > child->state[flow]). Rest of the words are 0 */ \
rTemp1s = (v4f)_mm_cmpnle_ps(rTemp1s, rS); \
 \
/* increasing child window start if child state less than state window cut off. */ \
rBeg = ((v4i)rBeg - (v4i)rTemp1s);


#define CheckWindowStart1() { \
CheckWindowStartBase(); \
\
/* this intrinsic gives sign of each word in binary indicating 1 for -ve sign and 0 for +ve */ \
/* if ad_adv is greater than 0, it indicates increase in child window start for some child path */ \
ad_Adv = _mm_movemask_ps(rTemp1s); \
\
}

#define CheckWindowStart2() { \
CheckWindowStartBase(); \
\
rTemp1i = (v4i)_mm_cmpeq_epi32((__m128i)rBeg, (__m128i)rEnd); \
rBeg += rTemp1i; \
}


#define CheckWindowEnd() { \
 \
v4f rTemp1s = (v4f)(LD_VEC4I(-1) + rEnd); \
rTemp1s = (v4f)((v4i)rTemp1s | (v4i)rParNuc); \
rTemp1s = (v4f)_mm_cmpeq_epi32((__m128i)rTemp1s, (__m128i)LD_VEC4I(flow)); \
rTemp1s = (v4f)((v4i)rTemp1s & (v4i)rAlive);\
rTemp1s = (v4f)_mm_cmpnle_ps(rTemp1s, AD_MINFRAC);\
/* child->window_end < max_flow */ \
rS = (v4f)_mm_cmpnle_ps((v4f)rFlowEnd, (v4f)rEnd); \
/* flow == child->window_end-1 and child->window_end < max_flow and alive > kStateWindowCutoff */ \
rTemp1s = (v4f)((v4i)rTemp1s & (v4i)rS); \
 \
/* if non zero than an increase in window end for some child paths */ \
ad_Adv = _mm_movemask_ps(rTemp1s); \
/* increases the child window end */ \
rEnd -= (v4i)rTemp1s; /* - (-1)  is the same thing as +1 */ \
}

void TreephaserSSE::advanceState4(PathRec RESTRICT_PTR parent, int end)
{

  int idx = parent->flow;

  // max flows
  v4i rFlowEnd = LD_VEC4I(end);
  // parent flow
  v4i rNucCpy = LD_VEC4I(idx);

  // child flows or the flow at which child nuc incorporates (corresponds to 
  // find child flow in AdvanceState() in DPTreephaser.cpp
  v4i rNucIdx = ts_NextNuc4[idx].V;
  rNucIdx = (v4i)_mm_min_epi16((__m128i)rNucIdx, (__m128i)rFlowEnd);

  // compare parent flow and child flows 
  rNucCpy = (v4i)_mm_cmpeq_epi32((__m128i)rNucCpy, (__m128i)rNucIdx);

  // store max_flow in ad_FlowEnd
  ad_FlowEnd.V = rFlowEnd;

  // four child flows in four 32 bit integers
  ad_Idx.V = rNucIdx;

  // changes datatype from int to float without doing any conversion
  v4f rParNuc = (v4f)(rNucCpy);

 // set alive to 0 for all 4 Nuc paths
  v4f rAlive = LD_VEC4F(0);

  // penalties for each nuc corresponding to four childs
  v4f rPenNeg = rAlive;
  v4f rPenPos = rAlive;

  int parLast = parent->window_end;
  v4i rEnd = LD_VEC4I(parLast);
  parLast--;
  v4i rBeg = LD_VEC4I(parent->window_start);


  int flow = parent->window_start;
  int j = 1;
  ad_Adv = 1;
  float *parent_CalibA= &parent->calib_A[0];
  float *parent_CalibB= &parent->calib_B[0];
  v4f rS;

  // iterate over the flows from parent->window_start to (parent->window_end - 1)
  // break this loop if child->window_start does not increase for any of the child paths from 
  // parent->window_start
  while(flow < parLast) {
    advanceI(parent->state[flow], j, rParNuc, ts_Transition4[flow].V, rAlive, rS);

    CheckWindowStart1();

    CalcResidualI(parent, flow, j, rS, (LD_VEC4F(parent->pred[flow])+rS),rPenNeg,rPenPos);

    flow++;
    j++;
    if(ad_Adv == 0)
      break;
  }

  // if none of the child paths has increase in window start.
  // this loop is the same as the previous loop, just doesn't check for window start
  if(EXPECTED(ad_Adv == 0)) {

    // child window start
    ad_Beg.V = rBeg;

    // flow < parent->window_end - 1
    while(flow < parLast) {

      advanceI(parent->state[flow], j, rParNuc, ts_Transition4[flow].V, rAlive, rS);

      CalcResidualI(parent, flow, j, rS, (LD_VEC4F(parent->pred[flow])+rS),rPenNeg,rPenPos);

      flow++;
      j++;
    }

    // flow = parent->window_end - 1
    {
      advanceI(parent->state[flow], j, rParNuc, (v4i) ts_Transition4[flow].V, rAlive, rS);

      CalcResidualI(parent, flow, j, rS, (LD_VEC4F(parent->pred[flow])+rS),rPenNeg,rPenPos);

      CheckWindowEnd();

      flow++;
      j++;
    }

   // flow >= parent window end
    while((flow < end) && (ad_Adv != 0)) {
      rS = (v4f)(~(v4i)rParNuc & (v4i)ts_Transition4[flow].V); //_mm_andnot_ps(rTemp1s, ts_Transition4[flow]);
      rS *= rAlive;
      state_Buf[j].V=rS;
      rAlive -= rS;

      CalcResidualI(parent, flow, j, rS, rS,rPenNeg,rPenPos);

      CheckWindowEnd();

      flow++;
      j++;
    }

    rEnd = (v4i)_mm_min_epi16((__m128i)rEnd, (__m128i)ad_FlowEnd.V);
    ad_End.V = rEnd;

  } 
  // This branch is for if one of the child paths has an increase in window_start 
  // flow = (parent->window_end - 1)
  else {

    {
      advanceI(parent->state[flow], j, rParNuc, ts_Transition4[flow].V, rAlive, rS);

      CalcResidualI(parent, flow, j, rS, (LD_VEC4F(parent->pred[flow])+rS),rPenNeg,rPenPos);

      CheckWindowStart2();

      CheckWindowEnd();

      flow++;
      j++;
    }

    // flow >= parent->window_end
    while((flow < end) && (ad_Adv != 0)) {
      v4f rTemp1s = rParNuc;
      rTemp1s = _mm_andnot_ps(rTemp1s, ts_Transition4[flow].V);
      rTemp1s *= rAlive;
      rS = rTemp1s;
      state_Buf[j].V=rS;
      rAlive -= rS;

      CheckWindowStart2();

      CalcResidualI(parent, flow, j, rS, rS,rPenNeg,rPenPos);

      CheckWindowEnd();

      flow++;
      j++;
    }

    rEnd = (v4i)_mm_min_epi16((__m128i)rEnd, (__m128i)ad_FlowEnd.V);
    ad_Beg.V = rBeg;
    ad_End.V = rEnd;

  }
}

void TreephaserSSE::sumNormMeasures() {
  int i = num_flows_;
  float sum = 0.0f;
  rd_SqNormMeasureSum[i] = 0.0f;
  while(--i >= 0)
    rd_SqNormMeasureSum[i] = (sum += rd_NormMeasure[i]*rd_NormMeasure[i]);
}

// -------------------------------------------------

void TreephaserSSE::RecalibratePredictions(PathRec *maxPathPtr)
{
  // Distort predictions according to recalibration model
  int to_flow = min(maxPathPtr->flow+1, num_flows_);

  for (int flow=0; flow<to_flow; flow++) {
    maxPathPtr->pred[flow] =
        maxPathPtr->pred[flow] * maxPathPtr->calib_A[flow]
          + maxPathPtr->calib_B[flow];
  }

}

void TreephaserSSE::ResetRecalibrationStructures(int num_flows) {
  for (int p = 0; p <= 8; ++p) {
    setValueSSE(&(sv_PathPtr[p]->calib_A[0]), 1.0f, num_flows_);
	setZeroSSE(&(sv_PathPtr[p]->calib_B[0]), num_flows_*sizeof(float));
  }
}

// --------------------------------------------------

void TreephaserSSE::SolveRead(BasecallerRead& read, int begin_flow, int end_flow)
{
  copySSE(rd_NormMeasure, &(read.normalized_measurements[0]), num_flows_*sizeof(float));
  setZeroSSE(sv_PathPtr[MAX_PATHS]->pred, num_flows_*sizeof(float)); // Not necessary?
  copySSE(sv_PathPtr[MAX_PATHS]->sequence, &(read.sequence[0]), (int)read.sequence.size()*sizeof(char));
  sv_PathPtr[MAX_PATHS]->sequence_length = read.sequence.size();

  Solve(begin_flow, end_flow);

  int to_flow = min(sv_PathPtr[MAX_PATHS]->window_end, end_flow);
  read.sequence.resize(sv_PathPtr[MAX_PATHS]->sequence_length);
  copySSE(&(read.sequence[0]), sv_PathPtr[MAX_PATHS]->sequence, sv_PathPtr[MAX_PATHS]->sequence_length*sizeof(char));
  setZeroSSE(&(read.prediction[0]), num_flows_*sizeof(float));
  copySSE(&(read.prediction[0]), sv_PathPtr[MAX_PATHS]->pred, to_flow*sizeof(float));
}

// -------------------------------------------------

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
  parent->state_inphase[0] = 1.0f;

  int pathCnt = 1;
  float bestDist = 1e20;
  end_flow = min(end_flow, num_flows_);

  // Simulating beginning of the read  up to or one base past begin_flow
  if(begin_flow > 0) {

    static const int char_to_nuc[8] = {-1, 0, -1, 1, 3, -1, -1, 2};

    for (int base = 0; base < best->sequence_length; ++base) {
      parent->sequence_length++;
      parent->sequence[base] = best->sequence[base];
      if (base and parent->sequence[base] != parent->sequence[base-1])
        parent->last_hp = 0;
      parent->last_hp++;
      if (parent->last_hp > MAX_HPXLEN) { // Safety to not overrun recalibration array
        parent->last_hp = MAX_HPXLEN;
      }

      advanceStateInPlace(parent, char_to_nuc[best->sequence[base]&7], num_flows_);
      if (parent->flow >= num_flows_)
        break;
      int to_flow = min(parent->window_end, end_flow);
      sumVectFloatSSE(&parent->pred[parent->window_start], &parent->state[parent->window_start], to_flow-parent->window_start);
//      for(int k = parent->window_start; k < to_flow; ++k) {
//        if((k & 3) == 0) {
//          sumVectFloatSSE(&parent->pred[k], &parent->state[k], to_flow-k);
//          break;
//        }
//        parent->pred[k] += parent->state[k];
//      }
      // Recalibration part of the initial simulation: log coefficients for simulation part
      if(recalibrate_predictions_) {
        parent->calib_A[parent->flow] = (*As_).at(parent->flow).at(flow_order_.int_at(parent->flow)).at(parent->last_hp);
        parent->calib_B[parent->flow] = (*Bs_).at(parent->flow).at(flow_order_.int_at(parent->flow)).at(parent->last_hp);
      }
      if (parent->flow >= begin_flow)
        break;
    }

    // No point solving the read if we simulated the whole thing.
    if(parent->window_end < begin_flow or parent->flow >= num_flows_) {
      sv_PathPtr[MAX_PATHS] = parent;
      sv_PathPtr[0] = best;
      return true;
    }
    parent->res = sumOfSquaredDiffsFloatSSE(
      (float*)rd_NormMeasure, (float*)parent->pred, parent->window_start);
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
   int parent_flow = parent->flow;

    // compute child path flow states, predicted signal,negative and positive penalties
    advanceState4(parent, end_flow);

    int n = pathCnt;
    double bestpen = 25.0;
    for(int nuc = 0; nuc < 4; ++nuc) {
      PathRec RESTRICT_PTR child = sv_PathPtr[n];

      child->flow = min(ad_Idx.A[nuc], end_flow);
      child->window_start = ad_Beg.A[nuc];
      child->window_end = min(ad_End.A[nuc], end_flow);

      // Do not attempt to calculate child->last_hp in this loop; bad things happen
      if(child->flow >= end_flow or parent->last_hp >= MAX_HPXLEN or parent->sequence_length >= 2*MAX_VALS-10)
        continue;

      // pointer in the ad_Buf buffer pointing at the running sum of positive residuals at start of parent window
      //char RESTRICT_PTR pn = ad_Buf+nuc*4+(AD_NRES_OFS-16)-parent->window_start*16;
      int cwsIdx=child->window_start-parent->window_start;
      int cfIdx =child->flow-parent->window_start;
      int cweIdx=child->window_end-parent->window_start;

      // child path metric
      float metr = parent->res + pres_Buf[cwsIdx].A[nuc];//*((float RESTRICT_PTR)(pn+child->window_start*16+(AD_PRES_OFS-AD_NRES_OFS)));

      // sum of squared residuals for positive residuals for flows < child->flow
      float penPar = pres_Buf[cfIdx].A[nuc];//*((float RESTRICT_PTR)(pn+child->flow*16+(AD_PRES_OFS-AD_NRES_OFS)));

      // sum of squared residuals for negative residuals for flows < child->window_end
      float penNeg = nres_Buf[cweIdx].A[nuc];//*((float RESTRICT_PTR)(pn+child->window_end*16));

      // sum of squared residuals left of child window start
      child->res = metr + nres_Buf[cwsIdx].A[nuc];//*((float RESTRICT_PTR)(pn+child->window_start*16));
      
      metr += penNeg;

      // penPar corresponds to penalty1 in DPTreephaser.cpp
      penPar += penNeg;
      penNeg += penPar;

      // penalty = penalty1 + (kNegativeMultiplier = 2)*penNeg
      if(penNeg >= 20.0)
        continue;
 
      if(bestpen > penNeg)
        bestpen = penNeg;
      else if(penNeg-bestpen >= 0.2)
        continue;

      // child->path_metric > sum_of_squares_upper_bound
      if(metr > bestDist)
        continue;

      float newSignal = rd_NormMeasure[child->flow];
      
      // XXX Right here we are having a memory overrun: We copied up to parent->flow but use until parent->window_end
      // Check 'dot' criterion
      if(child->flow < parent->window_end){
        if (recalibrate_predictions_)
          newSignal -= (parent->calib_A[child->flow]*parent->pred[child->flow]+parent->calib_B[child->flow]);
        else
          newSignal -= parent->pred[child->flow];
      }
   	  newSignal /= state_Buf[cfIdx+1].A[nuc];// *((float RESTRICT_PTR)(ad_Buf+nuc*4+AD_STATE_OFS + (child->flow-parent->window_start)*16));
      child->dotCnt = 0;
      if(newSignal < 0.3f) {
        if(parent->dotCnt > 0)
          continue;
        child->dotCnt = 1;
      }
      // child path survives at this point
      child->metr = float(metr);
      child->flowMetr = float(penPar);
      child->penalty = float(penNeg);
      child->nuc = nuc;
      ++n;
    }

    // XXX Right here we are having a memory overrun: We copied up to parent->flow but use until parent->window_end of calibA and calibB
    // Computing squared distance between parent's predicted signal and normalized measurements
    float dist = parent->res+(rd_SqNormMeasureSum[parent->window_end]-rd_SqNormMeasureSum[end_flow]);
	if (recalibrate_predictions_) {
		int i=parent->window_start;
	  dist += sumOfSquaredDiffsFloatSSE_recal((float RESTRICT_PTR)(&(rd_NormMeasure[i])),
											  (float RESTRICT_PTR)(&(parent->pred[i])),
											  (float RESTRICT_PTR)(&(parent->calib_A[i])),
											  (float RESTRICT_PTR)(&(parent->calib_B[i])),
											   parent->window_end-i);
	} else {
		int i=parent->window_start;
	  dist += sumOfSquaredDiffsFloatSSE((float RESTRICT_PTR)(&(rd_NormMeasure[i])),
										(float RESTRICT_PTR)(&(parent->pred[i])),
										 parent->window_end-i);
	}
    // Finished computing squared distance

    int bestPathIdx = -1;

    // current best path is parent path
    if(bestDist > dist) {
      bestPathIdx = parentPathIdx;
      parentPathIdx = -1;
    }

    int childPathIdx = -1;
    while(pathCnt < n) {
      PathRec RESTRICT_PTR child = sv_PathPtr[pathCnt];
      // Rule that depends on finding the best nuc
      if(child->penalty-bestpen >= 0.2f) {
        sv_PathPtr[pathCnt] = sv_PathPtr[--n];
        sv_PathPtr[n] = child;
      } 
      else if((childPathIdx < 0) && (parentPathIdx >= 0)) {
        sv_PathPtr[pathCnt] = sv_PathPtr[--n];
        sv_PathPtr[n] = child;
        childPathIdx = n;
      }
      // this is the child path to be kept 
      else {
        if (child->flow)
          child->flowMetr = (child->metr + 0.5f*child->flowMetr) / child->flow;
        //char RESTRICT_PTR p = ad_Buf+child->nuc*4+AD_STATE_OFS;
        {
        	int len=child->window_end-parent->window_start;
            copySSE_4th(&child->state[parent->window_start],&state_Buf[1].A[child->nuc],len);
            copySSE_4th(&child->pred[parent->window_start],&pred_Buf[1].A[child->nuc],len);
        }
//        for(int i = parent->window_start, j = 1, e = child->window_end; i < e; ++i, j ++) {
//		  child->state[i] = state_Buf[j].A[child->nuc];//*((float*)(ad_Buf+child->nuc*4+AD_STATE_OFS + j));
//          child->pred[i] = pred_Buf[j].A[child->nuc];  //*((float*)(ad_Buf+child->nuc*4+AD_PRED_OFS + j));
//        }
        copySSE(child->pred, parent->pred, parent->window_start << 2);

        copySSE(child->sequence, parent->sequence, parent->sequence_length);

        if(state_inphase_enabled_){
            if(child->flow > 0){
              int cpSize = (parent->flow+1)*sizeof(float);
              copySSE(child->state_inphase, parent->state_inphase, cpSize);
            }
            //extending from parent->state_inphase[parent->flow] to fill the gap
            for(int tempInd = parent->flow+1; tempInd < child->flow; tempInd++){
                child->state_inphase[tempInd] = max(child->state[child->flow],0.01f);
            }
            child->state_inphase[child->flow] = max(child->state[child->flow],0.01f);
        }

        child->sequence_length = parent->sequence_length + 1;
        child->sequence[parent->sequence_length] = flow_order_[child->flow];
        if (parent->sequence_length and child->sequence[parent->sequence_length] != child->sequence[parent->sequence_length-1])
          child->last_hp = 0;
        else
          child->last_hp = parent->last_hp;
        child->last_hp++;

        // copy whole vector to avoid memory access to fields that have been written to by (longer) previously discarded paths XXX
        // --> Reintroducing memory overrun since it seems to yield better performance
        if (recalibrate_predictions_) {
          if(child->flow > 0){
            // --- Reverting to old code with memory overrun
            int cpSize = (parent->flow+1) << 2;
            //memcpy(child->calib_A, parent->calib_A, cpSize);
            //memcpy(child->calib_B, parent->calib_B, cpSize);
            // ---
            copySSE(child->calib_A, parent->calib_A, cpSize);
            copySSE(child->calib_B, parent->calib_B, cpSize);
          }
          //explicitly fill zeros between parent->flow and child->flow;
          for(int tempInd = parent->flow + 1; tempInd < child->flow; tempInd++){
            child->calib_A[tempInd] = 1.0f;
            child->calib_B[tempInd] = 0.0f;
          }
          int hp_length = min(child->last_hp, MAX_HPXLEN);
          child->calib_A[child->flow] = (*As_).at(child->flow).at(flow_order_.int_at(child->flow)).at(hp_length);
          child->calib_B[child->flow] = (*Bs_).at(child->flow).at(flow_order_.int_at(child->flow)).at(hp_length);
        }
        ++pathCnt;
      }
    }

    // In the event, there is no best path, one of the child is copied to the parent
    if(childPathIdx >= 0) {
      PathRec RESTRICT_PTR child = sv_PathPtr[childPathIdx];
      parent_flow = parent->flow; //MJ
      parent->flow = child->flow;
      parent->window_end = child->window_end;
      parent->res = child->res;
      parent->metr = child->metr;
      (child->flow == 0) ? (parent->flowMetr == 0) : (parent->flowMetr = (child->metr + 0.5f*child->flowMetr) / child->flow);
      parent->dotCnt = child->dotCnt;
      //char RESTRICT_PTR p = ad_Buf+child->nuc*4+AD_STATE_OFS;
      for(int i = parent->window_start, j = 1, e = child->window_end; i < e; ++i, j ++) {
        parent->state[i] = state_Buf[j].A[child->nuc];//*((float*)(ad_Buf+child->nuc*4+AD_STATE_OFS+j));
        parent->pred[i] = pred_Buf[j].A[child->nuc];//*((float*)(ad_Buf+child->nuc*4+AD_PRED_OFS+j));
      }

      parent->sequence[parent->sequence_length] = flow_order_[parent->flow];
      if (parent->sequence_length and parent->sequence[parent->sequence_length] != parent->sequence[parent->sequence_length-1])
        parent->last_hp = 0;
      parent->last_hp++;
      parent->sequence_length++;

      //update calib_A and calib_B for parent
      if (recalibrate_predictions_) {
        for(int tempInd = parent_flow + 1; tempInd < child->flow; tempInd++){
          parent->calib_A[tempInd] = 1.0f;
          parent->calib_B[tempInd] = 0.0f;
        }
        parent->calib_A[parent->flow] = (*As_).at(parent->flow).at(flow_order_.int_at(parent->flow)).at(parent->last_hp);
        parent->calib_B[parent->flow] = (*Bs_).at(parent->flow).at(flow_order_.int_at(parent->flow)).at(parent->last_hp);
      }

      if(state_inphase_enabled_){
          for(int tempInd = parent_flow+1; tempInd < parent->flow; tempInd++){
              parent->state_inphase[tempInd] = parent->state[parent->flow];
          }
          parent->state_inphase[parent->flow] = parent->state[parent->flow];
      }

      parent->window_start = child->window_start;
      parentPathIdx = -1;
    }

    // updating parent as best path
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

  // At the end change predictions according to recalibration model and reset data structures
  if (recalibrate_predictions_) {
    RecalibratePredictions(sv_PathPtr[MAX_PATHS]);
    ResetRecalibrationStructures(end_flow);
  }

  return false;
}


void TreephaserSSE::WindowedNormalize(BasecallerRead& read, int num_steps)
{
//  int num_flows = read.raw_measurements.size();
  float median_set[windowSize_];

  // Estimate and correct for additive offset

  float next_normalizer = 0;
  int estim_flow = 0;
  int apply_flow = 0;

  for (int step = 0; step <= num_steps; ++step) {

    int window_end = estim_flow + windowSize_;
    int window_middle = estim_flow + windowSize_ / 2;
    if (window_middle > num_flows_)
      break;

    float normalizer = next_normalizer;

    int median_set_size = 0;
    for (; estim_flow < window_end and estim_flow < num_flows_ and estim_flow < sv_PathPtr[MAX_PATHS]->window_end; ++estim_flow)
      if (sv_PathPtr[MAX_PATHS]->pred[estim_flow] < 0.3)
        median_set[median_set_size++] = read.raw_measurements[estim_flow] - sv_PathPtr[MAX_PATHS]->pred[estim_flow];

    if (median_set_size > 5) {
      //cout << step << ":" << median_set_size << ":" << windowSize_ << endl;
      std::nth_element(median_set, median_set + median_set_size/2, median_set + median_set_size);
      next_normalizer = median_set[median_set_size / 2];
      if (step == 0)
        normalizer = next_normalizer;
    }

    float delta = (next_normalizer - normalizer) / static_cast<float>(windowSize_);

    for (; apply_flow < window_middle and apply_flow < num_flows_; ++apply_flow) {
      //cout << apply_flow << ":" << window_middle << ":" << num_flows_ << endl;
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

    int window_end = estim_flow + windowSize_;
    int window_middle = estim_flow + windowSize_ / 2;
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

    float delta = (next_normalizer - normalizer) / static_cast<float>(windowSize_);

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


// ------------------------------------------------------------------------
// Compute quality metrics
// Why does this function completely ignore recalibration?

void  TreephaserSSE::ComputeQVmetrics(BasecallerRead& read)
{
  static const char nuc_int_to_char[5] = "ACGT";
  int num_flows = flow_order_.num_flows();
  read.state_inphase.assign(num_flows, 1);
  read.state_total.assign(num_flows, 1);

  if (read.sequence.empty())
    return;
  int num_bases = read.sequence.size();
  read.penalty_mismatch.assign(num_bases, 0);
  read.penalty_residual.assign(num_bases, 0);

  PathRec RESTRICT_PTR parent = sv_PathPtr[0];
  PathRec RESTRICT_PTR children[4] = {sv_PathPtr[1], sv_PathPtr[2], sv_PathPtr[3], sv_PathPtr[4]};
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

  float recent_state_inphase = 1;
  float recent_state_total = 1;

  // main loop for base calling
  for (int solution_flow = 0, base = 0; solution_flow < num_flows; ++solution_flow) {
      for (; base<num_bases and read.sequence[base]==flow_order_[solution_flow]; ++base) {
          if(recalibrate_predictions_) {
            parent->calib_A[parent->flow] = (*As_).at(parent->flow).at(flow_order_.int_at(parent->flow)).at(parent->last_hp);
            parent->calib_B[parent->flow] = (*Bs_).at(parent->flow).at(flow_order_.int_at(parent->flow)).at(parent->last_hp);
          }
          // compute child path flow states, predicted signal,negative and positive penalties
          advanceState4(parent, num_flows);

          float penalty[4] = { 0, 0, 0, 0 };
          int called_nuc = -1;
      for(int nuc = 0; nuc < 4; ++nuc) {
        PathRec RESTRICT_PTR child = children[nuc];
        if (nuc_int_to_char[nuc] == flow_order_[solution_flow])
          called_nuc = nuc;
        child->flow = min(ad_Idx.A[nuc], flow_order_.num_flows());
        child->window_end = min(ad_End.A[nuc], flow_order_.num_flows());
        child->window_start = min(ad_Beg.A[nuc], child->window_end);

        // Apply easy termination rules
        if (child->flow >= num_flows || parent->last_hp >= MAX_HPXLEN ) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        // pointer in the ad_Buf buffer pointing at the running sum of positive residuals at start of parent window
//        char RESTRICT_PTR pn = ad_Buf+nuc*4+(AD_NRES_OFS-16)-parent->window_start*16;

        // sum of squared residuals for positive residuals for flows < child->flow
        float penPar = pres_Buf[child->flow-parent->window_start].A[nuc];// *((float RESTRICT_PTR)(ad_Buf+nuc*4+(AD_PRES_OFS)+(child->flow-parent->window_start-1)*16));

        // sum of squared residuals for negative residuals for flows < child->window_end
        float penNeg = nres_Buf[child->window_end-parent->window_start].A[nuc];// *((float RESTRICT_PTR)(ad_Buf+nuc*4+AD_NRES_OFS+(child->window_end-parent->window_start-1)*16));

        penalty[nuc] = penPar + penNeg;
      }

      // find current incorporating base
      assert(called_nuc > -1);
      assert(children[called_nuc]->flow == solution_flow);

      PathRec RESTRICT_PTR childToKeep = children[called_nuc];
      //copy
//      char RESTRICT_PTR p = ad_Buf+ called_nuc*4 + AD_STATE_OFS;

      recent_state_total = 0;
      for(int i = parent->window_start, j = 1, e = childToKeep->window_end; i < e; ++i, j ++) {
        childToKeep->state[i] = state_Buf[j].A[called_nuc];// *((float*)(p+j*16));
        childToKeep->pred[i] = pred_Buf[j].A[called_nuc];// *((float*)(p+j*16+(AD_PRED_OFS-AD_STATE_OFS)));
        recent_state_total += childToKeep->state[i];
      }
      //sse implementation with aligned memory; no gain as the number of elements to be summed up is small
//      recent_state_total = vecSumSSE(state_Buf, countStates);

      copySSE(childToKeep->pred, parent->pred, parent->window_start << 2);

      if (childToKeep->flow == parent->flow)
        childToKeep->last_hp = parent->last_hp + 1;
      else
        childToKeep->last_hp = 1;

      recent_state_inphase = childToKeep->state[solution_flow];

      // Get delta penalty to next best solution
      read.penalty_mismatch[base] = -1; // min delta penalty to earlier base hypothesis
      read.penalty_residual[base] = 0;

      if (solution_flow - parent->window_start > 0)
        read.penalty_residual[base] = penalty[called_nuc] / (solution_flow - parent->window_start);

      for (int nuc = 0; nuc < 4; ++nuc) {
        if (nuc == called_nuc)
            continue;
        float penalty_mismatch = penalty[called_nuc] - penalty[nuc];
        read.penalty_mismatch[base] = max(read.penalty_mismatch[base], penalty_mismatch);
      }

      // Called state is the starting point for next base
      PathRec RESTRICT_PTR swap = parent;
      parent = children[called_nuc];
      children[called_nuc] = swap;
    }
    read.state_inphase[solution_flow] = max(recent_state_inphase, 0.01f);
    read.state_total[solution_flow] = max(recent_state_total, 0.01f);
    }

  if(recalibrate_predictions_) {
    RecalibratePredictions(parent);
    ResetRecalibrationStructures(num_flows_);
  }
  setZeroSSE(&read.prediction[0], num_flows_*sizeof(float));
  copySSE(&read.prediction[0], parent->pred, parent->window_end*sizeof(float));
}


void  TreephaserSSE::ComputeQVmetrics_flow(BasecallerRead& read, vector<int>& flow_to_base, const bool flow_predictors_)
{
  static const char nuc_int_to_char[5] = "ACGT";
  int num_flows = flow_order_.num_flows();
  read.state_inphase.assign(num_flows, 1);
  read.state_total.assign(num_flows, 1);

  if (read.sequence.empty())
    return;
  int num_bases = read.sequence.size();
  read.penalty_mismatch.assign(num_bases, 0);
  read.penalty_residual.assign(num_bases, 0);
  if (flow_predictors_) {
      read.penalty_mismatch_flow.assign(num_flows, 0);
      read.penalty_residual_flow.assign(num_flows, 0);
  }

  PathRec RESTRICT_PTR parent = sv_PathPtr[0];
  PathRec RESTRICT_PTR children[4] = {sv_PathPtr[1], sv_PathPtr[2], sv_PathPtr[3], sv_PathPtr[4]};
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

  float recent_state_inphase = 1;
  float recent_state_total = 1;

  // main loop for base calling
  for (int solution_flow = 0, base = 0; solution_flow < num_flows; ++solution_flow) {
      for (; base<num_bases and read.sequence[base]==flow_order_[solution_flow]; ++base) {
          if(recalibrate_predictions_) {
            parent->calib_A[parent->flow] = (*As_).at(parent->flow).at(flow_order_.int_at(parent->flow)).at(parent->last_hp);
            parent->calib_B[parent->flow] = (*Bs_).at(parent->flow).at(flow_order_.int_at(parent->flow)).at(parent->last_hp);
          }
          // compute child path flow states, predicted signal,negative and positive penalties
          advanceState4(parent, num_flows);

          float penalty[4] = { 0, 0, 0, 0 };
          int called_nuc = -1;
      for(int nuc = 0; nuc < 4; ++nuc) {
        PathRec RESTRICT_PTR child = children[nuc];
        if (nuc_int_to_char[nuc] == flow_order_[solution_flow])
          called_nuc = nuc;
        child->flow = min(ad_Idx.A[nuc], flow_order_.num_flows());
        child->window_end = min(ad_End.A[nuc], flow_order_.num_flows());
        child->window_start = min(ad_Beg.A[nuc], child->window_end);

        // Apply easy termination rules
        if (child->flow >= num_flows || parent->last_hp >= MAX_HPXLEN ) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        // pointer in the ad_Buf buffer pointing at the running sum of positive residuals at start of parent window
//        char RESTRICT_PTR pn = ad_Buf+nuc*4+(AD_NRES_OFS-16)-parent->window_start*16;

        // sum of squared residuals for positive residuals for flows < child->flow
        float penPar = pres_Buf[child->flow-parent->window_start].A[nuc];// *((float RESTRICT_PTR)(ad_Buf+nuc*4+(AD_PRES_OFS)+(child->flow-parent->window_start-1)*16));

        // sum of squared residuals for negative residuals for flows < child->window_end
        float penNeg = nres_Buf[child->window_end-parent->window_start].A[nuc];// *((float RESTRICT_PTR)(ad_Buf+nuc*4+AD_NRES_OFS+(child->window_end-parent->window_start-1)*16));

        penalty[nuc] = penPar + penNeg;
      }

      // find current incorporating base
      assert(called_nuc > -1);
      assert(children[called_nuc]->flow == solution_flow);

      PathRec RESTRICT_PTR childToKeep = children[called_nuc];
      //copy
//      char RESTRICT_PTR p = ad_Buf+ called_nuc*4 + AD_STATE_OFS;

      recent_state_total = 0;
      for(int i = parent->window_start, j = 1, e = childToKeep->window_end; i < e; ++i, j ++) {
        childToKeep->state[i] = state_Buf[j].A[called_nuc];// *((float*)(p+j*16));
        childToKeep->pred[i] = pred_Buf[j].A[called_nuc];// *((float*)(p+j*16+(AD_PRED_OFS-AD_STATE_OFS)));
        recent_state_total += childToKeep->state[i];
      }
      //sse implementation with aligned memory; no gain as the number of elements to be summed up is small
//      recent_state_total = vecSumSSE(state_Buf, countStates);

      copySSE(childToKeep->pred, parent->pred, parent->window_start << 2);

      if (childToKeep->flow == parent->flow)
        childToKeep->last_hp = parent->last_hp + 1;
      else
        childToKeep->last_hp = 1;

      recent_state_inphase = childToKeep->state[solution_flow];

      // Get delta penalty to next best solution
      read.penalty_mismatch[base] = -1; // min delta penalty to earlier base hypothesis
      read.penalty_residual[base] = 0;

      if (solution_flow - parent->window_start > 0)
        read.penalty_residual[base] = penalty[called_nuc] / (solution_flow - parent->window_start);

      for (int nuc = 0; nuc < 4; ++nuc) {
        if (nuc == called_nuc)
            continue;
        float penalty_mismatch = penalty[called_nuc] - penalty[nuc];
        read.penalty_mismatch[base] = max(read.penalty_mismatch[base], penalty_mismatch);
      }

      // Called state is the starting point for next base
      PathRec RESTRICT_PTR swap = parent;
      parent = children[called_nuc];
      children[called_nuc] = swap;
    }
    read.state_inphase[solution_flow] = max(recent_state_inphase, 0.01f);
    read.state_total[solution_flow] = max(recent_state_total, 0.01f);
    }

  if (flow_predictors_) { //if (flow_predictors_)
      //vector<int> flows_to_proc;
      for (int solution_flow = 0; solution_flow < num_flows; ++solution_flow) {
          int curr_base = flow_to_base[solution_flow];
          if (curr_base >= 0) {
              // copy from what's stored in read.penalty_mismatch[base]
              read.penalty_mismatch_flow[solution_flow] = read.penalty_mismatch[curr_base];
              read.penalty_residual_flow[solution_flow] = read.penalty_residual[curr_base];
              /*
              int nFlows = flows_to_proc.size();
              if (nFlows>0) {
              for (int i=0; i<nFlows; ++i) {
                  int flow = flows_to_proc[i];
                  read.penalty_mismatch_flow[flow] = 0;
                  read.penalty_residual_flow[flow] = 0;
                  }
              flows_to_proc.clear();
              */
              }
          else {
              //flows_to_proc.push_back(solution_flow);
              continue;
            }
          }
      }

  if(recalibrate_predictions_) {
    RecalibratePredictions(parent);
    ResetRecalibrationStructures(num_flows_);
  }
  setZeroSSE(&read.prediction[0], num_flows_*sizeof(float));
  copySSE(&read.prediction[0], parent->pred, parent->window_end*sizeof(float));
}



