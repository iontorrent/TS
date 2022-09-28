/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "TreephaserVEC.h"
#include <x86intrin.h>

#include <vector>
#include <algorithm>
#include <math.h>
#include <cstring>
#include <cassert>
#include <stdint.h>

//#define PRINT_DEBUG 1
//#define DO_DEBUG (dbgX==1080 && dbgY==2133)

#define DFLT_CACHE 0x100000

using namespace std;

ALWAYS_INLINE float Sqr(float val) {
  return val*val;
}

#define memcpy_C(d,s,l)  memcpy(d,s,l)

#define memcpy_F(d,s,l)  { \
   float *dst=(d); float *src=(s); int size=(l); \
   while (size--)  dst[size] = src[size]; }

#define memcpy_4F(d,s,l)  { \
   float *dst=(d); float *src=(s); int size=(l); \
   while (size--)  dst[size] = src[4*size]; }

#define memcpy_4F4(d,s,l)  { \
   float *dst=(d); float *src=(s); int size=(l); \
   while (size--)  dst[4*size] = src[4*size]; }


#define memset_F(d, v, s) { \
  float *dst=(d); float val=v; int size=s; \
  while (size--) dst[size] = val; }

#define memset_C(d, v, s) memset(d,v,s)

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

  register v4f sumV;
  sumV = (v4f){sum,0,0,0};
  v4f *src1V=(v4f *)src1;
  v4f *src2V=(v4f *)src2;
  while(count--) {
    v4f r1 = src1V[count] - src2V[count];
    sumV += r1*r1;
  }

  sum = (sumV[3]+sumV[1])+(sumV[2]+sumV[0]);
  return sum;
}


// ----------------------------------------------------------------------------

// Constructor used in variant caller
TreephaserVEC::TreephaserVEC()
  : flow_order_("TACG", 4), my_cf_(-1.0), my_ie_(-1.0), As_(NULL), Bs_(NULL),
    bad_path_limit_(10000), many_path_limit_(10000),
    initial_paths_(-1),max_metr_diff_(0.3)
{
  SetNormalizationWindowSize(38);
  SetFlowOrder(flow_order_);
}

// Constructor used in Basecaller
TreephaserVEC::TreephaserVEC(const ion::FlowOrder& flow_order, const int windowSize)
  : my_cf_(-1.0), my_ie_(-1.0), As_(NULL), Bs_(NULL),
    bad_path_limit_(10000), many_path_limit_(10000),
    initial_paths_(-1),max_metr_diff_(0.3)
{
  SetNormalizationWindowSize(windowSize);
  SetFlowOrder(flow_order);
}

// ----------------------------------------------------------------
// Initilizes all float variables to NAN so that they cause mayhem if we read out of bounds
// and so that valgrind does not complain about uninitialized variables
void TreephaserVEC::InitializeVariables() {

  
  // Initializing the other variables of the object
  for (unsigned int idx=0; idx<4; idx++) {
    ad_FlowEnd[idx] = 0;
    flow_Buf[idx] = 0;
    winEnd_Buf[idx] = 0;
    winStart_Buf[idx] = 0;
  }
  for (unsigned int val=0; val<MAX_VALS; val++) {
    rd_NormMeasure[val]      = 0;
    rd_SqNormMeasureSum[val] = 0;
  }
  ad_Adv = 0;
}


// ----------------------------------------------------------------

// Initialize Object
void TreephaserVEC::SetFlowOrder(const ion::FlowOrder& flow_order)
{
  flow_order_ = flow_order;
  num_flows_ = flow_order.num_flows();

  // For some perverse reason cppcheck does not like this loop
  for (int path = 0; path < MAX_ACT_PATHS; path++)
    sv_PathPtr[path] = &(sv_pathBuf[path]);

  // -- For valgrind and debugging & to make cppcheck happy
  InitializeVariables();
  // --

  v4i nextIdx = LD_VEC4I(num_flows_);
  for(int flow = num_flows_-1; flow >= 0; --flow) {
    nextIdx[flow_order_.int_at(flow)] = flow;
    ts_NextNuc[flow] = nextIdx;
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

  pm_model_available_              = false;
  recalibrate_predictions_         = false;
  state_inphase_enabled_           = false;
  skip_recal_during_normalization_ = false;
}

// ----------------------------------------------------------------

void TreephaserVEC::SetModelParameters(double cf, double ie)
{
  if (cf == my_cf_ and ie == my_ie_)
    return;
  
  double dist[4] = { 0.0, 0.0, 0.0, 0.0 };

  for(int flow = 0; flow < num_flows_; ++flow) {
    dist[flow_order_.int_at(flow)] = 1.0;
    ts_Transition[flow][0] = float(dist[0]*(1-ie));
    dist[0] *= cf;
    ts_Transition[flow][1] = float(dist[1]*(1-ie));
    dist[1] *= cf;
    ts_Transition[flow][2] = float(dist[2]*(1-ie));
    dist[2] *= cf;
    ts_Transition[flow][3] = float(dist[3]*(1-ie));
    dist[3] *= cf;
  }
  my_cf_ = cf;
  my_ie_ = ie;
}

// ----------------------------------------------------------------

void TreephaserVEC::NormalizeAndSolve(BasecallerRead& read)
{
  memcpy_F(rd_NormMeasure, &read.raw_measurements[0], num_flows_);
  // Disable recalibration during normalization stage if requested
  if (skip_recal_during_normalization_)
    recalibrate_predictions_ = false;

  Solve(ts_StepBeg[0], ts_StepEnd[0], ts_StepBeg[1], initial_paths_, max_metr_diff_);
  WindowedNormalize(read, 0);
  for(int step = 0; step < ts_StepCnt; ++step) {
    bool is_final = Solve(ts_StepBeg[step], ts_StepEnd[step], ts_StepBeg[step+1], initial_paths_, max_metr_diff_);
    WindowedNormalize(read, step);
    if (is_final)
      break;
  }

  //final stage of solve and calculate the state_inphase for QV prediction
  state_inphase_enabled_ = true;
  // And turn recalibration back on (if available) for the final solving part
  EnableRecalibration();

  Solve(ts_StepBeg[ts_StepCnt], ts_StepEnd[ts_StepCnt], ts_StepEnd[ts_StepCnt], -1, max_metr_diff_);

  int to_flow = min(sv_PathPtr[BEST_PATH]->window_end, num_flows_); // Apparently window_end can be larger than num_flows_
  read.sequence.resize(sv_PathPtr[BEST_PATH]->sequence_length);
  memcpy_C(&read.sequence[0], sv_PathPtr[BEST_PATH]->sequence, sv_PathPtr[BEST_PATH]->sequence_length);
  memcpy_F(&read.normalized_measurements[0], rd_NormMeasure, num_flows_);
  memset_F(&read.prediction[to_flow], 0, num_flows_-to_flow);
  memcpy_F(&read.prediction[0], sv_PathPtr[BEST_PATH]->pred, to_flow);
  memset_F(&read.state_inphase[to_flow], 0, num_flows_-to_flow);
  memcpy_F(&read.state_inphase[0], sv_PathPtr[BEST_PATH]->state_inphase, to_flow);

  // copy inphase population and reset state_inphase flag
  if(state_inphase_enabled_){
    for (int p = 0; p <= 8; ++p) {
      memset_F(&(sv_PathPtr[p]->state_inphase[0]), 0, num_flows_);
    }
  }
  state_inphase_enabled_ = false;
}



void TreephaserVEC::advanceState4 (PathRecV RESTRICT_PTR path, int end)
{

  ALIGN(64)v4f nres2_Buf[MAX_VALS];
  ALIGN(64)v4f pres2_Buf[MAX_VALS];

  // child flows or the flow at which child nuc incorporates (corresponds to
  // find child flow in AdvanceState() in DPTreephaser.cpp
  flow_Buf = (v4i)_mm_min_epi16((__m128i)ts_NextNuc[path->flow], (__m128i)LD_VEC4I(end));

  // compare parent flow and child flows
  v4f rParNuc = (v4f)_mm_cmpeq_epi32((__m128i)LD_VEC4I(path->flow), (__m128i)flow_Buf);

  // set alive to 0 for all 4 Nuc paths
  v4f alive = LD_VEC4F(0);

  // penalties for each nuc corresponding to four childs
  v4f rPenNeg = alive;
  v4f rPenPos = alive;
  nres2_Buf[0]= alive;
  pres2_Buf[0]= alive;
  //path->pred_Buf[path->window_start]=alive;
  const v4f minFrac = LD_VEC4F( 1e-6 );

  int endi = path->window_end;
  int flow = path->window_start;
  v4i e =LD_VEC4I(endi);
  v4i b =LD_VEC4I(flow);


  v4i bInc = LD_VEC4I(-1);
  v4i eInc = bInc;
  int lastNuc=path->nuc;

  v4f *ps=path->state;
  v4f *tst=ts_Transition;
  v4f *nsb=&nres2_Buf[0];
  v4f *psb=&pres2_Buf[0];
  v4f *pdb=path->pred_Buf;
  float *rdn=rd_NormMeasureAdj;
  int j=1;

  while (flow < endi) {
    // advance
    float state=*(float *)&ps[flow][lastNuc];
    alive += state;
    v4f s = (v4f)((v4i)LD_VEC4F(state) & (v4i)rParNuc) +
		      alive * (v4f) (~(v4i)rParNuc & (v4i)tst[flow]);
    ps[flow] = s;
    alive -= s;

    // check window start
    bInc = (bInc & (v4i) (s < minFrac));
    b -= bInc; // minus -1 is the same as +1

    v4f pred = s + pdb[flow][lastNuc];
    pdb[flow] = pred;

    // compute residuals
    v4f res = rdn[flow] - pred; // difference between real signal and prediction
    v4i resSel = (v4i) (_mm_srai_epi32 ((__m128i ) res, 31)); // fills with sign bits.. negative residuals are all 1's
    res = res*res;
    rPenNeg += (v4f) (resSel & (v4i) res);
    rPenPos += (v4f) (~resSel & (v4i) res);
    nsb[j] = rPenNeg;
    psb[j] = rPenPos;

    flow++;
    j++;
  }

  b += (v4i) (e == b); // if (window_start==window_end) window_start--;
  eInc = eInc & (v4i) (alive > minFrac);
  e -= eInc; // minus -1 same as plus 1

  // flow >= path->window_end
  while ((flow < end) && _mm_movemask_ps ((v4f) eInc)) {
    v4f s = alive * (v4f) (~(v4i)rParNuc & (v4i)tst[flow]);
    ps[flow] = s;
    alive -= s;

    v4f pred = s;
    pdb[flow] = pred;
    v4f res = rdn[flow] - pred; // difference between real signal and prediction
    v4i resSel = (v4i) (_mm_srai_epi32 ((__m128i ) res, 31)); // fills with sign bits.. negative residuals are all 1's
    res = res*res;
    rPenNeg += (v4f) (resSel & (v4i) res);
    rPenPos += (v4f) (~resSel & (v4i) res);
    nsb[j] = rPenNeg;
    psb[j] = rPenPos;

    eInc = eInc & (v4i) (alive > minFrac);
    e -= eInc;
    flow++;
    j++;
  }

  nres_WE=rPenNeg;
  pres_WE=rPenPos;

  winStart_Buf = b;
  winEnd_Buf = (v4i) _mm_min_epi16 ((__m128i ) e, (__m128i ) LD_VEC4I(end));

  v4i wsi = winStart_Buf - path->window_start;
  v4i fli = flow_Buf - path->window_start;
  penNegV=nres_WE;
  for (int nuc = 0; nuc < 4; ++nuc) {

    // sum of squared residuals left of child window start
    resV[nuc]  = nres2_Buf[wsi[nuc]][nuc];
    metrV[nuc] = pres2_Buf[wsi[nuc]][nuc];

    // sum of squared residuals left of child->flow
    penParV[nuc] = pres2_Buf[fli[nuc]][nuc];

  }
  metrV+=path->res;
  resV += metrV;
  metrV += penNegV;
  penParV += penNegV;
  penNegV += penParV;
  distV = resV + nres_WE + pres_WE;
}

void TreephaserVEC::sumNormMeasures() {
  int i = num_flows_;
  float sum = 0.0f;
  rd_SqNormMeasureSum[i] = 0.0f;
  while(--i >= 0)
    rd_SqNormMeasureSum[i] = (sum += rd_NormMeasure[i]*rd_NormMeasure[i]);
}

// -------------------------------------------------

void TreephaserVEC::RecalibratePredictions(PathRecV *maxPathPtr)
{
  // Distort predictions according to recalibration model
  int to_flow = min(maxPathPtr->flow, num_flows_);

  for (int flow=0; flow<to_flow; flow++) {
    maxPathPtr->pred[flow] =
        maxPathPtr->pred[flow] * maxPathPtr->calib_A[flow]
          + maxPathPtr->calib_B[flow];
  }

}

void TreephaserVEC::ResetRecalibrationStructures(int num_flows) {
  for (int p = 0; p <= 8; ++p) {
    memset_F(&(sv_PathPtr[p]->calib_A[0]), 1.0f, num_flows_);
	memset_F(&(sv_PathPtr[p]->calib_B[0]), 0, num_flows_);
  }
}

// --------------------------------------------------

void TreephaserVEC::SolveRead(BasecallerRead& read, int begin_flow, int end_flow)
{
  memcpy_F(rd_NormMeasure, &(read.normalized_measurements[0]), num_flows_);
  memset_F(sv_PathPtr[BEST_PATH]->pred, 0, num_flows_); // Not necessary?
  memcpy_C(sv_PathPtr[BEST_PATH]->sequence, &(read.sequence[0]), (int)read.sequence.size());
  sv_PathPtr[BEST_PATH]->sequence_length = read.sequence.size();

  Solve(begin_flow, end_flow,end_flow,-1,max_metr_diff_ );

  int to_flow = min(sv_PathPtr[BEST_PATH]->window_end, end_flow);
  read.sequence.resize(sv_PathPtr[BEST_PATH]->sequence_length);
  memcpy_C(&(read.sequence[0]), sv_PathPtr[BEST_PATH]->sequence, sv_PathPtr[BEST_PATH]->sequence_length);
  memset_F(&(read.prediction[0]), 0, num_flows_);
  memcpy_F(&(read.prediction[0]), sv_PathPtr[BEST_PATH]->pred, to_flow);
}


// -------------------------------------------------
PathRecV *TreephaserVEC::sortPaths(int & pathCnt, int &parentPathIdx, int &badPaths, int numActivePaths)
{
#ifdef PRINT_DEBUG
    int commonSeq=0;
    if(DO_DEBUG){
    	printf("pathCnt=%d  \n",pathCnt);
        float minMetr=25.0f;
        int  minFlow=1000;
        for(int i = 0; i < pathCnt; ++i){
        	if(sv_PathPtr[i]->flow < minFlow)
        		minFlow=sv_PathPtr[i]->flow;
        }
        int diff=0;
        for(;commonSeq<sv_PathPtr[0]->sequence_length && !diff;commonSeq++){
            for(int i = 0; i < pathCnt; ++i){
            	if(sv_PathPtr[0]->sequence[commonSeq] != sv_PathPtr[i]->sequence[commonSeq]){
            		diff=1;
            	}
            }
        }
        if(commonSeq)
        	commonSeq--;

        for(int i = 0; i < pathCnt; ++i){
          sv_PathPtr[i]->sequence[sv_PathPtr[i]->sequence_length]=0; // null-terminate
          printf("%s  (%d %s %d/%d-%d) sig(%.3f/%.3f) %.3f/%.3f/%.3f\n",(minFlow==sv_PathPtr[i]->flow?"--":"  "),
        		  commonSeq,&sv_PathPtr[i]->sequence[commonSeq],
				  sv_PathPtr[i]->window_start,
				  sv_PathPtr[i]->flow,
    			  sv_PathPtr[i]->window_end,
    			  sv_PathPtr[i]->pred[sv_PathPtr[i]->flow],
    			  sv_PathPtr[i]->state[sv_PathPtr[i]->flow],
    			  sv_PathPtr[i]->penalty,sv_PathPtr[i]->flowMetr,sv_PathPtr[i]->metr);
        }
        printf("\n");
    }
#endif

    int maxPaths = MAX_PATHS-4;
    if(numActivePaths > 0)
      maxPaths = min(maxPaths,numActivePaths);
   if(pathCnt > (maxPaths-1)) {
	  int m = sv_PathPtr[0]->flow;
	  int i = 1;
	  do {
		int n = sv_PathPtr[i]->flow;
		if(m < n)
		  m = n;
	  } while(++i < pathCnt);
	  if((m -= MAX_PATH_DELAY) > 0) {
		do {
		  if(sv_PathPtr[--i]->flow < m){
#ifdef PRINT_DEBUG
			sv_PathPtr[pathCnt]->sequence[sv_PathPtr[pathCnt]->sequence_length]=0; // null-terminate
			printf("Removing path %s too far behind\n",&sv_PathPtr[pathCnt]->sequence[commonSeq]);
#endif
			swap(sv_PathPtr[i], sv_PathPtr[--pathCnt]);
		  }
		} while(i > 0);
	  }
	}

	while(pathCnt > maxPaths) {
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
#ifdef PRINT_DEBUG
      if(DO_DEBUG){
        sv_PathPtr[j]->sequence[sv_PathPtr[j]->sequence_length]=0; // null-terminate
        printf("%s Removing path %s too many paths\n",(j>= 4?"****":"    "),&sv_PathPtr[j]->sequence[commonSeq]);
      }
#endif
	  swap(sv_PathPtr[j], sv_PathPtr[--pathCnt]);
	  badPaths++;
	}
    PathRecV *parent = sv_PathPtr[0];
    for(int i = 1; i < pathCnt; ++i){
      if(parent->metr > sv_PathPtr[i]->metr) {
        parent = sv_PathPtr[i];
        parentPathIdx = i;
      }
    }
    return (parent);
}


float TreephaserVEC::computeParentDist(PathRecV RESTRICT_PTR parent, int end_flow)
{
   // Right here we are having a memory overrun: We copied up to parent->flow but use until parent->window_end of calibA and calibB
   // Computing squared distance between parent's predicted signal and normalized measurements
   float dist = parent->res+(rd_SqNormMeasureSum[parent->window_end]-rd_SqNormMeasureSum[end_flow]);
   dist += sumOfSquaredDiffsFloatSSE((float*)(&(rd_NormMeasureAdj[parent->window_start])),
                                     (float*)(&(parent->pred[parent->window_start])),
                                      parent->window_end-parent->window_start);
   return dist;
}

void TreephaserVEC::CopyPath (PathRecV RESTRICT_PTR dest, PathRecV RESTRICT_PTR parent,
			 PathRecV RESTRICT_PTR child, int saveFlow,
			 int &numSaved, int cached_flow, int cached_flow_ws, int cached_flow_seq)
{
  int pws=parent->window_start;
  int cws=child->window_start;
  int cwe=child->window_end;
  int nuc=child->nuc;
  int pf=parent->flow;
  int cf=child->flow;
  int psl=parent->sequence_length;

  memcpy_4F(&dest->pred[pws],&parent->pred_Buf[pws][nuc],cws - pws+1);

  if (state_inphase_enabled_) {
    //extending from parent->state_inphase[pf] to fill the gap
    for (int tempInd = pf + 1; tempInd < cf; tempInd++) {
      dest->state_inphase[tempInd] = max ((float)dest->state[cf][nuc], 0.01f);
    }
    dest->state_inphase[cf] = max ((float)dest->state[cf][nuc], 0.01f);
  }

  dest->sequence_length = psl + 1;
  dest->sequence[psl] = flow_order_[cf];
  dest->sequence[dest->sequence_length]=0;

  if (psl and dest->sequence[psl] != dest->sequence[psl - 1])
    dest->last_hp = 0;
  else
    dest->last_hp = parent->last_hp;
  dest->last_hp++;

  dest->flowMetr = (cf == 0)?0:((child->metr + 0.5f * child->flowMetr) / cf);

  if (recalibrate_predictions_) {
    //explicitly fill zeros between pf and cf;
    for (int tempInd = pf + 1; tempInd < cf; tempInd++) {
      dest->calib_A[tempInd] = 1.0f;
      dest->calib_B[tempInd] = 0.0f;
    }
    int hp_length = min (dest->last_hp, MAX_HPXLEN);
    dest->calib_A[cf] = (*As_).at (cf).at (
	flow_order_.int_at (cf)).at (hp_length);
    dest->calib_B[cf] = (*Bs_).at (cf).at (
	flow_order_.int_at (cf)).at (hp_length);
    rd_NormMeasureAdj[cf] = (rd_NormMeasure[cf]-dest->calib_B[cf])/dest->calib_A[cf];
  }
  {
    dest->flow = child->flow;
    dest->window_start = child->window_start;
    dest->window_end = child->window_end;
    dest->res = child->res;
    dest->metr = child->metr;
    dest->newSignal = child->newSignal;
    dest->dist = child->dist;
    dest->nuc = child->nuc;
  }

  dest->cached_flow = min(dest->flow,cached_flow);
  dest->cached_flow_ws = min(dest->window_start,cached_flow_ws);
  dest->cached_flow_seq = min(dest->sequence_length,cached_flow_seq);


#ifdef PRINT_DEBUG
    if(DO_DEBUG){
      child->sequence[child->sequence_length]=0; // null-terminate
      if (singleChild != 1)
	    printf("adding child %s\n",child->sequence);
    }
#endif

}

void TreephaserVEC::CopyPathNew (PathRecV RESTRICT_PTR dest, PathRecV RESTRICT_PTR parent,
			 PathRecV RESTRICT_PTR child, int saveFlow,
			 int &numSaved, int cached_flow, int cached_flow_ws, int cached_flow_seq)
{
  int dcf = (dest->cached_flow&0xffff);
  int dws = (dest->cached_flow_ws&0xffff);
  int dseq= (dest->cached_flow_seq&0xffff);

  int pws=parent->window_start;
  int cws=child->window_start;
  int cwe=child->window_end;
  int nuc=child->nuc;
  int pf=parent->flow;
  int cf=child->flow;
  int psl=parent->sequence_length;
  int cwemcws=cwe-cws;
  int pfmdcf=pf-dcf+1;

  memcpy_4F(&dest->pred[pws],&parent->pred_Buf[pws][nuc],cws - pws+1);

  if (dest != parent)
  {
    // copy the beginning of the arrays as well
    memcpy_4F4(&dest->state[cws][nuc],&parent->state[cws][nuc],cwemcws);
    memcpy_F(&dest->pred[dws], &parent->pred[dws], pws - dws);
    memcpy_4F4(&dest->pred_Buf[cws][nuc],&parent->pred_Buf[cws][nuc],cwemcws);
    memcpy_C(&dest->sequence[dseq], &parent->sequence[dseq], psl - dseq);
  }

  if (state_inphase_enabled_) {
    if (dest != parent)
    {
      memcpy_F(&dest->state_inphase[dcf], &parent->state_inphase[dcf],pfmdcf);
    }

    //extending from parent->state_inphase[pf] to fill the gap
    for (int tempInd = pf + 1; tempInd < cf; tempInd++) {
      dest->state_inphase[tempInd] = max ((float)dest->state[cf][nuc], 0.01f);
    }
    dest->state_inphase[cf] = max ((float)dest->state[cf][nuc], 0.01f);
  }

  dest->sequence_length = psl + 1;
  dest->sequence[psl] = flow_order_[cf];
  dest->sequence[dest->sequence_length]=0;
  if (psl and dest->sequence[psl] != dest->sequence[psl - 1])
    dest->last_hp = 0;
  else
    dest->last_hp = parent->last_hp;
  dest->last_hp++;

  dest->flowMetr = (cf == 0)?0:((child->metr + 0.5f * child->flowMetr) / cf);

  if (recalibrate_predictions_) {
    if (dest != parent && cf > 0)
    {
      memcpy_F(&dest->calib_A[dcf], &parent->calib_A[dcf],pfmdcf);
      memcpy_F(&dest->calib_B[dcf], &parent->calib_B[dcf],pfmdcf);
    }
    //explicitly fill zeros between pf and cf;
    for (int tempInd = pf + 1; tempInd < cf; tempInd++) {
      dest->calib_A[tempInd] = 1.0f;
      dest->calib_B[tempInd] = 0.0f;
    }
    int hp_length = min (dest->last_hp, MAX_HPXLEN);
    dest->calib_A[cf] = (*As_).at (cf).at (
	flow_order_.int_at (cf)).at (hp_length);
    dest->calib_B[cf] = (*Bs_).at (cf).at (
	flow_order_.int_at (cf)).at (hp_length);
    rd_NormMeasureAdj[cf] = (rd_NormMeasure[cf]-dest->calib_B[cf])/dest->calib_A[cf];
  }
  if(dest != child)
  {
    dest->flow = child->flow;
    dest->window_start = child->window_start;
    dest->window_end = child->window_end;
    dest->res = child->res;
    dest->metr = child->metr;
    dest->newSignal = child->newSignal;
    dest->dist = child->dist;
    dest->nuc = child->nuc;
  }

  dest->saved = parent->saved;
  dest->cached_flow = min(dest->flow,cached_flow);
  dest->cached_flow_ws = min(dest->window_start,cached_flow_ws);
  dest->cached_flow_seq = min(dest->sequence_length,cached_flow_seq);

#ifdef PRINT_DEBUG
    if(DO_DEBUG){
      child->sequence[child->sequence_length]=0; // null-terminate
      if (singleChild != 1)
	    printf("adding child %s\n",child->sequence);
    }
#endif

}

void
TreephaserVEC::CopyPathDeep (PathRecV RESTRICT_PTR destPath,
			     PathRecV RESTRICT_PTR srcPath)
{
  memcpy_4F4(&destPath->state[srcPath->window_start][srcPath->nuc],
	     &srcPath->state[srcPath->window_start][srcPath->nuc],
	     srcPath->window_end -srcPath->window_start);
  memcpy_4F4(&destPath->pred_Buf[srcPath->window_start][srcPath->nuc],
	     &srcPath->pred_Buf[srcPath->window_start][srcPath->nuc],
	     srcPath->window_end - srcPath->window_start + 1);
  memcpy_F(destPath->pred,srcPath->pred,srcPath->window_end + 1);
  memcpy_4F(&destPath->pred[srcPath->window_start],
	    &destPath->pred_Buf[srcPath->window_start][srcPath->nuc],
	    srcPath->window_end - srcPath->window_start);
  memcpy_C(destPath->sequence,srcPath->sequence,srcPath->sequence_length + 1);
  if (state_inphase_enabled_) {
    memcpy_F(destPath->state_inphase,srcPath->state_inphase,srcPath->flow + 1);
  }

  destPath->sequence_length = srcPath->sequence_length;

  if (recalibrate_predictions_) {
    memcpy_F(destPath->calib_A,srcPath->calib_A,srcPath->flow + 1);
    memcpy_F(destPath->calib_B,srcPath->calib_B,srcPath->flow + 1);
  }

  destPath->flow = srcPath->flow;
  destPath->window_start = srcPath->window_start;
  destPath->window_end = srcPath->window_end;
  destPath->res = srcPath->res;
  destPath->metr = srcPath->metr;
  destPath->newSignal = srcPath->newSignal;
  destPath->dist = srcPath->dist;
  destPath->flowMetr = srcPath->flowMetr;
  destPath->nuc = srcPath->nuc;
  destPath->penalty = srcPath->penalty;
  destPath->last_hp = srcPath->last_hp;
  destPath->cached_flow=0;
  destPath->cached_flow_seq=0;
  destPath->cached_flow_ws=0;
}



// -------------------------------------------------
bool TreephaserVEC::Solve(int begin_flow, int end_flow, int saveFlow, int numActivePaths, float maxMetrDiff)
{
  sumNormMeasures();

  PathRecV RESTRICT_PTR parent = sv_PathPtr[0];
  PathRecV RESTRICT_PTR best = sv_PathPtr[BEST_PATH];


  for(int flow=begin_flow;flow<end_flow;flow++){
    rd_NormMeasureAdj[flow]=rd_NormMeasure[flow];
  }

  int cached_flow=DFLT_CACHE;
  int cached_flow_ws=DFLT_CACHE;
  int cached_flow_seq=DFLT_CACHE;

  int pathCnt = 1;
  int numSaved=0;
  float bestDist = 1e20;
  end_flow = min(end_flow, num_flows_);

  // Simulating beginning of the read  up to or one base past begin_flow
  if(begin_flow > 0) {
    int found=0;

    if(sv_PathPtr[SAVED_PATHS]->flow > 0){
	int pth=0;
	for(;pth<NUM_SAVED_PATHS;pth++){
	    if(sv_PathPtr[SAVED_PATHS+pth]->flow > 0 &&
		memcmp(sv_PathPtr[SAVED_PATHS+pth]->sequence,
		      sv_PathPtr[BEST_PATH]->sequence,
		      sv_PathPtr[SAVED_PATHS+pth]->sequence_length)==0){

	      best=sv_PathPtr[SAVED_PATHS+pth];
	      sv_PathPtr[SAVED_PATHS+pth]=parent;
	      sv_PathPtr[0]=best;
	      parent=best;
	      found=1;
	      break;
	    }
	}
    }

    if(!found) return true;

    parent->res = sumOfSquaredDiffsFloatSSE(
        (float*)rd_NormMeasure, (float*)parent->pred, parent->window_start);
  }else{
    for(int i=0;i<MAX_ACT_PATHS;i++){
      sv_PathPtr[i]->calib_A[0]=1.0;
      sv_PathPtr[i]->calib_B[0]=0;

      memset(sv_PathPtr[i]->pred,0,num_flows_*sizeof(sv_PathPtr[i]->pred[0]));
    }
    parent->flow = 0;
    parent->window_start = 0;
    parent->window_end = 1;
    parent->res = 0.0f;
    parent->metr = 0.0f;
    parent->flowMetr = 0.0f;
    parent->newSignal = 1.0f;
    parent->state[0] = LD_VEC4F(1.0f);
    parent->sequence_length = 0;
    parent->last_hp = 0;
    parent->pred[0] = 0.0f;
    parent->pred_Buf[0]=LD_VEC4F(0);
    parent->state_inphase[0] = 1.0f;
    parent->nuc=0;

  }

  for(int i=1;i<MAX_ACT_PATHS;i++){
    sv_PathPtr[i]->flow=0; // mark as invalid for restart
    //sv_PathPtr[i]->window_end = 1;
    sv_PathPtr[i]->sequence_length = 0;
    sv_PathPtr[i]->cached_flow=DFLT_CACHE;
    sv_PathPtr[i]->cached_flow_ws=DFLT_CACHE;
    sv_PathPtr[i]->cached_flow_seq=DFLT_CACHE;
  }
  sv_PathPtr[0]->cached_flow=DFLT_CACHE;
  sv_PathPtr[0]->cached_flow_ws=DFLT_CACHE;
  sv_PathPtr[0]->cached_flow_seq=DFLT_CACHE;


  parent->dist=computeParentDist(parent,end_flow);
  parent->saved=0;
  int badPaths=0;
  int manyPaths=0;

  do {
    int parentPathIdx=0;
    int childPathIdx=-1;
    float bestpen = 19.8;
    parent=sv_PathPtr[0];
    if(pathCnt>1)
      parent = sortPaths(pathCnt,parentPathIdx,badPaths,numActivePaths);
    else
      badPaths=0;

    int pf=parent->flow;
    int pws=parent->window_start;
    int pseq=parent->sequence_length;
    int numChild=0;
   if(!(parent->last_hp >= MAX_HPXLEN or parent->sequence_length >= 2*MAX_VALS-10)){
     advanceState4(parent, end_flow);

      for(int nuc = 0; nuc < 4; ++nuc) {
        bestpen = min(bestpen,(float)penNegV[nuc]);
      }

      int numChild=0;
      for(int nuc = 0; nuc < 4; ++nuc) {
	if(flow_Buf[nuc] >= end_flow)
	      continue;
	if(penNegV[nuc]-bestpen >= maxMetrDiff)
	  continue;

	float newSignal=rd_NormMeasureAdj[flow_Buf[nuc]] / parent->state[flow_Buf[nuc]][nuc];
	if(newSignal < 0.3f && parent->newSignal < 0.3f)
	  continue;
	distV[nuc] += rd_SqNormMeasureSum[winEnd_Buf[nuc]]-rd_SqNormMeasureSum[end_flow];

	// child path survives at this point
	PathRecV RESTRICT_PTR child = sv_PathPtr[pathCnt];
	child->flow = flow_Buf[nuc];
	child->window_start = winStart_Buf[nuc];
	child->window_end = winEnd_Buf[nuc];
	child->res = resV[nuc];
	child->newSignal = newSignal;
	child->metr = metrV[nuc];
	child->flowMetr = penParV[nuc];
	child->penalty = penNegV[nuc];
	child->nuc = nuc;
	child->dist = distV[nuc];
	numChild++;
	if((childPathIdx < 0) && (bestDist < parent->dist || (parent->dist >= child->dist))){
          sv_PathPtr[pathCnt] = sv_PathPtr[MAX_PATHS-1];
          sv_PathPtr[MAX_PATHS-1] = child;
          childPathIdx = MAX_PATHS-1;
          if(bestDist >= child->dist){
            best=parent;
            bestDist=child->dist;
          }
        }else{
          CopyPathNew(child,parent,child,saveFlow,numSaved,cached_flow,cached_flow_ws,cached_flow_seq);
          ++pathCnt;
        }
      }
   }

   if(childPathIdx >= 0) {
     CopyPath(parent,parent,sv_PathPtr[childPathIdx],saveFlow,numSaved,cached_flow,cached_flow_ws,cached_flow_seq);
   }else{
     if(bestDist >= parent->dist){
       bestDist = parent->dist;
       sv_PathPtr[parentPathIdx] = sv_PathPtr[--pathCnt];
       sv_PathPtr[pathCnt] = sv_PathPtr[BEST_PATH];
       sv_PathPtr[BEST_PATH] = parent;
       best=parent;
     }else{
       sv_PathPtr[parentPathIdx] = sv_PathPtr[--pathCnt];
       sv_PathPtr[pathCnt] = parent;
     }
   }
     if(pathCnt > 1){
	if(cached_flow==DFLT_CACHE){
         cached_flow=0;//pf;
         cached_flow_ws=0;//pws;
         cached_flow_seq=0;//pseq;
         for(int pthi=0;pthi<=BEST_PATH;pthi++){
   	   PathRecV RESTRICT_PTR pth = sv_PathPtr[pthi];
           pth->cached_flow=min(pth->cached_flow,cached_flow);
           pth->cached_flow_ws=min(pth->cached_flow_ws,cached_flow_ws);
           pth->cached_flow_seq=min(pth->cached_flow_seq,cached_flow_seq);
         }
       }
     }else if(cached_flow != DFLT_CACHE){
       cached_flow=DFLT_CACHE;
       cached_flow_ws=DFLT_CACHE;
       cached_flow_seq=DFLT_CACHE;
     }

     if(saveFlow > 0){
       if(best->flow >= saveFlow && (pathCnt == 1 || best->flow >= end_flow)){

	 if(numSaved == 0){
	   // save all the current paths
	   if(sv_PathPtr[BEST_PATH] != best){
	     CopyPathDeep(sv_PathPtr[BEST_PATH],best);
	     best=sv_PathPtr[BEST_PATH];
	   }
	   for(int i=0;i<pathCnt && i < NUM_SAVED_PATHS;i++){
	     swap(sv_PathPtr[SAVED_PATHS+i],sv_PathPtr[i]);
	     numSaved++;
	   }
	 }
	 break;
       }
       if(numSaved == 0 && best->flow >= (saveFlow-8) && pathCnt != 1){
	   for(int i=0;i<pathCnt && i < NUM_SAVED_PATHS;i++){
	     CopyPathDeep(sv_PathPtr[SAVED_PATHS+i],sv_PathPtr[i]);
	     numSaved++;
	   }
       }
     }

     if(numChild> 2)
       manyPaths++;
  } while(pathCnt > 0 && badPaths < bad_path_limit_ && manyPaths < many_path_limit_);

  for(int i=0;i<4;i++){
    if(best == sv_PathPtr[i]){
      sv_PathPtr[i]=sv_PathPtr[BEST_PATH];
      sv_PathPtr[BEST_PATH]=best;
    }
  }
  best=sv_PathPtr[BEST_PATH];
  memcpy_4F(&best->pred[best->window_start],
	    &best->pred_Buf[best->window_start][best->nuc],
	    best->window_end - best->window_start);

  // At the end change predictions according to recalibration model and reset data structures
  if (recalibrate_predictions_) {
    RecalibratePredictions(sv_PathPtr[BEST_PATH]);
    ResetRecalibrationStructures(end_flow);
  }
  if(saveFlow > 0 && numSaved==0)
    return true;
  else
    return false;
}


void TreephaserVEC::WindowedNormalize(BasecallerRead& read, int num_steps)
{
//  int num_flows = read.raw_measurements.size();
  float median_set[windowSize_];

  // Estimate and correct for additive offset

  float next_normalizer = 0;
  int estim_flow = 0;
  int apply_flow = 0;
  PathRecV *path=sv_PathPtr[BEST_PATH];
  int start_step=max(0,num_steps-1);

  for (int step = start_step; step <= num_steps; ++step) {

    int window_start = (step+0) * windowSize_;
    int window_end   = (step+1) * windowSize_;
    int apply_flow_start = max(0,window_start-(windowSize_/2));
    int apply_flow_end   = min(num_flows_,window_start+(windowSize_/2));

    float normalizer = next_normalizer;

    int median_set_size = 0;
    float average=0;
    for (estim_flow=window_start; estim_flow < window_end ; ++estim_flow){
      if (path->pred[estim_flow] < 0.3){
        average += read.raw_measurements[estim_flow] - path->pred[estim_flow];
        median_set_size++;
      }
    }

    if (median_set_size > 5) {
      next_normalizer = average/(float)median_set_size;
      if (step == 0)
        normalizer = next_normalizer;
      else
	normalizer = read.additive_correction[apply_flow_start-1];
    }

    float delta = (next_normalizer - normalizer) / (float)windowSize_;

    for (apply_flow=apply_flow_start; apply_flow < apply_flow_end; ++apply_flow) {
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

  for (int step = start_step; step <= num_steps; ++step) {

    int window_start = (step+0) * windowSize_;
    int window_end   = (step+1) * windowSize_;
    int apply_flow_start = max(0,window_start-(windowSize_/2));
    int apply_flow_end   = min(num_flows_,window_start+(windowSize_/2));

    float normalizer = next_normalizer;

    int median_set_size = 0;
    float average=0;
    for (estim_flow=window_start; estim_flow < window_end ; ++estim_flow){
      if (path->pred[estim_flow] > 0.5 and rd_NormMeasure[estim_flow] > 0){
        median_set[median_set_size++] = rd_NormMeasure[estim_flow] / path->pred[estim_flow];
        average+=rd_NormMeasure[estim_flow] / path->pred[estim_flow];
      }
    }

    if (median_set_size > 5) {
      average /= (float)median_set_size;
      std::nth_element(median_set, median_set + median_set_size/2, median_set + median_set_size);
      next_normalizer = (median_set[median_set_size / 2] + average)/2;
      if (step == 0)
        normalizer = next_normalizer;
      else
	normalizer = read.multiplicative_correction[apply_flow_start-1];
    }

    float delta = (next_normalizer - normalizer) / static_cast<float>(windowSize_);

    for (apply_flow=apply_flow_start; apply_flow < apply_flow_end; ++apply_flow) {
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


void  TreephaserVEC::ComputeQVmetrics_flow(BasecallerRead& read, vector<int>& flow_to_base, const bool flow_predictors_,const bool flow_quality)
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

  PathRecV RESTRICT_PTR parent = sv_PathPtr[0];
  parent->flow = 0;
  parent->window_start = 0;
  parent->window_end = 1;
  parent->res = 0.0f;
  parent->metr = 0.0f;
  parent->flowMetr = 0.0f;
  parent->newSignal = 1.0f;
  parent->state[0] = LD_VEC4F(1.0f);
  parent->sequence_length = 0;
  parent->last_hp = 0;
  parent->pred[0] = 0.0f;
  parent->pred_Buf[0]=LD_VEC4F(0);

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
      int called_nuc = -1;
      for(int nuc = 0; nuc < 4; ++nuc) {
        if (nuc_int_to_char[nuc] == flow_order_[solution_flow])
          called_nuc = nuc;
      }

      for(int i = parent->window_start, e = winStart_Buf[called_nuc]; i < e; ++i) {
        parent->pred[i] = parent->pred_Buf[i][called_nuc];
      }

      if (flow_Buf[called_nuc] == parent->flow)
        parent->last_hp = parent->last_hp + 1;
      else
        parent->last_hp = 1;

      recent_state_inphase = parent->state[solution_flow][called_nuc];

      // Get delta penalty to next best solution
      read.penalty_mismatch[base] = -1; // min delta penalty to earlier base hypothesis
      read.penalty_residual[base] = 0;

      if (solution_flow - parent->window_start > 0)
        read.penalty_residual[base] = penParV[called_nuc] / (solution_flow - parent->window_start);

      for (int nuc = 0; nuc < 4; ++nuc) {
        if (nuc == called_nuc)
            continue;
        float penalty_mismatch = penParV[called_nuc] - penParV[nuc];
        read.penalty_mismatch[base] = max(read.penalty_mismatch[base], penalty_mismatch);
      }

      parent->flow = min((int)(flow_Buf[called_nuc]), flow_order_.num_flows());
      parent->window_end = min((int)(winEnd_Buf[called_nuc]), flow_order_.num_flows());
      parent->window_start = min((int)(winStart_Buf[called_nuc]), parent->window_end);
      parent->nuc=called_nuc;
    }
    read.state_inphase[solution_flow] = max(recent_state_inphase, 0.01f);
  }
  for(int i = parent->window_start, e = parent->window_end; i < e; ++i) {
    parent->pred[i] = parent->pred_Buf[i][parent->nuc];
  }

  if (flow_predictors_ || flow_quality) { //if (flow_predictors_)
      //vector<int> flows_to_proc;
      for (int solution_flow = 0; solution_flow < num_flows; ++solution_flow) {
          int curr_base = flow_to_base[solution_flow];
          if (curr_base >= 0) {
              // copy from what's stored in read.penalty_mismatch[base]
              read.penalty_mismatch_flow[solution_flow] = read.penalty_mismatch[curr_base];
              read.penalty_residual_flow[solution_flow] = read.penalty_residual[curr_base];
	  }
      }
  }

  if(recalibrate_predictions_) {
    RecalibratePredictions(parent);
  }
  memset_F(&read.prediction[parent->window_end], 0, num_flows_-parent->window_end);
  memcpy_F(&read.prediction[0], &parent->pred[0], parent->window_end);
}



