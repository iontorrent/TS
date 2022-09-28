/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef TREEPHASERVEC_H
#define TREEPHASERVEC_H

#include "DPTreephaser.h"

#include <algorithm>
#include <vector>
#include "BaseCallerUtils.h"
#include "Vecs.h"

#ifdef _WIN32
  #define ALIGN(AAlignSize) __declspec(align(AAlignSize))
  #define ALWAYS_INLINE __forceinline
  #define EXPECTED(ABoolExpression) ABoolExpression
  #define UNEXPECTED(ABoolExpression) ABoolExpression
  #define RESTRICT_PTR * __restrict
#else
  #define ALIGN(AAlignSize) __attribute__((aligned(AAlignSize)))
  #define ALWAYS_INLINE inline __attribute__((always_inline))
  #define NEVER_INLINE inline __attribute__((noinline))
  #define EXPECTED(ABoolExpression) __builtin_expect((ABoolExpression), 1)
  #define UNEXPECTED(ABoolExpression) __builtin_expect((ABoolExpression), 0)
  #define RESTRICT_PTR * __restrict__
/*
  #ifndef _mm_castps_si128
    ALWAYS_INLINE __m128i _mm_castps_si128(__m128 __A) {
      return (__m128i) __A;
    }
  #endif

  #ifndef _mm_castsi128_ps
    ALWAYS_INLINE __m128 _mm_castsi128_ps(__m128i __A) {
      return (__m128) __A;
    }
  #endif
*/
#endif


//#define MAX_VALS 1020
#define MAX_VALS 2044
//#define STEP_SIZE 38
//#define MAX_STEPS 24
// MAX_STEPS set large enough to handle minimum window size
#define MAX_STEPS (1+(MAX_VALS/DPTreephaser::kMinWindowSize_))
#define MAX_PATHS 8
#define NUM_SAVED_PATHS 4
#define NUM_BEST_PATH 1
#define MAX_ACT_PATHS (MAX_PATHS+NUM_BEST_PATH+NUM_SAVED_PATHS)
#define BEST_PATH MAX_PATHS
#define SAVED_PATHS (MAX_PATHS+NUM_BEST_PATH)

#define MAX_PATH_DELAY 40

//#pragma pack(push, 1)
struct __attribute__((packed)) ALIGN(64) PathRecV {
  int flow;
  int window_start;
  int window_end;
  float res;
  float metr;
  float flowMetr;
  float newSignal;
  float penalty;
  int last_hp;
  int nuc;
  float dist;

  // cache entries
  int saved;
  int cached_flow;
  int cached_flow_ws;
  int cached_flow_seq;

  // per-flow metrics
  v4f state[MAX_VALS] ALIGN(64);
  float pred[MAX_VALS]  ALIGN(64);
  v4f pred_Buf[MAX_VALS]  ALIGN(64);
  float calib_A[MAX_VALS] ALIGN(64);
  float calib_B[MAX_VALS] ALIGN(64);
  float state_inphase[MAX_VALS] ALIGN(64);

  // per-basse metrics
  int  sequence_length;
  char sequence[2*MAX_VALS + 12]; // +12 makes the entire struct align well
};
//#pragma pack(pop)


class TreephaserVEC {
public:

  //! @brief      Default constructor
  TreephaserVEC();

  //! @brief      Constructor
  //! @param[in]  flow_order  Flow order object
  //! @param[in]  windowSize  Size of the normalization window to use.
  TreephaserVEC(const ion::FlowOrder& flow_order, const int windowSize);

  //! @brief      Set flow order and initialize internal variables
  //! @param[in]  flow_order  Flow order object
  void SetFlowOrder(const ion::FlowOrder& flow_order);

  void SetBadPathLimit(int bad_path_limit){bad_path_limit_=bad_path_limit;}

  void SetManyPathLimit(int many_path_limit){many_path_limit_=many_path_limit;}

  void SetInitalPaths(int initial_paths){initial_paths_=initial_paths;}

  void SetMaxMetrDiff(float max_metr_diff){max_metr_diff_=max_metr_diff;}

  //! @brief      Set the normalization window size
  //! @param[in]  windowSize  Size of the normalization window to use.
  inline void SetNormalizationWindowSize(const int windowSize) { windowSize_ = max(DPTreephaser::kMinWindowSize_, min(windowSize, DPTreephaser::kMaxWindowSize_));}

  //! @brief     Set phasing model parameters
  void SetModelParameters(double cf, double ie);

  //! @brief      Interface function similar to DPTreephaser Solve
  //! @param[out] read        Basecaller read to be solved
  //! @param[in]  begin_flow  Start solving at this flow
  //! @param[in]  end_flow    Do not solve for any flows past this one
  void SolveRead(BasecallerRead& read, int begin_flow, int end_flow);

  //! @brief      Iterative solving and normalization routine
  void NormalizeAndSolve(BasecallerRead& read);
//  PathRecV* parent;
//  int best;

  //! @brief      Set pointers to recalibration model
  bool SetAsBs(const vector<vector< vector<float> > > *As, const vector<vector< vector<float> > > *Bs){
    As_ = As;
    Bs_ = Bs;
    pm_model_available_ = (As_ != NULL) and (Bs_ != NULL);
    recalibrate_predictions_ = pm_model_available_; // We bothered loading the model, of course we want to use it!
    return pm_model_available_;
  };

  //! @brief     Enables the use of recalibration if a model is available.
  bool EnableRecalibration() {
    recalibrate_predictions_ = pm_model_available_;
    return pm_model_available_;
  };

  //! @brief     Disables the use of recalibration until a new model is set.
  void DisableRecalibration() {
    //pm_model_available_ = false;
    recalibrate_predictions_ = false;
  };

  //! @brief  Switch to disable / enable the use of recalibration during the normalization phase
  void SkipRecalDuringNormalization(bool skip_recal)
    { skip_recal_during_normalization_ = skip_recal; };

  //! @brief  Perform a more advanced simulation to generate QV predictors
  void  ComputeQVmetrics(BasecallerRead& read);
  void  ComputeQVmetrics_flow(BasecallerRead& read, vector<int>& flow_to_base, const bool flow_predictors_=false, const bool flow_quality = false);

  int nuc_char_to_int(char nuc) {if (nuc=='A') return 0; if (nuc=='C') return 1; if (nuc=='G') return 2; if (nuc=='T') return 3; return -1;}

protected:

  inline void CalcResidualI(PathRecV RESTRICT_PTR parent,
  		int flow, int j, v4f &rS, v4f rTemp1s_, v4f &rPenNeg, v4f &rPenPos);

  inline v4i CheckWindowStartBase(v4i rBeg, v4i rParNuc, int flow, v4f rS);
  inline void CheckWindowStart1(v4i & rBeg, v4i rParNuc, int flow, v4f rS);
  inline void CheckWindowStart2(v4i & rBeg, v4i rParNuc, int flow, v4f rS, v4i rEnd);
  inline void CheckWindowEnd(v4i rBeg, v4i rParNuc, int flow, v4f & rS, v4i & rEnd, v4f rAlive, v4i rFlowEnd);
  inline void advanceI( float parentState, int j, v4i rParNuc, v4f ts_tr, v4f & rAlive, v4f & rS);
  inline void advanceE(int j, v4i rParNuc, v4f ts_tr, v4f & rAlive, v4f & rS);
  void catchup (PathRecV RESTRICT_PTR parent,PathRecV RESTRICT_PTR best, int begin_flow, int end_flow);

  //! @brief     Solving a read
  bool  Solve(int begin_flow, int end_flow, int save_flow, int numActivePaths, float maxMetrDiff);
  //! @brief     Normalizing a read
  PathRecV *sortPaths(int & pathCnt, int &parentPathIdx, int & badPaths, int numAcitvePaths);
  float computeParentDist(PathRecV RESTRICT_PTR parent, int end_flow);
  void CopyPath(PathRecV RESTRICT_PTR dest, PathRecV RESTRICT_PTR parent,
		PathRecV RESTRICT_PTR child, int saveFlow, int &numSaved,
		int cached_flow, int cached_flow_ws, int cached_flow_seq);
  void CopyPathNew(PathRecV RESTRICT_PTR dest, PathRecV RESTRICT_PTR parent,
		PathRecV RESTRICT_PTR child, int saveFlow, int &numSaved,
		int cached_flow, int cached_flow_ws, int cached_flow_seq);
  void CopyPathDeep(PathRecV RESTRICT_PTR dest, PathRecV RESTRICT_PTR src);
  void  WindowedNormalize(BasecallerRead& read, int step);
  //! @brief     Make recalibration changes to predictions explicitly visible
  void  RecalibratePredictions(PathRecV *maxPathPtr);
  //! @brief     Resetting recalibration data structure
  void  ResetRecalibrationStructures(int num_flows);
  //! @brief      Initialize floating point array variables with a value
  void InitializeVariables();

  void  sumNormMeasures();
  void  advanceState4(PathRecV RESTRICT_PTR parent, int end);
  void  advanceStateInPlace(PathRecV RESTRICT_PTR path, int nuc, int end, int end_flow);
  inline float CalcNewSignal(int nuc, PathRecV RESTRICT_PTR parent);

  // There was a small penalty in making these arrays class members, as opposed to static variables
  ALIGN(64) v4i ts_NextNuc[MAX_VALS];
  ALIGN(64) v4f ts_Transition[MAX_VALS];

  ALIGN(64) float rd_NormMeasure[MAX_VALS];
  ALIGN(64) float rd_NormMeasureAdj[MAX_VALS];

  ALIGN(64) float rd_SqNormMeasureSum[MAX_VALS];

  ALIGN(64) PathRecV sv_pathBuf[MAX_ACT_PATHS];

  ALIGN(64) float ft_stepNorms[MAX_STEPS];

  ALIGN(16) v4i ad_FlowEnd;
  ALIGN(16) v4i flow_Buf;
  ALIGN(16) v4i winEnd_Buf;
  ALIGN(16) v4i winStart_Buf;
  ALIGN(64)v4f nres_WE;
  ALIGN(64)v4f pres_WE;
  ALIGN(64)v4f metrV={0,0,0,0};
  ALIGN(64)v4f penParV={0,0,0,0};
  ALIGN(64)v4f penNegV={0,0,0,0};
  ALIGN(64)v4f resV={0,0,0,0};
  ALIGN(64)v4f distV={0,0,0,0};

  ion::FlowOrder      flow_order_;                //!< Sequence of nucleotide flows

  PathRecV *sv_PathPtr[MAX_ACT_PATHS];
  int ad_Adv;
  int num_flows_;
  int ts_StepCnt;
  int ts_StepBeg[MAX_STEPS+1];
  int ts_StepEnd[MAX_STEPS+1];

  
  int      windowSize_;                         //!< Adaptive normalization window size
  double   my_cf_;                              //!< Stores the cf phasing parameter used to compute transitions
  double   my_ie_;                              //!< Stores the ie phasing parameter used to compute transitions
  const vector< vector< vector<float> > > *As_; //!< Pointer to recalibration structure: multiplicative constant
  const vector< vector< vector<float> > > *Bs_; //!< Pointer to recalibration structure: additive constant
  bool     pm_model_available_;                 //!< Signals availability of a recalibration model
  bool     recalibrate_predictions_;            //!< Switch to use recalibration model during metric generation
  bool     skip_recal_during_normalization_;    //!< Switch to skip recalibration during the normalization phase
  bool     state_inphase_enabled_;              //!< Switch to save inphase population of molecules
  int      bad_path_limit_;                      //!< number of dead end paths before aborting
  int      many_path_limit_;                     //!< number of >3 path decisions before aborting
  int      initial_paths_;                       //!< number of initial max paths to use in solve
  float    max_metr_diff_;                       //!< maximum metric difference to create a branch

};

#endif // TREEPHASERVEC_H
