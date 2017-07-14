/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef TREEPHASERSSE_H
#define TREEPHASERSSE_H

#include "DPTreephaser.h"

#include <algorithm>
#include <vector>
#include "BaseCallerUtils.h"

#ifdef _WIN32
  #define ALIGN(AAlignSize) __declspec(align(AAlignSize))
  #define ALWAYS_INLINE __forceinline
  #define EXPECTED(ABoolExpression) ABoolExpression
  #define UNEXPECTED(ABoolExpression) ABoolExpression
  #define RESTRICT_PTR * __restrict
#else
  #define ALIGN(AAlignSize) __attribute__((aligned(AAlignSize)))
  #define ALWAYS_INLINE inline __attribute__((always_inline))
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

#define MAX_PATH_DELAY 40

#pragma pack(push, 1)
struct PathRec {
  int flow;
  int window_start;
  int window_end;
  float res;
  float metr;
  float flowMetr;
  int dotCnt;
  float penalty;
  float state[MAX_VALS];
  float pred[MAX_VALS];
  char sequence[2*MAX_VALS + 12]; // +12 makes the enitre struct align well
  int  sequence_length;
  int last_hp;
  int nuc;
  float calib_A[MAX_VALS];
  float calib_B[MAX_VALS];
  float state_inphase[MAX_VALS];
};
#pragma pack(pop)


class TreephaserSSE {
public:

  //! @brief      Default constructor
  TreephaserSSE();

  //! @brief      Constructor
  //! @param[in]  flow_order  Flow order object
  //! @param[in]  windowSize  Size of the normalization window to use.
  TreephaserSSE(const ion::FlowOrder& flow_order, const int windowSize);

  //! @brief      Set flow order and initialize internal variables
  //! @param[in]  flow_order  Flow order object
  void SetFlowOrder(const ion::FlowOrder& flow_order);

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
  PathRec* parent;
  int best;

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
    pm_model_available_ = false;
    recalibrate_predictions_ = false;
    As_ = 0; Bs_ = 0;
  };

  //! @brief  Switch to disable / enable the use of recalibration during the normalization phase
  void SkipRecalDuringNormalization(bool skip_recal)
    { skip_recal_during_normalization_ = skip_recal; };

  //! @brief  Perform a more advanced simulation to generate QV predictors
  void  ComputeQVmetrics(BasecallerRead& read);
  void  ComputeQVmetrics_flow(BasecallerRead& read, vector<int>& flow_to_base, const bool flow_predictors_=false);

  int nuc_char_to_int(char nuc) {if (nuc=='A') return 0; if (nuc=='C') return 1; if (nuc=='G') return 2; if (nuc=='T') return 3; return -1;}

protected:

  //! @brief     Solving a read
  bool  Solve(int begin_flow, int end_flow);
  //! @brief     Normalizing a read
  void  WindowedNormalize(BasecallerRead& read, int step);
  //! @brief     Make recalibration changes to predictions explicitly visible
  void  RecalibratePredictions(PathRec *maxPathPtr);
  //! @brief     Resetting recalibration data structure
  void  ResetRecalibrationStructures(int num_flows);
  //! @brief      Initialize floating point array variables with a value
  void InitializeVariables(float init_val);

  void  sumNormMeasures();
  void  advanceState4(PathRec RESTRICT_PTR parent, int end);
  void  nextState(PathRec RESTRICT_PTR path, int nuc, int end);

  // There was a small penalty in making these arrays class members, as opposed to static variables
  ALIGN(64) short ts_NextNuc[4][MAX_VALS];
  ALIGN(64) float ts_Transition[4][MAX_VALS];
  ALIGN(64) int   ts_NextNuc4[MAX_VALS][4];
  ALIGN(64) float ts_Transition4[MAX_VALS][4];

  ALIGN(64) float rd_NormMeasure[MAX_VALS];
  ALIGN(64) float rd_SqNormMeasureSum[MAX_VALS];

  ALIGN(64) PathRec sv_pathBuf[MAX_PATHS+1];

  ALIGN(64) float ft_stepNorms[MAX_STEPS];

  ALIGN(64) float ad_MinFrac[4];
  ALIGN(16) int   ad_FlowEnd[4];
  ALIGN(16) int   ad_Idx[4];
  ALIGN(16) int   ad_End[4];
  ALIGN(16) int   ad_Beg[4];
  ALIGN(16) char  ad_Buf[4*MAX_VALS*4*sizeof(float)];

  ion::FlowOrder      flow_order_;                //!< Sequence of nucleotide flows

  PathRec *sv_PathPtr[MAX_PATHS+1];
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

};

#endif // TREEPHASERSSE_H
