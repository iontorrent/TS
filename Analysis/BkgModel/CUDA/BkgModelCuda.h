/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef BKGMODELCUDA_H
#define BKGMODELCUDA_H

#include "SignalProcessingMasterFitter.h"
#include "ObsoleteCuda.h"
#include "MultiLevMar.h"
#include "MathOptim.h"

// std headers
#include <iostream>

// cuda
#include "cuda_runtime.h"

using std::vector;

//
// Device Functions
//

template <typename T>
__device__ __host__
inline void clamp(T& x, T a, T b);

__device__
float poiss_cdf_approx(int n, float x);

__device__
float erf_approx( float x );

__device__
float exp_approx( float x, float* ExpApproxArray, int nElements );



//
// CUDA Background Model Class
//

class BkgModelCuda
{
public:
    // Constructor
    BkgModelCuda();
    BkgModelCuda(SignalProcessingMasterFitter&, int, CpuStep_t*);

    // Destructor
    ~BkgModelCuda();

    
    // Copy lookup tables to GPU constant memory
    static void InitializeConstantMemory(PoissonCDFApproxMemo& poiss_cache);

private:

    // create Stream for each individual cuda thread
    cudaStream_t stream;

    // Print debugging messages
    static const bool verbose_cuda = false;

    // Size of matrices, must be larger than biggest possible number of outputs
    // CUDA's memory controller performs well with over-allocated size of 64
    static const int mat_dim_bead = 32;
    static const int mat_dim_region = 64;

    // Crossover point where it's faster to copy beads to CPU, process, and copy back to GPU
    static const int min_gpu_batch = 4;

    // Size constants
    int num_fb, num_steps, num_beads, num_pts, num_ev;

    // Reference to parent model
    SignalProcessingMasterFitter& bkg;

    // Step list defined globally in BkgModel.cpp
    int current_step;
    CpuStep_t* step_list;

    // CPU data structures
    int active_bead_count;
    int* active_bead_list_host;
    bool* cont_proc_host;

    // Host memory (page locked)
    float* ival_host;
    float* fval_host;
    float* sbg_host;
    float* sbg_slope_host;
    double* jtj_host;
    double* rhs_host;
    int* fit_flag_host;
    int* req_flag_host;

    // GPU memory (inputs)
    int* flow_ndx_map_cuda;
    float* frame_number_cuda;
    float* delta_frame_cuda;
    float* dark_matter_compensator_cuda;
    float* EmphasisVectorByHomopolymer_cuda;            
    float* EmphasisScale_cuda;   
    int* buff_flow_cuda;
    float* residual_cuda;
    float* new_residual_cuda;
    int* req_done_cuda;
    int* fit_flag_cuda;
    int* req_flag_cuda;

    // GPU buffers
    FG_BUFFER_TYPE* fg_buffers_cuda;

    // GPU memory (constant LUT)
    int exp_approx_table_size;
    float* exp_approx_table_cuda;

    // GPU memory (I/O)
    float* scratch_space_cuda;
    float* ival_cuda;
    float* sbg_cuda;
    float* sbg_slope_cuda;
    float* ivtmp_cuda;
    float* lambda_cuda;
    bool* cont_proc_cuda;
    bool* well_complete_cuda;
    int* i_start_cuda;
    float* c_dntp_top_pc_cuda;
    float* clonal_call_scale_cuda;

    // GPU memory (matrix related)
    double* sum_cuda;
    double* jtj_cuda;
    double* rhs_cuda;
    double* jtj_lambda_cuda;
    double* delta_cuda;

    // Offset indexes
    delta_mat_output_line* output_list_local_cuda;

    // GPU memory (local parameters)
    int* active_bead_list_cuda;
    bead_params* eval_params_cuda;
    bead_params* params_nn_cuda;
    bead_state* params_state_cuda;
    bound_params* params_low_cuda;
    bound_params* params_high_cuda;

    // GPU memory (local parameters)
    reg_params* new_rp_cuda;
    reg_params* eval_rp_cuda;
    int* WhichEmphasis_cuda;

    // free GPU memory for all the members 
    void Clear();

    // create cuda arrays for texture binding
    void createCuSbgArray();
    void createCuDarkMatterArray();

    // Initialize a new well fit
    void InitializeFit();

    // Initialize per flow fit
    void InitializePerFlowFit();

    // PartialDeriv Step Functions
    void DfdgainStep();
    void DfderrStep();
    void YerrStep();
    void DfdtshStep();
    void FvalStep(CpuStep_t*);

    // Local Matrix Functions
    void SingularMatrixCheck(int);
    void CopyMatrices();
    void BuildLocalMatrices( BkgFitMatrixPacker* );
    void SolveAndAdjustLocalParameters( BkgFitMatrixPacker*, BkgFitMatrixPacker*, reg_params&, int&, int&, int&, int&, int );
    void FactorizeAndSolveMatrices(int num_outputs);
    void AdjustParameters(int num_outputs);
    void CalculateNewResidual();
    void CalculateNewRegionalResidual();
    void AdjustLambdaAndUpdateParameters( bool reg_fit, int iter, int max_reg_iter, int& req_done, int& num_fit, int nreg_group );

    // Regional Matrix Functions
    void BuildRegionalMatrix( BkgFitMatrixPacker*, int, int&, float& );
    void AccumulateToRegionalMatrix( BkgFitMatrixPacker* );
    void SolveAndAdjustRegionalParameters( BkgFitMatrixPacker*, int&, int&, float&, float&, float );
    void FactorizeAndSolveRegionalMatrix( int, float );
    float CalculateRegionalResidual();
    void AdjustRegionalParameters( bead_params*, reg_params* );

    // Function wrappers
    void DoStepDiff(CpuStep_t*,int);
    void CalcPartialDerivWithEmphasis(float* p, float dp);
    void MultiFlowComputeCumulativeIncorporationSignalCuda(float* ival, reg_params*, bool performNucRise);
    void MultiFlowComputeIncorporationPlusBackgroundCuda(float* fval, float* ival, float* sbg, reg_params*);
    void MultiCycleNNModelFluxPulse_tshiftPartialDeriv(float*, bead_params*, reg_params*, float*);

    // Build list of active beads for various steps of the fitting process
    void BuildActiveBeadList( bool, bool, bool, bool, bool, int, int&, float& );

    // new functions for v7
    float CalculateFitError();
    void CalcXtalkFlux();
    void ClearMultiFlowMembers();
    void AllocateSingleFlowMembers();
   
    void BuildActiveBeadListMinusIgnoredWells(bool check_cont_proc);
    void AllocateBinarySearchArrays(); 
    void InitializeArraysForBinarySearch(bool restart);
    void EvaluateAmplitudeFit(float* amp, float* err);
    void UpdateAmplitudeForEvaluation(float* amp);
    void CalculateFitErrorForBinarySearch(float* err);
    void ErrorForBinarySearch(float* err);
    void UpdateAp();
    void BinarySearchStepOne();
    void BinarySearchStepTwo(float min_step);
    void UpdateAmplitudeAfterBinarySearch();
    void ClearBinarySearchMemory();

    void DetermineSingleFlowFitType();
    void SetWellRegionParams();
    void CalcCDntpTop(bead_params* p, reg_params* rp);
    void SingleFlowBkgCorrect();
    void SetWeightVector();
    void Fit();
    void EvaluateFunc(float* fval, float* f1, float* k1, bool* flowDone, bool* 
        iterDone);
    void ComputeJacobianSingleFlow(float* jac, float* tmp, int paramIdx);
    void ComputeJTJSingleFlow(float* jac, bool* flowDone);
    void ComputeRHSSingleFlow(float* jac, float* err_vec, bool* flowDone);
    void ComputeLHSWithLambdaSingleFLow(bool* flowDone, bool* iterDone);
    void SolveSingleFlow(bool* flowDone, bool* iterDone);
    void AdjustAndClampParamsSingleFlow(float* f1, float* k1, bool* flowDone, bool* iterDone);
    void CalcResidualForSingleFLowFit(float* fval, float* err_vect, float* r, bool* flowDone);
    void AdjustParamFitIter(bool doKrate, int paramIdx, float* f1_new, float* k1_new);
    void AdjustParamBackFitIter(bool doKrate, int paramIdx, float* f1_new, float* k1_new);
    void UpdateParamsAndFlowValsAfterEachIter(bool* flowsToUpdate, bool* flowDone, 
        float* tmp_fval, float* f1, float* k1);
    void DoneTest(bool* flowDone);
    void CheckForIterationCompletion(bool* flowsToUpdate, bool* beadIter, bool* 
        flowDone, bool* iterDone, float* r1, float* r2);
    bool ScanBeadArrayForCompletion(bool* beadItr);
    void UpdateFinalParamsSingleFlow();
    void ClearSingleFlowMembers();
    void CalculateEmphasisVectorIndexForFlow(bead_params* p, int max_emphasis);
   
    // Projection search related functions
    void AllocateMemoryForProjectionSearch();
    void ClearMemoryForProjectionSearch();
    void InitializeProjectionSearchAmplitude();
    void UpdateProjectionAmplitude(bead_params* p);
    void ProjectOnAmplitudeVector(float* X, float* Y);
    void RedSolveHydrogenFlowInWellAndAdjustedForGain(float* write_buffer, float* read_buffer);
    void ObtainBackgroundCorrectedSignal();
    void PostBlueSolveBackgroundTraceSteps(float* write_buffer);
    void BlueSolveBackgroundTrace(float* write_buffer, float* read_buffer);
    void ComputeCumulativeIncorporationHydrogensForProjection(float* write_buffer, reg_params* rp); 

    // xtalk routines
    void AllocateMemoryForXtalk(int neis);
    void ComputeXtalkContributionFromEveryBead();
    void ComputeXtalkTraceForEveryBead(int neis);
    void RedHydrogenForXtalk(float* write_buffer, float* read_buffer, float tau);
    void DiminishIncorporationTraceForXtalk(float* write_buffer, float* read_buffer);
    void ApplyXtalkMultiplier(float* write_buffer, float multiplier); 
    void GenerateNeighbourMap();
    void ClearMemoryForXtalk();

    // Dot Product Setup routines
    void BuildLocalMatrices(int isRegFit, BkgFitMatrixPacker* fit);
    void SetUpDotMatrixProduct(BkgFitMatrixPacker* well_fit, BkgFitMatrixPacker* reg_fit);
    void SetUpMatrixDotProductForWell(BkgFitMatrixPacker* reg_fit);
    void SetUpMatrixDotProductForRegion(BkgFitMatrixPacker* reg_fit);
    void ClearDotProductSetup(BkgFitMatrixPacker* well_fit, BkgFitMatrixPacker* reg_fit);

private:

    bool clearMultiFlowAllocs;

    // binary search amplitude
    float* ac_cuda;
    float* ec_cuda;
    float* ap_cuda;
    float* ep_cuda;
    float* step_cuda;
    int* done_cuda;


    // members for single flow fit
    int numSingleFitParams;
    int ampParams;
    int krateParams;
    float* xtflux_cuda; // beads*flows*npts
    bool* isKrateFit_cuda;

    float* singleFlowFitKrateParams_cuda;
    float* singleFlowFitKrateParamMin_cuda;
    float* singleFlowFitKrateParamMax_cuda;

    float* singleFlowFitParams_cuda;
    float* singleFlowFitParamMin_cuda;
    float* singleFlowFitParamMax_cuda;

    float* tauB_cuda;
    float* sens_cuda;
    float* SP_cuda;
    float* fval_cuda;
    float* fgbuffers_float_cuda;
    float* weight_cuda;
    float* wtScale_cuda;
    int* done_cnt_cuda;

    // projection search amplitude
    float* correctedSignal_cuda;
    float* model_trace_cuda;
    float* projectionAmp_cuda;

    // xtalk data
    float* write_buffer_cuda;
    float* incorporation_trace_cuda;
    float* xtalk_nei_trace_cuda;
    int* numNeisPerBead_cuda;
    int* NeiMapPerBead_cuda;

    // Dot Product Setup
    int inst_buf_len;
    int* well_sub_inst_cuda;
    int* well_f1_offset_cuda;
    int* well_f2_offset_cuda;
    int* well_dotProd_len_cuda;
    int* well_col_cuda;
    int* well_row_cuda;
    AssyMatID* well_matId_cuda;
    int* reg_sub_inst_cuda;
    int* reg_f1_offset_cuda;
    int* reg_f2_offset_cuda;
    int* reg_dotProd_len_cuda;
    int* reg_col_cuda;
    int* reg_row_cuda;
    AssyMatID* reg_matId_cuda;


public:

    // Public entry functions
    int MultiFlowSpecializedLevMarFitParameters(int max_iter, int max_reg_iter, BkgFitMatrixPacker* well_fit, BkgFitMatrixPacker* reg_fit, float lamda_start, int clonal_restriction = 0);
    void BinarySearchAmplitude(float min_step, bool restart);
    void FitAmplitudePerFlow();
    void ProjectionSearch();
    void NewXtalk();
};

#endif // BKGMODELCUDA_H
