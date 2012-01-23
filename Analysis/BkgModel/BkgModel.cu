/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <sys/time.h>
#include <cblas.h>

#include "DNTPRiseModel.h"
#include "BkgModelCudaKernels.h"
#include "BkgFitMatrixPacker.h"


#define NO_CUDA_DEBUG
#ifndef NO_CUDA_DEBUG
#define CUDA_ERROR_CHECK()                                                                     \
{                                                                                              \
    cudaError_t err = cudaGetLastError();                                                      \
    if ( err != cudaSuccess && err != cudaErrorSetOnActiveProcess ) {                          \
        std::cout << " +----------------------------------------" << std::endl                 \
                  << " | ** CUDA ERROR! ** " << std::endl                                      \
                  << " | Error: " << err << std::endl                                          \
                  << " | Msg: " << cudaGetErrorString(err) << std::endl                        \
                  << " | File: " << __FILE__ << std::endl                                      \
                  << " | Line: " << __LINE__ << std::endl                                      \
                  << " +----------------------------------------" << std::endl << std::flush;  \
                  exit(-1); }                                                                  \
}
#else
#define CUDA_ERROR_CHECK() {}
#endif




//
// Constructor
//

BkgModelCuda::BkgModelCuda(BkgModel& _bkg, int _num_steps, CpuStep_t* _step_list) :
    bkg(_bkg), num_steps(_num_steps), step_list(_step_list)
{
    // Pull size constants from the host background model for naming consistency
    num_pts = bkg.time_c.npts;
    num_fb = bkg.my_flow.numfb;
    num_beads = bkg.my_beads.numLBeads;
    num_ev = bkg.emphasis_data.numEv;

    Timer t;

    if ( verbose_cuda )
        printf( "[Reg: %d,%d] [Initialization] ", bkg.region->col, bkg.region->col );


    // Allocate page locked host memory
    cudaMallocHost( (void**) &ival_host, sizeof(float) * num_pts * num_fb ); CUDA_ERROR_CHECK();
    cudaMallocHost( (void**) &fval_host, sizeof(float) * num_pts * num_fb ); CUDA_ERROR_CHECK();
    cudaMallocHost( (void**) &sbg_host, sizeof(float) * num_pts * num_fb ); CUDA_ERROR_CHECK();
    cudaMallocHost( (void**) &sbg_slope_host, sizeof(float) * num_pts * num_fb ); CUDA_ERROR_CHECK();
    cudaMallocHost( (void**) &cont_proc_host, sizeof(bool) * num_beads ); CUDA_ERROR_CHECK();
    cudaMallocHost( (void**) &active_bead_list_host, sizeof(int) * num_beads ); CUDA_ERROR_CHECK();
    cudaMallocHost( (void**) &jtj_host, sizeof(double) * mat_dim * mat_dim ); CUDA_ERROR_CHECK();
    cudaMallocHost( (void**) &rhs_host, sizeof(double) * mat_dim ); CUDA_ERROR_CHECK();
    cudaMallocHost( (void**) &fit_flag_host, sizeof(int) * num_beads ); CUDA_ERROR_CHECK();
    cudaMallocHost( (void**) &req_flag_host, sizeof(int) * num_beads ); CUDA_ERROR_CHECK();

    // Allocate GPU parameter arrays
    cudaMalloc( (void**) &active_bead_list_cuda, sizeof(int) * num_beads ); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &eval_params_cuda, sizeof(bead_params) * num_beads ); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &params_nn_cuda, sizeof(bead_params) * num_beads ); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &params_low_cuda, sizeof(bead_params) * num_beads ); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &params_high_cuda, sizeof(bead_params) * num_beads ); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &eval_rp_cuda, sizeof(reg_params) ); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &new_rp_cuda, sizeof(reg_params) ); CUDA_ERROR_CHECK();

    // Allocate GPU memory (I/O)
    cudaMalloc( (void**) &scratch_space_cuda, sizeof(float) * num_pts * num_fb * num_steps * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &ival_cuda, sizeof(float) * num_fb * num_pts * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &sbg_cuda, sizeof(float) * num_fb * num_pts); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &sbg_slope_cuda, sizeof(float) * num_fb * num_pts); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &ivtmp_cuda, sizeof(float) * num_fb * num_pts * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &cont_proc_cuda, sizeof(bool) * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &well_complete_cuda, sizeof(bool) * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &fit_flag_cuda, sizeof(int) * num_beads ); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &req_flag_cuda, sizeof(int) * num_beads ); CUDA_ERROR_CHECK();

    // Zero out GPU scratch space
    cudaMemset( scratch_space_cuda, 0, sizeof(float) * num_pts * num_fb * num_steps * num_beads );

    // Allocate GPU memory (input data)
    cudaMalloc( (void**) &flow_ndx_map_cuda, sizeof(int) * num_fb); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &frame_number_cuda, sizeof(float) * bkg.my_trace.imgFrames); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &delta_frame_cuda, sizeof(float) * bkg.my_trace.imgFrames); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &dark_matter_compensator_cuda, sizeof(float) * NUMNUC * num_pts); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &buff_flow_cuda, sizeof(int) * num_fb); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &EmphasisVectorByHomopolymer_cuda, sizeof(float) * num_ev * num_pts); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &EmphasisScale_cuda, sizeof(float) * num_ev); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &fg_buffers_cuda, sizeof(FG_BUFFER_TYPE) * num_fb * num_pts * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &residual_cuda, sizeof(float) * num_beads); CUDA_ERROR_CHECK(); 
    cudaMalloc( (void**) &new_residual_cuda, sizeof(float) * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &lambda_cuda, sizeof(float) * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc((void**) &c_dntp_top_pc_cuda, sizeof(float)*num_beads*num_pts*NUMNUC*ISIG_SUB_STEPS_MULTI_FLOW); CUDA_ERROR_CHECK();
    cudaMalloc((void**) &i_start_cuda, sizeof(int)*num_beads*NUMNUC); CUDA_ERROR_CHECK();
    cudaMalloc((void**) &clonal_call_scale_cuda, sizeof(float)*12); CUDA_ERROR_CHECK();

    // Allocate GPU memory (matrix steps)
    cudaMalloc((void**) &sum_cuda, sizeof(double) * num_beads ); CUDA_ERROR_CHECK();
    cudaMalloc((void**) &jtj_cuda, sizeof(double) * num_beads * mat_dim * mat_dim ); CUDA_ERROR_CHECK();
    cudaMalloc((void**) &rhs_cuda, sizeof(double) * num_beads * mat_dim ); CUDA_ERROR_CHECK();
    cudaMalloc((void**) &jtj_lambda_cuda, sizeof(double) * num_beads * mat_dim * mat_dim ); CUDA_ERROR_CHECK();
    cudaMalloc((void**) &delta_cuda, sizeof(double) * num_beads * mat_dim ); CUDA_ERROR_CHECK();

    // Allocate GPU memory (offset lists)
    cudaMalloc((void**) &output_list_local_cuda, sizeof(delta_mat_output_line) * mat_dim); CUDA_ERROR_CHECK();

    // Copy unchanging data structures to GPU memory
    cudaMemcpy( flow_ndx_map_cuda, bkg.my_flow.flow_ndx_map, sizeof(int) * num_fb, cudaMemcpyHostToDevice );
    cudaMemcpy( frame_number_cuda, bkg.time_c.frameNumber, sizeof(float) * bkg.my_trace.imgFrames, cudaMemcpyHostToDevice );
    cudaMemcpy( delta_frame_cuda, bkg.time_c.deltaFrame, sizeof(float) * bkg.my_trace.imgFrames, cudaMemcpyHostToDevice );
    cudaMemcpy( buff_flow_cuda, bkg.my_flow.buff_flow, sizeof(int) * num_fb, cudaMemcpyHostToDevice );
    cudaMemcpy( fg_buffers_cuda, bkg.my_trace.fg_buffers, sizeof(FG_BUFFER_TYPE) * num_fb * num_pts * num_beads, cudaMemcpyHostToDevice );
    cudaMemcpy( residual_cuda, bkg.lev_mar_fit->lm_state.residual, sizeof(float) * num_beads, cudaMemcpyHostToDevice );
    cudaMemcpy( clonal_call_scale_cuda, bkg.global_defaults.clonal_call_scale, sizeof(float) * 12, cudaMemcpyHostToDevice );
    

    memset(fval_host, 0, sizeof(float)*num_pts*num_fb);

#ifndef USE_CUDA_EXP
    // exp look up table is too large for constant memory
    cudaMalloc((void**) &exp_approx_table_cuda, sizeof(EXP_APPROX_TABLE)); CUDA_ERROR_CHECK();
    exp_approx_table_size = sizeof(EXP_APPROX_TABLE) / sizeof(float);
    cudaMemcpy(exp_approx_table_cuda, EXP_APPROX_TABLE, sizeof(EXP_APPROX_TABLE), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK();
#endif

    // Initialize constant memory with mathematical lookup tables
    //InitializeConstantMemory();

    if ( verbose_cuda )
        printf( "(%f s)\n", t.elapsed() );
}

//
// Destructor
//

BkgModelCuda::~BkgModelCuda()
{
    Clear();
}

void BkgModelCuda::Clear() {
 
    // Free CPU page locked arrays
    cudaFreeHost( cont_proc_host );
    cudaFreeHost( active_bead_list_host );
    cudaFreeHost( sbg_host );

    // Free GPU parameter arrays
    cudaFree( active_bead_list_cuda );
    cudaFree( params_nn_cuda );
    cudaFree( params_low_cuda );
    cudaFree( params_high_cuda );
    cudaFree( eval_rp_cuda ); 
    cudaFree( eval_params_cuda);

    // Free GPU memory (I/O)
    cudaFree( sbg_cuda );
    cudaFree( cont_proc_cuda );

    // Free GPU memory (input data)
    cudaFree(flow_ndx_map_cuda);
    cudaFree(frame_number_cuda);
    cudaFree(delta_frame_cuda);
    cudaFree(dark_matter_compensator_cuda);
    cudaFree(buff_flow_cuda);
    cudaFree(EmphasisVectorByHomopolymer_cuda);
    cudaFree(EmphasisScale_cuda);
    cudaFree(fg_buffers_cuda);
    cudaFree(lambda_cuda);
    cudaFree(c_dntp_top_pc_cuda);
    cudaFree(i_start_cuda);
    cudaFree(clonal_call_scale_cuda);

    // Free GPU memory (matrix steps)
    cudaFree(jtj_cuda);
    cudaFree(rhs_cuda);
    cudaFree(jtj_lambda_cuda);
    cudaFree(delta_cuda);

    #ifndef USE_CUDA_EXP
    cudaFree( exp_approx_table_cuda );
#endif
}

void BkgModelCuda::ClearBinarySearchMemory() {

    // Free GPU memory for binary search code
    cudaFree(ac_cuda);
    cudaFree(ap_cuda);
    cudaFree(ec_cuda);
    cudaFree(ep_cuda);
    cudaFree(step_cuda);
}

//
// Member Functions
//

void BkgModelCuda::InitializeConstantMemory(PoissonCDFApproxMemo& poiss_cache)
{
    cudaMemcpyToSymbol(POISS_0_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[0], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_1_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[1], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_2_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[2], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_3_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[3], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_4_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[4], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_5_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[5], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_6_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[6], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_7_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[7], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_8_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[8], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_9_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[9], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_10_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[10], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol(POISS_11_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[11], sizeof(float)*MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
#ifndef USE_CUDA_ERF
    cudaMemcpyToSymbol(ERF_APPROX_TABLE_CUDA, ERF_APPROX_TABLE, sizeof(ERF_APPROX_TABLE)); CUDA_ERROR_CHECK();
#endif

}


void BkgModelCuda::createCuSbgArray() {
  
}

void BkgModelCuda::createCuDarkMatterArray() {

}

void BkgModelCuda::InitializeFit()
{
    // Transfers input data that changes between fit processes
    cudaMemcpy( fg_buffers_cuda, bkg.my_trace.fg_buffers, sizeof(FG_BUFFER_TYPE) * num_fb * num_pts * num_beads, cudaMemcpyHostToDevice );
    cudaMemcpy( dark_matter_compensator_cuda, bkg.my_regions.dark_matter_compensator, sizeof(float) * NUMNUC* num_pts, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    cudaMemcpy( params_low_cuda, bkg.my_beads.params_low, sizeof(bead_params) * num_beads, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    cudaMemcpy( params_high_cuda, bkg.my_beads.params_high, sizeof(bead_params) * num_beads, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    cudaMemcpy( EmphasisScale_cuda, bkg.emphasis_data.EmphasisScale, sizeof(float) * num_ev, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    
    for(int i=0; i<num_ev; ++i) { 
    	cudaMemcpy( EmphasisVectorByHomopolymer_cuda + num_pts*i, bkg.emphasis_data.EmphasisVectorByHomopolymer[i], sizeof(float) * num_pts, cudaMemcpyHostToDevice );
	CUDA_ERROR_CHECK();
    }
}

void BkgModelCuda::InitializePerFlowFit() {

    // Transfers input data that changes between fit processes
    cudaMemcpy( dark_matter_compensator_cuda, bkg.my_regions.dark_matter_compensator, sizeof(float) * NUMNUC* num_pts, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    cudaMemcpy( EmphasisScale_cuda, bkg.emphasis_data.EmphasisScale, sizeof(float) * num_ev, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    for(int i=0; i<num_ev; ++i) { 
    	cudaMemcpy( EmphasisVectorByHomopolymer_cuda + num_pts*i, bkg.emphasis_data.EmphasisVectorByHomopolymer[i], sizeof(float) * num_pts, cudaMemcpyHostToDevice );
	CUDA_ERROR_CHECK();
    }
}


void BkgModelCuda::DfdgainStep()
{
    dim3 threads(num_pts);
    dim3 blocks(active_bead_count);

    DfdgainStep_k <<< blocks, threads>>> ( scratch_space_cuda, eval_params_cuda, EmphasisVectorByHomopolymer_cuda,
            active_bead_list_cuda, num_fb, num_pts, num_steps, current_step );

    CUDA_ERROR_CHECK();
}


void BkgModelCuda::DfderrStep()
{
    dim3 threads(num_pts);
    dim3 blocks(active_bead_count);

    DfderrStep_k <<< blocks, threads >>> ( scratch_space_cuda, eval_params_cuda, EmphasisVectorByHomopolymer_cuda,
            dark_matter_compensator_cuda, flow_ndx_map_cuda, active_bead_list_cuda,
            num_fb, num_pts, num_steps, current_step );

    CUDA_ERROR_CHECK();
}


void BkgModelCuda::YerrStep()
{
    const int beads_per_block = 12;
    dim3 threads(num_fb, beads_per_block); // 20 x 12 = 240
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    YerrStep_k <beads_per_block> <<< blocks, threads >>> ( scratch_space_cuda, eval_params_cuda, EmphasisVectorByHomopolymer_cuda, EmphasisScale_cuda, active_bead_list_cuda, fg_buffers_cuda, residual_cuda, num_fb, num_pts, num_steps, num_beads, current_step, active_bead_count );

    CUDA_ERROR_CHECK();
}


void BkgModelCuda::DfdtshStep()
{
    reg_params eval_rp;
    cudaMemcpy( &eval_rp, eval_rp_cuda, sizeof(reg_params), cudaMemcpyDeviceToHost );

    bkg.my_trace.GetShiftedSlope( eval_rp.tshift, sbg_slope_host );
    cudaMemcpy( sbg_slope_cuda, sbg_slope_host, sizeof(float)*num_fb*num_pts, cudaMemcpyHostToDevice );

    MultiCycleNNModelFluxPulse_tshiftPartialDeriv( scratch_space_cuda, eval_params_cuda, eval_rp_cuda, sbg_slope_cuda );
}


void BkgModelCuda::FvalStep( CpuStep_t* StepP )
{
    // Set correct ival pointer (cpu and cuda) for this step
    float* ival_ptr_cuda = ival_cuda;

    DoStepDiff(StepP, 1);

    if (StepP->doBoth)
    {
        // Don't overwrite ival
        if (StepP->doBoth != CalcFirst)
            ival_ptr_cuda = ivtmp_cuda;

        MultiFlowComputeCumulativeIncorporationSignalCuda(ival_ptr_cuda, eval_rp_cuda, true);
    }

    MultiFlowComputeIncorporationPlusBackgroundCuda( scratch_space_cuda, ival_ptr_cuda, sbg_cuda, eval_rp_cuda );

    if (StepP->doBoth != CalcFirst)
        CalcPartialDerivWithEmphasis(scratch_space_cuda, StepP->diff);

    DoStepDiff(StepP, 0);

    if (verbose_cuda) {
      float* debugArray = new float[num_beads*num_pts*num_fb*num_steps];
      cudaMemcpy(debugArray, scratch_space_cuda, sizeof(float)*num_beads*num_pts*num_fb*num_steps, cudaMemcpyDeviceToHost);
      //cudaMemcpy(debugArray, ival_ptr_cuda, sizeof(float)*num_beads*num_pts*num_fb, cudaMemcpyDeviceToHost);
      printf("GPU Fval Data");
      for (int i=0; i<1; ++i) {
          printf("Bead %d\n", i);
          for (int j=0; j<num_fb; ++j) {
              printf("Flow buffer %d\n", j);
              for (int k=0; k<num_pts; ++k) {
			printf("%3.3f ", debugArray[i*num_pts*num_fb*num_steps + j*num_pts + k]); 
	      }
              printf("\n");
          }
      }
    }
}


void BkgModelCuda::BuildLocalMatrices(BkgFitMatrixPacker* well_fit)
{
    // Zero out matricies
    cudaMemset(jtj_cuda, 0, sizeof(double) * num_beads * mat_dim * mat_dim);
    cudaMemset(rhs_cuda, 0, sizeof(double) * num_beads * mat_dim);

    mat_assembly_instruction* instruction_list = well_fit->getInstList();
    int num_instructions = well_fit->getNumInstr();

    for (int i = 0; i < num_instructions; i++)
    {
        cudaMemset(sum_cuda, 0, sizeof(double) * num_beads);

        for (int j = 0; j < instruction_list[i].cnt; j++)
        {
            // Get offsets
            int len = instruction_list[i].si[j].len;
            int f1_offset = instruction_list[i].si[j].src1 - bkg.my_scratch.scratchSpace;
            int f2_offset = instruction_list[i].si[j].src2 - bkg.my_scratch.scratchSpace;

            const int beads_per_block = 64;
            const int block_size = 8;
            dim3 threads(block_size, beads_per_block);
            dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block);

            DotProductMatrixSums_k < block_size, beads_per_block > <<< blocks, threads >>>( sum_cuda, scratch_space_cuda, f1_offset, f2_offset, len, active_bead_list_cuda, num_fb, num_pts, num_steps, active_bead_count );
        }

        AssyMatID matId = well_fit->my_fit_instructions.input[i].matId;
        int col = well_fit->my_fit_instructions.input[i].mat_col;
        int row = well_fit->my_fit_instructions.input[i].mat_row;

        const int beads_per_block = 128;
        dim3 threads(beads_per_block);
        dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block);

        WriteMatrixFromSums_k <beads_per_block> <<< blocks, threads >>> ( matId, row, col, sum_cuda, mat_dim, jtj_cuda, rhs_cuda, active_bead_list_cuda, active_bead_count );
        CUDA_ERROR_CHECK();
    }
}


//
// -- CUDA Kernel Wrappers --
//
// These functions configure and call the GPU kernels
//

void BkgModelCuda::DoStepDiff(CpuStep_t* step, int add)
{
    int len = step->len;
    float diff = step->diff;

    dim3 blocks(active_bead_count);
    dim3 threads(len);

    if ( len == 0 )
        return;

    if ( step->paramsOffset != -1 )
    {
        int offset = step->paramsOffset;

        if ( add )
            DoStepDiff_k < true > <<< blocks, len >>>( eval_params_cuda, offset, diff, active_bead_list_cuda );
        else
            DoStepDiff_k < false > <<< blocks, len >>>( eval_params_cuda, offset, diff, active_bead_list_cuda );
    }
    else if( step->regParamsOffset != -1 )
    {
        int offset = step->regParamsOffset;

        if ( add )
            DoStepDiffReg_k < true > <<< 1, len >>>( eval_rp_cuda, offset, diff );
        else
            DoStepDiffReg_k < false > <<< 1, len >>>( eval_rp_cuda, offset, diff );
    } else if (step->nucShapeOffset != -1)
    {
        int offset = step->nucShapeOffset;

        if ( add )
            DoStepDiffNuc_k < true > <<< 1, len >>>( eval_rp_cuda, offset, diff );
        else
            DoStepDiffNuc_k < false > <<< 1, len >>>( eval_rp_cuda, offset, diff );
    }

    CUDA_ERROR_CHECK();
}


void BkgModelCuda::CalcPartialDerivWithEmphasis(float* p, float dp)
{
    //dim3 threads(num_pts);
    //dim3 blocks(active_bead_count);

    const int beads_per_block = 16;
    dim3 threads(num_pts, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    CalcPartialDerivWithEmphasis_k <beads_per_block> <<< blocks, threads >>> ( p, EmphasisVectorByHomopolymer_cuda, num_pts * num_fb, dp, num_pts,
            eval_params_cuda, active_bead_list_cuda, current_step, num_fb, num_pts, num_steps, active_bead_count);

    CUDA_ERROR_CHECK();
}

void BkgModelCuda::MultiFlowComputeCumulativeIncorporationSignalCuda(float* ival, reg_params* rp, 
    bool performNucRise)
{

    if(performNucRise)
        CalcCDntpTop(eval_params_cuda, rp);

    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    int smem_size = num_fb * num_pts * beads_per_block * sizeof(float);

    MultiFlowComputeCumulativeIncorporationSignalNew_CUDA_k <beads_per_block> <<< blocks, threads, smem_size>>> (eval_params_cuda, rp, ival, flow_ndx_map_cuda, delta_frame_cuda, buff_flow_cuda, num_pts, num_fb, active_bead_list_cuda, active_bead_count,
exp_approx_table_cuda, exp_approx_table_size, c_dntp_top_pc_cuda, i_start_cuda);

    CUDA_ERROR_CHECK();
}


void BkgModelCuda::MultiFlowComputeIncorporationPlusBackgroundCuda( float* fval, float* ival, float* sbg, reg_params* rp )
{
    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    int smem_size = num_fb * num_pts * beads_per_block * sizeof(float);

    MultiCycleNNModelFluxPulse_base_CUDA_k <NUMFB, beads_per_block> <<< blocks, threads, smem_size >>> (
        bkg.lev_mar_fit->lm_state.restrict_clonal, clonal_call_scale_cuda, fval, 
        eval_params_cuda, rp, ival, sbg, flow_ndx_map_cuda, 
        delta_frame_cuda, dark_matter_compensator_cuda, 
	buff_flow_cuda, num_pts, active_bead_list_cuda, current_step, num_steps, active_bead_count );

    CUDA_ERROR_CHECK();
}

void BkgModelCuda::MultiCycleNNModelFluxPulse_tshiftPartialDeriv( float* output, bead_params* eval_params, reg_params* eval_rp, float* sbg )
{
    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    MultiCycleNNModelFluxPulse_tshiftPartialDeriv_k <<< blocks, threads >>>( active_bead_list_cuda, active_bead_count, scratch_space_cuda, eval_params, eval_rp, EmphasisVectorByHomopolymer_cuda, sbg, delta_frame_cuda, buff_flow_cuda, flow_ndx_map_cuda, num_fb, num_pts, num_steps, current_step );

    CUDA_ERROR_CHECK();
}


void BkgModelCuda::CalculateNewResidual()
{
    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    CalculateResidual_k <<< blocks, threads >>> ( scratch_space_cuda, eval_params_cuda, EmphasisVectorByHomopolymer_cuda,
            EmphasisScale_cuda, active_bead_list_cuda,
            fg_buffers_cuda, new_residual_cuda, num_fb, num_pts, num_steps, num_beads );

    CUDA_ERROR_CHECK();
}

void BkgModelCuda::CalculateNewRegionalResidual()
{
    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    CalculateRegionalResidual_k <<< blocks, threads >>> ( scratch_space_cuda, eval_params_cuda, EmphasisVectorByHomopolymer_cuda,
            EmphasisScale_cuda, active_bead_list_cuda,
            fg_buffers_cuda, new_residual_cuda, num_fb, num_pts, num_steps, num_beads );

    CUDA_ERROR_CHECK();
}



void BkgModelCuda::SingularMatrixCheck( int n )
{
    dim3 threads(1);
    dim3 blocks(active_bead_count);

    SingularMatrixCheck_k < mat_dim > <<< blocks, threads >>> ( active_bead_list_cuda, n, jtj_cuda, cont_proc_cuda );
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::CopyMatrices()
{
    dim3 threads(mat_dim);
    dim3 blocks(active_bead_count);

    CopyMatrices_k < mat_dim > <<< blocks, threads >>> ( active_bead_list_cuda, jtj_cuda, rhs_cuda, jtj_lambda_cuda, delta_cuda, eval_params_cuda, params_nn_cuda );
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::FactorizeAndSolveMatrices(int num_outputs)
{
    const int block_size = 32; // With a block size of 32, we have implicit warp synchronization

    dim3 threads( block_size );
    dim3 blocks( active_bead_count );

    FactorizeAndSolveMatrices_k < mat_dim, block_size > <<< blocks, threads >>> ( active_bead_list_cuda, num_outputs, lambda_cuda, jtj_lambda_cuda, delta_cuda, active_bead_count );
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::FactorizeAndSolveRegionalMatrix(int num_outputs, float reg_lambda)
{
    // Factorizes the single matrix located at jtj_cuda and scales diagonal by reg_lambda

    const int block_size = 32;

    dim3 blocks( 1 );
    dim3 threads( block_size );

    FactorizeAndSolveRegionalMatrix_k < mat_dim, block_size > <<< blocks, threads >>> ( num_outputs, reg_lambda, jtj_lambda_cuda, delta_cuda );
    CUDA_ERROR_CHECK();
}


void BkgModelCuda::AdjustParameters(int num_outputs)
{
    dim3 threads(num_outputs);
    dim3 blocks(active_bead_count);

    AdjustParameters_k < mat_dim > <<< blocks, threads >>> ( active_bead_list_cuda, eval_params_cuda, params_nn_cuda, num_outputs, lambda_cuda, jtj_lambda_cuda, delta_cuda, output_list_local_cuda, params_low_cuda, params_high_cuda );
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::AdjustRegionalParameters(bead_params* p, reg_params* rp)
{
    dim3 threads(1);
    dim3 blocks(active_bead_count);

    AdjustRegionalParameters_k <<< blocks, threads >>> ( active_bead_list_cuda, rp, p, params_low_cuda, params_high_cuda, num_fb );
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::AdjustLambdaAndUpdateParameters(bool is_reg_fit, int iter, int max_reg_iter, int& req_done, int& num_fit, int nreg_group)
{
    bool is_under_iter = (iter <= max_reg_iter);

    dim3 threads(1);
    dim3 blocks(active_bead_count);

    cudaMemset( fit_flag_cuda, 0, sizeof(int)*active_bead_count ); CUDA_ERROR_CHECK();
    cudaMemset( req_flag_cuda, 0, sizeof(int)*active_bead_count ); CUDA_ERROR_CHECK();

    AdjustLambdaAndUpdateParameters_k <<< blocks, threads >>> ( active_bead_list_cuda, cont_proc_cuda, well_complete_cuda, new_residual_cuda, residual_cuda, params_nn_cuda, eval_params_cuda, lambda_cuda, is_reg_fit, is_under_iter, num_fb, fit_flag_cuda, req_flag_cuda );

    CUDA_ERROR_CHECK();

    cudaMemcpy( &fit_flag_host[0], fit_flag_cuda, sizeof(int) * active_bead_count, cudaMemcpyDeviceToHost ); CUDA_ERROR_CHECK();
    cudaMemcpy( &req_flag_host[0], req_flag_cuda, sizeof(int) * active_bead_count, cudaMemcpyDeviceToHost ); CUDA_ERROR_CHECK();

    int new_fit = 0;
    int new_req = 0;

    for (int i=0; i < active_bead_count; i++)
    {
        new_fit += fit_flag_host[i];
        new_req += req_flag_host[i];
    }

    num_fit -= new_fit;
    req_done += new_req;
}

void BkgModelCuda::AccumulateToRegionalMatrix(BkgFitMatrixPacker* reg_fit)
{
    int beads = active_bead_count;
    while ( beads != 1)
    {
        // Parallel reduction to sum the matrices in log(n) steps
        int blocks =  beads / 2;
        AccumulateToRegionalMatrix_k <mat_dim> <<< blocks, mat_dim >>> ( active_bead_list_cuda, beads, jtj_cuda, rhs_cuda );
        CUDA_ERROR_CHECK();
        beads -= blocks;
    }

    if ( active_bead_list_host[0] != 0 )
    {
        // For simplicity, move the accumulated matrices to the start of jtj and rhs
        int ibd = active_bead_list_host[0];
        cudaMemcpy( jtj_cuda, jtj_cuda + ibd*mat_dim*mat_dim, sizeof(double)*mat_dim*mat_dim, cudaMemcpyDeviceToDevice );
        CUDA_ERROR_CHECK();
        cudaMemcpy( rhs_cuda, rhs_cuda + ibd*mat_dim, sizeof(double)*mat_dim, cudaMemcpyDeviceToDevice );
        CUDA_ERROR_CHECK();
    }
}

void BkgModelCuda::BuildRegionalMatrix(BkgFitMatrixPacker* reg_fit, int nreg_group, int& reg_wells, float& reg_error)
{
    // Using this active bead list, build the local matrix for each contributing bead
    BuildLocalMatrices(reg_fit);

    // Accumulate the local matrices from each contributing bead into one regional matrix
    AccumulateToRegionalMatrix(reg_fit);
}

float BkgModelCuda::CalculateRegionalResidual()
{
    // Note: This step is non-destructive and could just use params_nn directly
    cudaMemcpy( eval_params_cuda, params_nn_cuda, sizeof(bead_params)*num_beads, cudaMemcpyDeviceToDevice );

    current_step = 0;
    AdjustRegionalParameters( eval_params_cuda, new_rp_cuda);
    MultiFlowComputeCumulativeIncorporationSignalCuda( ival_cuda, new_rp_cuda, true);
    MultiFlowComputeIncorporationPlusBackgroundCuda( scratch_space_cuda, ival_cuda, sbg_cuda, new_rp_cuda );

    // Calculate residual for each active bead
    cudaMemset( new_residual_cuda, 0, sizeof(float)*num_beads ); CUDA_ERROR_CHECK();
    CalculateNewRegionalResidual();

    // Copy
    std::vector<float> temp_residual(num_beads, 0);
    cudaMemcpy( &temp_residual[0], new_residual_cuda, sizeof(float)*num_beads, cudaMemcpyDeviceToHost );
    CUDA_ERROR_CHECK();

    float res_sum = 0;
    for ( int i = 0; i < num_beads; i++ )
        res_sum += temp_residual[i];

    return res_sum;
}


void BkgModelCuda::BuildActiveBeadList(bool reg_proc, bool check_cont_proc, bool check_well_region_fit, bool check_residual, bool check_fit_error, int nreg_group, int& reg_wells, float& reg_error)
{
    //
    // This function counts, and builds a list, of the beads that active.  Active beads are simply contributing to
    // calculations and should always be equal to or less than the number of beads being fit.  Also note that this
    // function does not transfer the list to the GPU; that is tasked to the main function.
    //

    // Reset active bead count
    active_bead_count = 0;

    // Loop over all beads
    for ( int ibd = 0; ibd < num_beads; ibd++ )
    {
        // Skip ignored wells
        if ( bkg.lev_mar_fit->lm_state.well_ignored[ibd] == true )
            continue;
	
	if ( check_fit_error ) {
        	active_bead_list_host[active_bead_count++] = ibd;
		continue;
	}

        // Skip completed wells
        if ( bkg.lev_mar_fit->lm_state.well_completed[ibd] == true )
            continue;

        // Extra checks for per-well solve loop
        if ( check_cont_proc && cont_proc_host[ibd] )
            continue;

        // Extra checks for regional processing
        if ( reg_proc && bkg.lev_mar_fit->lm_state.region_group[ibd] != nreg_group  )
            continue;

        // Extra checks for regional processing
        if ( reg_proc && check_well_region_fit && bkg.lev_mar_fit->lm_state.well_region_fit[ibd] == false )
            continue;

        // This check is performed when building for the regional matrix
        if ( reg_proc && check_residual )
        {
            reg_error += bkg.lev_mar_fit->lm_state.residual[ibd];
            if ( bkg.lev_mar_fit->lm_state.residual[ibd] < bkg.lev_mar_fit->lm_state.avg_resid * 4.0f )
                reg_wells++;
            else
                continue;
        }

        // Add to list and increment active count
        active_bead_list_host[active_bead_count++] = ibd;
    }
}


int BkgModelCuda::MultiFlowSpecializedLevMarFitParameters(int max_iter, int max_reg_iter, BkgFitMatrixPacker* well_fit, BkgFitMatrixPacker* reg_fit, float lambda_start, int clonal_restriction)
{
    //
    // Public entry function to be called from the CPU.
    //

    int iter = 0;
    CpuStep_t *StepP;

    float reg_lambda_min = FLT_MIN;
    float reg_lambda = 0.0001;
    unsigned int PartialDeriv_mask = 0;
    unsigned int reg_mask = 0;
    unsigned int well_mask = 0;
    float hpmax = 1;
    float tshift_cache = -10.0;
    bkg.lev_mar_fit->lm_state.restrict_clonal = 0;
    int nreg_group = 0;

    // Timer object that is used to benchmark portions of the code
    Timer my_timer;

    bkg.lev_mar_fit->lm_state.InitializeLevMarFit(well_fit,reg_fit);

    if (well_fit != NULL)
        well_mask = well_fit->GetPartialDerivMask();
    if (reg_fit != NULL)
        reg_mask = reg_fit->GetPartialDerivMask();

    well_mask |= YERR | FVAL; // always fill in yerr and fval
    reg_mask |= YERR | FVAL; // always fill in yerr and fval

    // initialize fitting algorithm
    int numFit = 0;
    for (int i = 0; i < bkg.my_beads.numLBeads; i++)
    {
        bkg.lev_mar_fit->lm_state.lambda[i] = lambda_start;
        bkg.lev_mar_fit->lm_state.well_completed[i] = false;

        if (~bkg.lev_mar_fit->lm_state.well_ignored[i])
            bkg.lev_mar_fit->lm_state.fit_indicies[numFit++] = i;
    }

    // Copy lists to GPU
    cudaMemcpy( lambda_cuda, bkg.lev_mar_fit->lm_state.lambda, sizeof(float) * num_beads, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    cudaMemcpy( well_complete_cuda, bkg.lev_mar_fit->lm_state.well_completed, sizeof(bool) * num_beads, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();

    max_iter += max_reg_iter;

    memset(bkg.my_scratch.cur_xtflux_block, 0, sizeof(float)*num_pts*num_fb);

    // Initialize data that doens't change for this fit
    InitializeFit();

    // -- Iteration Loop --
    while ( (iter < max_iter) && (numFit != 0) )
    {
        reg_params eval_rp = bkg.my_regions.rp;

        float reg_error = 0.0;
        int reg_wells = 0;

        bool reg_proc = false;

        if ((reg_fit != NULL) && (well_fit != NULL)) 
        {
            if ((iter <= max_reg_iter) && ((iter & 1) == 1))
                reg_proc = true;
        }
	else
	    if (reg_fit != NULL)
	        reg_proc = true;

        // figure out what set of partial derivative terms need to be computed in this iteration
        if (reg_proc)
            PartialDeriv_mask = reg_mask;
        else
            PartialDeriv_mask = well_mask;

        int req_done = 0;

        // Initialize SBG and copy to GPU
        if (bkg.my_regions.rp.tshift != tshift_cache) {
            bkg.my_trace.GetShiftedBkg( bkg.my_regions.rp.tshift, sbg_host );
            tshift_cache = bkg.my_regions.rp.tshift;
        }
        cudaMemcpy( sbg_cuda, sbg_host, sizeof(float) * num_fb * num_pts, cudaMemcpyHostToDevice );

	// Need to look at....chances for error
        cudaMemcpy( params_nn_cuda, bkg.my_beads.params_nn, sizeof(bead_params) * num_beads, cudaMemcpyHostToDevice );
        cudaMemcpy( eval_params_cuda, params_nn_cuda, sizeof(bead_params) * num_beads, cudaMemcpyDeviceToDevice);
        cudaMemcpy( eval_rp_cuda, &eval_rp, sizeof(reg_params), cudaMemcpyHostToDevice );

        BuildActiveBeadList( reg_proc, false, false, false, true, nreg_group, reg_wells, reg_error );
        cudaMemcpy( active_bead_list_cuda, active_bead_list_host, sizeof(int) * active_bead_count, cudaMemcpyHostToDevice );
        CUDA_ERROR_CHECK();
	if (reg_proc) {
    		bkg.lev_mar_fit->lm_state.avg_resid = CalculateFitError();
	}
	
        // phase in clonality restriction from 0-mers on up if requested
        
        if ( clonal_restriction > 0 )
        {
            hpmax = (int) (iter / 4) + 2;

            if (hpmax > clonal_restriction)
                hpmax = clonal_restriction;

            bkg.lev_mar_fit->lm_state.restrict_clonal = hpmax - 0.5;
        }

        if ( verbose_cuda )
            printf("  --- [Reg: %d,%d] [Iter: %d/%d] Num Fit = %d --- \n", bkg.region->col, bkg.region->col, iter, max_iter, numFit );

        // --- PartialDeriv Phase ---

        my_timer.restart();

        // Build the active bead list for PartialDeriv calculations
        BuildActiveBeadList( reg_proc, false, false, false, false, nreg_group, reg_wells, reg_error );

        if ( verbose_cuda )
            printf("  > [%s] [PartialDeriv Step] Active = %d ",  (reg_proc?"R":"L"), active_bead_count );

        if ( active_bead_count > 0 )
        {
            // Copy active bead list to the GPU
            cudaMemcpy( active_bead_list_cuda, active_bead_list_host, sizeof(int) * active_bead_count, cudaMemcpyHostToDevice );
            CUDA_ERROR_CHECK();

            for ( current_step = 0; current_step < num_steps; current_step++ )
            {
                StepP = &step_list[current_step];

                if ((StepP->PartialDerivMask & PartialDeriv_mask) == 0)
                    continue;

                if (StepP->PartialDerivMask == DFDGAIN)
                    DfdgainStep();

                else if (StepP->PartialDerivMask == DFDERR)
                    DfderrStep();

                else if (StepP->PartialDerivMask == YERR)
                    YerrStep();

                else if (StepP->PartialDerivMask == DFDTSH)
                    DfdtshStep();

                else if (StepP->diff != 0 || StepP->PartialDerivMask == FVAL)
                    FvalStep( StepP );
            }

            // Residual is needed back on the CPU to build the regional matrix
            cudaMemcpy( bkg.lev_mar_fit->lm_state.residual, residual_cuda, sizeof(float)*num_beads, cudaMemcpyDeviceToHost );
            cudaMemcpy( &eval_rp, eval_rp_cuda, sizeof(reg_params), cudaMemcpyDeviceToHost );
        }

        if ( verbose_cuda ) {
            printf( "(%f s)\n", my_timer.elapsed() );
	    for (int i=0; i<active_bead_count; ++i) {
                int ibd = active_bead_list_host[i];
	    	bead_params dbg_params;
	    	cudaMemcpy( &dbg_params, eval_params_cuda + ibd, sizeof(bead_params), cudaMemcpyDeviceToHost );

		printf("bead %d  residual: %3.3f\n", ibd, bkg.lev_mar_fit->lm_state.residual[ibd]);
		for (int j=0; j<NUMFB; ++j) {
			printf("%3.2f ", dbg_params.Ampl[j]);
		}        
		printf("\n");
	    }
	}
        // --- Build Matrices Phase ---

        // Build active bead list for the matrix building
        BuildActiveBeadList( reg_proc, false, true, true, false, nreg_group, reg_wells, reg_error );

        my_timer.restart();

        if ( verbose_cuda )
            printf("  > [%s] [ Matrix ] Active = %d ",  (reg_proc?"R":"L"), active_bead_count );

        if ( active_bead_count > 0 )
        {
            // Copy active bead list to the GPU
            cudaMemcpy( active_bead_list_cuda, active_bead_list_host, sizeof(int) * active_bead_count, cudaMemcpyHostToDevice );

            if ( !reg_proc )
            {
                // Construct per-well matrices
                BuildLocalMatrices( well_fit );
            }
            else
            {
                // Construct regional matrix
                BuildRegionalMatrix( reg_fit, nreg_group, reg_wells, reg_error );
            }
        }

        if ( verbose_cuda )
            printf( "(%f s)\n", my_timer.elapsed() );

        // --- Per-well Solve ---

        if ( !reg_proc )
        {
            int num_outputs = well_fit->getNumOutputs();

            // Copy to output list to the GPU, this is used to adjust the paramters from the solved delta
            cudaMemcpy( output_list_local_cuda, well_fit->getOuputList(), sizeof(delta_mat_output_line) * num_outputs, cudaMemcpyHostToDevice );

            // Initialize cont_proc to false for every bead
            cudaMemset( cont_proc_cuda, 0, sizeof(bool) * num_beads);

            // Check matrices for singularity
            SingularMatrixCheck( num_outputs );

            // Copy cont_proc to CPU to build active bead list
            cudaMemcpy( cont_proc_host, cont_proc_cuda, sizeof(bool) * num_beads, cudaMemcpyDeviceToHost );

            // Build active bead list for the per well solves
            BuildActiveBeadList( reg_proc, true, false, false, false, nreg_group, reg_wells, reg_error );

            if ( active_bead_count > 0  )
            {
                // Process a large chunk of beads on the GPU
                if ( active_bead_count > min_gpu_batch )
                {
                    if ( verbose_cuda )
                        printf("  > [L] [Solve-GPU] Active = %d ", active_bead_count);

                    my_timer.restart();

                    // Iterate till the GPU wells have converged
                    while ( active_bead_count > min_gpu_batch  )
                    {
                        // Copy active bead list to the GPU
                        cudaMemcpy( active_bead_list_cuda, active_bead_list_host, sizeof(int) * active_bead_count, cudaMemcpyHostToDevice );
                        CUDA_ERROR_CHECK();

                        // For every active well copy: RHS to Delta, JTJ to JTJ Lambda, params_nn to eval_params
                        CopyMatrices();

                        // Solve for delta on every active well
                        FactorizeAndSolveMatrices(num_outputs);

                        // Adjust and clamp parameters for every active well
                        AdjustParameters(num_outputs);

                        // Calculate the new residual for every active well
                        current_step = 0;
                        MultiFlowComputeCumulativeIncorporationSignalCuda(ival_cuda, eval_rp_cuda, true);
                        MultiFlowComputeIncorporationPlusBackgroundCuda(scratch_space_cuda, ival_cuda, sbg_cuda, eval_rp_cuda);
                        CalculateNewResidual();

                        // Compare residuals so we can adjust lambda and update bead_params for every active well
                        AdjustLambdaAndUpdateParameters( (reg_fit != NULL), iter, max_reg_iter, req_done, numFit, nreg_group );

                        // Sycn list of complete of cont_proc wells with the CPU
                        cudaMemcpy( bkg.lev_mar_fit->lm_state.well_completed, well_complete_cuda, sizeof(bool) * num_beads, cudaMemcpyDeviceToHost ); CUDA_ERROR_CHECK();
                        cudaMemcpy( cont_proc_host, cont_proc_cuda, sizeof(bool) * num_beads, cudaMemcpyDeviceToHost ); CUDA_ERROR_CHECK();

                        // Rebuild active bead list dropping off beads that have completed or stalled in the iteration loop
                        BuildActiveBeadList( reg_proc, true, false, false, false, nreg_group, reg_wells, reg_error );
                    }

                    if ( verbose_cuda )
                        printf("(%f s)\n", my_timer.elapsed());
                }

                // A small number of beads remain to be processed on the CPU
                if ( active_bead_count > 0 )
                {
                    if ( verbose_cuda )
                        printf("  > [L] [Solve-CPU] Active = %d ", active_bead_count);

                    my_timer.restart();

                    // For each remaining active bead...
                    for (int i = 0; i < active_bead_count; i++)
                    {
                        // Get bead index
                        int ibd = active_bead_list_host[i];

                        // Copy this well's matrix off the GPU
                        cudaMemcpy( rhs_host, &rhs_cuda[ibd*mat_dim], sizeof(double)*mat_dim, cudaMemcpyDeviceToHost ); CUDA_ERROR_CHECK();
                        cudaMemcpy( jtj_host, &jtj_cuda[ibd*mat_dim*mat_dim], sizeof(double)*mat_dim*mat_dim, cudaMemcpyDeviceToHost ); CUDA_ERROR_CHECK();

                        // And related parameters...
                        cudaMemcpy( &bkg.lev_mar_fit->lm_state.lambda[ibd], &lambda_cuda[ibd], sizeof(float), cudaMemcpyDeviceToHost ); CUDA_ERROR_CHECK();
                        cudaMemcpy( &bkg.my_beads.params_nn[ibd], &params_nn_cuda[ibd], sizeof(bead_params), cudaMemcpyDeviceToHost ); CUDA_ERROR_CHECK();
                        cudaMemcpy( &bkg.lev_mar_fit->lm_state.residual[ibd], &residual_cuda[ibd], sizeof(float), cudaMemcpyDeviceToHost ); CUDA_ERROR_CHECK();

                        // Copy to well_fit's in JTJ and RHS matrices
                        for ( int ii=0; ii<num_outputs; ii++ )
                        {
                            well_fit->rhs(ii) = rhs_host[ii];
                            for ( int jj=0; jj<num_outputs; jj++ )
                                (*(well_fit->jtj))(jj,ii) = jtj_host[jj*mat_dim+ii];
                        }

                        // Iteration flag
                        bool cont_proc = false;

                        // Iterate until limit is reached or answer is found
                        while ( !cont_proc )
                        {
                            float achg = 0.0;
                            bead_params eval_params = bkg.my_beads.params_nn[ibd];

                            // Solve and adjust delta
                            if ( well_fit->GetOutput( reinterpret_cast<float*>( &eval_params ), bkg.lev_mar_fit->lm_state.lambda[ibd] ) != LinearSolverException )
                            {
                                // Bounds check new parameters
                                params_ApplyLowerBound( &eval_params, &bkg.my_beads.params_low[ibd] );
                                params_ApplyUpperBound( &eval_params, &bkg.my_beads.params_high[ibd] );

                                // Check residual
                                MultiFlowComputeCumulativeIncorporationSignal( &eval_params, &eval_rp, ival_host,bkg.my_regions,bkg.my_scratch.cur_bead_block,bkg.time_c,bkg.my_flow, bkg.math_poiss);
                                MultiFlowComputeIncorporationPlusBackground( &fval_host[0], &eval_params, &eval_rp, ival_host, sbg_host,bkg.my_regions,bkg.my_scratch.cur_buffer_block,bkg.time_c,bkg.my_flow ,bkg.use_vectorization, bkg.my_scratch.bead_flow_t);
                                bkg.lev_mar_fit->lm_state.ApplyClonalRestriction(&fval_host[0], &eval_params,bkg.time_c.npts);

                                float res = 0;
                                float scale = 0;
				float mean_res = 0;
                                for (int fnum=0; fnum < num_fb; fnum++)
                                {
                                    FG_BUFFER_TYPE *pfg = &bkg.my_trace.fg_buffers[num_fb*num_pts*ibd+fnum*num_pts];
                                    float eval;
                                    int emndx = eval_params.WhichEmphasis[fnum];
                                    float *em = bkg.emphasis_data.EmphasisVectorByHomopolymer[emndx];
				    float rerr = 0;
                                    scale += bkg.emphasis_data.EmphasisScale[emndx];
    

                                    for (int ii=0;ii<num_pts;ii++)
                                    {
                                        eval = ( float(pfg[ii]) - fval_host[ii+fnum*num_pts] );

					rerr += eval*eval;
					eval = eval*em[ii];
                                        res += eval*eval;
                                    }


				    eval_params.rerr[fnum] = sqrt(rerr/(float)num_pts);
				    mean_res += eval_params.rerr[fnum];
                                }
                                res = sqrt(res/scale);
				mean_res /= num_fb;
                    
				for (int fnum=0;fnum < num_fb;fnum++)
                        		eval_params.rerr[fnum] /= mean_res;

                                float achg = 0;

                                if ( res < bkg.lev_mar_fit->lm_state.residual[ibd] )
                                {
                                    for (int fb = 0; fb < num_fb; fb++)
                                    {
                                        float chg = fabs(bkg.my_beads.params_nn[ibd].Ampl[fb]*bkg.my_beads.params_nn[ibd].Copies - eval_params.Ampl[fb]*eval_params.Copies);
                                        if (chg > achg)
                                            achg = chg;
                                    }

                                    bkg.my_beads.params_nn[ibd] = eval_params;
                                    bkg.lev_mar_fit->lm_state.lambda[ibd] *= 0.10;

                                    if ( bkg.lev_mar_fit->lm_state.lambda[ibd] < FLT_MIN )
                                        bkg.lev_mar_fit->lm_state.lambda[ibd] = FLT_MIN;

                                    bkg.lev_mar_fit->lm_state.residual[ibd] = res;
                                    cont_proc = true;
                                }
                                else
                                {
                                    bkg.lev_mar_fit->lm_state.lambda[ibd] *= 10.0;
                                }
                            }
                            else
                            {
                                // Singular matrix!
                                bkg.lev_mar_fit->lm_state.lambda[ibd] *= 10.0;
                            }

                            if ( (achg < 0.001) && (bkg.lev_mar_fit->lm_state.lambda[ibd] >= 1E+10) )
                            {
                                req_done++;
                                if ( (reg_fit == NULL) || (iter > max_reg_iter) )
                                {
                                    bkg.lev_mar_fit->lm_state.well_completed[ibd] = true;
                                    numFit--;
                                    cont_proc = true;
                                }
                            }

                            if (((reg_fit != NULL) && (iter <= max_reg_iter)) && (bkg.lev_mar_fit->lm_state.lambda[ibd] >= 1E+8))
                                cont_proc = true;
                        }

                        // Sync the updated list of completed wells, lambda, and residual with the GPU
                        cudaMemcpy( &well_complete_cuda[ibd], &bkg.lev_mar_fit->lm_state.well_completed[ibd], sizeof(bool), cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
                        cudaMemcpy( &lambda_cuda[ibd],  &bkg.lev_mar_fit->lm_state.lambda[ibd], sizeof(float), cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
                        cudaMemcpy( &params_nn_cuda[ibd], &bkg.my_beads.params_nn[ibd], sizeof(bead_params), cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
                        cudaMemcpy( &residual_cuda[ibd], &bkg.lev_mar_fit->lm_state.residual[ibd], sizeof(float), cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
                    }

                    if ( verbose_cuda )
                        printf( "(%f s)\n", my_timer.elapsed() );
                }
            }
        }

        // --- Regional Solve ---

        if ( reg_proc )
        {
            // Rebuild active bead list with regional considerations
            BuildActiveBeadList( reg_proc, false, true, false, false, nreg_group, reg_wells, reg_error );

            if ( reg_wells > 10 )
            {
                // Copy active list to the GPU
                cudaMemcpy( active_bead_list_cuda, active_bead_list_host, sizeof(int) * active_bead_count, cudaMemcpyHostToDevice );
                CUDA_ERROR_CHECK();

                bool cont_proc = false;

                // Copy the regional JTJ and RHS to CPU
                cudaMemcpy( jtj_host, jtj_cuda, sizeof(double)*mat_dim*mat_dim, cudaMemcpyDeviceToHost );
                cudaMemcpy( rhs_host, rhs_cuda, sizeof(double)*mat_dim, cudaMemcpyDeviceToHost );

                // Move to reg_fit's JTJ and RHS
                int num_outputs = reg_fit->getNumOutputs();
                for (int ii = 0; ii < num_outputs ; ii++)
                {
                    reg_fit->rhs(ii) = rhs_host[ii];
                    for (int jj = 0; jj < num_outputs; jj++)
                        (*(reg_fit->jtj))(jj,ii) = jtj_host[ jj*mat_dim + ii ];
                }

                my_timer.restart();

                while ( !cont_proc )
                {
                    if ( verbose_cuda )
                        printf( "  > [R] [ Solve  ] Active = %d ", active_bead_count );

                    reg_params eval_rp = bkg.my_regions.rp;
                    memset(eval_rp.Ampl,0,sizeof(eval_rp.Ampl));
                    eval_rp.R = 0.0;
                    eval_rp.Copies = 0.0;

                    // Solve
                    if ( reg_fit->GetOutput( reinterpret_cast<float*>( &eval_rp ), reg_lambda ) != LinearSolverException )
                    {
                        // Clamp parameters
                        if (verbose_cuda)
                            printf("t_mid_nuc: %f, tshift: %f\n", eval_rp.nuc_shape.t_mid_nuc, eval_rp.tshift);

                        reg_params_ApplyLowerBound( &eval_rp, &bkg.my_regions.rp_low );
                        reg_params_ApplyUpperBound( &eval_rp, &bkg.my_regions.rp_high );

                        // make a copy so we can modify it to null changes in new_rp that will
                        // we push into individual bead parameters
                        reg_params new_rp = eval_rp;
                        cudaMemcpy( new_rp_cuda, &eval_rp, sizeof(reg_params), cudaMemcpyHostToDevice );

                        // Get updated shifted background
                        bkg.my_trace.GetShiftedBkg( new_rp.tshift, sbg_host );
                        tshift_cache = new_rp.tshift;
                        cudaMemcpy( sbg_cuda, sbg_host, sizeof(float)*num_pts*num_fb, cudaMemcpyHostToDevice );

                        // Calculate regional residual sum error
                        float new_reg_error = CalculateRegionalResidual();
                        if (verbose_cuda)
                            printf("regional residual error: %f, NumFit: %d\n", new_reg_error, active_bead_count);

                        // are the new parameters better?
                        if ( new_reg_error < reg_error )
                        {

                            // Update regional parameters
                            bkg.my_regions.rp = new_rp;

                            // Re-calculate current parameter values for each bead as necessary
                            AdjustRegionalParameters( params_nn_cuda, new_rp_cuda );

                            // Adjust lambda
                            reg_lambda /= 10.0;
                            if (reg_lambda < FLT_MIN)
                                reg_lambda = FLT_MIN;

                            if (reg_lambda < reg_lambda_min)
                                reg_lambda = reg_lambda_min;

                            cont_proc = true;
                        }
                        else
                        {
                            // It's not better...
                            reg_lambda *= 10.0;
                            if ( reg_lambda > 1000000000 )
                                cont_proc = true;
                        }
                    }
                    else
                    {
                        // Singular matrix
                        reg_lambda *= 10.0;
                        if (reg_lambda > 1000000000)
                            cont_proc = true;
                    }

                }

                if ( verbose_cuda )
                    printf("(%f s)\n", my_timer.elapsed());

                nreg_group++;
                nreg_group = nreg_group % bkg.lev_mar_fit->lm_state.num_region_groups;
            }
        }

        // if more than 1/2 the beads aren't improving any longer, stop trying to do the region-wide fit

        if ( (iter <= max_reg_iter) && (reg_fit != NULL) && (((float) req_done / numFit) > 0.5) )
            iter = max_reg_iter + 1;
        else
            iter++;

        // Final iteration outputs
        cudaMemcpy( bkg.my_beads.params_nn, params_nn_cuda, sizeof(bead_params) * num_beads, cudaMemcpyDeviceToHost );
        cudaMemcpy( bkg.lev_mar_fit->lm_state.lambda, lambda_cuda, sizeof(float) * num_beads, cudaMemcpyDeviceToHost );
        cudaMemcpy( bkg.lev_mar_fit->lm_state.residual, residual_cuda, sizeof(float) * num_beads, cudaMemcpyDeviceToHost );

	if (reg_proc)
    //@TODO:  Mohit, please check the mask state is in fact set correctly for the GPU as I suspect strongly it is not
    // This argues that we should be exporting this whole morass to CPU for spare beads
    // and we're doing the whole bead list here???
            IdentifyParameters(bkg.my_beads,bkg.my_regions,bkg.lev_mar_fit->lm_state.well_mask,bkg.lev_mar_fit->lm_state.reg_mask);
    }

    bkg.lev_mar_fit->lm_state.avg_resid = 0.0;
    for (int i = 0; i < bkg.my_beads.numLBeads; i++)
        bkg.lev_mar_fit->lm_state.avg_resid += bkg.lev_mar_fit->lm_state.residual[i];

    bkg.lev_mar_fit->lm_state.avg_resid /= bkg.my_beads.numLBeads;

    printf(".");
    fflush(stdout);
    bkg.lev_mar_fit->lm_state.restrict_clonal = 0;
 
    return (iter);
}


// New Routines for v7

void BkgModelCuda::FitAmplitudePerFlow() {

    ClearMultiFlowMembers();

    AllocateSingleFlowMembers();

    // assuming all fields are floats in both the structs....1 is subtracted dsince fit_d is false right
    // now and there dmult is not fitted for krate...ugly hack for now
    numSingleFitParams = ampParams + krateParams;

    // initialize max and min values for SingleFlowFit and SingleFlowFitKrate bead_params
    float pbound[ampParams];
       
    pbound[0] = MINAMPL;
    cudaMemcpy(singleFlowFitParamMin_cuda, &pbound[0], sizeof(float)*ampParams, 
					cudaMemcpyHostToDevice);
    pbound[0] = MAX_HPLEN-1;
    cudaMemcpy(singleFlowFitParamMax_cuda, &pbound[0], sizeof(float)*ampParams, 
					cudaMemcpyHostToDevice);

    // calculate shifted background
    bkg.my_trace.GetShiftedBkg( bkg.my_regions.rp.tshift, sbg_host);
    cudaMemcpy(sbg_cuda, sbg_host, sizeof(float)*num_pts*num_fb, cudaMemcpyHostToDevice);
        
    // calculate active bead list 
    BuildActiveBeadListMinusIgnoredWells(false);
    cudaMemcpy(active_bead_list_cuda, active_bead_list_host, sizeof(int)*active_bead_count, cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK();

    InitializeFit();

    // calculate proton flux from neighbours
    CalcXtalkFlux();

    //DNTP calculation for each nucleotide
    DntpRiseModel dntpMod(num_pts, bkg.my_regions.rp.nuc_shape.C, bkg.time_c.frameNumber, ISIG_SUB_STEPS_SINGLE_FLOW);
    float nucRise[num_pts*NUMNUC*ISIG_SUB_STEPS_SINGLE_FLOW];
    int i_start[NUMNUC];
 
    for (int NucID=0;NucID<NUMNUC;NucID++)
    {
        float t_mid_nuc = bkg.my_regions.rp.nuc_shape.t_mid_nuc + bkg.my_regions.rp.nuc_shape.t_mid_nuc_delay[NucID]* (bkg.my_regions.rp.nuc_shape.t_mid_nuc-VALVEOPENFRAME) / TZERODELAYMAGICSCALE;
        i_start[NucID]=dntpMod.CalcCDntpTop(
            &nucRise[NucID*num_pts*ISIG_SUB_STEPS_SINGLE_FLOW],
            t_mid_nuc,
            bkg.my_regions.rp.nuc_shape.sigma * bkg.my_regions.rp.nuc_shape.sigma_mult[NucID]);
    }

    cudaMemcpy(c_dntp_top_pc_cuda, &nucRise[0], 
           sizeof(float)*num_pts*NUMNUC*ISIG_SUB_STEPS_SINGLE_FLOW, cudaMemcpyHostToDevice); CUDA_ERROR_CHECK();
    cudaMemcpy(i_start_cuda, &i_start[0], sizeof(int)*NUMNUC, cudaMemcpyHostToDevice); CUDA_ERROR_CHECK();
    // end DNTP calculation
   

    cudaMemcpy(params_nn_cuda, bkg.my_beads.params_nn, sizeof(bead_params)*num_beads, cudaMemcpyHostToDevice); CUDA_ERROR_CHECK();
    cudaMemcpy(eval_rp_cuda, &(bkg.my_regions.rp), sizeof(reg_params), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK();

    DetermineSingleFlowFitType();
    
    // Set well/region parameters for each flow for a live bead
    SetWellRegionParams();
    SingleFlowBkgCorrect();
    SetWeightVector();

    Fit();
   
    ClearSingleFlowMembers(); 
}

void BkgModelCuda::ClearMultiFlowMembers() {

    // Free CPU page locked arrays
    cudaFreeHost( ival_host );
    cudaFreeHost( fval_host );
    cudaFreeHost( sbg_slope_host );
    cudaFreeHost( jtj_host );
    cudaFreeHost( rhs_host );
    cudaFreeHost( fit_flag_host );
    cudaFreeHost( req_flag_host );

    // Free GPU parameter arrays
    //cudaFree( eval_params_cuda );
    cudaFree( new_rp_cuda ); ;

    // Free GPU memory (I/O)
    cudaFree( scratch_space_cuda );
    cudaFree( ival_cuda );
    cudaFree( sbg_slope_cuda );
    cudaFree( ivtmp_cuda );
    cudaFree( well_complete_cuda );
    cudaFree( fit_flag_cuda );
    cudaFree( req_flag_cuda );

    // Free GPU memory (input data)
    cudaFree(residual_cuda);
    cudaFree(new_residual_cuda);

    // Free GPU memory (matrix steps)
    cudaFree(sum_cuda);

    // Free GPU memory (offset lists)
    cudaFree(output_list_local_cuda);

    cudaFree(lambda_cuda);
    lambda_cuda = NULL;

    cudaFree(jtj_lambda_cuda);
    jtj_lambda_cuda = NULL;

    cudaFree(delta_cuda);
    delta_cuda = NULL;

    cudaFree(jtj_cuda);
    jtj_cuda = NULL;
 
    cudaFree(rhs_cuda); 
    rhs_cuda = NULL;

    CUDA_ERROR_CHECK();
}

void BkgModelCuda::AllocateSingleFlowMembers() {
 
    ampParams = sizeof(BkgModSingleFlowFitParams)/sizeof(float);
    krateParams = (sizeof(BkgModSingleFlowFitKrateParams)/sizeof(float)) - 1;

    cudaMalloc((void**)&xtflux_cuda, sizeof(float)*num_beads*num_fb*num_pts); CUDA_ERROR_CHECK();
    cudaMemset(xtflux_cuda, 0, sizeof(float)*num_beads*num_fb*num_pts); CUDA_ERROR_CHECK();

    cudaMalloc((void**)&isKrateFit_cuda, sizeof(bool)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMemset(isKrateFit_cuda, 0, sizeof(bool) * num_fb * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&singleFlowFitParams_cuda, sizeof(float)*ampParams*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&singleFlowFitKrateParams_cuda, sizeof(float)*krateParams*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&singleFlowFitKrateParamMin_cuda, sizeof(float)*krateParams*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&singleFlowFitKrateParamMax_cuda, sizeof(float)*krateParams*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&singleFlowFitParamMin_cuda, sizeof(float)*ampParams); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&singleFlowFitParamMax_cuda, sizeof(float)*ampParams); CUDA_ERROR_CHECK();
   
    cudaMalloc((void**)&tauB_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&sens_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&SP_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &fgbuffers_float_cuda, sizeof(float) * num_fb * num_pts * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &fval_cuda, sizeof(float) * num_fb * num_pts * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &weight_cuda, sizeof(float) * num_fb * num_pts * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &wtScale_cuda, sizeof(float) * num_fb * num_beads); CUDA_ERROR_CHECK();
    cudaMalloc( (void**) &done_cnt_cuda, sizeof(int) * num_fb * num_beads); CUDA_ERROR_CHECK();
    cudaMemset(done_cnt_cuda, 1, sizeof(int) * num_fb * num_beads); CUDA_ERROR_CHECK();
    
    cudaMalloc( (void**) &lambda_cuda, sizeof(float) * num_fb * num_beads); CUDA_ERROR_CHECK();
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::ClearSingleFlowMembers() {
    cudaFree(xtflux_cuda);
    cudaFree(isKrateFit_cuda);
    cudaFree(singleFlowFitParams_cuda);
    cudaFree(singleFlowFitKrateParams_cuda);
    cudaFree(singleFlowFitKrateParamMin_cuda);
    cudaFree(singleFlowFitKrateParamMax_cuda);
    cudaFree(singleFlowFitParamMin_cuda);
    cudaFree(singleFlowFitParamMax_cuda);
    cudaFree(tauB_cuda);
    cudaFree(sens_cuda);
    cudaFree(SP_cuda);
    cudaFree(fval_cuda);
    cudaFree(fgbuffers_float_cuda);
    cudaFree(weight_cuda);
    cudaFree(wtScale_cuda);
    cudaFree(done_cnt_cuda);
}

void BkgModelCuda::AllocateBinarySearchArrays() {
    cudaMalloc((void**)&ac_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&ec_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&ap_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&ep_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&step_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK();
}

void BkgModelCuda::CalcXtalkFlux() {
      NewXtalk();
}

void BkgModelCuda::BinarySearchAmplitude(float min_step, bool restart) {

    AllocateBinarySearchArrays();

    memset(cont_proc_host, 0, sizeof(bool)*num_beads);
    cudaMemset(cont_proc_cuda, 0, sizeof(bool)*num_beads);
   
    BuildActiveBeadListMinusIgnoredWells(false);
    cudaMemcpy(active_bead_list_cuda, active_bead_list_host, sizeof(int)*active_bead_count, cudaMemcpyHostToDevice);

    InitializeFit();
    
    // copy active bead list to gpu
    cudaMemcpy( params_nn_cuda, bkg.my_beads.params_nn, sizeof(bead_params) * num_beads, cudaMemcpyHostToDevice );
    cudaMemcpy( eval_rp_cuda, &bkg.my_regions.rp, sizeof(reg_params), cudaMemcpyHostToDevice );
    
    InitializeArraysForBinarySearch(restart);
    cudaMemcpy( eval_params_cuda, params_nn_cuda, sizeof(bead_params) * num_beads, cudaMemcpyDeviceToDevice);


    bkg.my_trace.GetShiftedBkg( bkg.my_regions.rp.tshift, sbg_host);
    cudaMemcpy(sbg_cuda, sbg_host, sizeof(float)*num_pts*num_fb, cudaMemcpyHostToDevice);
    EvaluateAmplitudeFit(ac_cuda, ec_cuda);

    int iter = 0;
    int max_iter = 30;
    while (iter <= max_iter) {
   
        // figure out which direction to go in from here 
        UpdateAp();
        EvaluateAmplitudeFit(ap_cuda, ep_cuda);

        BinarySearchStepOne();
        EvaluateAmplitudeFit(ap_cuda, ep_cuda);

        BinarySearchStepTwo(min_step);

        cudaMemcpy(cont_proc_host, cont_proc_cuda, sizeof(bool)*num_beads, cudaMemcpyDeviceToHost);
        BuildActiveBeadListMinusIgnoredWells(true);
        cudaMemcpy(active_bead_list_cuda, active_bead_list_host, sizeof(int)*active_bead_count, cudaMemcpyHostToDevice);

        if (active_bead_count == 0)
            break;

        iter++;        
    }

    BuildActiveBeadListMinusIgnoredWells(false);

    if (verbose_cuda)
        printf("GPU: Active beads = %d\n", active_bead_count);

    cudaMemcpy(active_bead_list_cuda, active_bead_list_host, sizeof(int)*active_bead_count, cudaMemcpyHostToDevice);
    UpdateAmplitudeAfterBinarySearch();   
        
    cudaMemcpy( bkg.my_beads.params_nn, params_nn_cuda, sizeof(bead_params) * num_beads, cudaMemcpyDeviceToHost );

    ClearBinarySearchMemory();
}

void BkgModelCuda::EvaluateAmplitudeFit(float* amp, float* err) {
   
   UpdateAmplitudeForEvaluation(amp);

   CalculateFitErrorForBinarySearch(err); 
}
 
void BkgModelCuda::UpdateAmplitudeForEvaluation(float* amp) {
 
    const int beads_per_block = 8;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    UpdateAmplitudeForEvaluation_k <beads_per_block> <<<blocks, threads>>>(active_bead_list_cuda, amp, eval_params_cuda, num_beads, num_fb, active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::InitializeArraysForBinarySearch(bool restart) {

    const int beads_per_block = 8;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    InitializeArraysForBinarySearch_k <beads_per_block> <<<blocks, threads>>>(restart, params_nn_cuda, ac_cuda, step_cuda, 
       active_bead_list_cuda, num_beads, num_fb, active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::BuildActiveBeadListMinusIgnoredWells(bool check_cont_proc) {
    active_bead_count = 0;
    for (int ibd=0; ibd<num_beads; ++ibd) {
        if (bkg.lev_mar_fit->lm_state.well_ignored[ibd])
            continue;
 
        if (check_cont_proc && cont_proc_host[ibd])
            continue;

        active_bead_list_host[active_bead_count++] = ibd;
    }
}

float BkgModelCuda::CalculateFitError() {

    current_step = 0;
    MultiFlowComputeCumulativeIncorporationSignalCuda( ivtmp_cuda, eval_rp_cuda, true);
    MultiFlowComputeIncorporationPlusBackgroundCuda( scratch_space_cuda, ivtmp_cuda, sbg_cuda, eval_rp_cuda );

    // Calculate residual for each active bead
    cudaMemset( new_residual_cuda, 0, sizeof(float)*num_beads ); CUDA_ERROR_CHECK();
    CalculateNewRegionalResidual();

    // Copy
    std::vector<float> temp_residual(num_beads, 0);
    cudaMemcpy( &temp_residual[0], new_residual_cuda, sizeof(float)*num_beads, cudaMemcpyDeviceToHost );
    CUDA_ERROR_CHECK();

    float avg_error = 0;
    for ( int i = 0; i < num_beads; i++ )
        avg_error += temp_residual[i];

    return (avg_error/active_bead_count);
}

void BkgModelCuda::CalculateFitErrorForBinarySearch(float* err) {

    current_step = 0;
    MultiFlowComputeCumulativeIncorporationSignalCuda( ivtmp_cuda, eval_rp_cuda, true);
    MultiFlowComputeIncorporationPlusBackgroundCuda( scratch_space_cuda, ivtmp_cuda, sbg_cuda, eval_rp_cuda );
  
    // calculate error b/w current fit and actual data
    ErrorForBinarySearch(err);
}

void BkgModelCuda::ErrorForBinarySearch(float* err) {

    dim3 threads(num_pts);
    dim3 blocks(active_bead_count);

    int shared_mem = num_fb * num_pts * sizeof(float);
    ErrorForBinarySearchNew_k <<< blocks, threads, shared_mem >>> ( scratch_space_cuda, eval_params_cuda, EmphasisVectorByHomopolymer_cuda,
            active_bead_list_cuda, fg_buffers_cuda, err, num_fb, num_pts, num_steps);

    CUDA_ERROR_CHECK();
    
}

void BkgModelCuda::UpdateAp() {

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    UpdateAp_k <<< blocks, threads >>> ( ap_cuda, ac_cuda, active_bead_list_cuda, num_fb);
    CUDA_ERROR_CHECK();    
}

void BkgModelCuda::BinarySearchStepOne() {

    float min_a = 0.001; 
    float max_a = (MAX_HPLEN - 1) - 0.001;

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    BinarySearchStepOne_k <<< blocks, threads >>> ( ap_cuda, ac_cuda, ep_cuda, ec_cuda, step_cuda, 
              min_a, max_a, active_bead_list_cuda, num_fb);
    CUDA_ERROR_CHECK();       
}

void BkgModelCuda::BinarySearchStepTwo(float min_step) {

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    int shared_mem = sizeof(int)*num_fb;

    BinarySearchStepTwo_k <<< blocks, threads, shared_mem >>> ( ap_cuda, ac_cuda, ep_cuda, ec_cuda, step_cuda, 
              cont_proc_cuda, min_step, active_bead_list_cuda, num_fb);
    CUDA_ERROR_CHECK();       
}

void BkgModelCuda::UpdateAmplitudeAfterBinarySearch() {

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    UpdateAmplitudeAfterBinarySearch_k <<< blocks, threads >>> ( params_nn_cuda, ac_cuda, 
              active_bead_list_cuda, num_fb);
    CUDA_ERROR_CHECK();    
}

void BkgModelCuda::DetermineSingleFlowFitType() {

    const int beads_per_block = 8;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    DetermineSingleFlowFitType_k <beads_per_block> <<<blocks, threads>>> (bkg.relax_krate_constraint, 
        params_nn_cuda, isKrateFit_cuda, singleFlowFitKrateParamMin_cuda, singleFlowFitKrateParamMax_cuda, 
        active_bead_list_cuda, krateParams, num_fb, active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::SetWellRegionParams() {
    
    const int beads_per_block = 8;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    SetWellRegionParams_k <beads_per_block> <<<blocks, threads>>>(
        params_nn_cuda, eval_rp_cuda,  singleFlowFitParams_cuda, 
        singleFlowFitKrateParams_cuda, lambda_cuda, active_bead_list_cuda, 
        SP_cuda, sens_cuda, tauB_cuda, flow_ndx_map_cuda, 
        buff_flow_cuda, num_fb, num_pts, ampParams, krateParams, active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::SingleFlowBkgCorrect() {
   
    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    SingleFlowBkgCorrect_k<<<blocks, threads>>>(params_nn_cuda, eval_rp_cuda, sbg_cuda, xtflux_cuda, 
         dark_matter_compensator_cuda, fgbuffers_float_cuda, fg_buffers_cuda, delta_frame_cuda, tauB_cuda, 
         flow_ndx_map_cuda, buff_flow_cuda, active_bead_list_cuda, num_fb, num_pts);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::SetWeightVector() {

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    SetWeightVector_k<<<blocks, threads>>>(params_nn_cuda, EmphasisVectorByHomopolymer_cuda, weight_cuda, wtScale_cuda, 
          active_bead_list_cuda, num_fb, num_pts, num_ev);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::Fit() {


    int maxIter = NUMSINGLEFLOWITER;

    // temp allocations for GPU
    float* f1_new;
    float* k1_new;
    float* tmp_fval;
    float* jac_cuda;
    float* r1_cuda;
    float* r2_cuda;
    float* err_vect_cuda;

    cudaMalloc((void**)&f1_new, sizeof(float)*ampParams*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&k1_new, sizeof(float)*krateParams*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&tmp_fval, sizeof(float)*num_beads*num_fb*num_pts); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&err_vect_cuda, sizeof(float)*num_beads*num_fb*num_pts); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&r1_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&r2_cuda, sizeof(float)*num_beads*num_fb); CUDA_ERROR_CHECK(); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&jac_cuda, sizeof(float)*num_beads*num_fb*num_pts*numSingleFitParams); CUDA_ERROR_CHECK(); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&jtj_cuda, sizeof(double)*num_beads*num_fb*(ampParams*ampParams + krateParams*krateParams)); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&rhs_cuda, sizeof(double)*num_beads*num_fb*numSingleFitParams); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&jtj_lambda_cuda, sizeof(double)*num_beads*num_fb*(ampParams*ampParams + krateParams*krateParams)); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&delta_cuda, sizeof(double)*num_beads*num_fb*numSingleFitParams); CUDA_ERROR_CHECK();

    cudaMemset(fval_cuda, 0, sizeof(float)*num_beads*num_fb*num_pts);
    cudaMemset(tmp_fval, 0, sizeof(float)*num_beads*num_fb*num_pts);

    // Initialize the above matrices
    cudaMemset(jac_cuda, 0, sizeof(float)*num_beads*num_fb*num_pts*numSingleFitParams);
    cudaMemset(jtj_cuda, 0, sizeof(double)*num_beads*num_fb*(ampParams*ampParams + krateParams*krateParams));
    cudaMemset(rhs_cuda, 0, sizeof(double)*num_beads*num_fb*numSingleFitParams);
    cudaMemset(jtj_lambda_cuda, 0, sizeof(double)*num_beads*num_fb*(ampParams*ampParams + krateParams*krateParams));
    cudaMemset(delta_cuda, 0, sizeof(double)*num_beads*num_fb*numSingleFitParams);

    bool* flowsToUpdate_cuda;
    bool* iterDone_cuda;
    bool* flowDone_cuda;
    bool* beadIter_cuda;
    bool* beadIter_host;
   
    cudaMallocHost((void**)&beadIter_host, sizeof(bool)*num_beads); CUDA_ERROR_CHECK();
    cudaMemset(beadIter_host, 0, sizeof(bool)*num_beads);

    cudaMalloc((void**)&iterDone_cuda, sizeof(bool)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&flowDone_cuda, sizeof(bool)*num_beads*num_fb); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&beadIter_cuda, sizeof(bool)*num_beads); CUDA_ERROR_CHECK();
    cudaMalloc((void**)&flowsToUpdate_cuda, sizeof(bool)*num_beads*num_fb); CUDA_ERROR_CHECK();

    // Loop for fit iterations   
    // calculate active bead count
 
    int iter = 0;
    
    cudaMemset(flowDone_cuda, 0, sizeof(bool)*num_beads*num_fb);
    cudaMemset(iterDone_cuda, 0, sizeof(bool)*num_beads*num_fb);
    EvaluateFunc(fval_cuda, singleFlowFitParams_cuda, singleFlowFitKrateParams_cuda,
        flowDone_cuda, iterDone_cuda);
    
    /*float* fval_cpu = (float*)malloc(sizeof(float)*num_beads*num_fb*num_pts);
    cudaMemcpy(fval_cpu, fval_cuda, sizeof(float)*num_beads*num_fb*num_pts, 
                       cudaMemcpyDeviceToHost);

        for (int i=0; i<num_pts; ++i) {
            float y = (float)bkg.my_trace.fg_buffers[i];
            printf("%3.3f\n", y);
        }
        float* temp;
        for (int i=0; i<active_bead_count; ++i) {
           int ibd = active_bead_list_host[i];
           printf("Bead: %d\n", ibd);
           temp = fval_cpu + ibd*num_fb*num_pts;
           for (int j=0; j<num_fb; ++j) {
               printf("Flow: %d\n", j);
               for (int k=0; k<num_pts; ++k) {
                   printf("fval[%d] = %3.3f\n", k, temp[j*num_pts + k]);
               }
           }
        }*/


    // 1. Solve for bead_params
    // 2. adjust bead_params and check for nans
    // 3. evaluate and residual and check for done
  
    cudaMemset(cont_proc_cuda, 0, sizeof(bool)*num_beads);
    while (iter < maxIter) {
        if (active_bead_count == 0)
            break;

        //printf("Iteration = %d\n", iter);

        if (iter == 0) {
            CalcResidualForSingleFLowFit(fval_cuda, err_vect_cuda, r1_cuda, 
                flowDone_cuda);
        }

        

        cudaMemset(flowsToUpdate_cuda, 0, sizeof(bool)*num_beads*num_fb);
        cudaMemset(beadIter_cuda, 0, sizeof(bool)*num_beads);
    
        cudaMemcpy(f1_new, singleFlowFitParams_cuda, sizeof(float)*num_beads*
                               num_fb*ampParams, cudaMemcpyDeviceToDevice);
        cudaMemcpy(k1_new, singleFlowFitKrateParams_cuda, sizeof(float)*num_beads*
                       num_fb*krateParams, cudaMemcpyDeviceToDevice);
        // adjusting params and calling evaluate
        for (int i=0; i<numSingleFitParams; ++i) {
            
            // copy params to new params
            if (i == 0) {
                AdjustParamFitIter(false, i, f1_new, k1_new);
            }
            else {
                AdjustParamFitIter(true, (i-1), f1_new, k1_new);
            }

            // evaluate func using new params
            EvaluateFunc(tmp_fval, f1_new, k1_new, flowDone_cuda, iterDone_cuda);

            // compute jacobian
            ComputeJacobianSingleFlow(jac_cuda, tmp_fval, i);
            
            // copy params to new params
            if (i == 0) {
                AdjustParamBackFitIter(false, i, f1_new, k1_new);
            }
            else {
                AdjustParamBackFitIter(true, (i-1), f1_new, k1_new);
            }
        }

        ComputeJTJSingleFlow(jac_cuda, flowDone_cuda); 
        ComputeRHSSingleFlow(jac_cuda, err_vect_cuda, flowDone_cuda);

        // cont_proc till all flows for all active beads are done
        while (true) {
            cudaMemcpy(jtj_lambda_cuda, jtj_cuda, sizeof(double)*num_beads*num_fb*
                 (ampParams*ampParams + krateParams*krateParams), 
                 cudaMemcpyDeviceToDevice);
            ComputeLHSWithLambdaSingleFLow(flowDone_cuda, iterDone_cuda);

	    SolveSingleFlow(flowDone_cuda, iterDone_cuda);

	    AdjustAndClampParamsSingleFlow(f1_new, k1_new, flowDone_cuda, 
                iterDone_cuda);

	    EvaluateFunc(tmp_fval, f1_new, k1_new, flowDone_cuda, iterDone_cuda);

	    CalcResidualForSingleFLowFit(tmp_fval, err_vect_cuda, r2_cuda, 
                flowDone_cuda);

	    CheckForIterationCompletion(flowsToUpdate_cuda, beadIter_cuda, 
                flowDone_cuda, iterDone_cuda, r1_cuda, r2_cuda); 
           
	    // active bead list to be calculated before next iteration 
            cudaMemcpy(beadIter_host, beadIter_cuda, sizeof(bool)*num_beads, 
                cudaMemcpyDeviceToHost);
            CUDA_ERROR_CHECK();

	    if(ScanBeadArrayForCompletion(beadIter_host))
                break;
        }
 
        UpdateParamsAndFlowValsAfterEachIter(flowsToUpdate_cuda, flowDone_cuda, 
            tmp_fval, f1_new, k1_new);
        iter++;   
        
        cudaMemset(iterDone_cuda, 0, sizeof(bool)*num_beads*num_fb);
        
        /*double* delta_cpu = (double*)malloc(sizeof(double)*num_beads*num_fb*numSingleFitParams);
        cudaMemcpy(delta_cpu, delta_cuda, sizeof(double)*num_beads*num_fb*numSingleFitParams, 
                       cudaMemcpyDeviceToHost);
        double* temp;
        for (int i=0; i<active_bead_count; ++i) {
           int ibd = active_bead_list_host[i];
           printf("Bead: %d\n", ibd);
           temp = delta_cpu + ibd*num_fb*numSingleFitParams;
           for (int j=0; j<num_fb; ++j) {
               printf("Flow: %d\n", j);
               for (int k=0; k<numSingleFitParams; ++k) {
                   printf("delta[%d] = %3.3f\n", k, temp[j*numSingleFitParams + k]);
               }
           }
        }*/


        DoneTest(flowDone_cuda);
        cudaMemcpy(cont_proc_host, cont_proc_cuda, sizeof(bool)*num_beads, 
                               cudaMemcpyDeviceToHost); 
        CUDA_ERROR_CHECK();

        BuildActiveBeadListMinusIgnoredWells(true);
        if (active_bead_count == 0)
            break;
        cudaMemcpy(active_bead_list_cuda, active_bead_list_host, 
              sizeof(int)*active_bead_count, cudaMemcpyHostToDevice); 
        CUDA_ERROR_CHECK();

    }

    BuildActiveBeadListMinusIgnoredWells(false);
    cudaMemcpy(active_bead_list_cuda, active_bead_list_host, 
       sizeof(int)*active_bead_count, cudaMemcpyHostToDevice); CUDA_ERROR_CHECK();
    UpdateFinalParamsSingleFlow();
    cudaMemcpy(bkg.my_beads.params_nn, params_nn_cuda, sizeof(bead_params)*num_beads, 
        cudaMemcpyDeviceToHost);

    cudaMemset(flowDone_cuda, 0, sizeof(bool)*num_beads*num_fb);
    cudaMemset(iterDone_cuda, 0, sizeof(bool)*num_beads*num_fb);
    EvaluateFunc(fval_cuda, singleFlowFitParams_cuda, singleFlowFitKrateParams_cuda, 
        flowDone_cuda, iterDone_cuda);
    CalcResidualForSingleFLowFit(fval_cuda, err_vect_cuda, r1_cuda, flowDone_cuda);
  
    float* r1_host = (float*)malloc(sizeof(float)*num_beads*num_fb);
    cudaMemcpy(r1_host, r1_cuda, sizeof(float)*num_beads*num_fb, 
        cudaMemcpyDeviceToHost);
    float flow_res_mean[NUMFB] = {0.0};
    bkg.lev_mar_fit->lm_state.avg_resid = 0.0; 
    
    for (int i=0; i<num_beads; ++i) {
        if (bkg.lev_mar_fit->lm_state.well_ignored[i]) continue;
       
        float tot_res = 0.0;
	
        if (verbose_cuda)
            printf("Bead = %d \n", i);

        for (int fnum=0; fnum<num_fb; ++fnum) {
            tot_res += bkg.my_beads.params_nn[i].rerr[fnum] = r1_host[i*num_fb + fnum];
            flow_res_mean[fnum] += bkg.my_beads.params_nn[i].rerr[fnum];

            if (verbose_cuda)
                printf("Ampl = %3.3f kmult = %3.3f\n", bkg.my_beads.params_nn[i].Ampl[fnum], bkg.my_beads.params_nn[i].kmult[fnum]);
        }

        tot_res /= num_fb;
        bkg.lev_mar_fit->lm_state.avg_resid += tot_res; 
    }
 
    for (int fnum=0;fnum < NUMFB;fnum++)
        flow_res_mean[fnum] /= num_beads;

    for (int ibd = 0;ibd < num_beads;ibd++)
    {
        float tot_res = 0.0;

        for (int fnum=0;fnum < NUMFB;fnum++)
        {
            bkg.my_beads.params_nn[ibd].rerr[fnum] /= flow_res_mean[fnum];
            tot_res += bkg.my_beads.params_nn[ibd].rerr[fnum];
        }
        tot_res /= NUMFB;
    }

    bkg.lev_mar_fit->lm_state.avg_resid /= num_beads;

    // free cpu memory
    free(r1_host);

    // free gpu memory
    cudaFree(f1_new);
    cudaFree(k1_new);
    cudaFree(tmp_fval);
    cudaFree(jac_cuda);
    cudaFree(r1_cuda);
    cudaFree(r2_cuda);
    cudaFree(err_vect_cuda);
    cudaFree(iterDone_cuda);
    cudaFree(flowDone_cuda);
    cudaFree(flowsToUpdate_cuda);
    cudaFree(beadIter_cuda);
    cudaFreeHost(beadIter_host);
}


void BkgModelCuda::EvaluateFunc(float* fval, float* f1, float* k1, bool* flowDone,
    bool* iterDone) {
    
    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    EvaluateFunc_beadblocksNew_k <beads_per_block> <<<blocks, threads>>>(flowDone, iterDone, isKrateFit_cuda, 
        params_nn_cuda, eval_rp_cuda, f1, k1, fval, delta_frame_cuda, sens_cuda, 
        tauB_cuda, SP_cuda, c_dntp_top_pc_cuda, i_start_cuda, 
        flow_ndx_map_cuda, buff_flow_cuda, active_bead_list_cuda, 
        exp_approx_table_cuda, exp_approx_table_size, num_fb, num_pts, ampParams, 
        krateParams, active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::CalcResidualForSingleFLowFit(float* data, float* err_vec, float* r, bool* flowDone) {
    
    dim3 threads(num_pts);
    dim3 blocks(active_bead_count); 

    int shared_mem = num_fb * num_pts * sizeof(double);
    CalcResidualForSingleFLowFitNew_k <<<blocks, threads, shared_mem>>>(flowDone, fgbuffers_float_cuda, 
        data, err_vec, r, weight_cuda, wtScale_cuda, active_bead_list_cuda, num_fb, num_pts, 
        active_bead_count);

    CUDA_ERROR_CHECK();
}

void BkgModelCuda::ComputeJacobianSingleFlow(float* jac, float* tmp, int paramIdx) {
    
    const int beads_per_block = 8;
    dim3 threads(num_pts, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    ComputeJacobianSingleFlow_k <beads_per_block> <<<blocks, threads>>>(jac, tmp, fval_cuda, weight_cuda, 
         paramIdx, numSingleFitParams, active_bead_list_cuda, num_fb, num_pts, active_bead_count);
    CUDA_ERROR_CHECK();

}

void BkgModelCuda::ComputeJTJSingleFlow(float* jac, bool* flowDone) {
    
    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up
 
    ComputeJTJSingleFlow_k <beads_per_block> <<<blocks, threads>>>(jac, jtj_cuda, flowDone, 
        ampParams, krateParams, isKrateFit_cuda, active_bead_list_cuda, 
        num_fb, num_pts, active_bead_count);
    CUDA_ERROR_CHECK();

}

void BkgModelCuda::ComputeRHSSingleFlow(float* jac, float* err_vec, bool* flowDone) {

    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up
 
    ComputeRHSSingleFlow_k <beads_per_block> <<<blocks, threads>>>(rhs_cuda, jac, err_vec, flowDone, 
        ampParams, krateParams, isKrateFit_cuda, active_bead_list_cuda, 
        num_fb, num_pts, active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::ComputeLHSWithLambdaSingleFLow(bool* flowDone, bool* iterDone) {

    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up
 
    ComputeLHSWithLambdaSingleFLow_k <beads_per_block> <<<blocks, threads>>>(jtj_lambda_cuda, 
        lambda_cuda, flowDone, iterDone, ampParams, krateParams, isKrateFit_cuda, 
        active_bead_list_cuda, num_fb, num_pts, active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::SolveSingleFlow(bool* flowDone, bool* iterDone) {

    if (verbose_cuda)
        printf("Solve function\n");

    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up
 
    SolveSingleFlow_k <beads_per_block> <<<blocks, threads>>>(jtj_lambda_cuda, rhs_cuda, jtj_cuda, 
	delta_cuda, lambda_cuda, flowDone, iterDone, ampParams, krateParams, isKrateFit_cuda, 
        active_bead_list_cuda, num_fb, num_pts, active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::AdjustAndClampParamsSingleFlow(float* f1, float* k1, bool* flowDone, bool* iterDone) {

    if (verbose_cuda)
        printf("Adjust Params\n");

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    AdjustAndClampParamsSingleFlow_k<<<blocks, threads>>>(f1, k1, singleFlowFitParamMax_cuda, 
        singleFlowFitParamMin_cuda, singleFlowFitKrateParamMax_cuda, singleFlowFitKrateParamMin_cuda,
        delta_cuda, flowDone, iterDone, ampParams, krateParams, isKrateFit_cuda, 
        active_bead_list_cuda, num_fb, num_pts);
    CUDA_ERROR_CHECK();
}


void BkgModelCuda::AdjustParamFitIter(bool doKrate, int paramIdx, float* f1_new, float* k1_new) {
    
    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    AdjustParamFitIter_k<<<blocks, threads>>>(doKrate, paramIdx, f1_new, k1_new, active_bead_list_cuda, 
        num_fb, ampParams, krateParams);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::AdjustParamBackFitIter(bool doKrate, int paramIdx, float* f1_new, float* k1_new) {

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    AdjustParamBackFitIter_k<<<blocks, threads>>>(doKrate, paramIdx, f1_new, k1_new, singleFlowFitParams_cuda, 
        singleFlowFitKrateParams_cuda, active_bead_list_cuda, num_fb, ampParams, krateParams);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::CheckForIterationCompletion(bool* flowsToUpdate, bool* beadIter, 
    bool* flowDone, bool* iterDone, float* r1, float* r2) {

    if (verbose_cuda)
        printf("Checking for iteration completion\n");

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    int sharedMem = sizeof(int)*num_fb;

    CheckForIterationCompletion_k<<<blocks, threads, sharedMem>>>(isKrateFit_cuda, 
       flowsToUpdate, beadIter, flowDone, iterDone, r1, r2, lambda_cuda, 
       active_bead_list_cuda, num_fb);
    CUDA_ERROR_CHECK();
} 

bool BkgModelCuda::ScanBeadArrayForCompletion(bool* beadItr) {

    if (verbose_cuda)
        printf("Scanning bead array for bead iter completion\n");

    int beadCnt = 0;
    for (int i=0; i<active_bead_count; ++i) {
        int ibd = active_bead_list_host[i];
        if (beadItr[ibd])
            beadCnt++;
    }
    if (beadCnt == active_bead_count)
        return true;

    return false;
}

void BkgModelCuda::DoneTest(bool* flowDone) {

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    int sharedMem = sizeof(int)*num_fb;

    DoneTest_k<<<blocks, threads, sharedMem>>>(delta_cuda, isKrateFit_cuda, flowDone,
        cont_proc_cuda, active_bead_list_cuda, done_cnt_cuda, num_fb, ampParams, krateParams);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::UpdateFinalParamsSingleFlow() {
    
    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    UpdateFinalParamsSingleFlow_k<<<blocks, threads>>>(params_nn_cuda, singleFlowFitParams_cuda, 
        singleFlowFitKrateParams_cuda, EmphasisVectorByHomopolymer_cuda, weight_cuda, wtScale_cuda, isKrateFit_cuda, 
        active_bead_list_cuda, num_fb, num_pts, num_ev, ampParams, krateParams);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::UpdateParamsAndFlowValsAfterEachIter(bool* flowsToUpdate, 
    bool* flowDone, float* tmp_fval, float* f1, float* k1) {

    dim3 threads(num_fb);
    dim3 blocks(active_bead_count);

    UpdateParamsAndFlowValsAfterEachIter_k<<<blocks, threads>>>(flowsToUpdate, 
       flowDone, fval_cuda, tmp_fval, f1, k1, singleFlowFitParams_cuda, 
       singleFlowFitKrateParams_cuda, active_bead_list_cuda, num_fb, num_pts, 
       ampParams, krateParams);
    CUDA_ERROR_CHECK();

}

void BkgModelCuda::CalcCDntpTop(bead_params* p, reg_params* rp) {
 
    //cudaMemset(c_dntp_top_pc_cuda, 0, sizeof(float)*num_beads*NUMNUC*
    //               num_pts*ISIG_SUB_STEPS_MULTI_FLOW);
    const int beads_per_block = 32;
    dim3 threads(beads_per_block, NUMNUC);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    SplineFitDntpTop_k<beads_per_block><<<blocks, threads>>>(c_dntp_top_pc_cuda,
        i_start_cuda, p, rp, frame_number_cuda, num_pts, active_bead_list_cuda,
        active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::BlueSolveBackgroundTrace(float* write_buffer, float* read_buffer) {
    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    BlueSolveBackgroundTrace_k <NUMFB, beads_per_block> <<< blocks, threads>>> (
            eval_params_cuda, eval_rp_cuda, write_buffer, read_buffer, flow_ndx_map_cuda, delta_frame_cuda, 
	    buff_flow_cuda, num_pts, active_bead_list_cuda, active_bead_count );

    CUDA_ERROR_CHECK();
}

void BkgModelCuda::PostBlueSolveBackgroundTraceSteps(float* write_buffer) {
    dim3 threads(num_pts);
    dim3 blocks(active_bead_count); 

    PostBlueSolveBackgroundTraceSteps_k <<< blocks, threads>>> (
            eval_params_cuda, eval_rp_cuda, write_buffer, dark_matter_compensator_cuda, 
            flow_ndx_map_cuda,  num_pts, num_fb, active_bead_list_cuda, active_bead_count);

    CUDA_ERROR_CHECK();
}

void BkgModelCuda::ObtainBackgroundCorrectedSignal() {
    dim3 threads(num_pts);
    dim3 blocks(active_bead_count); 

    ObtainBackgroundCorrectedSignal_k <<<blocks, threads>>> (correctedSignal_cuda, 
        fg_buffers_cuda, ival_cuda, num_pts, num_fb, active_bead_list_cuda, active_bead_count);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::RedSolveHydrogenFlowInWellAndAdjustedForGain(float* write_buffer,
    float* read_buffer) {

    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    cudaMemset(write_buffer, 0, sizeof(float)*num_pts*num_fb*num_beads);
    RedSolveHydrogenFlowInWellAndAdjustedForGain_k <NUMFB, beads_per_block> <<< blocks, threads >>> (
            eval_params_cuda, eval_rp_cuda, write_buffer, read_buffer, i_start_cuda, flow_ndx_map_cuda, 
            delta_frame_cuda, buff_flow_cuda, num_pts, active_bead_list_cuda, active_bead_count );

    CUDA_ERROR_CHECK();
}

void BkgModelCuda::ProjectOnAmplitudeVector(float* X, float* Y) {
    dim3 threads(num_fb);
    dim3 blocks(active_bead_count); 

    ProjectOnAmplitudeVector_k <<< blocks, threads>>> (
            projectionAmp_cuda, X, Y, num_pts, num_fb, active_bead_list_cuda, 
            active_bead_count);

    CUDA_ERROR_CHECK();

}

void BkgModelCuda::UpdateProjectionAmplitude(bead_params* p) {
    dim3 threads(num_fb);
    dim3 blocks(active_bead_count); 

    UpdateProjectionAmplitude_k <<< blocks, threads>>> (
            p, projectionAmp_cuda, num_fb, active_bead_list_cuda, 
            active_bead_count);

    CUDA_ERROR_CHECK();


}

void BkgModelCuda::ComputeCumulativeIncorporationHydrogensForProjection(float* ival, 
    reg_params* rp)
{
    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    int smem_size = num_fb * num_pts * beads_per_block * sizeof(float);

    ComputeCumulativeIncorporationHydrogensForProjection_k <beads_per_block> <<< blocks, threads, smem_size >>> 
        (eval_params_cuda, rp, ival, projectionAmp_cuda, flow_ndx_map_cuda, delta_frame_cuda, 
        buff_flow_cuda, num_pts, num_fb, active_bead_list_cuda, 
        active_bead_count, exp_approx_table_cuda, exp_approx_table_size, 
        c_dntp_top_pc_cuda, i_start_cuda);

    CUDA_ERROR_CHECK();
}


void BkgModelCuda::InitializeProjectionSearchAmplitude() {
    dim3 threads(num_fb);
    dim3 blocks(active_bead_count); 

    InitializeProjectionSearchAmplitude_k <<< blocks, threads>>> (
            projectionAmp_cuda, num_fb, active_bead_list_cuda, 
            active_bead_count);

    CUDA_ERROR_CHECK();
}


void BkgModelCuda::AllocateMemoryForProjectionSearch() {
    cudaMalloc((void**)&correctedSignal_cuda, sizeof(float)*num_pts*num_fb*num_beads);
    cudaMalloc((void**)&model_trace_cuda, sizeof(float)*num_pts*num_fb*num_beads);
    cudaMalloc((void**)&projectionAmp_cuda, sizeof(float)*num_fb*num_beads);    
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::ClearMemoryForProjectionSearch() {
    cudaFree(correctedSignal_cuda);
    cudaFree(model_trace_cuda);
    cudaFree(projectionAmp_cuda);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::ProjectionSearch() {
   
    AllocateMemoryForProjectionSearch();
   
    BuildActiveBeadListMinusIgnoredWells(false);
    cudaMemcpy(active_bead_list_cuda, active_bead_list_host, sizeof(int)*active_bead_count, cudaMemcpyHostToDevice);

    InitializeFit();
    
    // copy active bead list to gpu
    cudaMemcpy( eval_rp_cuda, &bkg.my_regions.rp, sizeof(reg_params), cudaMemcpyHostToDevice );
    cudaMemcpy( eval_params_cuda, bkg.my_beads.params_nn, sizeof(bead_params) * num_beads, cudaMemcpyHostToDevice);


    bkg.my_trace.GetShiftedBkg( bkg.my_regions.rp.tshift,  sbg_host);
    cudaMemcpy(sbg_cuda, sbg_host, sizeof(float)*num_pts*num_fb, cudaMemcpyHostToDevice);

    BlueSolveBackgroundTrace(ival_cuda, sbg_cuda);
    PostBlueSolveBackgroundTraceSteps(ival_cuda);

    ObtainBackgroundCorrectedSignal();

    CalcCDntpTop(eval_params_cuda, eval_rp_cuda);

    InitializeProjectionSearchAmplitude();
    for (int projection_loop=0; projection_loop<2; projection_loop++) {
        ComputeCumulativeIncorporationHydrogensForProjection(ival_cuda, eval_rp_cuda);
        RedSolveHydrogenFlowInWellAndAdjustedForGain(model_trace_cuda, ival_cuda);
        ProjectOnAmplitudeVector(correctedSignal_cuda, model_trace_cuda);
    }

    UpdateProjectionAmplitude(eval_params_cuda);
    cudaMemcpy( bkg.my_beads.params_nn, eval_params_cuda, sizeof(bead_params) * num_beads, cudaMemcpyDeviceToHost );

    ClearMemoryForProjectionSearch();
}

/* New crosstalk routines */

void BkgModelCuda::NewXtalk() {

    AllocateMemoryForXtalk(bkg.xtalk_spec.nei_affected);

    // copy active bead list to gpu
    cudaMemcpy( eval_rp_cuda, &bkg.my_regions.rp, sizeof(reg_params), cudaMemcpyHostToDevice );
    cudaMemcpy( eval_params_cuda, bkg.my_beads.params_nn, sizeof(bead_params) * num_beads, cudaMemcpyHostToDevice);

    ComputeXtalkContributionFromEveryBead();

    // Apply xtalk to each bead by its neighbours

    GenerateNeighbourMap();

    ComputeXtalkTraceForEveryBead(bkg.xtalk_spec.nei_affected);


    /*float* temp;
    float* x = (float*)malloc(sizeof(float)*num_pts*num_fb*num_beads);
    cudaMemcpy(x, xtflux_cuda, sizeof(float)*num_pts*num_fb*num_beads, cudaMemcpyDeviceToHost);
    for (int i=0; i<num_beads; ++i) {
        if (bkg.lev_mar_fit->lm_state.well_ignored[i])
            continue;
      temp = x + i*num_fb*num_pts;
      for (int j=0; j<num_fb*num_pts; ++j) {
          printf("%.3f ", temp[j]);
      }
      printf("\n");
    }
    free(x);*/
    
    ClearMemoryForXtalk();
}

void BkgModelCuda::AllocateMemoryForXtalk(int neigs) {
    cudaMalloc((void**)&write_buffer_cuda, sizeof(float)*num_pts*num_fb*num_beads);
    cudaMalloc((void**)&incorporation_trace_cuda, sizeof(float)*num_pts*num_fb*num_beads);
    cudaMalloc((void**)&xtalk_nei_trace_cuda, sizeof(float)*num_pts*num_fb*num_beads*neigs);
    cudaMalloc((void**)&numNeisPerBead_cuda, sizeof(int)*num_beads);
    cudaMalloc((void**)&NeiMapPerBead_cuda, sizeof(int)*num_beads*neigs*2); // bead index and its neighbour id
    
    cudaMemset(incorporation_trace_cuda, 0, sizeof(float)*num_fb*num_pts*num_beads);
    cudaMemset(xtalk_nei_trace_cuda, 0, sizeof(float)*num_fb*num_pts*num_beads*neigs);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::ClearMemoryForXtalk() {
    cudaFree(write_buffer_cuda);
    cudaFree(incorporation_trace_cuda);
    cudaFree(xtalk_nei_trace_cuda);
    cudaFree(numNeisPerBead_cuda);
    cudaFree(NeiMapPerBead_cuda);
}


void BkgModelCuda::ComputeXtalkContributionFromEveryBead() {
    int numNeis = bkg.xtalk_spec.nei_affected;

    MultiFlowComputeCumulativeIncorporationSignalCuda(incorporation_trace_cuda, eval_rp_cuda, true);
    
    float* model_trace = NULL;
    for (int i=0; i<numNeis; ++i) {
      if (bkg.xtalk_spec.multiplier[i] > 0) {
        model_trace = xtalk_nei_trace_cuda + i*num_pts*num_fb*num_beads;
        cudaMemcpy(write_buffer_cuda, incorporation_trace_cuda, 
            sizeof(float)*num_pts*num_fb*num_beads, cudaMemcpyDeviceToDevice);
        RedHydrogenForXtalk(model_trace, write_buffer_cuda, bkg.xtalk_spec.tau_top[i]);
        DiminishIncorporationTraceForXtalk(write_buffer_cuda, model_trace);        
        RedHydrogenForXtalk(model_trace, write_buffer_cuda, bkg.xtalk_spec.tau_fluid[i]);
        ApplyXtalkMultiplier(model_trace, bkg.xtalk_spec.multiplier[i]);        
      }
    }    
}

void BkgModelCuda::GenerateNeighbourMap() {
  if (bkg.my_beads.ndx_map != NULL)
  {
    int neis = bkg.xtalk_spec.nei_affected;

    int* numNeisPerBead = (int*)malloc(sizeof(int)*num_beads);
    int* NeiMapPerBead = (int*)malloc(sizeof(int)*num_beads*neis*2); // bead index and its neighbour index
    memset(numNeisPerBead, 0, sizeof(int)*num_beads); 
    memset(NeiMapPerBead, 0, sizeof(int)*num_beads*neis*2); 

    int nn_ndx,cx,cy,ncx,ncy;
    int* neiPtr = NULL;
    for (int ibd=0; ibd<num_beads; ++ibd) {

      if (bkg.lev_mar_fit->lm_state.well_ignored[ibd]) continue;

      cx = bkg.my_beads.params_nn[ibd].x;
      cy = bkg.my_beads.params_nn[ibd].y;
      neiPtr = NeiMapPerBead + ibd*neis*2;

      // Iterate over the number of neighbors, accumulating hydrogen ions
      for (int nei_idx=0; nei_idx<neis; nei_idx++)
      {
	// phase for hex-packed
        if (!bkg.xtalk_spec.hex_packed)
          CrossTalkSpecification::NeighborByGridPhase(ncx,ncy,cx,cy,
	      bkg.xtalk_spec.cx[nei_idx],bkg.xtalk_spec.cy[nei_idx], 0); 
	else
          CrossTalkSpecification::NeighborByGridPhase(ncx,ncy,cx,cy,
	      bkg.xtalk_spec.cx[nei_idx],bkg.xtalk_spec.cy[nei_idx], (bkg.region->row+cy+1) % 2); // maybe????

	if ((ncx>-1) && (ncx <bkg.region->w) && (ncy>-1) && (ncy<bkg.region->h)) // neighbor within region
	{
	  if ((nn_ndx=bkg.my_beads.ndx_map[ncy*bkg.region->w+ncx])!=-1) // bead present
	  {
	    numNeisPerBead[ibd]++;
	    neiPtr[0] = nn_ndx;
	    neiPtr[1] = nei_idx;
	    neiPtr += 2;
	  }
	}
      }
    }
 
    cudaMemcpy(numNeisPerBead_cuda, numNeisPerBead, sizeof(int)*num_beads, cudaMemcpyHostToDevice); 
    cudaMemcpy(NeiMapPerBead_cuda, NeiMapPerBead, sizeof(int)*num_beads*neis*2, cudaMemcpyHostToDevice); 
    free(numNeisPerBead);
    free(NeiMapPerBead);
  }
}

void BkgModelCuda::ComputeXtalkTraceForEveryBead(int neis) {

    const int beads_per_block = 2;
    dim3 threads(beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up
  
    int smem_size = sizeof(float) * num_fb * num_pts * beads_per_block; 
    int dataSize = num_beads*num_fb*num_pts;
    ComputeXtalkTraceForEveryBead_k <beads_per_block> <<<blocks, threads, smem_size>>>(
        xtflux_cuda, xtalk_nei_trace_cuda, dataSize, neis, numNeisPerBead_cuda, NeiMapPerBead_cuda, 
        active_bead_list_cuda, active_bead_count, num_fb*num_pts);
    CUDA_ERROR_CHECK();
}

void BkgModelCuda::RedHydrogenForXtalk(float* write_buffer, float* read_buffer, float tau) {

    const int beads_per_block = 3;
    dim3 threads(num_fb, beads_per_block);
    dim3 blocks((active_bead_count + beads_per_block - 1) / beads_per_block); // Simple integer round up

    int smem_size = num_fb * num_pts * beads_per_block * sizeof(float);
    int flowFrameProduct = num_pts * num_fb;
    RedHydrogenForXtalk_k <NUMFB, beads_per_block> <<< blocks, threads, smem_size >>> (
        eval_params_cuda, eval_rp_cuda, write_buffer, read_buffer, tau, i_start_cuda, 
        delta_frame_cuda, flow_ndx_map_cuda, num_pts, active_bead_list_cuda, active_bead_count,
        flowFrameProduct);

    CUDA_ERROR_CHECK();
}

void BkgModelCuda::DiminishIncorporationTraceForXtalk(float* write_buffer, float* read_buffer) {
    const int numThreads = 512;
    int dataSize = num_fb*num_pts*num_beads;
    dim3 threads(numThreads);
    dim3 blocks((dataSize%numThreads == 0) ? dataSize/numThreads 
                      : dataSize/numThreads + 1); 

    DiminishIncorporationTraceForXtalk_k <numThreads> <<< blocks, threads >>> (
        write_buffer, read_buffer, dataSize);

    CUDA_ERROR_CHECK();

}

void BkgModelCuda::ApplyXtalkMultiplier(float* write_buffer, float multiplier) {
    const int numThreads = 512;
    int dataSize = num_fb*num_pts*num_beads;
    dim3 threads(numThreads);
    dim3 blocks((dataSize%numThreads == 0) ? dataSize/numThreads 
                      : dataSize/numThreads + 1); 

    ApplyXtalkMultiplier_k <numThreads> <<< blocks, threads >>> (
        write_buffer, multiplier, dataSize);

    CUDA_ERROR_CHECK();
}
