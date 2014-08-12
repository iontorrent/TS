/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MULTIFLOWMODEL_H
#define MULTIFLOWMODEL_H

#include "BkgMagicDefines.h"
#include "TimeCompression.h"
#include "FlowBuffer.h"
#include "BeadParams.h"
#include "BeadTracker.h"
#include "BeadScratch.h"
#include "RegionParams.h"
#include "RegionTracker.h"
#include "DiffEqModel.h"
#include "DiffEqModelVec.h"
#include "MathOptim.h"

#include <vector>


namespace MathModel {

// set up to "vectorize" model by pre-computing blocks of flow parameters
void FillBufferParamsBlockFlows(
    buffer_params_block_flows *my_buff, BeadParams *p, const reg_params *reg_p, 
    const int *flow_ndx_map, int flow_block_start,
    int flow_block_size);
void FillIncorporationParamsBlockFlows(
    incorporation_params_block_flows *my_inc, BeadParams *p, reg_params *reg_p,
    const int *flow_ndx_map, int flow_block_start,
    int flow_block_size);
void ApplyDarkMatter(float *fval,const reg_params *reg_p, const std::vector<float>& dark_matter_compensator, 
    const int *flow_ndx_map, int npts,
    int flow_block_size);
void ApplyPCADarkMatter ( float *fval,BeadParams *p, const std::vector<float>& dark_matter_compensator, int npts,
    int flow_block_size);
void MultiFlowComputeTraceGivenIncorporationAndBackground(
    float *fval, BeadParams *p, const reg_params *reg_p, float *ival, float *sbg, 
    RegionTracker &my_regions, buffer_params_block_flows &cur_buffer_block, 
    const TimeCompression &time_c, const FlowBufferInfo &my_flow,
    bool use_vectorization, int bead_flow_t,
    int flow_block_size, int flow_block_start );

void MultiFlowComputeCumulativeIncorporationSignal(
    struct BeadParams *p,struct reg_params *reg_p, float *ivalPtr,
    NucStep &cache_step, incorporation_params_block_flows &cur_bead_block, 
    const TimeCompression &time_c, const FlowBufferInfo &my_flow,  PoissonCDFApproxMemo *math_poiss,
    int flow_block_size, int flow_block_start);

void MultiCorrectBeadBkg(
    float *block_signal_corrected, BeadParams *p,
    const BeadScratchSpace &my_scratch, 
    const buffer_params_block_flows &my_cur_buffer_block,
    const FlowBufferInfo &my_flow,
    const TimeCompression &time_c, const RegionTracker &my_regions, float *sbg, bool use_vectorization,
    int flow_block_size);

////
void AccumulateSingleNeighborExcessHydrogen(float *my_xtflux, float *neighbor_signal, BeadParams *p, reg_params *reg_p,
                                            BeadScratchSpace &my_scratch, 
                                            buffer_params_block_flows &my_cur_buffer_block,
                                            TimeCompression &time_c,
                                            RegionTracker &my_regions, const FlowBufferInfo & my_flow,
                                            bool use_vectorization,
                                            float tau_top, float tau_bulk, float multiplier,
                                            int flow_block_size, int flow_block_start );
                                            
void AccumulateSingleNeighborExcessHydrogenOneParameter ( float *my_xtflux, float *neighbor_signal,
                                                          BeadParams *p, reg_params *reg_p,
    BeadScratchSpace &my_scratch,
    buffer_params_block_flows &my_cur_buffer_block,
    TimeCompression &time_c,
    RegionTracker &my_regions, const FlowBufferInfo & my_flow,
    bool use_vectorization,
     float multiplier , bool rescale_flag,
     int flow_block_size, int flow_block_start );
     
void AccumulateSingleNeighborXtalkTrace(float *my_xtflux, BeadParams *p, reg_params *reg_p,
                                        BeadScratchSpace &my_scratch, 
                                        incorporation_params_block_flows & my_cur_bead_block,
                                        TimeCompression &time_c, RegionTracker &my_regions,
                                        const FlowBufferInfo & my_flow, PoissonCDFApproxMemo *math_poiss, bool use_vectorization,
                                        float tau_top, float tau_bulk, float multiplier,
                                        int flow_block_size, int flow_block_start );

} // namespace

#endif // MULTIFLOWMODEL_H
