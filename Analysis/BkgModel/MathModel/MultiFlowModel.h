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

#include <math.h>
#include <stdlib.h>
#include <stdio.h>


// set up to "vectorize" model by pre-computing blocks of flow parameters
void FillBufferParamsBlockFlows(buffer_params_block_flows *my_buff, bead_params *p, reg_params *reg_p, int *flow_ndx_map, int *buff_flow);
void FillIncorporationParamsBlockFlows(incorporation_params_block_flows *my_inc, bead_params *p, reg_params *reg_p,int *flow_ndx_map, int *buff_flow);
void ApplyDarkMatter(float *fval,reg_params *reg_p, float *dark_matter_compensator, int *flow_ndx_map, int *buff_flow, int npts);
// 2nd-order background function with non-uniform bead well
void MultiFlowComputeIncorporationPlusBackground(float *fval,struct bead_params *p,struct reg_params *reg_p, float *ival, float *sbg, 
                                                           RegionTracker &my_regions, buffer_params_block_flows &cur_buffer_block, 
                                                           TimeCompression &time_c, flow_buffer_info &my_flow,
                                                           bool use_vectorization, int bead_flow_t);
void MultiFlowComputeCumulativeIncorporationSignal(struct bead_params *p,struct reg_params *reg_p, float *ivalPtr,
                                                             RegionTracker &my_regions, incorporation_params_block_flows &cur_bead_block, 
                                                             TimeCompression &time_c, flow_buffer_info &my_flow,  PoissonCDFApproxMemo *math_poiss );                                                           
void AccumulateSingleNeighborXtalkTrace(float *my_xtflux, bead_params *p, reg_params *reg_p, 
                                                  BeadScratchSpace &my_scratch, TimeCompression &time_c, RegionTracker &my_regions, 
                                                flow_buffer_info my_flow, PoissonCDFApproxMemo *math_poiss, bool use_vectorization,
                                                  float tau_top, float tau_bulk, float multiplier);
void MultiCorrectBeadBkg(float *block_signal_corrected, bead_params *p,
                         BeadScratchSpace &my_scratch, flow_buffer_info &my_flow, TimeCompression &time_c, RegionTracker &my_regions, float *sbg, bool use_vectorization);
void AccumulateSingleNeighborExcessHydrogen(float *my_xtflux, float *neighbor_signal, bead_params *p, reg_params *reg_p,
                                                  BeadScratchSpace &my_scratch, TimeCompression &time_c,
                                                  RegionTracker &my_regions, flow_buffer_info my_flow,
                                                   bool use_vectorization,
                                                  float tau_top, float tau_bulk, float multiplier);    
#endif // MULTIFLOWMODEL_H