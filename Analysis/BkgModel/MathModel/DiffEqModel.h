/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DIFFEQMODEL_H
#define DIFFEQMODEL_H

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "MathOptim.h"
#include "MathUtil.h"
#include "Hydrogen.h"


//#define FRAMESPERSEC 15.0f
//#define n_to_uM_conv    (0.000062f)

// we do this a lot
// expand a bead parameters for a block of flows
// this is used in MultiFlowModel
struct incorporation_params_block_flows {
    int NucID[NUMFB]; // technically redundant
    float SP[NUMFB];
    float sens[NUMFB];
    float d[NUMFB];
    float kr[NUMFB];
    float kmax[NUMFB];
    float C[NUMFB];
    float molecules_to_micromolar_conversion[NUMFB]; // cannot reasonably differ by flow(?)
    float *nuc_rise_ptr[NUMFB];
    float *ival_output[NUMFB];
};

struct buffer_params_block_flows {
    float etbR[NUMFB]; // buffering ratio of empty to bead
    float tauB[NUMFB]; // buffering of bead itself
};


void RedSolveHydrogenFlowInWell(float *vb_out, float *red_hydrogen, int len, int i_start,float *deltaFrame, float tauB);
void BlueSolveBackgroundTrace(float *vb_out, float *blue_hydrogen, int len, float *deltaFrame, float tauB, float etbR);
void NewBlueSolveBackgroundTrace (double *vb_out, const double *blue_hydrogen, int len, const double *deltaFrame, float tauB, float etbR);
void PurpleSolveTotalTrace(float *vb_out, float *blue_hydrogen, float *red_hydrogen, int len, float *deltaFrame, float tauB, float etbR);

// some fun utilities
void IntegrateRedFromRedTraceObserved (float *red_hydrogen, float *red_obs, int len, int i_start, float *deltaFrame, float tauB);
void IntegrateRedFromObservedTotalTrace ( float *red_hydrogen, float *purple_obs, float *blue_hydrogen,  int len, float *deltaFrame, float tauB, float etbR);

// compute the trace for a single flow
void RedTrace(float *red_out, float *ivalPtr, int npts,
              float *deltaFrameSeconds, float *deltaFrame, float *nuc_rise_ptr, int SUB_STEPS, int my_start,
              float C, float A, float SP, float kr, float kmax, float d, float molecules_to_micromolar_conversion, float sens, float gain, float tauB,
              PoissonCDFApproxMemo *math_poiss);
              
              
#endif // DIFFEQMODEL_H
