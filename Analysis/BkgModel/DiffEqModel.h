/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DIFFEQMODEL_H
#define DIFFEQMODEL_H

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "MathOptim.h"
#include "MathUtil.h"
#include "Hydrogen.h"

#define FRAMESPERSEC 15.0
#define n_to_uM_conv    (0.000062f)

// we do this a lot
// expand a bead parameters for a block of flows
// this is used in MultiFlowModel
struct incorporation_params_block_flows {
    int NucID[NUMFB]; // technically redundant
    float SP[NUMFB] __attribute__ ((aligned (16)));
    float sens[NUMFB] __attribute__ ((aligned (16)));
    float d[NUMFB] __attribute__ ((aligned (16)));
    float kr[NUMFB] __attribute__ ((aligned (16)));
    float kmax[NUMFB] __attribute__ ((aligned (16)));
    float C[NUMFB] __attribute__ ((aligned (16)));
    float *nuc_rise_ptr[NUMFB];
    float *ival_output[NUMFB];
};

struct buffer_params_block_flows {
    float etbR[NUMFB] __attribute__ ((aligned (16))); // buffering ratio of empty to bead
    float tauB[NUMFB] __attribute__ ((aligned (16))); // buffering of bead itself
};


void RedSolveHydrogenFlowInWell(float *vb_out, float *red_hydrogen, int len, int i_start,float *deltaFrame, float tauB);
void BlueSolveBackgroundTrace(float *vb_out, float *blue_hydrogen, int len, float *deltaFrame, float tauB, float etbR);
void PurpleSolveTotalTrace(float *vb_out, float *blue_hydrogen, float *red_hydrogen, int len, float *deltaFrame, float tauB, float etbR);


#endif // DIFFEQMODEL_H