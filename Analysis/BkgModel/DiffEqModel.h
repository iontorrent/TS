/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DIFFEQMODEL_H
#define DIFFEQMODEL_H

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "MathOptim.h"

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


// provide differential equation solving functions so everyone can play equally
void ComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrame, 
                                             float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C, 
                                             float A, float SP, 
                                             float kr, float kmax, float d, PoissonCDFApproxMemo *math_poiss=NULL,bool do_simple=true);
// two options here                                             
void SimplifyComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrame, 
                                             float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C, 
                                             float A, float SP, 
                                             float kr, float kmax, float d, PoissonCDFApproxMemo *math_poiss=NULL);    
                                           
void ComplexComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrame, 
                                             float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C, 
                                             float A, float SP, 
                                             float kr, float kmax, float d, PoissonCDFApproxMemo *math_poiss=NULL);
void ParallelSimpleComputeCumulativeIncorporationHydrogens(float **ival_offset, int npts, float *deltaFrameSeconds,
        float **nuc_rise_ptr, int SUB_STEPS, int *my_start, 
        float *A, float *SP,
        float *kr, float *kmax, float *d, PoissonCDFApproxMemo *math_poiss);        
void RedSolveHydrogenFlowInWell(float *vb_out, float *red_hydrogen, int len, int i_start,float *deltaFrame, float tauB);
void BlueSolveBackgroundTrace(float *vb_out, float *blue_hydrogen, int len, float *deltaFrame, float tauB, float etbR);
void PurpleSolveTotalTrace(float *vb_out, float *blue_hydrogen, float *red_hydrogen, int len, float *deltaFrame, float tauB, float etbR);

// Mark may go ahead and vectorize this as he likes
void MultiplyVectorByScalar(float *my_vec, float my_scalar, int len);
void MultiplyVectorByVector(float *my_vec, float *my_other_vec, int len);
void AddScaledVector(float *start_vec, float *my_vec, float my_scalar, int len);
float CheckVectorDiffSSQ(float *my_vec, float *my_other_vec, int len);
void CopyVector(float *destination_vec, float *my_vec, int len);
void AccumulateVector(float *destination_vec, float *my_vec, int len);         
void DiminishVector(float *destination_vec, float *my_vec, int len);

void CALC_PartialDeriv(float *p1,float *p2,int len,float dp);
void CALC_PartialDeriv_W_EMPHASIS(float *p1,float *p2,float *pe,int len,float dp,float pelen);
void CALC_PartialDeriv_W_EMPHASIS_LONG(float *p1,float *p2,float *pe,int len,float dp);
    
#endif // DIFFEQMODEL_H