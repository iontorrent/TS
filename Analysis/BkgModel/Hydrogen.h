/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef HYDROGEN_H
#define HYDROGEN_H

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "MathOptim.h"

#define FRAMESPERSEC 15.0
#define n_to_uM_conv    (0.000062f)


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

void DerivativeComputeCumulativeIncorporationHydrogens(float *ival_offset, float *da_offset, float *dk_offset, int npts, float *deltaFrameSeconds,
        float *nuc_rise_ptr, int SUB_STEPS, int my_start,
        Dual A, float SP,
        Dual kr, float kmax, float d,PoissonCDFApproxMemo *math_poiss);

// add using dual numbers
// combine diffeq and incorporation for direct RedTrace computation - probably should be in own module
// note repeated code?  Interior function call for "one step" may be needed.
void DerivativeRedTrace(float *red_out, float *ival_offset, float *da_offset, float *dk_offset,
                        int npts, float *deltaFrameSeconds, float *deltaFrame,
        float *nuc_rise_ptr, int SUB_STEPS, int my_start,
        Dual A, float SP,
        Dual kr, float kmax, float d,
                        float sens, float gain, float tauB,
                        PoissonCDFApproxMemo *math_poiss);        
        
#endif // HYDROGEN_H