/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef HYDROGEN_H
#define HYDROGEN_H

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "MathOptim.h"

//#define FRAMESPERSEC 15.0f
//#define n_to_uM_conv    (0.000062f)


// provide differential equation solving functions so everyone can play equally
void ComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrame,
                                             float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
                                             float A, float SP,
                                             float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss=NULL,bool do_simple=true);
// two options here
void SimplifyComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrame,
                                             float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
                                             float A, float SP,
                                             float kr, float kmax, float d, float molecules_to_micromolar_conversion,PoissonCDFApproxMemo *math_poiss=NULL);

void ComplexComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrame,
                                             float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
                                             float A, float SP,
                                             float kr, float kmax, float d, float molecules_to_micromolar_conversion,PoissonCDFApproxMemo *math_poiss=NULL);

void ParallelSimpleComputeCumulativeIncorporationHydrogens(float **ival_offset, int npts, float *deltaFrameSeconds,
        float **nuc_rise_ptr, int SUB_STEPS, int *my_start,
        float *A, float *SP,
        float *kr, float *kmax, float *d, float *molecules_to_micromolar_conversion,PoissonCDFApproxMemo *math_poiss);        


        
void SuperSimplifyComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrameSeconds,
        float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
        float A, float SP,
        float kr, float kmax, float d,float molecules_to_micromolar_conversion,PoissonCDFApproxMemo *math_poiss);

        
#endif // HYDROGEN_H