/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef HYDROGEN_H
#define HYDROGEN_H

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "MathOptim.h"

namespace MathModel {

// provide differential equation solving functions so everyone can play equally
void ComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, const float *deltaFrame,
                                             const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
                                             float A, float SP,
                                             float kr, float kmax, float d, float molecules_to_micromolar_conversion,
                                             PoissonCDFApproxMemo *math_poiss, int incorporationModelType );
// two options here
void SimplifyComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, const float *deltaFrame,
                                             const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
                                             float A, float SP,
                                             float kr, float kmax, float d, float molecules_to_micromolar_conversion,PoissonCDFApproxMemo *math_poiss=NULL);

void ComplexComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, const float *deltaFrame,
                                             const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
                                             float A, float SP,
                                             float kr, float kmax, float d, float molecules_to_micromolar_conversion,PoissonCDFApproxMemo *math_poiss=NULL);

void ReducedComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, const float *deltaFrame,
                                             const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
                                             float A, float SP,
                                             float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss=NULL);

void Reduced2ComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, const float *deltaFrame,
                                             const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
                                             float A, float SP,
                                             float kr, float kmax, float d, float molecules_to_micromolar_conversion,PoissonCDFApproxMemo *math_poiss=NULL);

void Reduced3ComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, const float *deltaFrame,
                                             const float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
                                             float A, float SP,
                                             float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss=NULL);

void ParallelSimpleComputeCumulativeIncorporationHydrogens(
        float **ival_offset, int npts, const float *deltaFrameSeconds,
        const float * const *nuc_rise_ptr, int SUB_STEPS, int *my_start,
        float *A, float *SP,
        float *kr, float *kmax, float *d, float *molecules_to_micromolar_conversion,
        PoissonCDFApproxMemo *math_poiss, int incorporationModelType);
        
void UnsignedParallelSimpleComputeCumulativeIncorporationHydrogens(
        float **ival_offset, int npts, const float *deltaFrameSeconds,
        const float * const *nuc_rise_ptr, int SUB_STEPS, int *my_start,
        float *A, float *SP,
        float *kr, float *kmax, float *d, float *molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss);


        
void SuperSimplifyComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, const float *deltaFrameSeconds,
        float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
        float A, float SP,
        float kr, float kmax, float d, float molecules_to_micromolar_conversion, PoissonCDFApproxMemo *math_poiss);

} // namespace
        
#endif // HYDROGEN_H
