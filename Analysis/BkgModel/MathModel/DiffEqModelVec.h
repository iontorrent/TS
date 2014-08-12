/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef DIFFEQMODELVEC_H 
#define DIFFEQMODELVEC_H 

namespace MathModel {

/* Vectorized Routine declarations */
 
void PurpleSolveTotalTrace_Vec( float **vb_out, float **blue_hydrogen, 
    float **red_hydrogen, int len, const float *deltaFrame, float *tauB, float *etbR, 
    float gain, int flow_block_size );

void RedSolveHydrogenFlowInWell_Vec(float * const *vb_out, const float * const *red_hydrogen, 
    int len, const float *deltaFrame, const float *tauB, int flow_block_size);

void BlueSolveBackgroundTrace_Vec(float **vb_out, float **blue_hydrogen, int len, 
    const float *deltaFrame, const float *tauB, const float *etbR, int flow_block_size);

} // namespace


#endif // DIFFEQMODELVEC_H
