/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef DIFFEQMODELVEC_H 
#define DIFFEQMODELVEC_H 



/* Vectorized Routine declarations */
 
void PurpleSolveTotalTrace_Vec(int numfb, float **vb_out, float **blue_hydrogen, 
    float **red_hydrogen, int len, float *deltaFrame, float *tauB, float *etbR, 
    float gain);

void RedSolveHydrogenFlowInWell_Vec(int numfb, float **vb_out, float **red_hydrogen, 
    int len, float *deltaFrame, float *tauB);

void BlueSolveBackgroundTrace_Vec(int numfb, float **vb_out, float **blue_hydrogen, int len, 
    float *deltaFrame, float *tauB, float *etbR);




#endif // DIFFEQMODELVEC_H
