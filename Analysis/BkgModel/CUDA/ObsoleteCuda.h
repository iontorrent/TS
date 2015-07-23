/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef OBSOLETECUDA_H
#define OBSOLETECUDA_H

//@TODO:  placeholder for things that CUDA believes it needs declared that the CPU code has evolved away from
// Lets us compile without having the CUDA code explode

struct BkgModSingleFlowFitParams {
    float Ampl;
};

struct BkgModSingleFlowFitKrateParams
{
  float Ampl;
  float kmult;
  float dmultX;
};



// this for example depends on well size
#define n_to_uM_conv    (0.000062f)


#endif // OBSOLETECUDA_H
