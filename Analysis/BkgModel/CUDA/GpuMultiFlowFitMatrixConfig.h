/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef GPUMULTIFLOWFITMATRIXCONFIG_H
#define GPUMULTIFLOWFITMATRIXCONFIG_H

#include "BkgFitStructures.h"

class GpuMultiFlowFitMatrixConfig
{
  public:
    GpuMultiFlowFitMatrixConfig(fit_descriptor*, CpuStep_t*, int);
    ~GpuMultiFlowFitMatrixConfig();

    int GetNumSteps() { return _numSteps; }
    int GetNumParamsToFit() { return _numParamsToFit; }
    unsigned int* GetJTJMatrixMapForDotProductComputation() { return _jtjMatrixBitMap; }
    unsigned int* GetParamIdxMap()  { return _paramIdxMap; }
    CpuStep_t* GetPartialDerivSteps() { return _partialDerivSteps; }

  private:
    void CreatePartialDerivStepsVector(fit_descriptor*, CpuStep_t*, int);
    void CreateAffectedFlowsVector(fit_descriptor*);
    void CreateBitMapForJTJMatrixComputation();

  private:
    CpuStep_t* _partialDerivSteps;   
    unsigned int* _jtjMatrixBitMap;
    unsigned int* _paramIdxMap;
    unsigned int* _affectedFlowsForParamsBitMap;
    unsigned int* _paramToStepMap;
    int _numParamsToFit;
    int _numSteps;
};

#endif // GPUMULTIFLOWFITMATRIXCONFIG_H
