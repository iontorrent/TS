/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef GPUMULTIFLOWFITMATRIXCONFIG_H
#define GPUMULTIFLOWFITMATRIXCONFIG_H

#include "BkgFitStructures.h"

class GpuMultiFlowFitMatrixConfig
{
  public:
    GpuMultiFlowFitMatrixConfig(const std::vector<fit_descriptor>& fds, CpuStep*, int maxSteps, int flow_key, int flow_block_size);
    ~GpuMultiFlowFitMatrixConfig();

    int GetNumSteps() { return _numSteps; }
    int GetNumParamsToFit() { return _numParamsToFit; }
    unsigned int* GetJTJMatrixMapForDotProductComputation() { return _jtjMatrixBitMap; }
    unsigned int* GetParamIdxMap()  { return _paramIdxMap; }
    CpuStep* GetPartialDerivSteps() { return _partialDerivSteps; }

  private:
    void CreatePartialDerivStepsVector(const std::vector<fit_descriptor>& fds, CpuStep*, int);
    void CreateAffectedFlowsVector(const std::vector<fit_descriptor>& fds, int flow_key, int flow_block_size);
    void CreateBitMapForJTJMatrixComputation();

  private:
    CpuStep* _partialDerivSteps;   
    unsigned int* _jtjMatrixBitMap;
    unsigned int* _paramIdxMap;
    unsigned int* _affectedFlowsForParamsBitMap;
    unsigned int* _paramToStepMap;
    int _numParamsToFit;
    int _numSteps;
};

#endif // GPUMULTIFLOWFITMATRIXCONFIG_H
