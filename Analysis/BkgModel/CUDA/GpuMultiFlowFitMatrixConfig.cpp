/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "CudaDefines.h"
#include "GpuMultiFlowFitMatrixConfig.h"

GpuMultiFlowFitMatrixConfig::GpuMultiFlowFitMatrixConfig(fit_descriptor* fd, CpuStep_t* Steps, int maxSteps)
{
   // num of partial derivative steps to compute for this fit descriptor
   _numSteps = GetNumParDerivStepsForFitDescriptor(fd);

   // number of actual partial derivative steps to compute
   _numSteps = _numSteps + 2; // Need to calculate FVAL and YERR always

   // calculate num of params to fit based on param sensitivity classification
   _numParamsToFit = GetNumParamsToFitForDescriptor(fd);

   // collect partial derivative steps from Steps structure in 
   // BkgFitStructures.cpp for this fit
   CreatePartialDerivStepsVector(fd, Steps, maxSteps);
   
  _paramIdxMap = new unsigned int[_numParamsToFit];
  _affectedFlowsForParamsBitMap = new unsigned int[_numParamsToFit];
  _paramToStepMap = new unsigned int[_numParamsToFit];
  _jtjMatrixBitMap = new unsigned int[_numParamsToFit*_numParamsToFit];

  CreateAffectedFlowsVector(fd);
  CreateBitMapForJTJMatrixComputation();
}

GpuMultiFlowFitMatrixConfig::~GpuMultiFlowFitMatrixConfig()
{
  delete [] _partialDerivSteps;
  delete [] _jtjMatrixBitMap;
  delete [] _paramIdxMap;
  delete [] _affectedFlowsForParamsBitMap;
  delete [] _paramToStepMap;
}

void GpuMultiFlowFitMatrixConfig::CreatePartialDerivStepsVector(fit_descriptor* fd, CpuStep_t* Steps, int maxSteps)
{
  _partialDerivSteps = new CpuStep_t[_numSteps];

  if (Steps[0].PartialDerivMask == FVAL)
    _partialDerivSteps[0] = Steps[0];

  if (Steps[maxSteps - 1].PartialDerivMask == YERR) 
    _partialDerivSteps[_numSteps - 1] = Steps[maxSteps - 1];
  
  for (int i=1; fd[i-1].comp != TBL_END; ++i) 
  {
    for (int j=0; j<maxSteps; ++j) 
    {
      if ((unsigned int)fd[i-1].comp == Steps[j].PartialDerivMask)
      {
        _partialDerivSteps[i] = Steps[j];
        break;
      }
    }
  } 
}

void GpuMultiFlowFitMatrixConfig::CreateAffectedFlowsVector(fit_descriptor* fd)
{
  unsigned int paramIdx = 0;  
  unsigned int actualStep = 1;
  for (int i=0; fd[i].comp != TBL_END; ++i)
  {
    switch(fd[i].ptype)
    {
      case ParamTypeAllFlow:
        _paramIdxMap[paramIdx] = fd[i].param_ndx;
        _paramToStepMap[paramIdx] = actualStep;
        _affectedFlowsForParamsBitMap[paramIdx] = 0;
        for (int j=0; j<NUMFB; ++j) 
        { 
          _affectedFlowsForParamsBitMap[paramIdx] |= (1 << j);
          //printf("%u %u %x\n", _paramToStepMap[paramIdx], paramIdx, _affectedFlowsForParamsBitMap[paramIdx]);
        }
        paramIdx++;
      break;
      case ParamTypeNotKey:
        // create an independent paramter per flow except for key flows
        for (int j=KEY_LEN; j<NUMFB; ++j)
        {
          _paramIdxMap[paramIdx] = fd[i].param_ndx + (j-KEY_LEN);
          _paramToStepMap[paramIdx] = actualStep;
          _affectedFlowsForParamsBitMap[paramIdx] = 0;
          _affectedFlowsForParamsBitMap[paramIdx] |= (1 << j);
          //printf("%u %u %x\n", _paramToStepMap[paramIdx], paramIdx, _affectedFlowsForParamsBitMap[paramIdx]);
          paramIdx++;
        }
        break;
      case ParamTypeAllButFlow0:
        // create an independent paramter per flow except for the first flow
        for (int j=1; j < NUMFB; ++j)
        {
	  _paramIdxMap[paramIdx] = fd[i].param_ndx+ (j-1);
	  _paramToStepMap[paramIdx] = actualStep;
	  _affectedFlowsForParamsBitMap[paramIdx] = 0;
	  _affectedFlowsForParamsBitMap[paramIdx] |= (1 << j);
	  paramIdx++;
	}
	break;
      case ParamTypePerFlow:
        // create an independent paramter per flow
        for (int j=0; j < NUMFB; ++j)
        {
          _paramIdxMap[paramIdx] = fd[i].param_ndx + j;
          _paramToStepMap[paramIdx] = actualStep;
          _affectedFlowsForParamsBitMap[paramIdx] = 0;
          _affectedFlowsForParamsBitMap[paramIdx] |= (1 << j);
          paramIdx++;
        }
        break;
        // need to correct for nuc order
      case ParamTypePerNuc:
      case ParamTypeAFlows:
      case ParamTypeCFlows:
      case ParamTypeGFlows:
      default:
        break;
    }
    actualStep++;
  }
}

void GpuMultiFlowFitMatrixConfig::CreateBitMapForJTJMatrixComputation()
{
  for (int i=0; i<_numParamsToFit; ++i)
  {
    for (int j=0; j<_numParamsToFit; ++j)
    {
      _jtjMatrixBitMap[i*_numParamsToFit + j] = 
        (_paramToStepMap[i] << PARAM1_STEPIDX_SHIFT) | (_paramToStepMap[j] << PARAM2_STEPIDX_SHIFT) |
                (_affectedFlowsForParamsBitMap[i] & _affectedFlowsForParamsBitMap[j]);  
    }
  }
}
