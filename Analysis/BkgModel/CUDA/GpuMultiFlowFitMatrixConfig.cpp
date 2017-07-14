/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "CudaDefines.h"
#include "GpuMultiFlowFitMatrixConfig.h"

GpuMultiFlowFitMatrixConfig::GpuMultiFlowFitMatrixConfig(const std::vector<fit_descriptor>& fds, CpuStep* Steps, int maxSteps, int flow_key, int flow_block_size)
{
   // num of partial derivative steps to compute for this fit descriptor
   _numSteps = BkgFitStructures::GetNumParDerivStepsForFitDescriptor(fds);

   // number of actual partial derivative steps to compute
   _numSteps = _numSteps + 2; // Need to calculate FVAL and YERR always

   // calculate num of params to fit based on param sensitivity classification
   _numParamsToFit = BkgFitStructures::GetNumParamsToFitForDescriptor(fds, flow_key, flow_block_size);

   // collect partial derivative steps from Steps structure in 
   // BkgFitStructures.cpp for this fit
   CreatePartialDerivStepsVector(fds, Steps, maxSteps);
   
  _paramIdxMap = new unsigned int[_numParamsToFit];
  _affectedFlowsForParamsBitMap = new unsigned int[_numParamsToFit];
  _paramToStepMap = new unsigned int[_numParamsToFit];
  _jtjMatrixBitMap = new unsigned int[_numParamsToFit*_numParamsToFit];

  CreateAffectedFlowsVector(fds, flow_key, flow_block_size);
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

void GpuMultiFlowFitMatrixConfig::CreatePartialDerivStepsVector(const std::vector<fit_descriptor>& fds, CpuStep* Steps, int maxSteps)
{
  _partialDerivSteps = new CpuStep[_numSteps];

  if (Steps[0].PartialDerivMask == FVAL)
    _partialDerivSteps[0] = Steps[0];

  if (Steps[maxSteps - 1].PartialDerivMask == YERR) 
    _partialDerivSteps[_numSteps - 1] = Steps[maxSteps - 1];
  
  for (int i=1; fds[i-1].comp != TBL_END; ++i) 
  {
    for (int j=0; j<maxSteps; ++j) 
    {
      if ((unsigned int)fds[i-1].comp == Steps[j].PartialDerivMask)
      {
        _partialDerivSteps[i] = Steps[j];
        break;
      }
    }
  } 
}

void GpuMultiFlowFitMatrixConfig::CreateAffectedFlowsVector(
    const std::vector<fit_descriptor>& fds, 
    int flow_key, 
    int flow_block_size
  )
{
  BeadParams dummyBead;
  reg_params dummyReg;

  unsigned int paramIdx = 0;  
  unsigned int actualStep = 1;
  for (int i=0; fds[i].comp != TBL_END; ++i)
  {
    // Calculate a base index for reaching into the BeadParams / reg_params structure.
    unsigned int baseIndex = fds[i].bead_params_func ? ( dummyBead.*( fds[i].bead_params_func ))() - reinterpret_cast< float * >( & dummyBead )
                                                    : ( dummyReg .*( fds[i].reg_params_func  ))() - reinterpret_cast< float * >( & dummyReg );
            
    switch(fds[i].ptype)
    {
      case ParamTypeAllFlow:
        _paramIdxMap[paramIdx] = baseIndex;
        _paramToStepMap[paramIdx] = actualStep;
        _affectedFlowsForParamsBitMap[paramIdx] = 0;
        for (int j=0; j<flow_block_size; ++j) 
        { 
          _affectedFlowsForParamsBitMap[paramIdx] |= (1 << j);
          //printf("%u %u %x\n", _paramToStepMap[paramIdx], paramIdx, _affectedFlowsForParamsBitMap[paramIdx]);
        }
        paramIdx++;
      break;
      case ParamTypeNotKey:
        // create an independent paramter per flow except for key flows
        for (int j=flow_key; j<flow_block_size; ++j)
        {
          _paramIdxMap[paramIdx] = baseIndex + (j-flow_key);
          _paramToStepMap[paramIdx] = actualStep;
          _affectedFlowsForParamsBitMap[paramIdx] = 0;
          _affectedFlowsForParamsBitMap[paramIdx] |= (1 << j);
          //printf("%u %u %x\n", _paramToStepMap[paramIdx], paramIdx, _affectedFlowsForParamsBitMap[paramIdx]);
          paramIdx++;
        }
        break;
      case ParamTypeAllButFlow0:
        // create an independent paramter per flow except for the first flow
        for (int j=1; j < flow_block_size; ++j)
        {
          _paramIdxMap[paramIdx] = baseIndex+ (j-1);
          _paramToStepMap[paramIdx] = actualStep;
          _affectedFlowsForParamsBitMap[paramIdx] = 0;
          _affectedFlowsForParamsBitMap[paramIdx] |= (1 << j);
          paramIdx++;
        }
        break;
      case ParamTypePerFlow:
        // create an independent paramter per flow
        for (int j=0; j < flow_block_size; ++j)
        {
          _paramIdxMap[paramIdx] = baseIndex + j;
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
