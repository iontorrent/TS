/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "GpuMultiFlowFitControl.h"

unsigned int GpuMultiFlowFitControl::_maxBeads = 216*224;
unsigned int GpuMultiFlowFitControl::_maxFrames = MAX_COMPRESSED_FRAMES;

void GpuMultiFlowFitControl::clear()
{
  // delete gpu matrix configs here
  map<const char*, GpuMultiFlowFitMatrixConfig*>::iterator it;
  for (it=_matrixConfig.begin(); it!=_matrixConfig.end(); ++it)
  {
    delete (*it).second;
  } 
}

GpuMultiFlowFitControl::GpuMultiFlowFitControl()
{
   // create post key matrix config for gpu
   _maxSteps = 0;
   _maxParams = 0;

   CreatePostKeyFitGpuMatrixConfig();
   CreateFitInitialGpuMatrixConfig();
}

GpuMultiFlowFitControl::~GpuMultiFlowFitControl()
{
  clear();
}

void GpuMultiFlowFitControl::CreatePostKeyFitGpuMatrixConfig()
{
  GpuMultiFlowFitMatrixConfig* config = new GpuMultiFlowFitMatrixConfig(_fitParams.fit_well_post_key_descriptor, 
                                                  _fitParams.Steps, _fitParams.NumSteps);
  DetermineMaxSteps(config->GetNumSteps());
  DetermineMaxParams(config->GetNumParamsToFit());
  _matrixConfig.insert(pair<const char*, GpuMultiFlowFitMatrixConfig*>("FitWellPostKey", config));
}

void GpuMultiFlowFitControl::CreateFitInitialGpuMatrixConfig()
{
  GpuMultiFlowFitMatrixConfig* config = new GpuMultiFlowFitMatrixConfig(
                                                  _fitParams.fit_well_ampl_buffering_descriptor, 
                                                  _fitParams.Steps, _fitParams.NumSteps);
  DetermineMaxSteps(config->GetNumSteps());
  DetermineMaxParams(config->GetNumParamsToFit());
  _matrixConfig.insert(pair<const char*, GpuMultiFlowFitMatrixConfig*>("FitWellAmplBuffering", config));
}

void GpuMultiFlowFitControl::DetermineMaxSteps(int steps)
{
  if (_maxSteps < steps)
    _maxSteps = steps; 
}

void GpuMultiFlowFitControl::DetermineMaxParams(int params)
{
  if (_maxParams < params)
    _maxParams = params; 
}
