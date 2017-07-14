/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "GpuMultiFlowFitControl.h"
#include <pthread.h>


unsigned int GpuMultiFlowFitControl::_maxBeads = 216*224;
unsigned int GpuMultiFlowFitControl::_maxFrames = MAX_COMPRESSED_FRAMES_GPU;
bool GpuMultiFlowFitControl::_gpuTraceXtalk = false;


GpuMultiFlowFitControl::GpuMultiFlowFitControl()
{
   _maxSteps = 7;
   _maxParams = 21;
   _activeFlowKey = 7;
   _activeFlowMax = 20;
}

void GpuMultiFlowFitControl::SetFlowParams( int flow_key, int flow_block_size )
{
  // Update the active matrix config.
  _activeFlowKey = flow_key;
  _activeFlowMax = flow_block_size;
}

GpuMultiFlowFitControl::~GpuMultiFlowFitControl()
{
  // delete gpu matrix configs here
  map<MatrixIndex, GpuMultiFlowFitMatrixConfig*>::iterator it;

  for( it = _allMatrixConfig.begin() ; it != _allMatrixConfig.end() ; ++it )
  {
    delete it->second;
  }
}

GpuMultiFlowFitMatrixConfig* GpuMultiFlowFitControl::createConfig(
  const string &fitName, 
  const master_fit_type_table* levMarSparseMatrices)
{
  // Build the configuration.
  GpuMultiFlowFitMatrixConfig* config = 
    new GpuMultiFlowFitMatrixConfig(const_cast<master_fit_type_table*>(levMarSparseMatrices)->GetFitDescriptorByName(fitName.c_str()), 
                                     BkgFitStructures::Steps, BkgFitStructures::NumSteps, 
                                     _activeFlowKey, _activeFlowMax);
  // Accumulate maximum values.
  DetermineMaxSteps(config->GetNumSteps());
  DetermineMaxParams(config->GetNumParamsToFit());

  // Find the proper map to put it in.
  _allMatrixConfig[ MatrixIndex(_activeFlowKey, _activeFlowMax, fitName ) ] = config;
  return config;
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

GpuMultiFlowFitMatrixConfig* GpuMultiFlowFitControl::GetMatrixConfig(
  const std::string &name,
  const master_fit_type_table *levMarSparseMatrices) { 
     
  GpuMultiFlowFitMatrixConfig *config =  _allMatrixConfig[MatrixIndex( _activeFlowKey, _activeFlowMax, name)];

  if (config)
    return config;
  else
    return createConfig(name, levMarSparseMatrices);
}
