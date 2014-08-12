/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "GpuMultiFlowFitControl.h"
#include <pthread.h>


unsigned int GpuMultiFlowFitControl::_maxBeads = 216*224;
unsigned int GpuMultiFlowFitControl::_maxFrames = MAX_COMPRESSED_FRAMES_GPU;
bool GpuMultiFlowFitControl::_gpuTraceXtalk = false;


GpuMultiFlowFitControl::GpuMultiFlowFitControl()
{
   // create post key matrix config for gpu
   _maxSteps = 0;
   _maxParams = 0;
}

void GpuMultiFlowFitControl::SetFlowParams( int flow_key, int flow_block_size )
{
  // Update the active matrix config.
  _activeFlowKey = flow_key;
  _activeFlowMax = flow_block_size;
  if ( ! GetMatrixConfig( "FitWellPostKey" ) )
    CreatePostKeyFitGpuMatrixConfig( flow_key, flow_block_size );
  if ( ! GetMatrixConfig( "FitWellAmplBuffering" ) )
    CreateFitInitialGpuMatrixConfig( flow_key, flow_block_size );
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

void GpuMultiFlowFitControl::CreatePostKeyFitGpuMatrixConfig( int flow_key, int flow_block_size )
{
  // Build the configuration.
  GpuMultiFlowFitMatrixConfig* config = 
    new GpuMultiFlowFitMatrixConfig( BkgFitStructures::fit_well_post_key_descriptor, 
                                     BkgFitStructures::Steps, BkgFitStructures::NumSteps, 
                                     flow_key, flow_block_size);

  // Accumulate maximum values.
  DetermineMaxSteps(config->GetNumSteps());
  DetermineMaxParams(config->GetNumParamsToFit());

  // Find the proper map to put it in.
  _allMatrixConfig[ MatrixIndex( flow_key, flow_block_size, "FitWellPostKey" ) ] = config;
}

void GpuMultiFlowFitControl::CreateFitInitialGpuMatrixConfig( int flow_key, int flow_block_size )
{
  // Build the configuration.
  GpuMultiFlowFitMatrixConfig* config = 
    new GpuMultiFlowFitMatrixConfig( BkgFitStructures::fit_well_ampl_buffering_descriptor, 
                                     BkgFitStructures::Steps, BkgFitStructures::NumSteps, 
                                     flow_key, flow_block_size);

  // Accumulate maximum values.
  DetermineMaxSteps(config->GetNumSteps());
  DetermineMaxParams(config->GetNumParamsToFit());

  // Find the proper map to put it in.
  _allMatrixConfig[ MatrixIndex( flow_key, flow_block_size, "FitWellAmplBuffering" ) ] = config;
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
