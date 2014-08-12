/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GPUMULTIFLOWFITCONTROL_H
#define GPUMULTIFLOWFITCONTROL_H

#include <map>
#include "GpuMultiFlowFitMatrixConfig.h"
#include "FlowSequence.h"

using namespace std;

// This used to be a Singleton class. Now it's not.
class GpuMultiFlowFitControl
{

private:

  //private copy/= constructors to prevent usage
  GpuMultiFlowFitControl(GpuMultiFlowFitControl const&);  
  GpuMultiFlowFitControl& operator=(GpuMultiFlowFitControl const&);

  
  public:
  GpuMultiFlowFitControl();
  ~GpuMultiFlowFitControl();

    // When something sets the flow params, it adjusts the current instance.
    void SetFlowParams( int flow_key, int flow_block_size );

    unsigned int GetMaxSteps() const { return _maxSteps; }
    unsigned int GetMaxParamsToFit() const { return _maxParams; }

    static void SetMaxBeads(unsigned int maxBeads) { _maxBeads = maxBeads; }
    static void SetMaxFrames(unsigned int maxFrames) { _maxFrames = maxFrames; }
    static void SetChemicalXtalkCorrectionForPGM(bool doXtalk) { _gpuTraceXtalk = doXtalk; }
    static unsigned int GetMaxBeads() { return _maxBeads; } 
    static unsigned int GetMaxFrames() { return _maxFrames; } 
    static bool doGPUTraceLevelXtalk() { return _gpuTraceXtalk; }

    GpuMultiFlowFitMatrixConfig* GetMatrixConfig(const std::string & name) { 
      return _allMatrixConfig[MatrixIndex( _activeFlowKey, _activeFlowMax, name ) ]; 
    }
  private:
    void CreateFitInitialGpuMatrixConfig( int flow_key, int flow_block_size ); 
    void CreatePostKeyFitGpuMatrixConfig( int flow_key, int flow_block_size );
    void DetermineMaxSteps(int steps);
    void DetermineMaxParams(int params); 

  // data members
  struct MatrixIndex {
    int flow_key;
    int flow_block_size;
    std::string name;

    MatrixIndex( int k, int m, std::string n ) :
      flow_key( k ), flow_block_size( m ), name( n ) {
    }

    bool operator < ( const MatrixIndex & that ) const {
      if ( this->flow_key < that.flow_key )   return true;
      if ( this->flow_key > that.flow_key )   return false;
      if ( this->flow_block_size < that.flow_block_size )   return true;
      if ( this->flow_block_size > that.flow_block_size )   return false;
      return this->name < that.name;
    }
  };

  int _activeFlowKey, _activeFlowMax;
  std::map<MatrixIndex, GpuMultiFlowFitMatrixConfig* > _allMatrixConfig;
    int _maxSteps;
    int _maxParams;
 
  static unsigned int _maxBeads;
  static unsigned int _maxFrames;
  static bool _gpuTraceXtalk;
};

#endif // GPUMULTIFLOWFITCONTROL_H
