/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GPUMULTIFLOWFITCONTROL_H
#define GPUMULTIFLOWFITCONTROL_H

#include <map>
#include "GpuMultiFlowFitMatrixConfig.h"

using namespace std;

class GpuMultiFlowFitControl
{
  public:
    GpuMultiFlowFitControl();
    ~GpuMultiFlowFitControl();
    void clear();

    GpuMultiFlowFitMatrixConfig* GetMatrixConfig(const char* name) { return _matrixConfig[name]; }
    unsigned int GetMaxSteps() { return _maxSteps; }
    unsigned int GetMaxParamsToFit() { return _maxParams; }

    static void SetMaxBeads(unsigned int maxBeads) { _maxBeads = maxBeads; }
    static void SetMaxFrames(unsigned int maxFrames) { _maxFrames = maxFrames; }
    static unsigned int GetMaxBeads() { return _maxBeads; } 
    static unsigned int GetMaxFrames() { return _maxFrames; } 

  private:
    void CreateFitInitialGpuMatrixConfig(); 
    void CreatePostKeyFitGpuMatrixConfig();
    void DetermineMaxSteps(int steps);
    void DetermineMaxParams(int params); 

  // data members
  public:
    const BkgFitStructures _fitParams;
  private:
    map<const char*, GpuMultiFlowFitMatrixConfig*> _matrixConfig;
    int _maxSteps;
    int _maxParams;
 
    static unsigned int _maxBeads;
    static unsigned int _maxFrames;
};

#endif // GPUMULTIFLOWFITCONTROL_H
