/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GPUMULTIFLOWFITCONTROL_H
#define GPUMULTIFLOWFITCONTROL_H

#include <map>
#include "GpuMultiFlowFitMatrixConfig.h"

using namespace std;

class GpuMultiFlowFitControl   // Singleton class
{

private:

  static GpuMultiFlowFitControl * _pInstance;
  
  GpuMultiFlowFitControl();
  ~GpuMultiFlowFitControl();
  //private copy/= constructors to prevent usage
  GpuMultiFlowFitControl(GpuMultiFlowFitControl const&);  
  GpuMultiFlowFitControl& operator=(GpuMultiFlowFitControl const&);

  
  public:


    static GpuMultiFlowFitControl * Instance();


    void clear();

    GpuMultiFlowFitMatrixConfig* GetMatrixConfig(std::string name) { return _matrixConfig[name]; }
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
    map<std::string, GpuMultiFlowFitMatrixConfig*> _matrixConfig;
    int _maxSteps;
    int _maxParams;
 
    static unsigned int _maxBeads;
    static unsigned int _maxFrames;
};

#endif // GPUMULTIFLOWFITCONTROL_H
