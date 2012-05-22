/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FILTERCONTROLOPTS_H
#define FILTERCONTROLOPTS_H

#include <vector>
#include <string>
#include <map>
#include <set>
#include "Region.h"
#include "IonVersion.h"
#include "Utils.h"

class FilterControlOpts{
  public:
    int percentPositiveFlowsFilterTraining;
    int percentPositiveFlowsFilterCalling;
    int percentPositiveFlowsFilterTFs;
    bool KEYPASSFILTER;
    // Options related to filtering reads by percentage of positive flows
    // Options related to filtering reads by putative clonality
    int clonalFilterTraining;
    int clonalFilterSolving;
    // Options related to filtering reads by CAFIE residuals
    int cafieResFilterTraining;
    int cafieResFilterCalling;
    int cafieResFilterTFs;
    int cafieResMaxFlow;
    int cafieResMinFlow;
    double cafieResMaxValue;
    bool cafieResMaxValueOverride; // Will be true if the value is explicitly set on command line
    std::map<std::string,double> cafieResMaxValueByFlowOrder; // For holding flow-specific values.
    // too short!
    int minReadLength;

    // unfiltered summary
    int nUnfilteredLib;
    char *unfilteredLibDir;
    char *beadSummaryFile;

    void DefaultFilterControl();
    void RecognizeFlow(char *flowFormula);
    ~FilterControlOpts();
};

#endif // FILTERCONTROLOPTS_H