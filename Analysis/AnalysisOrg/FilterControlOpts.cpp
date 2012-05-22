/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FilterControlOpts.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h> //EXIT_FAILURE
#include <ctype.h>  //tolower
#include <libgen.h> //dirname, basename
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>

void FilterControlOpts::DefaultFilterControl()
{
  nUnfilteredLib = 100000;
  unfilteredLibDir = strdup ("unfiltered");
  beadSummaryFile = strdup ("beadSummary.unfiltered.txt");
  KEYPASSFILTER = true;

  minReadLength = 8;
  // Options related to filtering reads by percentage of positive flows
  percentPositiveFlowsFilterTraining = 0;  // Should the ppf filter be used when initially estimating CAFIE params?
  percentPositiveFlowsFilterCalling = 1;   // Should the ppf filter be used when writing the SFF?
  percentPositiveFlowsFilterTFs = 0;       // If the ppf filter is on, should it be applied to TFs?
  // Options related to filtering reads by putative clonality
  clonalFilterTraining = 0;  // Should the clonality filter be used when initially estimating CAFIE params?
  clonalFilterSolving  = 0;  // Should the clonality filter be used when solving library reads?
  // Options related to filtering reads by CAFIE residuals
  cafieResFilterTraining = 0;  // Should the cafieRes filter be used when initially estimating CAFIE params?
  cafieResFilterCalling = 1;   // Should the cafieRes filter be used when writing the SFF?
  cafieResFilterTFs = 0;       // If the cafie residual filter is on, should it be applied to TFs?
  cafieResMaxFlow = 60;
  cafieResMinFlow = 12;
  cafieResMaxValue = 0.06;
  cafieResMaxValueOverride = false;
  cafieResMaxValueByFlowOrder[std::string ("TACG") ] = 0.06;  // regular flow order
  cafieResMaxValueByFlowOrder[std::string ("TACGTACGTCTGAGCATCGATCGATGTACAGC") ] = 0.08;  // xdb flow order
}


FilterControlOpts::~FilterControlOpts()
{
  if (unfilteredLibDir)
    free (unfilteredLibDir);
  if (beadSummaryFile)
    free (beadSummaryFile);
}


void FilterControlOpts::RecognizeFlow (char *flowFormula)
{
  // Set some options that depend of flow order (unless explicitly set in command line)
  std::map<std::string,double>::iterator it;
  // cafieResMaxValue
  it = cafieResMaxValueByFlowOrder.find (std::string (flowFormula));
  if (!cafieResMaxValueOverride && it != cafieResMaxValueByFlowOrder.end())
  {
    cafieResMaxValue = it->second;
  }

  // Test some dependencies between options
  if (cafieResMinFlow >= cafieResMaxFlow)
  {
    fprintf (stderr, "value of --cafie-residual-filter-min-flow must be strictly less than that of --cafie-residual-filter-max-flow.\n");
    exit (EXIT_FAILURE);
  }
}

