/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HypothesisEvaluator.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#ifndef HYPOTHESISEVALUATOR_H
#define HYPOTHESISEVALUATOR_H

#include "SpliceVariantHypotheses.h"
/*
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <assert.h>
#include "stdlib.h"
#include "ctype.h"
#include "ClassifyVariant.h"
#include "ExtendedReadInfo.h"


using namespace std;
using namespace ion;*/


// Function to calculate signal predictions
int CalculateHypPredictions(
		PersistingThreadObjects  &thread_objects,
		const Alignment          &my_read,
        const InputStructures    &global_context,
        const vector<string>     &Hypotheses,
        const vector<bool>       &same_as_null_hypothesis,
        vector<vector<float> >   &predictions,
        vector<vector<float> >   &normalizedMeasurements,
        int flow_upper_bound);

// Does what the name says
void InitializeBasecallers(PersistingThreadObjects &thread_objects,
                         const Alignment         &my_read,
	                       const InputStructures   &global_context);

// Solve for hard and soft clipped bases at the start of the read, before start_flow
int GetStartOfMasterRead(PersistingThreadObjects  &thread_objects,
		                    const Alignment          &my_read,
	                      const InputStructures    &global_context,
	                      const vector<string>     &Hypotheses,
	                      const int                &nFlows,
	                      BasecallerRead           &master_read);

// Print out some messages
void PredictionGenerationVerbose(const vector<string>         &Hypotheses,
		                         const vector<BasecallerRead> &hypothesesReads,
		                         const Alignment              &my_read,
		                         const vector<vector<float> > &predictions,
		                         const int                    &prefix_size,
		                         const InputStructures        &global_context);

#endif // HYPOTHESISEVALUATOR_H
