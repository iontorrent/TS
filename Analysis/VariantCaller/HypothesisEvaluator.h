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
void CalculateHypPredictions(
		PersistingThreadObjects  &thread_objects,
		const Alignment          &my_read,
        const InputStructures    &global_context,
        const vector<string>     &Hypotheses,
	    const int                &hyp_same_as_null,
        vector<vector<float> >   &predictions,
        vector<float>            &normalizedMeasurements,
		int                      &min_last_flow,
		int                      &max_last_flow,
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
