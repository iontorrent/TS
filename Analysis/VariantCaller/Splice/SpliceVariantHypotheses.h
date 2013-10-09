/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef SPLICEVARIANTHYPOTHESES_H
#define SPLICEVARIANTHYPOTHESES_H


//#include "ExtendedReadInfo.h"
//#include "HypothesisEvaluator.h"

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
#include "MiscUtil.h"

using namespace std;
using namespace ion;


bool SpliceVariantHypotheses(ExtendedReadInfo &current_read, const MultiAlleleVariantIdentity &variant_identity,
                        const LocalReferenceContext &local_context, PersistingThreadObjects &thread_objects,
                        int &splice_start_flow, int &splice_end_flow, vector<string> &my_hypotheses,
                        const InputStructures &global_context);


bool SpliceAddVariantAlleles(const ExtendedReadInfo &current_read, const string pretty_alignment,
                             const MultiAlleleVariantIdentity &variant_identity,
                             const LocalReferenceContext &local_context, vector<string> &my_hypotheses,
                             unsigned int pretty_idx, int DEBUG);


void IncrementAlignmentIndices(const char aln_symbol, int &ref_idx, int &read_idx);

void DecrementAlignmentIndices(const char aln_symbol, int &ref_idx, int &read_idx);

void IncrementFlow(const ion::FlowOrder &flow_order, const char &nuc, int &flow);

void IncrementFlows(const ion::FlowOrder &flow_order, const char &nuc, vector<int> &flows);

int GetSpliceFlows(ExtendedReadInfo &current_read, const InputStructures &global_context,
                   vector<string> &my_hypotheses, int splice_start_idx, vector<int> splice_end_idx,
                   int &splice_start_flow);

string SpliceDoRealignement (PersistingThreadObjects &thread_objects, const ExtendedReadInfo &current_read,
		                     long variant_position, int DEBUG);

#endif // SPLICEVARIANTHYPOTHESES_H
