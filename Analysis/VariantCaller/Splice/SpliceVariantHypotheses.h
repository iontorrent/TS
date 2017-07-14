/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef SPLICEVARIANTHYPOTHESES_H
#define SPLICEVARIANTHYPOTHESES_H


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

class EnsembleEval;

bool SpliceVariantHypotheses(const Alignment &current_read, const EnsembleEval &my_ensemble,
                        const LocalReferenceContext &local_context, PersistingThreadObjects &thread_objects,
                        int &splice_start_flow, int &splice_end_flow, vector<string> &my_hypotheses,
                        vector<bool> & same_as_null_hypothesis, bool & changed_alignment, const InputStructures &global_context,
                        const ReferenceReader &ref_reader);


bool SpliceAddVariantAlleles(const Alignment &current_read, const string& pretty_alignment,
                             const EnsembleEval &my_ensemble,
                             const LocalReferenceContext &local_context, vector<string> &my_hypotheses,
                             unsigned int pretty_idx, int DEBUG);


void IncrementAlignmentIndices(const char aln_symbol, int &ref_idx, int &read_idx);

void DecrementAlignmentIndices(const char aln_symbol, int &ref_idx, int &read_idx);

void IncrementFlow(const ion::FlowOrder &flow_order, const char &nuc, int &flow);

void IncrementFlows(const ion::FlowOrder &flow_order, const char &nuc, vector<int> &flows);

int GetSpliceFlows(const Alignment &current_read, const InputStructures &global_context,
                   vector<string> &my_hypotheses, vector<bool> & same_as_null_hypothesis,
                   int splice_start_idx, vector<int> splice_end_idx, int &splice_start_flow);

string SpliceDoRealignement (PersistingThreadObjects &thread_objects, const Alignment &current_read, long variant_position,
		                     bool &changed_alignment, int DEBUG, const ReferenceReader &ref_reader, int chr_idx);

#endif // SPLICEVARIANTHYPOTHESES_H
