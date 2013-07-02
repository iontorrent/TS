/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef SPLICEVARIANTHYPOTHESES_H
#define SPLICEVARIANTHYPOTHESES_H


#include "ExtendedReadInfo.h"
#include "HypothesisEvaluator.h"


bool SpliceVariantHypotheses(const ExtendedReadInfo &current_read, const AlleleIdentity &variant_identity,
                        const LocalReferenceContext &local_context, PersistingThreadObjects &thread_objects,
                        vector<string> &my_hypotheses, const InputStructures &global_context);


bool SpliceAddVariantAlleles(const ExtendedReadInfo &current_read, const string pretty_alignment,
                             const AlleleIdentity &variant_identity, const LocalReferenceContext &local_context,
                             vector<string> &my_hypotheses, unsigned int pretty_idx, int DEBUG);


void IncrementAlignmentIndices(const char aln_symbol, int &ref_idx, int &read_idx);


void DecrementAlignmentIndices(const char aln_symbol, int &ref_idx, int &read_idx);


string SpliceDoRealignement (PersistingThreadObjects &thread_objects, const ExtendedReadInfo &current_read,
		                     long variant_position, int DEBUG);

#endif // SPLICEVARIANTHYPOTHESES_H
