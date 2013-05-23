/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef ENSEMBLEGLUE_H
#define ENSEMBLEGLUE_H

#include "api/BamReader.h"

#include "../Analysis/file-io/ion_util.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>

#include "ClassifyVariant.h"
#include "StackEngine.h"
#include "DecisionTreeData.h"
#include "json/json.h"

using namespace std;

//void GlueInVariants(StackPlus &my_data, HypothesisStack &hypothesis_stack, AlleleIdentity &variant_identity, const string &local_contig_sequence);
void GlueOutputVariant(EnsembleEval &my_ensemble, ExtendParameters *parameters, int _alt_allele_index);
void JustOneDiagnosis(EnsembleEval &my_ensemble, string &out_dir);
// in case we have multiple alleles as candidates
int TrySolveAllAllelesVsRef(EnsembleEval &my_ensemble, const string & local_contig_sequence, int DEBUG);

#endif // ENSEMBLEGLUE_H
