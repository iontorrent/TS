/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef DIAGNOSTICJSON_H
#define DIAGNOSTICJSON_H
#include "api/BamReader.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>

#include "StackEngine.h"
#include "json/json.h"

using namespace std;

void JustOneDiagnosis(const EnsembleEval &my_ensemble, const InputStructures &global_context,
    const string &out_dir, bool rich_diag);

#endif // DIAGNOSTICJSON_H
