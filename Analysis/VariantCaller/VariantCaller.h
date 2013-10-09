/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantCaller.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#ifndef VARIANTCALLER_H
#define VARIANTCALLER_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include "api/api_global.h"
#include "api/BamAux.h"
#include "api/BamConstants.h"
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/SamReadGroup.h"
#include "api/SamReadGroupDictionary.h"
#include "api/SamSequence.h"
#include "api/SamSequenceDictionary.h"

#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <Variant.h>

#include "stats.h"
#include <armadillo>


#include "HypothesisEvaluator.h"
#include "ExtendParameters.h"

#include "InputStructures.h"
#include "HandleVariant.h"
#include "ThreadedVariantQueue.h"
#include "Splice/ErrorMotifs.h"

using namespace std;
using namespace BamTools;
using namespace ion;

#endif // VARIANTCALLER_H
