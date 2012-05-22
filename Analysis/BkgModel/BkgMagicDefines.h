/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMAGICDEFINES_H
#define BKGMAGICDEFINES_H

#include <sys/types.h>

// the single most widely abused macro
// chunk size of number of flows
#define NUMFB                   20

// define this here to make sure it is consistent across codebase
#define MAX_HPLEN 11
#define MAX_POISSON_TABLE_ROW 512
#define POISSON_TABLE_STEP 0.05f

// magic numbers
#define SCALEOFBUFFERINGCHANGE 1000.0f
#define MINTAUB 4.0f
#define MAXTAUB 65.0f

#define MIN_RDR_HIGH_LIMIT  2.0f

#define SINGLE_BKG_IMAGE

#define ISIG_SUB_STEPS_SINGLE_FLOW  (2)
#define ISIG_SUB_STEPS_MULTI_FLOW   (1)

#define KEY_LEN                 7
#define NUMNUC                  4
#define WELL_COMPLETED_COUNT    8
#define FRAME_AVERAGE           1
#define FIRSTNFLOWSFORERRORCALC 12

// index into a lookup table for nucleotide parameters
#define TNUCINDEX 0
#define ANUCINDEX 1
#define CNUCINDEX 2
#define GNUCINDEX 3
#define DEFAULTNUCINDEX 0

#define MAXCLONALMODIFYPOINTSERROR 6

#define NUMBEADSPERGROUP 199

#define MINAMPL 0.001f
// TODO: this is what I want for proton, but it breaks the GPU right now
// #define MINAMPL -0.5f
#define MAXAMPL MAX_HPLEN-1

#define NUMSINGLEFLOWITER 40

#define MAX_BOUND_CHECK(param) {if (bound->param < cur->param) cur->param = bound->param;}
#define MIN_BOUND_CHECK(param) {if (bound->param > cur->param) cur->param = bound->param;}

#define MAX_BOUND_PAIR_CHECK(param,bparam) {if (bound->bparam < cur->param) cur->param = bound->bparam;}
#define MIN_BOUND_PAIR_CHECK(param,bparam) {if (bound->bparam > cur->param) cur->param = bound->bparam;}

#define EFFECTIVEINFINITY 1000



#define SENSMULTIPLIER 0.00002f
#define COPYMULTIPLIER 1E+6f

#define CRUDEXEMPHASIS 0.0f
#define FINEXEMPHASIS 1.0f

#define MAXRCHANGE 0.02f
#define MAXCOPYCHANGE 0.98f

typedef int16_t FG_BUFFER_TYPE;


#define MAGIC_OFFSET_FOR_EMPTY_TRACE 4.0f
#define DEFAULT_FRAME_TSHIFT 3

#define WASHOUT_THRESHOLD 2.0
#define WASHOUT_FLOW_DETECTION 6

#define MAGIC_MAX_CLONAL_HP_LEVEL 5

// speedup flags to accumulate 2x all together
#define CENSOR_ZERO_EMPHASIS 1
#define CENSOR_THRESHOLD 0.01f
#define MIN_CENSOR 1

// levmar state
#define UNINITIALIZED -1
#define MEAN_PER_FLOW 997

// time compression
#define MAX_COMPRESSED_FRAMES 41

#endif // BKGMAGICDEFINES_H
