/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMAGICDEFINES_H
#define BKGMAGICDEFINES_H

#include <sys/types.h>

// define this here to make sure it is consistent across codebase
#define MAX_HPLEN 11
#define MAX_POISSON_TABLE_ROW 512
#define POISSON_TABLE_STEP 0.05

#define CHIP_GAIN     0.62
#define NERSTIAN_GAIN 55.0
#define CNT_TO_MV     69.0
#define CON_SCALE     4.5E+10
#define BASE_PH      7.5
#define FRAMESPERSECOND 15.0
// magic numbers
#define SCALEOFBUFFERINGCHANGE 1000.0
#define MINTAUB 4.0
#define MAXTAUB 65.0

#define MIN_RDR_HIGH_LIMIT  2.0

//#define TANGO_BUILD

#ifdef TANGO_BUILD
#define NUMFB                   20
#define SINGLE_BKG_IMAGE
#else
#define NUMFB                   20
#define SINGLE_BKG_IMAGE
#endif

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

#define MINAMPL 0.001
#define MAXAMPL MAX_HPLEN-1

#define NUMSINGLEFLOWITER 40

#define MAX_BOUND_CHECK(param) {if (bound->param < cur->param) cur->param = bound->param;}
#define MIN_BOUND_CHECK(param) {if (bound->param > cur->param) cur->param = bound->param;}

#define MAX_BOUND_PAIR_CHECK(param,bparam) {if (bound->bparam < cur->param) cur->param = bound->bparam;}
#define MIN_BOUND_PAIR_CHECK(param,bparam) {if (bound->bparam > cur->param) cur->param = bound->bparam;}

#define EFFECTIVEINFINITY 1000

#define VALVEOPENFRAME 15.0
#define TZERODELAYMAGICSCALE 20.7

#define SENSMULTIPLIER 0.00002
#define COPYMULTIPLIER 1E+6

#define CRUDEXEMPHASIS 0.0
#define FINEXEMPHASIS 1.0

#define MAXRCHANGE 0.02
#define MAXCOPYCHANGE 0.98

typedef int16_t FG_BUFFER_TYPE;


#define MAGIC_OFFSET_FOR_EMPTY_TRACE 4.0
#define DEFAULT_FRAME_TSHIFT 3

#define WASHOUT_THRESHOLD 2.0
#define WASHOUT_FLOW_DETECTION 6

#define MAGIC_MAX_CLONAL_HP_LEVEL 5

// speedup flags to accumulate 2x all together
#define CENSOR_ZERO_EMPHASIS 1
#define CENSOR_THRESHOLD 0.01
#define MIN_CENSOR 1

// levmar state
#define UNINITIALIZED -1
#define MEAN_PER_FLOW 997



#endif // BKGMAGICDEFINES_H
