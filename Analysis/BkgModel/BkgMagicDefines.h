/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMAGICDEFINES_H
#define BKGMAGICDEFINES_H

#include <sys/types.h>
#include "SystemMagicDefines.h"

// Amount in frames forward from t0 to use for VFC compression in TimeCompression.h
// @hack - Note this must be kept in synch with T0_LEFT_OFFSET in DifferentialSeparator.cpp
#define VFC_T0_OFFSET 6

// when handling >tables< use the table values
// plus one for table size
#define MAX_POISSON_TABLE_COL (MAX_HPXLEN+1) 
#define MAX_LUT_TABLE_COL (MAX_POISSON_TABLE_COL+1)
// when used to index into the last column
#define LAST_POISSON_TABLE_COL (MAX_HPXLEN)
#define MAX_POISSON_TABLE_ROW 512
#define POISSON_TABLE_STEP 0.05f

// magic numbers
#define SCALEOFBUFFERINGCHANGE 1000.0f 
#define xMINTAUB 4.0f //-vm:
#define xMAXTAUB 65.0f

#define MIN_RDR_HIGH_LIMIT  10.0f // -vm: 
#define MIN_RDR_HIGH_LIMIT_OLD  2.0f 

// try to define some timing issues
const float MIN_INCORPORATION_TIME_PI = 6.0f;
const float MIN_INCORPORATION_TIME_PER_MER_PI_VERSION_ONE = 2.0f;
const float MIN_INCORPORATION_TIME_PER_MER_PI_VERSION_TWO = 1.75f;

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

// In limited testing NUM_DM_PCA == 6 produced the best results, but
// a value of 4 has the least impact on the rest of the software (it keeps
// the memory needed for the dark matter compensator the same as before, and
// possibly is less risky for plug-ins/debug
#define NUM_DM_PCA   (4)
#define TARGET_PCA_SET_SIZE   (2000)
#define MIN_PCA_SET_SIZE      (250)
#define MIN_AVG_DM_SET_SIZE   (50)

// TODO: this is what I want for proton, but it breaks the GPU right now
// #define MINAMPL -0.5f

// >must be less than the poisson table size<
// >can be any float smaller than this<
//#define MAXAMPL LAST_POISSON_TABLE_COL

#define NUMSINGLEFLOWITER_LEVMAR 40
#define NUMSINGLEFLOWITER_GAUSSNEWTON 8

#define MAX_BOUND_CHECK(param) {if (bound->param < this->param) this->param = bound->param;}
#define MIN_BOUND_CHECK(param) {if (bound->param > this->param) this->param = bound->param;}

#define MAX_BOUND_PAIR_CHECK(param,bparam) {if (bound->bparam < this->param) this->param = bound->bparam;}
#define MIN_BOUND_PAIR_CHECK(param,bparam) {if (bound->bparam > this->param) this->param = bound->bparam;}

#define EFFECTIVEINFINITY 1000
#define SMALLINFINITY 100
#define SAFETYZERO 0.000001f

#define SENSMULTIPLIER 0.00002f
#define COPYMULTIPLIER 1E+6f

#define CRUDEXEMPHASIS 0.0f
#define FINEXEMPHASIS 1.0f

#define MAXRCHANGE 0.02f
#define MAXCOPYCHANGE 0.98f

typedef int16_t FG_BUFFER_TYPE;


#define MAGIC_OFFSET_FOR_EMPTY_TRACE 1.0f
#define DEFAULT_FRAME_TSHIFT 0 // obsolete

#define WASHOUT_THRESHOLD 2.0
#define WASHOUT_FLOW_DETECTION 6

#define MAGIC_CLONAL_CALL_ARRAY_SIZE 12
#define MAGIC_MAX_CLONAL_HP_LEVEL 5
#define NO_NONCLONAL_PENALTY 0
#define FULL_NONCLONAL_PENALTY 5

#define NUMEMPHASISPARAMETERS 8

// helpers

#define NO_ADDITIONAL_WELL_ITERATIONS 0
#define HAPPY_ALL_BEADS 3
#define SMALL_LAMBDA 0.1f
#define LARGER_LAMBDA 1.0f
#define BIG_LAMBDA 10.0f

// speedup flags to accumulate 2x all together
#define CENSOR_ZERO_EMPHASIS 1
#define CENSOR_THRESHOLD 0.001f
#define MIN_CENSOR 1

// levmar state
#define UNINITIALIZED -1
#define MEAN_PER_FLOW 997

// time compression
//#define MAX_COMPRESSED_FRAMES 41
// Incrase MAX_COMPRESSED_FRAMES to 51 to make sure it works. Will remove MAX_COMPRESSED_FRAMES eventually and use max_frames
#define MAX_COMPRESSED_FRAMES 61
// to accommodate exponential tail fit large number of frames in GPU code
#define MAX_COMPRESSED_FRAMES_GPU 61
#define MAX_UNCOMPRESSED_FRAMES_GPU 110
#define MAX_PREALLOC_COMPRESSED_FRAMES_GPU 48

// Just for the GPU version, how many flows can be in a block (like numfb).
#define MAX_NUM_FLOWS_IN_BLOCK_GPU 32

// random values to keep people from iterating
#define TIME_FOR_NEXT_BLOCK -1
#define TIME_TO_DO_UPSTREAM 555
#define TIME_TO_DO_MULTIFLOW_REGIONAL_FIT 999
#define TIME_TO_DO_MULTIFLOW_FIT_ALL_WELLS 888
#define TIME_TO_DO_REMAIN_MULTI_FLOW_FIT_STEPS 666
#define TIME_TO_DO_DOWNSTREAM 457
#define TIME_TO_DO_PREWELL 234
#define TIME_TO_DO_EXPORT 777

#define LARGE_PRIME 104729
#define SMALL_PRIME 541


// please get rid of these soon - bkgmodel should treat these as global parameters
// to support possible changes for Proton
// we are experimenting with different sampling rates
// please use the deltaFrameSeconds in time-compression as a central source for this conversion(?)
#define FRAMESPERSEC 15.0f

// Number of sample beads to calculate generic xtalk for simple model in
// trace level crosstalk correction
#define GENERIC_SIMPLE_XTALK_SAMPLE 100

#endif // BKGMAGICDEFINES_H
