/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GLOBALDEFAULTSFORBKGMODEL_H
#define GLOBALDEFAULTSFORBKGMODEL_H

#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <sys/stat.h>
#include <ctype.h>

#include "BkgMagicDefines.h"
#include "MathOptim.h"

class GlobalDefaultsForBkgModel
{
  public:
    // this holds all the annoying static variables so I can ignore them
    // when I want to override them at a regional level
    // for example, emphasis vectors may vary by location on the chip
    // and not just in the obvious ways
    
    // plausibly a shared object
    static int flow_order_len;      // length of entire flow order sequence (might be > NUMFB)
  static int *glob_flow_ndx_map;  // maps flow number within a cycle to nucleotide (flow_order_len values)
  static char *flowOrder;         // entire flow order as a char array (flow_order_len values)
  // ugly
   static char *chipType; // Yes, this is available through Image, theoretically, but I need to know this before I see the first image passsed.
 
  // not plausible at all
  // various parameters used to initialize the model
  static float sens_default;
  static float molecules_to_micromolar_conv;
  static float tau_R_m_default;
  static float tau_R_o_default;
  
   static  float  krate_default[NUMNUC];
  static  float  d_default[NUMNUC];
  static  float  kmax_default[NUMNUC];
  static float sigma_mult_default[NUMNUC];
  static float t_mid_nuc_delay_default[NUMNUC];
  
  static float nuc_flow_frame_width;
  static int time_left_avg;
  static int time_start_detail;
  static int time_stop_detail;
  
  static  float  krate_adj_limit;
  static float dampen_kmult;
  static float kmult_low_limit;
  static float kmult_hi_limit;
  
  static float emphasis_ampl_default;
  static float emphasis_width_default;
  static int numEv;               // number of emphasis vectors allocated
  static float emp[];  // parameters for emphasis vector generation
  
    // weighting applied to clonal restriction on different hp lengths
  static float clonal_call_scale[];
  static float clonal_call_penalty;
  // why would this be global again?
  static  bool no_RatioDrift_fit_first_20_flows;
  
  static char *xtalk_name; // really bad, but I can't pass anything through analysis at all!!!
  static bool var_kmult_only; // always do variable kmult override
  static bool generic_test_flag; // control any features that I'm just testing
  static int choose_time;

  static bool projection_search_enable;

  static float ssq_filter;
  
// Here's a bunch of functions to make us happy
  static void   SetGoptDefaults(char *gopt);
  static void   ReadEmphasisVectorFromFile(char *experimentName);

  static void SetDampenKmult(float damp){dampen_kmult = damp;};
 static void    SetKrateDefaults(float *krate_values)
  {
    for (int i=0;i < NUMNUC;i++)
      krate_default[i] = krate_values[i];
  }
  static void    SetDDefaults(float *d_values)
  {
    for (int i=0;i < NUMNUC;i++)
      d_default[i] = d_values[i];
  }
 static void    SetKmaxDefaults(float *kmax_values)
  {
    for (int i=0;i < NUMNUC;i++)
      kmax_default[i] = kmax_values[i];
  }

 static void    FixRdrInFirst20Flows(bool fixed_RatioDrift) { no_RatioDrift_fit_first_20_flows = fixed_RatioDrift; }
  static void StaticCleanup(void)
  {
    if (glob_flow_ndx_map != NULL) delete [] glob_flow_ndx_map;
    if (flowOrder != NULL) free(flowOrder);
  }
  static void SetFlowOrder(char *_flowOrder);
  static int GetNucNdx(int flow_ndx)
  {
    return glob_flow_ndx_map[flow_ndx%flow_order_len];
  }
  // special functions for double-tap flows
static int IsDoubleTap(int flow)
{
  // may need to refer to earlier flows
  if (flow==0)
    return(1); // can't be double tap
    
   if (glob_flow_ndx_map[flow%flow_order_len]==glob_flow_ndx_map[(flow-1+flow_order_len)%flow_order_len])
     return(0);
   return(1);
}
  static bool GetVarKmultControl(){return(var_kmult_only);};
  static void SetVarKmultControl(bool _var_kmult_only){var_kmult_only = _var_kmult_only;};
  static void SetGenericTestFlag(bool _generic_test_flag){generic_test_flag = _generic_test_flag;};

  static void ReadXtalk(char *name);
  static void SetChipType(char *name);
  static void GetFlowOrderBlock(int *my_flow, int i_start, int i_stop);
};

#endif // GLOBALDEFAULTSFORBKGMODEL_H
