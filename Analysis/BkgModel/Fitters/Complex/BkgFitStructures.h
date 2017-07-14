/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITSTRUCTURES_H
#define BKGFITSTRUCTURES_H

#include "BkgFitOptim.h"

class BkgFitStructures{
  BkgFitStructures();   // Not implemented. This class holds on to some global structures.
                        // There's never a reason to build one of these.
public:
  
  static CpuStep Steps[];
  static int NumSteps;

  /*
  static fit_descriptor fit_well_ampl_descriptor[];
  static fit_descriptor fit_well_ampl_buffering_descriptor[];
  static fit_descriptor fit_well_post_key_descriptor[];
  static fit_descriptor fit_well_all_descriptor[];
  static fit_descriptor fit_well_post_key_descriptor_nodmult[];

  // region
  static fit_descriptor fit_region_tmidnuc_plus_descriptor[];
  
  static fit_descriptor fit_region_init2_descriptor[];
  static fit_descriptor fit_region_init2_taue_descriptor[];
  static fit_descriptor fit_region_init2_noRatioDrift_descriptor[];
  static fit_descriptor fit_region_init2_taue_NoRDR_descriptor[];
  
  static fit_descriptor fit_region_full_descriptor[];
  static fit_descriptor fit_region_full_taue_descriptor[];
  static fit_descriptor fit_region_full_noRatioDrift_descriptor[];
  static fit_descriptor fit_region_full_taue_NoRDR_descriptor[];
  
  static fit_descriptor fit_region_time_varying_descriptor[];
  static fit_descriptor fit_region_darkness_descriptor[];
  
  static fit_descriptor fit_region_init2_taue_NoD_descriptor[];
  static fit_descriptor fit_region_init2_taue_NoRDR_NoD_descriptor[];
  static fit_descriptor fit_region_full_taue_NoD_descriptor[];
  static fit_descriptor fit_region_full_taue_NoRDR_NoD_descriptor[];

 std::vector<fit_descriptor> fit_well_ampl_descriptor;
 std::vector<fit_descriptor> fit_well_ampl_buffering_descriptor;
 std::vector<fit_descriptor> fit_well_post_key_descriptor;
 std::vector<fit_descriptor> fit_well_post_key_descriptor_nodmult;

  // region
 std::vector<fit_descriptor> fit_region_tmidnuc_plus_descriptor;

 std::vector<fit_descriptor> fit_region_init2_descriptor;
 std::vector<fit_descriptor> fit_region_init2_taue_descriptor;
 std::vector<fit_descriptor> fit_region_init2_noRatioDrift_descriptor;
 std::vector<fit_descriptor> fit_region_init2_taue_NoRDR_descriptor;

 std::vector<fit_descriptor> fit_region_full_descriptor;
 std::vector<fit_descriptor> fit_region_full_taue_descriptor;
 std::vector<fit_descriptor> fit_region_full_noRatioDrift_descriptor;
 std::vector<fit_descriptor> fit_region_full_taue_NoRDR_descriptor;

 std::vector<fit_descriptor> fit_region_time_varying_descriptor;
 std::vector<fit_descriptor> fit_region_darkness_descriptor;

 std::vector<fit_descriptor> fit_region_init2_taue_NoD_descriptor;
 std::vector<fit_descriptor> fit_region_init2_taue_NoRDR_NoD_descriptor;
 std::vector<fit_descriptor> fit_region_full_taue_NoD_descriptor;
 std::vector<fit_descriptor> fit_region_full_taue_NoRDR_NoD_descriptor;
  */

  static int GetNumParamsToFitForDescriptor(const std::vector<fit_descriptor>& fds, int flow_key, int flow_block_size);
  static int GetNumParDerivStepsForFitDescriptor(const std::vector<fit_descriptor>& fds);
};


//#define NUMERIC_PartialDeriv_CALC
void BuildMatrix(BkgFitMatrixPacker *fit, bool accum, bool debug);

#endif // BKGFITSTRUCTURES_H

