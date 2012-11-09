/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITSTRUCTURES_H
#define BKGFITSTRUCTURES_H

#include "BkgFitOptim.h"

struct BkgFitStructures{
  BkgFitStructures();
  
  CpuStep_t* Steps;
  int NumSteps;

  fit_descriptor* fit_well_ampl_descriptor;
  fit_descriptor* fit_well_ampl_buffering_descriptor;
  fit_descriptor* fit_well_post_key_descriptor;

  // region
  fit_descriptor* fit_region_tmidnuc_plus_descriptor;
  
  fit_descriptor* fit_region_init2_descriptor;
  fit_descriptor* fit_region_init2_taue_descriptor;
  fit_descriptor* fit_region_init2_noRatioDrift_descriptor;
  fit_descriptor* fit_region_init2_taue_NoRDR_descriptor;
  
  fit_descriptor* fit_region_full_descriptor;
  fit_descriptor* fit_region_full_taue_descriptor;
  fit_descriptor* fit_region_full_noRatioDrift_descriptor;
  fit_descriptor* fit_region_full_taue_NoRDR_descriptor;
  
  fit_descriptor* fit_region_time_varying_descriptor;
  fit_descriptor* fit_region_darkness_descriptor;
  
  master_fit_type_table* bkg_model_fit_type;
};


fit_instructions *GetFitInstructionsByName(char *name);
fit_descriptor *GetFitDescriptorByName(const char* name);
int GetNumParamsToFitForDescriptor(fit_descriptor *fd);
int GetNumParDerivStepsForFitDescriptor(fit_descriptor* fd);

// TODO: PartialDeriv 'affected flows' has to be munged for tango flow order!!! */
void InitializeLevMarSparseMatrices(int *my_nuc_block);
void CleanupLevMarSparseMatrices(void);

//#define NUMERIC_PartialDeriv_CALC
void BuildMatrix(BkgFitMatrixPacker *fit,bool accum, bool debug);

#endif // BKGFITSTRUCTURES_H

