/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITSTRUCTURES_H
#define BKGFITSTRUCTURES_H

#include "BkgFitOptim.h"

struct BkgFitStructures{
    BkgFitStructures();

    CpuStep_t* Steps;
    int NumSteps;

    fit_descriptor* fit_Ampl_descriptor;
    fit_descriptor* fit_R_descriptor;
    fit_descriptor* fit_initial_descriptor;
    fit_descriptor* fit_post_key_descriptor;
    fit_descriptor* fit_known_seq_descriptor;
    fit_descriptor* fit_well_w_bkg_descriptor;
    fit_descriptor* fit_well_minimal_descriptor;
    fit_descriptor* fit_region_init1_descriptor;
    fit_descriptor* fit_region_init2_descriptor;
    fit_descriptor* fit_region_full_descriptor;
    fit_descriptor* fit_region_init2_noRatioDrift_descriptor;
    fit_descriptor* fit_region_full_noRatioDrift_descriptor;
    fit_descriptor* fit_region_slim_descriptor;
    fit_descriptor* fit_region_slim_err_descriptor;

    master_fit_type_table* bkg_model_fit_type;
};


fit_instructions *GetFitInstructionsByName(char *name);

// TODO: PartialDeriv 'affected flows' has to be munged for tango flow order!!! */
void InitializeLevMarSparseMatrices(int *my_nuc_block);
void CleanupLevMarSparseMatrices(void);

//#define NUMERIC_PartialDeriv_CALC
void BuildMatrix(BkgFitMatrixPacker *fit,bool accum, bool debug);

#endif // BKGFITSTRUCTURES_H

