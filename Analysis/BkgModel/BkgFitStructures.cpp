/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgFitStructures.h"
#include "BkgFitOptim.h"

static CpuStep_t Steps[] =
{
  // Index into the intitial float parameters in a particular bead
  {FVAL,       "FVAL",       NULL,  0.00,  CalcFirst,  NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM,     SPECIALCALCULATION}, // fills in fval
  
  {DFDA,       "DFDA",       NULL,  0.01,  CalcBoth,   offsetof(bead_params,Ampl[FIRSTINDEX]), NOTREGIONPARAM,       NOTNUCRISEPARAM ,NUMFB},
  {DFDP,       "DFDP",       NULL,  0.01,  CalcBoth,   offsetof(bead_params,Copies),       NOTREGIONPARAM,       NOTNUCRISEPARAM,SINGLETON},
  {DFDR,       "DFDR",       NULL,  0.01,  CalcBase,   offsetof(bead_params,R),       NOTREGIONPARAM,       NOTNUCRISEPARAM,SINGLETON},
  {DFDDKR,     "DFDDKR",     NULL,  0.01,  CalcBoth,   offsetof(bead_params,kmult[FIRSTINDEX]),   NOTREGIONPARAM,       NOTNUCRISEPARAM,NUMFB},
  {DFDPDM,     "DFDPDM",     NULL,  0.01,  CalcBoth,   offsetof(bead_params,dmult),      NOTREGIONPARAM,       NOTNUCRISEPARAM,SINGLETON},
  
  // bead parameters done as special calculations
  {DFDTSH,     "DFDTSH",     NULL,  0.01,  CalcNone,   NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM,SPECIALCALCULATION},
  {DFDGAIN,    "DFDGAIN",    NULL,  0.00,  CalcNone,   NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM,SPECIALCALCULATION},
 
  // Index into the initial float parameters in a region
// enzyme kinetics parameters
  {DFDD,       "DFDD",       NULL,  1.00,  CalcBoth,   NOTBEADPARAM,  offsetof(reg_params,d[FIRSTINDEX]),     NOTNUCRISEPARAM,NUMNUC},
  {DFDKRATE,   "DFDKRATE",   NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  offsetof(reg_params,krate[FIRSTINDEX]), NOTNUCRISEPARAM,NUMNUC},
  {DFDKMAX,    "DFDKMAX",    NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  offsetof(reg_params,kmax[FIRSTINDEX]),  NOTNUCRISEPARAM,NUMNUC},
  
  {DFDPDR,     "DFDPDR",     NULL, 0.001,  CalcBoth,   NOTBEADPARAM,  offsetof(reg_params,CopyDrift),     NOTNUCRISEPARAM,SINGLETON},
  // buffering hyperparameters
  {DFDRDR,     "DFDRDR",     NULL,  0.01,  CalcBase,   NOTBEADPARAM,  offsetof(reg_params,RatioDrift),      NOTNUCRISEPARAM,SINGLETON},
  {DFDMR,      "DFDMR",      NULL, 0.001,  CalcBase,   NOTBEADPARAM,  offsetof(reg_params,NucModifyRatio[FIRSTINDEX]),    NOTNUCRISEPARAM,NUMNUC},
  {DFDTAUMR,   "DFDTAUMR",   NULL,  0.01,  CalcBase,   NOTBEADPARAM,  offsetof(reg_params,tau_R_m),  NOTNUCRISEPARAM,SINGLETON},
  {DFDTAUOR,   "DFDTAUOR",   NULL,  0.01,  CalcBase,   NOTBEADPARAM,  offsetof(reg_params,tau_R_o),  NOTNUCRISEPARAM,SINGLETON},
  
  // index into the nuc_rise parameters into a particular region
  {DFDT0,      "DFDT0",      NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  NOTREGIONPARAM, offsetof(nuc_rise_params,t_mid_nuc),    SINGLETON},
  {DFDSIGMA,   "DFDSIGMA",   NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  NOTREGIONPARAM, offsetof(nuc_rise_params,sigma),    SINGLETON},
  {DFDT0DLY,   "DFDT0DLY",   NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  NOTREGIONPARAM, offsetof(nuc_rise_params,t_mid_nuc_delay[FIRSTINDEX]), NUMNUC},
  {DFDSMULT,   "DFDSMULT",   NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  NOTREGIONPARAM, offsetof(nuc_rise_params,sigma_mult[FIRSTINDEX]),   NUMNUC},

  {DFDERR,     "DFDERR",     NULL,  0.00,  CalcNone,   NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM, SPECIALCALCULATION},
  {YERR,       "YERR",       NULL,  0.00,  CalcNone,   NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM, SPECIALCALCULATION}
};


//@TODO BAD CODE STYLE global variable
static int NumSteps = sizeof(Steps) /sizeof(Steps[FIRSTINDEX]);



static fit_descriptor fit_Ampl_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDA,          & ((bead_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {TBL_END,       0,                                                        ParamTableEnd    },
};

static fit_descriptor fit_R_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDR,          & ((bead_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {TBL_END,       0,                                                        ParamTableEnd    },
};

static fit_descriptor fit_initial_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDA,          & ((bead_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {DFDR,          & ((bead_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {TBL_END,       0,                                                        ParamTableEnd    },
};

static fit_descriptor fit_post_key_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDA,          & ((bead_params *)(NULL))->Ampl[7]    - (float *) NULL,           ParamTypeNotKey  }, // Not TRUE!!! Cannot be guaranteed 8+ are not key
  {DFDR,          & ((bead_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {DFDP,          & ((bead_params *)(NULL))->Copies          - (float *) NULL,           ParamTypeAllFlow },
  {DFDPDM,        & ((bead_params *)(NULL))->dmult         - (float *) NULL,           ParamTypeAllFlow  },
  {TBL_END,       0,                                                        ParamTableEnd    },
};

static fit_descriptor fit_known_seq_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDR,          & ((bead_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {DFDPDM,        & ((bead_params *)(NULL))->dmult         - (float *) NULL,           ParamTypeAllFlow  },
  {TBL_END,       0,                                                        ParamTableEnd    },
};

static fit_descriptor fit_well_w_bkg_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDA,          & ((bead_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {TBL_END,       0,                                                        ParamTableEnd    },
};

static fit_descriptor fit_well_minimal_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDA,          & ((bead_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {DFDR,          & ((bead_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {TBL_END,       0,                                                        ParamTableEnd    },
};

static fit_descriptor fit_region_init1_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDT0,         & ((reg_params *)(NULL))->nuc_shape.t_mid_nuc      - (float *) NULL,           ParamTypeAllFlow },
  {DFDR,          & ((reg_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {DFDP,          & ((reg_params *)(NULL))->Copies          - (float *) NULL,           ParamTypeAllFlow },
  {DFDA,          & ((reg_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {TBL_END,       0,                                                            ParamTableEnd    },
};

static fit_descriptor fit_region_init2_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDT0,         & ((reg_params *)(NULL))->nuc_shape.t_mid_nuc      - (float *) NULL,           ParamTypeAllFlow },
  {DFDKRATE,      & ((reg_params *)(NULL))->krate[FIRSTINDEX]   - (float *) NULL,           ParamTypePerNuc  },
  {DFDD,          & ((reg_params *)(NULL))->d[FIRSTINDEX]       - (float *) NULL,           ParamTypePerNuc  },
  {DFDSIGMA,      & ((reg_params *)(NULL))->nuc_shape.sigma      - (float *) NULL,           ParamTypeAllFlow },
  {DFDTSH,        & ((reg_params *)(NULL))->tshift     - (float *) NULL,           ParamTypeAllFlow },
  {DFDR,          & ((reg_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {DFDA,          & ((reg_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {DFDP,          & ((reg_params *)(NULL))->Copies          - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUMR,      & ((reg_params *)(NULL))->tau_R_m    - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUOR,      & ((reg_params *)(NULL))->tau_R_o    - (float *) NULL,           ParamTypeAllFlow },
  {DFDRDR,        & ((reg_params *)(NULL))->RatioDrift        - (float *) NULL,           ParamTypeAllFlow },
  {DFDPDR,        & ((reg_params *)(NULL))->CopyDrift        - (float *) NULL,           ParamTypeAllFlow },
  {DFDMR,         & ((reg_params *)(NULL))->NucModifyRatio[FIRSTINDEX]      - (float *) NULL,           ParamTypePerNuc  },
  {TBL_END,       0,                                                               ParamTableEnd    },
};

static fit_descriptor fit_region_full_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDT0,         & ((reg_params *)(NULL))->nuc_shape.t_mid_nuc      - (float *) NULL,           ParamTypeAllFlow },
  {DFDKRATE,      & ((reg_params *)(NULL))->krate[FIRSTINDEX]   - (float *) NULL,           ParamTypePerNuc  },
  {DFDD,          & ((reg_params *)(NULL))->d[FIRSTINDEX]       - (float *) NULL,           ParamTypePerNuc  },
  {DFDSIGMA,      & ((reg_params *)(NULL))->nuc_shape.sigma      - (float *) NULL,           ParamTypeAllFlow },
  {DFDTSH,        & ((reg_params *)(NULL))->tshift     - (float *) NULL,           ParamTypeAllFlow },
  {DFDR,          & ((reg_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {DFDA,          & ((reg_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {DFDP,          & ((reg_params *)(NULL))->Copies          - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUMR,      & ((reg_params *)(NULL))->tau_R_m    - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUOR,      & ((reg_params *)(NULL))->tau_R_o    - (float *) NULL,           ParamTypeAllFlow },
  {DFDRDR,        & ((reg_params *)(NULL))->RatioDrift        - (float *) NULL,           ParamTypeAllFlow },
  {DFDPDR,        & ((reg_params *)(NULL))->CopyDrift        - (float *) NULL,           ParamTypeAllFlow },
  {DFDMR,         & ((reg_params *)(NULL))->NucModifyRatio[FIRSTINDEX]      - (float *) NULL,           ParamTypePerNuc  },
  {TBL_END,       0,                                                            ParamTableEnd    },
};

static fit_descriptor fit_region_init2_noRatioDrift_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDT0,         & ((reg_params *)(NULL))->nuc_shape.t_mid_nuc      - (float *) NULL,           ParamTypeAllFlow },
  {DFDKRATE,      & ((reg_params *)(NULL))->krate[FIRSTINDEX]   - (float *) NULL,           ParamTypePerNuc  },
  {DFDD,          & ((reg_params *)(NULL))->d[FIRSTINDEX]       - (float *) NULL,           ParamTypePerNuc  },
  {DFDSIGMA,      & ((reg_params *)(NULL))->nuc_shape.sigma      - (float *) NULL,           ParamTypeAllFlow },
  {DFDTSH,        & ((reg_params *)(NULL))->tshift     - (float *) NULL,           ParamTypeAllFlow },
  {DFDR,          & ((reg_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {DFDA,          & ((reg_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {DFDP,          & ((reg_params *)(NULL))->Copies          - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUMR,      & ((reg_params *)(NULL))->tau_R_m    - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUOR,      & ((reg_params *)(NULL))->tau_R_o    - (float *) NULL,           ParamTypeAllFlow },
//  {DFDRDR,        & ((reg_params *)(NULL))->RatioDrift        - (float *) NULL,           ParamTypeAllFlow },
  {DFDPDR,        & ((reg_params *)(NULL))->CopyDrift        - (float *) NULL,           ParamTypeAllFlow },
  {DFDMR,         & ((reg_params *)(NULL))->NucModifyRatio[FIRSTINDEX]      - (float *) NULL,           ParamTypePerNuc  },
//  {DFDKMAX,       & ((reg_params *)(NULL))->kmax[FIRSTINDEX]   - (float *) NULL,            ParamTypePerNuc  },
  {DFDT0DLY,      & ((reg_params *)(NULL))->nuc_shape.t_mid_nuc_delay[FIRSTINDEX]   - (float *) NULL,           ParamTypePerNuc  },
  {DFDSMULT,      & ((reg_params *)(NULL))->nuc_shape.sigma_mult[FIRSTINDEX]   - (float *) NULL,      ParamTypePerNuc  },
  {TBL_END,       0,                                                               ParamTableEnd    },
};

static fit_descriptor fit_region_full_noRatioDrift_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDT0,         & ((reg_params *)(NULL))->nuc_shape.t_mid_nuc      - (float *) NULL,           ParamTypeAllFlow },
  {DFDKRATE,      & ((reg_params *)(NULL))->krate[FIRSTINDEX]   - (float *) NULL,           ParamTypePerNuc  },
  {DFDD,          & ((reg_params *)(NULL))->d[FIRSTINDEX]       - (float *) NULL,           ParamTypePerNuc  },
  {DFDSIGMA,      & ((reg_params *)(NULL))->nuc_shape.sigma      - (float *) NULL,           ParamTypeAllFlow },
  {DFDTSH,        & ((reg_params *)(NULL))->tshift     - (float *) NULL,           ParamTypeAllFlow },
  {DFDR,          & ((reg_params *)(NULL))->R          - (float *) NULL,           ParamTypeAllFlow },
  {DFDA,          & ((reg_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {DFDP,          & ((reg_params *)(NULL))->Copies          - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUMR,      & ((reg_params *)(NULL))->tau_R_m    - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUOR,      & ((reg_params *)(NULL))->tau_R_o    - (float *) NULL,           ParamTypeAllFlow },
//  {DFDRDR,        & ((reg_params *)(NULL))->RatioDrift        - (float *) NULL,           ParamTypeAllFlow },
  {DFDPDR,        & ((reg_params *)(NULL))->CopyDrift        - (float *) NULL,           ParamTypeAllFlow },
  {DFDMR,         & ((reg_params *)(NULL))->NucModifyRatio[FIRSTINDEX]      - (float *) NULL,           ParamTypePerNuc  },
//  {DFDKMAX,       & ((reg_params *)(NULL))->kmax[FIRSTINDEX]   - (float *) NULL,            ParamTypePerNuc  },
  {DFDT0DLY,      & ((reg_params *)(NULL))->nuc_shape.t_mid_nuc_delay[FIRSTINDEX]   - (float *) NULL,           ParamTypePerNuc  },
  {DFDSMULT,      & ((reg_params *)(NULL))->nuc_shape.sigma_mult[FIRSTINDEX]   - (float *) NULL,      ParamTypePerNuc  },
  {TBL_END,       0,                                                            ParamTableEnd    },
};

static fit_descriptor fit_region_slim_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDT0,         & ((reg_params *)(NULL))->nuc_shape.t_mid_nuc      - (float *) NULL,           ParamTypeAllFlow },
  {DFDRDR,        & ((reg_params *)(NULL))->RatioDrift        - (float *) NULL,           ParamTypeAllFlow },
  {DFDPDR,        & ((reg_params *)(NULL))->CopyDrift        - (float *) NULL,           ParamTypeAllFlow },
//    {DFDERR,        &((reg_params *)(NULL))->darkness[FIRSTINDEX]     -(float *)NULL,           ParamTypeAllFlow },
  {TBL_END,       0,                                                            ParamTableEnd    },
};

static fit_descriptor fit_region_slim_err_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDERR,        & ((reg_params *)(NULL))->darkness[FIRSTINDEX]     - (float *) NULL,           ParamTypeAllFlow },
  {DFDA,          & ((reg_params *)(NULL))->Ampl[FIRSTINDEX]    - (float *) NULL,           ParamTypePerFlow },
  {TBL_END,       0,                                                            ParamTableEnd    },
};

static master_fit_type_table bkg_model_fit_type[] =
{
//  { "name",               &fit_descriptor,                        NULL,           {NULL,0,NULL,0}    },
  // individual well fits
  { "FitAmpl",             fit_Ampl_descriptor,                   NULL,           {NULL,0,NULL,0}    },
  { "FitR",                fit_R_descriptor,                      NULL,           {NULL,0,NULL,0}    },
  { "FitInitial",          fit_initial_descriptor,                NULL,           {NULL,0,NULL,0}    },
  { "FitPostKey",              fit_post_key_descriptor,                    NULL,           {NULL,0,NULL,0}    },
  { "FitKnownSequence",    fit_known_seq_descriptor,              NULL,           {NULL,0,NULL,0}    },
  { "FitWellWithBkg",      fit_well_w_bkg_descriptor,             NULL,           {NULL,0,NULL,0}    },
  { "FitWellMinimal",      fit_well_minimal_descriptor,           NULL,           {NULL,0,NULL,0}    },
  // region-wide fits
  { "FitRegionInit1",      fit_region_init1_descriptor,           NULL,           {NULL,0,NULL,0}    },
  { "FitRegionInit2",      fit_region_init2_descriptor,           NULL,           {NULL,0,NULL,0}    },
  { "FitRegionFull",       fit_region_full_descriptor,            NULL,           {NULL,0,NULL,0}    },
  { "FitRegionInit2NoRDR", fit_region_init2_noRatioDrift_descriptor,     NULL,           {NULL,0,NULL,0}    },
  { "FitRegionFullNoRDR",  fit_region_full_noRatioDrift_descriptor,      NULL,           {NULL,0,NULL,0}    },
  { "FitRegionSlim",       fit_region_slim_descriptor,            NULL,           {NULL,0,NULL,0}    },
  { "FitRegionSlimErr",    fit_region_slim_err_descriptor,        NULL,           {NULL,0,NULL,0}    },
  { NULL, NULL, NULL, {NULL,0,NULL,0} },  // end of table
};


BkgFitStructures::BkgFitStructures()
{
    Steps    = ::Steps;
    NumSteps = ::NumSteps;

    fit_Ampl_descriptor                      = ::fit_Ampl_descriptor;
    fit_R_descriptor                         = ::fit_R_descriptor;
    fit_initial_descriptor                   = ::fit_initial_descriptor;
    fit_post_key_descriptor                  = ::fit_post_key_descriptor;
    fit_known_seq_descriptor                 = ::fit_known_seq_descriptor;
    fit_well_w_bkg_descriptor                = ::fit_well_w_bkg_descriptor;
    fit_well_minimal_descriptor              = ::fit_well_minimal_descriptor;
    fit_region_init1_descriptor              = ::fit_region_init1_descriptor;
    fit_region_init2_descriptor              = ::fit_region_init2_descriptor;
    fit_region_full_descriptor               = ::fit_region_full_descriptor;
    fit_region_init2_noRatioDrift_descriptor = ::fit_region_init2_noRatioDrift_descriptor;
    fit_region_full_noRatioDrift_descriptor  = ::fit_region_full_noRatioDrift_descriptor;
    fit_region_slim_descriptor               = ::fit_region_slim_descriptor;
    fit_region_slim_err_descriptor           = ::fit_region_slim_err_descriptor;

    bkg_model_fit_type                       = ::bkg_model_fit_type;                    
}

fit_instructions *GetFitInstructionsByName(char *name)
{
  for (int i=0;bkg_model_fit_type[i].name != NULL;i++)
  {
    if (strcmp(bkg_model_fit_type[i].name,name) == 0)
      return &bkg_model_fit_type[i].fi;
  }

  return NULL;
}

// @TODO:  Potential bug here due to historical optimization
// does not update for further blocks of flows
// Needs to update as blocks of flows arrive
// fit instructions are optimized for first block of flows only
// can be in error for later blocks of flows.
void InitializeLevMarSparseMatrices(int *my_nuc_block)
{

  // go through the master table of fit types and generate all the build
  // instructions for each type of fitting we are going to do
  for (int i=0;bkg_model_fit_type[i].name != NULL;i++)
    CreateBuildInstructions(&bkg_model_fit_type[i], my_nuc_block);

//    DumpBuildInstructionTable(bkg_model_fit_type[6].mb);
}

void CleanupLevMarSparseMatrices(void)
{
  for (int i=0;bkg_model_fit_type[i].name != NULL;i++)
  {
    struct master_fit_type_table *ft = &bkg_model_fit_type[i];

    // make sure there is a high-level descriptor for this row
    // if there wasn't one, then the row might contain a hard link to
    // a statically allocated matrix build instruction which we don't
    // want to free
    if (ft->fd != NULL)
    {
      if (ft->mb != NULL)
      {
        delete [](ft->mb);
        ft->mb = NULL;
      }
    }

    if (ft->fi.input != NULL)
      delete [] ft->fi.input;

    if (ft->fi.output != NULL)
      delete [] ft->fi.output;

    ft->fi.input  = NULL;
    ft->fi.output = NULL;
  }
  
  // TODO:  This is really bad as well!!!!
  // Objects should be isolated or passed to functions that operate on them.

}

//#define NUMERIC_PartialDeriv_CALC
void BuildMatrix(BkgFitMatrixPacker *fit,bool accum, bool debug)
{
#if 1
    (void) debug;
    fit->BuildMatrix(accum);
#else
    mat_assembly_instruction *pinst = fit->instList;
    int lineInc=0;

    // build JTJ and RHS matricies
    for (int i=0;i < fit->nInstr;i++)
    {
        double sum=0.0;

        for (int j=0;j < pinst->cnt;j++)
            sum += cblas_sdot(pinst->si[j].len,pinst->si[j].src1,1,pinst->si[j].src2,1);
        if (accum)
            * (pinst->dst) += sum;
        else
            * (pinst->dst) = sum;

        if (debug)
        {
            char *src1Name = findName(pinst->si[0].src1);
            char *src2Name = findName(pinst->si[0].src2);

            printf("%d(%s--%s)(%lf) ",i,src1Name,src2Name,* (pinst->dst));

            if (lineInc++ > 6)
            {
                printf("\n  ");
                lineInc = 0;
            }
        }
        pinst++;
    }
#endif
}

