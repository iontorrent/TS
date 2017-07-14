/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <cstddef>
#include <string>
#include <vector>
#include "BkgFitStructures.h"

using namespace std;

CpuStep BkgFitStructures::Steps[] =
{
  // Index into the intitial float parameters in a particular bead
  {FVAL,       "FVAL",       NULL,  0.00,  CalcFirst,  NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM,     CpuStep::SpecialCalculation}, // fills in fval
// basic properties of beads
  {DFDR,       "DFDR",       NULL,  0.01,  CalcBase,   &BeadParams::AccessR,       NOTREGIONPARAM,       NOTNUCRISEPARAM,CpuStep::Singleton},
  {DFDP,       "DFDP",       NULL,  0.01,  CalcBoth,   &BeadParams::AccessCopies,       NOTREGIONPARAM,       NOTNUCRISEPARAM,CpuStep::Singleton},
  {DFDPDM,     "DFDPDM",     NULL,  0.01,  CalcBoth,   &BeadParams::AccessDmult,      NOTREGIONPARAM,       NOTNUCRISEPARAM,CpuStep::Singleton},
 // per flow updated parameters
  {DFDA,       "DFDA",       NULL,  0.01,  CalcBoth,   &BeadParams::AccessAmpl, NOTREGIONPARAM,       NOTNUCRISEPARAM ,CpuStep::PerFlow},
  {DFDDKR,     "DFDDKR",     NULL,  0.01,  CalcBoth,   &BeadParams::AccessKmult,   NOTREGIONPARAM,       NOTNUCRISEPARAM,CpuStep::PerFlow},
  
  // bead parameters done as special calculations
  {DFDGAIN,    "DFDGAIN",    NULL,  0.00,  CalcNone,   NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM,CpuStep::SpecialCalculation},
 
  // Index into the initial float parameters in a region
  // special timing for empty trace
  {DFDTSH,     "DFDTSH",     NULL,  0.01,  CalcNone,   NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM,CpuStep::SpecialCalculation},
  // index into the nuc_rise parameters into a particular region
  {DFDT0,      "DFDT0",      NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  NOTREGIONPARAM, &nuc_rise_params::AccessTMidNuc,    CpuStep::Singleton},
  {DFDSIGMA,   "DFDSIGMA",   NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  NOTREGIONPARAM, &nuc_rise_params::AccessSigma,    CpuStep::Singleton},
  {DFDT0DLY,   "DFDT0DLY",   NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  NOTREGIONPARAM, &nuc_rise_params::AccessTMidNucDelay, CpuStep::PerNuc},
  {DFDSMULT,   "DFDSMULT",   NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  NOTREGIONPARAM, &nuc_rise_params::AccessSigmaMult,   CpuStep::PerNuc},
// enzyme kinetics parameters
  {DFDD,       "DFDD",       NULL,  1.00,  CalcBoth,   NOTBEADPARAM,  &reg_params::AccessD,     NOTNUCRISEPARAM,CpuStep::PerNuc},
  {DFDKRATE,   "DFDKRATE",   NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  &reg_params::AccessKrate, NOTNUCRISEPARAM,CpuStep::PerNuc},
  {DFDKMAX,    "DFDKMAX",    NULL,  0.01,  CalcBoth,   NOTBEADPARAM,  &reg_params::AccessKmax,  NOTNUCRISEPARAM,CpuStep::PerNuc},
  
  // buffering hyperparameters
  {DFDMR,      "DFDMR",      NULL, 0.001,  CalcBase,   NOTBEADPARAM,  &reg_params::AccessNucModifyRatio,    NOTNUCRISEPARAM,CpuStep::PerNuc},
  // one way of making well buffering consistent across region
  {DFDTAUMR,   "DFDTAUMR",   NULL,  0.01,  CalcBase,   NOTBEADPARAM,  &reg_params::AccessTauRM,  NOTNUCRISEPARAM,CpuStep::Singleton},
  {DFDTAUOR,   "DFDTAUOR",   NULL,  0.01,  CalcBase,   NOTBEADPARAM,  &reg_params::AccessTauRO,  NOTNUCRISEPARAM,CpuStep::Singleton},
  // alternate parameter to stabilize wells
  {DFDTAUE,   "DFDTAUE",   NULL,  0.01,  CalcBase,   NOTBEADPARAM,  &reg_params::AccessTauE,  NOTNUCRISEPARAM,CpuStep::Singleton},
  
  // time varying parameters
   {DFDRDR,     "DFDRDR",     NULL,  0.01,  CalcBase,   NOTBEADPARAM,  &reg_params::AccessRatioDrift,      NOTNUCRISEPARAM,CpuStep::Singleton},
   {DFDPDR,     "DFDPDR",     NULL, 0.001,  CalcBoth,   NOTBEADPARAM,  &reg_params::AccessCopyDrift,     NOTNUCRISEPARAM,CpuStep::Singleton},
  
  // special for lev-mar calculations
  {DFDERR,     "DFDERR",     NULL,  0.00,  CalcNone,   NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM, CpuStep::SpecialCalculation},
  {YERR,       "YERR",       NULL,  0.00,  CalcNone,   NOTBEADPARAM, NOTREGIONPARAM, NOTNUCRISEPARAM, CpuStep::SpecialCalculation}
};


int BkgFitStructures::NumSteps = sizeof(Steps) /sizeof(Steps[FIRSTINDEX]);


/*
fit_descriptor BkgFitStructures::fit_well_ampl_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
  {DFDA,      & BeadParams::AccessAmpl, 0,      ParamTypePerFlow },
  {TBL_END,   0,                         0,      ParamTableEnd    },
};


fit_descriptor BkgFitStructures::fit_well_ampl_buffering_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
  {DFDR,      & BeadParams::AccessR,    0,      ParamTypeAllFlow },
  {DFDA,      & BeadParams::AccessAmpl, 0,      ParamTypePerFlow },
  {TBL_END,   0,                         0,      ParamTableEnd    },
};


fit_descriptor BkgFitStructures::fit_well_post_key_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
  {DFDR,      & BeadParams::AccessR,           0,   ParamTypeAllFlow },
  {DFDP,      & BeadParams::AccessCopies,      0,   ParamTypeAllFlow },
  {DFDPDM,    & BeadParams::AccessDmult,       0,   ParamTypeAllFlow  },
  {DFDA,      & BeadParams::AccessAmplPostKey, 0,   ParamTypeNotKey  }, // Not TRUE!!! Cannot be guaranteed 8+ are not key
  {TBL_END,   0,                                0,   ParamTableEnd    },
};


fit_descriptor BkgFitStructures::fit_well_all_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
  {DFDR,      & BeadParams::AccessR,           0,   ParamTypeAllFlow },
  {DFDP,      & BeadParams::AccessCopies,      0,   ParamTypeAllFlow },
  {DFDPDM,    & BeadParams::AccessDmult,       0,   ParamTypeAllFlow  },
  {DFDA,      & BeadParams::AccessAmpl, 0,   ParamTypePerFlow  }, // Not TRUE!!! Cannot be guaranteed 8+ are not key
  {DFDDKR,      & BeadParams::AccessKmult, 0,   ParamTypePerFlow  }, // Not TRUE!!! Cannot be guaranteed 8+ are not key
  {TBL_END,   0,                                0,   ParamTableEnd    },
};


fit_descriptor BkgFitStructures::fit_well_post_key_descriptor_nodmult[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
  {DFDR,          & BeadParams::AccessR,       0,           ParamTypeAllFlow },
  {DFDP,          & BeadParams::AccessCopies,  0,           ParamTypeAllFlow },
  //{DFDPDM,        & ((BeadParams *)(NULL))->dmult         - (float *) NULL,           ParamTypeAllFlow  },
  {DFDA,          & BeadParams::AccessAmplPostKey,    0,           ParamTypeNotKey  }, // Not TRUE!!! Cannot be guaranteed 8+ are not key
   {TBL_END,   0,                                0,   ParamTableEnd    },
};



fit_descriptor BkgFitStructures::fit_region_tmidnuc_plus_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
// bead modifying parameters
  {DFDR,     0,   & reg_params::AccessR,              ParamTypeAllFlow },
  {DFDP,     0,   & reg_params::AccessCopies,         ParamTypeAllFlow },
  {DFDA,     0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// timing of nuc rise shape
  {DFDT0,    0,   & reg_params::AccessTMidNuc,        ParamTypeAllFlow },
  {TBL_END,  0,   0,                                  ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_init2_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
// Parameters modifying beads
  {DFDR,     0,   & reg_params::AccessR,              ParamTypeAllFlow },
  {DFDP,     0,   & reg_params::AccessCopies,         ParamTypeAllFlow },
  {DFDA,     0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,   0,   & reg_params::AccessTShift,         ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,    0,   & reg_params::AccessTMidNuc,        ParamTypeAllFlow },
  {DFDSIGMA, 0,   & reg_params::AccessSigma,          ParamTypeAllFlow },
// Enzyme kinetics
  {DFDKRATE, 0,   & reg_params::AccessKrate,          ParamTypePerNuc  },
  {DFDD,     0,   & reg_params::AccessD,              ParamTypePerNuc  },
// buffering parameters
  {DFDMR,    0,   & reg_params::AccessNucModifyRatio, ParamTypePerNuc  },
  {DFDTAUMR, 0,   & reg_params::AccessTauRM,          ParamTypeAllFlow },
  {DFDTAUOR, 0,   & reg_params::AccessTauRO,          ParamTypeAllFlow },
// time varying parameters
  {DFDRDR,   0,   & reg_params::AccessRatioDrift,     ParamTypeAllFlow },
  {DFDPDR,   0,   & reg_params::AccessCopyDrift,      ParamTypeAllFlow },
  {TBL_END,  0,   0,                                  ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_init2_taue_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
// Parameters modifying beads
  {DFDR,     0,   & reg_params::AccessR,              ParamTypeAllFlow },
  {DFDP,     0,   & reg_params::AccessCopies,         ParamTypeAllFlow },
  {DFDA,     0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,   0,   & reg_params::AccessTShift,         ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,    0,   & reg_params::AccessTMidNuc,        ParamTypeAllFlow },
  {DFDSIGMA, 0,   & reg_params::AccessSigma,          ParamTypeAllFlow },
// Enzyme kinetics
  {DFDKRATE, 0,   & reg_params::AccessKrate,          ParamTypePerNuc  },
  {DFDD,     0,   & reg_params::AccessD,              ParamTypePerNuc  },
// buffering parameters
  {DFDMR,    0,   & reg_params::AccessNucModifyRatio, ParamTypePerNuc  },
//{DFDTAUMR, 0,   & reg_params::AccessTauRM,          ParamTypeAllFlow },
//{DFDTAUOR, 0,   & reg_params::AccessTauRO,          ParamTypeAllFlow },
  {DFDTAUE,  0,   & reg_params::AccessTauE,           ParamTypeAllFlow },
// time varying parameters
  {DFDRDR,   0,   & reg_params::AccessRatioDrift,     ParamTypeAllFlow },
  {DFDPDR,   0,   & reg_params::AccessCopyDrift,      ParamTypeAllFlow },
  {TBL_END,  0,   0,                                  ParamTableEnd    },
};


fit_descriptor BkgFitStructures::fit_region_init2_taue_NoRDR_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
// Parameters modifying beads
  {DFDR,     0,   & reg_params::AccessR,              ParamTypeAllFlow },
  {DFDP,     0,   & reg_params::AccessCopies,         ParamTypeAllFlow },
  {DFDA,     0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,   0,   & reg_params::AccessTShift,         ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,    0,   & reg_params::AccessTMidNuc,        ParamTypeAllFlow },
  {DFDSIGMA, 0,   & reg_params::AccessSigma,          ParamTypeAllFlow },
// Enzyme kinetics
  {DFDKRATE, 0,   & reg_params::AccessKrate,          ParamTypePerNuc  },
  {DFDD,     0,   & reg_params::AccessD,              ParamTypePerNuc  },
// buffering parameters
  {DFDMR,    0,   & reg_params::AccessNucModifyRatio, ParamTypePerNuc  },
//{DFDTAUMR, 0,   & reg_params::AccessTauRM,          ParamTypeAllFlow },
//{DFDTAUOR, 0,   & reg_params::AccessTauRO,          ParamTypeAllFlow },
  {DFDTAUE,  0,   & reg_params::AccessTauE,           ParamTypeAllFlow },
// time varying parameters
//{DFDRDR,   0,   & reg_params::AccessRatioDrift,     ParamTypeAllFlow },
  {DFDPDR,   0,   & reg_params::AccessCopyDrift,      ParamTypeAllFlow },
  {TBL_END,  0,   0,                                  ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_full_taue_NoRDR_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
// parameters modifying beads
  {DFDR,     0,   & reg_params::AccessR,              ParamTypeAllFlow },
  {DFDP,     0,   & reg_params::AccessCopies,         ParamTypeAllFlow },
  {DFDA,     0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,   0,   & reg_params::AccessTShift,         ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,    0,   & reg_params::AccessTMidNuc,        ParamTypeAllFlow },
  {DFDSIGMA, 0,   & reg_params::AccessSigma,          ParamTypeAllFlow },
// enzyme kinetics
  {DFDKRATE, 0,   & reg_params::AccessKrate,          ParamTypePerNuc  },
  {DFDD,     0,   & reg_params::AccessD,              ParamTypePerNuc  },
// buffering parameters
  {DFDMR,    0,   & reg_params::AccessNucModifyRatio, ParamTypePerNuc  },
//{DFDTAUMR, 0,   & reg_params::AccessTauRM,          ParamTypeAllFlow },
//{DFDTAUOR, 0,   & reg_params::AccessTauRO,          ParamTypeAllFlow },
  {DFDTAUE,  0,   & reg_params::AccessTauE,           ParamTypeAllFlow },
// time varying parameters
//{DFDRDR,   0,   & reg_params::AccessRatioDrift,     ParamTypeAllFlow },
  {DFDPDR,   0,   & reg_params::AccessCopyDrift,      ParamTypeAllFlow },
  {TBL_END,  0,   0,                                  ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_full_taue_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
// parameters modifying beads
  {DFDR,     0,   & reg_params::AccessR,              ParamTypeAllFlow },
  {DFDP,     0,   & reg_params::AccessCopies,         ParamTypeAllFlow },
  {DFDA,     0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,   0,   & reg_params::AccessTShift,         ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,    0,   & reg_params::AccessTMidNuc,        ParamTypeAllFlow },
  {DFDSIGMA, 0,   & reg_params::AccessSigma,          ParamTypeAllFlow },
// enzyme kinetics
  {DFDKRATE, 0,   & reg_params::AccessKrate,          ParamTypePerNuc  },
  {DFDD,     0,   & reg_params::AccessD,              ParamTypePerNuc  },
// buffering parameters
  {DFDMR,    0,   & reg_params::AccessNucModifyRatio, ParamTypePerNuc  },
//{DFDTAUMR, 0,   & reg_params::AccessTauRM,          ParamTypeAllFlow },
//{DFDTAUOR, 0,   & reg_params::AccessTauRO,          ParamTypeAllFlow },
  {DFDTAUE,  0,   & reg_params::AccessTauE,           ParamTypeAllFlow },
// time varying parameters
  {DFDRDR,   0,   & reg_params::AccessRatioDrift,     ParamTypeAllFlow },
  {DFDPDR,   0,   & reg_params::AccessCopyDrift,      ParamTypeAllFlow },
  {TBL_END,  0,   0,                                  ParamTableEnd    },
};


fit_descriptor BkgFitStructures::fit_region_full_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
// parameters modifying beads
  {DFDR,     0,   & reg_params::AccessR,              ParamTypeAllFlow },
  {DFDP,     0,   & reg_params::AccessCopies,         ParamTypeAllFlow },
  {DFDA,     0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,   0,   & reg_params::AccessTShift,         ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,    0,   & reg_params::AccessTMidNuc,        ParamTypeAllFlow },
  {DFDSIGMA, 0,   & reg_params::AccessSigma,          ParamTypeAllFlow },
// enzyme kinetics
  {DFDKRATE, 0,   & reg_params::AccessKrate,          ParamTypePerNuc  },
  {DFDD,     0,   & reg_params::AccessD,              ParamTypePerNuc  },
// buffering parameters
  {DFDMR,    0,   & reg_params::AccessNucModifyRatio, ParamTypePerNuc  },
  {DFDTAUMR, 0,   & reg_params::AccessTauRM,          ParamTypeAllFlow },
  {DFDTAUOR, 0,   & reg_params::AccessTauRO,          ParamTypeAllFlow },
  // time varying parameters
  {DFDRDR,   0,   & reg_params::AccessRatioDrift,     ParamTypeAllFlow },
  {DFDPDR,   0,   & reg_params::AccessCopyDrift,      ParamTypeAllFlow },
  {TBL_END,  0,   0,                                  ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_init2_noRatioDrift_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
// bead modifying parameters
  {DFDR,     0,   & reg_params::AccessR,              ParamTypeAllFlow },
  {DFDP,     0,   & reg_params::AccessCopies,         ParamTypeAllFlow },
  {DFDA,     0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,   0,   & reg_params::AccessTShift,         ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,    0,   & reg_params::AccessTMidNuc,        ParamTypeAllFlow },
  {DFDSIGMA, 0,   & reg_params::AccessSigma,          ParamTypeAllFlow },
  {DFDT0DLY, 0,   & reg_params::AccessTMidNucDelay,   ParamTypePerNuc  },
  {DFDSMULT, 0,   & reg_params::AccessSigmaMult,      ParamTypePerNuc  },
// enzyme kinetics
  {DFDKRATE, 0,   & reg_params::AccessKrate,          ParamTypePerNuc  },
  {DFDD,     0,   & reg_params::AccessD,              ParamTypePerNuc  },
// buffering parameters
  {DFDMR,    0,   & reg_params::AccessNucModifyRatio, ParamTypePerNuc  },
  {DFDTAUMR, 0,   & reg_params::AccessTauRM,          ParamTypeAllFlow },
  {DFDTAUOR, 0,   & reg_params::AccessTauRO,          ParamTypeAllFlow },
// time varying parameters
//{DFDRDR,   0,   & reg_params::AccessRatioDrift,     ParamTypeAllFlow },
  {DFDPDR,   0,   & reg_params::AccessCopyDrift,      ParamTypeAllFlow },
//{DFDKMAX,  0,   & reg_params::AccessKmax,           ParamTypePerNuc  },
  {TBL_END,  0,   0,                                  ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_full_noRatioDrift_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
// bead modifying parameters
  {DFDR,     0,   & reg_params::AccessR,              ParamTypeAllFlow },
  {DFDP,     0,   & reg_params::AccessCopies,         ParamTypeAllFlow },
  {DFDA,     0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace timing
  {DFDTSH,   0,   & reg_params::AccessTShift,         ParamTypeAllFlow },
// parameters modifying nuc rise
  {DFDT0,    0,   & reg_params::AccessTMidNuc,        ParamTypeAllFlow },
  {DFDSIGMA, 0,   & reg_params::AccessSigma,          ParamTypeAllFlow },
  {DFDT0DLY, 0,   & reg_params::AccessTMidNucDelay,   ParamTypePerNuc  },
  {DFDSMULT, 0,   & reg_params::AccessSigmaMult,      ParamTypePerNuc  },
// enzyme kinetics
  {DFDKRATE, 0,   & reg_params::AccessKrate,          ParamTypePerNuc  },
  {DFDD,     0,   & reg_params::AccessD,              ParamTypePerNuc  },
// buffering parameters
  {DFDMR,    0,   & reg_params::AccessNucModifyRatio, ParamTypePerNuc  },
  {DFDTAUMR, 0,   & reg_params::AccessTauRM,          ParamTypeAllFlow },
  {DFDTAUOR, 0,   & reg_params::AccessTauRO,          ParamTypeAllFlow },
// time varying parameters
//{DFDRDR,   0,    & reg_params::AccessRatioDrift,    ParamTypeAllFlow },
  {DFDPDR,   0,    & reg_params::AccessCopyDrift,     ParamTypeAllFlow },
//{DFDKMAX,  0,   & reg_params::AccessKmax,           ParamTypePerNuc  },
  {TBL_END,  0,  0,                                   ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_time_varying_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
  {DFDT0,    0,    & reg_params::AccessTMidNuc,       ParamTypeAllFlow },
// time varying parameters
  {DFDRDR,   0,    & reg_params::AccessRatioDrift,    ParamTypeAllFlow },
  {DFDPDR,   0,    & reg_params::AccessCopyDrift,     ParamTypeAllFlow },
//{DFDERR,   0,    & reg_params::AccessDarkness,      ParamTypeAllFlow },
  {TBL_END,  0,    0,                                 ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_darkness_descriptor[] =
{
//{PartialDerivComponent, bead_params_func, reg_params_func, ParameterSensitivityClassification}
  {DFDERR,   0,  & reg_params::AccessDarkness,   ParamTypeAllFlow },
  {DFDA,     0,  & reg_params::AccessAmpl,       ParamTypePerFlow },
  {TBL_END,  0,  0,                              ParamTableEnd    },
};


fit_descriptor BkgFitStructures::fit_region_init2_taue_NoD_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,               ParameterSensitivityClassification}
// Parameters modifying beads
  {DFDR,   0,        & reg_params::AccessR,           ParamTypeAllFlow },
  {DFDP,   0,       & reg_params::AccessCopies,           ParamTypeAllFlow },
  {DFDA,   0,       & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,  0,      & reg_params::AccessTShift,           ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,    0,     & reg_params::AccessTMidNuc,           ParamTypeAllFlow },
  {DFDSIGMA,  0,    & reg_params::AccessSigma,           ParamTypeAllFlow },
// Enzyme kinetics
  {DFDKRATE,  0,    & reg_params::AccessKrate,           ParamTypePerNuc  },
  //{DFDD,    0,      & ((reg_params *)(NULL))->d[FIRSTINDEX]       - (float *) NULL,           ParamTypePerNuc  },
// buffering parameters
  {DFDMR,   0,      & reg_params::AccessNucModifyRatio,           ParamTypePerNuc  },
  //  {DFDTAUMR,  0,    & ((reg_params *)(NULL))->tau_R_m    - (float *) NULL,           ParamTypeAllFlow },
  //{DFDTAUOR,   0,   & ((reg_params *)(NULL))->tau_R_o    - (float *) NULL,           ParamTypeAllFlow },
   {DFDTAUE,   0,   & reg_params::AccessTauE,           ParamTypeAllFlow },
// time varying parameters
  {DFDRDR,   0,     & reg_params::AccessRatioDrift,           ParamTypeAllFlow },
  {DFDPDR,    0,    & reg_params::AccessCopyDrift,           ParamTypeAllFlow },
  {TBL_END,   0,    0,                                                               ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_init2_taue_NoRDR_NoD_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
// Parameters modifying beads
  {DFDR,   0,        & reg_params::AccessR,           ParamTypeAllFlow },
  {DFDP,   0,       & reg_params::AccessCopies,           ParamTypeAllFlow },
  {DFDA,   0,       & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,  0,      & reg_params::AccessTShift,           ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,    0,     & reg_params::AccessTMidNuc,           ParamTypeAllFlow },
  {DFDSIGMA,  0,    & reg_params::AccessSigma,           ParamTypeAllFlow },
// Enzyme kinetics
  {DFDKRATE,  0,    & reg_params::AccessKrate,           ParamTypePerNuc  },
  //{DFDD,    0,      & ((reg_params *)(NULL))->d[FIRSTINDEX]       - (float *) NULL,           ParamTypePerNuc  },
// buffering parameters
  {DFDMR,   0,      & reg_params::AccessNucModifyRatio,           ParamTypePerNuc  },
  //  {DFDTAUMR,  0,    & ((reg_params *)(NULL))->tau_R_m    - (float *) NULL,           ParamTypeAllFlow },
  //{DFDTAUOR,   0,   & ((reg_params *)(NULL))->tau_R_o    - (float *) NULL,           ParamTypeAllFlow },
   {DFDTAUE,   0,   & reg_params::AccessTauE,           ParamTypeAllFlow },
// time varying parameters
//  {DFDRDR,   0,     & reg_params::AccessRatioDrift,           ParamTypeAllFlow },
  {DFDPDR,    0,    & reg_params::AccessCopyDrift,           ParamTypeAllFlow },
  {TBL_END,   0,    0,                                                               ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_full_taue_NoD_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
// parameters modifying beads
  {DFDR,      0,    & reg_params::AccessR,           ParamTypeAllFlow },
  {DFDP,       0,   & reg_params::AccessCopies,           ParamTypeAllFlow },
  {DFDA,       0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,    0,    & reg_params::AccessTShift,           ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,     0,    & reg_params::AccessTMidNuc,           ParamTypeAllFlow },
  {DFDSIGMA,  0,    & reg_params::AccessSigma,           ParamTypeAllFlow },
// enzyme kinetics
  {DFDKRATE,   0,   & reg_params::AccessKrate,           ParamTypePerNuc  },
  //{DFDD,     0,     & ((reg_params *)(NULL))->d[FIRSTINDEX]       - (float *) NULL,           ParamTypePerNuc  },
// buffering parameters
  {DFDMR,     0,    & reg_params::AccessNucModifyRatio,           ParamTypePerNuc  },
  //{DFDTAUMR,  0,    & ((reg_params *)(NULL))->tau_R_m    - (float *) NULL,           ParamTypeAllFlow },
  //{DFDTAUOR,   0,   & ((reg_params *)(NULL))->tau_R_o    - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUE,  0,    & reg_params::AccessTauE,           ParamTypeAllFlow },
// time varying parameters
  {DFDRDR,   0,     & reg_params::AccessRatioDrift,           ParamTypeAllFlow },
  {DFDPDR,   0,     & reg_params::AccessCopyDrift,           ParamTypeAllFlow },
  {TBL_END,  0,     0,                                                            ParamTableEnd    },
};

fit_descriptor BkgFitStructures::fit_region_full_taue_NoRDR_NoD_descriptor[] =
{
//  {PartialDerivComponent,  param_ndx,                                          ParameterSensitivityClassification}
// parameters modifying beads
  {DFDR,      0,    & reg_params::AccessR,           ParamTypeAllFlow },
  {DFDP,       0,   & reg_params::AccessCopies,           ParamTypeAllFlow },
  {DFDA,       0,   & reg_params::AccessAmpl,           ParamTypePerFlow },
// parameter modifying empty trace
  {DFDTSH,    0,    & reg_params::AccessTShift,           ParamTypeAllFlow },
// timing of nuc rise shape
  {DFDT0,     0,    & reg_params::AccessTMidNuc,           ParamTypeAllFlow },
  {DFDSIGMA,  0,    & reg_params::AccessSigma,           ParamTypeAllFlow },
// enzyme kinetics
  {DFDKRATE,   0,   & reg_params::AccessKrate,           ParamTypePerNuc  },
  //{DFDD,     0,     & ((reg_params *)(NULL))->d[FIRSTINDEX]       - (float *) NULL,           ParamTypePerNuc  },
// buffering parameters
  {DFDMR,     0,    & reg_params::AccessNucModifyRatio,           ParamTypePerNuc  },
  //{DFDTAUMR,  0,    & ((reg_params *)(NULL))->tau_R_m    - (float *) NULL,           ParamTypeAllFlow },
  //{DFDTAUOR,   0,   & ((reg_params *)(NULL))->tau_R_o    - (float *) NULL,           ParamTypeAllFlow },
  {DFDTAUE,  0,    & reg_params::AccessTauE,           ParamTypeAllFlow },
// time varying parameters
//  {DFDRDR,   0,     & reg_params::AccessRatioDrift,           ParamTypeAllFlow },
  {DFDPDR,   0,     & reg_params::AccessCopyDrift,           ParamTypeAllFlow },
  {TBL_END,  0,     0,                                                            ParamTableEnd    },
};

master_fit_type_entry master_fit_type_table::base_bkg_model_fit_type[] =
{
//{"name",                     &fit_descriptor,                              NULL, {NULL,0,NULL,0}},
  // individual well fits
  {"FitWellAmpl",               BkgFitStructures::fit_well_ampl_descriptor,                    NULL, {NULL,0,NULL,0}},
  {"FitWellAmplBuffering",      BkgFitStructures::fit_well_ampl_buffering_descriptor,          NULL, {NULL,0,NULL,0}},
  {"FitWellPostKey",            BkgFitStructures::fit_well_post_key_descriptor,                NULL, {NULL,0,NULL,0}},
  {"FitWellPostKeyNoDmult",     BkgFitStructures::fit_well_post_key_descriptor_nodmult,        NULL, {NULL,0,NULL,0}},

  // region-wide fits
  {"FitRegionTmidnucPlus",      BkgFitStructures::fit_region_tmidnuc_plus_descriptor,          NULL, {NULL,0,NULL,0}},
  {"FitRegionInit2",            BkgFitStructures::fit_region_init2_descriptor,                 NULL, {NULL,0,NULL,0}},
  {"FitRegionInit2TauE",        BkgFitStructures::fit_region_init2_taue_descriptor,            NULL, {NULL,0,NULL,0}},
  {"FitRegionInit2TauENoRDR",   BkgFitStructures::fit_region_init2_taue_NoRDR_descriptor,      NULL, {NULL,0,NULL,0}},
  {"FitRegionFull",             BkgFitStructures::fit_region_full_descriptor,                  NULL, {NULL,0,NULL,0}},
  {"FitRegionFullTauE",         BkgFitStructures::fit_region_full_taue_descriptor,             NULL, {NULL,0,NULL,0}},
  {"FitRegionFullTauENoRDR",    BkgFitStructures::fit_region_full_taue_NoRDR_descriptor,       NULL, {NULL,0,NULL,0}},
  {"FitRegionInit2NoRDR",       BkgFitStructures::fit_region_init2_noRatioDrift_descriptor,    NULL, {NULL,0,NULL,0}},
  {"FitRegionFullNoRDR",        BkgFitStructures::fit_region_full_noRatioDrift_descriptor,     NULL, {NULL,0,NULL,0}},
  {"FitRegionTimeVarying",      BkgFitStructures::fit_region_time_varying_descriptor,          NULL, {NULL,0,NULL,0}},
  {"FitRegionDarkness",         BkgFitStructures::fit_region_darkness_descriptor,              NULL, {NULL,0,NULL,0}},

  //region-wide fits without diffusion
  {"FitRegionInit2TauENoD",     BkgFitStructures::fit_region_init2_taue_NoD_descriptor,        NULL, {NULL,0,NULL,0}},
  {"FitRegionInit2TauENoRDRNoD",BkgFitStructures::fit_region_init2_taue_NoRDR_NoD_descriptor,  NULL, {NULL,0,NULL,0}},
  {"FitRegionFullTauENoD",      BkgFitStructures::fit_region_full_taue_NoD_descriptor,         NULL, {NULL,0,NULL,0}},
  {"FitRegionFullTauENoRDRNoD", BkgFitStructures::fit_region_full_taue_NoRDR_NoD_descriptor,   NULL, {NULL,0,NULL,0}},

  { NULL, NULL, NULL, {NULL,0,NULL,0} },  // end of table
};
*/


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

int BkgFitStructures::GetNumParamsToFitForDescriptor(
  const std::vector<fit_descriptor>& fds, 
  int flow_key, 
  int flow_block_size)
{
  int numParamsToFit = 0;
  if (fds.size() > 0) {
    for (int i=0; fds[i].comp != TBL_END; ++i) 
    {
      switch (fds[i].ptype)
      {
        case ParamTypeAFlows:
	case ParamTypeCFlows:
	case ParamTypeGFlows:
	case ParamTypeAllFlow:
          numParamsToFit++;
	  break;
	case ParamTypeNotKey:
	  numParamsToFit += max( 0, flow_block_size-flow_key );
	  break;
	case ParamTypePerFlow:
	  numParamsToFit += flow_block_size;
	  break;
	case ParamTypePerNuc:
	  numParamsToFit += NUMNUC;
	  break;
	case ParamTypeAllButFlow0:
	  numParamsToFit += flow_block_size-1;
	  break;
	default:
	  break;
      }
    }
  }
  return numParamsToFit;
}

int BkgFitStructures::GetNumParDerivStepsForFitDescriptor(const std::vector<fit_descriptor>& fds) {
  int numParDerivSteps = 0;
  if (fds.size() > 0) {
    for (int i=0; fds[i].comp != TBL_END; ++i) 
    {
      numParDerivSteps++;
    }
  }
  return numParDerivSteps;
}
