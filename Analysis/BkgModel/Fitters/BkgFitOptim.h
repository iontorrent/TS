/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITOPTIM_H
#define BKGFITOPTIM_H

#include <stdio.h>
#include <stdlib.h>
#include "BkgMagicDefines.h"
#include "BkgFitMatrixPacker.h"
#include "BeadParams.h"
#include "RegionParams.h"


typedef enum
{
  CalcBase = 0,
  CalcBoth = 1,
  CalcFirst = 2,
  CalcNone = 3
}CalcEnums;


#define SPECIALCALCULATION 0
#define NOTBEADPARAM -1
#define NOTREGIONPARAM -1
#define NOTNUCRISEPARAM -1
#define SINGLETON 1
#define FIRSTINDEX 0

#define POSTKEY 7

typedef struct
{
    unsigned int PartialDerivMask;
    char *name;
    float *ptr;
    float diff;
    int   doBoth;
    int paramsOffset;
    int regParamsOffset;
    int nucShapeOffset; // this is to deal with the goofy way the optimization is handled
    int len;
}CpuStep_t;

typedef struct
{
    unsigned int PartialDerivMask;
    float diff;
    int origStep;
}Step_t;



/* Making the tables that configure the matrix operations was getting horribly tedious, so I now have some relatively
   simple code that builds the table.  This code should be called once at startup because the tables are shared
   by all objects of the class
   */
struct mat_table_build_instr
{
  PartialDerivComponent comp;
  int param_ndx;
  int affected_flows[NUMFB];
};

/* typedef used to classify each parameter we are going to fit.  The high-level fit specification
   breaks parameters down into four basic groups.  Those that are an indePartialDerivnent parameter per flow,
   those that are an indepdendent parameter per nucleotide, those that are the same across all
   flows, and a special class that is independent per nuc flow, but is specifically excluded from the key.
   This information is used, along with the number of flow buffers and the flow order to construct
   build instructions for each type of fit.  This allows easy reconfiguration of the software for
   different numbers of flow buffers at compile time, and changes to the flow order at run time.  */
typedef enum
{
  ParamTypePerFlow,
  ParamTypePerNuc,
  ParamTypeAllFlow,
  ParamTypeNotKey,
  ParamTypeAFlows,
  ParamTypeCFlows,
  ParamTypeGFlows,
  ParamTypeAllButFlow0,
  ParamTableEnd,
} ParameterSensitivityClassification;


struct fit_descriptor
{
  PartialDerivComponent comp;
  int param_ndx;
  ParameterSensitivityClassification ptype;
};

struct master_fit_type_table
{
  // nice human-readable descriptive name for what the fit attempts to do
  char *name;
  // high-level fit descriptor list.  One entry in the list for each parameter to be
  // fit, along with a classification of the parameter that indicates whether it's one-per-flow
  // or one-per-nuc, etc.,...  This high level description is used to build the
  // mat_table_build_instr table.
  struct fit_descriptor *fd;
  // mid-level matrix build instructions.  This intermediate level table contains multiple entries
  // for some parameters.  (i.e., the Ampl parameter, which is independent per flow is broken out
  // in this table to one entry per flow, whereas it was a single line in the fit_descriptor....)
  // This is done dynamically because it makes it much easier to re-configure the software for a
  // different number of flow buffers.  If the total number of flow buffers change, the number of entries
  // in the fit_descriptor table doesn't change, but the number of entries in the mat_table_build_instr
  // does change.  This also makes it easier to handle run-time configuration of the flow order.
  struct mat_table_build_instr *mb;
  // the lowest-level fit instruction table.  These tables contain one entry for all the permutations
  // of each parameter with all other parameters.  These are built from the mat_table_build_instr
  // tables to make life much easier, and this low level is used to initialize the BkgFitMatrixPacker
  // class
  fit_instructions fi;
};



void CreateBuildInstructions(struct master_fit_type_table *mfte, int *my_nuc);
void InitializeLevMarFitter(struct mat_table_build_instr *btbl,fit_instructions *instr);
void DumpBuildInstructionTable(struct mat_table_build_instr *tbl);

#endif // BKGFITOPTIM_H