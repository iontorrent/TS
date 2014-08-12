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


#define NOTBEADPARAM        ( float * ( BeadParams    ::* )() )( 0 )
#define NOTREGIONPARAM      ( float * ( reg_params     ::* )() )( 0 )
#define NOTNUCRISEPARAM     ( float * ( nuc_rise_params::* )() )( 0 )
#define FIRSTINDEX 0

struct CpuStep
{
    enum Length {
      SpecialCalculation,
      Singleton,
      PerNuc,
      PerFlow
    };

    unsigned int PartialDerivMask;
    char *name;
    float *ptr;
    float diff;
    int   doBoth;

    // The somewhat odd syntax below is C++ for "pointer to a member function".
    // This is all to deal with the goofy way the optimization is handled.
    float * ( BeadParams::*     paramsFunc    )();
    float * ( reg_params::*      regParamsFunc )();
    float * ( nuc_rise_params::* nucShapeFunc  )();
    Length length;
};

typedef struct
{
    unsigned int PartialDerivMask;
    float diff;
    int origStep;
}Step_t;



/* Making the tables that configure the matrix operations was getting horribly tedious, 
   so I now have some relatively simple code that builds the table.  This code should be 
   called once at startup because the tables are shared by all objects of the class.
   */
class mat_table_build_instr
{
  mat_table_build_instr( const mat_table_build_instr & );               // Don't do this.
  mat_table_build_instr & operator=( const mat_table_build_instr & );   // Don't do this, either.

  int * affected_flows;
public:
  mat_table_build_instr() 
  { 
    affected_flows = 0; 
    comp = TBL_END;
    array_index = 0;
    bead_params_func = 0;
    reg_params_func = 0;
  }
  ~mat_table_build_instr() { delete [] affected_flows; }
  void realloc( int size ) { delete [] affected_flows; affected_flows = new int[size]; }
  
  PartialDerivComponent comp;
  float * ( BeadParams::* bead_params_func )();
  float * ( reg_params::*  reg_params_func  )();
  int array_index;

  int GetAffectedFlow( int which ) const { return affected_flows[which]; }
  void SetAffectedFlow( int which, int value ) { affected_flows[which] = value; }
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
  float * ( BeadParams::* bead_params_func )();
  float * ( reg_params::*  reg_params_func  )();
  ParameterSensitivityClassification ptype;
};

struct master_fit_type_entry
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
  mat_table_build_instr *mb;
  // the lowest-level fit instruction table.  These tables contain one entry for all the permutations
  // of each parameter with all other parameters.  These are built from the mat_table_build_instr
  // tables to make life much easier, and this low level is used to initialize the BkgFitMatrixPacker
  // class
  fit_instructions fi;

  void CreateBuildInstructions(const int *my_nuc, int flow_key, int flow_block_size);
};

class master_fit_type_table
{
  // All of the table entries.
  master_fit_type_entry *data;

  // No copying or assignment.
  master_fit_type_table( const master_fit_type_table & );
  master_fit_type_table & operator=( const master_fit_type_table & );

  // The base data that we start with when making our array.
  static master_fit_type_entry base_bkg_model_fit_type[];
public:

  // TODO: PartialDeriv 'affected flows' has to be munged for tango flow order!!! */
  // Make me a new one! (Used to be InitializeLevMarSparseMatrices())
  master_fit_type_table( const class FlowMyTears & tears, 
                         int flow_start, int flow_key, int flow_block_size );

  // Cleanup! (Used to be CleanupLevMarSparseMatrices()).
  ~master_fit_type_table();

  fit_instructions *GetFitInstructionsByName(char *name);
  fit_descriptor *GetFitDescriptorByName(const char* name);
};



void InitializeLevMarFitter(struct mat_table_build_instr *btbl,fit_instructions *instr, int flow_block_size);
void DumpBuildInstructionTable(struct mat_table_build_instr *tbl, int flow_block_size);

#endif // BKGFITOPTIM_H
