/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITMATRIXPACKER_H
#define BKGFITMATRIXPACKER_H


#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "BkgMagicDefines.h"
#include "BeadParams.h"
#include "RegionParams.h"
#include <map>

class BkgFitMatDat;

//#define BKG_FIT_MAX_FLOWS   20

// identifiers for all the partial derivatives that might be used in the fit
// this needs to be a bit-field
typedef enum 
  {
    TBL_END    =    0x0,
    // per-well fitting PartialDeriv components
    DFDP        =0x00000001,
    DFDR        =0x00000002,
    DFDPDM      =0x00000004, 
    DFDGAIN     =0x00000008,
    DFDA        =0x00000010,
    DFDDKR      =0x00000020,

    // per-region fitting PartialDeriv components

    DFDSIGMA    =0x00001000,
    DFDTSH      =0x00002000,
    DFDERR      =0x00004000,
    DFDT0       =0x00008000, 
    DFDTAUMR    =0x00010000,
    DFDTAUOR    =0x00020000,

    DFDRDR      =0x00040000,
    DFDKRATE    =0x00080000,
    DFDSENS     =0x00100000,
    DFDD        =0x00200000,
    DFDPDR      =0x00400000,
    DFDMR       =0x00800000,
    DFDKMAX     =0x01000000,
    DFDT0DLY    =0x02000000,
    DFDSMULT    =0x04000000,
    DFDTAUE     =0x08000000,
    // special cases (function value and fit error)
    YERR        =0x10000000,
    FVAL        =0x20000000,


  } PartialDerivComponent;

// identifiers for the two matricies that must be constructed
typedef enum {
  JTJ_MAT,
  RHS_MAT
} AssyMatID;

// item used in list of partial derivatives.  This list forms the linkage between
// the logical PartialDerivComponent value specified in the table (at link time), and the 
// run time address of the component's data
struct PartialDeriv_comp_list_item {
  PartialDerivComponent comp;
  float *addr;
};

// structure that holds all the information to describe how a single jtj or rhs
// matrix element can be constructed from pieceis of two different PartialDeriv components
class mat_assy_input_line {
  mat_assy_input_line( const mat_assy_input_line & );             // Don't do this.
  mat_assy_input_line & operator=( const mat_assy_input_line & ); // Don't do this, either.

  int * affected_flows;
public:
  mat_assy_input_line() { 
    affected_flows = 0;
    comp1 = comp2 = TBL_END;
    mat_row = mat_col = 0;
    matId = JTJ_MAT;
  }
  ~mat_assy_input_line() { if (affected_flows) delete [] affected_flows; }
  void realloc( int size ) { delete [] affected_flows; affected_flows = new int[size]; }

  PartialDerivComponent comp1;
  PartialDerivComponent comp2;

  int GetAffectedFlow( int which ) { return affected_flows[which]; }
  void SetAffectedFlow( int which, int value ) { affected_flows[which] = value; }

  AssyMatID matId;
  int mat_row;
  int mat_col;
  std::string derivName;
};


struct sub_instr {
  int   len;      // numer of consecutive data points to mult-add together
  float *src1;
  float *src2;
};

// the run-time information used to control the matrix building process.  Once the
// address of each PartialDeriv component is known, a list of mat_assembly_instructions is built
// an each item indicates the basic information needed to construct each non-zero 
// matrix element
struct mat_assembly_instruction {
  int cnt;        // number of sub_instr blocks to process
  struct sub_instr *si;
  double *dst;    // output address for dot-product value
};

// structure that holds the output mapping of the solution matrix (the delta vector)
// to the params structure
struct delta_mat_output_line {
  delta_mat_output_line() : delta_ndx(0), bead_params_func(NULL), reg_params_func(NULL), array_index(0) {}
  int delta_ndx;
  float * ( BeadParams::* bead_params_func )();
  float * ( reg_params::*  reg_params_func  )();
  int array_index;
  std::string name;
};

// collection of all input and output instruction information for a single fit configuration
struct fit_instructions {
  mat_assy_input_line *input;
  int    input_len;
  struct delta_mat_output_line *output;
  int    output_len;

  fit_instructions() {
    input = NULL;
    input_len = 0;
    output = NULL;
    output_len = 0;
  }
};

typedef enum {
  LinearSolverException,
  LinearSolverSuccess

} LinearSolverResult;

class BkgFitMatrixPacker
{
 public:
  BkgFitMatrixPacker(int imgLen,fit_instructions &fi,PartialDeriv_comp_list_item *PartialDeriv_list,int PartialDeriv_list_len, int flow_block_size);

  LinearSolverResult GetOutput(BeadParams *bp, reg_params *rp,double lambda, double regularizer);

  unsigned int GetPartialDerivMask(void) {return PartialDeriv_mask;}
	
  ~BkgFitMatrixPacker();
  BkgFitMatDat *data;
  /* Mat<double> *jtj; */
  /* Col<double> *rhs; */
  /* Col<double> *delta; */

  double *GetDestMatrixPtr(AssyMatID mat_id,int row,int col);

  std::vector<std::string> compNames;

 private:


  float *GetPartialDerivComponent(PartialDerivComponent comp);

  void CreateInstrFromInputLine(mat_assembly_instruction *p,mat_assy_input_line *src,int imgLen, int flow_block_size);

  delta_mat_output_line *outputList;
  int nOutputs;

  mat_assembly_instruction *instList;
  int nInstr;

  PartialDeriv_comp_list_item *PartialDeriv_list;
  int nPartialDeriv;

  unsigned int PartialDeriv_mask;
  int numException;

 public:

  fit_instructions& my_fit_instructions;
  delta_mat_output_line* getOuputList() const { return outputList; }
  mat_assembly_instruction* getInstList() const { return instList; }
  void BuildMatrix(bool accum);
  int getNumInstr() const { return nInstr; }
  int getNumOutputs() const { return nOutputs; }
  int getNumException() const { return numException; }
  void resetNumException() { numException = 0; }
  void SetDataRhs(float, int);
  void SetDataJtj(float, int, int);
};


#endif // BKGFITMATRIXPACKER_H
