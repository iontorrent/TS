/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgFitMatrixPacker.h"
#include "BkgFitMatDat.h"
#include <armadillo>
#include <MathUtil.h>
#include <DotProduct.h>

using namespace arma;

BkgFitMatrixPacker::~BkgFitMatrixPacker()
{
  for (int i=0;i<nInstr;i++)
  {
    delete [] instList[i].si;
  }

  delete [] instList;
  delete data;
}

BkgFitMatrixPacker:: BkgFitMatrixPacker (int imgLen,fit_instructions &fi,PartialDeriv_comp_list_item *PartialDeriv_list,int PartialDeriv_list_len, int flow_block_size)
    : my_fit_instructions (fi)
{
  nInstr = fi.input_len;
  instList = new mat_assembly_instruction[nInstr];
  this->PartialDeriv_list = PartialDeriv_list;
  nPartialDeriv = PartialDeriv_list_len;

  // the output_len is the total number of fitted parameters, and it is equal to the
  // dimensions of the jtj and rhs matricies
  // jtj  =  (Mat<double>*) new Mat<double>(fi.output_len, fi.output_len);
  // rhs = new Col<double>;
  // delta = new Col<double>
  data = new BkgFitMatDat();
  data->rhs->set_size (fi.output_len);
  data->delta->set_size (fi.output_len);
  numException = 0;
  (*data->jtj).zeros (fi.output_len, fi.output_len);

  PartialDeriv_mask = 0;

  for (int i=0;i<fi.input_len;i++)
  {
    instList[i].si = NULL;
    CreateInstrFromInputLine (instList+i,&fi.input[i],imgLen, flow_block_size);
    PartialDeriv_mask |= fi.input[i].comp1 | fi.input[i].comp2;
  }

  nOutputs = fi.output_len;
  outputList = fi.output;
  for (int i=0; i<nOutputs; ++i )
      compNames.push_back(outputList[i].name);
}


void BkgFitMatrixPacker::BuildMatrix (bool accum)
{
  mat_assembly_instruction *pinst = instList;

  // build JTJ and RHS matricies
  for (int i=0;i < nInstr;i++)
  {
    double sum=0.0;

    for (int j=0;j < pinst->cnt;j++)
        sum += DotProduct (pinst->si[j].len,pinst->si[j].src1,pinst->si[j].src2);

    if (accum)
      * (pinst->dst) += sum;
    else
      * (pinst->dst) = sum;

    pinst++;
  }
}

LinearSolverResult BkgFitMatrixPacker::GetOutput (BeadParams *bp, reg_params *rp, double lambda, double regularizer)
{
  bool delta_ok = true;
  Mat<double> jtj_lambda;

  jtj_lambda = trans (*data->jtj) + (*data->jtj) + (lambda-1.0) *diagmat (*data->jtj) + regularizer* eye(nOutputs,nOutputs);

 
  try
  {
 // not necessary - armadillo does fast-solvers for small matrix size
    {
      if (!solve (*data->delta,jtj_lambda,*data->rhs))
      {
        data->delta->set_size (nOutputs);
        data->delta->zeros (nOutputs);
        delta_ok = false;
        numException++;
      }
    }
  }
  catch (std::runtime_error& le)
  {
    data->delta->set_size (nOutputs);
    data->delta->zeros (nOutputs);
    delta_ok = false;
  }

  for (int i=0;i < nOutputs;i++)
  {
    if (std::isnan (data->delta->at (i)))
    {
      delta_ok = false;
      break;
    }
  }

  if (delta_ok)
  {
    // put outputs in the right place
    for (int i=0;i < nOutputs;i++){
      // What is that right place?
      float *dptr = outputList[i].bead_params_func ? (bp->*( outputList[i].bead_params_func ))() 
                                                   : (rp->*( outputList[i].reg_params_func  ))();
      dptr += outputList[i].array_index;
      // safe extraction double->double
      double tmp_eval = data->delta->at (outputList[i].delta_ndx);
      // float added to double safe promotion
      tmp_eval += *dptr;
      if ((tmp_eval<(-FLT_MAX)) or (tmp_eval>FLT_MAX))
        tmp_eval = *dptr;
      // now that tmp_eval is safe, put back
      *dptr = tmp_eval ;
    }
  }
  else
  {
    data->delta->set_size (nOutputs);
    data->delta->zeros (nOutputs);
  }
  // if (numException >0)
  //   std::cout << "BkgFitMatrixPacker: numException = " << numException << std::endl;
  if (delta_ok)
    return LinearSolverSuccess;
  else
    return LinearSolverException;
}

double* BkgFitMatrixPacker::GetDestMatrixPtr (AssyMatID mat_id,int row,int col)
{
  double *ret = NULL;

  switch (mat_id)
  {
    case JTJ_MAT:
      ret = & (data->jtj->at (row,col));
      break;
    case RHS_MAT:
      ret = & (data->rhs->at (row));
      break;
  }

  return (ret);
}

float* BkgFitMatrixPacker::GetPartialDerivComponent (PartialDerivComponent comp)
{
  for (int i=0;i < nPartialDeriv;i++)
  {
    if (PartialDeriv_list[i].comp == comp)
      return (PartialDeriv_list[i].addr);
  }

  return NULL;
}

void BkgFitMatrixPacker::CreateInstrFromInputLine (mat_assembly_instruction *p,mat_assy_input_line *src,int imgLen, int flow_block_size)
{
  float *src1_base = GetPartialDerivComponent (src->comp1);
  float *src2_base = GetPartialDerivComponent (src->comp2);
  p->dst = GetDestMatrixPtr (src->matId,src->mat_row,src->mat_col);
  if (p->dst == NULL)
  {
    fprintf (stderr,"Unable to find PDE component\n");
  }

  int nstart =-1;
  int nlen   = 0;
  int cnt = 0;

  // we do this twice...once to figure out how many there are...
  for (int i=0;i<flow_block_size;i++)
  {
    if (src->GetAffectedFlow(i))
    { 
      if (nlen == 0)
        nstart = i;
      nlen++;
    }
    else
    {

      if (nlen != 0)
      {
        // we just found a contiguous block of data points to dot-product together
        // convet it to a sub_instr
        cnt++;
        nlen = 0;
      }
    }
  }

  // special case...if all flows are affected
  if (nlen != 0)
    cnt++;
  // ...and a second time to actually fill in the data
  struct sub_instr *si = new struct sub_instr[cnt];
  nstart = -1;
  nlen = 0;
  cnt = 0;
  for (int i=0;i<flow_block_size;i++)
  {
    if (src->GetAffectedFlow(i))
    {
      if (nlen == 0)
        nstart = i;
      nlen++;
    }
    else
    {
      if (nlen != 0)
      {
        // we just found a contiguous block of data points to dot-product together
        // convet it to a sub_instr

        si[cnt].len = nlen*imgLen;
        si[cnt].src1 = src1_base + nstart*imgLen;
        si[cnt].src2 = src2_base + nstart*imgLen;

        cnt++;
        nlen = 0;
      }
    }
  }

  // special case...if all flows are affected
  if (nlen != 0)
  {
    si[cnt].len  = nlen*imgLen;
    si[cnt].src1 = src1_base + nstart*imgLen;
    si[cnt].src2 = src2_base + nstart*imgLen;
    cnt++;
  }
  if (p->si)
  {
    delete [] p->si;
  }
  p->si = si;
  p->cnt = cnt;
}

void BkgFitMatrixPacker::SetDataRhs (float value, int i)
{
  data->rhs->at (i) = value;
}

void BkgFitMatrixPacker::SetDataJtj (float value, int row, int col)
{
  data->jtj->at (row,col) = value;
}
