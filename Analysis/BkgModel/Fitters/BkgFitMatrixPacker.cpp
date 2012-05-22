/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgFitMatrixPacker.h"
#include "BkgFitMatDat.h"

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

BkgFitMatrixPacker:: BkgFitMatrixPacker (int imgLen,fit_instructions &fi,PartialDeriv_comp_list_item *PartialDeriv_list,int PartialDeriv_list_len)
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
    CreateInstrFromInputLine (instList+i,&fi.input[i],imgLen);
    PartialDeriv_mask |= fi.input[i].comp1 | fi.input[i].comp2;
  }

  nOutputs = fi.output_len;
  outputList = fi.output;
}
void BkgFitMatrixPacker::BuildMatrix (bool accum)
{
  mat_assembly_instruction *pinst = instList;

  // build JTJ and RHS matricies
  for (int i=0;i < nInstr;i++)
  {
    double sum=0.0;

    for (int j=0;j < pinst->cnt;j++)
      sum += cblas_sdot (pinst->si[j].len,pinst->si[j].src1,1,pinst->si[j].src2,1);
    if (accum)
      * (pinst->dst) += sum;
    else
      * (pinst->dst) = sum;

    pinst++;
  }
}

LinearSolverResult BkgFitMatrixPacker::GetOutput (float *dptr,double lambda)
{
  bool delta_ok = true;
  Mat<double> jtj_lambda;
  //Mat<double> jtj_lambda = symmatl(*jtj);

  //const float REGULARIZER_VAL = 0.0; // do regularization to prevent 0 values on the diagonal and stabilize convergence
  jtj_lambda = trans (*data->jtj) + (*data->jtj) + (lambda-1.0) *diagmat (*data->jtj);

  // apply lambda parameter to the diagonal terms of the jtj matrix
  //for (int i=0;i < nOutputs;i++)
  //jtj_lambda(i,i) = (1.0+lambda)*( jtj_lambda(i,i)/2+ REGULARIZER_VAL);
  // solve equation...don't do IP because I want to preserve the zero fields in jtj
  // also...I don't know if IP works when the 'A' matrix is a symmetric one like I have defined

  try
  {
    // these special cases handle the relatively trivial 1 and 2 parameter solutions
    // in a faster way
    if (nOutputs == 1)
    {
      data->delta->at (0) = data->rhs->at (0) /jtj_lambda.at (0,0);
    }
    else if (nOutputs == 2)
    {
      double a,b,c,d,det;
      a = jtj_lambda.at (0,0);
      b = jtj_lambda.at (0,1);
      c = jtj_lambda.at (1,0);
      d = jtj_lambda.at (1,1);
      det = 1.0 / (a*d - b*c);
      data->delta->at (0) = (d* (data->rhs->at (0))-b* (data->rhs->at (1))) *det;
      data->delta->at (1) = (-c*data->rhs->at (0) +a*data->rhs->at (1)) *det;
    }
    else
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
  catch (std::runtime_error le)
  {
    data->delta->set_size (nOutputs);
    data->delta->zeros (nOutputs);
    delta_ok = false;
  }

  for (int i=0;i < nOutputs;i++)
  {
    if (isnan (data->delta->at (i)))
    {
      delta_ok = false;
      break;
    }
  }

  if (delta_ok)
  {
    // put outputs in the right place
    for (int i=0;i < nOutputs;i++)
      dptr[outputList[i].param_ndx] += data->delta->at (outputList[i].delta_ndx);
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
  double *ret;

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

void BkgFitMatrixPacker::CreateInstrFromInputLine (mat_assembly_instruction *p,mat_assy_input_line *src,int imgLen)
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
  for (int i=0;i<NUMFB;i++)
  {
    if (src->affected_flows[i])
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
  for (int i=0;i<NUMFB;i++)
  {
    if (src->affected_flows[i])
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
