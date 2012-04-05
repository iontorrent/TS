/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITMATRIXPACKER_H
#define BKGFITMATRIXPACKER_H

#include <string.h>
#include <stdlib.h>
#include <float.h>

#include "BkgMagicDefines.h"

// some of the code uses <complex.h>, and in <complex.h> 'I' is defined and this 
// interferes w/ lapackpp.  I undef it here in case anyone above has included <complex.h>
#undef I

#include <lapackpp.h>
#include <spdfd.h>
#ifdef ION_USE_MKL
#include <mkl_cblas.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

//#define BKG_FIT_MAX_FLOWS   20

// identifiers for all the partial derivatives that might be used in the fit
// this needs to be a bit-field
typedef enum 
{
    TBL_END    =    0x0,
    // per-well fitting PartialDeriv components
    DFDR        =0x00000001,
    DFDA        =0x00000002,
    DFDT0       =0x00000004,
    DFDP        =0x00000008,
    DFDGAIN     =0x00000010,
    DFDPDM      =0x00000020,

    // per-region fitting PartialDeriv components
    DFDSIGMA    =0x00001000,
    DFDTSH      =0x00002000,
    DFDERR      =0x00004000,
    DFDDKR      =0x00008000,
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
struct mat_assy_input_line {
    PartialDerivComponent comp1;
    PartialDerivComponent comp2;
    int affected_flows[NUMFB];

    AssyMatID matId;
    int mat_row;
    int mat_col;
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
    int delta_ndx;
    int param_ndx;
};

// collection of all input and output instruction information for a single fit configuration
struct fit_instructions {
    struct mat_assy_input_line *input;
    int    input_len;
    struct delta_mat_output_line *output;
    int    output_len;
};

typedef enum {
    LinearSolverException,
    LinearSolverSuccess

} LinearSolverResult;

class BkgFitMatrixPacker
{
public:
    BkgFitMatrixPacker(int imgLen,fit_instructions &fi,PartialDeriv_comp_list_item *PartialDeriv_list,int PartialDeriv_list_len)
    : my_fit_instructions(fi)
    {
        nInstr = fi.input_len;
        instList = new mat_assembly_instruction[nInstr];

        this->PartialDeriv_list = PartialDeriv_list;
        nPartialDeriv = PartialDeriv_list_len;

        // the output_len is the total number of fitted parameters, and it is equal to the
        // dimensions of the jtj and rhs matricies
//        jtj.resize(fi.output_len,fi.output_len);
        jtj = new LaSpdMatDouble(fi.output_len,fi.output_len);
        rhs.resize(fi.output_len);
        delta.resize(fi.output_len);
	
        // TODO: There is an easier way to do this, but it wasn't working for me...something stupid...
        for (int i=0;i < fi.output_len;i++)
            for (int j=0;j < fi.output_len;j++)
                (*jtj)(i,j) = 0.0;

        PartialDeriv_mask = 0;

        for (int i=0;i<fi.input_len;i++)
	  {
	    instList[i].si = NULL;
            CreateInstrFromInputLine(instList+i,&fi.input[i],imgLen);
            PartialDeriv_mask |= fi.input[i].comp1 | fi.input[i].comp2;
        }

        nOutputs = fi.output_len;
        outputList = fi.output;
    }

    void BuildMatrix(bool accum)
    {
        mat_assembly_instruction *pinst = instList;

        // build JTJ and RHS matricies
        for (int i=0;i < nInstr;i++)
        {
            double sum=0.0;

            for (int j=0;j < pinst->cnt;j++)
                sum += cblas_sdot(pinst->si[j].len,pinst->si[j].src1,1,pinst->si[j].src2,1);
            if (accum)
                *(pinst->dst) += sum;
            else
                *(pinst->dst) = sum;

            pinst++;
        }
    }

    LinearSolverResult GetOutput(float *dptr,double lambda)
    {
      // Jacobian is J
      // Approximate Hessian matrix of partial derivatives is J'J = jtj
      //  Solve for delta: (J'J + lamda.I).delta = (rhs = J'(x - f(phat))
        bool delta_ok = true;
//        LaSymmMatDouble jtj_lambda;
        LaSpdMatDouble jtj_lambda;
      	LaSpdFactDouble jtj_lambda_fact;

        jtj_lambda.copy(*jtj);

        const float REGULARIZER_VAL = 0.0; // do regularization to prevent 0 values on the diagonal and stabilize convergence
        // apply lambda parameter to the diagonal terms of the jtj matrix
        for (int i=0;i < nOutputs;i++)
            jtj_lambda(i,i) = (1.0+lambda)*( jtj_lambda(i,i)+ REGULARIZER_VAL);

        // solve equation...don't do IP because I want to preserve the zero fields in jtj
        // also...I don't know if IP works when the 'A' matrix is a symmetric one like I have defined
        try {
                // these special cases handle the relatively trivial 1 and 2 parameter solutions
                // in a faster way
                if (nOutputs == 1)
                {    
                    delta(0) = rhs(0)/jtj_lambda(0,0);
                }
                else if (nOutputs == 2)
                {
                    double a,b,c,d,det;
                    a = jtj_lambda(0,0);
                    b = jtj_lambda(0,1);
                    c = jtj_lambda(1,0);
                    d = jtj_lambda(1,1);
                    det = 1.0 / (a*d - b*c);
                    delta(0) = (d*rhs(0)-b*rhs(1))*det;
                    delta(1) = (-c*rhs(0)+a*rhs(1))*det;
                }
                else
                {
                    LaSpdMatFactorize(jtj_lambda,jtj_lambda_fact);
                    LaLinearSolve(jtj_lambda_fact,delta,rhs);
                }
        }
        catch (LaException le) {
            delta.resize(nOutputs);

            delta = 0.0;
            delta_ok = false;
        }

        for (int i=0;i < nOutputs;i++)
        {
            if (isnan(delta(i)))
            {    
                delta_ok = false;
                break;
            }
        }

        if (delta_ok)
        {
            // put outputs in the right place
            for (int i=0;i < nOutputs;i++)
                dptr[outputList[i].param_ndx] += delta(outputList[i].delta_ndx);
        }
        else
            delta = 0.0;

        if (delta_ok)
            return LinearSolverSuccess;
        else
            return LinearSolverException;
    }

    unsigned int GetPartialDerivMask(void) {return PartialDeriv_mask;}

    ~BkgFitMatrixPacker() {
      for (int i=0;i<nInstr;i++) {
	delete [] instList[i].si;
      }
      
      delete [] instList;
      delete jtj;
    }

    LaSpdMatDouble  *jtj;
//    LaSymmMatDouble jtj;
    LaVectorDouble  rhs;
    LaVectorDouble  delta;

private:

    double *GetDestMatrixPtr(AssyMatID mat_id,int row,int col)
    {
        double *ret;

        switch (mat_id) {
        case JTJ_MAT:
            ret = &(*jtj)(row,col);
            break;
        case RHS_MAT:
            ret = &rhs(row);
            break;
        }

        return(ret);
    }

    float *GetPartialDerivComponent(PartialDerivComponent comp)
    {
        for (int i=0;i < nPartialDeriv;i++)
        {
            if (PartialDeriv_list[i].comp == comp)
                return(PartialDeriv_list[i].addr);
        }

        return NULL;
    }

    void CreateInstrFromInputLine(mat_assembly_instruction *p,mat_assy_input_line *src,int imgLen)
    {
        float *src1_base = GetPartialDerivComponent(src->comp1);
        float *src2_base = GetPartialDerivComponent(src->comp2);
        p->dst = GetDestMatrixPtr(src->matId,src->mat_row,src->mat_col);
    
        if (p->dst == NULL)
        {
            fprintf(stderr,"Unable to find PDE component\n");
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
	if (p->si) { delete [] p->si; }
        p->si = si;
        p->cnt = cnt;
    }

    delta_mat_output_line *outputList;
    int nOutputs;

    mat_assembly_instruction *instList;
    int nInstr;

    PartialDeriv_comp_list_item *PartialDeriv_list;
    int nPartialDeriv;

    unsigned int PartialDeriv_mask;

public:

    // Simple get methods for the CUDA object
    fit_instructions& my_fit_instructions;
    delta_mat_output_line* getOuputList() { return outputList; }
    mat_assembly_instruction* getInstList() { return instList; }
    int getNumInstr() { return nInstr; }
    int getNumOutputs() { return nOutputs; }


};


#endif // BKGFITMATRIXPACKER_H
