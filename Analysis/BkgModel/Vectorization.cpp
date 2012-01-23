/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "Vectorization.h"
#include <string.h>
#include <algorithm>
#include <math.h>

void PurpleSolveTotalTrace_Vec(int numfb, float **vb_out, float **blue_hydrogen, float **red_hydrogen, int len, float *deltaFrame, float *tauB, float *etbR, float gain)
{
    v4sf dv,dvn,dv_rs;
    v4sf aval;
    v4sf dt;
    v4sf shift_ratio;
    v4sf tauBV;
    v4sf ttauBV;
    v4sf total;
    v4sf rh,bh;
    v4sf one={1.0,1.0,1.0,1.0};
    v4sf two={2.0,2.0,2.0,2.0};
    int  i,fb;
 
    float aligned_etbr[numfb] __attribute__ ((aligned (16)));
    float aligned_tau[numfb] __attribute__ ((aligned (16)));
    memcpy(aligned_etbr, etbR, sizeof(float)*numfb);
    memcpy(aligned_tau, tauB, sizeof(float)*numfb);

    for(fb=0;fb<numfb;fb+=VEC_INC)
    {
        shift_ratio = *(v4sf*)&aligned_etbr[fb] - one;
        aval = (v4sf){0,0,0,0};
        dv = (v4sf){0,0,0,0};
        dv_rs = (v4sf){0,0,0,0};
        tauBV = *(v4sf*)&aligned_tau[fb];
        ttauBV = tauBV*two;
        for (i=0;i < len;i++)
        {
            dt = (v4sf){deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};
 
            LOAD_4FLOATS_FLOWS(rh,red_hydrogen,fb,i,numfb);
            LOAD_4FLOATS_FLOWS(bh,blue_hydrogen,fb,i,numfb);
 
	    aval = dt/ ttauBV;

	    dvn = (rh + shift_ratio * bh - dv_rs/tauBV - dv*aval) / (one+aval);
	    dv_rs = dv_rs + (dv+dvn) * (dt/two);
	    dv = dvn;
	    total = (dv+bh);

	    // record the result
	    UNLOAD_4FLOATS_FLOWS(vb_out,total,fb,i,numfb);
	}
    }

}

void BlueSolveBackgroundTrace_Vec(int numfb, float **vb_out, float **blue_hydrogen, int len, 
    float *deltaFrame, float *tauB, float *etbR)
{
    v4sf dv,dvn,dv_rs;
    v4sf aval;
    v4sf dt;
    v4sf shift_ratio;
    v4sf tauBV;
    v4sf ttauBV;
    v4sf total;
    v4sf bh;
    v4sf one={1.0,1.0,1.0,1.0};
    v4sf two={2.0,2.0,2.0,2.0};
    int  i,fb;
 
    float aligned_etbr[numfb] __attribute__ ((aligned (16)));
    float aligned_tau[numfb] __attribute__ ((aligned (16)));
    memcpy(aligned_etbr, etbR, sizeof(float)*numfb);
    memcpy(aligned_tau, tauB, sizeof(float)*numfb);

    for(fb=0;fb<numfb;fb+=VEC_INC)
    {
        shift_ratio = *(v4sf *)&aligned_etbr[fb] - one;
        aval = (v4sf){0,0,0,0};
        dv = (v4sf){0,0,0,0};
        dv_rs = (v4sf){0,0,0,0};
        tauBV = *(v4sf *)&aligned_tau[fb];
        ttauBV = tauBV*two;
        for (i=0;i < len;i++)
        {
            dt = (v4sf){deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};
 
            LOAD_4FLOATS_FLOWS(bh,blue_hydrogen,fb,i,numfb);
 
	    aval = dt/ ttauBV;

	    dvn = (shift_ratio * bh - dv_rs/tauBV - dv*aval) / (one+aval);
	    dv_rs = dv_rs + (dv+dvn) * (dt/two);
	    dv = dvn;
	    total = (dv+bh);

	    // record the result
	    UNLOAD_4FLOATS_FLOWS(vb_out,total,fb,i,numfb);
	}
    }

}


void RedSolveHydrogenFlowInWell_Vec(int numfb, float **vb_out, float **red_hydrogen, int len, float *deltaFrame, float tauB)
{
    v4sf dv,dvn,dv_rs;
    v4sf aval;
    v4sf dt;
    v4sf tauBV;
    v4sf ttauBV;
    v4sf rh;
    v4sf one={1.0,1.0,1.0,1.0};
    v4sf two={2.0,2.0,2.0,2.0};
    int  i,fb;
 
    tauBV = (v4sf){tauB, tauB, tauB, tauB};
    ttauBV = tauBV*two;
    for(fb=0;fb<numfb;fb+=VEC_INC)
    {
        aval = (v4sf){0,0,0,0};
        dv = (v4sf){0,0,0,0};
        dv_rs = (v4sf){0,0,0,0};
        for (i=0;i < len;i++)
        {
            dt = (v4sf){deltaFrame[i],deltaFrame[i],deltaFrame[i],deltaFrame[i]};
 
            LOAD_4FLOATS_FLOWS(rh,red_hydrogen,fb,i,numfb);
 
	    aval = dt/ ttauBV;

	    dvn = (rh - dv_rs/tauBV - dv*aval) / (one+aval);
	    dv_rs = dv_rs + (dv+dvn) * (dt/two);
	    dv = dvn;

	    // record the result
	    UNLOAD_4FLOATS_FLOWS(vb_out,dv,fb,i,numfb);
	}
    }

}



void MultiplyVectorByScalar_Vec(float *my_vec, float my_scalar, int len) {
    v4sf dest;
    v4sf mul = {my_scalar,my_scalar,my_scalar,my_scalar};
    for (int i=0; i<len; i+=VEC_INC) {
        LOAD_4FLOATS_FRAMES(dest, my_vec, i, len);
        dest *= mul;
        UNLOAD_4FLOATS_FRAMES(my_vec, dest, i, len);
    }
}
 
void Dfderr_Step_Vec(int numfb, float** dst, float** et, float** em, int len) {
    v4sf dst_v, et_v, em_v;
    int i, fb;
    for(fb=0;fb<numfb;fb+=VEC_INC) {
        for (i=0;i < len;i++) {
            LOAD_4FLOATS_FLOWS(et_v,et,fb,i,numfb); 
            LOAD_4FLOATS_FLOWS(em_v,em,fb,i,numfb);

            dst_v = et_v * em_v;

            UNLOAD_4FLOATS_FLOWS(dst,dst_v,fb,i,numfb); 
        }
    }
}

void Dfdgain_Step_Vec(int numfb, float** dst, float** src, float** em, int len, float gain) {
    v4sf dst_v, src_v, em_v;
    v4sf gain_v = (v4sf){gain, gain, gain, gain};
    int i, fb;
    for(fb=0;fb<numfb;fb+=VEC_INC) {
        for (i=0;i < len;i++) {
            LOAD_4FLOATS_FLOWS(src_v,src,fb,i,numfb); 
            LOAD_4FLOATS_FLOWS(em_v,em,fb,i,numfb);

            dst_v = src_v * em_v / gain_v;

            UNLOAD_4FLOATS_FLOWS(dst,dst_v,fb,i,numfb); 
        }
    }
}
