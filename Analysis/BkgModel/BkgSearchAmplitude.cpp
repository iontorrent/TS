/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgSearchAmplitude.h"


void SearchAmplitude::EvaluateAmplitudeFit(bead_params *p, float *avals,float *error_by_flow,float *sbg)
{
    bead_params eval_params = *p; // temporary copy
    reg_params *reg_p = &my_regions->rp;

    params_SetAmplitude(&eval_params,avals);
    // evaluate the function

    MultiFlowComputeCumulativeIncorporationSignal(&eval_params,reg_p,my_scratch->ival,*my_regions,my_scratch->cur_bead_block,*time_c,*my_flow,math_poiss);
    MultiFlowComputeIncorporationPlusBackground(my_scratch->fval,&eval_params,reg_p,my_scratch->ival,sbg,*my_regions,my_scratch->cur_buffer_block,*time_c,*my_flow,use_vectorization, my_scratch->bead_flow_t);

    my_scratch->FillEmphasis(p->WhichEmphasis,emphasis_data->EmphasisVectorByHomopolymer,emphasis_data->EmphasisScale);

    my_scratch->CalculateFitError(error_by_flow,my_flow->numfb);
}

void InitializeBinarySearch(bead_params *p, bool restart, float *ac, float *step, bool *done)
{
    if (restart)
        for (int fnum=0;fnum<NUMFB;fnum++)
        {
            ac[fnum] = 0.5;
            done[fnum] = false;
            step[fnum] = 1.0;
            p->WhichEmphasis[fnum] = 0;
        }
    else
        for (int fnum=0;fnum<NUMFB;fnum++)
        {
            ac[fnum] =p->Ampl[fnum];
            done[fnum] = false;
            step[fnum] = 0.02;
            p->WhichEmphasis[fnum] = 0;
        }
}

void TakeStepBinarySearch(float *ap, float *ep, float *ac, float *ec,float *step,  float deltaAmp, float min_a, float max_a)
{
    for (int i=0;i < NUMFB;i++)
    {
        float slope = (ep[i] - ec[i]) /deltaAmp;

        if (slope < 0.0)
        {
            // step up to lower err
            ap[i] = ac[i] + step[i];
        }
        else
        {
            // step up to lower err
            ap[i] = ac[i] - step[i];
        }

        if (ap[i] < min_a) ap[i] = min_a;
        if (ap[i] > max_a) ap[i] = max_a;
    }
}

void UpdateStepBinarySearch(float *ac, float *ec, float *ap, float *ep, float *step, float min_step, bool *done, int &done_cnt)
{
    for (int i=0;i < NUMFB;i++)
    {
        if (ep[i] < ec[i])
        {
            ac[i] = ap[i];
            ec[i] = ep[i];
        }
        else
        {
            step[i] /= 2.0;
            if (step[i] < min_step)
            {
                done[i] = true;
                done_cnt--;
            }
        }
    }
}

void SearchAmplitude::BinarySearchOneBead(bead_params *p, float min_step, bool restart, float *sbg)
{
    float ac[NUMFB] __attribute__ ((aligned (16)));
    float ec[NUMFB] __attribute__ ((aligned (16)));
    float ap[NUMFB] __attribute__ ((aligned (16)));
    float ep[NUMFB] __attribute__ ((aligned (16)));
    float step[NUMFB] __attribute__ ((aligned (16)));
    float min_a = 0.001;
    float max_a = (MAX_HPLEN - 1)-0.001;
    bool done[NUMFB];
    int done_cnt;
    int iter;
    float deltaAmp = 0.00025;
    
        
        InitializeBinarySearch(p,restart,ac,step,done);
        
        EvaluateAmplitudeFit(p, ac,ec,sbg);

        done_cnt = NUMFB; // number remaining to finish

        iter = 0;

        while ((done_cnt > 0) && (iter <= 30))
        {
            // figure out which direction to go in from here
            for (int fnum=0;fnum<NUMFB;fnum++)
                ap[fnum] = ac[fnum] + deltaAmp;

            EvaluateAmplitudeFit(p, ap,ep,sbg);
            // computes derivative to determine direction
            TakeStepBinarySearch(ap,ep,ac,ec,step,deltaAmp,min_a,max_a);

            // determine if new location is better
            EvaluateAmplitudeFit(p, ap,ep,sbg);

            UpdateStepBinarySearch(ac,ec,ap,ep, step, min_step, done, done_cnt);

            iter++;
        }
        params_SetAmplitude(p,ac);

        // It is important to have amplitude zero so that regional fits do the right thing for other parameters
        params_ApplyAmplitudeZeros(p,my_flow->dbl_tap_map);
}

void SearchAmplitude::BinarySearchAmplitude(BeadTracker &my_beads, float min_step,bool restart)
{
    // should be using a specific region pointer, not the universal one, to match the bead list
    // as we may generate more region pointers from more beads
    my_trace->GetShiftedBkg(my_regions->rp.tshift,my_scratch->shifted_bkg);

    my_scratch->ResetXtalkToZero();
    for (int ibd=0;ibd < my_beads.numLBeads;ibd++)
    {
        // want this for all beads, even if they are polyclonal.
        //@TODO this is of course really weird as we have several options for fitting one flow at a time
        // which might be easily parallelizable and not need this weird global level binary search
        my_scratch->FillObserved(my_trace->fg_buffers,ibd);  // set scratch space for this bead

        BinarySearchOneBead(&my_beads.params_nn[ibd],min_step,restart,my_scratch->shifted_bkg);
    }
}

SearchAmplitude::SearchAmplitude()
{
      math_poiss = NULL;
      my_trace = NULL;
      my_scratch = NULL;
      my_regions = NULL;
      time_c = NULL;
      my_flow = NULL;
      emphasis_data = NULL;
    
      use_vectorization = true;
}

SearchAmplitude::~SearchAmplitude()
{
  // nothing allocated here, just get rid of everything
        math_poiss = NULL;
      my_trace = NULL;
      my_scratch = NULL;
      my_regions = NULL;
      time_c = NULL;
      my_flow = NULL;
      emphasis_data = NULL;
}



// project X onto Y
// possibly I need to include "emphasis" function here
float project_vector(float *X, float *Y, float *em, int len)
{
    float numX,denX;
    numX=denX = 0;
    for (int i=0; i<len; i++)
    {
        numX += X[i]*Y[i]*em[i]*em[i];
        denX += Y[i]*Y[i]*em[i]*em[i];
    }
    return(numX/denX);
}

float em_diff(float *X, float *Y, float *em, int len)
{
    float tot=0.0;
    float eval;
    for (int i=0; i<len; i++)
    {
        eval = (X[i]-Y[i])*em[i];
        tot += eval*eval;
    }
    return(tot);
}

