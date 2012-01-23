/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MultiLevMar.h"

MultiFlowLevMar::MultiFlowLevMar(BkgModel &_bkg):
  bkg(_bkg)
{
    
}


int MultiFlowLevMar::MultiFlowSpecializedLevMarFitParameters(int max_iter, int max_reg_iter, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction)
{
    int iter = 0;
    float BackgroundDebugTapA[bkg.my_scratch.bead_flow_t]; // disambiguate from well pointers, these are purely local debugging values
    float BackgroundDebugTapB[bkg.my_scratch.bead_flow_t];
    float fvdbg[bkg.my_scratch.bead_flow_t];
    float ival[bkg.my_scratch.bead_flow_t];
    float sbg[bkg.my_scratch.bead_flow_t];
    float tshift_cache = -10.0;

    lm_state.InitializeLevMarFit(well_fit,reg_fit);

    lm_state.SetupActiveBeadList(lambda_start);

    memset(BackgroundDebugTapB,0,sizeof(float[bkg.my_scratch.bead_flow_t]));

    // add regional iterations to total number of iterations
    // because we're going around the loop alternating region and well fits
    max_iter += max_reg_iter;

    bkg.my_scratch.ResetXtalkToZero();
    // loop until either all beads are done...or until we hit the maximum allowable iterations
    while ((iter < max_iter) && (lm_state.ActiveBeads > 0))
    {
#ifdef FIT_ITERATION_DEBUG_TRACE
        bkg.DebugIterations();
#endif

        bool reg_proc = (well_fit==NULL); // if no well fit, always do region processing
        bool well_only_fit = (reg_fit==NULL) || (iter>max_reg_iter);

        if (!well_only_fit) {
            if ((iter & 1) == 1)
                reg_proc = true; // half the time do region fit if we are not doing well_only
        }

        // phase in clonality restriction from 0-mers on up if requested
        lm_state.PhaseInClonalRestriction(iter,clonal_restriction);
        // go through each bead and do the next iteration of the fitting algorithm
        // we only fit the wells we have to and drop off ones that are finished as soon
        // as possible.
        if (bkg.my_regions.rp.tshift != tshift_cache)
        {
            bkg.my_trace.GetShiftedBkg(bkg.my_regions.rp.tshift,sbg);
            tshift_cache = bkg.my_regions.rp.tshift;
        }
        lm_state.reg_error = 0.0;
        reg_params eval_rp = bkg.my_regions.rp;
        if (reg_proc)
        {
            // do my region iteration
            lm_state.avg_resid = CalculateAverageResidual(sbg);
            int reg_wells = 0;
            reg_wells = LevMarAccumulateRegionDerivsForActiveBeadList(ival,sbg,eval_rp,
                        reg_fit, lm_state.reg_mask,
                        iter, fvdbg, BackgroundDebugTapA, BackgroundDebugTapB);
            // solve per-region equation and adjust parameters
            if (reg_wells > 10)
            {
                LevMarFitRegion(tshift_cache, sbg, reg_fit);
                lm_state.IncrementRegionGroup();
            }
            IdentifyParameters(bkg.my_beads,bkg.my_regions, lm_state.well_mask, lm_state.reg_mask);

        }
        else
        {
            // do my well iteration, see how many didn't improve
            int req_done = 0;
            req_done = LevMarFitToActiveBeadList(
                           ival, sbg, well_only_fit,
                           eval_rp,
                           well_fit, lm_state.well_mask,
                           iter, fvdbg, BackgroundDebugTapA, BackgroundDebugTapB);
            // if more than 1/2 the beads aren't improving any longer, stop trying to do the
            // region-wide fit
            if (!well_only_fit && (((float) req_done/lm_state.ActiveBeads) > 0.5))
                iter = max_reg_iter; // which will be incremented later
        }
        iter++;
    }

    lm_state.FinalComputeAndSetAverageResidual();
    lm_state.restrict_clonal = 0.0; // we only restrict clonal within this routine
    return (iter);
}

float MultiFlowLevMar::CalculateAverageResidual(float *sbg)
{
    float avg_error = 0.0;
    int cnt = 0;

    for (int ibd=0;ibd < bkg.my_beads.numLBeads;ibd++){
        if (bkg.my_beads.params_nn[ibd].clonal_read)
        {
            bkg.my_scratch.FillObserved(bkg.my_trace.fg_buffers,bkg.my_beads.params_nn[ibd].trace_ndx); // set scratch space for this bead
            FillScratchForEval(&bkg.my_beads.params_nn[ibd],&bkg.my_regions.rp,sbg);
            avg_error += bkg.my_scratch.CalculateFitError(NULL,NUMFB);
            cnt++;
        }
    }

    return (avg_error / cnt);
}



void MultiFlowLevMar::FillScratchForEval(struct bead_params *p,struct reg_params *reg_p, float *sbg)
{
    // evaluate the function
    //bkg.MultiFlowComputeTotalSignalTrace(bkg.my_scratch.fval,p,reg_p,sbg);
    MultiFlowComputeCumulativeIncorporationSignal(p,reg_p,bkg.my_scratch.ival,bkg.my_regions,bkg.my_scratch.cur_bead_block,bkg.time_c,bkg.my_flow,bkg.math_poiss);
    MultiFlowComputeIncorporationPlusBackground(bkg.my_scratch.fval,p,reg_p,bkg.my_scratch.ival,sbg,bkg.my_regions,bkg.my_scratch.cur_buffer_block,bkg.time_c,bkg.my_flow,bkg.use_vectorization, bkg.my_scratch.bead_flow_t);

    // add clonal restriction here to penalize non-integer clonal reads
    // this of course does not belong here and should be in the optimizer section of the code
    lm_state.ApplyClonalRestriction(bkg.my_scratch.fval, p,bkg.time_c.npts);
    bkg.my_scratch.FillEmphasis(p->WhichEmphasis,bkg.emphasis_data.EmphasisVectorByHomopolymer,bkg.emphasis_data.EmphasisScale);
}

// reg_proc = TRUE
int MultiFlowLevMar::LevMarAccumulateRegionDerivsForActiveBeadList(
    float *ival, float *sbg,
    reg_params &eval_rp,
    BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
    int iter, float *fvdbg, float *BackgroundDebugTapA, float *BackgroundDebugTapB)
{
    int reg_wells = 0;
    int defend_against_infinity = 0;
    lm_state.advance_bd = true; // always true here
    //@TODO nuc_rise step computed here, applied to all beads individually
    for (int nbd=0;(nbd < lm_state.ActiveBeads) && (defend_against_infinity<lm_state.numLBeads);defend_against_infinity++)
    {
        // get the index to the next bead to fit in the data matricies
        int ibd = lm_state.fit_indicies[nbd];

        // if this iteration is a region-wide parameter fit, then only process beads
        // in the selection sub-group
        if (!lm_state.ValidBeadGroup(ibd))
        {
            nbd++;
            continue;
        }

        // get the current parameter values for this bead
        bead_params eval_params = bkg.my_beads.params_nn[ibd];
        ComputePartialDerivatives(eval_params,eval_rp,ibd, PartialDeriv_mask,ival, sbg);

        if ((ibd == bkg.my_beads.DEBUG_BEAD) && (bkg.my_debug.trace_dbg_file != NULL))
        {
            bkg.DebugBeadIteration(eval_params,eval_rp, iter, ibd);
            bkg.my_trace.GetShiftedBkg(bkg.my_regions.rp.tshift,BackgroundDebugTapA);
            //CalcXtalkFlux(ibd,BackgroundDebugTapB);
            memcpy(fvdbg,bkg.my_scratch.fval,sizeof(float) *bkg.my_scratch.bead_flow_t);
        }

        // if doing region-wide fitting, continue building the regional fit matricies
        if (lm_state.well_region_fit[ibd])
        {
            lm_state.reg_error += lm_state.residual[ibd];

            if (lm_state.residual[ibd] < lm_state.avg_resid*4.0) // only use "well-behaved" wells at any iteration
            {
                // if  reg_wells>0, continue same matrix, otherwise start a new one
                BuildMatrix(reg_fit, (reg_wells>0),(ibd==bkg.my_beads.DEBUG_BEAD));
                reg_wells++;
            }
        }

        // if we just removed a bead from the fitting process, don't advance since a
        // new bead is already in place at the current index (it was placed there in
        // place of the bead that was just removed)
        if (lm_state.advance_bd)
            nbd++;
    }
    return(reg_wells); // number of live wells fit for region
}



void MultiFlowLevMar::LevMarFitRegion( float &tshift_cache, float *sbg,
                                BkgFitMatrixPacker *reg_fit)
{
    bool cont_proc = false;

    reg_params eval_rp;
    int defend_against_infinity = 0; // make sure we can't get trapped forever

    while (!cont_proc && defend_against_infinity<EFFECTIVEINFINITY)
    {
        defend_against_infinity++;
        eval_rp = bkg.my_regions.rp;
        ResetRegionBeadShiftsToZero(&eval_rp);

        if (reg_fit->GetOutput((float *)(&eval_rp),lm_state.reg_lambda) != LinearSolverException)
        {
            reg_params_ApplyLowerBound(&eval_rp,&bkg.my_regions.rp_low);
            reg_params_ApplyUpperBound(&eval_rp,&bkg.my_regions.rp_high);

            // make a copy so we can modify it to null changes in new_rp that will
            // we push into individual bead parameters
            reg_params new_rp = eval_rp;

            bkg.my_trace.GetShiftedBkg(new_rp.tshift,sbg);
            tshift_cache = new_rp.tshift;

            float new_reg_error = TryNewRegionalParameters(&new_rp, sbg);
            // are the new parameters better?
            if (new_reg_error < lm_state.reg_error)
            {
                // it's better...apply the change to all the beads and the region
                // update regional parameters
                bkg.my_regions.rp = new_rp;

                // re-calculate current parameter values for each bead as necessary
                UpdateBeadParametersFromRegion(&new_rp);

                lm_state.ReduceRegionStep();
                cont_proc = true;
            }
            else
            {
              cont_proc = lm_state.IncreaseRegionStep();
            }
        }
        else
        {
            cont_proc = lm_state.IncreaseRegionStep();
        }
    }
    if (defend_against_infinity>100)
        printf("RegionLevMar taking a while: %d\n",defend_against_infinity);
}



float MultiFlowLevMar::TryNewRegionalParameters(reg_params *new_rp, float *sbg)
{
    float new_reg_error = 0.0;
    //@TODO compute nuc rise step here to avoid recomputing

    // calculate new parameters for everything and re-check residual error
    for (int nbd=0;nbd<lm_state.ActiveBeads;nbd++)
    {
        int ibd = lm_state.fit_indicies[nbd];
        // if this iteration is a region-wide parameter fit, then only process beads
        // in the selection sub-group
        if (!lm_state.ValidBeadGroup(ibd) || (lm_state.well_region_fit[ibd]==false))
            continue;

        bead_params eval_params = bkg.my_beads.params_nn[ibd];

        // apply the region-wide adjustments to each individual well
        UpdateOneBeadFromRegion(&eval_params,&bkg.my_beads.params_high[ibd], &bkg.my_beads.params_low[ibd],new_rp,bkg.my_flow.dbl_tap_map);
        
        bkg.my_scratch.FillObserved(bkg.my_trace.fg_buffers, bkg.my_beads.params_nn[ibd].trace_ndx);
        FillScratchForEval(&eval_params,new_rp,sbg);
        // calculate error b/w current fit and actual data
        new_reg_error += bkg.my_scratch.CalculateFitError(NULL,NUMFB);
    }
    return(new_reg_error);
}


void MultiFlowLevMar::UpdateBeadParametersFromRegion(reg_params *new_rp)
{
    for (int nbd=0;nbd<lm_state.ActiveBeads;nbd++)
    {
        int ibd = lm_state.fit_indicies[nbd];
        UpdateOneBeadFromRegion(&bkg.my_beads.params_nn[ibd],&bkg.my_beads.params_high[ibd],&bkg.my_beads.params_low[ibd],new_rp, bkg.my_flow.dbl_tap_map);
    }
}

// reg_proc ==FALSE
int MultiFlowLevMar::LevMarFitToActiveBeadList(
    float *ival, float *sbg,
    bool well_only_fit,
    reg_params &eval_rp,
    BkgFitMatrixPacker *well_fit, unsigned int PartialDeriv_mask,
    int iter, float *fvdbg, float *BackgroundDebugTapA, float *BackgroundDebugTapB)
{
    int defend_against_infinity=0;
    int req_done = 0; // beads have improved?
    //@TODO nuc_rise step computed here, applied to all beads individually
    for (int nbd=0;(nbd < lm_state.ActiveBeads) && (defend_against_infinity<bkg.my_beads.numLBeads);defend_against_infinity++)
    {
        // get the index to the next bead to fit in the data matricies
        int ibd = lm_state.fit_indicies[nbd];

        // get the current parameter values for this bead
        bead_params eval_params = bkg.my_beads.params_nn[ibd];
        ComputePartialDerivatives(eval_params,eval_rp,ibd, PartialDeriv_mask,ival, sbg);

        if ((ibd == bkg.my_beads.DEBUG_BEAD) && (bkg.my_debug.trace_dbg_file != NULL))
        {
            bkg.DebugBeadIteration(eval_params,eval_rp, iter, ibd);
            bkg.my_trace.GetShiftedBkg(bkg.my_regions.rp.tshift,BackgroundDebugTapA);
            //CalcXtalkFlux(ibd,BackgroundDebugTapB);
            memcpy(fvdbg,bkg.my_scratch.fval,sizeof(float) *bkg.my_scratch.bead_flow_t);
        }

        // assemble jtj matrix and rhs matrix for per-well fitting
        // only the non-zero elements of computed
        // automatically start a new matrix
        BuildMatrix(well_fit,false, (ibd == bkg.my_beads.DEBUG_BEAD));
        // only if not doing regional processing do a bead step
        req_done += LevMarFitOneBead(ibd,nbd,sbg, eval_rp,well_fit,well_only_fit);

        // if we just removed a bead from the fitting process, don't advance since a
        // new bead is already in place at the current index (it was placed there in
        // place of the bead that was just removed)
        if (lm_state.advance_bd)
            nbd++;
    }
    return(req_done);
}


// the decision logic per iteration is particularly tortured in these routines
// and needs to be revisited
int MultiFlowLevMar::LevMarFitOneBead(int ibd, int &nbd,
                                float *sbg,
                               reg_params &eval_rp,
                               BkgFitMatrixPacker *well_fit,
                               bool well_only_fit)
{
    int bead_not_improved = 0;
    bead_params eval_params;
    lm_state.advance_bd = true;
    int defend_against_infinity=0;
    // we only need to re-calculate the PartialDeriv's if we actually adjust something
    // if the fit didn't improve...adjust lambda and retry right here, that way
    // we can save all the work of re-calculating the PartialDeriv's
    bool cont_proc = false;


    while ((!cont_proc) && (defend_against_infinity<EFFECTIVEINFINITY))
    {
        defend_against_infinity++;
        eval_params = bkg.my_beads.params_nn[ibd];
        bkg.my_scratch.FillObserved(bkg.my_trace.fg_buffers,eval_params.trace_ndx);
        float achg = 0.0;

        // solve equation and adjust parameters
        if (well_fit->GetOutput((float *)(&eval_params),lm_state.lambda[ibd]) != LinearSolverException)
        {
            // bounds check new parameters
            params_ApplyLowerBound(&eval_params,&bkg.my_beads.params_low[ibd]);
            params_ApplyUpperBound(&eval_params,&bkg.my_beads.params_high[ibd]);
            params_ApplyAmplitudeZeros(&eval_params, bkg.my_flow.dbl_tap_map); // double-tap

            FillScratchForEval(&eval_params, &eval_rp,sbg);
            float res = EvaluateNewBeadParameters(&eval_params);

            if (res < (lm_state.residual[ibd]))
            {
                achg=CheckSignificantSignalChange(&bkg.my_beads.params_nn[ibd],&eval_params,NUMFB);
                bkg.my_beads.params_nn[ibd] = eval_params;

                lm_state.ReduceBeadLambda(ibd);
                lm_state.residual[ibd] = res;

                cont_proc = true;
            }
            else
            {
                lm_state.IncreaseBeadLambda(ibd);
            }
        }
        else
        {
            if ((ibd == bkg.my_beads.DEBUG_BEAD) && (bkg.my_debug.trace_dbg_file != NULL))
            {
                fprintf(bkg.my_debug.trace_dbg_file,"singular matrix\n");
                fflush(bkg.my_debug.trace_dbg_file);
            }

            lm_state.IncreaseBeadLambda(ibd);
        }
        // if signal isn't making much progress, and we're ready to abandon lev-mar, deal with it
        if ((achg < 0.001) && (lm_state.lambda[ibd] >= lm_state.lambda_max))
        {
            bead_not_improved = 1;
            if (well_only_fit)
            {
                // this well is finished
                lm_state.FinishCurrentBead(ibd,nbd);
                cont_proc = true;
            }
        }

        // if regional fitting...we can get stuck here if we can't improve until the next
        // regional fit
        if (!well_only_fit && (lm_state.lambda[ibd] >= lm_state.lambda_escape))
            cont_proc = true;
    }
    if (defend_against_infinity>100)
        printf("Problem with bead %d %d\n", ibd, defend_against_infinity);
    return(bead_not_improved);
}


float MultiFlowLevMar::EvaluateNewBeadParameters(bead_params *eval_p)
{
    float res = 0.0;
    float scale = 0.0;
    
    for (int fnum=0;fnum <NUMFB;fnum++)
    {
        float eval;
        float rerr = 0;

        for (int i=0;i<bkg.time_c.npts;i++)   // real data only
        {
            int ti = i+fnum*bkg.time_c.npts;
            eval = (bkg.my_scratch.observed[ti]-bkg.my_scratch.fval[ti]);
            rerr += eval*eval;
            eval = eval*bkg.my_scratch.custom_emphasis[ti];
            res += eval*eval;
        }
        scale += bkg.my_scratch.custom_emphasis_scale[fnum];
        eval_p->rerr[fnum] = sqrt(rerr/ (float) bkg.time_c.npts);
    }
    RescaleRerr(eval_p,NUMFB);

    return(sqrt(res/scale));
}


float MultiFlowLevMar::ComputeLevMarResidual(float *y_minus_f_emphasized)
{
    float res = 0.0;
    float scale = 0.0;
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
      for (int i=0; i<bkg.time_c.npts; i++)
      {
        int ti = i+fnum*bkg.time_c.npts;
        res += y_minus_f_emphasized[ti]*y_minus_f_emphasized[ti];
      }
      scale += bkg.my_scratch.custom_emphasis_scale[fnum];
    }
    return(sqrt(res/scale));
}

// arguably this is part of "scratch space" operations and should be part of that object
void MultiFlowLevMar::ComputePartialDerivatives(bead_params &eval_params, reg_params &eval_rp, int ibd, unsigned int PartialDeriv_mask, float *ival, float *sbg)
{
    float *output;
    CpuStep_t *StepP;

    // make custom emphasis vector for this well using pointers to the per-HP vectors
    bkg.my_scratch.FillEmphasis(eval_params.WhichEmphasis,bkg.emphasis_data.EmphasisVectorByHomopolymer,bkg.emphasis_data.EmphasisScale);
    bkg.my_scratch.FillObserved(bkg.my_trace.fg_buffers, eval_params.trace_ndx);
    for (int step=0;step<bkg.fit_control.fitParams.NumSteps;step++)
    {
        StepP = &bkg.fit_control.fitParams.Steps[step];
        if ((StepP->PartialDerivMask & PartialDeriv_mask) == 0)
            continue; // only do the steps we are interested in.

        output = bkg.my_scratch.scratchSpace + step*bkg.my_scratch.bead_flow_t;

        ComputeOnePartialDerivative(output, StepP,  eval_params, eval_rp, ibd, ival, sbg);

        if ((ibd == bkg.my_beads.DEBUG_BEAD) && (bkg.my_debug.trace_dbg_file != NULL))
        {
            bkg.doDebug(StepP->name,StepP->diff,StepP->ptr);
        }
    }
}


void MultiFlowLevMar::ComputeOnePartialDerivative(float *output, CpuStep_t *StepP,  
                                           bead_params &eval_params, reg_params &eval_rp, int ibd, float *ival, float *sbg)
{
        if (StepP->PartialDerivMask == DFDGAIN)
        {
            Dfdgain_Step(output, &eval_params);
        }
        else if (StepP->PartialDerivMask == DFDERR)
        {
            Dfderr_Step(output, &eval_params);
        }
        else if (StepP->PartialDerivMask == YERR)
        {
            // this subtroutine uses both "eval_params" and "my_beads.params_nn", which is weird and possibly fixable
            // my_scratch.FillEmphasis(eval_params.WhichEmphasis,emphasis_data.EmphasisVectorByHomopolymer,emphasis_data.EmphasisScale);
            // this function does too much!
            Dfyerr_Step(output, &bkg.my_beads.params_nn[ibd]); // ??? why do we touch actual params and not eval params?
            lm_state.residual[ibd] = ComputeLevMarResidual(output);
        }
        else if (StepP->PartialDerivMask == DFDTSH)
        {
            // explicit calculation of PartialDeriv w.r.t. tshift
            float neg_sbg_slope[bkg.my_scratch.bead_flow_t];
            bkg.my_trace.GetShiftedSlope(eval_rp.tshift, neg_sbg_slope);
            MultiFlowComputePartialDerivOfTimeShift(output,&eval_params,&eval_rp,neg_sbg_slope);
        }
        else if (StepP->PartialDerivMask == FVAL) // set up the baseline for everything else
        {
            // fill in the function value & incorporation trace
            MultiFlowComputeCumulativeIncorporationSignal(&eval_params,&eval_rp,ival,bkg.my_regions,bkg.my_scratch.cur_bead_block,bkg.time_c,bkg.my_flow,bkg.math_poiss);
            MultiFlowComputeIncorporationPlusBackground(output,&eval_params,&eval_rp,ival,sbg,bkg.my_regions,bkg.my_scratch.cur_buffer_block,bkg.time_c,bkg.my_flow,bkg.use_vectorization, bkg.my_scratch.bead_flow_t);
                // add clonal restriction here to penalize non-integer clonal reads
            lm_state.ApplyClonalRestriction(output, &eval_params,bkg.time_c.npts);
        }
        else if (StepP->diff != 0)
        {
            float ivtmp[bkg.my_scratch.bead_flow_t];
            float backup[bkg.my_scratch.bead_flow_t]; // more than we need
            DoStepDiff(1,backup, StepP,&eval_params,&eval_rp);
            float *iv = ival;

            if (StepP->doBoth)
            {
                iv = ivtmp;
                //@TODO nuc rise recomputation?
                MultiFlowComputeCumulativeIncorporationSignal(&eval_params,&eval_rp,iv,bkg.my_regions,bkg.my_scratch.cur_bead_block,bkg.time_c,bkg.my_flow,bkg.math_poiss);
            }

            MultiFlowComputeIncorporationPlusBackground(output,&eval_params,&eval_rp,iv,sbg,bkg.my_regions,bkg.my_scratch.cur_buffer_block,bkg.time_c,bkg.my_flow,bkg.use_vectorization, bkg.my_scratch.bead_flow_t);
                // add clonal restriction here to penalize non-integer clonal reads
            lm_state.ApplyClonalRestriction(output, &eval_params,bkg.time_c.npts);

            CALC_PartialDeriv_W_EMPHASIS_LONG(bkg.my_scratch.fval,output,bkg.my_scratch.custom_emphasis,bkg.my_scratch.bead_flow_t,StepP->diff);

            DoStepDiff(0,backup, StepP,&eval_params,&eval_rp); // reset parameter to default value
        }
}


void MultiFlowLevMar::Dfdgain_Step(float *output,bead_params *eval_p) {

// partial w.r.t. gain is the function value divided by the current gain

    float* src[NUMFB];
    float* dst[NUMFB];
    float* em[NUMFB];
    // set up across flows
    for (int fnum=0;fnum < NUMFB;fnum++)
    {
        src[fnum] = &bkg.my_scratch.fval[bkg.time_c.npts*fnum];
        dst[fnum] = &output[bkg.time_c.npts*fnum];
        int emndx = eval_p->WhichEmphasis[fnum];
        em[fnum] = bkg.emphasis_data.EmphasisVectorByHomopolymer[emndx];
    }
    // execute in parallel
    if (bkg.use_vectorization)
        Dfdgain_Step_Vec(NUMFB, dst, src, em, bkg.time_c.npts, eval_p->gain);
    else {
        for (int fnum=0; fnum<NUMFB; fnum++) {
            for (int i=0;i < bkg.time_c.npts;i++)
                (dst[fnum])[i] = (src[fnum])[i]*(em[fnum])[i]/eval_p->gain;
        }
    }
}

void MultiFlowLevMar::Dfderr_Step(float *output, bead_params *eval_p)
{
    // partial w.r.t. darkness is the dark_matter_compensator multiplied by the emphasis

    float* dst[NUMFB];
    float* et[NUMFB];
    float* em[NUMFB];
    // set up
    for (int fnum=0;fnum < NUMFB;fnum++)
    {
        dst[fnum] = &output[bkg.time_c.npts*fnum];
        int emndx = eval_p->WhichEmphasis[fnum];
        em[fnum] = bkg.emphasis_data.EmphasisVectorByHomopolymer[emndx];
        et[fnum] = &bkg.my_regions.dark_matter_compensator[bkg.my_flow.flow_ndx_map[fnum]*bkg.time_c.npts];
    }
    //execute
    if (bkg.use_vectorization)
        Dfderr_Step_Vec(NUMFB, dst, et, em, bkg.time_c.npts);
    else {
        for (int fnum=0; fnum<NUMFB;fnum++) {
            for (int i=0;i < bkg.time_c.npts;i++)
                (dst[fnum])[i] = (et[fnum])[i]*(em[fnum])[i];
        }
    }
}

void MultiFlowLevMar::Dfyerr_Step(float *y_minus_f_emphasized, bead_params *actual_p)
{

    for (int fnum=0;fnum < NUMFB;fnum++)
    {
        float rerr = 0.0;
        float eval;

        for (int i=0;i<bkg.time_c.npts;i++)   // real data only
        {
            int ti= i+fnum*bkg.time_c.npts;
            eval = bkg.my_scratch.observed[ti]-bkg.my_scratch.fval[ti];
            rerr += eval*eval;
            eval = eval*bkg.my_scratch.custom_emphasis[ti];
            y_minus_f_emphasized[ti] = eval;

        }
        actual_p->rerr[fnum] = sqrt(rerr/ (float) bkg.time_c.npts);
    }
    RescaleRerr(actual_p,NUMFB);
}

// this is the one PartialDeriv that really isn't computed very well w/ the stansard numeric computation method
void MultiFlowLevMar::MultiFlowComputePartialDerivOfTimeShift(float *fval,struct bead_params *p,struct reg_params *reg_p, float *neg_sbg_slope)
{

    int fnum;

    float *vb_out;
    float *flow_deriv;

    // parallel fill one bead parameter for block of flows
    FillBufferParamsBlockFlows(&bkg.my_scratch.cur_buffer_block,p,reg_p,bkg.my_flow.flow_ndx_map,bkg.my_flow.buff_flow);
    bkg.my_scratch.FillEmphasis(p->WhichEmphasis,bkg.emphasis_data.EmphasisVectorByHomopolymer,bkg.emphasis_data.EmphasisScale);

    for (fnum=0;fnum<NUMFB;fnum++)
    {
        vb_out = fval + fnum*bkg.time_c.npts;              // get ptr to start of the function evaluation for the current flow
        flow_deriv = &neg_sbg_slope[fnum*bkg.time_c.npts];                   // get ptr to pre-shifted slope

        // now this diffeq looks like all the others
        // because linearity of derivatives gets passed through
        // flow_deriv in this case is in fact the local derivative with respect to time of the background step
        // so it can be passed through the equation as though it were a background term
        BlueSolveBackgroundTrace(vb_out,flow_deriv,bkg.time_c.npts,bkg.time_c.deltaFrame,bkg.my_scratch.cur_buffer_block.tauB[fnum],bkg.my_scratch.cur_buffer_block.etbR[fnum]);
        // isolate gain and emphasis so we can reuse diffeq code
    }
    MultiplyVectorByVector(fval,bkg.my_scratch.custom_emphasis,bkg.my_scratch.bead_flow_t);
   // gain invariant so can be done to all at once
    MultiplyVectorByScalar(fval,p->gain,bkg.my_scratch.bead_flow_t);
}

void DoStepDiff(int add, float *archive, CpuStep_t *step, struct bead_params *p, struct reg_params *reg_p)
{
    int i;
    float *dptr = NULL;
    // choose my parameter
    if (step->paramsOffset != NOTBEADPARAM)
    {
        dptr = (float *)((char *) p + step->paramsOffset);
    }
    else if (step->regParamsOffset != NOTREGIONPARAM)
    {
        dptr = (float *)((char *) reg_p + step->regParamsOffset);
    } else if (step->nucShapeOffset != NOTNUCRISEPARAM)
    {
        dptr = (float *)((char *) &(reg_p->nuc_shape) + step->nucShapeOffset);
    }
    // update my parameter
    if (dptr != NULL)
    {
        for (i=0;i<step->len;i++)
        {
            if (add){
                archive[i] = *dptr;
                *dptr += step->diff; // change
            }else
                *dptr = archive[i]; // reset
            dptr++;
        }
    }
}

void UpdateOneBeadFromRegion(bead_params *p, bead_params *hi, bead_params *lo, reg_params *new_rp, int *dbl_tap_map)
{
        for (int i=0;i < NUMFB;i++)
            p->Ampl[i] += new_rp->Ampl[i];
        p->R += new_rp->R;
        p->Copies += new_rp->Copies;

        params_ApplyLowerBound(p,lo);
        params_ApplyUpperBound(p,hi);
        params_ApplyAmplitudeZeros(p,dbl_tap_map);  // double-taps
}

void IdentifyParameters(BeadTracker &my_beads, RegionTracker &my_regions, unsigned int well_mask, unsigned int reg_mask)
{
            if ((well_mask & DFDPDM)>0)  // only if actually fitting dmult do we need to do this step 
              IdentifyDmult(my_beads,my_regions);
            if ((reg_mask & DFDMR) >0) // only if actually fitting NucMultiplyRatio do I need to identify NMR so we don't slide around randomly
              IdentifyNucMultiplyRatio(my_beads, my_regions);
}

void IdentifyDmult(BeadTracker &my_beads, RegionTracker &my_regions)
{
    float mean_dmult = my_beads.CenterDmult();
    
    for (int nnuc=0;nnuc < NUMNUC;nnuc++)
    {
        my_regions.rp.d[nnuc] *= mean_dmult;
    }
}

void IdentifyNucMultiplyRatio(BeadTracker &my_beads, RegionTracker &my_regions)
{
  // provide identifiability constraint
  float mean_x = 0.0;
  for (int nnuc=0; nnuc<NUMNUC; nnuc++)
  {
    mean_x += my_regions.rp.NucModifyRatio[nnuc];
  }
  mean_x /=NUMNUC;
  
  for (int nnuc=0; nnuc<NUMNUC; nnuc++)
  {
    my_regions.rp.NucModifyRatio[nnuc] /=mean_x;
  }
  
  my_beads.RescaleRatio(1.0/mean_x);
}
