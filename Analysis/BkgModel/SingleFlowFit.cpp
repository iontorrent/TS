/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SingleFlowFit.h"


single_flow_optimizer::single_flow_optimizer()
{
    // this can't get allocated until after we know how many data points we will
    // be processing
    oneFlowFit = NULL;
    oneFlowFitKrate = NULL;
    for (int i=0; i<NUMFB; i++)
      decision_threshold[i] = 0;
}

void single_flow_optimizer::Delete()
{
    if (oneFlowFitKrate!=NULL) delete oneFlowFitKrate;
    if (oneFlowFit!=NULL) delete oneFlowFit;
}

single_flow_optimizer::~single_flow_optimizer()
{
  Delete();
}

void single_flow_optimizer::FillDecisionThreshold(float *nuc_threshold, int *my_nucs)
{
  for (int i=0; i<NUMFB; i++)
    decision_threshold[i] = nuc_threshold[my_nucs[i]];
}

void single_flow_optimizer::SetLowerLimitAmplFit(float AmplLim,float krateLim, float dmultLim)
{
    // not all may apply

    pbound.Ampl = AmplLim;
    oneFlowFit->SetParamMin(pbound);
    pboundkrate.Ampl = AmplLim;
    pboundkrate.kmult = krateLim;  // conservative variation in kmult
    pboundkrate.dmultX = dmultLim;
    oneFlowFitKrate->SetParamMin(pboundkrate);
}


void single_flow_optimizer::SetUpperLimitAmplFit(float AmplLim,float krateLim, float dmultLim)
{
    // not all may apply

    pbound.Ampl = AmplLim;
    oneFlowFit->SetParamMax(pbound);
    pboundkrate.Ampl = AmplLim;
    pboundkrate.kmult = krateLim;  // conservative variation in kmult
    pboundkrate.dmultX = dmultLim;
    oneFlowFitKrate->SetParamMax(pboundkrate);
}

void single_flow_optimizer::AllocLevMar(TimeCompression &time_c, PoissonCDFApproxMemo *math_poiss, float damp_kmult)
{
    oneFlowFit = new BkgModSingleFlowFit(time_c.npts,time_c.frameNumber,time_c.deltaFrame,time_c.deltaFrameSeconds,math_poiss);
   oneFlowFitKrate = new BkgModSingleFlowFitKrate(time_c.npts,time_c.frameNumber,time_c.deltaFrame,time_c.deltaFrameSeconds, math_poiss, false); // @TODO try a warped fit

   // set up levmar state before going off on a wild adventure
    oneFlowFit->SetLambdaThreshold(10.0);
    oneFlowFitKrate->SetLambdaThreshold(1.0);

    SetLowerLimitAmplFit(MINAMPL,0.65,0.5);
    SetUpperLimitAmplFit(MAX_HPLEN-1,1.75,1.0);
   
    float my_prior[2] = {0.0, 1.0}; // amplitude, kmult
    float my_damper[2] = {0.0,damp_kmult*time_c.npts}; // must make a difference to change kmult
    oneFlowFitKrate->SetPrior(my_prior);
    oneFlowFitKrate->SetDampers(my_damper);
}

void single_flow_optimizer::FitKrateOneFlow(int fnum, float *evect, bead_params *p, float *signal_corrected, float *nucRise, int *i_start, 
                                      flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{
         int NucID = my_flow.flow_ndx_map[fnum];

          oneFlowFitKrate->SetWeightVector(evect);
            oneFlowFitKrate->SetLambdaStart(1E-20);
            oneFlowFitKrate->SetWellRegionParams(p,&my_regions.rp,fnum,
                                                 NucID,my_flow.buff_flow[fnum],
                                                 i_start[NucID],&nucRise[NucID*time_c.npts*ISIG_SUB_STEPS_SINGLE_FLOW]);
            // evaluate the fancier model
            oneFlowFitKrate->Fit(NUMSINGLEFLOWITER,signal_corrected);
            p->Ampl[fnum] = oneFlowFitKrate->params.Ampl;
            p->kmult[fnum] = oneFlowFitKrate->params.kmult;
          
            // store output for later
        oneFlowFitKrate->SetWeightVector(emphasis_data.EmphasisVectorByHomopolymer[emphasis_data.numEv-1]);
        p->rerr[fnum] = sqrt(oneFlowFitKrate->GetResidual(signal_corrected)); // backwards compatibility
}



void single_flow_optimizer::FitThisOneFlow(int fnum, float *evect, bead_params *p,  float *signal_corrected, float *nucRise, int *i_start,
                                           flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{ 
        int NucID = my_flow.flow_ndx_map[fnum];

        oneFlowFit->SetWeightVector(evect);
        // SetWellParams should leave data invariant
        oneFlowFit->SetLambdaStart(1E-20);
        oneFlowFit->SetWellRegionParams(p,&my_regions.rp,fnum,
                                        NucID,my_flow.buff_flow[fnum],
                                        i_start[NucID],&nucRise[NucID*time_c.npts*ISIG_SUB_STEPS_SINGLE_FLOW]);

        oneFlowFit->Fit(NUMSINGLEFLOWITER,signal_corrected);  // Not enough evidence to warrant krate fitting to this flow, do the simple thing.
        p->Ampl[fnum] = oneFlowFit->params.Ampl;

        // re-calculate residual based on a the highest hp weighting vector (which is the most flat)
        oneFlowFit->SetWeightVector(emphasis_data.EmphasisVectorByHomopolymer[emphasis_data.numEv-1]);
        p->rerr[fnum] = sqrt(oneFlowFit->GetResidual(signal_corrected)); // backwards compatibility
}

void single_flow_optimizer::FitOneFlow(int fnum, float *evect, bead_params *p,  float *signal_corrected, float *nucRise, int *i_start,
                                           flow_buffer_info &my_flow, TimeCompression &time_c, EmphasisClass &emphasis_data,RegionTracker &my_regions)
{
        bool krate_fit = (((p->Copies*p->Ampl[fnum]) > decision_threshold[fnum])); 
        // krate_fit = true; // if using new style of optimizer
       if (krate_fit)
       {
         FitKrateOneFlow(fnum,evect,p,signal_corrected,nucRise, i_start,my_flow,time_c,emphasis_data,my_regions);
       }
       else
       {
        FitThisOneFlow(fnum,evect,p, signal_corrected, nucRise, i_start,my_flow,time_c,emphasis_data,my_regions);
       }
}
