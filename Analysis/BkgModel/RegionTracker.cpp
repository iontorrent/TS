/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "RegionTracker.h"

RegionTracker::RegionTracker()
{
    nuc_rise_fine_step = NULL;
    nuc_rise_coarse_step = NULL;
    dark_matter_compensator  = NULL;
    nuc_flow_t = 0;
    for (int NucID=0;NucID<NUMNUC;NucID++)
    {
        i_start_coarse_step[NucID] = 0;
        i_start_fine_step[NucID] = 0;
    }
}

void RegionTracker::AllocScratch(int npts)
{
    // set up buffer here as time_c tells me how many points I have
     nuc_flow_t = NUMNUC * npts; // some objects repeat per nucleotide
   if (nuc_rise_coarse_step==NULL)
        nuc_rise_coarse_step=new float[nuc_flow_t*ISIG_SUB_STEPS_MULTI_FLOW];
    if (nuc_rise_fine_step==NULL)
        nuc_rise_fine_step=new float[nuc_flow_t*ISIG_SUB_STEPS_SINGLE_FLOW];
    dark_matter_compensator  = new float[nuc_flow_t];
   ResetDarkMatter();
}

void RegionTracker::ResetDarkMatter()
{
    memset(dark_matter_compensator,0,sizeof(float[nuc_flow_t]));
}

void RegionTracker::Delete()
{
    if (nuc_rise_coarse_step !=NULL) delete [] nuc_rise_coarse_step;
    if (nuc_rise_fine_step !=NULL) delete [] nuc_rise_fine_step;
    
    if (dark_matter_compensator != NULL) delete [] dark_matter_compensator;
}


void RegionTracker::CalculateNucRiseFineStep(reg_params *a_region, TimeCompression &time_c)
{
    DntpRiseModel dntpMod(time_c.npts,a_region->nuc_shape.C,time_c.frameNumber,ISIG_SUB_STEPS_SINGLE_FLOW);

    // compute the nuc rise model for each nucleotide...in the long run could be separated out
    // and computed region-wide
    // at the moment, this computation is identical for all wells.  If it were not, we couldn't
    // do it here.
    for (int NucID=0;NucID<NUMNUC;NucID++)
    {
        i_start_fine_step[NucID]=dntpMod.CalcCDntpTop(&nuc_rise_fine_step[NucID*time_c.npts*ISIG_SUB_STEPS_SINGLE_FLOW],
                                 GetModifiedMidNucTime(&(a_region->nuc_shape),NucID),
                                 GetModifiedSigma(&(a_region->nuc_shape),NucID));
    }
}

void RegionTracker::CalculateNucRiseCoarseStep( reg_params *my_region, TimeCompression &time_c)
{
    DntpRiseModel dntpMod(time_c.npts,my_region->nuc_shape.C,time_c.frameNumber,ISIG_SUB_STEPS_MULTI_FLOW);

    // compute the nuc rise model for each nucleotide...in the long run could be separated out
    // and computed region-wide
    // at the moment, this computation is identical for all wells.  If it were not, we couldn't
    // do it here.
    for (int NucID=0;NucID<NUMNUC;NucID++)
    {
        i_start_coarse_step[NucID]=dntpMod.CalcCDntpTop(&nuc_rise_coarse_step[NucID*time_c.npts*ISIG_SUB_STEPS_MULTI_FLOW],
                                   GetModifiedMidNucTime(&(my_region->nuc_shape),NucID),
                                   GetModifiedSigma(&(my_region->nuc_shape),NucID));
    }
}


void RegionTracker::RestrictRatioDrift()
{
    // we don't allow the RatioDrift term to increase after it's initial fitted value is determined
    rp_high.RatioDrift = rp.RatioDrift;

    if (rp_high.RatioDrift < MIN_RDR_HIGH_LIMIT) rp_high.RatioDrift = MIN_RDR_HIGH_LIMIT;
}

void RegionTracker::NormalizeDarkMatter(float *scale_factor, int npts)
{
   // now normalize all the averages.  If any don't contain enough to make a good average
    // replace the error term with all ones
    for (int nnuc=0;nnuc < NUMNUC;nnuc++)
    {
        float *et = &dark_matter_compensator[nnuc*npts];

        if (scale_factor[nnuc] > 1)
        {
            for (int i=0;i<npts;i++)
                et[i] = et[i]/scale_factor[nnuc];
        }
        else
        {
            for (int i=0;i<npts;i++)
                et[i] = 0.0;  // no data, no offset
        }
    }
  
}

RegionTracker::~RegionTracker()
{
  Delete();
}

void RegionTracker::DumpDarkMatterTitle(FILE *my_fp)
{
  // ragged columns because of variable time compression
  fprintf(my_fp, "col\trow\tNucID\t");
  for (int j=0; j<40; j++)
    fprintf(my_fp,"V%d\t",j);
  fprintf(my_fp,"Neat");
  fprintf(my_fp,"\n");
}

void RegionTracker::DumpDarkMatter(FILE *my_fp, int x, int y)
{
  // this is a little tricky across regions, as time compression is somewhat variable
  // 4 lines, one per nuc_flow
  for (int NucID=0; NucID<NUMNUC; NucID++)
  {
    fprintf(my_fp, "%d\t%d\t%d\t", x,y, NucID);
    int npts = nuc_flow_t/4;
    int j=0;
    for (; j<npts; j++)
      fprintf(my_fp,"%0.3f\t", dark_matter_compensator[NucID*npts+j]);
    int max_npts = 40;  // always at least this much time compression?
    for (;j<max_npts; j++)
       fprintf(my_fp,"%0.3f\t", 0.0);
    fprintf(my_fp,"%f",rp.darkness[0]);
    fprintf(my_fp, "\n");
  }
}

// Region parameters & box constraints

void RegionTracker::InitHighRegionParams(float t_mid_nuc_start)
{
    reg_params_setStandardHigh(&rp_high,t_mid_nuc_start);
}

void RegionTracker::InitLowRegionParams(float t_mid_nuc_start)
{
    reg_params_setStandardLow(&rp_low,t_mid_nuc_start);
}

void RegionTracker::InitModelRegionParams(float t_mid_nuc_start,float sigma_start, float dntp_concentration_in_uM,GlobalDefaultsForBkgModel &global_defaults)
{
    // s is overly complex
    reg_params_setStandardValue(&rp,t_mid_nuc_start,sigma_start, dntp_concentration_in_uM);
    reg_params_setKrate(&rp,global_defaults.krate_default);
    reg_params_setDiffusion(&rp,global_defaults.d_default);
    reg_params_setKmax(&rp,global_defaults.kmax_default);
    reg_params_setSens(&rp,global_defaults.sens_default);
    reg_params_setBuffModel(&rp,global_defaults.tau_R_m_default,global_defaults.tau_R_o_default);
    if (global_defaults.no_RatioDrift_fit_first_20_flows)
        reg_params_setNoRatioDriftValues(&rp);
}

void RegionTracker::InitRegionParams(float t_mid_nuc_start,float sigma_start, float dntp_concentration_in_uM,GlobalDefaultsForBkgModel &global_defaults)
{
    InitHighRegionParams(t_mid_nuc_start);
    InitLowRegionParams(t_mid_nuc_start);
    InitModelRegionParams(t_mid_nuc_start,sigma_start,dntp_concentration_in_uM,global_defaults);
    // bounds check all these to be sure
    reg_params_ApplyLowerBound(&rp,&rp_low);
    reg_params_ApplyUpperBound(&rp,&rp_high);
}