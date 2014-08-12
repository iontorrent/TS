/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DarkMatter.h"
#include "TraceCorrector.h"

using namespace std;

Axion::Axion (SignalProcessingMasterFitter &_bkg) :
    bkg (_bkg)
{

}


void Axion::AccumulateOneBead(BeadParams *p, reg_params *reg_p, int max_fnum, 
    float my_residual, float res_threshold, int flow_block_size, int flow_block_start )
{
    float tmp_res[bkg.region_data->my_scratch.bead_flow_t];
  
    bkg.region_data->my_scratch.FillObserved (bkg.region_data->my_trace, p->trace_ndx, flow_block_size);
    //params_IncrementHits(p);
    MathModel::MultiFlowComputeCumulativeIncorporationSignal (p,reg_p,bkg.region_data->my_scratch.ival,bkg.region_data->my_regions.cache_step,*bkg.region_data_extras.cur_bead_block,bkg.region_data->time_c,
      *bkg.region_data_extras.my_flow,bkg.math_poiss, flow_block_size, flow_block_start);
    MathModel::MultiFlowComputeTraceGivenIncorporationAndBackground (bkg.region_data->my_scratch.fval,p,reg_p,bkg.region_data->my_scratch.ival,bkg.region_data->my_scratch.shifted_bkg,bkg.region_data->my_regions,*bkg.region_data_extras.cur_buffer_block,bkg.region_data->time_c,
      *bkg.region_data_extras.my_flow,bkg.global_defaults.signal_process_control.use_vectorization, bkg.region_data->my_scratch.bead_flow_t, flow_block_size, flow_block_start);    // evaluate the function
    // calculate error b/w current fit and actual data
    bkg.region_data->my_scratch.MultiFlowReturnResiduals (tmp_res);  // wait, didn't I just compute the my_residual implicitly here?

    //@TODO: compute "my_residual" here so I can avoid using lev-mar state
    // aggregate error to find first-order correction for each nuc type
    for (int fnum=0;fnum < max_fnum;fnum++)
    {
      // if our bead has enough signal in this flow to be interesting we include it
      // if our bead is close enough in mean-square distance we include it
      // exclude small amplitude but high mean-square beads
      if ((my_residual>res_threshold) && (p->Ampl[fnum]<0.3) )
        continue;

      bkg.region_data->my_regions.missing_mass.AccumulateDarkMatter(
        &tmp_res[bkg.region_data->time_c.npts()*fnum],
        bkg.region_data_extras.my_flow->flow_ndx_map[fnum]);

    }
}


void Axion::AccumulateResiduals(reg_params *reg_p, int max_fnum, float *residual, float res_threshold, int flow_block_size, int flow_block_start)
{

  for (int ibd=0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
  {
    if ( bkg.region_data->my_beads.Sampled(ibd) ) {
      // get the current parameter values for this bead
      BeadParams *p = &bkg.region_data->my_beads.params_nn[ibd];
      AccumulateOneBead(p,reg_p, max_fnum, residual[ibd], res_threshold, flow_block_size, flow_block_start);
    }
  }
}

void Axion::CalculateDarkMatter (int max_fnum, float *residual, float res_threshold,
    int flow_block_size, int flow_block_start
  )
{
  bkg.region_data->my_regions.missing_mass.mytype = PerNucAverage;
  bkg.region_data->my_regions.missing_mass.ResetDarkMatter();
  // prequel, set up standard bits
  reg_params *reg_p = & bkg.region_data->my_regions.rp;
  bkg.region_data->my_scratch.FillShiftedBkg(*bkg.region_data->emptytrace, reg_p->tshift, bkg.region_data->time_c, true, flow_block_size);
  bkg.region_data->my_regions.cache_step.ForceLockCalculateNucRiseCoarseStep(
    reg_p,bkg.region_data->time_c,*bkg.region_data_extras.my_flow);

  AccumulateResiduals(reg_p,max_fnum, residual, res_threshold, flow_block_size, flow_block_start);

  bkg.region_data->my_regions.missing_mass.NormalizeDarkMatter ();
  // make sure everything is happy in the rest of the code
  bkg.region_data->my_regions.missing_mass.training_only = false; // now we've trained
  bkg.region_data->my_regions.cache_step.Unlock();
}

// TODO: should probably put this in math util or some other shared-location
void Axion::smooth_kern(float *out, float *in, float *kern, int dist, int npts)
{
   float sum;
   float scale;

   for (int i = 0; i < npts; i++) {
      sum = 0.0f;
      scale = 0.0f;

      for (int j = i - dist, k = 0; j <= (i + dist); j++, k++) {
         if ((j >= 0) && (j < npts)) {
            sum += kern[k] * in[j];
            scale += kern[k];
         }
      }
      out[i] = sum / scale;
   }
}

int Axion::Average0MerOneBead(int ibd,float *avg0p, int flow_block_size, int flow_block_start )
{
   float block_signal_corrected[bkg.region_data->my_trace.npts * flow_block_size];

   bkg.trace_bkg_adj->ReturnBackgroundCorrectedSignal (block_signal_corrected, ibd, flow_block_size,
      flow_block_start );

   float tmp[bkg.region_data->time_c.npts()];
   memset(tmp,0,sizeof(tmp));

   int ncnt=0;
   for (int fnum = 0; fnum < flow_block_size; fnum++)
      if (bkg.region_data->my_beads.params_nn[ibd].Ampl[fnum] < 0.5f)
      {
         for (int i = 0; i < bkg.region_data->time_c.npts(); i++)
            tmp[i] += block_signal_corrected[fnum * bkg.region_data->time_c.npts() + i];

         ncnt++;
      }

   // this will force the output to all 0's if we didn't find enough 0-mers to average
   float scale = 1.0f/ncnt;
   if (ncnt < 4)
      scale = 0.0f;

   for (int i = 0; i < bkg.region_data->time_c.npts(); i++)
      avg0p[i] = tmp[i]*scale;

   return(ncnt);
}

void Axion::PCACalc(arma::fmat &avg_0mers,bool region_sampled)
{
   int npts = bkg.region_data->time_c.npts();
   int numLBeads = bkg.region_data->my_beads.numLBeads;
   arma::fmat coeff;
   arma::fmat score;
   arma::fmat amat;
   arma::fmat vals;
   arma::fmat mat_obs;
   arma::fmat region_mean_0mer;
   float samp_rate = (float)TARGET_PCA_SET_SIZE/numLBeads;
   int icnt = 0;
   int ilast = -1;
   int total_vectors;
   float kern[3] = {1.0f,1.0f,1.0f};
   int target_sample_size;

   if (region_sampled)
      target_sample_size = NUMBEADSPERGROUP;
   else
      target_sample_size = TARGET_PCA_SET_SIZE;

   // if we don't have very many example traces...skip the PCA analysis and just resort to using the mean trace as a single vector
   if (numLBeads >= MIN_PCA_SET_SIZE)
   {
      // create an evenly distributed subset of traces for PCA analysis
      // TARGET_PCA_SET_SIZE is the desired number of traces to be used
      mat_obs.set_size(target_sample_size,npts);

      float samp = 0.0f;
      for (int ibd=0;(ibd < numLBeads) && (icnt < target_sample_size);ibd++)
      {
         int isamp = (int)samp;
         bool sample_read;

         if (region_sampled)
            sample_read = (bkg.region_data->my_beads.Sampled(ibd));
         else
            sample_read = (isamp > ilast);

         if (sample_read)
         {
            float trc[npts];
            float trc_smooth[npts];

            for (int i=0;i < npts;i++)
               trc[i] = avg_0mers(ibd,i);

            smooth_kern(trc_smooth, trc, kern, 1, bkg.region_data->time_c.npts());

            for (int i=0;i < npts;i++)
               mat_obs(icnt,i) = trc_smooth[i];

            ilast = icnt++;
         }

         samp += samp_rate;
      }
      // just in case we wound up with fewer than the target...resize it to only the valid
      // rows
      if (icnt < target_sample_size)
         mat_obs.resize(icnt,npts);

      bool success = false;

      try {
	arma::fmat X = mat_obs.t() * mat_obs;
	arma::fmat evec;
	arma::Col<float> eval;
	arma::eig_sym(eval, evec, X);
	amat.set_size(X.n_cols, NUM_DM_PCA);
	int count = 0;
	// Copy first N eigen vectors
	for (int vIx = X.n_cols - 1; vIx >= (int)(X.n_cols - (NUM_DM_PCA)) && vIx >= 0; vIx--) {
	  std::copy(evec.begin_col(vIx), evec.end_col(vIx), amat.begin_col(count++));
	}
	success = true;
      }
      catch (...) {
	// @todo - warning or something here?
       	success = false;
      }

      if (success)
	total_vectors = NUM_DM_PCA;
      else 
	total_vectors = 1;
   }
   else
   {
      // we shouldn't need mat_obs if we get here..but just in case set it to something reasonable
      mat_obs.set_size(1,npts);
      mat_obs.fill(1.0f);
      total_vectors = 1;
   }

   // If not doing pca set up the amat matrix (vector in this case really)
   if (total_vectors == 1) {
     if (region_sampled)
       region_mean_0mer = arma::mean(mat_obs,0).t();
     else
       region_mean_0mer = arma::mean(avg_0mers,0).t();
     amat.set_size(npts, total_vectors);
     amat(arma::span::all,0) = region_mean_0mer;
     vals = solve(amat,avg_0mers.t());
     vals = vals.t();
   }
   else {
     vals = avg_0mers * amat;
   }

   // store coefficients for each bead into the bead params structure
   for (int ibd=0;ibd < numLBeads;ibd++)
      for (int i=0;i < total_vectors;i++)
	bkg.region_data->my_beads.params_nn[ibd].pca_vals[i] = vals(ibd,i);

   // store the components of the PCA vector into the DarkHalo object
   for (int icomp=0;icomp < total_vectors;icomp++)
   {
      float tmp[npts];
      for (int i=0;i < npts;i++)
         tmp[i] = amat(i,icomp);

      bkg.region_data->my_regions.missing_mass.AccumulateDarkMatter(tmp,icomp);
   }
}

void Axion::CalculatePCADarkMatter (bool region_sampled, int flow_block_size, int flow_block_start )
{
   bkg.region_data->my_regions.missing_mass.mytype = PCAVector;
   bkg.region_data->my_regions.missing_mass.ResetDarkMatter();

   // if there are very few reads...then we shouldn't attempt this
   // if we bail here...all the vectors will be all-0s and all the
   // coefficients will be 0, so dark matter correction will become
   // a null operation
   if (bkg.region_data->my_beads.numLBeads < MIN_AVG_DM_SET_SIZE)
      return;

   bkg.region_data->my_scratch.FillShiftedBkg(*bkg.region_data->emptytrace, bkg.region_data->my_regions.rp.tshift, bkg.region_data->time_c, true, flow_block_size);
   bkg.region_data->my_regions.cache_step.ForceLockCalculateNucRiseCoarseStep(
    &bkg.region_data->my_regions.rp,bkg.region_data->time_c,*bkg.region_data_extras.my_flow);

   arma::fmat avg_0mers;

   avg_0mers.set_size(bkg.region_data->my_beads.numLBeads,bkg.region_data->time_c.npts());
   avg_0mers.fill(0.0f);

   // calculate the average 0-mer for each bead
   for (int ibd=0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
   {
      float avg_0mer[bkg.region_data->time_c.npts()];

      if ((region_sampled && bkg.region_data->my_beads.Sampled(ibd)) || ~region_sampled)
      {
         Average0MerOneBead(ibd,avg_0mer, flow_block_size, flow_block_start);

         // now add this to the list of observed traces
         for (int i=0;i < bkg.region_data->time_c.npts();i++)
            avg_0mers(ibd,i) = avg_0mer[i];
      }
   }

   // represent the average 0-mers using the NUM_DM_PCA most significant PCA vectors across all 0-mer signals
   PCACalc(avg_0mers,region_sampled);

   // make sure everything is happy in the rest of the code
   bkg.region_data->my_regions.cache_step.Unlock();
}

