/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "BkgModel.h"

RcppExport SEXP bkgModel(
  SEXP Rdat, 
  SEXP Rbkg, 
  SEXP RnFrame,
  SEXP RnFlow,
  SEXP RflowOrder, 
  SEXP RmaxFlow,
  SEXP Rsigma_guess,
  SEXP Rt0_guess,
  SEXP Rdntp_uM
) {

  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    RcppMatrix<double> dat(Rdat);
    int nWell             = dat.rows();
    int nCol              = dat.cols();
    RcppMatrix<double> bkg(Rbkg);
    int nFrame            = Rcpp::as<int>(RnFrame);
    int nFlow             = Rcpp::as<int>(RnFlow);
    const char* flowOrder = Rcpp::as<const char*>(RflowOrder);
    int maxFlow           = Rcpp::as<int>(RmaxFlow);
    float sigma_guess     = Rcpp::as<float>(Rsigma_guess);
    float t0_guess        = Rcpp::as<float>(Rt0_guess);

    float dntp_uM         = Rcpp::as<float>(Rdntp_uM);

    if(nWell <= 0) {
      std::string exception = "Empty matrix supplied, nothing to fit\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else if(bkg.rows() != nFrame) {
      std::string exception = "Number of rows for bkg matrix should equal number of frames\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else if(bkg.cols() != nFlow) {
      std::string exception = "Number of cols for bkg matrix should equal number of flows\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else if(maxFlow >= nFlow) {
      std::string exception = "maxFlow must be less than nFlow\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else if(nFlow*nFrame != nCol) {
      std::string exception = "Number of columns in signal matrix should equal nFrame * nFlow\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else if(strlen((char *)flowOrder) != 4) {
      std::string exception = "flowOrder must be of length 4\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else {
      // Allocate memory for returned data
      // Per-frame
      RcppVector<double> out_sim_bg(nFlow*nFrame);
      RcppVector<double> out_bg(nFlow*nFrame);
      // Per-well, per-flow
      RcppMatrix<double> out_sim_fg(nWell,nFlow*nFrame);
      RcppMatrix<double> out_sig(nWell,nFlow);
      // Per-well, per-flowBatch
      int nFlowBatch = ceil((double) (nFlow/ (double) NUMFB));
      RcppMatrix<double> out_R(nWell,nFlowBatch);
      RcppMatrix<double> out_dt(nWell,nFlowBatch);
      RcppMatrix<double> out_Copies(nWell,nFlowBatch);
      RcppMatrix<double> out_gain(nWell,nFlowBatch);
      // Per-region, per-flowBatch
      RcppMatrix<double> out_krate(NUMNUC,nFlowBatch);
      RcppMatrix<double> out_d(NUMNUC,nFlowBatch);
      RcppVector<double> out_sigma(nFlowBatch);
      RcppVector<double> out_tshift(nFlowBatch);
      // For returning model inputs
      RcppMatrix<double> out_sens(NUMNUC,nFlowBatch);
      double out_C = 0;

      // Initialize background model.
      // For now we make sure we only ever do this once, there is something
      // broken in the memory handling and if called twice it will seg fault.
      // In future we might change this given changes in BkgModel.cpp
      static bool first=true;
       static PoissonCDFApproxMemo poiss_cache; // math routines the bkg model needs to do a lot
      if(first) {
        first = false;
        
        poiss_cache.Allocate(MAX_HPLEN+1,512,0.05);
        poiss_cache.GenerateValues(); // fill out my table
        int my_nuc_block[NUMFB];
        // TODO: Objects should be isolated!!!!
        GlobalDefaultsForBkgModel::SetFlowOrder((char *) flowOrder);
        GlobalDefaultsForBkgModel::GetFlowOrderBlock(my_nuc_block,0,NUMFB);
        InitializeLevMarSparseMatrices(my_nuc_block);
      }
      BkgModel *bkgmodel = new BkgModel(nWell,nFrame,sigma_guess,t0_guess,dntp_uM);

      // poisson cache
      bkgmodel->math_poiss = &poiss_cache;

      if(maxFlow < 0)
        maxFlow = nFlow-1;
      struct bead_params p;
      struct reg_params rp;
      short *imgBuffer = new short[nWell * nFrame];
      short *bkgBuffer = new short[nFrame];
      int fitFramesPerFlow = 0;
      for(int iFlow=0, iFlowBatch=0; iFlow <= maxFlow; iFlow++) {
        // copy well data into imgBuffer
        int flowOffset = iFlow * nFrame;
        for(int iWell=0, iImg=0; iWell<nWell; iWell++)
          for(int iFrame=0; iFrame<nFrame; iFrame++, iImg++)
            imgBuffer[iImg] = (short) dat(iWell,flowOffset+iFrame);
        // copy average background data into bkgBufer and save it for return in out_bg
        for(int iFrame=0; iFrame<nFrame; iFrame++) {
          bkgBuffer[iFrame] = (short) bkg(iFrame,iFlow);
          out_bg(iFlow*nFrame+iFrame) = bkg(iFrame,iFlow);
        }
        bool last_flow = (iFlow == maxFlow);
        bkgmodel->ProcessImage(imgBuffer,bkgBuffer,iFlow,last_flow,false);

        // fitted-paramters are only updated after a block of NUMFB
        // flows has been processed or if last_flow == true
        if ((((1+iFlow) % NUMFB) == 0) || last_flow) {
          // Store per-region parameters
          bkgmodel->GetRegParams(&rp);
          for (int iNuc=0; iNuc < NUMNUC; iNuc++) {
            out_krate(iNuc,iFlowBatch) = rp.krate[iNuc];
            out_d(iNuc,iFlowBatch)     = rp.d[iNuc];
            out_sens(iNuc,iFlowBatch)  = rp.sens;
          }
          out_sigma(iFlowBatch)        = rp.nuc_shape.sigma;
          out_C                        = rp.nuc_shape.C;
          out_tshift(iFlowBatch)       = rp.tshift;
          // Store per-well parameters
          for (int iWell=0; iWell < nWell; iWell++) {
            bkgmodel->GetParams(iWell,&p);
            for (int iFlowParam=0, iFlowOut=iFlow-NUMFB+1; iFlowParam < NUMFB; iFlowParam++, iFlowOut++) {
              out_sig(iWell,iFlowOut)    = p.Ampl[iFlowParam]*p.Copies;
            }
            out_R(iWell,iFlowBatch)      = p.R;
            out_dt(iWell,iFlowBatch)     = 0;
            out_Copies(iWell,iFlowBatch)      = p.Copies;
            out_gain(iWell,iFlowBatch)   = p.gain;
            // Simulate data from the fitted model
            float *fg,*bg,*feval,*isig,*pf;
            int tot_frms = bkgmodel->GetModelEvaluation(iWell,&p,&rp,&fg,&bg,&feval,&isig,&pf);
            fitFramesPerFlow = floor(tot_frms/(double)NUMFB);
            for (int iFlowParam=0, iFlowOut=iFlow-NUMFB+1; iFlowParam < NUMFB; iFlowParam++, iFlowOut++) {
              for (int iFrame=0; iFrame < fitFramesPerFlow; iFrame++) {
                out_sim_fg(iWell,iFlowOut*nFrame+iFrame) = feval[iFlowParam*fitFramesPerFlow+iFrame];
                if(iWell==0)
                  out_sim_bg(iFlowOut*nFrame+iFrame) = bg[iFlowParam*fitFramesPerFlow+iFrame];
              }
            }
          }
          iFlowBatch++;
        }
      }
      RcppMatrix<double> out_sim_fg_trimmed(nWell,nFlow*fitFramesPerFlow);
      RcppVector<double> out_sim_bg_trimmed(nFlow*fitFramesPerFlow);
      for (int iWell=0; iWell < nWell; iWell++) {
        for (int iFlow=0; iFlow  < nFlow; iFlow++) {
          for (int iFrame=0; iFrame < fitFramesPerFlow; iFrame++) {
            out_sim_fg_trimmed(iWell,iFlow*fitFramesPerFlow+iFrame) = out_sim_fg(iWell,iFlow*nFrame+iFrame);
            if(iWell==0)
              out_sim_bg_trimmed(iFlow*fitFramesPerFlow+iFrame) = out_sim_bg(iFlow*nFrame+iFrame);
          }
        }
      }
      delete [] imgBuffer;
      delete [] bkgBuffer;
      delete bkgmodel;
      // This is commented out for now - see comments above relating to InitializeBkgModelTables()
      //CleanupBkgModelTables();

      RcppResultSet rs;
      rs.add("sig",        out_sig);
      rs.add("bkg",        out_bg);
      rs.add("nFitFrame",  fitFramesPerFlow);
      rs.add("fitFg",      out_sim_fg_trimmed);
      rs.add("fitBg",      out_sim_bg_trimmed);
      rs.add("nFlowBatch", nFlowBatch);
      rs.add("R",          out_R);
      rs.add("dt",         out_dt);
      rs.add("Copies",     out_Copies);
      rs.add("gain",       out_gain);
      rs.add("tshift",     out_tshift);
      rs.add("krate",      out_krate);
      rs.add("d",          out_d);
      rs.add("sigma",      out_sigma);
      rs.add("sens",       out_sens);
      rs.add("C",          out_C);
      ret = rs.getReturnList();
    }
  } catch(std::exception& ex) {
    exceptionMesg = copyMessageToR(ex.what());
  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
