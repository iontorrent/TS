/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <Rcpp.h>
#include "PhaseFitCfIe.h"

RcppExport SEXP phaseFit(SEXP RseqString, SEXP RseqFlow, SEXP Rsig, SEXP RflowCycle, SEXP RnucConc, SEXP Rcf, SEXP Rie, SEXP Rdr, SEXP RhpScale, SEXP RnFlow, SEXP RmaxAdv, SEXP RdroopType, SEXP RmaxIter, SEXP RfitType, SEXP RignoreHPs, SEXP RflowWeight, SEXP RresType, SEXP RresSummary, SEXP RmaxErr, SEXP RextraTaps)
{
  SEXP ret = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    string flowCycle       = Rcpp::as<string>(RflowCycle);
    unsigned int nFlow     = (unsigned int) Rcpp::as<int>(RnFlow);
    unsigned int extraTaps = (unsigned int) Rcpp::as<int>(RextraTaps);
    unsigned int maxAdv    = (unsigned int) Rcpp::as<int>(RmaxAdv);
    int maxIter            = Rcpp::as<int>(RmaxIter);
    double maxErr          = Rcpp::as<double>(RmaxErr);
    string drType          = Rcpp::as<string>(RdroopType);
    string resType         = Rcpp::as<string>(RresType);
    string resSummary      = Rcpp::as<string>(RresSummary);
    string fitType         = Rcpp::as<string>(RfitType);
    bool ignoreHPs         = Rcpp::as<bool>(RignoreHPs);
    Rcpp::StringVector  seqString(RseqString);
    Rcpp::NumericMatrix seqFlow(RseqFlow);
    Rcpp::NumericMatrix sig(Rsig);
    Rcpp::NumericMatrix cc(RnucConc);
    Rcpp::NumericVector cf(Rcf);
    Rcpp::NumericVector ie(Rie);
    Rcpp::NumericVector dr(Rdr);
    Rcpp::NumericVector hpScale(RhpScale);
    Rcpp::NumericVector flowWeight(RflowWeight);

    // This block determines if the sequence has been specified by a vector of strings
    // or by a matrix of integer flow values with one row per read
    int nSeqFlow   = seqFlow.rows();
    int nSeqString = seqString.size();
    // annoyingly the RcppSringVector constructor doesn't handle a zero-length vector, so we
    // currently 'encode' that by a string vector of length 1 with just an empty sequence
    if((nSeqString == 1) && (seqString(0)==""))
      nSeqString = 0;
    int nSeq=0;
    string seqSource = "";
    if((nSeqString == 0) && (nSeqFlow == 0)) {
      std::string exception = "both seqString and seqFlow specify no sequences\n";
      exceptionMesg = strdup(exception.c_str());
    } else if((nSeqString > 0) && (nSeqFlow > 0)) {
      std::string exception = "both seqString and seqFlow specify sequences - use only one\n";
      exceptionMesg = strdup(exception.c_str());
    } else if(nSeqString > 0) {
      nSeq = nSeqString;
      seqSource = "String";
    } else {
      nSeq = nSeqFlow;
      seqSource = "Flow";
    }
    
    // Ensure we have a valid droop type
    DroopType droopType = ONLY_WHEN_INCORPORATING;
    bool badDroopType = false;
    if(drType == "ONLY_WHEN_INCORPORATING") {
      droopType = ONLY_WHEN_INCORPORATING;
    } else if(drType == "EVERY_FLOW") {
      droopType = EVERY_FLOW;
    } else {
      badDroopType = true;
    }
  
    // Ensure we have a valid residual type
    ResidualType residualType = SQUARED;
    bool badResidualType = false;
    if(resType == "SQUARED") {
      residualType = SQUARED;
    } else if(resType == "ABSOLUTE") {
      residualType = ABSOLUTE;
    } else if(resType == "GEMAN_MCCLURE") {
      residualType = GEMAN_MCCLURE;
    } else {
      badResidualType = true;
    }
  
    // Ensure we have a valid residual summary
    ResidualSummary residualSummary = MEAN;
    bool badResidualSummary = false;
    if(resSummary == "MEAN") {
      residualSummary = MEAN;
    } else if(resSummary == "MEDIAN") {
      residualSummary = MEDIAN;
    } else if(resSummary == "MEAN_OF_MEDIAN") {
      residualSummary = MEAN_OF_MEDIAN;
    } else if(resSummary == "MEDIAN_OF_MEAN") {
      residualSummary = MEDIAN_OF_MEAN;
    } else {
      badResidualSummary = true;
    }
  
    if(badDroopType) {
      std::string exception = "bad droop type supplied\n";
      exceptionMesg = strdup(exception.c_str());
    } else if(badResidualType) {
      std::string exception = "bad residual type supplied\n";
      exceptionMesg = strdup(exception.c_str());
    } else if(badResidualSummary) {
      std::string exception = "bad residual summary supplied\n";
      exceptionMesg = strdup(exception.c_str());
    } else if(sig.rows() != nSeq) {
      std::string exception = "number of entries in signal vector should be equal to the number of sequences\n";
    } else if(sig.cols() != (int) nFlow) {
      std::string exception = "number of cols in signal matrix should be equal to number of flows\n";
    } else if(cc.rows() != (int) N_NUCLEOTIDES) {
      std::string exception = "concentration matrix should have 4 rows\n";
      exceptionMesg = strdup(exception.c_str());
    } else if(cc.cols() != (int) N_NUCLEOTIDES) {
      std::string exception = "concentration matrix should have 4 columns\n";
      exceptionMesg = strdup(exception.c_str());
    } else {

      // Start with typecasts of some of the inputs

      // recast seqString
      vector<string> seqStringMod(seqString.size());
      for(int i=0; i<seqString.size(); i++)
        seqStringMod[i] = seqString(i);

      // recast cf
      weight_vec_t cfMod(cf.size());
      for(int i=0; i<cf.size(); i++)
        cfMod[i] = cf(i);

      // recast ie
      weight_vec_t ieMod(ie.size());
      for(int i=0; i<ie.size(); i++)
        ieMod[i] = ie(i);

      // recast dr
      weight_vec_t drMod(dr.size());
      for(int i=0; i<dr.size(); i++)
        drMod[i] = dr(i);

      // recast hpScale
      weight_vec_t hpScaleMod(hpScale.size());
      for(int i=0; i<hpScale.size(); i++)
        hpScaleMod[i] = hpScale(i);

      // recast flowWeight
      weight_vec_t flowWeightMod(flowWeight.size());
      for(int i=0; i<flowWeight.size(); i++)
        flowWeightMod[i] = flowWeight(i);
      
      // recast nuc concentration
      vector<weight_vec_t> ccMod(cc.rows());
      for(int iRow=0; iRow < cc.rows(); iRow++) {
        ccMod[iRow].resize(cc.cols());
        for(unsigned int iCol=0; iCol < N_NUCLEOTIDES; iCol++)
          ccMod[iRow][iCol] = cc(iRow,iCol);
      }

      // recast signal
      vector<weight_vec_t> sigMod(sig.rows());
      for(int iRead=0; iRead < sig.rows(); iRead++) {
        sigMod[iRead].resize(sig.cols());
        for(int iFlow=0; iFlow < sig.cols(); iFlow++)
          sigMod[iRead][iFlow] = sig(iRead,iFlow);
      }

      // recast seqFlow
      vector<weight_vec_t> seqFlowMod(seqFlow.rows());
      for(int iRead=0; iRead < seqFlow.rows(); iRead++) {
        seqFlowMod[iRead].resize(seqFlow.cols());
        for(int iFlow=0; iFlow < seqFlow.cols(); iFlow++)
          seqFlowMod[iRead][iFlow] = seqFlow(iRead,iFlow);
      }

      vector<weight_vec_t> residualRaw,residualWeighted;
      float residualSummarized = 0;
      int nIter = 0;
      std::map<std::string,SEXP> map ;
      if(fitType == "CfIe") {
        PhaseFitCfIe pFit;
        pFit.SetExtraTaps(extraTaps);
        pFit.InitializeFlowSpecs(flowCycle,ccMod,cfMod,ieMod,drMod,nFlow,droopType,maxAdv);
        for(int iRead=0; iRead < nSeq; iRead++) {
          if(seqSource == "String")
            pFit.AddRead(seqStringMod[iRead],sigMod[iRead]);
          else
            pFit.AddRead(seqFlowMod[iRead],sigMod[iRead],maxErr);
        }

        // Set initial parameter estimates
        CfIeParam param_in;
        param_in.cf = (float) cfMod[0];
        param_in.ie = (float) ieMod[0];
        pFit.SetParam(param_in);

        // Set min parameter estimates
        CfIeParam param_min;
        param_min.cf = (float) 0;
        param_min.ie = (float) 0;
        pFit.SetParamMin(param_min);

        // Set max parameter estimates
        CfIeParam param_max;
        param_max.cf = (float) 0.5;
        param_max.ie = (float) 0.5;
        pFit.SetParamMax(param_max);

        // Do the fit
        pFit.SetResidualType(residualType);
        pFit.SetResidualSummary(residualSummary);
        pFit.SetIgnoreHPs(ignoreHPs);
        pFit.SetFlowWeight(flowWeightMod);
        nIter = pFit.LevMarFit(maxIter);
        residualSummarized = pFit.GetSummarizedResidual();
        residualRaw = pFit.GetRawResiduals();
        residualWeighted = pFit.GetWeightedResiduals();

        // retrieve parameter estimates
        CfIeParam param_out = pFit.GetParam();
        map["param.cf"] = Rcpp::wrap( param_out.cf );
        map["param.ie"] = Rcpp::wrap( param_out.ie );
      } else if (fitType == "CfIeDr") {
        PhaseFitCfIeDr pFit;
        pFit.SetExtraTaps(extraTaps);
        pFit.InitializeFlowSpecs(flowCycle,ccMod,cfMod,ieMod,drMod,nFlow,droopType,maxAdv);
        for(int iRead=0; iRead < nSeq; iRead++) {
          if(seqSource == "String")
            pFit.AddRead(seqStringMod[iRead],sigMod[iRead]);
          else
            pFit.AddRead(seqFlowMod[iRead],sigMod[iRead],maxErr);
        }

        // Set initial parameter estimates
        CfIeDrParam param_in;
        param_in.cf = (float) cfMod[0];
        param_in.ie = (float) ieMod[0];
        param_in.dr = (float) drMod[0];
        pFit.SetParam(param_in);

        // Set min parameter estimates
        CfIeDrParam param_min;
        param_min.cf = (float) 0;
        param_min.ie = (float) 0;
        param_min.dr = (float) 0;
        pFit.SetParamMin(param_min);

        // Set max parameter estimates
        CfIeDrParam param_max;
        param_max.cf = (float) 0.5;
        param_max.ie = (float) 0.5;
        param_max.dr = (float) 0.5;
        pFit.SetParamMax(param_max);

        // Do the fit
        pFit.SetResidualType(residualType);
        pFit.SetResidualSummary(residualSummary);
        pFit.SetIgnoreHPs(ignoreHPs);
        pFit.SetFlowWeight(flowWeightMod);
        nIter = pFit.LevMarFit(maxIter);
        residualSummarized = pFit.GetSummarizedResidual();
        residualRaw = pFit.GetRawResiduals();
        residualWeighted = pFit.GetWeightedResiduals();

        // retrieve parameter estimates
        CfIeDrParam param_out = pFit.GetParam();
        map["param.cf"] = Rcpp::wrap( param_out.cf );
        map["param.ie"] = Rcpp::wrap( param_out.ie );
        map["param.dr"] = Rcpp::wrap( param_out.dr );
      } else if (fitType == "CfIeDrHpScale") {
        PhaseFitCfIeDrHpScale pFit;
        pFit.SetExtraTaps(extraTaps);
        pFit.InitializeFlowSpecs(flowCycle,ccMod,cfMod,ieMod,drMod,nFlow,droopType,maxAdv);
        for(int iRead=0; iRead < nSeq; iRead++) {
          if(seqSource == "String")
            pFit.AddRead(seqStringMod[iRead],sigMod[iRead]);
          else
            pFit.AddRead(seqFlowMod[iRead],sigMod[iRead],maxErr);
        }

        // Set initial parameter estimates
        CfIeDrHpScaleParam param_in;
        param_in.cf      = (float) cfMod[0];
        param_in.ie      = (float) ieMod[0];
        param_in.dr      = (float) drMod[0];
        param_in.hpScale = (float) hpScaleMod[0];
        pFit.SetParam(param_in);

        // Set min parameter estimates
        CfIeDrHpScaleParam param_min;
        param_min.cf      = (float) 0;
        param_min.ie      = (float) 0;
        param_min.dr      = (float) 0;
        param_min.hpScale = (float) 0.5;
        pFit.SetParamMin(param_min);

        // Set max parameter estimates
        CfIeDrHpScaleParam param_max;
        param_max.cf      = (float) 0.5;
        param_max.ie      = (float) 0.5;
        param_max.dr      = (float) 0.5;
        param_max.hpScale = (float) 1.5;
        pFit.SetParamMax(param_max);

        // Do the fit
        pFit.SetResidualType(residualType);
        pFit.SetResidualSummary(residualSummary);
        pFit.SetIgnoreHPs(ignoreHPs);
        pFit.SetFlowWeight(flowWeightMod);
        nIter = pFit.LevMarFit(maxIter);
        residualSummarized = pFit.GetSummarizedResidual();
        residualRaw = pFit.GetRawResiduals();
        residualWeighted = pFit.GetWeightedResiduals();

        // retrieve parameter estimates
        CfIeDrHpScaleParam param_out = pFit.GetParam();
        map["param.cf"]      = Rcpp::wrap( param_out.cf );
        map["param.ie"]      = Rcpp::wrap( param_out.ie );
        map["param.dr"]      = Rcpp::wrap( param_out.dr );
        map["param.hpScale"] = Rcpp::wrap( param_out.hpScale );
      } else if (fitType == "HpScale") {
        PhaseFitHpScale pFit;
        pFit.SetExtraTaps(extraTaps);
        pFit.InitializeFlowSpecs(flowCycle,ccMod,cfMod,ieMod,drMod,nFlow,droopType,maxAdv);
        for(int iRead=0; iRead < nSeq; iRead++) {
          if(seqSource == "String")
            pFit.AddRead(seqStringMod[iRead],sigMod[iRead]);
          else
            pFit.AddRead(seqFlowMod[iRead],sigMod[iRead],maxErr);
        }

        // Set initial parameter estimates
        HpScaleParam param_in;
        param_in.hpScale = (float) hpScaleMod[0];
        pFit.SetParam(param_in);

        // Set min parameter estimates
        HpScaleParam param_min;
        param_min.hpScale = (float) 0.5;
        pFit.SetParamMin(param_min);

        // Set max parameter estimates
        HpScaleParam param_max;
        param_max.hpScale = (float) 1.5;
        pFit.SetParamMax(param_max);

        // Do the fit
        pFit.SetResidualType(residualType);
        pFit.SetResidualSummary(residualSummary);
        pFit.SetIgnoreHPs(ignoreHPs);
        pFit.SetFlowWeight(flowWeightMod);
        nIter = pFit.LevMarFit(maxIter);
        residualSummarized = pFit.GetSummarizedResidual();
        residualRaw = pFit.GetRawResiduals();
        residualWeighted = pFit.GetWeightedResiduals();

        // retrieve parameter estimates
        HpScaleParam param_out = pFit.GetParam();
        map["param.hpScale"]      = Rcpp::wrap( param_out.hpScale );
      } else if (fitType == "CfIeDrHpScale4") {
        if(hpScaleMod.size()==1) {
          hpScaleMod.assign(4,hpScaleMod.front());
        }
        PhaseFitCfIeDrHpScale4 pFit;
        pFit.SetExtraTaps(extraTaps);
        pFit.InitializeFlowSpecs(flowCycle,ccMod,cfMod,ieMod,drMod,nFlow,droopType,maxAdv);
        for(int iRead=0; iRead < nSeq; iRead++) {
          if(seqSource == "String")
            pFit.AddRead(seqStringMod[iRead],sigMod[iRead]);
          else
            pFit.AddRead(seqFlowMod[iRead],sigMod[iRead],maxErr);
        }

        // Set initial parameter estimates
        CfIeDrHpScale4Param param_in;
        param_in.cf       = (float) cfMod[0];
        param_in.ie       = (float) ieMod[0];
        param_in.dr       = (float) drMod[0];
        param_in.hpScaleA = (float) hpScaleMod[0];
        param_in.hpScaleC = (float) hpScaleMod[1];
        param_in.hpScaleG = (float) hpScaleMod[2];
        param_in.hpScaleT = (float) hpScaleMod[3];
        pFit.SetParam(param_in);

        // Set min parameter estimates
        CfIeDrHpScale4Param param_min;
        param_min.cf       = (float) 0;
        param_min.ie       = (float) 0;
        param_min.dr       = (float) 0;
        param_min.hpScaleA = (float) 0.5;
        param_min.hpScaleC = (float) 0.5;
        param_min.hpScaleG = (float) 0.5;
        param_min.hpScaleT = (float) 0.5;
        pFit.SetParamMin(param_min);

        // Set max parameter estimates
        CfIeDrHpScale4Param param_max;
        param_max.cf       = (float) 0.5;
        param_max.ie       = (float) 0.5;
        param_max.dr       = (float) 0.5;
        param_max.hpScaleA = (float) 1.5;
        param_max.hpScaleC = (float) 1.5;
        param_max.hpScaleG = (float) 1.5;
        param_max.hpScaleT = (float) 1.5;
        pFit.SetParamMax(param_max);

        // Do the fit
        pFit.SetResidualType(residualType);
        pFit.SetResidualSummary(residualSummary);
        pFit.SetIgnoreHPs(ignoreHPs);
        pFit.SetFlowWeight(flowWeightMod);
        nIter = pFit.LevMarFit(maxIter);
        residualSummarized = pFit.GetSummarizedResidual();
        residualRaw = pFit.GetRawResiduals();
        residualWeighted = pFit.GetWeightedResiduals();

        // retrieve parameter estimates
        CfIeDrHpScale4Param param_out = pFit.GetParam();
        map["param.cf"]      = Rcpp::wrap( param_out.cf );
        map["param.ie"]      = Rcpp::wrap( param_out.ie );
        map["param.dr"]      = Rcpp::wrap( param_out.dr );
        Rcpp::NumericVector out_hpScale(N_NUCLEOTIDES);
        out_hpScale(0) = param_out.hpScaleA;
        out_hpScale(1) = param_out.hpScaleC;
        out_hpScale(2) = param_out.hpScaleG;
        out_hpScale(3) = param_out.hpScaleT;
        map["param.hpScale"] = Rcpp::wrap( out_hpScale );
      } else if (fitType == "HpScale4") {
        if(hpScaleMod.size()==1) {
          hpScaleMod.assign(4,hpScaleMod.front());
        }
        PhaseFitHpScale4 pFit;
        pFit.SetExtraTaps(extraTaps);
        pFit.InitializeFlowSpecs(flowCycle,ccMod,cfMod,ieMod,drMod,nFlow,droopType,maxAdv);
        for(int iRead=0; iRead < nSeq; iRead++) {
          if(seqSource == "String")
            pFit.AddRead(seqStringMod[iRead],sigMod[iRead]);
          else
            pFit.AddRead(seqFlowMod[iRead],sigMod[iRead],maxErr);
        }

        // Set initial parameter estimates
        HpScale4Param param_in;
        param_in.hpScaleA = (float) hpScaleMod[0];
        param_in.hpScaleC = (float) hpScaleMod[1];
        param_in.hpScaleG = (float) hpScaleMod[2];
        param_in.hpScaleT = (float) hpScaleMod[3];
        pFit.SetParam(param_in);

        // Set min parameter estimates
        HpScale4Param param_min;
        param_min.hpScaleA = (float) 0.5;
        param_min.hpScaleC = (float) 0.5;
        param_min.hpScaleG = (float) 0.5;
        param_min.hpScaleT = (float) 0.5;
        pFit.SetParamMin(param_min);

        // Set max parameter estimates
        HpScale4Param param_max;
        param_max.hpScaleA = (float) 1.5;
        param_max.hpScaleC = (float) 1.5;
        param_max.hpScaleG = (float) 1.5;
        param_max.hpScaleT = (float) 1.5;
        pFit.SetParamMax(param_max);

        // Do the fit
        pFit.SetResidualType(residualType);
        pFit.SetResidualSummary(residualSummary);
        pFit.SetIgnoreHPs(ignoreHPs);
        pFit.SetFlowWeight(flowWeightMod);
        nIter = pFit.LevMarFit(maxIter);
        residualSummarized = pFit.GetSummarizedResidual();
        residualRaw = pFit.GetRawResiduals();
        residualWeighted = pFit.GetWeightedResiduals();

        // retrieve parameter estimates
        HpScale4Param param_out = pFit.GetParam();
        Rcpp::NumericVector out_hpScale(N_NUCLEOTIDES);
        out_hpScale(0) = param_out.hpScaleA;
        out_hpScale(1) = param_out.hpScaleC;
        out_hpScale(2) = param_out.hpScaleG;
        out_hpScale(3) = param_out.hpScaleT;
        map["param.hpScale"] = Rcpp::wrap( out_hpScale );
      } else if (fitType == "NucContam") {
        PhaseFitNucContam pFit;
        pFit.SetExtraTaps(extraTaps);
        pFit.InitializeFlowSpecs(flowCycle,ccMod,cfMod,ieMod,drMod,nFlow,droopType,maxAdv);
        for(int iRead=0; iRead < nSeq; iRead++) {
          if(seqSource == "String")
            pFit.AddRead(seqStringMod[iRead],sigMod[iRead]);
          else
            pFit.AddRead(seqFlowMod[iRead],sigMod[iRead],maxErr);
        }

        // Set initial parameter estimates
        NucContamParam param_in;
        param_in.C_in_A = (float) ccMod[0][1];
        param_in.G_in_A = (float) ccMod[0][2];
        param_in.T_in_A = (float) ccMod[0][3];
        param_in.A_in_C = (float) ccMod[1][0];
        param_in.G_in_C = (float) ccMod[1][2];
        param_in.T_in_C = (float) ccMod[1][3];
        param_in.A_in_G = (float) ccMod[2][0];
        param_in.C_in_G = (float) ccMod[2][1];
        param_in.T_in_G = (float) ccMod[2][3];
        param_in.A_in_T = (float) ccMod[3][0];
        param_in.C_in_T = (float) ccMod[3][1];
        param_in.G_in_T = (float) ccMod[3][2];
        pFit.SetParam(param_in);

        // Set min parameter estimates
        NucContamParam param_min;
        param_min.C_in_A = 0;
        param_min.G_in_A = 0;
        param_min.T_in_A = 0;
        param_min.A_in_C = 0;
        param_min.G_in_C = 0;
        param_min.T_in_C = 0;
        param_min.A_in_G = 0;
        param_min.C_in_G = 0;
        param_min.T_in_G = 0;
        param_min.A_in_T = 0;
        param_min.C_in_T = 0;
        param_min.G_in_T = 0;
        pFit.SetParamMin(param_min);

        // Set max parameter estimates
        NucContamParam param_max;
        param_max.C_in_A = 0.5;
        param_max.G_in_A = 0.5;
        param_max.T_in_A = 0.5;
        param_max.A_in_C = 0.5;
        param_max.G_in_C = 0.5;
        param_max.T_in_C = 0.5;
        param_max.A_in_G = 0.5;
        param_max.C_in_G = 0.5;
        param_max.T_in_G = 0.5;
        param_max.A_in_T = 0.5;
        param_max.C_in_T = 0.5;
        param_max.G_in_T = 0.5;
        pFit.SetParamMax(param_max);

        // Do the fit
        pFit.SetResidualType(residualType);
        pFit.SetResidualSummary(residualSummary);
        pFit.SetIgnoreHPs(ignoreHPs);
        pFit.SetFlowWeight(flowWeightMod);
        nIter = pFit.LevMarFit(maxIter);
        residualSummarized = pFit.GetSummarizedResidual();
        residualRaw = pFit.GetRawResiduals();
        residualWeighted = pFit.GetWeightedResiduals();

        // retrieve parameter estimates
        NucContamParam param_out = pFit.GetParam();
        Rcpp::NumericMatrix conc_out(N_NUCLEOTIDES,N_NUCLEOTIDES);
        for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++)
          conc_out(iNuc,iNuc) = 1;
        conc_out(0,1) = param_out.C_in_A;
        conc_out(0,2) = param_out.G_in_A;
        conc_out(0,3) = param_out.T_in_A;
        conc_out(1,0) = param_out.A_in_C;
        conc_out(1,2) = param_out.G_in_C;
        conc_out(1,3) = param_out.T_in_C;
        conc_out(2,0) = param_out.A_in_G;
        conc_out(2,1) = param_out.C_in_G;
        conc_out(2,3) = param_out.T_in_G;
        conc_out(3,0) = param_out.A_in_T;
        conc_out(3,1) = param_out.C_in_T;
        conc_out(3,2) = param_out.G_in_T;
        map["param.conc"] = Rcpp::wrap( conc_out );
      } else if (fitType == "NucContamIe") {
        PhaseFitNucContamIe pFit;
        pFit.SetExtraTaps(extraTaps);
        pFit.InitializeFlowSpecs(flowCycle,ccMod,cfMod,ieMod,drMod,nFlow,droopType,maxAdv);
        for(int iRead=0; iRead < nSeq; iRead++) {
          if(seqSource == "String")
            pFit.AddRead(seqStringMod[iRead],sigMod[iRead]);
          else
            pFit.AddRead(seqFlowMod[iRead],sigMod[iRead],maxErr);
        }

        // Set initial parameter estimates
        NucContamIeParam param_in;
        param_in.ie     = (float) ieMod[0];
        param_in.C_in_A = (float) ccMod[0][1];
        param_in.G_in_A = (float) ccMod[0][2];
        param_in.T_in_A = (float) ccMod[0][3];
        param_in.A_in_C = (float) ccMod[1][0];
        param_in.G_in_C = (float) ccMod[1][2];
        param_in.T_in_C = (float) ccMod[1][3];
        param_in.A_in_G = (float) ccMod[2][0];
        param_in.C_in_G = (float) ccMod[2][1];
        param_in.T_in_G = (float) ccMod[2][3];
        param_in.A_in_T = (float) ccMod[3][0];
        param_in.C_in_T = (float) ccMod[3][1];
        param_in.G_in_T = (float) ccMod[3][2];
        pFit.SetParam(param_in);

        // Set min parameter estimates
        NucContamIeParam param_min;
        param_min.ie     = 0;
        param_min.C_in_A = 0;
        param_min.G_in_A = 0;
        param_min.T_in_A = 0;
        param_min.A_in_C = 0;
        param_min.G_in_C = 0;
        param_min.T_in_C = 0;
        param_min.A_in_G = 0;
        param_min.C_in_G = 0;
        param_min.T_in_G = 0;
        param_min.A_in_T = 0;
        param_min.C_in_T = 0;
        param_min.G_in_T = 0;
        pFit.SetParamMin(param_min);

        // Set max parameter estimates
        NucContamIeParam param_max;
        param_max.ie     = 0.5;
        param_max.C_in_A = 0.5;
        param_max.G_in_A = 0.5;
        param_max.T_in_A = 0.5;
        param_max.A_in_C = 0.5;
        param_max.G_in_C = 0.5;
        param_max.T_in_C = 0.5;
        param_max.A_in_G = 0.5;
        param_max.C_in_G = 0.5;
        param_max.T_in_G = 0.5;
        param_max.A_in_T = 0.5;
        param_max.C_in_T = 0.5;
        param_max.G_in_T = 0.5;
        pFit.SetParamMax(param_max);

        // Do the fit
        pFit.SetResidualType(residualType);
        pFit.SetResidualSummary(residualSummary);
        pFit.SetIgnoreHPs(ignoreHPs);
        pFit.SetFlowWeight(flowWeightMod);
        nIter = pFit.LevMarFit(maxIter);
        residualSummarized = pFit.GetSummarizedResidual();
        residualRaw = pFit.GetRawResiduals();
        residualWeighted = pFit.GetWeightedResiduals();

        // retrieve parameter estimates
        NucContamIeParam param_out = pFit.GetParam();
        double ie_out = param_out.ie;
        Rcpp::NumericMatrix conc_out(N_NUCLEOTIDES,N_NUCLEOTIDES);
        for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++)
          conc_out(iNuc,iNuc) = 1;
        conc_out(0,1) = param_out.C_in_A;
        conc_out(0,2) = param_out.G_in_A;
        conc_out(0,3) = param_out.T_in_A;
        conc_out(1,0) = param_out.A_in_C;
        conc_out(1,2) = param_out.G_in_C;
        conc_out(1,3) = param_out.T_in_C;
        conc_out(2,0) = param_out.A_in_G;
        conc_out(2,1) = param_out.C_in_G;
        conc_out(2,3) = param_out.T_in_G;
        conc_out(3,0) = param_out.A_in_T;
        conc_out(3,1) = param_out.C_in_T;
        conc_out(3,2) = param_out.G_in_T;
        map["param.ie"]   = Rcpp::wrap( ie_out );
        map["param.conc"] = Rcpp::wrap( conc_out );
      } else if (fitType == "CfIe4") {
        PhaseFitCfIe4 pFit;
        pFit.SetExtraTaps(extraTaps);
        pFit.InitializeFlowSpecs(flowCycle,ccMod,cfMod,ieMod,drMod,nFlow,droopType,maxAdv);
        for(int iRead=0; iRead < nSeq; iRead++) {
          if(seqSource == "String")
            pFit.AddRead(seqStringMod[iRead],sigMod[iRead]);
          else
            pFit.AddRead(seqFlowMod[iRead],sigMod[iRead],maxErr);
        }

        // Set initial parameters
        CfIe4Param param_in;
        param_in.cf  = (float) cfMod[0];
        param_in.ieA = (float) ieMod[0];
        if(ieMod.size() == N_NUCLEOTIDES) {
          param_in.ieC = (float) ieMod[1];
          param_in.ieG = (float) ieMod[2];
          param_in.ieT = (float) ieMod[3];
        } else {
          param_in.ieC = (float) ieMod[0];
          param_in.ieG = (float) ieMod[0];
          param_in.ieT = (float) ieMod[0];
        }
        pFit.SetParam(param_in);

        // Set min parameters
        CfIe4Param param_min;
        param_min.cf  = 0;
        param_min.ieA = 0;
        param_min.ieC = 0;
        param_min.ieG = 0;
        param_min.ieT = 0;
        pFit.SetParamMin(param_min);

        // Set max parameters
        CfIe4Param param_max;
        param_max.cf  = 0.5;
        param_max.ieA = 0.5;
        param_max.ieC = 0.5;
        param_max.ieG = 0.5;
        param_max.ieT = 0.5;
        pFit.SetParamMax(param_max);

        // Do the fit
        pFit.SetResidualType(residualType);
        pFit.SetResidualSummary(residualSummary);
        pFit.SetIgnoreHPs(ignoreHPs);
        pFit.SetFlowWeight(flowWeightMod);
        nIter = pFit.LevMarFit(maxIter);
        residualSummarized = pFit.GetSummarizedResidual();
        residualRaw = pFit.GetRawResiduals();
        residualWeighted = pFit.GetWeightedResiduals();

        // Retrieve parameter estimates
        CfIe4Param param_out = pFit.GetParam();
        map["param.cf"] = Rcpp::wrap( param_out.cf );
        Rcpp::NumericVector ie_out(N_NUCLEOTIDES);
        ie_out(0) = param_out.ieA;
        ie_out(1) = param_out.ieC;
        ie_out(2) = param_out.ieG;
        ie_out(3) = param_out.ieT;
        map["param.ie"] = Rcpp::wrap( ie_out );
      } else {
        std::string exception = "bad droop type supplied\n";
        exceptionMesg = strdup(exception.c_str());
      }

      // Add some other outputs to the returned data
      map["nIter"] = Rcpp::wrap( nIter );
      map["residualSummarized"] = Rcpp::wrap( residualSummarized );
      Rcpp::NumericMatrix residualRaw_out(nSeq,nFlow);
      Rcpp::NumericMatrix residualWeighted_out(nSeq,nFlow);
      for(int iRead=0; iRead < nSeq; iRead++) {
        for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
          residualRaw_out(iRead,iFlow)      = (double) residualRaw[iRead][iFlow];
          residualWeighted_out(iRead,iFlow) = (double) residualWeighted[iRead][iFlow];
        }
      }
      map["residualRaw"] = Rcpp::wrap( residualRaw_out );
      map["residualWeighted"] = Rcpp::wrap( residualWeighted_out );

      ret = Rcpp::wrap( map ) ;
    }
  } catch(std::exception& ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
