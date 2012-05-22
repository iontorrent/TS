/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SeparatorInterface.h"


void DoDiffSeparatorFromCLO (DifferentialSeparator *diffSeparator, CommandLineOpts &clo, Mask *maskPtr, string &analysisLocation, SequenceItem *seqList, int numSeqListItems)
{
  DifSepOpt opts;

  opts.bfType = clo.bfd_control.bfType;
  opts.bfDat = clo.bfd_control.bfDat;
  opts.bfBgDat = clo.bfd_control.bfBgDat;
  opts.resultsDir = clo.sys_context.dat_source_directory;
  opts.outData = analysisLocation;
  opts.analysisDir =  analysisLocation;
  opts.ignoreChecksumErrors = clo.img_control.ignoreChecksumErrors;
  opts.noduds = clo.bfd_control.noduds;
  opts.outputDebug = clo.bfd_control.bfOutputDebug;
  opts.minRatioLiveWell = clo.bfd_control.bfMinLiveRatio;
  opts.doRecoverSdFilter = clo.bfd_control.skipBeadfindSdRecover == 0;

  opts.tfFilterQuantile = clo.bfd_control.bfTfFilterQuantile;
  opts.libFilterQuantile = clo.bfd_control.bfLibFilterQuantile;

  opts.flowOrder = clo.flow_context.flowOrder; // 5th duplicated code instance of translating flow order to nucs
  if (!opts.outData.empty())
  {
    string sep = "";
    if (*opts.outData.rbegin() != '/')
    {
      sep = "/";
    }
    opts.outData = opts.outData + sep + "separator";
  }
  else
  {
    opts.outData = "separator";
  }
  opts.mask = maskPtr;
  opts.useSignalReference = clo.bfd_control.useSignalReference;
  cout << "Out Data: " << opts.outData << endl;
  cout << "Analysis location: " << opts.analysisDir << endl;
  diffSeparator->SetKeys (seqList, numSeqListItems, clo.bfd_control.bfMinLiveLibSnr, clo.bfd_control.bfMinLiveTfSnr);
  if (clo.bfd_control.beadfindLagOneFilt > 0)
  {
    opts.filterLagOneSD = true;
  }
  if (clo.bfd_control.beadfindThumbnail == 1)
  {
    opts.t0MeshStep = clo.loc_context.regionXSize;
    opts.bfMeshStep = clo.loc_context.regionXSize;
    opts.tauEEstimateStep = clo.loc_context.regionXSize;
    opts.useMeshNeighbors = 0;
    opts.regionXSize = clo.loc_context.regionXSize;
    opts.regionYSize = clo.loc_context.regionYSize;
  }
  diffSeparator->Run (opts);
}


void SetupForBkgModelTiming (DifferentialSeparator *diffSeparator, std::vector<float> &smooth_t0_est, RegionTiming *region_timing,
                             Region *region_list, int numRegions, ImageSpecClass &my_image_spec, Mask *maskPtr, bool doSmoothing)
{
  // compute timing information
  AvgKeyIncorporation *keyIncorporation = NULL;

  //Create a mask that tracks the pinned pixels discovered in each image
  maskPtr->CalculateLiveNeighbors();

  // Setup t0 estimation from beadfind to pass to background model
  std::vector<float> sep_t0_est;
  sep_t0_est = diffSeparator->GetT0();
  smooth_t0_est = sep_t0_est;
  if (doSmoothing)
  {
    printf ("smoothing t0 estimate from separator.......");
    NNSmoothT0Estimate (maskPtr,my_image_spec.rows,my_image_spec.cols,sep_t0_est,smooth_t0_est);
    printf ("done.\n");
  }
  // do some incorporation signal modeling
  keyIncorporation = diffSeparator;
  //FillRegionalTimingParameters(region_timing, region_list, numRegions, keyIncorporation);
  threadedFillRegionalTimingParameters (region_timing,region_list,numRegions,keyIncorporation);
  keyIncorporation = NULL;
}


void getTausFromSeparator (Mask *maskPtr, DifferentialSeparator *diffSeparator, std::vector<float> &tauB, std::vector<float> &tauE)
{
  const KeyBulkFit *tempKbf;
  std::vector<float> tempTauB;
  std::vector<float> tempTauE;
  tempTauB.reserve (maskPtr->W() * maskPtr->H());
  tempTauE.reserve (maskPtr->W() * maskPtr->H());
  for (size_t ik = 0; ik < (size_t) maskPtr->W() * maskPtr->H(); ++ik)
  {
    tempKbf = diffSeparator->GetBulkFit (ik);
    // get the average here
    float avgTauB = 0;
    float avgTauE = 0;
    if (tempKbf != NULL)
    {
      avgTauB = tempKbf->param.at (TraceStore<double>::A_NUC,0) +tempKbf->param.at (TraceStore<double>::C_NUC,0) +tempKbf->param.at (TraceStore<double>::G_NUC,0) +tempKbf->param.at (TraceStore<double>::T_NUC,0);
      avgTauB /= 4;
      avgTauE = tempKbf->param.at (TraceStore<double>::A_NUC,1) +tempKbf->param.at (TraceStore<double>::C_NUC,1) +tempKbf->param.at (TraceStore<double>::G_NUC,1) +tempKbf->param.at (TraceStore<double>::T_NUC,1);
      avgTauE /= 4;
    }
    //cout << "avgTauB=" << avgTauB << "\t" << "avgTauE=" << avgTauE << endl;
    tempTauB.push_back (avgTauB);
    tempTauE.push_back (avgTauE);
  }
  tauB = tempTauB;
  tauE = tempTauE;
}





// distance to NN-smooth the t_zero (not t_mid_nuc!!!) estimate from the separator
#define SEPARATOR_T0_ESTIMATE_SMOOTH_DIST 15


// Brute-force NN averaging used to smooth the t0 estimate from the separator.  This algorithm can be sped-up
// considerably by sharing summations across multiple data points, similar to what is done in the image class for
// neighbor-subtracting images.
void NNSmoothT0Estimate (Mask *mask,int imgRows,int imgCols,std::vector<float> &sep_t0_est,std::vector<float> &output_t0_est)
{
  for (int r=0;r < imgRows;r++)
  {
    for (int c=0;c < imgCols;c++)
    {
      // OK..we're going to compute the Neighbor-average for the well at (r,c)
      float sum = 0.0;
      int nsum = 0;
      int lr = r-SEPARATOR_T0_ESTIMATE_SMOOTH_DIST;
      int ur = r+SEPARATOR_T0_ESTIMATE_SMOOTH_DIST;
      int lc = c-SEPARATOR_T0_ESTIMATE_SMOOTH_DIST;
      int uc = c+SEPARATOR_T0_ESTIMATE_SMOOTH_DIST;
      lr = (lr < 0?0:lr);
      lc = (lc < 0?0:lc);
      ur = (ur >= imgRows?imgRows-1:ur);
      uc = (uc >= imgCols?imgCols-1:uc);

      for (int sr=lr;sr <= ur;sr++)
        for (int sc=lc;sc <= uc;sc++)
          if (!mask->Match (sc,sr, (MaskType) (MaskPinned | MaskIgnore | MaskExclude)))
          {
            sum += sep_t0_est[sc+sr*imgCols];
            nsum++;
          }

      // if there we're no wells to form and average from, just copy the value from the original
      // un-smoothed vector
      if (nsum > 0)
        output_t0_est[c+r*imgCols] = sum / nsum;
      else
        output_t0_est[c+r*imgCols] = sep_t0_est[c+r*imgCols];
    }
  }
}
