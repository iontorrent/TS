/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SeparatorInterface.h"


void DoDiffSeparatorFromCLO (DifferentialSeparator *diffSeparator, CommandLineOpts &inception_state, Mask *maskPtr, string &analysisLocation, SequenceItem *seqList, int numSeqListItems)
{
  DifSepOpt opts;

  opts.bfType = inception_state.bfd_control.bfType;
  opts.bfDat = inception_state.bfd_control.bfDat;
  opts.bfBgDat = inception_state.bfd_control.bfBgDat;
  opts.resultsDir = inception_state.sys_context.dat_source_directory;
  opts.outData = analysisLocation;
  opts.analysisDir =  analysisLocation;
  opts.ignoreChecksumErrors = inception_state.img_control.ignoreChecksumErrors;
  opts.noduds = inception_state.bfd_control.noduds;
  opts.outputDebug = inception_state.bfd_control.bfOutputDebug;
  opts.minRatioLiveWell = inception_state.bfd_control.bfMinLiveRatio;
  opts.doRecoverSdFilter = inception_state.bfd_control.skipBeadfindSdRecover == 0;
  opts.nCores = inception_state.bfd_control.numThreads;
  opts.useSeparatorRef = inception_state.bfd_control.beadfindUseSepRef == 1;
  opts.tfFilterQuantile = inception_state.bfd_control.bfTfFilterQuantile;
  opts.libFilterQuantile = inception_state.bfd_control.bfLibFilterQuantile;
  opts.doSdat = inception_state.img_control.doSdat;
  opts.minTfPeakMax = inception_state.bfd_control.minTfPeakMax;
  opts.minLibPeakMax = inception_state.bfd_control.minLibPeakMax;
  opts.sdAsBf = inception_state.bfd_control.sdAsBf;
  opts.bfMult = inception_state.bfd_control.bfMult;
  opts.flowOrder = inception_state.flow_context.flowOrder; // 5th duplicated code instance of translating flow order to nucs
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
  cout << "Out Data: " << opts.outData << endl;
  cout << "Analysis location: " << opts.analysisDir << endl;
  diffSeparator->SetKeys (seqList, numSeqListItems, inception_state.bfd_control.bfMinLiveLibSnr, inception_state.bfd_control.bfMinLiveTfSnr);
  if (inception_state.bfd_control.beadfindLagOneFilt > 0)
  {
    opts.filterLagOneSD = true;
  }
  if (inception_state.bfd_control.beadfindThumbnail == 1)
  {
    opts.t0MeshStep = 50; // inception_state.loc_context.regionXSize;
    opts.bfMeshStep = 50; // inception_state.loc_context.regionXSize;
    opts.clusterMeshStep = 100;
    opts.tauEEstimateStep = inception_state.loc_context.regionXSize;
    opts.useMeshNeighbors = 0;
    opts.regionXSize = 50; //inception_state.loc_context.regionXSize;
    opts.regionYSize = 50; //inception_state.loc_context.regionYSize;
  }
  diffSeparator->Run (opts);
}


void SetupForBkgModelTiming (DifferentialSeparator *diffSeparator, std::vector<float> &smooth_t0_est, std::vector<RegionTiming> &region_timing, std::vector<Region>& region_list, ImageSpecClass &my_image_spec, Mask *maskPtr, bool doSmoothing, int numThreads)
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
  threadedFillRegionalTimingParameters (region_timing,region_list,keyIncorporation,numThreads);
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

void IsolatedBeadFind (
  SlicedPrequel &my_prequel_setup,
  ImageSpecClass &my_image_spec,
  Region &wholeChip,
  CommandLineOpts &inception_state,
  char *results_folder, string &analysisLocation,
  SeqListClass &my_keys,
  TrackProgress &my_progress )
{
  /*********************************************************************
  // Beadfind Section
   *********************************************************************/
  Mask bfmask ( 1, 1 );
  Mask *maskPtr = &bfmask;
  // beadfind has responsibility for defining the exclusion mask
  SetExcludeMask ( inception_state.loc_context,maskPtr,ChipIdDecoder::GetChipType(),my_image_spec.rows,my_image_spec.cols );
  bool beadfind_done = false;
  if ( inception_state.mod_control.reusePriorBeadfind )
  {
    // do not do any bead find
    fprintf ( stdout, "No new Beadfind, using prior beadfind data\n" );

    beadfind_done = true;
  }

  if ( ( !beadfind_done ) & ( inception_state.bfd_control.beadfindType == "differential" ) )
  {
    DifferentialSeparator *diffSeparator=NULL;

    diffSeparator = new DifferentialSeparator();
    DoDiffSeparatorFromCLO ( diffSeparator, inception_state, maskPtr, analysisLocation, my_keys.seqList,my_keys.numSeqListItems );
    // now actually set up the mask I want
    maskPtr->Copy ( diffSeparator->GetMask() );
    
    SetupForBkgModelTiming ( diffSeparator, my_prequel_setup.smooth_t0_est, my_prequel_setup.region_timing, my_prequel_setup.region_list, my_image_spec, maskPtr, inception_state.bfd_control.beadfindThumbnail == 0, inception_state.bfd_control.numThreads);
    // Get the average of tauB and tauE for wells
    // Add getTausFromSeparator here

    my_prequel_setup.tauB.reserve(maskPtr->W() * maskPtr->H());
    my_prequel_setup.tauE.reserve(maskPtr->W() * maskPtr->H());
    if ( inception_state.mod_control.passTau )
      getTausFromSeparator ( maskPtr,diffSeparator,my_prequel_setup.tauB,my_prequel_setup.tauE );

    if ( diffSeparator !=NULL )
    {
      delete diffSeparator;
      diffSeparator = NULL;
    }
    beadfind_done = true;
    my_prequel_setup.WriteBeadFindForSignalProcessing();

    maskPtr->UpdateBeadFindOutcomes ( wholeChip, my_prequel_setup.bfMaskFile.c_str(), !inception_state.bfd_control.SINGLEBF, 0, my_prequel_setup.bfStatsFile.c_str() ); // writes the beadfind mask in maskPtr

    my_progress.ReportState ( "Beadfind Complete" );

    // save state parameters
    string stateFile =  analysisLocation + "analysisState.json";
    string imageSpecFile =  analysisLocation + "imageState.h5";
    ProgramState state ( stateFile );
    state.Save ( inception_state,my_keys,my_image_spec );
    state.WriteState();
    //save image state
    CaptureImageState imgState ( imageSpecFile );
    imgState.CleanUpOldFile();
    imgState.WriteImageSpec ( my_image_spec, inception_state.img_control.maxFrames );
    imgState.WriteXTCorrection();
    if ( inception_state.img_control.gain_correct_images )
      imgState.WriteImageGainCorrection ( my_image_spec.rows, my_image_spec.cols );    

    if ( inception_state.mod_control.BEADFIND_ONLY )
    {
      fprintf ( stdout, "Beadfind Only Mode has completed successfully\n" );
      exit ( EXIT_SUCCESS );
    }
  }

  if ( !beadfind_done )
  {
    fprintf ( stderr, "Don't recognize --beadfind-type %s\n", inception_state.bfd_control.beadfindType.c_str() );
    exit ( EXIT_FAILURE );
  }
}

