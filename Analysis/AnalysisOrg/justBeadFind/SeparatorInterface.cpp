/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SeparatorInterface.h"
#include "ChipReduction.h"
#include "NNAvg.h"
#include "SampleStats.h"
#include "ChipIdDecoder.h"

void DoDiffSeparatorFromCLO (DifferentialSeparator *diffSeparator, CommandLineOpts &inception_state, Mask *maskPtr, string &analysisLocation, SequenceItem *seqList, int numSeqListItems)
{
  DifSepOpt opts;
  opts.doGainCorrect = inception_state.img_control.gain_correct_images;
  opts.useBeadfindGainCorrect = inception_state.bfd_control.useBeadfindGainCorrection;
  opts.useDataCollectGainCorrect = inception_state.bfd_control.useDatacollectGainCorrectionFile;
  opts.doubleTapFlows = inception_state.bfd_control.doubleTapFlows;
  opts.filterNoisyCols = inception_state.bfd_control.filterNoisyCols;
  opts.predictFlowStart = inception_state.bfd_control.predictFlowStart;
  opts.predictFlowEnd = inception_state.bfd_control.predictFlowEnd;
  opts.bfType = inception_state.bfd_control.bfType;
  opts.bfDat = inception_state.bfd_control.bfDat;
  opts.bfBgDat = inception_state.bfd_control.bfBgDat;
  opts.resultsDir = inception_state.sys_context.dat_source_directory;
  opts.outData = analysisLocation;
  opts.analysisDir =  analysisLocation;
  opts.ignoreChecksumErrors = inception_state.img_control.ignoreChecksumErrors;
  opts.noduds = inception_state.bfd_control.noduds;
  opts.outputDebug = inception_state.bfd_control.bfOutputDebug;
  if (opts.outputDebug == 0 && inception_state.bkg_control.pest_control.bkg_debug_files) {
    opts.outputDebug = 1;
  }
  opts.useSignalReference = inception_state.bfd_control.useSignalReference;
  opts.minRatioLiveWell = inception_state.bfd_control.bfMinLiveRatio;
  opts.doRecoverSdFilter = inception_state.bfd_control.skipBeadfindSdRecover == 0;
  opts.nCores = inception_state.bfd_control.numThreads;
  opts.useSeparatorRef = inception_state.bfd_control.beadfindUseSepRef == 1;
  opts.minTfPeakMax = inception_state.bfd_control.minTfPeakMax;
  opts.minLibPeakMax = inception_state.bfd_control.minLibPeakMax;
  opts.sdAsBf = inception_state.bfd_control.sdAsBf;
  opts.bfMult = inception_state.bfd_control.bfMult;
  opts.flowOrder = inception_state.flow_context.flowOrder; // 5th duplicated code instance of translating flow order to nucs
  opts.isThumbnail = inception_state.bfd_control.beadfindThumbnail == 1;
  opts.doComparatorCorrect = inception_state.img_control.col_flicker_correct;
  opts.aggressive_cnc = inception_state.img_control.aggressive_cnc;
  opts.blobFilter = inception_state.bfd_control.blobFilter == 1;
  opts.col_pair_pixel_xtalk_correct = inception_state.img_control.col_pair_pixel_xtalk_correct;
  opts.pair_xtalk_fraction = inception_state.img_control.pair_xtalk_fraction;
  opts.corr_noise_correct = inception_state.img_control.corr_noise_correct;

  opts.acqPrefix = inception_state.img_control.acqPrefix;
  opts.datPostfix = inception_state.img_control.datPostfix;

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
  diffSeparator->SetKeys (seqList, numSeqListItems, 
                          inception_state.bfd_control.bfMinLiveLibSnr, inception_state.bfd_control.bfMinLiveTfSnr, 
                          inception_state.bfd_control.minLibPeakMax, inception_state.bfd_control.minTfPeakMax);
  opts.smoothTrace = inception_state.bfd_control.beadfindSmoothTrace;

  opts.t0MeshStepX = inception_state.bfd_control.meshStepX;
  opts.t0MeshStepY = inception_state.bfd_control.meshStepY;
  opts.clusterMeshStepX = inception_state.bfd_control.meshStepX;
  opts.clusterMeshStepY = inception_state.bfd_control.meshStepY;
  opts.tauEEstimateStepX = inception_state.bfd_control.meshStepX;
  opts.tauEEstimateStepY = inception_state.bfd_control.meshStepY;
  opts.useMeshNeighbors = 0;
  opts.regionXSize = inception_state.loc_context.regionXSize;
  opts.regionYSize = inception_state.loc_context.regionYSize;
  opts.beadfindAcqThreshold = inception_state.bfd_control.beadfindAcqThreshold;
  opts.beadfindBfThreshold = inception_state.bfd_control.beadfindBfThreshold;

  // For saving dcOffset and NucStep
  opts.nucStepDir = string ( opts.analysisDir + string ( "/NucStepFromBeadfind" ));
  if ( mkdir ( opts.nucStepDir.c_str(), 0777 ) && ( errno != EEXIST ) )
  {
    perror ( opts.nucStepDir.c_str() );
    return;
  }

  diffSeparator->Run (opts);
}


void SetupForBkgModelTiming (DifferentialSeparator *diffSeparator, std::vector<float> &smooth_t0_est, 
                             std::vector<RegionTiming> &region_timing, std::vector<Region>& region_list, 
                             ImageSpecClass &my_image_spec, Mask *maskPtr, bool doSmooth, int numThreads, int x_clip, int y_clip)
{
  // compute timing information
  AvgKeyIncorporation *keyIncorporation = NULL;

  // Setup t0 estimation from beadfind to pass to background model
  std::vector<float> sep_t0_est;
  sep_t0_est = diffSeparator->GetT0();
  smooth_t0_est = sep_t0_est;
  if (doSmooth) {
    printf ("smoothing t0 estimate from separator.......");
    NNSmoothT0EstimateFast (maskPtr,my_image_spec.rows,my_image_spec.cols, sep_t0_est, smooth_t0_est, x_clip, y_clip);
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
  std::vector<float> tempTauB;
  std::vector<float> tempTauE;
  tempTauB.reserve (maskPtr->W() * maskPtr->H());
  tempTauE.reserve (maskPtr->W() * maskPtr->H());
  for (size_t ik = 0; ik < (size_t) maskPtr->W() * maskPtr->H(); ++ik)
  {
    float avgTauB = diffSeparator->GetTauB(ik);
    float avgTauE = diffSeparator->GetTauE(ik);
    if (!isfinite(avgTauB)) { avgTauB = 0; }
    if (!isfinite(avgTauE)) { avgTauE = 0; }
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

void NNSmoothT0EstimateFast (Mask *mask,int imgRows,int imgCols,std::vector<float> &sep_t0_est,
                             std::vector<float> &output_t0_est, int x_clip, int y_clip)
{
  NNAvg smoother(imgRows, imgCols, 1);
  std::vector<char> bad_wells(sep_t0_est.size());
  std::vector<char>::iterator bw = bad_wells.begin(); 
  for(std::vector<float>::iterator i = sep_t0_est.begin(); i != sep_t0_est.end(); ++i, ++bw) {
    *i > 0 ? *bw = 0 : *bw = 1;
  }
  smoother.CalcCumulativeSum(&sep_t0_est[0], &bad_wells[0]);
  smoother.CalcNNAvg(y_clip, x_clip, SEPARATOR_T0_ESTIMATE_SMOOTH_DIST+1, SEPARATOR_T0_ESTIMATE_SMOOTH_DIST+1, -1.0f);
  std::vector<float>::iterator out_start = output_t0_est.begin();
  SampleStats<double> sample;
  for (int r=0;r < imgRows;r++) {
    for (int c=0;c < imgCols;c++) {
      *out_start = smoother.GetNNAvg(r, c, 0);
      sample.AddValue(*out_start);
      out_start++;
    }
  }
  fprintf(stdout, "Mean smoothed t0 is: %.8f\n", sample.GetMean());
}

void IsolatedBeadFind (
  SlicedPrequel &my_prequel_setup,
  ImageSpecClass &my_image_spec,
  Region &wholeChip,
  CommandLineOpts &inception_state,
  char *results_folder, string &analysisLocation,
  SeqListClass &my_keys,
  TrackProgress &my_progress,
  string& chipType)
{
  /*********************************************************************
  // Beadfind Section
   *********************************************************************/
  Mask bfmask ( 1, 1 );
  Mask *maskPtr = &bfmask;
  // beadfind has responsibility for defining the exclusion mask
  SetExcludeMask ( inception_state.loc_context,maskPtr,(char*)chipType.c_str(),my_image_spec.rows,my_image_spec.cols, inception_state.bfd_control.exclusionMaskFile, inception_state.bfd_control.beadfindThumbnail );
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
    int x_clip = my_image_spec.cols;
    int y_clip = my_image_spec.rows;
    if (inception_state.bfd_control.beadfindThumbnail == 1) {
      x_clip = 100;
      y_clip = 100;
    }
    SetupForBkgModelTiming ( diffSeparator, my_prequel_setup.smooth_t0_est, my_prequel_setup.region_timing, 
                             my_prequel_setup.region_list, my_image_spec, maskPtr,
                             inception_state.bfd_control.beadfindThumbnail != 1,
                             inception_state.bfd_control.numThreads, x_clip, y_clip);
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

