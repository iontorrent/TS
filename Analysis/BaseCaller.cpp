/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BaseCaller.h"
#include "WellFileManipulation.h"
#include "MaskSample.h"

using namespace std;



struct SFFscratch {
  char *destinationDir;
  char *fn_sfflib;
  char *fn_sfftf;
  char *experimentName;
  char *sffLIBFileName;
  char *sffTFFileName;
};

void ConstructSFFscratch(SFFscratch &sff_scratch_files, CommandLineOpts &clo, char *experimentName)
{
  static char *sffLIBFileName = "rawlib.sff";
  static char *sffTFFileName = "rawtf.sff";
  sff_scratch_files.experimentName = strdup(experimentName);
  sff_scratch_files.sffLIBFileName = strdup(sffLIBFileName);
  sff_scratch_files.sffTFFileName = strdup(sffTFFileName);

  ClearStaleSFFFiles();
  if (clo.LOCAL_WELLS_FILE) {
    // override current experimentName with /tmp
    sff_scratch_files.destinationDir = strdup("/tmp");
    char fTemplate[256] = { 0 };
    int tmpFH = 0;

    sprintf(fTemplate, "/tmp/%d_%sXXXXXX", getpid(), sffLIBFileName);
    tmpFH = mkstemp(fTemplate);
    if (tmpFH > 0)
      close(tmpFH);
    else
      exit(EXIT_FAILURE);
    sff_scratch_files.fn_sfflib = strdup(basename(fTemplate));

    sprintf(fTemplate, "/tmp/%d_%sXXXXXX", getpid(), sffTFFileName);
    tmpFH = mkstemp(fTemplate);
    if (tmpFH > 0)
      close(tmpFH);
    else
      exit(EXIT_FAILURE);
    sff_scratch_files.fn_sfftf = strdup(basename(fTemplate));

  } else {
    // use experimentName for destination directory
    sff_scratch_files.destinationDir = strdup(experimentName);
    sff_scratch_files.fn_sfflib = strdup(sffLIBFileName);
    sff_scratch_files.fn_sfftf = strdup(sffTFFileName);
  }
}

void CopyScratchSFFToFinalDestination(SFFscratch &sff_scratch_files, CommandLineOpts &clo)
{
  if (clo.LOCAL_WELLS_FILE) {
    //Move SFF files from temp location
    char src[MAX_PATH_LENGTH];
    char dst[MAX_PATH_LENGTH];

    sprintf(src, "/tmp/%s", sff_scratch_files.fn_sfflib);
    sprintf(dst, "%s/%s", sff_scratch_files.experimentName, sff_scratch_files.sffLIBFileName);
    CopyFile(src, dst);
    unlink(src);

    sprintf(src, "/tmp/%s", sff_scratch_files.fn_sfftf);
    sprintf(dst, "%s/%s", sff_scratch_files.experimentName, sff_scratch_files.sffTFFileName);
    CopyFile(src, dst);
    unlink(src);
  }
}
void DestroySFFScratch(SFFscratch &sff_scratch_files)
{
  if (sff_scratch_files.destinationDir)
    free(sff_scratch_files.destinationDir);
  if (sff_scratch_files.fn_sfflib)
    free(sff_scratch_files.fn_sfflib);
  if (sff_scratch_files.fn_sfftf)
    free(sff_scratch_files.fn_sfftf);
  if (sff_scratch_files.experimentName)
    free(sff_scratch_files.experimentName);
  if (sff_scratch_files.sffLIBFileName)
    free(sff_scratch_files.sffLIBFileName);
  if (sff_scratch_files.sffTFFileName)
    free(sff_scratch_files.sffTFFileName);
}

//
// Main tasks performed in base calling section of the analysis:
//  - Polyclonal filter training
//  - Phasing parameter estimation for library reads & TFs
//  - Actual base calling: dephasing, filtering, QV calculation, TF classification
//
void GenerateBasesFromWells(CommandLineOpts &clo, RawWells &rawWells, Flow *flw, Mask *maskPtr, SequenceItem *seqList, int rows, int cols,
    char *experimentName, TrackProgress &my_progress)
{

  BaseCaller basecaller(&clo, &rawWells, flw->GetFlowOrder(), maskPtr, rows, cols, my_progress.fpLog);

  rawWells.Close();
  SetWellsToLiveBeadsOnly(rawWells,maskPtr);
  rawWells.OpenForIncrementalRead();
  rawWells.ResetCurrentRegionWell();
  MemUsage("RawWellsBasecalling");
  // Find distribution of clonal reads for use in read filtering:
  vector<int> keyIonogram(seqList[1].Ionogram, seqList[1].Ionogram+seqList[1].usableKeyFlows);
  basecaller.FindClonalPopulation(experimentName, keyIonogram);
  MemUsage("ClonalPopulation");
  my_progress.ReportState("Polyclonal Filter Training Complete");

  // Library CF/IE/DR parameter estimation
  MemUsage("BeforePhaseEstimation");
  basecaller.DoPhaseEstimation(seqList);
  MemUsage("AfterPhaseEstimation");
  my_progress.ReportState("Phase Parameter Estimation Complete");

  // Write SFF files to local directory, rather than directly to Report directory
  SFFscratch sff_scratch_files;
  ConstructSFFscratch(sff_scratch_files, clo, experimentName);
  MemUsage("BeforeBasecalling");
  // Perform base calling and save results to appropriate sff files
  basecaller.DoThreadedBasecalling(sff_scratch_files.destinationDir, sff_scratch_files.fn_sfflib, sff_scratch_files.fn_sfftf);
  MemUsage("AfterBasecalling");
  CopyScratchSFFToFinalDestination(sff_scratch_files, clo);
  DestroySFFScratch(sff_scratch_files);

  // Generate TF-related and phase estimation-related files:
  // TFTracking.txt, cafieMetrics.txt, cafieRegions.txt
  basecaller.generateTFTrackingFile(experimentName);
  basecaller.generateCafieMetricsFile(experimentName);
  basecaller.generateCafieRegionsFile(experimentName);

  my_progress.ReportState("Basecalling Complete");

  basecaller.saveBaseCallerJson(experimentName);

}



BaseCaller::BaseCaller(CommandLineOpts *_clo, RawWells *_rawWells, const char *_flowOrder, Mask *_maskPtr, int _rows, int _cols, FILE *fpLog)
  : tfs(_flowOrder)
{
  clo = _clo;
  rawWells = _rawWells;
  maskPtr = _maskPtr;
  rows = _rows;
  cols = _cols;

  if (clo->TFoverride != NULL) {
    if (tfs.LoadConfig(clo->TFoverride) == true) {
      fprintf(stdout, "Loading TFs from '%s'\n", clo->TFoverride);
    } else {
      fprintf(stderr, "Warning!\n");
      fprintf(stderr, "Specified TF config file, '%s', was not found.\n", clo->TFoverride);
      fprintf(fpLog, "Specified TF config file, '%s', was not found.\n", clo->TFoverride);
    }
  } else {
    char *tfConfigFileName = tfs.GetTFConfigFile();
    if (tfConfigFileName == NULL) {
      fprintf(stderr, "Warning!\n");
      fprintf(stderr, "No TF config file was found.\n");
      fprintf(fpLog, "No TF config file was found.\n");
    } else {
      tfs.LoadConfig(tfConfigFileName);
      fprintf(stdout, "Loading TFs from '%s'\n", tfConfigFileName);
      free(tfConfigFileName);
    }
  }
  tfInfo = tfs.Info();
  numTFs = tfs.Num();

  flowOrder = _flowOrder;
  chipID = ChipIdDecoder::GetGlobalChipId();
  numRegions = clo->cfiedrRegionsX * clo->cfiedrRegionsY;

  numFlows = clo->GetNumFlows();
  if (clo->numCafieSolveFlows)
    numFlows = std::min(numFlows, clo->numCafieSolveFlows);

  numFlowsTFClassify = std::min(numFlows, clo->cafieFlowMax);

  cf.assign(numRegions,0.0);
  ie.assign(numRegions,0.0);
  droop.assign(numRegions,0.0);

  TFCount.assign(numTFs, 0);
  avgTFSignal.assign(numTFs, vector<double>(numFlowsTFClassify, 0.0));
  avgTFSignalSquared.assign(numTFs, vector<double>(numFlowsTFClassify, 0.0));
  avgTFCorrected.assign(numTFs, vector<double>(numFlowsTFClassify, 0.0));
  tfSignalHist.assign(numTFs, vector<int>(numFlowsTFClassify*maxTFSignalHist,0));

  tfCallCorrect.assign(numTFs, vector<int>(4*maxTFHPHist,0));
  tfCallUnder.assign(numTFs, vector<int>(4*maxTFHPHist,0));
  tfCallOver.assign(numTFs, vector<int>(4*maxTFHPHist,0));
  tfCallCorrect2.assign(numTFs, vector<int>(4*maxTFHPHist,0));
  tfCallUnder2.assign(numTFs, vector<int>(4*maxTFHPHist,0));
  tfCallOver2.assign(numTFs, vector<int>(4*maxTFHPHist,0));

  tfCallCorrect3.assign(numTFs, vector<int>(maxTFSparklineFlows,0));
  tfCallTotal3.assign(numTFs, vector<int>(maxTFSparklineFlows,0));

  numWorkers = 1;
  commonInputMutex = NULL;
  commonOutputMutex = NULL;
  wellStatFileFP = NULL;
  phaseResid = NULL;
  numWellsCalled = 0;

  numKeyBasesTF = strlen(clo->tfKey);
  numKeyFlowsTF = seqToFlow(clo->tfKey, numKeyBasesTF, keyFlowTF, MAX_KEY_FLOWS,
      (char *) flowOrder.c_str(), flowOrder.size());

  numKeyBasesLib = strlen(clo->libKey);
  numKeyFlowsLib = seqToFlow(clo->libKey, numKeyBasesLib, keyFlowLib, MAX_KEY_FLOWS,
      (char *) flowOrder.c_str(), flowOrder.size());

  nextRegionX = 0;
  nextRegionY = 0;

  numClasses = 0;
}

BaseCaller::~BaseCaller()
{
}



void BaseCaller::FindClonalPopulation(char *experimentName, const vector<int>& keyIonogram)
{
  if (clo->clonalFilterSolving or clo->clonalFilterTraining) {
    filter_counts counts;
    int nlib = maskPtr->GetCount(static_cast<MaskType> (MaskLib));
    counts._nsamp = min(nlib, clo->nUnfilteredLib);
    make_filter(clonalPopulation, counts, nlib, *maskPtr, *rawWells, keyIonogram);
    cout << counts << endl;
  }
}



void BaseCaller::DoPhaseEstimation(SequenceItem *seqList)
{

  printf("Phase estimation mode = %s\n", clo->libPhaseEstimator.c_str());

  if (clo->libPhaseEstimator == "override") {

    // user sets the library values
    for (int r = 0; r < numRegions; r++) {
      cf[r] = clo->LibcfOverride;
      ie[r] = clo->LibieOverride;
      droop[r] = clo->LibdrOverride;
    }

  } else if ((clo->libPhaseEstimator == "nel-mead-treephaser") || (clo->libPhaseEstimator == "nel-mead-adaptive-treephaser")){

    int numWorkers = std::max(numCores(), 2);
    if (clo->singleCoreCafie)
      numWorkers = 1;

    RegionAnalysis regionAnalysis;
    regionAnalysis.analyze(&cf, &ie, &droop, rawWells, maskPtr, seqList, clo, flowOrder, numFlows, numWorkers);

  } else {
    ION_ABORT("Requested phase estimator is not recognized");
  }
}



void BaseCaller::DoThreadedBasecalling(char *resultsDir, char *sffLIBFileName, char *sffTFFileName)
{

  numWorkers = 1;
  if (clo->singleCoreCafie == false) {
    numWorkers = 2*numCores();
//    numWorkers = 24*2;
    // Limit threads to half the number of cores with minimum of 4 threads
    numWorkers = (numWorkers > 4 ? numWorkers : 4);
  }

  //
  // Step 1. Open wells and sff files
  //

  // Open the wells file
  rawWells->ResetCurrentWell();
  assert(numFlows <= (int)rawWells->NumFlows());
  assert (wellsReader.OpenForRead2(rawWells, cols, rows, clo->regionXSize, clo->regionYSize));
  int numWellRegions = wellsReader.numRegionsX * wellsReader.numRegionsY;


  if (!clo->basecallSubset.empty()) {
    printf("Basecalling limited to user-specified set of %d wells. No random library sff will be generated\n",
        (int)clo->basecallSubset.size());

  } else { // Generate random library subset
    MaskSample<well_index_t> randomLib(*maskPtr, MaskLib, clo->nUnfilteredLib);
    randomLibSet.insert(randomLib.Sample().begin(), randomLib.Sample().end());
  }

  // If we have randomly-sampled unfiltered reads, write them out
  if (!randomLibSet.empty()) {
    string unfilteredLibDir = string(clo->GetExperimentName()) + "/" + string(clo->unfilteredLibDir);
    if (mkdir(unfilteredLibDir.c_str(), 0777) && (errno != EEXIST)) {
      randomLibSet.clear(); // This will ensure no writes for now
      ION_WARN("*Warning* - problem making directory " + unfilteredLibDir + " for unfiltered lib results")
    } else {

      string unfilteredSffName = string(clo->runId) + ".lib.unfiltered.untrimmed.sff";
      randomLibSFF.OpenForWrite(unfilteredLibDir.c_str(),unfilteredSffName.c_str(),
          numWellRegions, numFlows, flowOrder.c_str(), clo->libKey);

      string filterStatusFileName = unfilteredLibDir + string("/") + string(clo->runId) + string(".filterStatus.txt");
      filterStatus.open(filterStatusFileName.c_str());
      filterStatus << "col";
      filterStatus << "\t" << "row";
      filterStatus << "\t" << "highRes";
      filterStatus << "\t" << "valid";
      filterStatus << endl;
    }
  }

  libSFF.OpenForWrite(resultsDir, sffLIBFileName, numWellRegions, numFlows, flowOrder.c_str(), clo->libKey);
  tfSFF.OpenForWrite(resultsDir, sffTFFileName, numWellRegions, numFlows, flowOrder.c_str(), clo->tfKey);


  for (well_index_t bfmaskIndex = 0; bfmaskIndex < (well_index_t)rows*cols; bfmaskIndex++) {
    (*maskPtr)[bfmaskIndex] &= MaskAll
        - MaskFilteredBadPPF - MaskFilteredShort - MaskFilteredBadKey - MaskFilteredBadResidual - MaskKeypass;
  }
  //
  // Step 2. Initialize read filtering
  //

  numClasses = numTFs + 2;
  classCountPolyclonal.assign(numClasses,0);
  classCountHighPPF.assign(numClasses,0);
  classCountZeroBases.assign(numClasses,0);
  classCountTooShort.assign(numClasses,0);
  classCountFailKeypass.assign(numClasses,0);
  classCountHighResidual.assign(numClasses,0);
  classCountValid.assign(numClasses,0);
  classCountTotal.assign(numClasses,0);
  className.resize(numClasses);
  classKey.resize(numClasses);
  classFilterPolyclonal.resize(numClasses);
  classFilterHighResidual.resize(numClasses);

  className[0] = "lib";
  classKey[0] = clo->libKey;
  classFilterPolyclonal[0] = clo->clonalFilterSolving;
  classFilterHighResidual[0] = clo->cafieResFilterCalling;

  for (int iBin = 1; iBin < (numTFs + 2); iBin++) {
    if (iBin == numTFs+1)
      className[iBin] = "TF-??";
    else
      className[iBin] = tfInfo[iBin-1].name;
    classKey[iBin] = clo->tfKey;
    classFilterPolyclonal[iBin] = clo->clonalFilterSolving && clo->percentPositiveFlowsFilterTFs;
    classFilterHighResidual[iBin] = clo->cafieResFilterCalling && clo->cafieResFilterTFs;
  }

  //
  // Step 3. Open miscellaneous results files
  //

  // Set up phase residual file
  phaseResid = NULL;
  if (clo->doCafieResidual) {
    string wellsExtension = ".wells";
    string phaseResidualExtension = ".cafie-residuals";
    string phaseResidualPath = clo->wellsFileName;
    int matchPos = phaseResidualPath.size() - wellsExtension.size();
    if ((matchPos >= 0) && (wellsExtension == phaseResidualPath.substr(matchPos, wellsExtension.size()))) {
      phaseResidualPath.replace(matchPos, wellsExtension.size(), phaseResidualExtension);
    } else {
      phaseResidualPath = string("1") + phaseResidualExtension;
    }
    string phaseResidualDir;
    string phaseResidualFile;
    FillInDirName(phaseResidualPath,phaseResidualDir,phaseResidualFile);

    phaseResid = new RawWells(phaseResidualDir.c_str(), phaseResidualFile.c_str());
    phaseResid->CreateEmpty(numFlows, flowOrder.c_str(), rows, cols);
    phaseResid->OpenForWrite();
  }

  // Set up wellStats file (if necessary)
  OpenWellStatFile();

  //
  // Step 4. Execute threaded basecalling
  //

  // Prepare threads
  pthread_mutex_t _commonInputMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t _commonOutputMutex = PTHREAD_MUTEX_INITIALIZER;
  commonInputMutex    = &_commonInputMutex;
  commonOutputMutex   = &_commonOutputMutex;
  numWellsCalled = 0;
  nextRegionX = 0;
  nextRegionY = 0;

  time_t startBasecall;
  time(&startBasecall);

  // Launch threads
  pthread_t workerID[numWorkers];
  for (unsigned int iWorker = 0; iWorker < numWorkers; iWorker++)
    if (pthread_create(&workerID[iWorker], NULL, doBasecall, this))
      ION_ABORT("*Error* - problem starting thread");

  // Join threads
  for (unsigned int iWorker = 0; iWorker < numWorkers; iWorker++)
    pthread_join(workerID[iWorker], NULL);

  time_t endBasecall;
  time(&endBasecall);

  //
  // Step 5. Close files and print out some statistics
  //

  if (!clo->basecallSubset.empty()) {
    cout << endl << "BASECALLING: called " << numWellsCalled << " of " << clo->basecallSubset.size() << " wells in "
        << difftime(endBasecall,startBasecall) << " seconds with " << numWorkers << " threads" << endl;
  } else {
    cout << endl << "BASECALLING: called " << numWellsCalled << " of " << (rows*cols) << " wells in "
        << difftime(endBasecall,startBasecall) << " seconds with " << numWorkers << " threads" << endl;
  }

  writePrettyText(cout);
  writeTSV(clo->beadSummaryFile);

  libSFF.Close();
  printf("Generated library SFF with %d reads\n", libSFF.NumReads());
  tfSFF.Close();
  printf("Generated TF SFF with %d reads\n", tfSFF.NumReads());

  // Close files
  if (wellStatFileFP)
    fclose(wellStatFileFP);
  if(phaseResid) {
    phaseResid->Close();
    delete phaseResid;
  }

  if (!randomLibSet.empty()) {
    filterStatus.close();
    randomLibSFF.Close();
    printf("Generated random unfiltered library SFF with %d reads\n", randomLibSFF.NumReads());
  }

}



void *doBasecall(void *input)
{
  static_cast<BaseCaller*> (input)->BasecallerWorker();
  return NULL;
}

void BaseCaller::BasecallerWorker()
{

  // initialize the per base quality score generator
  PerBaseQual pbq;
  pthread_mutex_lock(commonInputMutex);
  if (!pbq.Init(chipID, flowOrder, clo->phredTableFile)) {
    pthread_mutex_unlock(commonInputMutex);
    ION_ABORT("*Error* - perBaseQualInit failed");
  }
  pthread_mutex_unlock(commonInputMutex);

  // Things we must produce for each read
  hpLen_vec_t calledHpLen(numFlows, 0);
  weight_vec_t keyNormSig(numFlows, 0);
  weight_vec_t residual(numFlows, 0);
  double multiplier = 1.0;

  std::deque<int> wellX;
  std::deque<int> wellY;
  std::deque<std::vector<float> > wellMeasurements;

  while (1) {

    //
    // Step 1. Load a region worth of reads
    //

    pthread_mutex_lock(commonInputMutex);

    if (nextRegionY >= wellsReader.numRegionsY) {
      pthread_mutex_unlock(commonInputMutex);
      break;
    }

    int currentRegionX = nextRegionX;
    int currentRegionY = nextRegionY;

    wellsReader.loadRegion(wellX, wellY, wellMeasurements, currentRegionX, currentRegionY, maskPtr);

    if (currentRegionX == 0)
      printf("% 5d/% 5d: ", currentRegionY*clo->regionYSize, rows);

    if (wellX.size() == 0)
      printf("  ");
    else if (wellX.size() < 750)
      printf(". ");
    else if (wellX.size() < 1500)
      printf("o ");
    else if (wellX.size() < 2250)
      printf("# ");
    else
      printf("$ ");

    int currentRegion = currentRegionX + wellsReader.numRegionsX * currentRegionY;
    nextRegionX++;
    if (nextRegionX == wellsReader.numRegionsX) {
      nextRegionX = 0;
      nextRegionY++;
      printf("\n");
    }
    fflush(stdout);
    pthread_mutex_unlock(commonInputMutex);

    deque<SFFWriterWell> randomLibReads;
    deque<SFFWriterWell> libReads;
    deque<SFFWriterWell> tfReads;

    deque<int>::iterator x = wellX.begin();
    deque<int>::iterator y = wellY.begin();
    deque<std::vector<float> >::iterator measurements = wellMeasurements.begin();

    for (; x != wellX.end() ; x++, y++, measurements++) {

      //
      // Step 2. Retrieve additional information needed to process this read
      //

      if (!clo->basecallSubset.empty())
        if (clo->basecallSubset.count(pair<unsigned short, unsigned short>(*x,*y)) == 0)
          continue;

      well_index_t readIndex = (*x) + (*y) * cols;
      bool isTF = maskPtr->Match(readIndex, MaskTF);
      bool isLib = maskPtr->Match(readIndex, MaskLib);
      if (!isTF && !isLib)
        continue;
      bool isRandomLib = randomLibSet.count(readIndex) > 0;

      // TODO: Improved, more general (x,y) -> cafie lookup
      unsigned short cafieYinc = ceil(rows / (double) clo->cfiedrRegionsY);
      unsigned short cafieXinc = ceil(cols / (double) clo->cfiedrRegionsX);
      read_region_t iRegion = (*y / cafieYinc) + (*x / cafieXinc) * clo->cfiedrRegionsY;
      double        currentCF = cf[iRegion];
      double        currentIE = ie[iRegion];
      double        currentDR = droop[iRegion];

      int *keyFlow = keyFlowLib;
      int numKeyFlows = numKeyFlowsLib;
      int numKeyBases = numKeyBasesLib;
      if (isTF) {
        keyFlow = keyFlowTF;
        numKeyFlows = numKeyFlowsTF;
        numKeyBases = numKeyBasesTF;
      }

      //
      // Step 3. Perform base calling and quality value calculation
      //

      SFFWriterWell readResults;
      stringstream wellNameStream;
      wellNameStream << clo->runId << ":" << (*y) << ":" << (*x);
      readResults.name = wellNameStream.str();
      readResults.clipQualLeft = numKeyBases + 1;
      readResults.clipQualRight = 0;
      readResults.clipAdapterLeft = 0;
      readResults.clipAdapterRight = 0;
      readResults.flowIonogram.resize(numFlows);

      bool isReadPolyclonal = false;
      bool isReadHighPPF    = false;

//      if ((clo->basecaller == "dp-treephaser") ||
//          (clo->basecaller == "treephaser-adaptive") ||
//          (clo->basecaller == "treephaser-swan")) {

        BasecallerRead read;
        read.SetDataAndKeyNormalize(&(measurements->at(0)), numFlows, keyFlow, numKeyFlows - 1);

        if (clo->clonalFilterSolving) {
          float ppf = percent_positive(read.measurements.begin() + 12, read.measurements.begin() + 72);
          float ssq = sum_fractional_part(read.measurements.begin() + 12, read.measurements.begin() + 72);
		  if(ppf > 0.84)
		  	isReadHighPPF = true;
		  else if(!clonalPopulation.is_clonal(ppf, ssq))
            isReadPolyclonal = true;
        }

        if (classFilterPolyclonal[0] && isLib && isReadPolyclonal && !isRandomLib) {
          pthread_mutex_lock(commonOutputMutex);
		  if(isReadHighPPF)
            classCountHighPPF[0]++;
		  if(isReadPolyclonal)
            classCountPolyclonal[0]++;
          (*maskPtr)[readIndex] |= MaskFilteredBadPPF;
          classCountTotal[0]++;
          numWellsCalled++;
          pthread_mutex_unlock(commonOutputMutex);
          continue;
        }

        DPTreephaser dpTreephaser(flowOrder.c_str(), numFlows, 8);

        if (clo->basecaller == "dp-treephaser")
          dpTreephaser.SetModelParameters(currentCF, currentIE, currentDR);
        else
          dpTreephaser.SetModelParameters(currentCF, currentIE, 0); // Adaptive normalization

        // Execute the iterative solving-normalization routine
        if (clo->basecaller == "dp-treephaser")
          dpTreephaser.NormalizeAndSolve4(read, numFlows);
        else if (clo->basecaller == "treephaser-adaptive")
          dpTreephaser.NormalizeAndSolve3(read, numFlows); // Adaptive normalization
        else
          dpTreephaser.NormalizeAndSolve5(read, numFlows); // sliding window adaptive normalization

        // one more pass to get quality metrics
        dpTreephaser.ComputeQVmetrics(read);

        for (int iFlow = 0; iFlow < numFlows; iFlow++) {
          keyNormSig[iFlow] = read.measurements[iFlow];   // Temporary TF debug
    //        keyNormSig[iFlow] = read.normalizedMeasurements[iFlow];
          calledHpLen[iFlow] = read.solution[iFlow];
          residual[iFlow] = read.normalizedMeasurements[iFlow] - read.prediction[iFlow];
          float perFlowAdjustment = residual[iFlow] / dpTreephaser.oneMerHeight[iFlow];
          if (perFlowAdjustment > 0.49)
            perFlowAdjustment = 0.49;
          else if (perFlowAdjustment < -0.49)
            perFlowAdjustment = -0.49;
          if ((perFlowAdjustment < 0) && (calledHpLen[iFlow] == 0))
            perFlowAdjustment = 0;
          readResults.flowIonogram[iFlow] = perFlowAdjustment + calledHpLen[iFlow];
        }

        multiplier = read.keyNormalizer;

        readResults.numBases = 0;
        for (int iFlow = 0; iFlow < numFlows; iFlow++)
          readResults.numBases += calledHpLen[iFlow];

        readResults.baseFlowIndex.reserve(readResults.numBases);
        readResults.baseCalls.reserve(readResults.numBases);
        readResults.baseQVs.reserve(readResults.numBases);

        unsigned int prev_used_flow = 0;
        for (int iFlow = 0; iFlow < numFlows; iFlow++) {
          for (hpLen_t hp = 0; hp < calledHpLen[iFlow]; hp++) {
            readResults.baseFlowIndex.push_back(1 + iFlow - prev_used_flow);
            readResults.baseCalls.push_back(flowOrder[iFlow % flowOrder.length()]);
            prev_used_flow = iFlow + 1;
          }
        }
        // Calculation of quality values
        pbq.setWellName(readResults.name);
        pbq.GenerateQualityPerBaseTreephaser(dpTreephaser.penaltyResidual, dpTreephaser.penaltyMismatch,
            readResults.flowIonogram, residual, readResults.baseFlowIndex);
        pbq.GetQualities(readResults.baseQVs);

//      } else { // Unrecognized
//        ION_ABORT("Requested basecaller is not recognized");
//      }

      //
      // Step 4. If this read is a TF, classify it
      //
      //  TODO: All TF classification and metric processing should be done outside Analysis.
      //        Potential places: TFMapper, plugins
      //

      read_class_t  iClass = 0;
      if (isTF) {

        int bestTF = numTFs;
        double bestScore = clo->minTFScore;

        int seqFlows = std::min(numFlowsTFClassify, 40); // don't want to compare more than 40 flows, too many errors
        if (clo->alternateTFMode == 1)
          seqFlows = (int) (numFlowsTFClassify * 0.9 + 0.5); // just fit to 90% of the flows to get good confidence in the basecalls, the last few are lower qual due to CF

        for (int tf = 0; tf < numTFs; tf++) {
          int numTestFlows = std::min(seqFlows, tfInfo[tf].flows); // don't compare more than this TF's flows
          if (numTestFlows <= clo->minTFFlows)
            continue;

          int correct = 0;
          for (int iFlow = 0; iFlow < numTestFlows; iFlow++) {
            if (clo->alternateTFMode == 1) {
              if (calledHpLen[iFlow] == tfInfo[tf].Ionogram[iFlow])
                correct++;
            } else {
              if ((calledHpLen[iFlow] > 0) == (tfInfo[tf].Ionogram[iFlow] > 0))
                correct++;
            }
          }
          double score = (double) correct / (double) numTestFlows;

          if (score > bestScore) {
            bestScore = score;
            bestTF = tf;
          }
        }

        if (bestTF < numTFs) {
          pthread_mutex_lock(commonOutputMutex);
          TFCount[bestTF]++;
          for (int iFlow = 0; (iFlow < numFlowsTFClassify) && (iFlow < numFlows); iFlow++) {
            //avgTFSignal[bestTF][iFlow] += keyNormSig[iFlow];
            avgTFSignal[bestTF][iFlow] += read.normalizedMeasurements[iFlow];
            avgTFSignalSquared[bestTF][iFlow] += read.normalizedMeasurements[iFlow] * read.normalizedMeasurements[iFlow];

            avgTFCorrected[bestTF][iFlow] += readResults.flowIonogram[iFlow];

            int quantizedTFSignal = (int) rint(40.0 * read.normalizedMeasurements[iFlow]);
            quantizedTFSignal = min(max(quantizedTFSignal,0),maxTFSignalHist-1);
            tfSignalHist[bestTF][iFlow * maxTFSignalHist + quantizedTFSignal]++;
          }

          for (int iFlow = 0; (iFlow < numFlowsTFClassify) && (iFlow < numFlows); iFlow++) {

            char base = flowOrder[iFlow % flowOrder.length()];
            int baseIdx = 0;
            if      (base == 'A') baseIdx = 0;
            else if (base == 'C') baseIdx = 1;
            else if (base == 'G') baseIdx = 2;
            else                  baseIdx = 3;

            if (calledHpLen[iFlow] == tfInfo[bestTF].Ionogram[iFlow])
              tfCallCorrect[bestTF][tfInfo[bestTF].Ionogram[iFlow] + baseIdx*maxTFHPHist]++;
            else if (calledHpLen[iFlow] > tfInfo[bestTF].Ionogram[iFlow])
              tfCallOver[bestTF][tfInfo[bestTF].Ionogram[iFlow] + baseIdx*maxTFHPHist]++;
            else
              tfCallUnder[bestTF][tfInfo[bestTF].Ionogram[iFlow] + baseIdx*maxTFHPHist]++;
          }

          for (int iFlow = 0; (iFlow < (tfInfo[bestTF].flows-2)) && (iFlow < numFlows); iFlow++) {

            char base = flowOrder[iFlow % flowOrder.length()];
            int baseIdx = 0;
            if      (base == 'A') baseIdx = 0;
            else if (base == 'C') baseIdx = 1;
            else if (base == 'G') baseIdx = 2;
            else                  baseIdx = 3;

            if (calledHpLen[iFlow] == tfInfo[bestTF].Ionogram[iFlow])
              tfCallCorrect2[bestTF][tfInfo[bestTF].Ionogram[iFlow] + baseIdx*maxTFHPHist]++;
            else if (calledHpLen[iFlow] > tfInfo[bestTF].Ionogram[iFlow])
              tfCallOver2[bestTF][tfInfo[bestTF].Ionogram[iFlow] + baseIdx*maxTFHPHist]++;
            else
              tfCallUnder2[bestTF][tfInfo[bestTF].Ionogram[iFlow] + baseIdx*maxTFHPHist]++;
          }

          // Sparkline data

          int lastCalledFlow = 0;
          for (int iFlow = 0; iFlow < numFlows; iFlow++)
            if (calledHpLen[iFlow] > 0)
              lastCalledFlow = iFlow;

          for (int iFlow = 0; (iFlow <= lastCalledFlow) && (iFlow < tfInfo[bestTF].flows) && (iFlow < maxTFSparklineFlows); iFlow++) {
            tfCallTotal3[bestTF][iFlow]++;
            if (calledHpLen[iFlow] == tfInfo[bestTF].Ionogram[iFlow])
              tfCallCorrect3[bestTF][iFlow]++;
          }

          readClass[readIndex] = 1+bestTF;
          pthread_mutex_unlock(commonOutputMutex);
        }
        iClass = 1 + bestTF;
      }

      //
      // Step 5. Calculate/save read metrics and apply filters
      //

      double ppf = getPPF(calledHpLen, PERCENT_POSITIVE_FLOWS_N);
      double medAbsResidual = getMedianAbsoluteCafieResidual(residual, CAFIE_RESIDUAL_FLOWS_N);

      bool isFailKeypass = false;
      for (int iFlow = 0; iFlow < (numKeyFlows-1); iFlow++)
        if (keyFlow[iFlow] != calledHpLen[iFlow])
          isFailKeypass = true;

      pthread_mutex_lock(commonOutputMutex);

      bool isReadValid = false;
      bool isReadHighResidual = false;

      if(readResults.numBases == 0) {
        classCountZeroBases[iClass]++;
        (*maskPtr)[readIndex] |= MaskFilteredShort;

      } else if(readResults.numBases < clo->minReadLength) {
        classCountTooShort[iClass]++;
        (*maskPtr)[readIndex] |= MaskFilteredShort;

      } else if(clo->KEYPASSFILTER && isFailKeypass) {
        classCountFailKeypass[iClass]++;
        (*maskPtr)[readIndex] |= MaskFilteredBadKey;

      } else if (classFilterPolyclonal[iClass] && (isReadHighPPF || isReadPolyclonal)) {
		if(isReadHighPPF)
          classCountHighPPF[iClass]++;
		else if(isReadPolyclonal)
          classCountPolyclonal[iClass]++;
        (*maskPtr)[readIndex] |= MaskFilteredBadPPF;

      } else if(classFilterHighResidual[iClass] && (medAbsResidual > clo->cafieResMaxValue)) {
        classCountHighResidual[iClass]++;
        isReadHighResidual = true;
        (*maskPtr)[readIndex] |= MaskFilteredBadResidual;

      } else {
        classCountValid[iClass]++;
        isReadValid = true;
        (*maskPtr)[readIndex] |= (MaskType) MaskKeypass;
      }

      classCountTotal[iClass]++;
      numWellsCalled++;

      WriteWellStatFileEntry(isLib?MaskLib:MaskTF, keyFlow, numKeyFlows, *x, *y, keyNormSig, readResults.numBases,
          currentCF, currentIE, currentDR, multiplier, ppf, !isReadPolyclonal, medAbsResidual);

      if (clo->doCafieResidual && (phaseResid != NULL))
        for (int iFlow = 0; iFlow < numFlows; iFlow++)
          phaseResid->WriteFlowgram(iFlow, *x, *y, residual[iFlow]);

      pthread_mutex_unlock(commonOutputMutex);

      //
      // Step 6. Save the basecalling results to appropriate sff files
      //

      if (isRandomLib) {
        pthread_mutex_lock(commonOutputMutex);
        filterStatus << (*x);
        filterStatus << "\t" << (*y);
        filterStatus << "\t" << (int) isReadHighResidual;
        filterStatus << "\t" << (int) isReadValid;
        filterStatus << endl;
        pthread_mutex_unlock(commonOutputMutex);

        randomLibReads.push_back(SFFWriterWell());

        if (isReadValid)
          readResults.copyTo(randomLibReads.back());
        else
          readResults.moveTo(randomLibReads.back());
      }

      if (isReadValid) {
        if (isLib) {
          libReads.push_back(SFFWriterWell());
          readResults.moveTo(libReads.back());
        } else {
          tfReads.push_back(SFFWriterWell());
          readResults.moveTo(tfReads.back());
        }
      }
    }

    libSFF.WriteRegion(currentRegion,libReads);
    tfSFF.WriteRegion(currentRegion,tfReads);

    if (!randomLibSet.empty())
      randomLibSFF.WriteRegion(currentRegion,randomLibReads);
  }
}




// Return percentage of positive flows
double getPPF(hpLen_vec_t &predictedExtension, unsigned int nFlowsToAssess) {
  double ppf = 0;

  unsigned int nFlow = min(predictedExtension.size(), (size_t) nFlowsToAssess);
  for (unsigned int iFlow = 0; iFlow < nFlow; iFlow++)
    if (predictedExtension[iFlow] > 0)
      ppf++;
  if(nFlow > 0)
    ppf /= (double) nFlow;

  return ppf;
}

double getMedianAbsoluteCafieResidual(vector<weight_t> &residual, unsigned int nFlowsToAssess) {
  double medAbsCafieRes = 0;

  unsigned int nFlow = min(residual.size(), (size_t) nFlowsToAssess);
  if (nFlow > 0) {
    vector<double> absoluteResid(nFlow);
    for (unsigned int iFlow = 0; iFlow < nFlow; iFlow++) {
      absoluteResid[iFlow] = abs(residual[iFlow]);
    }
    medAbsCafieRes = ionStats::median(absoluteResid);
  }

  return medAbsCafieRes;
}



