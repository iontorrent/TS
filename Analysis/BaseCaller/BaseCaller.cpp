/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <time.h>
#include "BaseCaller.h"
#include "WellFileManipulation.h"
#include "MaskSample.h"
#include "mixed.h"

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
  if (clo.sys_context.LOCAL_WELLS_FILE) {
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
    sff_scratch_files.destinationDir = strdup(clo.sys_context.basecaller_output_directory);
    sff_scratch_files.fn_sfflib = strdup(sffLIBFileName);
    sff_scratch_files.fn_sfftf = strdup(sffTFFileName);
  }
}

void CopyScratchSFFToFinalDestination(SFFscratch &sff_scratch_files, CommandLineOpts &clo)
{
  if (clo.sys_context.LOCAL_WELLS_FILE) {
    //Move SFF files from temp location
    char src[MAX_PATH_LENGTH];
    char dst[MAX_PATH_LENGTH];

    sprintf(src, "/tmp/%s", sff_scratch_files.fn_sfflib);
    sprintf(dst, "%s/%s", clo.sys_context.basecaller_output_directory, sff_scratch_files.sffLIBFileName);
    CopyFile(src, dst);
    unlink(src);

    sprintf(src, "/tmp/%s", sff_scratch_files.fn_sfftf);
    sprintf(dst, "%s/%s", clo.sys_context.basecaller_output_directory, sff_scratch_files.sffTFFileName);
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
void GenerateBasesFromWells(CommandLineOpts &clo, RawWells &rawWells, Mask *maskPtr, SequenceItem *seqList, int rows, int cols,
    char *experimentName, TrackProgress &my_progress)
{

  BaseCaller basecaller(&clo, &rawWells, clo.flow_context.flowOrder, maskPtr, rows, cols, my_progress.fpLog); // 7th duplicated code instance for flow

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
  basecaller.generateCafieRegionsFile(clo.sys_context.basecaller_output_directory);

  my_progress.ReportState("Basecalling Complete");

  basecaller.saveBaseCallerJson(clo.sys_context.basecaller_output_directory);

}



BaseCaller::BaseCaller(CommandLineOpts *_clo, RawWells *_rawWells, const char *_flowOrder, Mask *_maskPtr, int _rows, int _cols, FILE *fpLog)
{
  clo = _clo;
  rawWells = _rawWells;
  maskPtr = _maskPtr;
  rows = _rows;
  cols = _cols;

  flowOrder = _flowOrder;
  chipID = ChipIdDecoder::GetGlobalChipId();
  numRegions = clo->cfe_control.cfiedrRegionsX * clo->cfe_control.cfiedrRegionsY;

  numFlows = clo->GetNumFlows();
  if (clo->cfe_control.numCafieSolveFlows)
    numFlows = std::min(numFlows, clo->cfe_control.numCafieSolveFlows);

  cf.assign(numRegions,0.0);
  ie.assign(numRegions,0.0);
  droop.assign(numRegions,0.0);

  numWorkers = 1;
  commonOutputMutex = NULL;
  wellStatFileFP = NULL;
  phaseResid = NULL;
  numWellsCalled = 0;

  className[0] = "lib";
  classKeyBases[0] = clo->key_context.libKey;  //should draw from seqList
  className[1] = "tf";
  classKeyBases[1] = clo->key_context.tfKey; //should draw from seqList
  for (int iClass = 0; iClass < numClasses; iClass++) {
    classKeyBasesLength[iClass] = classKeyBases[iClass].length();
    classKeyFlowsLength[iClass] = seqToFlow(classKeyBases[iClass].c_str(), classKeyBasesLength[iClass], classKeyFlows[iClass], MAX_KEY_FLOWS,
        (char *) flowOrder.c_str(), flowOrder.size()); // already executed in seqList?
  }

  classFilterPolyclonal[0] = clo->flt_control.clonalFilterSolving;
  classFilterHighResidual[0] = clo->flt_control.cafieResFilterCalling;

  classFilterPolyclonal[1] = clo->flt_control.clonalFilterSolving && clo->flt_control.percentPositiveFlowsFilterTFs;
  classFilterHighResidual[1] = clo->flt_control.cafieResFilterCalling && clo->flt_control.cafieResFilterTFs;

  nextRegionX = 0;
  nextRegionY = 0;
}

BaseCaller::~BaseCaller()
{
}



void BaseCaller::FindClonalPopulation(char *experimentName, const vector<int>& keyIonogram)
{
  if (clo->flt_control.clonalFilterSolving or clo->flt_control.clonalFilterTraining) {
    filter_counts counts;
    int nlib = maskPtr->GetCount(static_cast<MaskType> (MaskLib));
    counts._nsamp = min(nlib, clo->flt_control.nUnfilteredLib);
    make_filter(clonalPopulation, counts, *maskPtr, *rawWells, keyIonogram);
    cout << counts << endl;
  }
}



void BaseCaller::DoPhaseEstimation(SequenceItem *seqList)
{

  printf("Phase estimation mode = %s\n", clo->cfe_control.libPhaseEstimator.c_str());

  if (clo->cfe_control.libPhaseEstimator == "override") {

    // user sets the library values
    for (int r = 0; r < numRegions; r++) {
      cf[r] = clo->cfe_control.LibcfOverride;
      ie[r] = clo->cfe_control.LibieOverride;
      droop[r] = clo->cfe_control.LibdrOverride;
    }

  } else if ((clo->cfe_control.libPhaseEstimator == "nel-mead-treephaser") || (clo->cfe_control.libPhaseEstimator == "nel-mead-adaptive-treephaser")){

    int numWorkers = std::max(numCores(), 2);
    if (clo->cfe_control.singleCoreCafie)
      numWorkers = 1;

    RegionAnalysis regionAnalysis;
    regionAnalysis.analyze(&cf, &ie, &droop, rawWells, maskPtr, seqList, clo, flowOrder, numFlows, numWorkers);

  } else {
    ION_ABORT("Requested phase estimator is not recognized");
  }

  // Save phase estimates to BaseCaller.json

  float CFmean = 0;
  float IEmean = 0;
  float DRmean = 0;
  int count = 0;

  for (int r = 0; r < numRegions; r++) {
    basecallerJson["Phasing"]["CFbyRegion"][r] = cf[r];
    basecallerJson["Phasing"]["IEbyRegion"][r] = ie[r];
    basecallerJson["Phasing"]["DRbyRegion"][r] = droop[r];
    if (cf[r] || ie[r] || droop[r]) {
      CFmean += cf[r];
      IEmean += ie[r];
      DRmean += droop[r];
      count++;
    }
  }
  basecallerJson["Phasing"]["RegionRows"] = clo->cfe_control.cfiedrRegionsY;
  basecallerJson["Phasing"]["RegionCols"] = clo->cfe_control.cfiedrRegionsX;

  basecallerJson["Phasing"]["CF"] = count ? (CFmean/count) : 0;
  basecallerJson["Phasing"]["IE"] = count ? (IEmean/count) : 0;
  basecallerJson["Phasing"]["DR"] = count ? (DRmean/count) : 0;
}

bool BaseCaller::LoadRegion(std::deque<int> &wellX, std::deque<int> &wellY, std::deque<std::vector<float> > &wellMeasurements, int &currentRegion, std::string &msg)
{
	if (nextRegionY >= wellsReader.numRegionsY) {
		return false;
	}

	int currentRegionX = nextRegionX;
	int currentRegionY = nextRegionY;

	wellsReader.loadRegion(wellX, wellY, wellMeasurements, currentRegionX, currentRegionY, maskPtr);

	msg = std::string();
	if (currentRegionX == 0) {
		char buff[32];
		sprintf(buff, "% 5d/% 5d: ", currentRegionY*clo->loc_context.regionYSize, rows);
		msg += buff;
	}

	if (wellX.size() == 0)
		msg += "  ";
	else if (wellX.size() < 750)
		msg += ". ";
	else if (wellX.size() < 1500)
		msg += "o ";
	else if (wellX.size() < 2250)
		msg += "# ";
	else
		msg += "$ ";

	currentRegion = currentRegionX + wellsReader.numRegionsX * currentRegionY;
	nextRegionX++;
	if (nextRegionX == wellsReader.numRegionsX) {
		nextRegionX = 0;
		nextRegionY++;
		msg += "\n";
	}

	return true;
}
  
typedef struct {
	BaseCaller *baseCaller;
	std::deque<int> wellX;
	std::deque<int> wellY;
	std::deque<std::vector<float> > wellMeasurements;
	PerBaseQual pbq;
	unsigned int iWorker;
	int currentRegion;
	std::string msg;
	bool started;
	bool completed;
} BaseCallerThreadData;

void BaseCaller::DoThreadedBasecalling(char *resultsDir, char *sffLIBFileName, char *sffTFFileName)
{

  numWorkers = 1;
  if (clo->cfe_control.singleCoreCafie == false) {
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
  assert (wellsReader.OpenForRead2(rawWells, cols, rows, clo->loc_context.regionXSize, clo->loc_context.regionYSize));
  int numWellRegions = wellsReader.numRegionsX * wellsReader.numRegionsY;


  if (!clo->cfe_control.basecallSubset.empty()) {
    printf("Basecalling limited to user-specified set of %d wells. No random library sff will be generated\n",
        (int)clo->cfe_control.basecallSubset.size());

  } else { // Generate random library subset
    MaskSample<well_index_t> randomLib(*maskPtr, MaskLib, clo->flt_control.nUnfilteredLib);
    randomLibSet.insert(randomLib.Sample().begin(), randomLib.Sample().end());
  }

  // If we have randomly-sampled unfiltered reads, write them out
  if (!randomLibSet.empty()) {
    string unfilteredLibDir = string(clo->sys_context.basecaller_output_directory) + "/" + string(clo->flt_control.unfilteredLibDir);
    if (mkdir(unfilteredLibDir.c_str(), 0777) && (errno != EEXIST)) {
      randomLibSet.clear(); // This will ensure no writes for now
      ION_WARN("*Warning* - problem making directory " + unfilteredLibDir + " for unfiltered lib results")
    } else {

      string unfilteredSffName = string(clo->sys_context.runId) + ".lib.unfiltered.untrimmed.sff";
      randomLibSFF.OpenForWrite(unfilteredLibDir.c_str(),unfilteredSffName.c_str(),
          numWellRegions, numFlows, flowOrder.c_str(), clo->key_context.libKey);

      string filterStatusFileName = unfilteredLibDir + string("/") + string(clo->sys_context.runId) + string(".filterStatus.txt");
      filterStatus.open(filterStatusFileName.c_str());
      filterStatus << "col";
      filterStatus << "\t" << "row";
      filterStatus << "\t" << "highRes";
      filterStatus << "\t" << "valid";
      filterStatus << endl;
    }
  }

  libSFF.OpenForWrite(resultsDir, sffLIBFileName, numWellRegions, numFlows, flowOrder.c_str(), clo->key_context.libKey);  // should be iterating over keys?
  tfSFF.OpenForWrite(resultsDir, sffTFFileName, numWellRegions, numFlows, flowOrder.c_str(), clo->key_context.tfKey);


  for (well_index_t bfmaskIndex = 0; bfmaskIndex < (well_index_t)rows*cols; bfmaskIndex++) {
    (*maskPtr)[bfmaskIndex] &= MaskAll
        - MaskFilteredBadPPF - MaskFilteredShort - MaskFilteredBadKey - MaskFilteredBadResidual - MaskKeypass;
  }
  //
  // Step 2. Initialize read filtering
  //

  for (int iClass = 0; iClass < numClasses; iClass++) {
    classCountPolyclonal[iClass] = classCountHighPPF[iClass] = classCountZeroBases[iClass] = classCountTooShort[iClass] = 0;
    classCountFailKeypass[iClass] = classCountHighResidual[iClass] = classCountValid[iClass] = classCountTotal[iClass] = 0;
  }

  //
  // Step 3. Open miscellaneous results files
  //

  // Set up phase residual file
  phaseResid = NULL;
  if (clo->cfe_control.doCafieResidual) {
    string wellsExtension = ".wells";
    string phaseResidualExtension = ".cafie-residuals";
    string phaseResidualPath = clo->sys_context.wellsFileName;
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
    phaseResid->SetChunk(0, rows, 0, cols, 0, numFlows);
  }

  // Set up wellStats file (if necessary)
  OpenWellStatFile();

  //
  // Step 4. Execute threaded basecalling
  //

  // Prepare threads
  pthread_mutex_t _commonOutputMutex = PTHREAD_MUTEX_INITIALIZER;
  commonOutputMutex   = &_commonOutputMutex;
  numWellsCalled = 0;
  nextRegionX = 0;
  nextRegionY = 0;

  time_t startBasecall;
  time(&startBasecall);

  // NB: this takes twice as much memory as before
  BaseCallerThreadData baseCallerThreadData[2*numWorkers]; // NB: one more for to load in data...
  pthread_t workerID[numWorkers];
  int loadDataI = numWorkers;
  int loadDataN = numWorkers;
  
  // reserve
  for (unsigned int iWorker = 0; iWorker < numWorkers; iWorker++) { 
	  baseCallerThreadData[iWorker].started = false;
	  baseCallerThreadData[iWorker].completed = false;
  }
  
  // Launch threads
  int numRunning = 0;
  while (true) { // NB: this makes threading lock free

	  /*
	  printf("numRunning=%d numWorkers=%d\n", numRunning, numWorkers); // HERE
	  fflush(stdout);
	  */
	  // Join threads
	  for (unsigned int iWorker = 0; iWorker < numWorkers; iWorker++) {
		  if (baseCallerThreadData[iWorker].completed || !baseCallerThreadData[iWorker].started) {
			  bool wasStarted = baseCallerThreadData[iWorker].started;
			  // join
			  if (baseCallerThreadData[iWorker].started) {
				  pthread_join(workerID[iWorker], NULL);
	        numRunning--;
			  }
			  baseCallerThreadData[iWorker].completed = false; // always set to false so that we do not consider ever again...

			  // NB: it would be great to have a thread that its only job is to read in data
			  if (0 < loadDataN && loadDataN == loadDataI) { // load more data
				  for (loadDataI = 0; loadDataI < loadDataN; loadDataI++) {
					  if (!LoadRegion(baseCallerThreadData[numWorkers+loadDataI].wellX,
							  baseCallerThreadData[numWorkers+loadDataI].wellY,
							  baseCallerThreadData[numWorkers+loadDataI].wellMeasurements,
							  baseCallerThreadData[numWorkers+loadDataI].currentRegion,
							  baseCallerThreadData[numWorkers+loadDataI].msg)) {
						  loadDataN = loadDataI;
						  break;
					  }
				  }
				  loadDataI = 0;
			  }

			  // is there more data?
			  if (loadDataI < loadDataN) {
				  // print status
				  printf("%s", baseCallerThreadData[numWorkers+loadDataI].msg.c_str());
				  fflush(stdout);
				  // copy region
				  baseCallerThreadData[iWorker].wellX.swap(baseCallerThreadData[numWorkers+loadDataI].wellX);
				  baseCallerThreadData[iWorker].wellY.swap(baseCallerThreadData[numWorkers+loadDataI].wellY);
				  baseCallerThreadData[iWorker].wellMeasurements.swap(baseCallerThreadData[numWorkers+loadDataI].wellMeasurements);
				  baseCallerThreadData[iWorker].currentRegion = baseCallerThreadData[numWorkers+loadDataI].currentRegion;
				  if (!baseCallerThreadData[iWorker].pbq.Init(chipID, flowOrder, clo->cfe_control.phredTableFile)) {
					  ION_ABORT("*Error* - perBaseQualInit failed");
				  }
				  // copy thread data
				  baseCallerThreadData[iWorker].baseCaller = this;
				  baseCallerThreadData[iWorker].iWorker = iWorker;
				  
				  // launch the thread
				  if (pthread_create(&workerID[iWorker], NULL, doBasecall, &baseCallerThreadData[iWorker]))
					  ION_ABORT("*Error* - problem starting thread");
				  numRunning++;
				  baseCallerThreadData[iWorker].started = true;
				  
				  loadDataI++;
			  }

			  if (wasStarted) {
				  iWorker = -1; // start a the beginning, searching for joining threads
			  }
		  }
	  }
	  
	  // sleep
	  struct timespec req;
	  req.tv_sec = 0;
	  //req.tv_nsec = 100000000; // 100 milliseconds
	  //req.tv_nsec = 10000000; // 10 milliseconds
	  req.tv_nsec = 1000000; // 1 milliseconds
	  nanosleep(&req, NULL);
	  //sleep(1); // TODO: how long?

	  if (numRunning <= 0)
	    break;
  }

  time_t endBasecall;
  time(&endBasecall);

  //
  // Step 5. Close files and print out some statistics
  //

  if (!clo->cfe_control.basecallSubset.empty()) {
    cout << endl << "BASECALLING: called " << numWellsCalled << " of " << clo->cfe_control.basecallSubset.size() << " wells in "
        << difftime(endBasecall,startBasecall) << " seconds with " << numWorkers << " threads" << endl;
  } else {
    cout << endl << "BASECALLING: called " << numWellsCalled << " of " << (rows*cols) << " wells in "
        << difftime(endBasecall,startBasecall) << " seconds with " << numWorkers << " threads" << endl;
  }

  writePrettyText(cout);
//  writeTSV(clo->beadSummaryFile);

  libSFF.Close();
  printf("Generated library SFF with %d reads\n", libSFF.NumReads());
  tfSFF.Close();
  printf("Generated TF SFF with %d reads\n", tfSFF.NumReads());

  // Close files
  if (wellStatFileFP)
    fclose(wellStatFileFP);
  if(phaseResid) {
    phaseResid->WriteWells();
    phaseResid->WriteRanks();
    phaseResid->WriteInfo();
    phaseResid->Close();
    delete phaseResid;
  }

  if (!randomLibSet.empty()) {
    filterStatus.close();
    randomLibSFF.Close();
    printf("Generated random unfiltered library SFF with %d reads\n", randomLibSFF.NumReads());
  }

  for (int iClass = 0; iClass < numClasses; iClass++) {
    basecallerJson["BeadSummary"][className[iClass]]["key"] = classKeyBases[iClass];
    basecallerJson["BeadSummary"][className[iClass]]["polyclonal"] = classCountPolyclonal[iClass];
    basecallerJson["BeadSummary"][className[iClass]]["highPPF"] = classCountHighPPF[iClass];
    basecallerJson["BeadSummary"][className[iClass]]["zero"] = classCountZeroBases[iClass];
    basecallerJson["BeadSummary"][className[iClass]]["short"] = classCountTooShort[iClass];
    basecallerJson["BeadSummary"][className[iClass]]["badKey"] = classCountFailKeypass[iClass];
    basecallerJson["BeadSummary"][className[iClass]]["highRes"] = classCountHighResidual[iClass];
    basecallerJson["BeadSummary"][className[iClass]]["valid"] = classCountValid[iClass];
  }

}

void *doBasecall(void *input)
{
  BaseCallerThreadData *data = static_cast<BaseCallerThreadData*>(input); 
  data->baseCaller->BasecallerWorker(data->wellX, data->wellY, data->wellMeasurements, data->pbq, data->currentRegion);
  data->completed = true;
  return NULL;
}

void BaseCaller::BasecallerWorker(std::deque<int> &wellX, std::deque<int> &wellY, std::deque<std::vector<float> > &wellMeasurements, PerBaseQual &pbq, int currentRegion)
{

  // initialize the per base quality score generator

  // Things we must produce for each read
  hpLen_vec_t calledHpLen(numFlows, 0);
  weight_vec_t keyNormSig(numFlows, 0);
  weight_vec_t residual(numFlows, 0);
  double multiplier = 1.0;

  // Process the data
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

	  if (!clo->cfe_control.basecallSubset.empty())
		  if (clo->cfe_control.basecallSubset.count(pair<unsigned short, unsigned short>(*x,*y)) == 0)
			  continue;

	  well_index_t readIndex = (*x) + (*y) * cols;
	  bool isTF = maskPtr->Match(readIndex, MaskTF);
	  bool isLib = maskPtr->Match(readIndex, MaskLib);
	  if (!isTF && !isLib)
		  continue;
	  bool isRandomLib = randomLibSet.count(readIndex) > 0;

    int  iClass = isLib ? 0 : 1;

	  // TODO: Improved, more general (x,y) -> cafie lookup
	  unsigned short cafieYinc = ceil(rows / (double) clo->cfe_control.cfiedrRegionsY);
	  unsigned short cafieXinc = ceil(cols / (double) clo->cfe_control.cfiedrRegionsX);
	  read_region_t iRegion = (*y / cafieYinc) + (*x / cafieXinc) * clo->cfe_control.cfiedrRegionsY;
	  double        currentCF = cf[iRegion];
	  double        currentIE = ie[iRegion];
	  double        currentDR = droop[iRegion];

	  //
	  // Step 3. Perform base calling and quality value calculation
	  //

	  SFFWriterWell readResults;
	  stringstream wellNameStream;
	  wellNameStream << clo->sys_context.runId << ":" << (*y) << ":" << (*x);
	  readResults.name = wellNameStream.str();
	  readResults.clipQualLeft = classKeyBasesLength[iClass] + 1;
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
	  read.SetDataAndKeyNormalize(&(measurements->at(0)), numFlows, classKeyFlows[iClass], classKeyFlowsLength[iClass] - 1);

	  if (clo->flt_control.clonalFilterSolving) {
          vector<float>::const_iterator first = read.measurements.begin() + mixed_first_flow();
          vector<float>::const_iterator last  = read.measurements.begin() + mixed_last_flow();
		  float ppf = percent_positive(first, last);
		  float ssq = sum_fractional_part(first, last);
		  if(ppf > mixed_ppf_cutoff())
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

	  if (clo->cfe_control.basecaller == "dp-treephaser")
		  dpTreephaser.SetModelParameters(currentCF, currentIE, currentDR);
	  else
		  dpTreephaser.SetModelParameters(currentCF, currentIE, 0); // Adaptive normalization

	  // Execute the iterative solving-normalization routine
	  if (clo->cfe_control.basecaller == "dp-treephaser")
		  dpTreephaser.NormalizeAndSolve4(read, numFlows);
	  else if (clo->cfe_control.basecaller == "treephaser-adaptive")
		  dpTreephaser.NormalizeAndSolve3(read, numFlows); // Adaptive normalization
	  else
		  dpTreephaser.NormalizeAndSolve5(read, numFlows); // sliding window adaptive normalization

	  // one more pass to get quality metrics
	  dpTreephaser.ComputeQVmetrics(read);

	  for (int iFlow = 0; iFlow < numFlows; iFlow++) {
		  keyNormSig[iFlow] = read.normalizedMeasurements[iFlow];
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
	  // Step 4. Calculate/save read metrics and apply filters
	  //

	  double ppf = getPPF(calledHpLen, PERCENT_POSITIVE_FLOWS_N);
	  double medAbsResidual = getMedianAbsoluteCafieResidual(residual, CAFIE_RESIDUAL_FLOWS_N);

	  bool isFailKeypass = false;
	  for (int iFlow = 0; iFlow < (classKeyFlowsLength[iClass]-1); iFlow++)
		  if (classKeyFlows[iClass][iFlow] != calledHpLen[iFlow])
			  isFailKeypass = true;

	  pthread_mutex_lock(commonOutputMutex);

	  bool isReadValid = false;
	  bool isReadHighResidual = false;

	  if(readResults.numBases == 0) {
		  classCountZeroBases[iClass]++;
		  (*maskPtr)[readIndex] |= MaskFilteredShort;

	  } else if(readResults.numBases < clo->flt_control.minReadLength) {
		  classCountTooShort[iClass]++;
		  (*maskPtr)[readIndex] |= MaskFilteredShort;

	  } else if(clo->flt_control.KEYPASSFILTER && isFailKeypass) {
		  classCountFailKeypass[iClass]++;
		  (*maskPtr)[readIndex] |= MaskFilteredBadKey;

	  } else if (classFilterPolyclonal[iClass] && (isReadHighPPF || isReadPolyclonal)) {
		  if(isReadHighPPF)
			  classCountHighPPF[iClass]++;
		  else if(isReadPolyclonal)
			  classCountPolyclonal[iClass]++;
		  (*maskPtr)[readIndex] |= MaskFilteredBadPPF;

	  } else if(classFilterHighResidual[iClass] && (medAbsResidual > clo->flt_control.cafieResMaxValue)) {
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

	  WriteWellStatFileEntry(isLib?MaskLib:MaskTF, classKeyFlows[iClass], classKeyFlowsLength[iClass], *x, *y, keyNormSig, readResults.numBases,
			  currentCF, currentIE, currentDR, multiplier, ppf, !isReadPolyclonal, medAbsResidual);

	  if (clo->cfe_control.doCafieResidual && (phaseResid != NULL))
		  for (int iFlow = 0; iFlow < numFlows; iFlow++)
			  phaseResid->WriteFlowgram(iFlow, *x, *y, residual[iFlow]);

	  pthread_mutex_unlock(commonOutputMutex);

	  //
	  // Step 5. Save the basecalling results to appropriate sff files
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



