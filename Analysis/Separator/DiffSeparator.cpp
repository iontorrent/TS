/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include "BFReference.h"
#include "Mask.h"
#include "ZeromerDiff.h"
#include "KeyClassifier.h"
#include "SampleKeyReporter.h"
#include "IncorpReporter.h"
#include "AvgKeyReporter.h"
#include "RegionAvgKeyReporter.h"
#include "DualGaussMixModel.h"
#include "KeyClassifyJob.h"
#include "KeyClassifyTauEJob.h"
#include "LoadTracesJob.h"
#include "FillCriticalFramesJob.h"
#include "Traces.h"
#include "Image.h"
#include "OptArgs.h"
#include "KClass.h"
#include "PJobQueue.h"
#include "DifferentialSeparator.h"

void usage() {
  cout << "DiffSeparator - Separates wells into bead & empties and classifies them based" << endl;
  cout << "on keys into library, TF based on key SNR score using signal based on approximation" << endl;
  cout << "to standard differential equation based signal processing. Can also do some filtering" << endl;
  cout << "and reporting based on mean signal, mean trace, model fit, etc." << endl;
  cout << "  " << endl;
  cout << "usage:" << endl;
  cout << " DiffSeparator --mask-file /opt/ion/config/exclusionMask_314.bin \\ " << endl;
  cout << "               --results-dir /path/to/dat/files \\ " << endl;
  cout << "               --out-prefix B10-165 \\ " << endl;
  cout << "  " << endl;
  cout << "options: " << endl;
  cout << "  --out-prefix - Name to prefix output files with. [required]    " << endl;
  cout << "  --mask-file - Exclusion mask to start with, assume no excluded wells if missing" << endl;
  cout << "  --min-snr - Min snr to be classified as a particular key [default: 4] " << endl;
  cout << "  --num-cores - How many cores to run on? 1 for debugging. [default: use numCores()]" << endl;
  cout << "  --max-mad - Maximum mean absolute deviation to allow for filtering." << endl;
  cout << "  --min-taue-snr - Minimum bead snr to be used for calculating bulk buffering." << endl;
  cout << "  --taue-step - Size of regions to use for bulk buffering estimation [default: 128]" << endl;
  cout << "  --report-wells - File with 0 based index of wells to report for. [default: none] " << endl;
  cout << "  --report-step - Report on every Nth well [default: 0]." << endl;
  cout << "  --do-mean-filter - Filter out wells that have very small mean trace [default: true]" << endl;
  cout << "  --do-sig-sd-filter - Filter out empty wells with high variation [default: false]" << endl;
  cout << "  --do-empty-sig-filter - Filter empty well signal outliers [default: false]" << endl;  
  cout << "  --sig-sd-filter-mult - Mult of IQR/2 plus median for sd threshold. [default: 5]" << endl;
  cout << "  --do-mad-filter - Filter out poor fitting welss in zeromers [default: true]" << endl;
  cout << "  --do-recover-sd-filter - Mark low signal beads as emtpy for reference. Useful " << endl;
  cout << "                           for 316 when many wells can be marked bead [default: false]" << endl;
  cout << "  --do-remove-low-signal - Mark low signal keypass beads as ignore to avoid putting" << endl;
  cout << "                           noisy beads into processing pipeline" << endl;
	cout << "  --just-beadfind - Just do beadfind on bfmetric, no separating." << endl;
	cout << "  --cluster-trim - What percentage of trimming to do for outliers before clustering [.02]" << endl;
	cout << "  --bf-threshold - What ownership level call a bead (legacy .35)  [.5]" << endl;
  cout << "" << endl;
  exit(1);
}

int main(int argc, const char *argv[]) {
  // Fill in references
  KClass kc;
  BFReference reference;
  Mask mask;
  DifSepOpt  dOpts;
  OptArgs opts;  
	string predictDat;
  opts.ParseCmdLine(argc, argv);
  opts.GetOption(dOpts.doMeanFilter, "true", '-', "do-mean-filter");
  opts.GetOption(dOpts.doSigVarFilter, "true", '-', "do-sig-sd-filter");
  opts.GetOption(dOpts.doMadFilter, "true", '-', "do-mad-filter");
  opts.GetOption(dOpts.doEmptyCenterSignal, "false", '-', "do-empty-sig-filter");
  opts.GetOption(dOpts.doRecoverSdFilter, "false", '-', "do-recover-sd-filter");
  opts.GetOption(dOpts.doRemoveLowSignalFilter, "false", '-', "do-remove-low-signal");
  opts.GetOption(dOpts.maskFile, "", '-', "mask-file");
  opts.GetOption(dOpts.minSnr, "6", '-', "min-snr");
  opts.GetOption(dOpts.nCores, "-1", '-', "num-cores");
  opts.GetOption(dOpts.maxMad, "30", '-', "max-mad");
  opts.GetOption(dOpts.minTauESnr, "7", '-', "min-taue-snr");
  opts.GetOption(dOpts.tauEEstimateStep, "50", '-', "taue-step");
  opts.GetOption(dOpts.resultsDir, "", '-', "results-dir");
  opts.GetOption(dOpts.sigSdMult, "4", '-', "sig-sd-filter-mult");
  opts.GetOption(dOpts.outData, "", '-', "out-prefix");
  opts.GetOption(dOpts.wellsReportFile, "", '-', "report-wells");
  opts.GetOption(dOpts.reportStepSize, "0", '-', "report-step");
  opts.GetOption(dOpts.justBeadfind, "false", '-', "just-beadfind");
  opts.GetOption(dOpts.clusterTrim, ".02", '-', "cluster-trim");
	opts.GetOption(dOpts.bfThreshold, ".5", '-', "bf-threshold");
	opts.GetOption(dOpts.bfMeshStep, "50", '-', "bf-mesh-step");
	opts.GetOption(dOpts.bfNeighbors, "2", '-', "bf-neighbors");
	opts.GetOption(dOpts.samplingStep, "10", '-', "sample-step");
	opts.GetOption(predictDat, "", '-', "predict-dat");
	opts.GetOption(dOpts.regionYSize, "50", '-', "row-step");
	opts.GetOption(dOpts.regionXSize, "50", '-', "col-step");
  opts.GetOption(dOpts.help, "false", 'h', "help");
  opts.GetOption(dOpts.bfType, "", '-', "beadfind-basis");
  opts.CheckNoLeftovers();
  if (dOpts.resultsDir.empty() || dOpts.outData.empty() || dOpts.help) {
    usage();
  }

	DifferentialSeparator separator;
	separator.Run(dOpts);
	if (predictDat != "") {
          //	  Separator.Predictflow(Predictdat, Dopts.Outdata, Dopts.Ignorechecksumerrors, dOpts);
	}
	return 0;
}
