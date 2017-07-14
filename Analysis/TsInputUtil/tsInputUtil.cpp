/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "json/json.h"
#include "Utils.h"
#include "OptArgs.h"

#define TS_INPUT_UTIL_VERSION "1.0.0"

using namespace std;
using namespace Json;

enum OptType {
    OT_BOOL = 0,
    OT_INT,
    OT_FLOAT,
    OT_DOUBLE,
    OT_STRING,
    OT_VECTOR_INT,
    OT_VECTOR_FLOAT,
    OT_VECTOR_DOUBLE,
    OT_UNKOWN
};

void saveJson(const Json::Value & json, const string& filename_json)
{
	ofstream out(filename_json.c_str(), ios::out);
	if (out.good())
	{
		out << json.toStyledString();
	}
	else
	{
        cout << "tsinpututil ERROR: unable to write JSON file " << filename_json << endl;
	}

    out.close();
}

void usageMain()
{
    cerr << "tsinpututil - operations of TS inputs" << endl;
	cerr << "Usage: " << endl
       << "  tsinpututil [operation]" << endl
       << "    operation:" << endl
       << "              create" << endl
       << "              edit" << endl
       << "              validate" << endl
       << "              diff"  <<endl;
	exit(1);
}

void usageCreate()
{
    cerr << "tsinpututil create - create json files per chip type from input ts_dbData.json" << endl;
    cerr << "Usage: " << endl
       << "  tsinpututil create input_ts_dbData.json output_directory" << endl;
    exit(1);
}

void usageEdit()
{
    cerr << "tsinpututil edit - edit json file" << endl;
    cerr << "Usage: " << endl
       << "  tsinpututil edit input_json [option_arguments]" << endl;
    exit(1);
}

void usageValidate()
{
    cerr << "tsinpututil validate - validate json file" << endl;
    cerr << "Usage: " << endl
       << "  tsinpututil validate input_json" << endl;
    exit(1);
}

void usageDiff()
{
    cerr << "tsinpututil diff - compare inpute parameters in 2 log files" << endl;
    cerr << "Usage: " << endl
       << "  tsinpututil diff input_log1 input_log2" << endl;
    exit(1);
}

int main(int argc, const char *argv[]) 
{
	if(argc < 2)
	{
        usageMain();
	}
	
	if(argc == 2)
	{
		string option = argv[1];
		if("-h" == option)
		{
            usageMain();
		}
		else if("-v" == option)
		{
            cerr << "tsinpututil version: " << TS_INPUT_UTIL_VERSION << endl;
            usageMain();
		}
	}

    Value jsonBase(objectValue);
    jsonBase["chipType"] = "";
    jsonBase["BkgModelControlOpts"]["n-unfiltered-lib"]["type"] = OT_INT;
    jsonBase["BkgModelControlOpts"]["n-unfiltered-lib"]["value"] = 100000;
    jsonBase["BkgModelControlOpts"]["n-unfiltered-lib"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["n-unfiltered-lib"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["xtalk-correction"]["type"] = OT_BOOL;
    jsonBase["BkgModelControlOpts"]["xtalk-correction"]["value"] = true;
    jsonBase["BkgModelControlOpts"]["xtalk-correction"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["xtalk-correction"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["bkg-dont-emphasize-by-compression"]["type"] = OT_INT;
    jsonBase["BkgModelControlOpts"]["bkg-dont-emphasize-by-compression"]["value"] = 1;
    jsonBase["BkgModelControlOpts"]["bkg-dont-emphasize-by-compression"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["bkg-dont-emphasize-by-compression"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["nokey"]["type"] = OT_BOOL;
    jsonBase["BkgModelControlOpts"]["nokey"]["value"] = false;
    jsonBase["BkgModelControlOpts"]["nokey"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["nokey"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["clonal-filter-bkgmodel"]["type"] = OT_BOOL;
    jsonBase["BkgModelControlOpts"]["clonal-filter-bkgmodel"]["value"] = true;
    jsonBase["BkgModelControlOpts"]["clonal-filter-bkgmodel"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["clonal-filter-bkgmodel"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["clonal-filter-debug"]["type"] = OT_BOOL;
    jsonBase["BkgModelControlOpts"]["clonal-filter-debug"]["value"] = false;
    jsonBase["BkgModelControlOpts"]["clonal-filter-debug"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["clonal-filter-debug"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["clonal-filter-use-last-iter-params"]["type"] = OT_BOOL;
    jsonBase["BkgModelControlOpts"]["clonal-filter-use-last-iter-params"]["value"] = false;
    jsonBase["BkgModelControlOpts"]["clonal-filter-use-last-iter-params"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["clonal-filter-use-last-iter-params"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["filter-extreme-ppf-only"]["type"] = OT_BOOL;
    jsonBase["BkgModelControlOpts"]["filter-extreme-ppf-only"]["value"] = false;
    jsonBase["BkgModelControlOpts"]["filter-extreme-ppf-only"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["filter-extreme-ppf-only"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["mixed-first-flow"]["type"] = OT_INT;
    jsonBase["BkgModelControlOpts"]["mixed-first-flow"]["value"] = 12;
    jsonBase["BkgModelControlOpts"]["mixed-first-flow"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["mixed-first-flow"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["mixed-last-flow"]["type"] = OT_INT;
    jsonBase["BkgModelControlOpts"]["mixed-last-flow"]["value"] = 72;
    jsonBase["BkgModelControlOpts"]["mixed-last-flow"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["mixed-last-flow"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["max-iterations"]["type"] = OT_INT;
    jsonBase["BkgModelControlOpts"]["max-iterations"]["value"] = 30;
    jsonBase["BkgModelControlOpts"]["max-iterations"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["max-iterations"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["mixed-model-option"]["type"] = OT_INT;
    jsonBase["BkgModelControlOpts"]["mixed-model-option"]["value"] = 0;
    jsonBase["BkgModelControlOpts"]["mixed-model-option"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["mixed-model-option"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["mixed-stringency"]["type"] = OT_DOUBLE;
    jsonBase["BkgModelControlOpts"]["mixed-stringency"]["value"] = 0.5;
    jsonBase["BkgModelControlOpts"]["mixed-stringency"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["mixed-stringency"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["sigproc-regional-smoothing-alpha"]["type"] = OT_DOUBLE;
    jsonBase["BkgModelControlOpts"]["sigproc-regional-smoothing-alpha"]["value"] = 1.0;
    jsonBase["BkgModelControlOpts"]["sigproc-regional-smoothing-alpha"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["sigproc-regional-smoothing-alpha"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["sigproc-regional-smoothing-gamma"]["type"] = OT_DOUBLE;
    jsonBase["BkgModelControlOpts"]["sigproc-regional-smoothing-gamma"]["value"] = 1.0;
    jsonBase["BkgModelControlOpts"]["sigproc-regional-smoothing-gamma"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["sigproc-regional-smoothing-gamma"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["restart-region-params-file"]["type"] = OT_STRING;
    jsonBase["BkgModelControlOpts"]["restart-region-params-file"]["value"] = "";
    jsonBase["BkgModelControlOpts"]["restart-region-params-file"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["restart-region-params-file"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["bkg-washout-threshold"]["type"] = OT_DOUBLE;
    jsonBase["BkgModelControlOpts"]["bkg-washout-threshold"]["value"] = 2.0;
    jsonBase["BkgModelControlOpts"]["bkg-washout-threshold"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["bkg-washout-threshold"]["max"] = "";
    jsonBase["BkgModelControlOpts"]["bkg-washout-flow-detection"]["type"] = OT_INT;
    jsonBase["BkgModelControlOpts"]["bkg-washout-flow-detection"]["value"] = 6;
    jsonBase["BkgModelControlOpts"]["bkg-washout-flow-detection"]["min"] = "";
    jsonBase["BkgModelControlOpts"]["bkg-washout-flow-detection"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpuworkload"]["type"] = OT_DOUBLE;
    jsonBase["GpuControlOpts"]["gpuworkload"]["value"] = 1.0;
    jsonBase["GpuControlOpts"]["gpuworkload"]["min"] = 0.0;
    jsonBase["GpuControlOpts"]["gpuworkload"]["max"] = 1.0;
    jsonBase["GpuControlOpts"]["gpu-num-streams"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-num-streams"]["value"] = 2;
    jsonBase["GpuControlOpts"]["gpu-num-streams"]["min"] = 1;
    jsonBase["GpuControlOpts"]["gpu-num-streams"]["max"] = 16;
    jsonBase["GpuControlOpts"]["gpu-amp-guess"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-amp-guess"]["value"] = 1;
    jsonBase["GpuControlOpts"]["gpu-amp-guess"]["min"] = 0;
    jsonBase["GpuControlOpts"]["gpu-amp-guess"]["max"] = 1;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit"]["value"] = 1;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit"]["min"] = 0;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit"]["max"] = 1;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-blocksize"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-blocksize"]["value"] = -1;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-blocksize"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-blocksize"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-l1config"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-l1config"]["value"] = -1;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-l1config"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-l1config"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit"]["value"] = 1;
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit"]["min"] = 0;
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit"]["max"] = 1;
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-blocksize"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-blocksize"]["value"] = 128;
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-blocksize"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-blocksize"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-l1config"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-l1config"]["value"] = -1;
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-l1config"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-l1config"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-type"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-type"]["value"] = 3;
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-type"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-type"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-hybrid-fit-iter"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-hybrid-fit-iter"]["value"] = 3;
    jsonBase["GpuControlOpts"]["gpu-hybrid-fit-iter"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-hybrid-fit-iter"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-partial-deriv-blocksize"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-partial-deriv-blocksize"]["value"] = 128;
    jsonBase["GpuControlOpts"]["gpu-partial-deriv-blocksize"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-partial-deriv-blocksize"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-partial-deriv-l1config"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-partial-deriv-l1config"]["value"] = -1;
    jsonBase["GpuControlOpts"]["gpu-partial-deriv-l1config"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-partial-deriv-l1config"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-use-all-devices"]["type"] = OT_BOOL;
    jsonBase["GpuControlOpts"]["gpu-use-all-devices"]["value"] = false;
    jsonBase["GpuControlOpts"]["gpu-use-all-devices"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-use-all-devices"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-verbose"]["type"] = OT_BOOL;
    jsonBase["GpuControlOpts"]["gpu-verbose"]["value"] = false;
    jsonBase["GpuControlOpts"]["gpu-verbose"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-verbose"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-device-ids"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-device-ids"]["value"] = Value(arrayValue);
    jsonBase["GpuControlOpts"]["gpu-device-ids"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-device-ids"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-fitting-only"]["type"] = OT_BOOL;
    jsonBase["GpuControlOpts"]["gpu-fitting-only"]["value"] = true;
    jsonBase["GpuControlOpts"]["gpu-fitting-only"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-fitting-only"]["max"] = "";
    jsonBase["GpuControlOpts"]["post-fit-handshake-worker"]["type"] = OT_BOOL;
    jsonBase["GpuControlOpts"]["post-fit-handshake-worker"]["value"] = true;
    jsonBase["GpuControlOpts"]["post-fit-handshake-worker"]["min"] = "";
    jsonBase["GpuControlOpts"]["post-fit-handshake-worker"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-flow-by-flow"]["type"] = OT_BOOL;
    jsonBase["GpuControlOpts"]["gpu-flow-by-flow"]["value"] = false;
    jsonBase["GpuControlOpts"]["gpu-flow-by-flow"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-flow-by-flow"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-switch-to-flow-by-flow-at"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-switch-to-flow-by-flow-at"]["value"] = 20;
    jsonBase["GpuControlOpts"]["gpu-switch-to-flow-by-flow-at"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-switch-to-flow-by-flow-at"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-num-history-flows"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-num-history-flows"]["value"] = 10;
    jsonBase["GpuControlOpts"]["gpu-num-history-flows"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-num-history-flows"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-force-multi-flow-fit"]["type"] = OT_BOOL;
    jsonBase["GpuControlOpts"]["gpu-force-multi-flow-fit"]["value"] = false;
    jsonBase["GpuControlOpts"]["gpu-force-multi-flow-fit"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-force-multi-flow-fit"]["max"] = "";
    jsonBase["GpuControlOpts"]["gpu-memory-per-proc"]["type"] = OT_INT;
    jsonBase["GpuControlOpts"]["gpu-memory-per-proc"]["value"] = 0;
    jsonBase["GpuControlOpts"]["gpu-memory-per-proc"]["min"] = "";
    jsonBase["GpuControlOpts"]["gpu-memory-per-proc"]["max"] = "";
    jsonBase["SignalProcessingBlockControl"]["wells-compression"]["type"] = OT_INT;
    jsonBase["SignalProcessingBlockControl"]["wells-compression"]["value"] = 0;
    jsonBase["SignalProcessingBlockControl"]["wells-compression"]["min"] = 0;
    jsonBase["SignalProcessingBlockControl"]["wells-compression"]["max"] = 10;
    jsonBase["SignalProcessingBlockControl"]["wells-save-freq"]["type"] = OT_INT;
    jsonBase["SignalProcessingBlockControl"]["wells-save-freq"]["value"] = -1;
    jsonBase["SignalProcessingBlockControl"]["wells-save-freq"]["min"] = "";
    jsonBase["SignalProcessingBlockControl"]["wells-save-freq"]["max"] = "";
    jsonBase["SignalProcessingBlockControl"]["wells-save-flow"]["type"] = OT_INT;
    jsonBase["SignalProcessingBlockControl"]["wells-save-flow"]["value"] = 60;
    jsonBase["SignalProcessingBlockControl"]["wells-save-flow"]["min"] = 20;
    jsonBase["SignalProcessingBlockControl"]["wells-save-flow"]["max"] = 2000;
    jsonBase["SignalProcessingBlockControl"]["restart-from"]["type"] = OT_STRING;
    jsonBase["SignalProcessingBlockControl"]["restart-from"]["value"] = "";
    jsonBase["SignalProcessingBlockControl"]["restart-from"]["min"] = "";
    jsonBase["SignalProcessingBlockControl"]["restart-from"]["max"] = "";
    jsonBase["SignalProcessingBlockControl"]["restart-next"]["type"] = OT_STRING;
    jsonBase["SignalProcessingBlockControl"]["restart-next"]["value"] = "";
    jsonBase["SignalProcessingBlockControl"]["restart-next"]["min"] = "";
    jsonBase["SignalProcessingBlockControl"]["restart-next"]["max"] = "";
    jsonBase["SignalProcessingBlockControl"]["restart-check"]["type"] = OT_BOOL;
    jsonBase["SignalProcessingBlockControl"]["restart-check"]["value"] = true;
    jsonBase["SignalProcessingBlockControl"]["restart-check"]["min"] = "";
    jsonBase["SignalProcessingBlockControl"]["restart-check"]["max"] = "";
    jsonBase["SignalProcessingBlockControl"]["numcputhreads"]["type"] = OT_INT;
    jsonBase["SignalProcessingBlockControl"]["numcputhreads"]["value"] = 0;
    jsonBase["SignalProcessingBlockControl"]["numcputhreads"]["min"] = "";
    jsonBase["SignalProcessingBlockControl"]["numcputhreads"]["max"] = "";
    jsonBase["SignalProcessingBlockControl"]["bkg-bfmask-update"]["type"] = OT_BOOL;
    jsonBase["SignalProcessingBlockControl"]["bkg-bfmask-update"]["value"] = true;
    jsonBase["SignalProcessingBlockControl"]["bkg-bfmask-update"]["min"] = "";
    jsonBase["SignalProcessingBlockControl"]["bkg-bfmask-update"]["max"] = "";
    jsonBase["SignalProcessingBlockControl"]["sigproc-compute-flow"]["type"] = OT_STRING;
    jsonBase["SignalProcessingBlockControl"]["sigproc-compute-flow"]["value"] = "";
    jsonBase["SignalProcessingBlockControl"]["sigproc-compute-flow"]["min"] = "";
    jsonBase["SignalProcessingBlockControl"]["sigproc-compute-flow"]["max"] = "";
    jsonBase["TraceControl"]["bkg-use-duds"]["type"] = OT_BOOL;
    jsonBase["TraceControl"]["bkg-use-duds"]["value"] = false;
    jsonBase["TraceControl"]["bkg-use-duds"]["min"] = "";
    jsonBase["TraceControl"]["bkg-use-duds"]["max"] = "";
    jsonBase["TraceControl"]["bkg-empty-well-normalization"]["type"] = OT_BOOL;
    jsonBase["TraceControl"]["bkg-empty-well-normalization"]["value"] = false;
    jsonBase["TraceControl"]["bkg-empty-well-normalization"]["min"] = "";
    jsonBase["TraceControl"]["bkg-empty-well-normalization"]["max"] = "";
    jsonBase["TraceControl"]["trim-ref-trace"]["type"] = OT_STRING;
    jsonBase["TraceControl"]["trim-ref-trace"]["value"] = "";
    jsonBase["TraceControl"]["trim-ref-trace"]["min"] = "";
    jsonBase["TraceControl"]["trim-ref-trace"]["max"] = "";
    jsonBase["DebugMe"]["bkg-debug-param"]["type"] = OT_INT;
    jsonBase["DebugMe"]["bkg-debug-param"]["value"] = 1;
    jsonBase["DebugMe"]["bkg-debug-param"]["min"] = "";
    jsonBase["DebugMe"]["bkg-debug-param"]["max"] = "";
    jsonBase["DebugMe"]["bkg-debug-nsamples"]["type"] = OT_INT;
    jsonBase["DebugMe"]["bkg-debug-nsamples"]["value"] = 9;
    jsonBase["DebugMe"]["bkg-debug-nsamples"]["min"] = "";
    jsonBase["DebugMe"]["bkg-debug-nsamples"]["max"] = "";
    jsonBase["DebugMe"]["bkg-debug-region"]["type"] = OT_VECTOR_INT;
    jsonBase["DebugMe"]["bkg-debug-region"]["value"] = Value(arrayValue);
    jsonBase["DebugMe"]["bkg-debug-region"]["value"].append(-1);
    jsonBase["DebugMe"]["bkg-debug-region"]["value"].append(-1);
    jsonBase["DebugMe"]["bkg-debug-region"]["min"] = "";
    jsonBase["DebugMe"]["bkg-debug-region"]["max"] = "";
    jsonBase["DebugMe"]["bkg-debug-trace-sse"]["type"] = OT_STRING;
    jsonBase["DebugMe"]["bkg-debug-trace-sse"]["value"] = "";
    jsonBase["DebugMe"]["bkg-debug-trace-sse"]["min"] = "";
    jsonBase["DebugMe"]["bkg-debug-trace-sse"]["max"] = "";
    jsonBase["DebugMe"]["bkg-debug-trace-rcflow"]["type"] = OT_STRING;
    jsonBase["DebugMe"]["bkg-debug-trace-rcflow"]["value"] = "";
    jsonBase["DebugMe"]["bkg-debug-trace-rcflow"]["min"] = "";
    jsonBase["DebugMe"]["bkg-debug-trace-rcflow"]["max"] = "";
    jsonBase["DebugMe"]["bkg-debug-trace-xyflow"]["type"] = OT_STRING;
    jsonBase["DebugMe"]["bkg-debug-trace-xyflow"]["value"] = "";
    jsonBase["DebugMe"]["bkg-debug-trace-xyflow"]["min"] = "";
    jsonBase["DebugMe"]["bkg-debug-trace-xyflow"]["max"] = "";
    jsonBase["DebugMe"]["bkg-dbg-trace"]["type"] = OT_VECTOR_INT;
    jsonBase["DebugMe"]["bkg-dbg-trace"]["value"] = Value(arrayValue);
    jsonBase["DebugMe"]["bkg-dbg-trace"]["min"] = "";
    jsonBase["DebugMe"]["bkg-dbg-trace"]["max"] = "";
    jsonBase["DebugMe"]["debug-bead-only"]["type"] = OT_BOOL;
    jsonBase["DebugMe"]["debug-bead-only"]["value"] = true;
    jsonBase["DebugMe"]["debug-bead-only"]["min"] = "";
    jsonBase["DebugMe"]["debug-bead-only"]["max"] = "";
    jsonBase["DebugMe"]["region-vfrc-debug"]["type"] = OT_BOOL;
    jsonBase["DebugMe"]["region-vfrc-debug"]["value"] = false;
    jsonBase["DebugMe"]["region-vfrc-debug"]["min"] = "";
    jsonBase["DebugMe"]["region-vfrc-debug"]["max"] = "";
    jsonBase["DebugMe"]["bkg-debug-files"]["type"] = OT_BOOL;
    jsonBase["DebugMe"]["bkg-debug-files"]["value"] = false;
    jsonBase["DebugMe"]["bkg-debug-files"]["min"] = "";
    jsonBase["DebugMe"]["bkg-debug-files"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-type"]["type"] = OT_STRING;
    jsonBase["BeadfindControlOpts"]["beadfind-type"]["value"] = "differential";
    jsonBase["BeadfindControlOpts"]["beadfind-type"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-type"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["use-beadmask"]["type"] = OT_STRING;
    jsonBase["BeadfindControlOpts"]["use-beadmask"]["value"] = "";
    jsonBase["BeadfindControlOpts"]["use-beadmask"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["use-beadmask"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["exclusion-mask"]["type"] = OT_STRING;
    jsonBase["BeadfindControlOpts"]["exclusion-mask"]["value"] = "";
    jsonBase["BeadfindControlOpts"]["exclusion-mask"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["exclusion-mask"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadmask-categorized"]["type"] = OT_BOOL;
    jsonBase["BeadfindControlOpts"]["beadmask-categorized"]["value"] = false;
    jsonBase["BeadfindControlOpts"]["beadmask-categorized"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadmask-categorized"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-basis"]["type"] = OT_STRING;
    jsonBase["BeadfindControlOpts"]["beadfind-basis"]["value"] = "naoh";
    jsonBase["BeadfindControlOpts"]["beadfind-basis"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-basis"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-dat"]["type"] = OT_STRING;
    jsonBase["BeadfindControlOpts"]["beadfind-dat"]["value"] = "beadfind_pre_0003.dat";
    jsonBase["BeadfindControlOpts"]["beadfind-dat"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-dat"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-bgdat"]["type"] = OT_STRING;
    jsonBase["BeadfindControlOpts"]["beadfind-bgdat"]["value"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-bgdat"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-bgdat"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-sdasbf"]["type"] = OT_BOOL;
    jsonBase["BeadfindControlOpts"]["beadfind-sdasbf"]["value"] = true;
    jsonBase["BeadfindControlOpts"]["beadfind-sdasbf"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-sdasbf"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-bfmult"]["type"] = OT_DOUBLE;
    jsonBase["BeadfindControlOpts"]["beadfind-bfmult"]["value"] = 1.0;
    jsonBase["BeadfindControlOpts"]["beadfind-bfmult"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-bfmult"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-minlive"]["type"] = OT_DOUBLE;
    jsonBase["BeadfindControlOpts"]["beadfind-minlive"]["value"] = 0.0001;
    jsonBase["BeadfindControlOpts"]["beadfind-minlive"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-minlive"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-filt-noisy-col"]["type"] = OT_STRING;
    jsonBase["BeadfindControlOpts"]["beadfind-filt-noisy-col"]["value"] = "none";
    jsonBase["BeadfindControlOpts"]["beadfind-filt-noisy-col"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-filt-noisy-col"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-minlivesnr"]["type"] = OT_DOUBLE;
    jsonBase["BeadfindControlOpts"]["beadfind-minlivesnr"]["value"] = 4.0;
    jsonBase["BeadfindControlOpts"]["beadfind-minlivesnr"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-minlivesnr"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-min-tf-snr"]["type"] = OT_DOUBLE;
    jsonBase["BeadfindControlOpts"]["beadfind-min-tf-snr"]["value"] = 7.0;
    jsonBase["BeadfindControlOpts"]["beadfind-min-tf-snr"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-min-tf-snr"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-tf-min-peak"]["type"] = OT_DOUBLE;
    jsonBase["BeadfindControlOpts"]["beadfind-tf-min-peak"]["value"] = 40.0;
    jsonBase["BeadfindControlOpts"]["beadfind-tf-min-peak"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-tf-min-peak"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-lib-min-peak"]["type"] = OT_DOUBLE;
    jsonBase["BeadfindControlOpts"]["beadfind-lib-min-peak"]["value"] = 10.0;
    jsonBase["BeadfindControlOpts"]["beadfind-lib-min-peak"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-lib-min-peak"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-lib-filt"]["type"] = OT_DOUBLE;
    jsonBase["BeadfindControlOpts"]["beadfind-lib-filt"]["value"] = 1.0;
    jsonBase["BeadfindControlOpts"]["beadfind-lib-filt"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-lib-filt"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-tf-filt"]["type"] = OT_DOUBLE;
    jsonBase["BeadfindControlOpts"]["beadfind-tf-filt"]["value"] = 1.0;
    jsonBase["BeadfindControlOpts"]["beadfind-tf-filt"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-tf-filt"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-skip-sd-recover"]["type"] = OT_INT;
    jsonBase["BeadfindControlOpts"]["beadfind-skip-sd-recover"]["value"] = 1;
    jsonBase["BeadfindControlOpts"]["beadfind-skip-sd-recover"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-skip-sd-recover"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-sep-ref"]["type"] = OT_BOOL;
    jsonBase["BeadfindControlOpts"]["beadfind-sep-ref"]["value"] = false;
    jsonBase["BeadfindControlOpts"]["beadfind-sep-ref"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-sep-ref"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-smooth-trace"]["type"] = OT_BOOL;
    jsonBase["BeadfindControlOpts"]["beadfind-smooth-trace"]["value"] = false;
    jsonBase["BeadfindControlOpts"]["beadfind-smooth-trace"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-smooth-trace"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-diagnostics"]["type"] = OT_INT;
    jsonBase["BeadfindControlOpts"]["beadfind-diagnostics"]["value"] = 2;
    jsonBase["BeadfindControlOpts"]["beadfind-diagnostics"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-diagnostics"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-gain-correction"]["type"] = OT_BOOL;
    jsonBase["BeadfindControlOpts"]["beadfind-gain-correction"]["value"] = true;
    jsonBase["BeadfindControlOpts"]["beadfind-gain-correction"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-gain-correction"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["datacollect-gain-correction"]["type"] = OT_BOOL;
    jsonBase["BeadfindControlOpts"]["datacollect-gain-correction"]["value"] = false;
    jsonBase["BeadfindControlOpts"]["beadfind-blob-filter"]["type"] = OT_BOOL;
    jsonBase["BeadfindControlOpts"]["beadfind-blob-filter"]["value"] = false;
    jsonBase["BeadfindControlOpts"]["beadfind-blob-filter"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-blob-filter"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-predict-start"]["type"] = OT_INT;
    jsonBase["BeadfindControlOpts"]["beadfind-predict-start"]["value"] = -1;
    jsonBase["BeadfindControlOpts"]["beadfind-predict-start"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-predict-start"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-predict-end"]["type"] = OT_INT;
    jsonBase["BeadfindControlOpts"]["beadfind-predict-end"]["value"] = -1;
    jsonBase["BeadfindControlOpts"]["beadfind-predict-end"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-predict-end"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-sig-ref-type"]["type"] = OT_INT;
    jsonBase["BeadfindControlOpts"]["beadfind-sig-ref-type"]["value"] = -1;
    jsonBase["BeadfindControlOpts"]["beadfind-sig-ref-type"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-sig-ref-type"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-zero-flows"]["type"] = OT_STRING;
    jsonBase["BeadfindControlOpts"]["beadfind-zero-flows"]["value"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-zero-flows"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-zero-flows"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-num-threads"]["type"] = OT_INT;
    jsonBase["BeadfindControlOpts"]["beadfind-num-threads"]["value"] =-1 ;
    jsonBase["BeadfindControlOpts"]["beadfind-num-threads"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-num-threads"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["bfold"]["type"] = OT_BOOL;
    jsonBase["BeadfindControlOpts"]["bfold"]["value"] = true;
    jsonBase["BeadfindControlOpts"]["bfold"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["bfold"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["noduds"]["type"] = OT_BOOL;
    jsonBase["BeadfindControlOpts"]["noduds"]["value"] = false;
    jsonBase["BeadfindControlOpts"]["noduds"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["noduds"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfindfile"]["type"] = OT_STRING;
    jsonBase["BeadfindControlOpts"]["beadfindfile"]["value"] = "";
    jsonBase["BeadfindControlOpts"]["beadfindfile"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfindfile"]["max"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-mesh-step"]["type"] = OT_VECTOR_INT;
    jsonBase["BeadfindControlOpts"]["beadfind-mesh-step"]["value"] = Value(arrayValue);
    jsonBase["BeadfindControlOpts"]["beadfind-mesh-step"]["min"] = "";
    jsonBase["BeadfindControlOpts"]["beadfind-mesh-step"]["max"] = "";
    jsonBase["ImageControlOpts"]["pca-test"]["type"] = OT_STRING;
    jsonBase["ImageControlOpts"]["pca-test"]["value"] = "";
    jsonBase["ImageControlOpts"]["pca-test"]["min"] = "";
    jsonBase["ImageControlOpts"]["pca-test"]["max"] = "";
    jsonBase["ImageControlOpts"]["acqPrefix"]["type"] = OT_STRING;
    jsonBase["ImageControlOpts"]["acqPrefix"]["value"] = "acq_";
    jsonBase["ImageControlOpts"]["acqPrefix"]["min"] = "";
    jsonBase["ImageControlOpts"]["acqPrefix"]["max"] = "";
    jsonBase["ImageControlOpts"]["dat-postfix"]["type"] = OT_STRING;
    jsonBase["ImageControlOpts"]["dat-postfix"]["value"] = "dat";
    jsonBase["ImageControlOpts"]["dat-postfix"]["min"] = "";
    jsonBase["ImageControlOpts"]["dat-postfix"]["max"] = "";
    jsonBase["ImageControlOpts"]["col-flicker-correct"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["col-flicker-correct"]["value"] = false;
    jsonBase["ImageControlOpts"]["col-flicker-correct"]["min"] = "";
    jsonBase["ImageControlOpts"]["col-flicker-correct"]["max"] = "";
    jsonBase["ImageControlOpts"]["col-flicker-correct-verbose"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["col-flicker-correct-verbose"]["value"] = false;
    jsonBase["ImageControlOpts"]["col-flicker-correct-verbose"]["min"] = "";
    jsonBase["ImageControlOpts"]["col-flicker-correct-verbose"]["max"] = "";
    jsonBase["ImageControlOpts"]["col-flicker-correct-aggressive"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["col-flicker-correct-aggressive"]["value"] = false;
    jsonBase["ImageControlOpts"]["col-flicker-correct-aggressive"]["min"] = "";
    jsonBase["ImageControlOpts"]["col-flicker-correct-aggressive"]["max"] = "";
    jsonBase["ImageControlOpts"]["img-gain-correct"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["img-gain-correct"]["value"] = false;
    jsonBase["ImageControlOpts"]["img-gain-correct"]["min"] = "";
    jsonBase["ImageControlOpts"]["img-gain-correct"]["max"] = "";
    jsonBase["ImageControlOpts"]["smoothing-file"]["type"] = OT_STRING;
    jsonBase["ImageControlOpts"]["smoothing-file"]["value"] = "";
    jsonBase["ImageControlOpts"]["smoothing-file"]["min"] = "";
    jsonBase["ImageControlOpts"]["smoothing-file"]["max"] = "";
    jsonBase["ImageControlOpts"]["smoothing"]["type"] = OT_STRING;
    jsonBase["ImageControlOpts"]["smoothing"]["value"] = "";
    jsonBase["ImageControlOpts"]["smoothing"]["min"] = "";
    jsonBase["ImageControlOpts"]["smoothing"]["max"] = "";
    jsonBase["ImageControlOpts"]["ignore-checksum-errors"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["ignore-checksum-errors"]["value"] = false;
    jsonBase["ImageControlOpts"]["ignore-checksum-errors"]["min"] = "";
    jsonBase["ImageControlOpts"]["ignore-checksum-errors"]["max"] = "";
    jsonBase["ImageControlOpts"]["ignore-checksum-errors-1frame"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["ignore-checksum-errors-1frame"]["value"] = false;
    jsonBase["ImageControlOpts"]["ignore-checksum-errors-1frame"]["min"] = "";
    jsonBase["ImageControlOpts"]["ignore-checksum-errors-1frame"]["max"] = "";
    jsonBase["ImageControlOpts"]["output-pinned-wells"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["output-pinned-wells"]["value"] = false;
    jsonBase["ImageControlOpts"]["output-pinned-wells"]["min"] = "";
    jsonBase["ImageControlOpts"]["output-pinned-wells"]["max"] = "";
    jsonBase["ImageControlOpts"]["flowtimeoffset"]["type"] = OT_INT;
    jsonBase["ImageControlOpts"]["flowtimeoffset"]["value"] = 1000;
    jsonBase["ImageControlOpts"]["flowtimeoffset"]["min"] = "";
    jsonBase["ImageControlOpts"]["flowtimeoffset"]["max"] = "";
    jsonBase["ImageControlOpts"]["nn-subtract-empties"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["nn-subtract-empties"]["value"] = false;
    jsonBase["ImageControlOpts"]["nn-subtract-empties"]["min"] = "";
    jsonBase["ImageControlOpts"]["nn-subtract-empties"]["max"] = "";
    jsonBase["ImageControlOpts"]["nnmask"]["type"] = OT_VECTOR_INT;
    jsonBase["ImageControlOpts"]["nnmask"]["value"] = Value(arrayValue);
    jsonBase["ImageControlOpts"]["nnmask"]["value"].append(1);
    jsonBase["ImageControlOpts"]["nnmask"]["value"].append(3);
    jsonBase["ImageControlOpts"]["nnmask"]["min"] = "";
    jsonBase["ImageControlOpts"]["nnmask"]["max"] = "";
    jsonBase["ImageControlOpts"]["nnmaskwh"]["type"] = OT_VECTOR_INT;
    jsonBase["ImageControlOpts"]["nnmaskwh"]["value"] = Value(arrayValue);
    jsonBase["ImageControlOpts"]["nnmaskwh"]["value"].append(1);
    jsonBase["ImageControlOpts"]["nnmaskwh"]["value"].append(1);
    jsonBase["ImageControlOpts"]["nnmaskwh"]["value"].append(12);
    jsonBase["ImageControlOpts"]["nnmaskwh"]["value"].append(8);
    jsonBase["ImageControlOpts"]["nnmaskwh"]["min"] = "";
    jsonBase["ImageControlOpts"]["nnmaskwh"]["max"] = "";
    jsonBase["ImageControlOpts"]["hilowfilter"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["hilowfilter"]["value"] = false;
    jsonBase["ImageControlOpts"]["hilowfilter"]["min"] = "";
    jsonBase["ImageControlOpts"]["hilowfilter"]["max"] = "";
    jsonBase["ImageControlOpts"]["total-timeout"]["type"] = OT_INT;
    jsonBase["ImageControlOpts"]["total-timeout"]["value"] = 0;
    jsonBase["ImageControlOpts"]["total-timeout"]["min"] = "";
    jsonBase["ImageControlOpts"]["total-timeout"]["max"] = "";
    jsonBase["ImageControlOpts"]["readaheaddat"]["type"] = OT_INT;
    jsonBase["ImageControlOpts"]["readaheaddat"]["value"] = 0;
    jsonBase["ImageControlOpts"]["readaheaddat"]["min"] = "";
    jsonBase["ImageControlOpts"]["readaheaddat"]["max"] = "";
    jsonBase["ImageControlOpts"]["no-threaded-file-access"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["no-threaded-file-access"]["value"] = false;
    jsonBase["ImageControlOpts"]["no-threaded-file-access"]["min"] = "";
    jsonBase["ImageControlOpts"]["no-threaded-file-access"]["max"] = "";
    jsonBase["ImageControlOpts"]["frames"]["type"] = OT_INT;
    jsonBase["ImageControlOpts"]["frames"]["value"] = -1;
    jsonBase["ImageControlOpts"]["frames"]["min"] = "";
    jsonBase["ImageControlOpts"]["frames"]["max"] = "";
    jsonBase["ImageControlOpts"]["col-doubles-xtalk-correct"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["col-doubles-xtalk-correct"]["value"] = false;
    jsonBase["ImageControlOpts"]["col-doubles-xtalk-correct"]["min"] = "";
    jsonBase["ImageControlOpts"]["col-doubles-xtalk-correct"]["max"] = "";
    jsonBase["ImageControlOpts"]["pair-xtalk-coeff"]["type"] = OT_DOUBLE;
    jsonBase["ImageControlOpts"]["pair-xtalk-coeff"]["value"] = 0.0;
    jsonBase["ImageControlOpts"]["pair-xtalk-coeff"]["min"] = "";
    jsonBase["ImageControlOpts"]["pair-xtalk-coeff"]["max"] = "";
    jsonBase["ImageControlOpts"]["fluid-potential-correct"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["fluid-potential-correct"]["value"] = false;
    jsonBase["ImageControlOpts"]["fluid-potential-correct"]["min"] = "";
    jsonBase["ImageControlOpts"]["fluid-potential-correct"]["max"] = "";
    jsonBase["ImageControlOpts"]["fluid-potential-threshold"]["type"] = OT_DOUBLE;
    jsonBase["ImageControlOpts"]["fluid-potential-threshold"]["value"] = 1.0;
    jsonBase["ImageControlOpts"]["fluid-potential-threshold"]["min"] = "";
    jsonBase["ImageControlOpts"]["fluid-potential-threshold"]["max"] = "";
    jsonBase["ImageControlOpts"]["corr-noise-correct"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["corr-noise-correct"]["value"] = true;
    jsonBase["ImageControlOpts"]["corr-noise-correct"]["min"] = "";
    jsonBase["ImageControlOpts"]["corr-noise-correct"]["max"] = "";
    jsonBase["ImageControlOpts"]["mask-datacollect-exclude-regions"]["type"] = OT_BOOL;
    jsonBase["ImageControlOpts"]["mask-datacollect-exclude-regions"]["value"] = false;
    jsonBase["ImageControlOpts"]["mask-datacollect-exclude-regions"]["min"] = "";
    jsonBase["ImageControlOpts"]["mask-datacollect-exclude-regions"]["max"] = "";
    jsonBase["ModuleControlOpts"]["bfonly"]["type"] = OT_BOOL;
    jsonBase["ModuleControlOpts"]["bfonly"]["value"] = false;
    jsonBase["ModuleControlOpts"]["bfonly"]["min"] = "";
    jsonBase["ModuleControlOpts"]["bfonly"]["max"] = "";
    jsonBase["ModuleControlOpts"]["from-beadfind"]["type"] = OT_BOOL;
    jsonBase["ModuleControlOpts"]["from-beadfind"]["value"] = false;
    jsonBase["ModuleControlOpts"]["from-beadfind"]["min"] = "";
    jsonBase["ModuleControlOpts"]["from-beadfind"]["max"] = "";
    jsonBase["ModuleControlOpts"]["pass-tau"]["type"] = OT_BOOL;
    jsonBase["ModuleControlOpts"]["pass-tau"]["value"] = true;
    jsonBase["ModuleControlOpts"]["pass-tau"]["min"] = "";
    jsonBase["ModuleControlOpts"]["pass-tau"]["max"] = "";
    jsonBase["SpatialContext"]["region-size"]["type"] = OT_VECTOR_INT;
    jsonBase["SpatialContext"]["region-size"]["value"] = Value(arrayValue);
    jsonBase["SpatialContext"]["region-size"]["min"] = "";
    jsonBase["SpatialContext"]["region-size"]["max"] = "";
    jsonBase["SpatialContext"]["cropped"]["type"] = OT_VECTOR_INT;
    jsonBase["SpatialContext"]["cropped"]["value"] = Value(arrayValue);
    jsonBase["SpatialContext"]["cropped"]["min"] = "";
    jsonBase["SpatialContext"]["cropped"]["max"] = "";
    jsonBase["SpatialContext"]["analysis-region"]["type"] = OT_VECTOR_INT;
    jsonBase["SpatialContext"]["analysis-region"]["value"] = Value(arrayValue);
    jsonBase["SpatialContext"]["analysis-region"]["min"] = "";
    jsonBase["SpatialContext"]["analysis-region"]["max"] = "";
    jsonBase["SpatialContext"]["cropped-region-origin"]["type"] = OT_VECTOR_INT;
    jsonBase["SpatialContext"]["cropped-region-origin"]["value"] = Value(arrayValue);
    jsonBase["SpatialContext"]["cropped-region-origin"]["min"] = "";
    jsonBase["SpatialContext"]["cropped-region-origin"]["max"] = "";
    jsonBase["FlowContext"]["flow-order"]["type"] = OT_STRING;
    jsonBase["FlowContext"]["flow-order"]["value"] = "";
    jsonBase["FlowContext"]["flow-order"]["min"] = "";
    jsonBase["FlowContext"]["flow-order"]["max"] = "";
    jsonBase["FlowContext"]["flowlimit"]["type"] = OT_INT;
    jsonBase["FlowContext"]["flowlimit"]["value"] = -1;
    jsonBase["FlowContext"]["flowlimit"]["min"] = "";
    jsonBase["FlowContext"]["flowlimit"]["max"] = "";
    jsonBase["FlowContext"]["start-flow-plus-interval"]["type"] = OT_VECTOR_INT;
    jsonBase["FlowContext"]["start-flow-plus-interval"]["value"] = Value(arrayValue);
    jsonBase["FlowContext"]["start-flow-plus-interval"]["value"].append(0);
    jsonBase["FlowContext"]["start-flow-plus-interval"]["value"].append(0);
    jsonBase["FlowContext"]["start-flow-plus-interval"]["min"] = "";
    jsonBase["FlowContext"]["start-flow-plus-interval"]["max"] = "";
    jsonBase["KeyContext"]["librarykey"]["type"] = OT_STRING;
    jsonBase["KeyContext"]["librarykey"]["value"] = "";
    jsonBase["KeyContext"]["librarykey"]["min"] = "";
    jsonBase["KeyContext"]["librarykey"]["max"] = "";
    jsonBase["KeyContext"]["tfkey"]["type"] = OT_STRING;
    jsonBase["KeyContext"]["tfkey"]["value"] = "";
    jsonBase["KeyContext"]["tfkey"]["min"] = "";
    jsonBase["KeyContext"]["tfkey"]["max"] = "";
    jsonBase["ObsoleteOpts"]["nuc-correct"]["type"] = OT_INT;
    jsonBase["ObsoleteOpts"]["nuc-correct"]["value"] = 0;
    jsonBase["ObsoleteOpts"]["nuc-correct"]["min"] = "";
    jsonBase["ObsoleteOpts"]["nuc-correct"]["max"] = "";
    jsonBase["ObsoleteOpts"]["use-pinned"]["type"] = OT_BOOL;
    jsonBase["ObsoleteOpts"]["use-pinned"]["value"] = false;
    jsonBase["ObsoleteOpts"]["use-pinned"]["min"] = "";
    jsonBase["ObsoleteOpts"]["use-pinned"]["max"] = "";
    jsonBase["ObsoleteOpts"]["forcenn"]["type"] = OT_INT;
    jsonBase["ObsoleteOpts"]["forcenn"]["value"] = 0;
    jsonBase["ObsoleteOpts"]["forcenn"]["min"] = "";
    jsonBase["ObsoleteOpts"]["forcenn"]["max"] = "";
    jsonBase["SystemContext"]["local-wells-file"]["type"] = OT_BOOL;
    jsonBase["SystemContext"]["local-wells-file"]["value"] = false;
    jsonBase["SystemContext"]["local-wells-file"]["min"] = "";
    jsonBase["SystemContext"]["local-wells-file"]["max"] = "";
    jsonBase["SystemContext"]["well-stat-file"]["type"] = OT_STRING;
    jsonBase["SystemContext"]["well-stat-file"]["value"] = "";
    jsonBase["SystemContext"]["well-stat-file"]["min"] = "";
    jsonBase["SystemContext"]["well-stat-file"]["max"] = "";
    jsonBase["SystemContext"]["stack-dump-file"]["type"] = OT_STRING;
    jsonBase["SystemContext"]["stack-dump-file"]["value"] = "";
    jsonBase["SystemContext"]["stack-dump-file"]["min"] = "";
    jsonBase["SystemContext"]["stack-dump-file"]["max"] = "";
    jsonBase["SystemContext"]["wells-format"]["type"] = OT_STRING;
    jsonBase["SystemContext"]["wells-format"]["value"] = "hdf5";
    jsonBase["SystemContext"]["wells-format"]["min"] = "";
    jsonBase["SystemContext"]["wells-format"]["max"] = "";
    jsonBase["SystemContext"]["output-dir"]["type"] = OT_STRING;
    jsonBase["SystemContext"]["output-dir"]["value"] = "";
    jsonBase["SystemContext"]["output-dir"]["min"] = "";
    jsonBase["SystemContext"]["output-dir"]["max"] = "";
    jsonBase["SystemContext"]["explog-path"]["type"] = OT_STRING;
    jsonBase["SystemContext"]["explog-path"]["value"] = "";
    jsonBase["SystemContext"]["explog-path"]["min"] = "";
    jsonBase["SystemContext"]["explog-path"]["max"] = "";
    jsonBase["SystemContext"]["no-subdir"]["type"] = OT_BOOL;
    jsonBase["SystemContext"]["no-subdir"]["value"] = true;
    jsonBase["SystemContext"]["no-subdir"]["min"] = "";
    jsonBase["SystemContext"]["no-subdir"]["max"] = "";
    jsonBase["SystemContext"]["dat-source-directory"]["type"] = OT_STRING;
    jsonBase["SystemContext"]["dat-source-directory"]["value"] = "";
    jsonBase["SystemContext"]["dat-source-directory"]["min"] = "";
    jsonBase["SystemContext"]["dat-source-directory"]["max"] = "";
    jsonBase["SystemContext"]["region-list"]["type"] = OT_VECTOR_INT;
    jsonBase["SystemContext"]["region-list"]["value"] = Value(arrayValue);
    jsonBase["SystemContext"]["region-list"]["min"] = "";
    jsonBase["SystemContext"]["region-list"]["max"] = "";
    jsonBase["SystemContext"]["wells-save-queue-size"]["type"] = OT_INT;
    jsonBase["SystemContext"]["wells-save-queue-size"]["value"] = 0;
    jsonBase["SystemContext"]["wells-save-queue-size"]["min"] = "";
    jsonBase["SystemContext"]["wells-save-queue-size"]["max"] = "";
    jsonBase["SystemContext"]["wells-save-as-ushort"]["type"] = OT_BOOL;
    jsonBase["SystemContext"]["wells-save-as-ushort"]["value"] = true;
    jsonBase["SystemContext"]["wells-save-as-ushort"]["min"] = "";
    jsonBase["SystemContext"]["wells-save-as-ushort"]["max"] = "";
    jsonBase["SystemContext"]["wells-convert-low"]["type"] = OT_DOUBLE;
    jsonBase["SystemContext"]["wells-convert-low"]["value"] = -5.0;
    jsonBase["SystemContext"]["wells-convert-low"]["min"] = "";
    jsonBase["SystemContext"]["wells-convert-low"]["max"] = "";
    jsonBase["SystemContext"]["wells-convert-high"]["type"] = OT_DOUBLE;
    jsonBase["SystemContext"]["wells-convert-high"]["value"] = 28.0;
    jsonBase["SystemContext"]["wells-convert-high"]["min"] = "";
    jsonBase["SystemContext"]["wells-convert-high"]["max"] = "";
    jsonBase["SystemContext"]["wells-save-number-copies"]["type"] = OT_BOOL;
    jsonBase["SystemContext"]["wells-save-number-copies"]["value"] = true;
    jsonBase["SystemContext"]["wells-save-number-copies"]["min"] = "";
    jsonBase["SystemContext"]["wells-save-number-copies"]["max"] = "";
    jsonBase["SystemContext"]["wells-convert-with-copies"]["type"] = OT_BOOL;
    jsonBase["SystemContext"]["wells-convert-with-copies"]["value"] = true;
    jsonBase["SystemContext"]["wells-convert-with-copies"]["min"] = "";
    jsonBase["SystemContext"]["wells-convert-with-copies"]["max"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["gopt"]["type"] = OT_STRING;
    jsonBase["GlobalDefaultsForBkgModel"]["gopt"]["value"] = "default";
    jsonBase["GlobalDefaultsForBkgModel"]["gopt"]["min"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["gopt"]["max"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["bkg-dont-emphasize-by-compression"]["type"] = OT_BOOL;
    jsonBase["GlobalDefaultsForBkgModel"]["bkg-dont-emphasize-by-compression"]["value"] = false;
    jsonBase["GlobalDefaultsForBkgModel"]["bkg-dont-emphasize-by-compression"]["min"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["bkg-dont-emphasize-by-compression"]["max"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["xtalk"]["type"] = OT_STRING;
    jsonBase["GlobalDefaultsForBkgModel"]["xtalk"]["value"] = "disable";
    jsonBase["GlobalDefaultsForBkgModel"]["xtalk"]["min"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["xtalk"]["max"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["bkg-well-xtalk-name"]["type"] = OT_STRING;
    jsonBase["GlobalDefaultsForBkgModel"]["bkg-well-xtalk-name"]["value"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["bkg-well-xtalk-name"]["min"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["bkg-well-xtalk-name"]["max"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["barcode-spec-file"]["type"] = OT_STRING;
    jsonBase["GlobalDefaultsForBkgModel"]["barcode-spec-file"]["value"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["barcode-spec-file"]["min"] = "";
    jsonBase["GlobalDefaultsForBkgModel"]["barcode-spec-file"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-kmult-adj-low-hi"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["bkg-kmult-adj-low-hi"]["value"] = 2.0;
    jsonBase["LocalSigProcControl"]["bkg-kmult-adj-low-hi"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-kmult-adj-low-hi"]["max"] = "";
    jsonBase["LocalSigProcControl"]["kmult-low-limit"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["kmult-low-limit"]["value"] = 0.65;
    jsonBase["LocalSigProcControl"]["kmult-low-limit"]["min"] = "";
    jsonBase["LocalSigProcControl"]["kmult-low-limit"]["max"] = "";
    jsonBase["LocalSigProcControl"]["kmult-hi-limit"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["kmult-hi-limit"]["value"] = 1.75;
    jsonBase["LocalSigProcControl"]["kmult-hi-limit"]["min"] = "";
    jsonBase["LocalSigProcControl"]["kmult-hi-limit"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-copy-stringency"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["bkg-copy-stringency"]["value"] = 1.0;
    jsonBase["LocalSigProcControl"]["bkg-copy-stringency"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-copy-stringency"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-min-sampled-beads"]["type"] = OT_INT;
    jsonBase["LocalSigProcControl"]["bkg-min-sampled-beads"]["value"] = 100;
    jsonBase["LocalSigProcControl"]["bkg-min-sampled-beads"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-min-sampled-beads"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-max-rank-beads"]["type"] = OT_INT;
    jsonBase["LocalSigProcControl"]["bkg-max-rank-beads"]["value"] = 100000;
    jsonBase["LocalSigProcControl"]["bkg-max-rank-beads"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-max-rank-beads"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-post-key-train"]["type"] = OT_INT;
    jsonBase["LocalSigProcControl"]["bkg-post-key-train"]["value"] = 2;
    jsonBase["LocalSigProcControl"]["bkg-post-key-train"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-post-key-train"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-post-key-step"]["type"] = OT_INT;
    jsonBase["LocalSigProcControl"]["bkg-post-key-step"]["value"] = 2;
    jsonBase["LocalSigProcControl"]["bkg-post-key-step"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-post-key-step"]["max"] = "";
    jsonBase["LocalSigProcControl"]["clonal-filter-bkgmodel"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["clonal-filter-bkgmodel"]["value"] = true;
    jsonBase["LocalSigProcControl"]["clonal-filter-bkgmodel"]["min"] = "";
    jsonBase["LocalSigProcControl"]["clonal-filter-bkgmodel"]["max"] = "";
    jsonBase["LocalSigProcControl"]["clonal-filter-bkgmodel"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["clonal-filter-bkgmodel"]["value"] = true;
    jsonBase["LocalSigProcControl"]["clonal-filter-bkgmodel"]["min"] = "";
    jsonBase["LocalSigProcControl"]["clonal-filter-bkgmodel"]["max"] = "";
    jsonBase["LocalSigProcControl"]["barcode-flag"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["barcode-flag"]["value"] = false;
    jsonBase["LocalSigProcControl"]["barcode-flag"]["min"] = "";
    jsonBase["LocalSigProcControl"]["barcode-flag"]["max"] = "";
    jsonBase["LocalSigProcControl"]["barcode-debug"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["barcode-debug"]["value"] = false;
    jsonBase["LocalSigProcControl"]["barcode-debug"]["min"] = "";
    jsonBase["LocalSigProcControl"]["barcode-debug"]["max"] = "";
    jsonBase["LocalSigProcControl"]["barcode-radius"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["barcode-radius"]["value"] = 0.75;
    jsonBase["LocalSigProcControl"]["barcode-radius"]["min"] = "";
    jsonBase["LocalSigProcControl"]["barcode-radius"]["max"] = "";
    jsonBase["LocalSigProcControl"]["barcode-tie"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["barcode-tie"]["value"] = 0.5;
    jsonBase["LocalSigProcControl"]["barcode-tie"]["min"] = "";
    jsonBase["LocalSigProcControl"]["barcode-tie"]["max"] = "";
    jsonBase["LocalSigProcControl"]["barcode-penalty"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["barcode-penalty"]["value"] = 2000.0;
    jsonBase["LocalSigProcControl"]["barcode-penalty"]["min"] = "";
    jsonBase["LocalSigProcControl"]["barcode-penalty"]["max"] = "";
    jsonBase["LocalSigProcControl"]["kmult-penalty"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["kmult-penalty"]["value"] = 100.0;
    jsonBase["LocalSigProcControl"]["kmult-penalty"]["min"] = "";
    jsonBase["LocalSigProcControl"]["kmult-penalty"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-use-proton-well-correction"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["bkg-use-proton-well-correction"]["value"] = false;
    jsonBase["LocalSigProcControl"]["bkg-use-proton-well-correction"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-use-proton-well-correction"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-per-flow-time-tracking"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["bkg-per-flow-time-tracking"]["value"] = false;
    jsonBase["LocalSigProcControl"]["bkg-per-flow-time-tracking"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-per-flow-time-tracking"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-fit"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-fit"]["value"] = false;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-fit"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-fit"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-adj"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-adj"]["value"] = true;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-adj"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-adj"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-tau-adj"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-tau-adj"]["value"] = true;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-tau-adj"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-tau-adj"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-limit"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-limit"]["value"] = 0.2;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-limit"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-limit"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-lower"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-lower"]["value"] = 10.0;
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-lower"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-exp-tail-bkg-lower"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-pca-dark-matter"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["bkg-pca-dark-matter"]["value"] = false;
    jsonBase["LocalSigProcControl"]["bkg-pca-dark-matter"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-pca-dark-matter"]["max"] = "";
    jsonBase["LocalSigProcControl"]["regional-sampling"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["regional-sampling"]["value"] = false;
    jsonBase["LocalSigProcControl"]["regional-sampling"]["min"] = "";
    jsonBase["LocalSigProcControl"]["regional-sampling"]["max"] = "";
    jsonBase["LocalSigProcControl"]["regional-sampling-type"]["type"] = OT_INT;
    jsonBase["LocalSigProcControl"]["regional-sampling-type"]["value"] = 1;
    jsonBase["LocalSigProcControl"]["regional-sampling-type"]["min"] = "";
    jsonBase["LocalSigProcControl"]["regional-sampling-type"]["max"] = "";
    jsonBase["LocalSigProcControl"]["num-regional-samples"]["type"] = OT_INT;
    jsonBase["LocalSigProcControl"]["num-regional-samples"]["value"] = 400;
    jsonBase["LocalSigProcControl"]["num-regional-samples"]["min"] = "";
    jsonBase["LocalSigProcControl"]["num-regional-samples"]["max"] = "";
    jsonBase["LocalSigProcControl"]["dark-matter-correction"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["dark-matter-correction"]["value"] = true;
    jsonBase["LocalSigProcControl"]["dark-matter-correction"]["min"] = "";
    jsonBase["LocalSigProcControl"]["dark-matter-correction"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-prefilter-beads"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["bkg-prefilter-beads"]["value"] = false;
    jsonBase["LocalSigProcControl"]["bkg-prefilter-beads"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-prefilter-beads"]["max"] = "";
    jsonBase["LocalSigProcControl"]["vectorize"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["vectorize"]["value"] = true;
    jsonBase["LocalSigProcControl"]["vectorize"]["min"] = "";
    jsonBase["LocalSigProcControl"]["vectorize"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-ampl-lower-limit"]["type"] = OT_DOUBLE;
    jsonBase["LocalSigProcControl"]["bkg-ampl-lower-limit"]["value"] = 0.001;
    jsonBase["LocalSigProcControl"]["bkg-ampl-lower-limit"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-ampl-lower-limit"]["max"] = "";
    jsonBase["LocalSigProcControl"]["limit-rdr-fit"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["limit-rdr-fit"]["value"] = false;
    jsonBase["LocalSigProcControl"]["limit-rdr-fit"]["min"] = "";
    jsonBase["LocalSigProcControl"]["limit-rdr-fit"]["max"] = "";
    jsonBase["LocalSigProcControl"]["use-alternative-etbr-equation"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["use-alternative-etbr-equation"]["value"] = false;
    jsonBase["LocalSigProcControl"]["use-alternative-etbr-equation"]["min"] = "";
    jsonBase["LocalSigProcControl"]["use-alternative-etbr-equation"]["max"] = "";
    jsonBase["LocalSigProcControl"]["fitting-taue"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["fitting-taue"]["value"] = false;
    jsonBase["LocalSigProcControl"]["fitting-taue"]["min"] = "";
    jsonBase["LocalSigProcControl"]["fitting-taue"]["max"] = "";
    jsonBase["LocalSigProcControl"]["use-safe-buffer-model"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["use-safe-buffer-model"]["value"] = false;
    jsonBase["LocalSigProcControl"]["use-safe-buffer-model"]["min"] = "";
    jsonBase["LocalSigProcControl"]["use-safe-buffer-model"]["max"] = "";
    jsonBase["LocalSigProcControl"]["suppress-copydrift"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["suppress-copydrift"]["value"] = false;
    jsonBase["LocalSigProcControl"]["suppress-copydrift"]["min"] = "";
    jsonBase["LocalSigProcControl"]["suppress-copydrift"]["max"] = "";
    jsonBase["LocalSigProcControl"]["incorporation-type"]["type"] = OT_INT;
    jsonBase["LocalSigProcControl"]["incorporation-type"]["value"] = 0;
    jsonBase["LocalSigProcControl"]["incorporation-type"]["min"] = "";
    jsonBase["LocalSigProcControl"]["incorporation-type"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-single-gauss-newton"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["bkg-single-gauss-newton"]["value"] = true;
    jsonBase["LocalSigProcControl"]["bkg-single-gauss-newton"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-single-gauss-newton"]["max"] = "";
    jsonBase["LocalSigProcControl"]["fit-region-kmult"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["fit-region-kmult"]["value"] = false;
    jsonBase["LocalSigProcControl"]["fit-region-kmult"]["min"] = "";
    jsonBase["LocalSigProcControl"]["fit-region-kmult"]["max"] = "";
    jsonBase["LocalSigProcControl"]["stop-beads"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["stop-beads"]["value"] = false;
    jsonBase["LocalSigProcControl"]["stop-beads"]["min"] = "";
    jsonBase["LocalSigProcControl"]["stop-beads"]["max"] = "";
    jsonBase["LocalSigProcControl"]["revert-regional-sampling"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["revert-regional-sampling"]["value"] = false;
    jsonBase["LocalSigProcControl"]["revert-regional-sampling"]["min"] = "";
    jsonBase["LocalSigProcControl"]["revert-regional-sampling"]["max"] = "";
    jsonBase["LocalSigProcControl"]["always-start-slow"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["always-start-slow"]["value"] = true;
    jsonBase["LocalSigProcControl"]["always-start-slow"]["min"] = "";
    jsonBase["LocalSigProcControl"]["always-start-slow"]["max"] = "";
    jsonBase["LocalSigProcControl"]["bkg-recompress-tail-raw-trace"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["bkg-recompress-tail-raw-trace"]["value"] = false;
    jsonBase["LocalSigProcControl"]["bkg-recompress-tail-raw-trace"]["min"] = "";
    jsonBase["LocalSigProcControl"]["bkg-recompress-tail-raw-trace"]["max"] = "";
    jsonBase["LocalSigProcControl"]["double-tap-means-zero"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["double-tap-means-zero"]["value"] = true;
    jsonBase["LocalSigProcControl"]["double-tap-means-zero"]["min"] = "";
    jsonBase["LocalSigProcControl"]["double-tap-means-zero"]["max"] = "";
    jsonBase["LocalSigProcControl"]["skip-first-flow-block-regional-fitting"]["type"] = OT_BOOL;
    jsonBase["LocalSigProcControl"]["skip-first-flow-block-regional-fitting"]["value"] = false;
    jsonBase["LocalSigProcControl"]["skip-first-flow-block-regional-fitting"]["min"] = "";
    jsonBase["LocalSigProcControl"]["skip-first-flow-block-regional-fitting"]["max"] = "";
/*
    Value jsonSimple(objectValue);
    Value::Members groups = jsonBase.getMemberNames();
    for(Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
    {
        if(*it1 == "chipType")
        {
            jsonSimple["chipType"] = jsonBase["chipType"];
        }
        else
        {
            Value::Members items = jsonBase[*it1].getMemberNames();
            for(Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
            {
                jsonSimple[*it2] = jsonBase[*it1][*it2]["value"];

            }
        }
    }
    string filename_json("./tsInputBase_simple.json");
    saveJson(jsonSimple, filename_json);
*/
    char buf[4000];
    string act = argv[1];
    if(act == "create")
    {
        if(argc != 4)
        {
            cerr << "tsinpututil ERROR: wrong option for tsinpututil create." << endl;
            usageCreate();
        }

        struct stat sb2;
        if(!(stat(argv[2], &sb2) == 0 && S_ISREG(sb2.st_mode)))
        {
            cerr << "tsinpututil ERROR: " << argv[2] << " does not exist or it is not a regular file." << endl;
            exit(1);
        }
        struct stat sb3;
        if(!(stat(argv[3], &sb3) == 0 && S_ISDIR(sb3.st_mode)))
        {
            cerr << "tsinpututil ERROR: " << argv[3] << " does not exist or it is not a directory." << endl;
            exit(1);
        }

        string dbDataName = argv[2];
        string outputDir = argv[3];
        string beadfindJson0(outputDir);
        beadfindJson0 += "/args_";
        string analysisJson0(outputDir);
        analysisJson0 += "/args_";
        string chipType;
        int nline = -1;
        bool skip = true;
        int beginLine = -1;
        int counter = -1;
        map<string, int> mapChips;

        ifstream ifs(dbDataName.c_str());
        while(ifs.getline(buf, 4000))
        {
            ++nline;
            if(skip)
            {
                ++beginLine;
            }

            string sline = buf;
            int index0 = sline.find("\"model\": \"rundb.analysisargs\""); // enter block
            int indexName = sline.find("\"name\": \"default_"); // for default chip type
            int indexChip = sline.find("\"chipType\""); // for chip type name
            int indexBeadfind = sline.find("\"beadfindargs\""); // for justBeadFind full chip
            int indexAnalysis = sline.find("\"analysisargs\""); // for Analysis full chip
            if(index0 >= 0)
            {
                skip = false;
                counter = 0;
            }
            if(counter > -1  && counter < 3)
            {
                if(indexName >= 0)
                {
                    ++counter;
                    int index1 = sline.find("_");
                    if(index1 > 16)
                    {
                        chipType = sline.substr(index1 + 1, sline.length() - index1 - 1);
                        int index2 = chipType.find("\"");
                        if(index2 > 2)
                        {
                            chipType = chipType.substr(0, index2);
                        }
                        int index3 = chipType.find("_");
                        if(index3 > 2)
                        {
                            chipType = chipType.substr(0, index3);
                        }
                    }
                    else
                    {
                        counter = -1;
                    }
                }
                else if(indexChip >= 0)
                {
                    int index1 = sline.find(":");
                    if(index1 > 9)
                    {
                        string chipType2 = sline.substr(index1 + 3, sline.length() - index1 - 3);
                        int index2 = chipType2.find("\"");
                        if(index2 > 2)
                        {
                            chipType2 = chipType2.substr(0, index2);
                        }
                        if(chipType2 != chipType)
                        {
                            counter = -1;
                            chipType = "";
                        }
                        map<string, int>::iterator iter = mapChips.find(chipType);
                        if(iter == mapChips.end())
                        {
                             mapChips[chipType] = nline;
                        }
                        else
                        {
                            counter = -1;
                            chipType = "";
                        }
                    }
                    else
                    {
                        counter = -1;
                        chipType = "";
                    }
                }
                else if(indexBeadfind >= 0)
                {
                    ++counter;
                    int index1 = sline.find("justBeadFind");
                    if(index1 > 16)
                    {
                        string args = sline.substr(index1, sline.length() - index1);
                        int index2 = args.find("\"");
                        if(index2 > 11)
                        {
                            args = args.substr(0, index2);
                        }

                        vector<string> vArgs;
                        split(args, ' ', vArgs);
                        int argn = vArgs.size();

                        char** argv2 = new char*[argn];
                        for(int i = 0; i < argn; ++i)
                        {
                            int len = vArgs[i].length() + 1;
                            argv2[i] = new char[len];
                            sprintf(argv2[i], "%s", vArgs[i].c_str());
                        }

                        OptArgs opts;
                        opts.ParseCmdLine(argn,  (const char**)argv2);

                        for(int i = 0; i < argn; ++i)
                        {
                            delete [] argv2[i];
                        }
                        delete [] argv2;

                        Value jsonChip(objectValue);
                        jsonChip["chipType"] = chipType;
                        Value::Members groups = jsonBase.getMemberNames();
                        for(Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
                        {
                            if(*it1 == "chipType" || *it1 == "BkgModelControlOpts" || *it1 == "GlobalDefaultsForBkgModel" || *it1 == "GpuControlOpts"
                            || *it1 == "LocalSigProcControl" || *it1 == "ObsoleteOpts"|| *it1 == "SignalProcessingBlockControl" || *it1 == "TraceControl")
                            {
                                continue;
                            }

                            Value::Members items = jsonBase[*it1].getMemberNames();
                            for(Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
                            {
                                jsonChip[*it1][*it2] = jsonBase[*it1][*it2]["value"];
                                if(opts.HasOption('-', *it2))
                                {
                                    if(jsonBase[*it1][*it2]["type"] == OT_BOOL)
                                    {
                                        bool val = opts.GetFirstBoolean('-', *it2, false);
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_STRING)
                                    {
                                        string val = opts.GetFirstString('-', *it2, "ValueIsNotSet");
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_INT)
                                    {
                                        int val = opts.GetFirstInt('-', *it2, 99999);
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_FLOAT)
                                    {
                                        float val = opts.GetFirstDouble('-', *it2, 99999.99);
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_DOUBLE)
                                    {
                                        double val = opts.GetFirstDouble('-', *it2, 99999.99);
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_INT)
                                    {
                                        vector<int> val;
                                        opts.GetOption(val, "ValueIsNotSet", '-', *it2);
                                        jsonChip[*it1][*it2].clear();
                                        for(size_t i = 0; i < val.size(); ++i)
                                        {
                                            jsonChip[*it1][*it2].append(val[i]);
                                        }
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_FLOAT)
                                    {
                                        vector<double> val;
                                        opts.GetOption(val, "ValueIsNotSet", '-', *it2);
                                        jsonChip[*it1][*it2].clear();
                                        for(size_t i = 0; i < val.size(); ++i)
                                        {
                                            jsonChip[*it1][*it2].append(val[i]);
                                        }
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_DOUBLE)
                                    {
                                        vector<double> val;
                                        opts.GetOption(val, "ValueIsNotSet", '-', *it2);
                                        jsonChip[*it1][*it2].clear();
                                        for(size_t i = 0; i < val.size(); ++i)
                                        {
                                            jsonChip[*it1][*it2].append(val[i]);
                                        }
                                    }
                                    else
                                    {
                                        cerr << "tsinpututil WARNING: cannot set beadfining option - " << *it2 << endl;
                                    }
                                }
                            }
                        }

                        string beadfindJson(beadfindJson0);
                        beadfindJson += chipType;
                        beadfindJson += "_beadfind.json";
                        saveJson(jsonChip, beadfindJson);
                        cout << "\nSave beadfining arguments for chip " << chipType << " to " << beadfindJson << endl;
                    }
                }
                else if(indexAnalysis >= 0)
                {
                    ++counter;
                    int index1 = sline.find("Analysis");
                    if(index1 > 16)
                    {
                        string args = sline.substr(index1, sline.length() - index1);
                        int index2 = args.find("\"");
                        if(index2 > 7)
                        {
                            args = args.substr(0, index2);
                        }

                        vector<string> vArgs;
                        split(args, ' ', vArgs);
                        int argn = vArgs.size();

                        char** argv2 = new char*[argn];
                        for(int i = 0; i < argn; ++i)
                        {
                            int len = vArgs[i].length() + 1;
                            argv2[i] = new char[len];
                            sprintf(argv2[i], "%s", vArgs[i].c_str());
                        }

                        OptArgs opts;
                        opts.ParseCmdLine(argn, (const char**)argv2);

                        for(int i = 0; i < argn; ++i)
                        {
                            delete [] argv2[i];
                        }
                        delete [] argv2;

                        Value jsonChip(objectValue);
                        jsonChip["chipType"] = chipType;
                        Value::Members groups = jsonBase.getMemberNames();
                        for(Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
                        {
                            if(*it1 == "chipType" || *it1 == "BeadfindControlOpts")
                            {
                                continue;
                            }

                            Value::Members items = jsonBase[*it1].getMemberNames();
                            for(Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
                            {
                                jsonChip[*it1][*it2] = jsonBase[*it1][*it2]["value"];
                                if(opts.HasOption('-', *it2))
                                {
                                    if(jsonBase[*it1][*it2]["type"] == OT_BOOL)
                                    {
                                        bool val = opts.GetFirstBoolean('-', *it2, false);
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_STRING)
                                    {
                                        string val = opts.GetFirstString('-', *it2, "ValueIsNotSet");
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_INT)
                                    {
                                        int val = opts.GetFirstInt('-', *it2, 99999);
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_FLOAT)
                                    {
                                        float val = opts.GetFirstDouble('-', *it2, 99999.99);
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_DOUBLE)
                                    {
                                        double val = opts.GetFirstDouble('-', *it2, 99999.99);
                                        jsonChip[*it1][*it2] = val;
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_INT)
                                    {
                                        vector<int> val;
                                        opts.GetOption(val, "ValueIsNotSet", '-', *it2);
                                        jsonChip[*it1][*it2].clear();
                                        for(size_t i = 0; i < val.size(); ++i)
                                        {
                                            jsonChip[*it1][*it2].append(val[i]);
                                        }
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_FLOAT)
                                    {
                                        vector<double> val;
                                        opts.GetOption(val, "ValueIsNotSet", '-', *it2);
                                        jsonChip[*it1][*it2].clear();
                                        for(size_t i = 0; i < val.size(); ++i)
                                        {
                                            jsonChip[*it1][*it2].append(val[i]);
                                        }
                                    }
                                    else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_DOUBLE)
                                    {
                                        vector<double> val;
                                        opts.GetOption(val, "ValueIsNotSet", '-', *it2);
                                        jsonChip[*it1][*it2].clear();
                                        for(size_t i = 0; i < val.size(); ++i)
                                        {
                                            jsonChip[*it1][*it2].append(val[i]);
                                        }
                                    }
                                    else
                                    {
                                        cerr << "tsinpututil WARNING: cannot set analysis option - " << *it2 << endl;
                                    }
                                }
                            }
                        }

                        string analysisJson(analysisJson0);
                        analysisJson += chipType;
                        analysisJson += "_analysis.json";
                        saveJson(jsonChip, analysisJson);
                        cout << "\nSave analysis arguments for chip " << chipType << " to " << analysisJson << endl;
                    }
                }
            }
        }

        ifs.close();

        nline = -1;
        string dbDataName2(dbDataName);
        dbDataName2 += ".modified";

        string argsBeadfindFile;
        Value jsonBeadfind;
        Reader readerBeadfind;
        string argsAnalysisFile;
        Value jsonAnalysis;
        Reader readerAnalysis;

        string args_json_beadfind;
        string args_json_analysis;
        string args_json_beadfindtn;
        string args_json_analysistn;

        ifstream ifs0(dbDataName.c_str());
        ofstream ofs(dbDataName2.c_str());
        while(ifs0.getline(buf, 4000))
        {
            ++nline;

            string sline = buf;
            if(nline < beginLine)
            {
                ofs << sline << endl;
                continue;
            }

            int index0 = sline.find("\"model\": \"rundb.analysisargs\""); // enter block
            int indexChip = sline.find("\"chipType\""); // for chip type name
            int indexBeadfind = sline.find("\"beadfindargs\""); // for justBeadFind full chip
            int indexAnalysis = sline.find("\"analysisargs\""); // for Analysis full chip
            int indexBeadfindTn = sline.find("\"thumbnailbeadfindargs\""); // for justBeadFind thumbnail
            int indexAnalysisTn = sline.find("\"thumbnailanalysisargs\""); // for Analysis thumbnail
            if(index0 >= 0)
            {
                counter = 0;
            }

            if(counter > -1)
            {
                if(indexChip >= 0)
                {
                    int index1 = sline.find(":");
                    if(index1 > 9)
                    {
                        string chipType = sline.substr(index1 + 3, sline.length() - index1 - 3);
                        int index2 = chipType.find("\"");
                        if(index2 > 2)
                        {
                            chipType = chipType.substr(0, index2);
                        }

                        argsBeadfindFile = beadfindJson0;
                        argsAnalysisFile = analysisJson0;
                        argsBeadfindFile += chipType;
                        argsBeadfindFile += "_beadfind.json";
                        ifstream ifsb(argsBeadfindFile.c_str());
                        readerBeadfind.parse(ifsb, jsonBeadfind, false);
                        ifsb.close();
                        argsAnalysisFile += chipType;
                        argsAnalysisFile += "_analysis.json";
                        cout << argsAnalysisFile << endl;
                        ifstream ifsa(argsAnalysisFile.c_str());
                        readerAnalysis.parse(ifsa, jsonAnalysis, false);
                        ifsa.close();

                        args_json_beadfind = "      \"beadfindargs\"               : \"justBeadFind --args-json /opt/ion/config/args_";
                        args_json_analysis = "      \"analysisargs\"               : \"Analysis --args-json /opt/ion/config/args_";
                        args_json_beadfindtn = "      \"thumbnailbeadfindargs\"      : \"justBeadFind --args-json /opt/ion/config/args_";
                        args_json_analysistn = "      \"thumbnailanalysisargs\"      : \"Analysis --args-json /opt/ion/config/args_";

                        args_json_beadfind += chipType;
                        args_json_beadfind += "_beadfind.json";
                        args_json_analysis += chipType;
                        args_json_analysis += "_analysis.json";
                        args_json_beadfindtn += chipType;
                        args_json_beadfindtn += "_beadfind.json";
                        args_json_analysistn += chipType;
                        args_json_analysistn += "_analysis.json";
                    }
                }
                else if(indexBeadfind >= 0)
                {
                    ++counter;
                    int index1 = sline.find("justBeadFind");
                    if(index1 > 16)
                    {
                        string args = sline.substr(index1, sline.length() - index1);
                        int index2 = args.find("\"");
                        if(index2 > 11)
                        {
                            args = args.substr(0, index2);
                        }

                        vector<string> vArgs;
                        split(args, ' ', vArgs);
                        int argn = vArgs.size();

                        char** argv2 = new char*[argn];
                        for(int i = 0; i < argn; ++i)
                        {
                            int len = vArgs[i].length() + 1;
                            argv2[i] = new char[len];
                            sprintf(argv2[i], "%s", vArgs[i].c_str());
                        }

                        OptArgs opts;
                        opts.ParseCmdLine(argn,  (const char**)argv2);

                        for(int i = 0; i < argn; ++i)
                        {
                            delete [] argv2[i];
                        }
                        delete [] argv2;

                        Value::Members groups = jsonBeadfind.getMemberNames();
                        for(Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
                        {
                            if(*it1 != "chipType")
                            {
                                Value::Members items = jsonBeadfind[*it1].getMemberNames();
                                for(Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
                                {
                                    if(opts.HasOption('-', *it2))
                                    {
                                        if(jsonBase[*it1][*it2]["type"] == OT_BOOL)
                                        {
                                            bool val0 = jsonBeadfind[*it1][*it2].asBool();
                                            bool val = opts.GetFirstBoolean('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_beadfind += " --";
                                                args_json_beadfind += (*it2);
                                                if(val == false)
                                                {
                                                    args_json_beadfind += " false";
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_STRING)
                                        {
                                            string val0 = jsonBeadfind[*it1][*it2].asString();
                                            string val = opts.GetFirstString('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_beadfind += " --";
                                                args_json_beadfind += (*it2);
                                                args_json_beadfind += " ";
                                                args_json_beadfind += val;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_INT)
                                        {
                                            int val0 = jsonBeadfind[*it1][*it2].asInt();
                                            int val = opts.GetFirstInt('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_beadfind += " --";
                                                args_json_beadfind += (*it2);
                                                sprintf(buf, " %d", val);
                                                args_json_beadfind += buf;
                                            }
                                         }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_FLOAT)
                                        {
                                            double val0 = jsonBeadfind[*it1][*it2].asFloat();
                                            float val = opts.GetFirstDouble('-', *it2, val0);
                                            if(val != (float)val0)
                                            {
                                                args_json_beadfind += " --";
                                                args_json_beadfind += (*it2);
                                                sprintf(buf, " %f", val);
                                                args_json_beadfind += buf;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_DOUBLE)
                                        {
                                            double val0 = jsonBeadfind[*it1][*it2].asDouble();
                                            double val = opts.GetFirstDouble('-', *it2, val0);
                                            if(val != (float)val0)
                                            {
                                                args_json_beadfind += " --";
                                                args_json_beadfind += (*it2);
                                                sprintf(buf, " %f", val);
                                                args_json_beadfind += buf;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_INT)
                                        {
                                            Value val0 = jsonBeadfind[*it1][*it2];
                                            vector<int> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    int m0 = (*it0).asInt();
                                                    int m = val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_beadfind += " --";
                                                args_json_beadfind += (*it2);
                                                sprintf(buf, " %d", val[0]);
                                                args_json_beadfind += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%d", val[i]);
                                                    args_json_beadfind += buf;
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_FLOAT)
                                        {
                                            Value val0 = jsonBeadfind[*it1][*it2];
                                            vector<double> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    float m0 = (*it0).asFloat();
                                                    float m = (float)val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_beadfind += " --";
                                                args_json_beadfind += (*it2);
                                                sprintf(buf, " %f", val[0]);
                                                args_json_beadfind += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%f", val[i]);
                                                    args_json_beadfind += buf;
                                                }

                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_DOUBLE)
                                        {
                                            Value val0 = jsonBeadfind[*it1][*it2];
                                            vector<double> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    double m0 = (*it0).asDouble();
                                                    double m = (float)val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_beadfind += " --";
                                                args_json_beadfind += (*it2);
                                                sprintf(buf, " %f", val[0]);
                                                args_json_beadfind += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%f", val[i]);
                                                    args_json_beadfind += buf;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            cerr << "tsinpututil WARNING: cannot set option - " << *it2 << endl;
                                        }
                                    }
                                }
                            }
                        }
                        args_json_beadfind += "\",";
                        sline = args_json_beadfind;
                    }
                }
                else if(indexAnalysis >= 0)
                {
                    ++counter;
                    int index1 = sline.find("Analysis");
                    if(index1 > 16)
                    {
                        string args = sline.substr(index1, sline.length() - index1);
                        int index2 = args.find("\"");
                        if(index2 > 11)
                        {
                            args = args.substr(0, index2);
                        }

                        vector<string> vArgs;
                        split(args, ' ', vArgs);
                        int argn = vArgs.size();

                        char** argv2 = new char*[argn];
                        for(int i = 0; i < argn; ++i)
                        {
                            int len = vArgs[i].length() + 1;
                            argv2[i] = new char[len];
                            sprintf(argv2[i], "%s", vArgs[i].c_str());
                        }

                        OptArgs opts;
                        opts.ParseCmdLine(argn,  (const char**)argv2);

                        for(int i = 0; i < argn; ++i)
                        {
                            delete [] argv2[i];
                        }
                        delete [] argv2;

                        Value::Members groups = jsonAnalysis.getMemberNames();
                        for(Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
                        {
                            if(*it1 != "chipType")
                            {
                                Value::Members items = jsonAnalysis[*it1].getMemberNames();
                                for(Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
                                {
                                    if(opts.HasOption('-', *it2))
                                    {
                                        if(jsonBase[*it1][*it2]["type"] == OT_BOOL)
                                        {
                                            bool val0 = jsonAnalysis[*it1][*it2].asBool();
                                            bool val = opts.GetFirstBoolean('-', *it2, val0);

                                            if(val != val0)
                                            {
                                                args_json_analysis += " --";
                                                args_json_analysis += (*it2);
                                                if(val == false)
                                                {
                                                    args_json_analysis += " false";
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_STRING)
                                        {
                                            string val0 = jsonAnalysis[*it1][*it2].asString();
                                            string val = opts.GetFirstString('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_analysis += " --";
                                                args_json_analysis += (*it2);
                                                args_json_analysis += " ";
                                                args_json_analysis += val;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_INT)
                                        {
                                            int val0 = jsonAnalysis[*it1][*it2].asInt();
                                            int val = opts.GetFirstInt('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_analysis += " --";
                                                args_json_analysis += (*it2);
                                                sprintf(buf, " %d", val);
                                                args_json_analysis += buf;
                                            }
                                         }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_FLOAT)
                                        {
                                            double val0 = jsonAnalysis[*it1][*it2].asFloat();
                                            float val = opts.GetFirstDouble('-', *it2, val0);
                                            if(val != (float)val0)
                                            {
                                                args_json_analysis += " --";
                                                args_json_analysis += (*it2);
                                                sprintf(buf, " %f", val);
                                                args_json_analysis += buf;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_DOUBLE)
                                        {
                                            double val0 = jsonAnalysis[*it1][*it2].asDouble();
                                            double val = opts.GetFirstDouble('-', *it2, val0);
                                            if(val != (float)val0)
                                            {
                                                args_json_analysis += " --";
                                                args_json_analysis += (*it2);
                                                sprintf(buf, " %f", val);
                                                args_json_analysis += buf;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_INT)
                                        {
                                            Value val0 = jsonAnalysis[*it1][*it2];
                                            vector<int> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    int m0 = (*it0).asInt();
                                                    int m = val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_analysis += " --";
                                                args_json_analysis += (*it2);
                                                sprintf(buf, " %d", val[0]);
                                                args_json_analysis += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%d", val[i]);
                                                    args_json_analysis += buf;
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_FLOAT)
                                        {
                                            Value val0 = jsonAnalysis[*it1][*it2];
                                            vector<double> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    float m0 = (*it0).asFloat();
                                                    float m = (float)val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_analysis += " --";
                                                args_json_analysis += (*it2);
                                                sprintf(buf, " %f", val[0]);
                                                args_json_analysis += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%f", val[i]);
                                                    args_json_analysis += buf;
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_DOUBLE)
                                        {
                                            Value val0 = jsonAnalysis[*it1][*it2];
                                            vector<double> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    double m0 = (*it0).asDouble();
                                                    double m = (float)val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_analysis += " --";
                                                args_json_analysis += (*it2);
                                                sprintf(buf, " %f", val[0]);
                                                args_json_analysis += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%f", val[i]);
                                                    args_json_analysis += buf;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            cerr << "tsinpututil WARNING: cannot set option - " << *it2 << endl;
                                        }
                                    }
                                }
                            }
                        }
                        args_json_analysis += "\",";
                        sline = args_json_analysis;
                    }
                }
                else if(indexBeadfindTn >= 0)
                {
                    ++counter;
                    int index1 = sline.find("justBeadFind");
                    if(index1 > 16)
                    {
                        string args = sline.substr(index1, sline.length() - index1);
                        int index2 = args.find("\"");
                        if(index2 > 11)
                        {
                            args = args.substr(0, index2);
                        }

                        vector<string> vArgs;
                        split(args, ' ', vArgs);
                        int argn = vArgs.size();

                        char** argv2 = new char*[argn];
                        for(int i = 0; i < argn; ++i)
                        {
                            int len = vArgs[i].length() + 1;
                            argv2[i] = new char[len];
                            sprintf(argv2[i], "%s", vArgs[i].c_str());
                        }

                        OptArgs opts;
                        opts.ParseCmdLine(argn,  (const char**)argv2);

                        for(int i = 0; i < argn; ++i)
                        {
                            delete [] argv2[i];
                        }
                        delete [] argv2;

                        Value::Members groups = jsonBeadfind.getMemberNames();
                        for(Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
                        {
                            if(*it1 != "chipType")
                            {
                                Value::Members items = jsonBeadfind[*it1].getMemberNames();
                                for(Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
                                {
                                    if(opts.HasOption('-', *it2))
                                    {
                                        if(jsonBase[*it1][*it2]["type"] == OT_BOOL)
                                        {
                                            bool val0 = jsonBeadfind[*it1][*it2].asBool();
                                            bool val = opts.GetFirstBoolean('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_beadfindtn += " --";
                                                args_json_beadfindtn += (*it2);
                                                if(val == false)
                                                {
                                                    args_json_beadfindtn += " false";
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_STRING)
                                        {
                                            string val0 = jsonBeadfind[*it1][*it2].asString();
                                            string val = opts.GetFirstString('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_beadfindtn += " --";
                                                args_json_beadfindtn += (*it2);
                                                args_json_beadfindtn += " ";
                                                args_json_beadfindtn += val;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_INT)
                                        {
                                            int val0 = jsonBeadfind[*it1][*it2].asInt();
                                            int val = opts.GetFirstInt('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_beadfindtn += " --";
                                                args_json_beadfindtn += (*it2);
                                                sprintf(buf, " %d", val);
                                                args_json_beadfindtn += buf;
                                            }
                                         }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_FLOAT)
                                        {
                                            double val0 = jsonBeadfind[*it1][*it2].asFloat();
                                            float val = opts.GetFirstDouble('-', *it2, val0);
                                            if(val != (float)val0)
                                            {
                                                args_json_beadfindtn += " --";
                                                args_json_beadfindtn += (*it2);
                                                sprintf(buf, " %f", val);
                                                args_json_beadfindtn += buf;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_DOUBLE)
                                        {
                                            double val0 = jsonBeadfind[*it1][*it2].asDouble();
                                            double val = opts.GetFirstDouble('-', *it2, val0);
                                            if(val != (float)val0)
                                            {
                                                args_json_beadfindtn += " --";
                                                args_json_beadfindtn += (*it2);
                                                sprintf(buf, " %f", val);
                                                args_json_beadfindtn += buf;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_INT)
                                        {
                                            Value val0 = jsonBeadfind[*it1][*it2];
                                            vector<int> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    int m0 = (*it0).asInt();
                                                    int m = val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_beadfindtn += " --";
                                                args_json_beadfindtn += (*it2);
                                                sprintf(buf, " %d", val[0]);
                                                args_json_beadfindtn += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%d", val[i]);
                                                    args_json_beadfindtn += buf;
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_FLOAT)
                                        {
                                            Value val0 = jsonBeadfind[*it1][*it2];
                                            vector<double> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    float m0 = (*it0).asFloat();
                                                    float m = (float)val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_beadfindtn += " --";
                                                args_json_beadfindtn += (*it2);
                                                sprintf(buf, " %f", val[0]);
                                                args_json_beadfindtn += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%f", val[i]);
                                                    args_json_beadfindtn += buf;
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_DOUBLE)
                                        {
                                            Value val0 = jsonBeadfind[*it1][*it2];
                                            vector<double> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    double m0 = (*it0).asDouble();
                                                    double m = (float)val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_beadfindtn += " --";
                                                args_json_beadfindtn += (*it2);
                                                sprintf(buf, " %f", val[0]);
                                                args_json_beadfindtn += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%f", val[i]);
                                                    args_json_beadfindtn += buf;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            cerr << "tsinpututil WARNING: cannot set option - " << *it2 << endl;
                                        }
                                    }
                                }
                            }
                        }
                        args_json_beadfindtn += "\",";
                        sline = args_json_beadfindtn;
                    }
                }
                else if(indexAnalysisTn >= 0)
                {
                    ++counter;
                    int index1 = sline.find("Analysis");
                    if(index1 > 16)
                    {
                        string args = sline.substr(index1, sline.length() - index1);
                        int index2 = args.find("\"");
                        if(index2 > 11)
                        {
                            args = args.substr(0, index2);
                        }

                        vector<string> vArgs;
                        split(args, ' ', vArgs);
                        int argn = vArgs.size();

                        char** argv2 = new char*[argn];
                        for(int i = 0; i < argn; ++i)
                        {
                            int len = vArgs[i].length() + 1;
                            argv2[i] = new char[len];
                            sprintf(argv2[i], "%s", vArgs[i].c_str());
                        }

                        OptArgs opts;
                        opts.ParseCmdLine(argn,  (const char**)argv2);

                        for(int i = 0; i < argn; ++i)
                        {
                            delete [] argv2[i];
                        }
                        delete [] argv2;

                        Value::Members groups = jsonAnalysis.getMemberNames();
                        for(Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
                        {
                            if(*it1 != "chipType")
                            {
                                Value::Members items = jsonAnalysis[*it1].getMemberNames();
                                for(Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
                                {
                                    if(opts.HasOption('-', *it2))
                                    {
                                        if(jsonBase[*it1][*it2]["type"] == OT_BOOL)
                                        {
                                            bool val0 = jsonAnalysis[*it1][*it2].asBool();
                                            bool val = opts.GetFirstBoolean('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_analysistn += " --";
                                                args_json_analysistn += (*it2);
                                                if(val == false)
                                                {
                                                    args_json_analysistn += " false";
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_STRING)
                                        {
                                            string val0 = jsonAnalysis[*it1][*it2].asString();
                                            string val = opts.GetFirstString('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_analysistn += " --";
                                                args_json_analysistn += (*it2);
                                                args_json_analysistn += " ";
                                                args_json_analysistn += val;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_INT)
                                        {
                                            int val0 = jsonAnalysis[*it1][*it2].asInt();
                                            int val = opts.GetFirstInt('-', *it2, val0);
                                            if(val != val0)
                                            {
                                                args_json_analysistn += " --";
                                                args_json_analysistn += (*it2);
                                                sprintf(buf, " %d", val);
                                                args_json_analysistn += buf;
                                            }
                                         }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_FLOAT)
                                        {
                                            double val0 = jsonAnalysis[*it1][*it2].asFloat();
                                            float val = opts.GetFirstDouble('-', *it2, val0);
                                            if(val != (float)val0)
                                            {
                                                args_json_analysistn += " --";
                                                args_json_analysistn += (*it2);
                                                sprintf(buf, " %f", val);
                                                args_json_analysistn += buf;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_DOUBLE)
                                        {
                                            double val0 = jsonAnalysis[*it1][*it2].asDouble();
                                            double val = opts.GetFirstDouble('-', *it2, val0);
                                            if(val != (float)val0)
                                            {
                                                args_json_analysistn += " --";
                                                args_json_analysistn += (*it2);
                                                sprintf(buf, " %f", val);
                                                args_json_analysistn += buf;
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_INT)
                                        {
                                            Value val0 = jsonAnalysis[*it1][*it2];
                                            vector<int> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    int m0 = (*it0).asInt();
                                                    int m = val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_analysistn += " --";
                                                args_json_analysistn += (*it2);
                                                sprintf(buf, " %d", val[0]);
                                                args_json_analysistn += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%d", val[i]);
                                                    args_json_analysistn += buf;
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_FLOAT)
                                        {
                                            Value val0 = jsonAnalysis[*it1][*it2];
                                            vector<double> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    float m0 = (*it0).asFloat();
                                                    float m = (float)val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_analysistn += " --";
                                                args_json_analysistn += (*it2);
                                                sprintf(buf, " %f", val[0]);
                                                args_json_analysistn += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%f", val[i]);
                                                    args_json_analysistn += buf;
                                                }
                                            }
                                        }
                                        else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_DOUBLE)
                                        {
                                            Value val0 = jsonAnalysis[*it1][*it2];
                                            vector<double> val;
                                            opts.GetOption(val, "ValueIsNotSet", '-', *it2);

                                            bool diff = false;
                                            if(val.size() != val0.size())
                                            {
                                                diff = true;
                                            }
                                            else
                                            {
                                                int k = 0;
                                                for(Value::iterator it0 = val0.begin(); it0 != val0.end(); ++it0, ++k)
                                                {
                                                    double m0 = (*it0).asDouble();
                                                    double m = (float)val[k];
                                                    if(m != m0)
                                                    {
                                                        diff = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if(diff && val.size() > 0)
                                            {
                                                args_json_analysistn += " --";
                                                args_json_analysistn += (*it2);
                                                sprintf(buf, " %f", val[0]);
                                                args_json_analysistn += buf;
                                                for(size_t i = 1; i < val.size(); ++i)
                                                {
                                                    sprintf(buf, ",%f", val[i]);
                                                    args_json_analysistn += buf;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            cerr << "tsinpututil WARNING: cannot set option - " << *it2 << endl;
                                        }
                                    }
                                }
                            }
                        }
                        args_json_analysistn += "\",";
                        sline = args_json_analysistn;
                    }
                }
            }

            ofs << sline << endl;
        }
        ifs0.close();
        ofs.close();
    }
    else if(act == "edit")
    {
        if(argc < 3)
        {
            cerr << "tsinpututil ERROR: wrong option for tsinpututil edit." << endl;
            usageEdit();
        }

        struct stat sb2;
        if(!(stat(argv[2], &sb2) == 0 && S_ISREG(sb2.st_mode)))
        {
            cerr << "tsinpututil ERROR: " << argv[2] << " does not exist or it is not a regular file." << endl;
            exit(1);
        }

        string jsonFile = argv[2];
        ifstream ifs(jsonFile.c_str());

        Value json;
        Reader reader;
        reader.parse(ifs, json, false);
        ifs.close();

        int argc2 = argc - 3;
        if(argc2 > 0)
        {
            char** argv2 = new char*[argc2];
            int i2 = 0;
            for(int i = 3; i < argc; ++i, ++i2)
            {
                int len = strlen(argv[i]) + 1;
                argv2[i2] = new char[len];
                memcpy(argv2[i2], argv[i], len);
            }

            OptArgs opts;
            opts.ParseCmdLine(argc2, (const char**)argv2);

            for(int i2 = 0; i2 < argc2; ++i2)
            {
                delete [] argv2[i2];
            }
            delete [] argv2;

            Value::Members groups = json.getMemberNames();
            for(Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
            {
                if(*it1 != "chipType")
                {
                    Value::Members items = json[*it1].getMemberNames();
                    for(Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
                    {
                        if(opts.HasOption('-', *it2))
                        {
                            if(jsonBase[*it1][*it2]["type"] == OT_BOOL)
                            {
                                bool val0 = json[*it1][*it2].asBool();
                                bool val = opts.GetFirstBoolean('-', *it2, val0);
                                json[*it1][*it2] = val;
                                cout << *it1 << "::" << *it2 << " is set from " << val0 << " to " << val << endl;
                            }
                            else if(jsonBase[*it1][*it2]["type"] == OT_STRING)
                            {
                                string val0 = json[*it1][*it2].asString();
                                string val = opts.GetFirstString('-', *it2, val0);
                                json[*it1][*it2] = val;
                                cout << *it1 << "::" << *it2 << " is set from " << val0 << " to " << val << endl;
                            }
                            else if(jsonBase[*it1][*it2]["type"] == OT_INT)
                            {
                                int val0 = json[*it1][*it2].asInt();
                                int val = opts.GetFirstInt('-', *it2, val0);
                                json[*it1][*it2] = val;
                                cout << *it1 << "::" << *it2 << " is set from " << val0 << " to " << val << endl;
                             }
                            else if(jsonBase[*it1][*it2]["type"] == OT_FLOAT)
                            {
                                double val0 = json[*it1][*it2].asFloat();
                                float val = opts.GetFirstDouble('-', *it2, val0);
                                json[*it1][*it2] = val;
                                cout << *it1 << "::" << *it2 << " is set from " << val0 << " to " << val << endl;
                            }
                            else if(jsonBase[*it1][*it2]["type"] == OT_DOUBLE)
                            {
                                double val0 = json[*it1][*it2].asDouble();
                                double val = opts.GetFirstDouble('-', *it2, val0);
                                json[*it1][*it2] = val;
                                cout << *it1 << "::" << *it2 << " is set from " << val0 << " to " << val << endl;
                            }
                            else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_INT)
                            {
                                Value val0 = json[*it1][*it2];
                                string sval0;
                                if(val0.size() > 0)
                                {
                                    Value::iterator it0 = val0.begin();
                                    sprintf(buf, "%d", (*it0).asInt());
                                    sval0 += buf;
                                    ++it0;
                                    for(; it0 != val0.end(); ++it0)
                                    {
                                        sprintf(buf, ",%d", (*it0).asInt());
                                        sval0 += buf;
                                    }
                                }
                                vector<int> val;
                                opts.GetOption(val, "ValueIsNotSet", '-', *it2);
                                json[*it1][*it2].clear();
                                string sval;
                                if(val.size() > 0)
                                {
                                    json[*it1][*it2].append(val[0]);
                                    sprintf(buf, "%d", val[0]);
                                    sval += buf;
                                    for(size_t i = 1; i < val.size() - 1; ++i)
                                    {
                                        json[*it1][*it2].append(val[i]);
                                        sprintf(buf, ",%d", val[i]);
                                        sval += buf;
                                    }
                                }
                                cout << *it1 << "::" << *it2 << " is set from " << sval0 << " to " << sval << endl;
                            }
                            else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_FLOAT)
                            {
                                Value val0 = json[*it1][*it2];
                                string sval0;
                                if(val0.size() > 0)
                                {
                                    Value::iterator it0 = val0.begin();
                                    sprintf(buf, "%f", (*it0).asFloat());
                                    sval0 += buf;
                                    ++it0;
                                    for(; it0 != val0.end(); ++it0)
                                    {
                                        sprintf(buf, ",%f", (*it0).asFloat());
                                        sval0 += buf;
                                    }
                                }
                                vector<double> val;
                                opts.GetOption(val, "ValueIsNotSet", '-', *it2);
                                json[*it1][*it2].clear();
                                string sval;
                                if(val.size() > 0)
                                {
                                    json[*it1][*it2].append(val[0]);
                                    sprintf(buf, "%f", val[0]);
                                    sval += buf;
                                    for(size_t i = 1; i < val.size() - 1; ++i)
                                    {
                                        json[*it1][*it2].append(val[i]);
                                        sprintf(buf, ",%f", val[i]);
                                        sval += buf;
                                    }
                                }

                                cout << *it1 << "::" << *it2 << " is set from " << sval0 << " to " << sval << endl;
                            }
                            else if(jsonBase[*it1][*it2]["type"] == OT_VECTOR_DOUBLE)
                            {
                                Value val0 = json[*it1][*it2];
                                string sval0;
                                if(val0.size() > 0)
                                {
                                    Value::iterator it0 = val0.begin();
                                    sprintf(buf, "%f", (*it0).asDouble());
                                    sval0 += buf;
                                    ++it0;
                                    for(; it0 != val0.end(); ++it0)
                                    {
                                        sprintf(buf, ",%f", (*it0).asDouble());
                                        sval0 += buf;
                                    }
                                }
                                vector<double> val;
                                opts.GetOption(val, "ValueIsNotSet", '-', *it2);
                                json[*it1][*it2].clear();
                                string sval;
                                if(val.size() > 0)
                                {
                                    json[*it1][*it2].append(val[0]);
                                    sprintf(buf, "%f", val[0]);
                                    sval += buf;
                                    for(size_t i = 1; i < val.size() - 1; ++i)
                                    {
                                        json[*it1][*it2].append(val[i]);
                                        sprintf(buf, ",%f", val[i]);
                                        sval += buf;
                                    }
                                }
                                cout << *it1 << "::" << *it2 << " is set from " << sval0 << " to " << sval << endl;
                            }
                            else
                            {
                                cerr << "tsinpututil WARNING: cannot set option - " << *it2 << endl;
                            }
                        }
                    }
                }
            }
        }

        saveJson(json, jsonFile);
    }
    else if(act == "validate")
    {
        cerr << "tsinpututil ERROR: this function is under construction." << endl;
        usageValidate();
    }
    else if(act == "diff")
    {
        if(argc != 4)
        {
            cerr << "tsinpututil ERROR: wrong option for tsinpututil diff." << endl;
            usageDiff();
        }

        struct stat sb2;
        if(!(stat(argv[2], &sb2) == 0 && S_ISREG(sb2.st_mode)))
        {
            cerr << "tsinpututil ERROR: " << argv[2] << " does not exist or it is not a regular file." << endl;
            exit(1);
        }
        struct stat sb3;
        if(!(stat(argv[3], &sb3) == 0 && S_ISREG(sb3.st_mode)))
        {
            cerr << "tsinpututil ERROR: " << argv[3] << " does not exist or it is not a regular file." << endl;
            exit(1);
        }

        map<string, int> mapOptnames;
        map<string, string> mapOpt1;
        map<string, string> mapOpt2;

        Value::Members groups = jsonBase.getMemberNames();
        for(Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
        {
            if(*it1 != "chipType")
            {
                Value::Members items = jsonBase[*it1].getMemberNames();
                for(Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
                {
                    mapOptnames[*it2] = 0;
                }
            }
        }

        string logName1 = argv[2];
        string logName2 = argv[3];
        ifstream ifs1(logName1.c_str());
        ifstream ifs2(logName2.c_str());
        int nline1 = -1;
        int nline2 = -1;
        int flag1 = 0;
        int flag2 = 0;

        while(ifs1.getline(buf, 4000))
        {
            ++nline1;
            string sline = buf;
            int index1 = sline.find("Command line = justBeadFind"); // check justBeadFind
            if(index1 >= 0)
            {
                flag1 = 1;
                break;
            }
            int index2 = sline.find("Command line = Analysis"); // check Analysis
            if(index2 >= 0)
            {
                flag1 = 2;
                break;
            }
        }
        while(ifs2.getline(buf, 4000))
        {
            ++nline2;
            string sline = buf;
            int index = sline.find("Command line = justBeadFind"); // check justBeadFind
            if(index >= 0)
            {
                flag2 = 1;
                break;
            }
            int index2 = sline.find("Command line = Analysis"); // check Analysis
            if(index2 >= 0)
            {
                flag2 = 2;
                break;
            }
        }

        if(flag1 != flag2)
        {
            if(flag1 == 1)
            {
                if(flag2 == 2)
                {
                    cout << "tsinpututil WARNING: " << argv[2] << " runs justBeadFind, but " << argv[3] << " only runs Analysis." << endl;
                }
                else if(flag2 == 0)
                {
                    cout << "tsinpututil WARNING: " << argv[2] << " runs justBeadFind, but " << argv[3] << " runs neither justBeadFind nor Analysis." << endl;
                }

                while(ifs1.getline(buf, 4000))
                {
                    ++nline1;
                    string sline = buf;
                    int index2 = sline.find("Command line = Analysis"); // check Analysis
                    if(index2 >= 0)
                    {
                        flag1 = 2;
                        break;
                    }
                }
            }
            if(flag2 == 1)
            {
                if(flag1 == 2)
                {
                    cout << "tsinpututil WARNING: " << argv[3] << " runs justBeadFind, but " << argv[2] << " only runs Analysis." << endl;
                }
                else if(flag1 == 0)
                {
                    cout << "tsinpututil WARNING: " << argv[3] << " runs justBeadFind, but " << argv[2] << " runs neither justBeadFind nor Analysis." << endl;
                }

                while(ifs2.getline(buf, 4000))
                {
                    ++nline2;
                    string sline = buf;
                    int index2 = sline.find("Command line = Analysis"); // check Analysis
                    if(index2 >= 0)
                    {
                        flag2 = 2;
                        break;
                    }
                }
            }
        }

        int flagVal = 0;
        if(flag1 == flag2)
        {
            flagVal = flag1;

            while(ifs1.getline(buf, 4000))
            {
                ++nline1;
                string sline = buf;
                int index2 = sline.find("Command line = Analysis"); // check Analysis
                if(index2 >= 0)
                {
                    ++flag1;
                    break;
                }

                bool noOpt = true;
                for(map<string, int>::iterator iter0 = mapOptnames.begin(); noOpt && iter0 != mapOptnames.end(); ++iter0)
                {
                    if(iter0->second == 1)
                    {
                        continue;
                    }

                    string optName = iter0->first;
                    optName += " = ";
                    int index3 = sline.find(optName);
                    if(index3 >= 0)
                    {
                        noOpt = false;
                        int index4 = sline.find(" = ");
                        string sVal = sline.substr(index4 + 3, sline.length() - index4 - 3);
                        mapOpt1[iter0->first] = sVal;

                        iter0->second = 1;
                    }
                }
            }

            while(ifs2.getline(buf, 4000))
            {
                ++nline2;
                string sline = buf;
                int index2 = sline.find("Command line = Analysis"); // check Analysis
                if(index2 >= 0)
                {
                    ++flag2;
                    break;
                }

                bool noOpt = true;
                for(map<string, int>::iterator iter0 = mapOptnames.begin(); noOpt && iter0 != mapOptnames.end(); ++iter0)
                {
                    if(iter0->second >= 2)
                    {
                        continue;
                    }

                    string optName = iter0->first;
                    optName += " = ";
                    int index3 = sline.find(optName);
                    if(index3 >= 0)
                    {
                        noOpt = false;
                        int index4 = sline.find(" = ");
                        string sVal = sline.substr(index4 + 3, sline.length() - index4 - 3);
                        mapOpt2[iter0->first] = sVal;

                        iter0->second += 2;
                    }
                }
            }

            int count = 0;

            if(flagVal == 1)
            {
                cout << "The difference in justBeadFind" << endl;
            }
            else if(flagVal == 2)
            {
                cout << "The difference in Analysis" << endl;
            }

            for(map<string, int>::iterator iter0 = mapOptnames.begin(); iter0 != mapOptnames.end(); ++iter0)
            {
                string optName = iter0->first;
                if(optName == "dat-source-directory")
                {
                    continue;
                }

                if(iter0->second == 1)
                {
                     count++;
                     cout << setw(35) << iter0->first << " = " << mapOpt1[iter0->first] << setw(20) << "NO value in " << argv[3] << endl;
                }
                else if(iter0->second == 2)
                {
                     count++;
                     cout << setw(35) << iter0->first << " = " << "NO value in " << argv[2] << setw(20) << mapOpt2[iter0->first] << endl;
                }
                else if(iter0->second == 3)
                {
                     if(mapOpt1[iter0->first] != mapOpt2[iter0->first])
                     {
                         string val1 = mapOpt1[iter0->first];
                         int index5 = val1.find(" (");
                         if(index5 >= 0)
                         {
                             val1 = val1.substr(0, index5);
                         }

                         string val2 = mapOpt2[iter0->first];
                         int index6 = val2.find(" (");
                         if(index6 >= 0)
                         {
                             val2 = val2.substr(0, index6);
                         }

                         if(val1 != val2)
                         {
                             count++;
                             cout << setw(35) << iter0->first << " = " << mapOpt1[iter0->first] << setw(20) << mapOpt2[iter0->first] << endl;
                         }
                     }
                }
            }

            if(count == 0)
            {
                cout << " is 0" << endl;
            }

        }

        if(flagVal == 1)
        {

            mapOpt1.clear();
            mapOpt2.clear();

            for(map<string, int>::iterator iter0 = mapOptnames.begin(); iter0 != mapOptnames.end(); ++iter0)
            {
                iter0->second = 0;
            }

            while(ifs1.getline(buf, 4000))
            {
                ++nline1;
                string sline = buf;
                int index2 = sline.find("Command line = Analysis"); // check Analysis
                if(index2 >= 0)
                {
                    ++flag1;
                    break;
                }

                bool noOpt = true;
                for(map<string, int>::iterator iter0 = mapOptnames.begin(); noOpt && iter0 != mapOptnames.end(); ++iter0)
                {
                    if(iter0->second == 1)
                    {
                        continue;
                    }

                    string optName = iter0->first;
                    optName += " = ";
                    int index3 = sline.find(optName);
                    if(index3 >= 0)
                    {
                        noOpt = false;
                        int index4 = sline.find(" = ");
                        string sVal = sline.substr(index4 + 3, sline.length() - index4 - 3);
                        mapOpt1[iter0->first] = sVal;

                        iter0->second = 1;
                    }
                }
            }

            while(ifs2.getline(buf, 4000))
            {
                ++nline2;
                string sline = buf;
                int index2 = sline.find("Command line = Analysis"); // check Analysis
                if(index2 >= 0)
                {
                    ++flag2;
                    break;
                }

                bool noOpt = true;
                for(map<string, int>::iterator iter0 = mapOptnames.begin(); noOpt && iter0 != mapOptnames.end(); ++iter0)
                {
                    if(iter0->second >= 2)
                    {
                        continue;
                    }

                    string optName = iter0->first;
                    optName += " = ";
                    int index3 = sline.find(optName);
                    if(index3 >= 0)
                    {
                        noOpt = false;
                        int index4 = sline.find(" = ");
                        string sVal = sline.substr(index4 + 3, sline.length() - index4 - 3);
                        mapOpt2[iter0->first] = sVal;

                        iter0->second += 2;
                    }
                }
            }

            int count = 0;
            cout << "The difference in Analysis" << endl;

            for(map<string, int>::iterator iter0 = mapOptnames.begin(); iter0 != mapOptnames.end(); ++iter0)
            {
                string optName = iter0->first;
                if(optName == "dat-source-directory")
                {
                    continue;
                }

                if(iter0->second == 1)
                {
                     count++;
                     cout << setw(35) << iter0->first << " = " << mapOpt1[iter0->first] << setw(20) << "NO value in " << argv[3] << endl;
                }
                else if(iter0->second == 2)
                {
                     count++;
                     cout << setw(35) << iter0->first << " = " << "NO value in " << argv[2] << setw(20) << mapOpt2[iter0->first] << endl;
                }
                else if(iter0->second == 3)
                {
                     if(mapOpt1[iter0->first] != mapOpt2[iter0->first])
                     {
                         string val1 = mapOpt1[iter0->first];
                         int index5 = val1.find(" (");
                         if(index5 >= 0)
                         {
                             val1 = val1.substr(0, index5);
                         }

                         string val2 = mapOpt2[iter0->first];
                         int index6 = val2.find(" (");
                         if(index6 >= 0)
                         {
                             val2 = val2.substr(0, index6);
                         }

                         if(val1 != val2)
                         {
                             count++;
                             cout << setw(35) << iter0->first << " = " << mapOpt1[iter0->first] << setw(20) << mapOpt2[iter0->first] << endl;
                         }
                     }
                }
            }

            if(count == 0)
            {
                cout << " is 0" << endl;
            }
        }

        ifs1.close();
        ifs2.close();
    }
    else
    {
        cerr << "tsinpututil ERROR: wrong option" << endl;
        usageMain();
    }

	exit(0);
}
