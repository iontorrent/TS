/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>

#include "json/json.h"


#define PER_CHIP_JSON_EDITOR_VERSION "1.0.0"
#define WAIT_TIME 60

using namespace std;
using namespace Json;

enum OptType {
    OT_BOOL = 0,
    OT_INT,
    OT_DOUBLE,
    OT_STRING,
    OT_VECTOR_INT,
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
    cerr << "perchipjsoneditor - edit per chip json files" << endl;
    cerr << "Usage: " << endl
       << "  perchipjsoneditor [perChipJsonDirectory]" << endl
       << "    operation:" << endl
       << "              create" << endl
       << "              edit" << endl
       << "              validate" << endl
       << "              diff"  <<endl;
    exit(1);
}

int main(int argc, const char *argv[])
{
    map<string, OptType> mapOptType;
    mapOptType["acqPrefix"] = OT_STRING;
    mapOptType["always-start-slow"] = OT_BOOL;
    mapOptType["analysis-region"] = OT_VECTOR_INT;
    mapOptType["barcode-debug"] = OT_BOOL;
    mapOptType["barcode-flag"] = OT_BOOL;
    mapOptType["barcode-penalty"] = OT_DOUBLE;
    mapOptType["barcode-radius"] = OT_DOUBLE;
    mapOptType["barcode-spec-file"] = OT_STRING;
    mapOptType["barcode-tie"] = OT_DOUBLE;
    mapOptType["beadfind-basis"] = OT_STRING;
    mapOptType["beadfind-bfmult"] = OT_DOUBLE;
    mapOptType["beadfind-bgdat"] = OT_STRING;
    mapOptType["beadfind-blob-filter"] = OT_BOOL;
    mapOptType["beadfind-dat"] = OT_STRING;
    mapOptType["beadfind-diagnostics"] = OT_INT;
    mapOptType["beadfind-filt-noisy-col"] = OT_STRING;
    mapOptType["beadfind-gain-correction"] = OT_BOOL;
    mapOptType["beadfind-lib-filt"] = OT_DOUBLE;
    mapOptType["beadfind-lib-min-peak"] = OT_DOUBLE;
    mapOptType["beadfind-mesh-step"] = OT_VECTOR_INT;
    mapOptType["beadfind-min-tf-snr"] = OT_DOUBLE;
    mapOptType["beadfind-minlive"] = OT_DOUBLE;
    mapOptType["beadfind-minlivesnr"] = OT_DOUBLE;
    mapOptType["beadfind-num-threads"] = OT_INT;
    mapOptType["beadfind-predict-end"] = OT_INT;
    mapOptType["beadfind-predict-start"] = OT_INT;
    mapOptType["beadfind-sdasbf"] = OT_BOOL;
    mapOptType["beadfind-sep-ref"] = OT_BOOL;
    mapOptType["beadfind-sig-ref-type"] = OT_INT;
    mapOptType["beadfind-skip-sd-recover"] = OT_INT;
    mapOptType["beadfind-smooth-trace"] = OT_BOOL;
    mapOptType["beadfind-tf-filt"] = OT_DOUBLE;
    mapOptType["beadfind-tf-min-peak"] = OT_DOUBLE;
    mapOptType["beadfind-type"] = OT_STRING;
    mapOptType["beadfind-zero-flows"] = OT_STRING;
    mapOptType["beadfindfile"] = OT_STRING;
    mapOptType["beadmask-categorized"] = OT_BOOL;
    mapOptType["bfold"] = OT_BOOL;
    mapOptType["bfonly"] = OT_BOOL;
    mapOptType["bkg-ampl-lower-limit"] = OT_DOUBLE;
    mapOptType["bkg-bfmask-update"] = OT_BOOL;
    mapOptType["bkg-copy-stringency"] = OT_DOUBLE;
    mapOptType["bkg-dbg-trace"] = OT_VECTOR_INT;
    mapOptType["bkg-debug-files"] = OT_BOOL;
    mapOptType["bkg-debug-nsamples"] = OT_INT;
    mapOptType["bkg-debug-param"] = OT_INT;
    mapOptType["bkg-debug-region"] = OT_VECTOR_INT;
    mapOptType["bkg-debug-trace-rcflow"] = OT_STRING;
    mapOptType["bkg-debug-trace-sse"] = OT_STRING;
    mapOptType["bkg-debug-trace-xyflow"] = OT_STRING;
    mapOptType["bkg-dont-emphasize-by-compression"] = OT_BOOL;
    mapOptType["bkg-empty-well-normalization"] = OT_BOOL;
    mapOptType["bkg-exp-tail-bkg-adj"] = OT_BOOL;
    mapOptType["bkg-exp-tail-bkg-limit"] = OT_DOUBLE;
    mapOptType["bkg-exp-tail-bkg-lower"] = OT_DOUBLE;
    mapOptType["bkg-exp-tail-fit"] = OT_BOOL;
    mapOptType["bkg-exp-tail-tau-adj"] = OT_BOOL;
    mapOptType["bkg-kmult-adj-low-hi"] = OT_DOUBLE;
    mapOptType["bkg-max-rank-beads"] = OT_INT;
    mapOptType["bkg-min-sampled-beads"] = OT_INT;
    mapOptType["bkg-pca-dark-matter"] = OT_BOOL;
    mapOptType["bkg-per-flow-time-tracking"] = OT_BOOL;
    mapOptType["bkg-post-key-step"] = OT_INT;
    mapOptType["bkg-post-key-train"] = OT_INT;
    mapOptType["bkg-prefilter-beads"] = OT_BOOL;
    mapOptType["bkg-recompress-tail-raw-trace"] = OT_BOOL;
    mapOptType["bkg-single-gauss-newton"] = OT_BOOL;
    mapOptType["bkg-use-duds"] = OT_BOOL;
    mapOptType["bkg-use-proton-well-correction"] = OT_BOOL;
    mapOptType["bkg-washout-flow-detection"] = OT_INT;
    mapOptType["bkg-washout-threshold"] = OT_DOUBLE;
    mapOptType["bkg-well-xtalk-name"] = OT_STRING;
    mapOptType["clonal-filter-bkgmodel"] = OT_BOOL;
    mapOptType["clonal-filter-debug"] = OT_BOOL;
    mapOptType["clonal-filter-use-last-iter-params"] = OT_BOOL;
    mapOptType["col-doubles-xtalk-correct"] = OT_BOOL;
    mapOptType["col-flicker-correct"] = OT_BOOL;
    mapOptType["col-flicker-correct-aggressive"] = OT_BOOL;
    mapOptType["col-flicker-correct-verbose"] = OT_BOOL;
    mapOptType["corr-noise-correct"] = OT_BOOL;
    mapOptType["cropped"] = OT_VECTOR_INT;
    mapOptType["cropped-region-origin"] = OT_VECTOR_INT;
    mapOptType["dark-matter-correction"] = OT_BOOL;
    mapOptType["dat-postfix"] = OT_STRING;
    mapOptType["dat-source-directory"] = OT_STRING;
    mapOptType["datacollect-gain-correction"] = OT_BOOL;
    mapOptType["debug-bead-only"] = OT_BOOL;
    mapOptType["double-tap-means-zero"] = OT_BOOL;
    mapOptType["explog-path"] = OT_STRING;
    mapOptType["filter-extreme-ppf-only"] = OT_BOOL;
    mapOptType["fit-region-kmult"] = OT_BOOL;
    mapOptType["fitting-taue"] = OT_BOOL;
    mapOptType["flow-order"] = OT_STRING;
    mapOptType["flowlimit"] = OT_INT;
    mapOptType["flowtimeoffset"] = OT_INT;
    mapOptType["fluid-potential-correct"] = OT_BOOL;
    mapOptType["fluid-potential-threshold"] = OT_DOUBLE;
    mapOptType["forcenn"] = OT_INT;
    mapOptType["frames"] = OT_INT;
    mapOptType["from-beadfind"] = OT_BOOL;
    mapOptType["gopt"] = OT_STRING;
    mapOptType["gpu-amp-guess"] = OT_INT;
    mapOptType["gpu-device-ids"] = OT_INT;
    mapOptType["gpu-fitting-only"] = OT_BOOL;
    mapOptType["gpu-flow-by-flow"] = OT_BOOL;
    mapOptType["gpu-force-multi-flow-fit"] = OT_BOOL;
    mapOptType["gpu-hybrid-fit-iter"] = OT_INT;
    mapOptType["gpu-memory-per-proc"] = OT_INT;
    mapOptType["gpu-multi-flow-fit"] = OT_INT;
    mapOptType["gpu-multi-flow-fit-blocksize"] = OT_INT;
    mapOptType["gpu-multi-flow-fit-l1config"] = OT_INT;
    mapOptType["gpu-num-history-flows"] = OT_INT;
    mapOptType["gpu-num-streams"] = OT_INT;
    mapOptType["gpu-partial-deriv-blocksize"] = OT_INT;
    mapOptType["gpu-partial-deriv-l1config"] = OT_INT;
    mapOptType["gpu-single-flow-fit"] = OT_INT;
    mapOptType["gpu-single-flow-fit-blocksize"] = OT_INT;
    mapOptType["gpu-single-flow-fit-l1config"] = OT_INT;
    mapOptType["gpu-single-flow-fit-type"] = OT_INT;
    mapOptType["gpu-switch-to-flow-by-flow-at"] = OT_INT;
    mapOptType["gpu-use-all-devices"] = OT_BOOL;
    mapOptType["gpu-verbose"] = OT_BOOL;
    mapOptType["gpuworkload"] = OT_DOUBLE;
    mapOptType["hilowfilter"] = OT_BOOL;
    mapOptType["ignore-checksum-errors"] = OT_BOOL;
    mapOptType["ignore-checksum-errors-1frame"] = OT_BOOL;
    mapOptType["img-gain-correct"] = OT_BOOL;
    mapOptType["incorporation-type"] = OT_INT;
    mapOptType["kmult-hi-limit"] = OT_DOUBLE;
    mapOptType["kmult-low-limit"] = OT_DOUBLE;
    mapOptType["kmult-penalty"] = OT_DOUBLE;
    mapOptType["librarykey"] = OT_STRING;
    mapOptType["limit-rdr-fit"] = OT_BOOL;
    mapOptType["local-wells-file"] = OT_BOOL;
    mapOptType["mask-datacollect-exclude-regions"] = OT_BOOL;
    mapOptType["max-iterations"] = OT_INT;
    mapOptType["mixed-first-flow"] = OT_INT;
    mapOptType["mixed-last-flow"] = OT_INT;
    mapOptType["mixed-model-option"] = OT_INT;
    mapOptType["mixed-stringency"] = OT_DOUBLE;
    mapOptType["n-unfiltered-lib"] = OT_INT;
    mapOptType["nn-subtract-empties"] = OT_BOOL;
    mapOptType["nnmask"] = OT_VECTOR_INT;
    mapOptType["nnmaskwh"] = OT_VECTOR_INT;
    mapOptType["no-subdir"] = OT_BOOL;
    mapOptType["no-threaded-file-access"] = OT_BOOL;
    mapOptType["noduds"] = OT_BOOL;
    mapOptType["nokey"] = OT_BOOL;
    mapOptType["nuc-correct"] = OT_INT;
    mapOptType["num-regional-samples"] = OT_INT;
    mapOptType["numcputhreads"] = OT_INT;
    mapOptType["output-dir"] = OT_STRING;
    mapOptType["output-pinned-wells"] = OT_BOOL;
    mapOptType["pair-xtalk-coeff"] = OT_DOUBLE;
    mapOptType["pass-tau"] = OT_BOOL;
    mapOptType["pca-test"] = OT_STRING;
    mapOptType["post-fit-handshake-worker"] = OT_BOOL;
    mapOptType["readaheaddat"] = OT_INT;
    mapOptType["region-list"] = OT_VECTOR_INT;
    mapOptType["region-size"] = OT_VECTOR_INT;
    mapOptType["region-vfrc-debug"] = OT_BOOL;
    mapOptType["regional-sampling"] = OT_BOOL;
    mapOptType["regional-sampling-type"] = OT_INT;
    mapOptType["restart-check"] = OT_BOOL;
    mapOptType["restart-from"] = OT_STRING;
    mapOptType["restart-next"] = OT_STRING;
    mapOptType["restart-region-params-file"] = OT_STRING;
    mapOptType["revert-regional-sampling"] = OT_BOOL;
    mapOptType["sigproc-compute-flow"] = OT_STRING;
    mapOptType["sigproc-regional-smoothing-alpha"] = OT_DOUBLE;
    mapOptType["sigproc-regional-smoothing-gamma"] = OT_DOUBLE;
    mapOptType["skip-first-flow-block-regional-fitting"] = OT_BOOL;
    mapOptType["smoothing"] = OT_STRING;
    mapOptType["smoothing-file"] = OT_STRING;
    mapOptType["stack-dump-file"] = OT_STRING;
    mapOptType["start-flow-plus-interval"] = OT_VECTOR_INT;
    mapOptType["stop-beads"] = OT_BOOL;
    mapOptType["suppress-copydrift"] = OT_BOOL;
    mapOptType["tfkey"] = OT_STRING;
    mapOptType["total-timeout"] = OT_INT;
    mapOptType["trim-ref-trace"] = OT_STRING;
    mapOptType["use-alternative-etbr-equation"] = OT_BOOL;
    mapOptType["use-beadmask"] = OT_STRING;
    mapOptType["exclusion-mask"] = OT_STRING;
    mapOptType["use-pinned"] = OT_BOOL;
    mapOptType["use-safe-buffer-model"] = OT_BOOL;
    mapOptType["vectorize"] = OT_BOOL;
    mapOptType["well-stat-file"] = OT_STRING;
    mapOptType["wells-compression"] = OT_INT;
    mapOptType["wells-convert-high"] = OT_DOUBLE;
    mapOptType["wells-convert-low"] = OT_DOUBLE;
    mapOptType["wells-convert-with-copies"] = OT_BOOL;
    mapOptType["wells-format"] = OT_STRING;
    mapOptType["wells-save-as-ushort"] = OT_BOOL;
    mapOptType["wells-save-flow"] = OT_INT;
    mapOptType["wells-save-freq"] = OT_INT;
    mapOptType["wells-save-number-copies"] = OT_BOOL;
    mapOptType["wells-save-queue-size"] = OT_INT;
    mapOptType["xtalk"] = OT_STRING;
    mapOptType["xtalk-correction"] = OT_BOOL;

    if(argc == 2)
    {
        string option = argv[1];
        if("-h" == option)
        {
            usageMain();
        }
        else if("-v" == option)
        {
            cerr << "perchipjsoneditor version: " << PER_CHIP_JSON_EDITOR_VERSION << endl;
            usageMain();
        }
    }

    string jsonDir("../config");
    if(argc > 1)
    {
        jsonDir = argv[1];
    }

    struct stat sb;
    if(!(stat(jsonDir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)))
    {
        cerr << "ERROR: " << jsonDir << " does not exist or it is not a directory. Please provide the correct directory where per chip json files locate." << endl;
        cout << "Please provide correct per chip json file directory (or type q to quit): ";
        cin >> jsonDir;

        if(jsonDir == "q")
        {
            cout << "Bye." << endl;
            exit(0);
        }

        if(!(stat(jsonDir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)))
        {
            cerr << "ERROR: " << jsonDir << " does not exist or it is not a directory. Please re-run the program with correct directory where per chip json files locate." << endl;
            exit ( EXIT_FAILURE );
        }
    }

    vector<string> vecChipType;
    map<string, int> mapChipType;

    string cmdline("ls ");
    cmdline += jsonDir;
    cmdline += "/args_*.json > ./tmpFileList.txt";
    int ret = system(cmdline.c_str());
    if(ret != 0)
    {
        cerr << "ERROR: There is no per chip json file at " << jsonDir << endl;
        exit ( EXIT_FAILURE );
    }

    char buf[2048];
    ifstream ifs("./tmpFileList.txt");
    while(ifs.getline(buf, 2048))
    {
        string chipType = buf;
        int index = chipType.rfind("/");
        if(index >= 0)
        {
            chipType = chipType.substr(index + 1, chipType.length() - index - 1);
        }

        chipType = chipType.substr(5, chipType.length() - 5);
        index = chipType.find("_");
        if(index > 0)
        {
            chipType = chipType.substr(0, index);
        }

        map<string, int>::iterator iter = mapChipType.find(chipType);
        if(iter == mapChipType.end())
        {
            mapChipType[chipType] = 1;
            vecChipType.push_back(chipType);
        }
        else
        {
            mapChipType[chipType]++;
        }
    }

    ifs.close();
    cmdline = "rm ./tmpFileList.txt";
    system(cmdline.c_str());

    if(vecChipType.empty())
    {
        cerr << "ERROR: There is no per chip json file at " << jsonDir << endl;
        exit ( EXIT_FAILURE );
    }

    sort(vecChipType.begin(), vecChipType.end());
    cout << "Available chip types are:" << endl;
    for(vector<string>::iterator iter1 = vecChipType.begin(); iter1 != vecChipType.end() - 1; ++iter1)
    {
        cout << *iter1 << ", ";
    }

    cout << vecChipType.back() << endl;

    vector<string> editList;
    string chipTypes;
    cout << "Please select chip type to edit ('q' to quit; 'n' for new; 'a' for all; separate multiple chip types with ','):" << endl;
    cin >> chipTypes;

    if(chipTypes == "q")
    {
        cout << "Bye." << endl;
        exit(0);
    }
    else if(chipTypes == "n")
    {
        cout << "New chip type name: ";
        cin >> chipTypes;
        string chipBase;
        cout << "Base chip type (data will be copied to new chip type): ";
        cin >> chipBase;
        if(chipBase.length() == 0)
        {
            chipBase = "530";
        }

        string srcName(jsonDir);
        srcName += "/args_";
        srcName += chipBase;
        srcName += "_analysis.json";
        string desName(jsonDir);
        desName += "/args_";
        desName += chipTypes;
        desName += "_analysis.json";
        cmdline = "cp ";
        cmdline += srcName;
        cmdline += " ";
        cmdline += desName;
        cout << cmdline;
        system(cmdline.c_str());

        srcName = jsonDir;
        srcName += "/args_";
        srcName += chipBase;
        srcName += "_beadfind.json";
        desName = jsonDir;
        desName += "/args_";
        desName += chipTypes;
        desName += "_beadfind.json";
        cmdline = "cp ";
        cmdline += srcName;
        cmdline += " ";
        cmdline += desName;
        cout << cmdline;
        system(cmdline.c_str());

        editList.push_back(chipTypes);
    }
    else if(chipTypes == "a")
    {
        chipTypes = "all chip types";
        editList = vecChipType;
    }
    else
    {
        int index = chipTypes.find(",");
        while(index > 0)
        {
            string chipType = chipTypes.substr(0, index);
            editList.push_back(chipType);
            chipTypes = chipTypes.substr(index + 1, chipTypes.length() - index - 1);
            index = chipTypes.find(",");
        }
        editList.push_back(chipTypes);
    }

    string chipType = editList[0];

    string analysisName(jsonDir);
    analysisName += "/args_";
    analysisName += chipType;
    analysisName += "_analysis.json";
    string beadfindName(jsonDir);
    beadfindName += "/args_";
    beadfindName += chipType;
    beadfindName += "_beadfind.json";

    ifstream ifsa(analysisName.c_str());
    Value jsona;
    Reader readera;
    readera.parse(ifsa, jsona, false);
    ifsa.close();

    ifstream ifsb(beadfindName.c_str());
    Value jsonb;
    Reader readerb;
    readerb.parse(ifsb, jsonb, false);
    ifsb.close();

    map<string, int> modules;
    map<string, string> opts;
    vector<string> tnOpts;
    Value::Members groupsa = jsona.getMemberNames();
    for(Value::Members::iterator ita1 = groupsa.begin(); ita1 != groupsa.end(); ++ita1)
    {
        if(*ita1 == "chipType")
        {
            string chipType0 = jsona["chipType"].asString();
            if(chipType != chipType0) // for new chip type
            {
                jsona["chipType"] = chipType;
                jsonb["chipType"] = chipType;
            }
        }
        else if(*ita1 == "ThumbnailControl")
        {
            Value::Members itemstn = jsona["ThumbnailControl"].getMemberNames();
            for(Value::Members::iterator itn = itemstn.begin(); itn != itemstn.end(); ++itn)
            {
                tnOpts.push_back(*itn);
            }
        }
        else
        {
            modules[*ita1] = 1;
            Value::Members itemsa = jsona[*ita1].getMemberNames();
            for(Value::Members::iterator ita2 = itemsa.begin(); ita2 != itemsa.end(); ++ita2)
            {
                opts[*ita2] = (*ita1);
            }
        }
    }

    Value::Members groupsb = jsonb.getMemberNames();
    for(Value::Members::iterator itb1 = groupsb.begin(); itb1 != groupsb.end(); ++itb1)
    {
        if(*itb1 != "chipType")
        {
            if(*itb1 == "ThumbnailControl")
            {
                Value::Members itemstn = jsonb["ThumbnailControl"].getMemberNames();
                for(Value::Members::iterator itn = itemstn.begin(); itn != itemstn.end(); ++itn)
                {
                    size_t k = 0;
                    for(; k < tnOpts.size(); ++k)
                    {
                        if(tnOpts[k] == (*itn))
                        {
                            break;
                        }
                    }
                    if(k == tnOpts.size())
                    {
                        tnOpts.push_back(*itn);
                    }
                }

            }
            else
            {
                map<string, int>::iterator iter1 = modules.find(*itb1);
                if(iter1 == modules.end())
                {
                    modules[*itb1] = 2;
                    Value::Members itemsb = jsonb[*itb1].getMemberNames();
                    for(Value::Members::iterator itb2 = itemsb.begin(); itb2 != itemsb.end(); ++itb2)
                    {
                        opts[*itb2] = (*itb1);
                    }
                }
                else
                {
                    modules[*itb1] = 3;
                }
            }
        }
    }

    cout << "Working on chip type " << chipType << endl;
    cout << "Which option you want to work on: " << endl
         << "\tfor searching: search" << endl
         << "\tfor adding new option: new" << endl
         << "\tfor listing all options: list" << endl
         << "\tfor thumbnail options: thumbnail" << endl;

    int dtype = -1;
    string optin;
    cin >> optin;

    bool tn = false;

    if(optin == "thumbnail")
    {
        tn = true;

        cout << "Existing thumbnail options:" << endl;
        size_t k = 0;
        for(; k < tnOpts.size(); ++k)
        {
            cout << k + 1 << " - " << tnOpts[k] << endl;
        }

        cout << "Please select an option you want to work on (type 0 for adding new option):" << endl;
        int n = 0;
        cin >> n;
        if(n > 0)
        {
            optin = tnOpts[n - 1];
        }
        else
        {
            cout << "Please type new option name you want to work on (list for listing all options):" << endl;
            cin >> optin;

            if(optin == "list")
            {
                for(map<string, string>::iterator iter = opts.begin(); iter != opts.end(); ++iter)
                {
                    cout << iter->first << endl;
                }

                cout << "Please type the option you want to work on:" << endl;
                cin >> optin;
            }
        }
    }
    else if(optin == "list")
    {
        for(map<string, string>::iterator iter = opts.begin(); iter != opts.end(); ++iter)
        {
            cout << iter->first << endl;
        }

        cout << "Please type the option you want to work on:" << endl;
        cin >> optin;
    }
    else if(optin == "search")
    {
        cout << "Please type partial of the option word:" << endl;
        cin >> optin;
        vector<string> names;
        for(map<string, string>::iterator iter = opts.begin(); iter != opts.end(); ++iter)
        {
            int idxsearch = iter->first.find(optin);
            if(idxsearch >= 0)
            {
                names.push_back(iter->first);
            }
        }

        if(names.empty())
        {
            cout << "There is no option containing " << optin << endl;
            cout << "Here is the full list of options:" << endl;
            for(map<string, string>::iterator iter = opts.begin(); iter != opts.end(); ++iter)
            {
                cout << iter->first << endl;
            }

            cout << "Please type the option you want to work on:" << endl;
            cin >> optin;
        }
        else if(names.size() == 1)
        {
            optin = names[0];
            cout << "Working on option " << optin << endl;
        }
        else
        {
            cout << "Here is the list of options containing " << optin << endl;
            for(size_t i = 0; i < names.size(); ++i)
            {
                cout << i << " - " << names[i] << endl;
            }

            cout << "Please type the option you want to work on:" << endl;
            int k;
            cin >> k;
            optin = names[k];
        }
    }
    if(optin == "new")
    {
        cout << "Please type new option name:" << endl;
        cin >> optin;
        cout << "Which module you want to add " << optin << " to:" << endl;
        for(map<string, int>::iterator iter = modules.begin(); iter != modules.end(); ++ iter)
        {
            cout << iter->first << endl;
        }

        string mod;
        cin >> mod;
        opts[optin] = mod;

        cout << "option data type: 0 - bool; 1 - integer; 2 - double; 3 - string; 4 - vector of integers; 5 - vector 0f doubles." << endl;
        cin >> dtype;
    }

    int idx = -1;
    bool val0;
    int val1, vali;
    double val2, valf;
    string val3, val4, val5;

    string mod = opts[optin];
    string mod2 = mod;
    if(tn)
    {
        mod = "ThumbnailControl";
    }

    cout << "option's value:" << endl;
    if(dtype < 0)
    {
        dtype = mapOptType[optin];
    }

    switch (dtype)
    {
    case OT_BOOL:       
        cin >> val0;

        if(modules[mod2] == 1)
        {
            jsona[mod][optin] = val0;
        }
        else if(modules[mod2] == 2)
        {
            jsonb[mod][optin] = val0;
        }
        else if(modules[mod2] == 3)
        {
            jsona[mod][optin] = val0;
            jsonb[mod][optin] = val0;
        }

        break;

    case OT_INT:      
        cin >> val1;

        if(modules[mod2] == 1)
        {
            jsona[mod][optin] = val1;
        }
        else if(modules[mod2] == 2)
        {
            jsonb[mod][optin] = val1;
        }
        else if(modules[mod2] == 3)
        {
            jsona[mod][optin] = val1;
            jsonb[mod][optin] = val1;
        }

        break;

    case OT_DOUBLE:     
        cin >> val2;

        if(modules[mod2] == 1)
        {
            jsona[mod][optin] = val2;
        }
        else if(modules[mod2] == 2)
        {
            jsonb[mod][optin] = val2;
        }
        else if(modules[mod2] == 3)
        {
            jsona[mod][optin] = val2;
            jsonb[mod][optin] = val2;
        }

        break;

    case OT_STRING:      
        cin >> val3;

        if(modules[mod2] == 1)
        {
            jsona[mod][optin] = val3;
        }
        else if(modules[mod2] == 2)
        {
            jsonb[mod][optin] = val3;
        }
        else if(modules[mod2] == 3)
        {
            jsona[mod][optin] = val3;
            jsonb[mod][optin] = val3;
        }

        break;

    case OT_VECTOR_INT:
        if(modules[mod2] == 1)
        {
            jsona[mod][optin].clear();
        }
        else if(modules[mod2] == 2)
        {
            jsonb[mod][optin].clear();
        }
        else if(modules[mod2] == 3)
        {
            jsona[mod][optin].clear();
            jsonb[mod][optin].clear();
        }

        cin >> val3;
        val4 = val3;
        idx = val4.find(",");
        while(idx > 0)
        {
            string s2 = val4.substr(0, idx);
            vali = atoi(s2.c_str());
            if(modules[mod2] == 1)
            {
                jsona[mod][optin].append(vali);
            }
            else if(modules[mod2] == 2)
            {
                jsonb[mod][optin].append(vali);
            }
            else if(modules[mod2] == 3)
            {
                jsona[mod][optin].append(vali);
                jsonb[mod][optin].append(vali);
            }

            val4 = val4.substr(idx + 1, val4.length() - idx - 1);
            idx = val4.find(",");
        }

        vali = atoi(val4.c_str());
        if(modules[mod2] == 1)
        {
            jsona[mod][optin].append(vali);
        }
        else if(modules[mod2] == 2)
        {
            jsonb[mod][optin].append(vali);
        }
        else if(modules[mod2] == 3)
        {
            jsona[mod][optin].append(vali);
            jsonb[mod][optin].append(vali);
        }

        break;

    case OT_VECTOR_DOUBLE:
        if(modules[mod2] == 1)
        {
            jsona[mod][optin].clear();
        }
        else if(modules[mod2] == 2)
        {
            jsonb[mod][optin].clear();
        }
        else if(modules[mod2] == 3)
        {
            jsona[mod][optin].clear();
            jsonb[mod][optin].clear();
        }

        cin >> val3;
        val5 = val3;
        idx = val5.find(",");
        while(idx > 0)
        {
            string s2 = val5.substr(0, idx);
            valf = atof(s2.c_str());
            if(modules[mod2] == 1)
            {
                jsona[mod][optin].append(valf);
            }
            else if(modules[mod2] == 2)
            {
                jsonb[mod][optin].append(valf);
            }
            else if(modules[mod2] == 3)
            {
                jsona[mod][optin].append(valf);
                jsonb[mod][optin].append(valf);
            }

            val5 = val5.substr(idx + 1, val5.length() - idx - 1);
            idx = val5.find(",");
        }

        valf = atof(val5.c_str());
        if(modules[mod2] == 1)
        {
            jsona[mod][optin].append(valf);
        }
        else if(modules[mod2] == 2)
        {
            jsonb[mod][optin].append(valf);
        }
        else if(modules[mod2] == 3)
        {
            jsona[mod][optin].append(valf);
            jsonb[mod][optin].append(valf);
        }

        break;

    default:
        cout << "ERROR: Invalid option data type!" << endl;
        return(1);
    }

    saveJson(jsona, analysisName);
    saveJson(jsonb, beadfindName);

    for(size_t i = 1; i < editList.size(); ++i)
    {
        chipType = editList[i];
        cout << "Working on chip type " << chipType << endl;

        analysisName = jsonDir;
        analysisName += "/args_";
        analysisName += chipType;
        analysisName += "_analysis.json";
        beadfindName = jsonDir;
        beadfindName += "/args_";
        beadfindName += chipType;
        beadfindName += "_beadfind.json";

        ifstream ifsa2(analysisName.c_str());
        Value jsona2;
        Reader readera2;
        readera2.parse(ifsa2, jsona2, false);
        ifsa2.close();

        ifstream ifsb2(beadfindName.c_str());
        Value jsonb2;
        Reader readerb2;
        readerb2.parse(ifsb2, jsonb2, false);
        ifsb2.close();

        switch (dtype)
        {
        case OT_BOOL:
            if(modules[mod2] == 1)
            {
                jsona2[mod][optin] = val0;
            }
            else if(modules[mod2] == 2)
            {
                jsonb2[mod][optin] = val0;
            }
            else if(modules[mod2] == 3)
            {
                jsona2[mod][optin] = val0;
                jsonb2[mod][optin] = val0;
            }

            break;

        case OT_INT:
            if(modules[mod2] == 1)
            {
                jsona2[mod][optin] = val1;
            }
            else if(modules[mod2] == 2)
            {
                jsonb2[mod][optin] = val1;
            }
            else if(modules[mod2] == 3)
            {
                jsona2[mod][optin] = val1;
                jsonb2[mod][optin] = val1;
            }

            break;

        case OT_DOUBLE:
            if(modules[mod2] == 1)
            {
                jsona2[mod][optin] = val2;
            }
            else if(modules[mod2] == 2)
            {
                jsonb2[mod][optin] = val2;
            }
            else if(modules[mod2] == 3)
            {
                jsona2[mod][optin] = val2;
                jsonb2[mod][optin] = val2;
            }

            break;

        case OT_STRING:
            if(modules[mod2] == 1)
            {
                jsona2[mod][optin] = val3;
            }
            else if(modules[mod2] == 2)
            {
                jsonb2[mod][optin] = val3;
            }
            else if(modules[mod2] == 3)
            {
                jsona2[mod][optin] = val3;
                jsonb2[mod][optin] = val3;
            }

            break;

        case OT_VECTOR_INT:
            if(modules[mod2] == 1)
            {
                jsona2[mod][optin].clear();
            }
            else if(modules[mod2] == 2)
            {
                jsonb2[mod][optin].clear();
            }
            else if(modules[mod2] == 3)
            {
                jsona2[mod][optin].clear();
                jsonb2[mod][optin].clear();
            }

            val4 = val3;
            idx = val4.find(",");
            while(idx > 0)
            {
                string s2 = val4.substr(0, idx);
                vali = atoi(s2.c_str());
                if(modules[mod2] == 1)
                {
                    jsona2[mod][optin].append(vali);
                }
                else if(modules[mod2] == 2)
                {
                    jsonb2[mod][optin].append(vali);
                }
                else if(modules[mod2] == 3)
                {
                    jsona2[mod][optin].append(vali);
                    jsonb2[mod][optin].append(vali);
                }

                val4 = val4.substr(idx + 1, val4.length() - idx - 1);
                idx = val4.find(",");
            }

            vali = atoi(val4.c_str());
            if(modules[mod2] == 1)
            {
                jsona2[mod][optin].append(vali);
            }
            else if(modules[mod2] == 2)
            {
                jsonb2[mod][optin].append(vali);
            }
            else if(modules[mod2] == 3)
            {
                jsona2[mod][optin].append(vali);
                jsonb2[mod][optin].append(vali);
            }

            break;

        case OT_VECTOR_DOUBLE:
            if(modules[mod2] == 1)
            {
                jsona2[mod][optin].clear();
            }
            else if(modules[mod2] == 2)
            {
                jsonb2[mod][optin].clear();
            }
            else if(modules[mod2] == 3)
            {
                jsona2[mod][optin].clear();
                jsonb2[mod][optin].clear();
            }

            val5 = val3;
            idx = val5.find(",");
            while(idx > 0)
            {
                string s2 = val5.substr(0, idx);
                valf = atof(s2.c_str());
                if(modules[mod2] == 1)
                {
                    jsona2[mod][optin].append(valf);
                }
                else if(modules[mod2] == 2)
                {
                    jsonb2[mod][optin].append(valf);
                }
                else if(modules[mod2] == 3)
                {
                    jsona2[mod][optin].append(valf);
                    jsonb2[mod][optin].append(valf);
                }

                val5 = val5.substr(idx + 1, val5.length() - idx - 1);
                idx = val5.find(",");
            }

            valf = atof(val5.c_str());
            if(modules[mod2] == 1)
            {
                jsona2[mod][optin].append(valf);
            }
            else if(modules[mod2] == 2)
            {
                jsonb2[mod][optin].append(valf);
            }
            else if(modules[mod2] == 3)
            {
                jsona2[mod][optin].append(valf);
                jsonb2[mod][optin].append(valf);
            }

            break;

        default:
            break;
        }

        saveJson(jsona2, analysisName);
        saveJson(jsonb2, beadfindName);
    }

    return (0);
}
