/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include <iostream>
#include <sstream>
#include "BkgControlOpts.h"


void BkgModelControlOpts::DefaultBkgModelControl()
{
    bkg_debug_files = 0;
    bkgModelHdf5Debug = 1;
    bkgModelHdf5Debug_region_r = -1;
    bkgModelHdf5Debug_region_c = -1;
    bkgModel_xyflow_output = false;
    bkgModel_xyflow_fname_in_type = -1;
    bkgModel_xyflow_fname_in = "";
    bkg_model_emphasis_width = 32.0;
    bkg_model_emphasis_amplitude = 4.0;
    for (int i_nuc=0; i_nuc<4; i_nuc++) dntp_uM[i_nuc] = -1.0f;
    AmplLowerLimit = 0.001;
    bkgModelMaxIter = 17;
    gopt = "default"; // "default" enables per-chip optimizations; other options: "disable" use the old hard-coded defaults, "opt" used only during optimization, and path to any optimized param file would load the file.
    xtalk = "disable";
    //xtalk= NULL;
    for (int i=0;i<4;i++)
    {
        krate[i] = -1.0;
        diff_rate[i] = -1.0;
        kmax[i] = -1.0;
    }
    no_rdr_fit_first_20_flows = 0;
    fitting_taue = 0;
    var_kmult_only = 0;
    generic_test_flag = 0;
    fit_alternate = 0;
    fit_gauss_newton = 0;
    emphasize_by_compression=1; // by default turned to the old method
    BkgTraceDebugRegions.clear();
    bkgDebugParam = 0;

    enableXtalkCorrection = true;
    enable_dark_matter = true;
    enableBkgModelClonalFilter = true;
    updateMaskAfterBkgModel = true;

    // options for replay
    replayBkgModelData = false;
    recordBkgModelData = false;

    restart = false;
    restart_from = "";
    restart_next = "";
    restart_check = true;

    damp_kmult = 0;
    kmult_hi_limit = 1.75;
    kmult_low_limit = 0.65;
    krate_adj_threshold = 2.0;

    ssq_filter = 0.0f; // no filtering
    
    // how to do computation
    vectorize = 1;
    gpuControl.DefaultGpuControl();

    numCpuThreads = 0;
    readaheadDat = 0;
    saveWellsFrequency = 3;
    wellsCompression = 3;
    useProjectionSearchForSingleFlowFit = false;
    choose_time = 0; // default standard time compression
    exp_tail_fit = false;
    pca_dark_matter = false;

    use_dud_and_empty_wells_as_reference = false;
    proton_dot_wells_post_correction = false;
    empty_well_normalization = false;
    single_flow_fit_max_retry = 0;
    per_flow_t_mid_nuc_tracking = false;
    regional_sampling = false;
    //regional_sampling_type = REGIONAL_SAMPLING_SYSTEMATIC;
    regional_sampling_type = REGIONAL_SAMPLING_CLONAL_KEY_NORMALIZED;
    prefilter_beads = false;

    unfiltered_library_random_sample = 100000;

    // diagnostics
    debug_bead_only = 1;  // only debug bead
    region_vfrc_debug = 0; // off
    // emptyTrace outlier (wild trace) removal
    do_ref_trace_trim = false;
    span_inflator_min = 10;
    span_inflator_mult = 10;
    cutoff_quantile = .2;

    region_list = "";
    nokey = false;
}


bool BkgModelControlOpts::read_file_sse(HashTable_xyflow &xyf_hash, int numFlows)
{
    xyf_hash.clear();
    xyf_hash.setFilename(bkgModel_xyflow_fname_in);
    //xyf_hash.set_xyflow_limits(numFlows);
    std::ifstream infile;
    infile.open(bkgModel_xyflow_fname_in.c_str());
    if (infile.is_open()) { /* ok, proceed with output */
        std::string line;
        //std::getline (infile,line); // header line
        int nLines = 0;
        while(!infile.eof()){
          //read data from file
          std::getline (infile,line);
          if (line.length()==0)
              break;
          nLines++;
          std::vector<std::string> tokens;
          std::vector<std::string> elem;
          split(line,'\t',tokens);
          if (tokens.size()<9) {
              split(line,' ',tokens);
              if (tokens.size()<9) {
                  std::cerr << "read_file_sse() parsing error: not enough number of tokens in line " << nLines << ": " << line << std::endl << std::flush;
                  //assert(tokens.size()>=9);
                  break;
              }
          }
          split(tokens[1],':',elem);
          int nTokens = elem.size();
          int r = atoi(elem[nTokens-2].c_str());
          int c = atoi(elem[nTokens-1].c_str());
          int f = atoi(tokens[4].c_str());
          bool mm = tokens[6].compare("1")==0 ? 1:0;
          std::string hp = tokens[8];
          xyf_hash.insert_rcflow(r,c,f,mm,hp);
          xyf_hash.insert_rc(r,c);
          //std::cout << r << " " << c << " " << flow << std::endl;
        }
        std::cout << nLines << " lines read by read_file_sse()..."  << std::endl << std::flush;
        std::cout << "xyf_hash.size() = " << xyf_hash.size()  << std::endl << std::flush;
        std::cout << "xyf_hash.size_xy() = " << xyf_hash.size_xy()  << std::endl << std::flush;
        //xyf_hash.print();
    }
    else {
        std::cerr << "read_file_sse() open error!!!"  << std::endl << std::flush;
        return (false);
    }
    infile.close();
    return (true);
}


bool BkgModelControlOpts::read_file_rcflow(HashTable_xyflow &xyf_hash, int numFlows)
{
    xyf_hash.clear();
    xyf_hash.setFilename(bkgModel_xyflow_fname_in);
    //xyf_hash.set_xyflow_limits(numFlows);
    std::ifstream infile;
    infile.open(bkgModel_xyflow_fname_in.c_str());
    if (infile.is_open()) { /* ok, proceed with output */
        std::string line;
        int nLines = 0;
        while(!infile.eof()){
          //read data from file
          std::getline (infile,line);
          if (line.length()==0)
              break;
          nLines++;
          std::vector<std::string> tokens;
          std::vector<std::string> elem;
          split(line,'\t',tokens);
          if (tokens.size()<2) {
              split(line,' ',tokens);
              if (tokens.size()<2) {
                  std::cerr << "read_file_rcflow() parsing error: not enough number of tokens in line: " << line << std::endl << std::flush;
                  assert(tokens.size()>=2);
              }
          }
          split(tokens[0],':',elem);
          //int nTokens = elem.size();
          int r = atoi(tokens[0].c_str());
          int c = atoi(tokens[1].c_str());
          int f = atoi(tokens[2].c_str());
          xyf_hash.insert_rcflow(r,c,f);
          xyf_hash.insert_rc(r,c);
          //std::cout << r << " " << c << " " << flow << std::endl;
        }
        std::cout << nLines << " lines read by read_file_rcflow()..."  << std::endl << std::flush;
        //xyf_hash.print();
    }
    else {
        std::cerr << "read_file_rcflow() open error!!!"  << std::endl << std::flush;
        return (false);
    }
    infile.close();
    return (true);
}


bool BkgModelControlOpts::read_file_xyflow(HashTable_xyflow &xyf_hash, int numFlows)
{
    xyf_hash.clear();
    xyf_hash.setFilename(bkgModel_xyflow_fname_in);
    //xyf_hash.set_xyflow_limits(numFlows);
    std::ifstream infile;
    infile.open(bkgModel_xyflow_fname_in.c_str());
    if (infile.is_open()) { /* ok, proceed with output */
        std::string line;
        int nLines = 0;
        while(!infile.eof()){
          //read data from file
          std::getline (infile,line);
          if (line.length()==0)
              break;
          nLines++;
          std::vector<std::string> tokens;
          split(line,' ',tokens);
          int x = atoi(tokens[0].c_str());
          int y = atoi(tokens[1].c_str());
          int f = atoi(tokens[2].c_str());
          xyf_hash.insert_xyflow(x,y,f);
          xyf_hash.insert_xy(x,y);
          //std::cout << r << " " << c << " " << flow << std::endl;
        }
        std::cout << nLines << " lines read by read_file_xyflow()..."  << std::endl << std::flush;
        //xyf_hash.print();
    }
    else {
        std::cerr << "read_file_xyflow() open error!!!"  << std::endl << std::flush;
        return (false);
    }
    infile.close();
    return (true);
}

