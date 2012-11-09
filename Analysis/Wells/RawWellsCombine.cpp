/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
//
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <vector>
#include "IonVersion.h"
#include "LinuxCompat.h"
#include "Utils.h"
#include "RawWells.h"

int merge_wells(const std::vector<std::string>& wellsFileNames,
                const char* outputFileName,
                bool debug)
{

    //output
    RawWells wells_out("",outputFileName);
    //WellData
    std::vector<RawWells*> wells_in;
    std::vector<WellData*> wells_data;
    const char *flow_order = NULL;
    int total_nb_rows = 0;
    int total_nb_cols = 0;
    int total_nb_flows = 0;
        
    if (debug) {
        std::cerr << "Opening the files" << std::endl;
    }

    // open the files
    for (unsigned int f=0;f<wellsFileNames.size();f++) {
        RawWells *wells = new RawWells("", wellsFileNames[f].c_str());
        wells->OpenForIncrementalRead(); // read a chunk at a time
        wells->ResetCurrentRegionWell(); // initialize iterator over the wells
        wells_in.push_back(wells);
        wells_data.push_back(new WellData());

        int cur_nb_rows  = wells_in[f]->NumRows();
        int cur_nb_cols  = wells_in[f]->NumCols();
        //int cur_nb_wells = wells_in[f]->NumWells();
        int cur_nb_flows = wells_in[f]->NumFlows();
        const char* cur_flow_order = wells_in[f]->FlowOrder();
        if (0 == f) {
            flow_order = cur_flow_order;
            total_nb_rows = cur_nb_rows;
            total_nb_cols = cur_nb_cols;
            total_nb_flows = cur_nb_flows;

            if (debug) {
                std::cerr << flow_order << std::endl;
                std::cerr << total_nb_rows << std::endl;
                std::cerr << total_nb_cols << std::endl;
                std::cerr << total_nb_flows << std::endl;
            }
            
            // open the output file

            wells_out.CreateEmpty(cur_nb_flows, cur_flow_order, total_nb_rows, total_nb_cols);
            wells_out.SetChunk(0, total_nb_rows, 0, total_nb_cols, 0, total_nb_flows);
            wells_out.SetCompression(3); // by default
            wells_out.SetRows(total_nb_rows);
            wells_out.SetCols(total_nb_cols);
            wells_out.SetFlows(total_nb_flows);
            wells_out.SetFlowOrder(flow_order);
            wells_out.OpenForWrite();
        }
        else {
            if (total_nb_rows != cur_nb_rows) {
                fprintf (stderr, "# of rows did not match\n");
                exit (1);
            }
            else if (total_nb_cols != cur_nb_cols) {
                fprintf (stderr, "# of cols did not match\n");
                exit (1);
            }
            else if (total_nb_flows != cur_nb_flows) {
                fprintf (stderr, "# of flows did not match\n");
                exit (1);
            }
            else if (0 != strcmp(cur_flow_order, flow_order)){
                fprintf (stderr, "flow orders did not match\n");
                exit (1);
            }
        }
    }

    // iterate over the wells
    while(1) {
        bool finished = false;

        if (debug) {
            std::cerr << "Loading data" << std::endl;
        }

        // read over regions
        for (unsigned int f=0;f<wellsFileNames.size();f++) {
            if (wells_in[f]->ReadNextRegionData(wells_data[f])) {
                finished = true;
            }
            else if (finished) {
                fprintf (stderr, "Error: premature EOF!\n");
                exit (1);
            }
        }

        if (finished) {
            break;
        }
        
        if (debug) {
            std::cerr << "Combining data" << std::endl;
        }

        // combine
        // NB: uses the first file
        for (int flow=0; flow<total_nb_flows; flow++) { // for each flow
            for (unsigned int f=1;f<wellsFileNames.size();f++) { // for each file
                wells_data[0]->flowValues[flow] += wells_data[f]->flowValues[flow]; // sum
            }
            // average
            wells_data[0]->flowValues[flow] /= wells_data.size();
        }

        if (debug) {
            std::cerr << "Writing data" << std::endl;
        }

        // write the data
        for (int flow=0; flow<total_nb_flows; flow++) {
            wells_out.WriteFlowgram(flow, wells_data[0]->x, wells_data[0]->y, wells_data[0]->flowValues[flow]);
        }
        
    }
        
    if (debug) {
        std::cerr << "Finishing" << std::endl;
    }

    // write wells, ranks, and info
    wells_out.WriteWells();
    wells_out.WriteRanks();
    wells_out.WriteInfo();

    // close the files and destroy the data
    for (unsigned int f=0;f<wellsFileNames.size();f++) {
        // close
        wells_in[f]->Close();
        // destroy
        delete wells_in[f];
        wells_in[f] = NULL;
        wells_data[f]->flowValues = NULL;
        delete wells_data[f];
        wells_data[f] = NULL;

    }
    wells_out.Close();
        
    if (debug) {
        std::cerr << "Returning" << std::endl;
    }

    return 0;
}

int main (int argc, char *argv[])
{
    // process command-line args
    char* outputFileName = NULL;
    std::vector<std::string> wellsFileNames;
    bool debug = false;
    int c;
    while ( (c = getopt (argc, argv, "o:hvd")) != -1 )
        {
            switch (c)
                {
                case 'o': outputFileName = strdup(optarg); break;
                case 'h':
                    fprintf (stdout, "%s -o <out_filename> <in1.wells> <in2.wells> [...]\n", argv[0]);
                    exit (0);
                    break;
                case 'v':   //version
                    fprintf (stdout, "%s", IonVersion::GetFullVersion("MergeWells").c_str());
                    return (0);
                    break;
                case 'd':       // enable debug print outs
                    debug = true;
                    break;
                default:
                    fprintf (stdout, "whatever");
                    break;
                }
        }

    for (c = optind; c < argc; c++) {
        wellsFileNames.push_back(argv[c]);
    }

    if (wellsFileNames.size() < 2) {
        fprintf (stderr, "Two or more wells files need to be specified\n");
        exit (1);
    }
    else {
        for (unsigned int f=0;f<wellsFileNames.size();f++) {
            fprintf (stdout, "Reading from wells: %s\n", wellsFileNames[f].c_str());
        }
    }

    if (!outputFileName) {
        fprintf (stderr, "No output file specified\n");
        exit (1);
    }
    else {
        fprintf (stdout, "Writing into file: %s\n", outputFileName);
    }

    int ret = merge_wells(wellsFileNames,
                          outputFileName,
                          debug);

    free(outputFileName);

    return ret;
}
