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
#include <iostream>
#include "IonVersion.h"
#include "LinuxCompat.h"
#include "Utils.h"
#include "RawWells.h"

void output_debug_info(std::ostream &out, RawWells &wells) {
  int rowStart, height, colStart, width;
  wells.GetRegion(rowStart,height, colStart, width);
  out << "w:" << rowStart << std::endl;
  out << "w:" << height << std::endl;
  out << "w:" << colStart << std::endl;
  out << "w:" << width << std::endl;
}

int merge_wells(int total_nb_cols,
                int total_nb_rows,
                const std::vector<int>& xoffsets,
                const std::vector<int>& yoffsets,
                const std::vector<std::string>& folders,
                const char* wellsFileName,
                const char* outputFolder,
                const char* outputFileName,
                bool debug)
{

    //output
    RawWells wells_out(outputFolder,outputFileName);

    for (unsigned int f=0;f<folders.size();f++) {

        if (debug) {
            std::cerr << "folders: " << folders[f] << std::endl;
            std::cerr << "offsetx: " << xoffsets[f] << std::endl;
            std::cerr << "offsety: " << yoffsets[f] << std::endl;
        }
        RawWells wells_in(folders[f].c_str(), wellsFileName);
        bool stat = wells_in.OpenForRead();
        if (stat) {
            fprintf (stdout, "# ERROR: Could not open %s/%s\n", folders[f].c_str(), wellsFileName);
            exit (1);
        }
        int cur_nb_rows  = wells_in.NumRows();
        int cur_nb_cols  = wells_in.NumCols();
        int cur_nb_wells = wells_in.NumWells();
        int cur_nb_flows = wells_in.NumFlows();
        const char* cur_flow_order = wells_in.FlowOrder();

        if (debug) {
            std::cerr << "w:" << cur_nb_rows << std::endl;
            std::cerr << "w:" << cur_nb_cols << std::endl;
            std::cerr << "w:" << cur_nb_wells << std::endl;
            std::cerr << "w:" << cur_nb_flows << std::endl;
            std::cerr << "w:" << cur_flow_order << std::endl;
            output_debug_info(std::cerr, wells_in);
        }

        if (f==0) {

            wells_out.CreateEmpty(cur_nb_flows, cur_flow_order, total_nb_rows, total_nb_cols);
            wells_out.OpenForWrite();
            wells_out.SetCompression(0);

            if (debug) {
                std::cerr << "out:" << wells_out.NumWells() << std::endl;
                std::cerr << "out:" << wells_out.NumCols() << std::endl;
                std::cerr << "out:" << wells_out.NumRows() << std::endl;
                std::cerr << "out:" << wells_out.NumFlows() << std::endl;
                std::cerr << "out:" << wells_out.FlowOrder() << std::endl;
                output_debug_info(std::cerr, wells_out);
            }
        }

#if 1
        // slow solution, unable to set rank
        const WellData* well = NULL;
        for (int x=0; x<cur_nb_cols; x++) {
            for (int y=0; y<cur_nb_rows; y++) {
                well = wells_in.ReadXY(x, y);
                if (debug) {
                    std::cout << "in x: " << x << ", y: " << y << std::endl;
                }
                for (int flow=0; flow<cur_nb_flows; flow++) {
                    if (debug) {
                        std::cerr << well->flowValues[flow] << " ";
                    }
                    wells_out.WriteFlowgram(flow, x+xoffsets[f], y+yoffsets[f], well->flowValues[flow]);
                }
            }
        }
#else
        // fast solution, missing functionality, TODO
        WellData o;
        o.flowValues = (float *) malloc (wells_in.NumFlows() * sizeof(float));
        size_t count = 0;
        while(!wells_in.ReadNextData(&o)) {
            // mRows = std::max((uint64_t)o.y, mRows);
            // mCols = std::max((uint64_t)o.x, mCols);
            wells_out.mRankData[count] = o.rank;
            copy(o.flowValues, o.flowValues + wells_out.mFlows, wells_out.mFlowData.begin() + count * wells_out.mFlows);
            wells_out.mIndexes[count] = count;
            count++;
        }
#endif

        wells_in.Close();
    }
    if (debug) {
        std::cerr << "sout: flows: " << wells_out.NumFlows() << std::endl;
        std::cerr << "sout: wells: " << wells_out.NumWells() << std::endl;
        std::cerr << "sout: cols: " << wells_out.NumCols() << std::endl;
        std::cerr << "sout: rows: " << wells_out.NumRows() << std::endl;
    }
    wells_out.WriteLegacyWells(); 

    return 0;
}


// File IO, opens processParameters.txt in main folder (see GetProcessParam)
// Writes out one output wells file in /tmp
// extract offsets from folder names
// Reads in wells files from folders


int main (int argc, char *argv[])
{

    // process command-line args
    char* wellsFileName = NULL;
    char* outputFileName = NULL;
    char* outputFolder = NULL;
    std::vector<std::string> folders;
    bool debug = false;
    int c;
    outputFolder = strdup(".");
    while ( (c = getopt (argc, argv, "i:o:f:hvd")) != -1 )
        {
            switch (c)
                {
                case 'i': wellsFileName = strdup(optarg); break;
                case 'o': outputFileName = strdup(optarg); break;
                case 'f': outputFolder = strdup(optarg); break;
                case 'h':
                    fprintf (stdout, "%s -i in_filename -o out_filename folders \n", argv[0]);
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
        folders.push_back(argv[c]);
    }

    if (!wellsFileName) {
        fprintf (stderr, "No input file specified\n");
        exit (1);
    }
    else {
        fprintf (stdout, "Reading from file: %s\n", wellsFileName);
    }

    if (folders.size() < 1) {
        fprintf (stderr, "No input directories specified\n");
        exit (1);
    }
    else {
        for (unsigned int f=0;f<folders.size();f++) {
            fprintf (stdout, "Reading from folder: %s\n", folders[f].c_str());
        }
    }

    if (!outputFileName) {
        fprintf (stderr, "No output file specified\n");
        exit (1);
    }
    else {
        fprintf (stdout, "Writing into file: %s\n", outputFileName);
    }

    int total_nb_cols = 0;
    int total_nb_rows = 0;
    std::vector<int> xoffsets = std::vector<int>(folders.size());
    std::vector<int> yoffsets = std::vector<int>(folders.size());

    for (unsigned int f=0;f<folders.size();f++) {

        if (f==0) {
            // extract chip size
            char* size = GetProcessParam (folders[0].c_str(), "Chip" );
            total_nb_cols=atoi(strtok (size,","));
            total_nb_rows =atoi(strtok(NULL, ","));
            if (debug) {
                std::cout << "chip size: " << total_nb_cols << "," << total_nb_rows << std::endl;
            }
        }

        // extract offsets from folder name
        char* size = GetProcessParam (folders[f].c_str(), "Block" );
        xoffsets[f]=atoi(strtok (size,","));
        yoffsets[f]=atoi(strtok(NULL, ","));
        if (debug) {
            std::cerr << "xoffset: " << xoffsets[f] << std::endl;
            std::cerr << "yoffset: " << yoffsets[f] << std::endl;
        }
    }

    int ret = merge_wells(total_nb_cols,
                          total_nb_rows,
                          xoffsets,
                          yoffsets,
                          folders,
                          wellsFileName,
                          outputFolder,
                          outputFileName,
                          debug);

    free(wellsFileName);
    free(outputFolder);
    free(outputFileName);

    return ret;
}
