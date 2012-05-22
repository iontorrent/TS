/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
// basic sff merger

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include "IonVersion.h"
#include "Utils.h"

#include "sff.h"
#include "sff_file.h"
#include "sff_header.h"
#include "sff_read_header.h"
#include "sff_read.h"


void sff_add_offset(int xoffset, int yoffset, sff_read_header_t *rh) {
    char name_copy[rh->name_length+1];
    strcpy (name_copy,rh->name->s);
    char* basename = strtok (name_copy,":");
    int row = atoi(strtok(NULL, ":"));
    int col = atoi(strtok(NULL, ":"));
    free(rh->name->s);
    rh->name_length= asprintf(&rh->name->s, "%s:%d:%d", basename, row+yoffset, col+xoffset);
}

int merge_sff(
              const std::vector<int>& xoffset,
              const std::vector<int>& yoffset,
              const std::vector<char*>& folders,
              const char* mergeFileName,
              const char* inputSFFFileName,
              bool debug
              )
{

    sff_file_t* sff_file_in = NULL;
    sff_file_t* sff_file_out = NULL;
    std::vector<std::string> sffFileNames;

    // stores updated header information
    sff_header_t* fileheader_clone = NULL;

    // for each input sff file:
    for (unsigned int j=0; j<folders.size(); j++) {

        sffFileNames.push_back(std::string(folders[j])+"/"+std::string(inputSFFFileName));

        if (debug) {
            printf("Reading file: %s\n", sffFileNames[j].c_str());
        }
        sff_file_in = sff_fopen(sffFileNames[j].c_str(), "rb", NULL, NULL);


        /// header section ///
        if (debug) {
            sff_header_print(stdout, sff_file_in->header);
        }

        // write the initial/updated header information
        if (j==0) {
            // copy header from first sff file
            fileheader_clone = sff_header_clone(sff_file_in->header);
            sff_file_out = sff_fopen(mergeFileName, "wb", fileheader_clone, NULL);
        } else {
            assert(fileheader_clone->magic           == sff_file_in->header->magic);
            assert(fileheader_clone->version         == sff_file_in->header->version);
            assert(fileheader_clone->index_offset    == sff_file_in->header->index_offset);
            assert(fileheader_clone->index_length    == sff_file_in->header->index_length);
            sff_file_out->header->n_reads += sff_file_in->header->n_reads;
            assert(fileheader_clone->gheader_length  == sff_file_in->header->gheader_length);
            assert(fileheader_clone->key_length      == sff_file_in->header->key_length);
            assert(fileheader_clone->flow_length     == sff_file_in->header->flow_length);
            assert(fileheader_clone->flowgram_format == sff_file_in->header->flowgram_format);
            for(int i=0;i<fileheader_clone->key_length;i++) {
               assert(fileheader_clone->key->s[i] == sff_file_in->header->key->s[i]);
            }

            // assert(flow == flow);
            // assert(key == key);
        }
        fseek(sff_file_out->fp, 0, SEEK_SET);
        sff_header_write(sff_file_out->fp, sff_file_out->header);
        fseek(sff_file_out->fp, 0, SEEK_END);


        //// Reads section ////
        for(unsigned int i=0; i<sff_file_in->header->n_reads; i++) {
            sff_read_header_t* rh = sff_read_header_read(sff_file_in->fp);
            sff_read_t* rr = sff_read_read(sff_file_in->fp, sff_file_in->header, rh);

            if (debug) {
                sff_read_header_print(stdout, rh);
            }

            sff_add_offset(xoffset[j], yoffset[j], rh);

            sff_read_header_write(sff_file_out->fp, rh);
            sff_read_write(sff_file_out->fp, sff_file_out->header, rh, rr);

            sff_read_header_destroy(rh);
            sff_read_destroy(rr);
        }

        /// Cleanup ///
        sff_fclose(sff_file_in);
    }
    sff_fclose(sff_file_out);

    return 0;
}



int main(int argc, char *argv[])
{
    char* mergeFileName = NULL;
    char* inputSFFFileName = NULL;

    std::vector<int> xoffset;
    std::vector<int> yoffset;
    std::vector<char*> folders;
    bool debug = false;
    int c;
    while ( (c = getopt (argc, argv, "i:o:hvd")) != -1 )
        {
            switch (c)
                {
                case 'i': inputSFFFileName = strdup(optarg); break;
                case 'o': mergeFileName = strdup(optarg); break;
                case 'h':
                    fprintf (stdout, "%s -i in.sff -o path/out.sff folders/\n", argv[0]);
                    exit (0);
                    break;
                case 'v':   //version
                    fprintf (stdout, "%s", IonVersion::GetFullVersion("SFFProtonMerge").c_str());
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

    if (debug) {
        printf("Writing file: %s\n", mergeFileName);
    }

    for (unsigned int j=0; j<folders.size(); j++) {
        char* size = GetProcessParam (folders[j], "Block");
        xoffset.push_back(atoi(strtok(size,",")));
        yoffset.push_back(atoi(strtok(NULL,",")));

        if (debug) {
            std::cout << "folder: " << folders[j] << std::endl;
            std::cout << "block size: " << xoffset[j] << "," << yoffset[j] << std::endl;
        }
    }
    int ret = merge_sff(xoffset, yoffset, folders, mergeFileName, inputSFFFileName, debug);

    free(mergeFileName);

    return ret;
}
