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


int merge_sff(
              const std::vector<char*>& filenames,
              const char* mergeFileName,
              bool removeKey,
              bool debug
              )
{

    sff_file_t* sff_file_in = NULL;
    sff_file_t* sff_file_out = NULL;

    // for each input sff file:
    for (unsigned int j=0; j<filenames.size(); j++) {

        if (debug) {
            printf("Reading file: %s\n", filenames[j]);
        }
        sff_file_in = sff_fopen(filenames[j], "rb", NULL, NULL);


        /// header section ///
        if (debug) {
            sff_header_print(stdout, sff_file_in->header);
        }

        // Refuse to merge sffs with index. Just to be sure.
        assert(sff_file_in->header->index_offset == 0);
        assert(sff_file_in->header->index_length == 0);

        // write the initial/updated header information
        if (j==0) {
            // copy header from first sff file

            if (removeKey) {
                sff_header_t* fileheader_clone =  sff_header_init1(
                                     sff_file_in->header->n_reads,
                                     sff_file_in->header->flow_length,
                                     sff_file_in->header->flow->s,
                                     "");
                sff_file_out = sff_fopen(mergeFileName, "wb", fileheader_clone, NULL); // Note: sff_fopen internally clones the header
                sff_header_destroy(fileheader_clone);
            } else {
                sff_file_out = sff_fopen(mergeFileName, "wb", sff_file_in->header, NULL); // Note: sff_fopen internally clones the header
            }
        } else {
            assert(sff_file_out->header->magic           == sff_file_in->header->magic);
            assert(sff_file_out->header->version         == sff_file_in->header->version);
            sff_file_out->header->n_reads += sff_file_in->header->n_reads;
            if (!removeKey) {
              // skip that check if the key has been removed
              assert(sff_file_out->header->gheader_length  == sff_file_in->header->gheader_length); // header size ok to change if key removed
              assert(sff_file_out->header->key_length      == sff_file_in->header->key_length);
            }
            assert(sff_file_out->header->flow_length     == sff_file_in->header->flow_length);
            assert(sff_file_out->header->flowgram_format == sff_file_in->header->flowgram_format);
            if (!removeKey) {
              // skip that check if the key has been removed
              if (debug) {
                printf("%s,%s\n" , sff_file_out->header->key->s, sff_file_in->header->key->s);
              }
              for(int i=0;i<sff_file_out->header->key_length;i++) {
                assert(sff_file_out->header->key->s[i] == sff_file_in->header->key->s[i]);
              }
            }
            for(int i=0;i<sff_file_out->header->flow_length;i++) {
              assert(sff_file_out->header->flow->s[i] == sff_file_in->header->flow->s[i]);
            }
        }
        fseek(sff_file_out->fp, 0, SEEK_SET);
        sff_header_write(sff_file_out->fp, sff_file_out->header);
        fseek(sff_file_out->fp, 0, SEEK_END);

        if (debug) {
            sff_header_print(stdout, sff_file_out->header);
        }

        //// Reads section ////
        for(unsigned int i=0; i<sff_file_in->header->n_reads; i++) {
            sff_read_header_t* rh = sff_read_header_read(sff_file_in->fp);
            sff_read_t* rr = sff_read_read(sff_file_in->fp, sff_file_in->header, rh);

            if (debug) {
                sff_read_header_print(stdout, rh);
            }

            if (removeKey) {
                //Only permit SFF combining when the flow order is identical between all input files.
                //Keep flow values the same
                //Confirm that the key is a perfect prefix of the bases in the read, and strip the prefix away.  Die if this condition doesn't hold true.  This condition will be guaranteed so long as the "–k off" option is not used in Analysis – this option bypasses the requirement of a match to the library key.
                //Make the corresponding reduction in the quality scores and read length
                //Update the flow index field, making sure to sum up the values of the deleted entries to add them into the new first entry so that it points to the right flow.

                // update rh + rr

                // Check 1: There must be at least one extra base in this read beyond the key
                assert (rh->n_bases > sff_file_in->header->key_length);

                // Check 2: The read must start with the key
                for (unsigned int i=0; i<sff_file_in->header->key_length; i++) {
                  assert(sff_file_in->header->key->s[i] == rr->bases->s[i]);
                }

                // Adjust header and flow_index
                for (unsigned int i=0; i<sff_file_in->header->key_length; i++) {

                    rh->n_bases--;
                    if (rh->clip_adapter_left > 0) {
                        rh->clip_adapter_left--;
                    }
                    if (rh->clip_qual_left > 0) {
                        rh->clip_qual_left--;
                    }
                    if (rh->clip_adapter_right > 0) {
                        rh->clip_adapter_right--;
                    }
                    if (rh->clip_qual_right > 0) {
                        rh->clip_qual_right--;
                    }

                    rr->flow_index[i+1] += rr->flow_index[i];
                }

                // Shift flow_index, bases, and quality
                for (unsigned int i=0; i<rh->n_bases; i++) {
                    rr->flow_index[i] = rr->flow_index[i + sff_file_in->header->key_length];
                    rr->bases->s[i] = rr->bases->s[i + sff_file_in->header->key_length];
                    rr->quality->s[i] = rr->quality->s[i + sff_file_in->header->key_length];
                }

                // Update lengths of ion_strings. Double check they are right. Note: flow_index is not an ion_string
                rr->bases->l -= sff_file_in->header->key_length;
                rr->quality->l -= sff_file_in->header->key_length;
                assert(rr->bases->l == rh->n_bases);
                assert(rr->quality->l == rh->n_bases);

                if (debug) {
                    sff_read_header_print(stdout, rh);
                }
            }

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


void help() {
    fprintf (stdout, "SFFMerge -o path/out.sff in1.sff in2.sff\n");
}

int main(int argc, char *argv[])
{
    char* mergeFileName = NULL;

    std::vector<char*> filenames;
    bool debug = false;
    bool removeKey = false;
    int c;

    if (argc < 2) {
        help();
        exit(1);
    }
    while ( (c = getopt (argc, argv, "o:hvrd")) != -1 )
        {
            switch (c)
                {
                case 'o': mergeFileName = strdup(optarg); break;
                case 'h':
                    help();
                    exit (0);
                    break;
                case 'v':   //version
                    fprintf (stdout, "%s", IonVersion::GetFullVersion("SFFMerge").c_str());
                    return (0);
                    break;
                case 'd':       // enable debug print outs
                    debug = true;
                    break;
                case 'r':       // remove library key
                    removeKey = true;
                    break;
                default:
                    fprintf (stdout, "unknown option");
                    break;
                }
        }

    for (c = optind; c < argc; c++) {
        filenames.push_back(argv[c]);
    }

    if (debug) {
        printf("Writing file: %s\n", mergeFileName);
    }

    for (unsigned int j=0; j<filenames.size(); j++) {

        if (debug) {
            std::cout << "file: " << filenames[j] << std::endl;
        }
    }
    int ret = merge_sff(filenames, mergeFileName, removeKey, debug);

    free(mergeFileName);

    return ret;
}
