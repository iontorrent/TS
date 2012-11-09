/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "boost/scoped_array.hpp"

#include "sff.h"
#include "sff_file.h"
#include "sff_header.h"
#include "OptArgs.h"

#define BAM2SFF_VERSION "1.0.0"
#define DEAFAUL_QUALITY 30
#define BASE_BUFFER_SIZE  1000000

using namespace std;
using namespace BamTools;

char ComplementBase(char base)
{
    switch(base)
    {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        default:  return base;
    }
}

void usage()
{
  cout << "bam2sff - Converts bam file to sff file." << endl;
  cout << "Usage: " << endl
       //<< "  bam2sff [-k] [-o out.sff] in.bam" << endl
	   << "  bam2sff [-o out.sff] in.bam" << endl
       << "Options:" << endl
       << "  -h,--help              this message" << endl
       //<< "  -k,--suppress-key      remove key sequence and change flow values to 0." << endl
       << "  -o,--out-filename      specify output file name" << endl;
  exit(1);
}

int main(int argc, const char *argv[])
{
    OptArgs opts;
    opts.ParseCmdLine(argc, argv);
    bool help, suppressKey;
    string sffFile;
    string bamFile;
    vector<string> infiles;
    opts.GetOption(help,"false", 'h', "help");
    opts.GetOption(suppressKey,"false", 'k', "suppress-key");
    opts.GetOption(sffFile,"",'o',"out-filename");
    opts.GetLeftoverArguments(infiles);

    if(help || infiles.empty())
    {
        usage();
    }

    bamFile= infiles.front();

    if(sffFile.length() < 1)
    {
        sffFile = bamFile.substr(0, bamFile.length() - 3);
        sffFile += "sff";
    }

    BamReader bamReader;
    if (!bamReader.Open(bamFile))
    {
        cerr << "bam2sff ERROR: fail to open bam" << bamFile << endl;
        exit(1);
    }

    SamHeader samHeader = bamReader.GetHeader();
    if(!samHeader.HasReadGroups())
    {
        bamReader.Close();
        cerr << "bam2sff ERROR: there is no read group in " << bamFile << endl;
        exit(1);
    }

    string flow_order;
    string key;
    for (SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr )
    {
        if(itr->HasFlowOrder())
        {
            flow_order = itr->FlowOrder;
        }
        if(itr->HasKeySequence())
        {
            key = itr->KeySequence;
        }
    }

    uint16_t nFlows = flow_order.length();
    uint16_t nKeys = key.length();
    if(!nFlows || nKeys < 1)
    {
        bamReader.Close();
        cerr << "bam2sff ERROR: there is no floworder or key in " << bamFile << endl;
        exit(1);
    }

    sff_header_t* sff_header = sff_header_init1(0, nFlows, flow_order.c_str(), key.c_str());
    sff_file_t* sff_file = sff_fopen(sffFile.c_str(), "wb", sff_header, NULL);
    sff_header_destroy(sff_header);
    if(!sff_file)
    {
        bamReader.Close();
        cerr << "bam2sff ERROR: fail to open " << sffFile << endl;
        exit(1);
    }

    boost::scoped_array<uint16_t> flowgram(new uint16_t[nFlows]);
    boost::scoped_array<char> bases(new char[BASE_BUFFER_SIZE]);
    boost::scoped_array<char> qalities(new char[BASE_BUFFER_SIZE]);
    char* pBases = bases.get();
    char* pQualities = qalities.get();
    for(size_t i = 0; i < nKeys; ++i, ++pBases, ++pQualities)
    {
        (*pBases) = key[i];
        (*pQualities) = DEAFAUL_QUALITY;
    }
    boost::scoped_array<uint8_t> flow_index(new uint8_t[BASE_BUFFER_SIZE]);
    vector<uint16_t> flowInt(nFlows);

    sff_t* sff = sff_init1();
    sff->gheader = sff_file->header;
    sff->rheader->name = ion_string_init(0);
    sff->read->bases = ion_string_init(0);
    sff->read->quality = ion_string_init(0);
    sff->rheader->clip_qual_left = nKeys + 1;
    sff->rheader->clip_adapter_left = 0;

    int nReads = 0;
    BamAlignment alignment;
    while(bamReader.GetNextAlignment(alignment))
    {
        sff->rheader->name_length = alignment.Name.length();
        sff->rheader->name->s = (char*)alignment.Name.c_str();
        sff->rheader->n_bases = nKeys + alignment.Length;
        sff->rheader->clip_qual_right = 0;//sff->rheader->n_bases + 1;
        sff->rheader->clip_adapter_right = 0;//sff->rheader->n_bases + 1;

        if(!alignment.GetTag("FZ", flowInt))
        {
            sff_destroy(sff);
            sff_fclose(sff_file);
            bamReader.Close();
            cerr << "bam2sff ERROR: fail to open " << sffFile << endl;
            exit(1);
        }
        copy(flowInt.begin(), flowInt.end(), flowgram.get());
        sff->read->flowgram = flowgram.get();

        bool flag = false;
        int base2 = 0;
        if(16 == alignment.AlignmentFlag)
        {
            flag = true;
            base2 = alignment.Length - 1;
        }

        for(int base = 0; base < alignment.Length; ++base)
        {
            if(flag)
            {
                pBases[base] = ComplementBase(alignment.QueryBases[base2]);
                pQualities[base] = alignment.Qualities[base2] - 33;
                --base2;
            }
            else
            {
                pBases[base] = alignment.QueryBases[base2];
                pQualities[base] = alignment.Qualities[base2] - 33;
                ++base2;
            }
        }

        sff->read->bases->s = bases.get();
        sff->read->quality->s = qalities.get();
		
		uint32_t nFlow = 0;		
		uint32_t nBase = 0;
		char preBase = bases[nBase] - 1;
        int index = 0;

		while(nBase < sff->rheader->n_bases && nFlow < nFlows)
		{
			if(bases[nBase] == preBase)
			{	
				flow_index[nBase] = index;
				++nBase;
			}
			else
			{
                ++index;
				while(nFlow < nFlows && flow_order[nFlow] != bases[nBase])
				{
					++nFlow;
					++index;
				}
				
				flow_index[nBase] = index;
                index = 0;
				preBase = bases[nBase];
				++nBase;
				++nFlow;
			}			
		}

        sff->read->flow_index = flow_index.get();

        sff_write(sff_file, sff);
        ++nReads;
    }

    sff_file->header->n_reads = nReads;
    fseek(sff_file->fp, 0, SEEK_SET);
    sff_header_write(sff_file->fp, sff_file->header);

    free(sff->read);
    sff->read = NULL;
    free(sff->rheader->name);
    sff->rheader->name = NULL;
    sff_destroy(sff);

    sff_fclose(sff_file);
    bamReader.Close();

    return 0;
}
