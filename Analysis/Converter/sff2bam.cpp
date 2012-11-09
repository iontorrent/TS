/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/BamWriter.h"

#include "sff.h"
#include "sff_file.h"
#include "sff_header.h"
#include "OptArgs.h"

#define SFF2BAM_VERSION "1.0.0"

using namespace std;
using namespace BamTools;

void usage()
{
  cout << "sff2bam - Converts sff file(s) to unmapped bam file." << endl;
  cout << "Usage: " << endl
       << "  sff2bam [-c] [-o out.bam] in.sff [in2.sff ...]" << endl
       << "Options:" << endl
       << "  -h,--help              this message" << endl
       << "  -c,--combine-sffs      combine all input sff files to a single bam file" << endl
       << "                         all sffs must have same flow orders and same keys." << endl
       << "  -o,--out-filename      specify output file name" << endl;
  exit(1);
}

int main(int argc, const char *argv[])
{
    OptArgs opts;
    opts.ParseCmdLine(argc, argv);
    bool help, combineSffs;
    string sffFile;
    string bamFile;
    vector<string> infiles;
    opts.GetOption(help,"false", 'h', "help");
    opts.GetOption(combineSffs,"false", 'c', "combine-sffs");
    opts.GetOption(bamFile,"",'o',"out-filename");
    opts.GetLeftoverArguments(infiles);

    if(help || infiles.empty())
    {
        usage();
    }

	if((!combineSffs) && infiles.size() > 1)
	{
        cerr << "sff2bam ERROR: if you want to combine all sff files into a single bam file, please use option -c true." << endl;
        usage();
	}

    sffFile= infiles.front();

    if(bamFile.length() < 1)
    {
        bamFile = sffFile.substr(0, sffFile.length() - 3);
        bamFile += "bam";
    }

    sff_file_t* sff_file = sff_fopen(sffFile.c_str(), "r", NULL, NULL);
    if(!sff_file)
    {
        cerr << "sff2bam ERROR: fail to open " << sffFile << endl;
        exit(1);
    }

	// All sff files must have the same flow and key
	if(combineSffs && infiles.size() > 1)
	{
        for(size_t n = 1; n < infiles.size(); ++n)
		{
			sff_file_t* sff_file2 = sff_fopen(infiles[n].c_str(), "r", NULL, NULL);
			if(!sff_file2)
			{
				sff_fclose(sff_file);
				cerr << "sff2bam ERROR: fail to open " << infiles[n] << endl;
				exit(1);
			}

			if(strcmp(sff_file2->header->flow->s, sff_file->header->flow->s) != 0 ||
				strcmp(sff_file2->header->key->s, sff_file->header->key->s) != 0)
			{
				sff_fclose(sff_file);
				sff_fclose(sff_file2);
				cerr << "sff2bam ERROR: " << sffFile << " and " << infiles[n] << " have different flows or keys." << endl;
				exit(1);
			}

			sff_fclose(sff_file2);
		}
	}

    sff_t* sff = NULL;
    // Open 1st read for read group name
    sff = sff_read(sff_file);
    if(!sff)
    {
        sff_fclose(sff_file);
        cerr << "sff2bam ERROR: fail to read " << sffFile << endl;
        exit(1);
    }

    // Set up BAM header
    SamHeader sam_header;
    sam_header.Version = "1.4";
    sam_header.SortOrder = "unsorted";

    SamProgram sam_program("sff2bam");
    sam_program.Name = "sff2bam";
    sam_program.Version = SFF2BAM_VERSION;
    sam_program.CommandLine = "sff2bam";
    sam_header.Programs.Add(sam_program);

    string rgname = sff->rheader->name->s;
    int index = rgname.find(":");
    rgname = rgname.substr(0, index);

    SamReadGroup read_group(rgname);
    read_group.FlowOrder = sff->gheader->flow->s;
    read_group.KeySequence = sff->gheader->key->s;

    sam_header.ReadGroups.Add(read_group);

    RefVector refvec;
    BamWriter bamWriter;
    bamWriter.SetCompressionMode(BamWriter::Compressed);

    if(!bamWriter.Open(bamFile, sam_header, refvec))
    {
        sff_fclose(sff_file);
        cerr << "sff2bam ERROR: failed to open " << bamFile << endl;
        exit(1);
    }

    // Save 1st read
    BamAlignment bam_alignment0;
    bam_alignment0.SetIsMapped(false);
    bam_alignment0.Name = sff->rheader->name->s;
    size_t nBases = sff->rheader->n_bases + 1 - sff->rheader->clip_qual_left;
    if(sff->rheader->clip_qual_right > 0)
    {
        nBases = sff->rheader->clip_qual_right - sff->rheader->clip_qual_left;
    }
    if(nBases > 0)
    {
        bam_alignment0.QueryBases.reserve(nBases);
        bam_alignment0.Qualities.reserve(nBases);
        for (int base = sff->rheader->clip_qual_left - 1; base < sff->rheader->clip_qual_right - 1; ++base)
        {
            bam_alignment0.QueryBases.push_back(sff->read->bases->s[base]);
            bam_alignment0.Qualities.push_back(sff->read->quality->s[base] + 33);
        }
    }

    int clip_flow = 0;
    for (unsigned int base = 0; base < sff->rheader->clip_qual_left && base < sff->rheader->n_bases; ++base)
    {
        clip_flow += sff->read->flow_index[base];
    }
    if (clip_flow > 0)
    {
        clip_flow--;
    }

    bam_alignment0.AddTag("RG","Z", rgname);
    bam_alignment0.AddTag("PG","Z", string("sff2bam"));
    bam_alignment0.AddTag("ZF","i", clip_flow); // TODO: trim flow
    vector<uint16_t> flowgram0(sff->gheader->flow_length);
    copy(sff->read->flowgram, sff->read->flowgram + sff->gheader->flow_length, flowgram0.begin());
    bam_alignment0.AddTag("FZ", flowgram0);
    sff_destroy(sff);
    sff = NULL;

    bamWriter.SaveAlignment(bam_alignment0);

    // Save rest reads
    while(NULL != (sff = sff_read(sff_file)))
    {
        BamAlignment bam_alignment;
        bam_alignment.SetIsMapped(false);
        bam_alignment.Name = sff->rheader->name->s;   
        nBases = sff->rheader->n_bases + 1 - sff->rheader->clip_qual_left;
        if(sff->rheader->clip_qual_right > 0)
        {
            nBases = sff->rheader->clip_qual_right - sff->rheader->clip_qual_left;
        }
        if(nBases > 0)
        {
            bam_alignment.QueryBases.reserve(nBases);
            bam_alignment.Qualities.reserve(nBases);
            for (int base = sff->rheader->clip_qual_left - 1; base < sff->rheader->clip_qual_right - 1; ++base)
            {
                bam_alignment.QueryBases.push_back(sff->read->bases->s[base]);
                bam_alignment.Qualities.push_back(sff->read->quality->s[base] + 33);
            }
        }

        clip_flow = 0;
        for (unsigned int base = 0; base <= sff->rheader->clip_qual_left && base < sff->rheader->n_bases; ++base)
        {
            clip_flow += sff->read->flow_index[base];
        }
        if (clip_flow > 0)
        {
            clip_flow--;
        }

        bam_alignment.AddTag("RG","Z", rgname);
        bam_alignment.AddTag("PG","Z", string("sff2bam"));
        bam_alignment.AddTag("ZF","i", clip_flow); // TODO: trim flow
        vector<uint16_t> flowgram(sff->gheader->flow_length);
        copy(sff->read->flowgram, sff->read->flowgram + sff->gheader->flow_length, flowgram.begin());
        bam_alignment.AddTag("FZ", flowgram);
        sff_destroy(sff);
        sff = NULL;

        bamWriter.SaveAlignment(bam_alignment);
    }

	sff_fclose(sff_file);

	if(combineSffs && infiles.size() > 1)
	{
        for(size_t n = 1; n < infiles.size(); ++n)
		{
			sff_file_t* sff_file2 = sff_fopen(infiles[n].c_str(), "r", NULL, NULL);

			while(NULL != (sff = sff_read(sff_file2)))
			{
				BamAlignment bam_alignment;
				bam_alignment.SetIsMapped(false);
				bam_alignment.Name = sff->rheader->name->s;   
				nBases = sff->rheader->n_bases + 1 - sff->rheader->clip_qual_left;
				if(sff->rheader->clip_qual_right > 0)
				{
					nBases = sff->rheader->clip_qual_right - sff->rheader->clip_qual_left;
				}
				if(nBases > 0)
				{
					bam_alignment.QueryBases.reserve(nBases);
					bam_alignment.Qualities.reserve(nBases);
					for (int base = sff->rheader->clip_qual_left - 1; base < sff->rheader->clip_qual_right - 1; ++base)
					{
						bam_alignment.QueryBases.push_back(sff->read->bases->s[base]);
						bam_alignment.Qualities.push_back(sff->read->quality->s[base] + 33);
					}
				}

				clip_flow = 0;
				for (unsigned int base = 0; base <= sff->rheader->clip_qual_left && base < sff->rheader->n_bases; ++base)
				{
					clip_flow += sff->read->flow_index[base];
				}
				if (clip_flow > 0)
				{
					clip_flow--;
				}

				bam_alignment.AddTag("RG","Z", rgname);
				bam_alignment.AddTag("PG","Z", string("sff2bam"));
				bam_alignment.AddTag("ZF","i", clip_flow); // TODO: trim flow
				vector<uint16_t> flowgram(sff->gheader->flow_length);
				copy(sff->read->flowgram, sff->read->flowgram + sff->gheader->flow_length, flowgram.begin());
				bam_alignment.AddTag("FZ", flowgram);
				sff_destroy(sff);
				sff = NULL;

				bamWriter.SaveAlignment(bam_alignment);
			}

			sff_fclose(sff_file2);
		}
	}

    bamWriter.Close();    

    return 0;
}
