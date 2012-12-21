/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "boost/scoped_array.hpp"
#include <map>

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
  //<< "  -m,--multi-sffs sff output  output multi read group sff files" << endl;
  exit(1);
}

int main(int argc, const char *argv[])
{
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  bool help, suppressKey;
  string sffFile;
  string bamFile;
  //bool outm = false;
  vector<string> infiles;
  opts.GetOption(help,"false", 'h', "help");
  opts.GetOption(suppressKey,"false", 'k', "suppress-key");
  opts.GetOption(sffFile,"",'o',"out-filename");
  //opts.GetOption(outm,"false",'m',"multi-sffs");
  opts.GetLeftoverArguments(infiles);

  if(help || infiles.empty())
  {
    usage();
  }

  if(infiles.size() > 1 && sffFile.length() < 1)
  {
    cerr << "bam2sff ERROR: output sff file name must be provided with multi bam input files" << endl;
    exit(1);
  }

  bamFile= infiles.front();

  if(sffFile.length() < 1)
  {
    sffFile = bamFile.substr(0, bamFile.length() - 3);
    sffFile += "sff";
  }
  
//  bool multi = false;
  bool diffKey = false;
  string flow_order;
  string key;
  map<string, string> flow_orders;
  map<string, string> keys;
  BamReader bamReader;

  for(vector<string>::iterator iterbam = infiles.begin(); iterbam != infiles.end(); ++iterbam)
  {
	  bamFile = *iterbam;	  
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

	  for (SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr )
	  {
		if(!itr->HasFlowOrder())
		{
		  bamReader.Close();
		  cerr << "bam2sff ERROR: there is no flow order in read group " << itr->ID << " in " << bamFile << endl;
		  exit(1);
		}
		if((!itr->HasFlowOrder()) || (!itr->HasKeySequence()))
		{
		  bamReader.Close();
		  cerr << "bam2sff ERROR: there is no key sequence in read group " << itr->ID << " in " << bamFile << endl;
		  exit(1);
		}

		map<string, string>::iterator iterfo = flow_orders.find(itr->ID);
		if(iterfo == flow_orders.end())
		{
			flow_orders[itr->ID] = itr->FlowOrder;
		}
		else
		{
			if(flow_orders[itr->ID] != itr->FlowOrder)
			{
				  bamReader.Close();
				  cerr << "bam2sff ERROR: there are different flow orders for read group " << itr->ID << endl;
				  exit(1);
			}
		}

		map<string, string>::iterator iterk = keys.find(itr->ID);
		if(iterk == keys.end())
		{		
			keys[itr->ID] = itr->KeySequence;
		}
		else
		{
			if(keys[itr->ID] != itr->KeySequence)
			{
				  bamReader.Close();
				  cerr << "bam2sff ERROR: there are different keys for read group " << itr->ID << endl;
				  exit(1);
			}
		}

		if(flow_order.length() > 0)
		{
		  if(flow_order != itr->FlowOrder)
		  {
			bamReader.Close();
			cerr << "bam2sff ERROR: there are multiple read groups with different flow order length in " << bamFile << endl;
			//   << "\n\t\tto output multi sffs please re-run with option -m true." << endl;
			exit(1);
			//multi = true;
		  }
		}
		else
		{
		  flow_order = itr->FlowOrder;
		}

		if(key.length() > 0)
		{
		  if(key != itr->KeySequence)
		  {
			diffKey = true;
		  }
		}
		else
		{
		  key = itr->KeySequence;
		}
	  }

	  bamReader.Close();
  }

/*  if(multi) && (!outm))
  {
    bamReader.Close();
    cerr << "bam2sff ERROR: there are multiple read groups with different flow order length in " << bamFile << endl;
         << "\n\t\tto output multi sffs please re-run with option -m true." << endl;
    exit(1);
  }
*/
  sff_file_t* sff_file = NULL;
  map<string, sff_file_t*> sff_files;
  uint16_t maxflow = 0;
  uint16_t nKeys = key.length();
/*  if(multi)
  {
    cout << "bam2sff WARNING: there are multi read groups with different flow orders in " << bamFile << ". Multiple sff files will be generated with postfix of read group id to file name." << endl;

    map<string, string>::iterator iter1 = flow_orders.begin();
    map<string, string>::iterator iter2 = keys.begin();
    for(; iter1 != flow_orders.end() && iter2 != keys.end(); ++iter1, ++iter2)
    {
      uint16_t nFlows0 = (iter1->second).length();
      uint16_t nKeys0 = (iter2->second).length();
      if(0 == nFlows0 || 0 == nKeys0)
      {
        bamReader.Close();
        if(!sff_files.empty())
        {
          for(map<string, sff_file_t*>::iterator iter3 = sff_files.begin(); iter3 != sff_files.end(); ++iter3)
          {
            sff_fclose(iter3->second);
          }
        }
        cerr << "bam2sff ERROR: floworder or key is empty in " << bamFile << endl;
        exit(1);
      }

      if(maxflow < nFlows0)
      {
        maxflow = nFlows0;
      }

      sff_header_t* sff_header0 = sff_header_init1(0, nFlows0, (iter1->second).c_str(), (iter2->second).c_str());
      string sffFile0 = sffFile.substr(0, sffFile.length() - 4);
      sffFile0 += "_";
      sffFile0 += iter1->first;
      sffFile0 += ".sff";
      sff_file_t* sff_file0 = sff_fopen(sffFile0.c_str(), "wb", sff_header0, NULL);
      sff_header_destroy(sff_header0);
      if(!sff_file0)
      {
        bamReader.Close();
        if(!sff_files.empty())
        {
          for(map<string, sff_file_t*>::iterator iter3 = sff_files.begin(); iter3 != sff_files.end(); ++iter3)
          {
            sff_fclose(iter3->second);
          }
        }
        cerr << "bam2sff ERROR: fail to open " << sffFile0 << endl;
        exit(1);
      }

      sff_files[iter1->first] = sff_file0;
    }
  }
  else
  {*/
    uint16_t nFlows = flow_order.length();
    if(0 == nFlows)
    {
      //bamReader.Close();
      cerr << "bam2sff ERROR: there is no floworder in " << bamFile << endl;
      exit(1);
    }

    maxflow = nFlows;

    if(diffKey)
    {
      unsigned int n = 1;
      bool common = true;
      string key2;
      while(common && n <= key.length())
      {
        key2 = key.substr(0, n);
        for(map<string, string>::iterator iter4 = keys.begin(); iter4 != keys.end(); ++iter4)
        {
          string key3 = (iter4->second).substr(0, n);
          if(key2 != key3)
          {
            common = false;
            --n;
            break;
          }
        }
        ++n;
      }
      --n;

      if(n > 0)
      {
        key = key.substr(0, n);
      }
      else
      {
        key = "";
      }
    }

    nKeys = key.length();
    sff_header_t* sff_header = sff_header_init1(0, nFlows, flow_order.c_str(), key.c_str());
    sff_file = sff_fopen(sffFile.c_str(), "wb", sff_header, NULL);
    sff_header_destroy(sff_header);
    if(!sff_file)
    {
      bamReader.Close();
      cerr << "bam2sff ERROR: fail to open " << sffFile << endl;
      exit(1);
    }
//  }

  boost::scoped_array<uint16_t> flowgram(new uint16_t[maxflow]);
  boost::scoped_array<char> bases(new char[BASE_BUFFER_SIZE]);
  boost::scoped_array<char> qalities(new char[BASE_BUFFER_SIZE]);
  char* pBases = bases.get();
  char* pQualities = qalities.get();
  boost::scoped_array<uint8_t> flow_index(new uint8_t[BASE_BUFFER_SIZE]);
  vector<uint16_t> flowInt;
  vector<int16_t> new_flow_signal;

  sff_t* sff = sff_init1();

  sff->rheader->name = ion_string_init(0);
  sff->read->bases = ion_string_init(0);
  sff->read->quality = ion_string_init(0);

  int nReads = 0;
  int nReadsIgnore = 0;
  BamAlignment alignment;

  for(vector<string>::iterator iterbam = infiles.begin(); iterbam != infiles.end(); ++iterbam)
  {
	  bamFile = *iterbam;	  
	  bamReader.Open(bamFile);

	  while(bamReader.GetNextAlignment(alignment))
	  {
		int rgind = alignment.Name.find(":");
		string rgname = alignment.Name;
		if(rgind > 0)
		{
		  rgname = alignment.Name.substr(0, rgind);
		}
		if(alignment.HasTag("RG"))
		{
		  string rgname2;
		  if(alignment.GetTag("RG", rgname2))
		  {
			rgname = rgname2;
		  }
		}

		sff_file_t* sff_file2 = sff_file;
	/*    if(multi)
		{
		  map<string, sff_file_t*>::iterator iter3 = sff_files.find(rgname);
		  if(iter3 == sff_files.end())
		  {
			++nReadsIgnore;
			continue;
		  }

		  sff->gheader = (iter3->second)->header;
		  sff_file2 = iter3->second;
		}
		else
		{*/
		  sff->gheader = sff_file->header;
	//    }
		sff->rheader->name_length = alignment.Name.length();
		sff->rheader->name->s = (char*)alignment.Name.c_str();
		sff->rheader->n_bases = nKeys + alignment.Length;
		string key2 = key;
		sff->rheader->clip_qual_left = nKeys + 1;
		if(diffKey)
		{
		  map<string, string>::iterator iter2 = keys.find(rgname);
		  if(iter2 == keys.end())
		  {
			++nReadsIgnore;
			continue;
		  }
		  key2 = iter2->second;
		  sff->rheader->clip_qual_left = key2.length() + 1;
		  sff->rheader->n_bases = key2.length() + alignment.Length;
		}
		sff->rheader->clip_adapter_left = 0;
		sff->rheader->clip_qual_right = 0;//sff->rheader->n_bases + 1;
		sff->rheader->clip_adapter_right = 0;//sff->rheader->n_bases + 1;

		if(alignment.GetTag("ZM", new_flow_signal)) {
		  flowInt.assign(flow_order.length(),0);
		  for (unsigned int pos = 0; pos < flow_order.length() and pos < new_flow_signal.size(); ++pos)
			flowInt[pos] = (100 * max(0,(int)new_flow_signal[pos])) / 256;
		}
		else
		if(!alignment.GetTag("FZ", flowInt))
		{
		  bamReader.Close();
		  free(sff->read);
		  sff->read = NULL;
		  free(sff->rheader->name);
		  sff->rheader->name = NULL;
		  sff_destroy(sff);
/*		  if(multi)
		  {
			for(map<string, sff_file_t*>::iterator iter3 = sff_files.begin(); iter3 != sff_files.end(); ++iter3)
			{
			  sff_fclose(iter3->second);
			}
		  }
		  else
		  {*/
			sff_fclose(sff_file);
//		  }

		  cerr << "bam2sff ERROR: fail to get flow intensities." << endl;
		  exit(1);
		}
		copy(flowInt.begin(), flowInt.end(), flowgram.get());
		sff->read->flowgram = flowgram.get();

		unsigned int base = 0;
		for(; base < key2.length(); ++base)
		{
		  pBases[base] = key2[base];
		  pQualities[base] = DEAFAUL_QUALITY;
		}

		bool flag = false;
		int base2 = 0;
		if(16 == alignment.AlignmentFlag)
		{
		  flag = true;
		  base2 = alignment.Length - 1;
		}

		unsigned int sz = (unsigned int)alignment.Length + key2.length();
		for(; base < sz; ++base)
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
		string flow_order2(flow_order);

/*		if(multi)
		{
		  map<string, string>::iterator iter1 = flow_orders.find(rgname);
		  if(iter1 == flow_orders.end())
		  {
			++nReadsIgnore;
			continue;
		  }
		  flow_order2 = iter1->second;
		}
*/
		uint16_t nFlows2 = flow_order2.length();

		while(nBase < sff->rheader->n_bases && nFlow < nFlows2)
		{
		  if(bases[nBase] == preBase)
		  {
			flow_index[nBase] = index;
			++nBase;
		  }
		  else
		  {
			++index;
			while(nFlow < nFlows2 && flow_order2[nFlow] != bases[nBase])
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

		++nReads;

		sff_write(sff_file2, sff);
	  }

	  bamReader.Close();
  }

  sff_file->header->n_reads = nReads;
  fseek(sff_file->fp, 0, SEEK_SET);
  sff_header_write(sff_file->fp, sff_file->header);

  free(sff->read);
  sff->read = NULL;
  free(sff->rheader->name);
  sff->rheader->name = NULL;
  sff_destroy(sff);

//  bamReader.Close();

/*  if(multi)
  {
    if(!sff_files.empty())
    {
      for(map<string, sff_file_t*>::iterator iter3 = sff_files.begin(); iter3 != sff_files.end(); ++iter3)
      {
        sff_fclose(iter3->second);
      }
    }
  }
  else
  {*/
    sff_fclose(sff_file);
//  }

  if(0 < nReadsIgnore)
  {
    cout << "bam2sff WARNING: there are " << nReadsIgnore << " reads having read group name not listed in the bam header. They are not saved to sff files." << endl;
  }

  return 0;
}
