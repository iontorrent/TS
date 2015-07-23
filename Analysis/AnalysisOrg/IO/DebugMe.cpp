/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include "DebugMe.h"
#include "IonErr.h"
#include "Utils.h"

using namespace std;

DebugMe::DebugMe(){
  bkg_debug_files = 0;
  bkgModelHdf5Debug = 1;
  bkgModelHdf5Debug_region_r = -1;
  bkgModelHdf5Debug_region_c = -1;
  bkgModel_xyflow_output = false;
  bkgModel_xyflow_fname_in_type = -1;
  bkgModel_xyflow_fname_in = "";
  bkgDebugParam = 0;
  // diagnostics
  debug_bead_only = 1;  // only debug bead
  region_vfrc_debug = 0; // off
  BkgTraceDebugRegions.clear();

}


bool DebugMe::read_file_sse(HashTable_xyflow &xyf_hash, int numFlows) const
{
  if (! isFile(bkgModel_xyflow_fname_in.c_str())) {
      std::cerr << "read_file_sse() error: file " << bkgModel_xyflow_fname_in << " does not exist!!" << std::endl << std::flush;
      exit(1);
  }
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
      //bool mm = tokens[6].compare("1")==0 ? 1:0;
      bool mm = tokens[6].compare("REF")==0 ? 0:1;
      std::string hp = tokens[8];
      xyf_hash.insert_rcflow(r,c,f,mm,hp);
      xyf_hash.insert_rc(r,c);
      //std::cout << r << " " << c << " " << flow << std::endl;
    }
    std::cout << nLines << " lines read by read_file_sse()..."  << std::endl << std::flush;
    std::cout << "xyf_hash.size() = " << xyf_hash.size()  << std::endl << std::flush;
    std::cout << "xyf_hash.size_xy() = " << xyf_hash.size_xy()  << std::endl << std::flush;
    if ( xyf_hash.size() < xyf_hash.size_xy()) // something wrong, shouldn't happen, print out and exit
    {
        std::cout << "read_file_sse() error: xyf_hash.size(" << xyf_hash.size() << ") < xyf_hash.size_xy(" << xyf_hash.size_xy() << ")" << std::endl << std::flush;
        //xyf_hash.print();
        assert(xyf_hash.size() >= xyf_hash.size_xy());
        exit(1);
    }
  }
  else {
    std::cerr << "read_file_sse() processing error!!!"  << std::endl << std::flush;
    return (false);
  }
  infile.close();
  return (true);
}


bool DebugMe::read_file_rcflow(HashTable_xyflow &xyf_hash, int numFlows) const
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


bool DebugMe::read_file_xyflow(HashTable_xyflow &xyf_hash, int numFlows) const
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

void DebugMe::PrintHelp()
{
	printf ("     DebugMe\n");
    printf ("     --bkg-debug-param       INT               background modeling hdf5 debug param [1]\n");
	printf ("     --bkg-debug-region      INT VECTOR OF 2   background modeling hdf5 debug region [-1,-1]\n");
	printf ("     --bkg-dbg-trace         INT VECTOR OF 2   background modeling	trace debug region []\n");
	printf ("     --debug-bead-only       BOOL              debug bead only [true]\n");
	printf ("     --region-vfrc-debug     BOOL              region vfrc debug [false]\n");
	printf ("     --bkg-debug-files       BOOL              background debug files [false]\n");
    printf ("     --bkg-debug-trace-sse   FILE              background modeling xyflow file name, file type 1 []\n");
    printf ("     --bkg-debug-trace-rcflow            FILE  background modeling xyflow file name, file type 2 []\n");
    printf ("     --bkg-debug-trace-xyflow            FILE  background modeling xyflow file name, file type 3 []\n");
    printf ("\n");
}

void DebugMe::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	bkgModelHdf5Debug = RetrieveParameterInt(opts, json_params, '-', "bkg-debug-param", 1);
	vector<int> vec1;
	RetrieveParameterVectorInt(opts, json_params, '-', "bkg-debug-region", "-1,-1", vec1);
	if(vec1.size() == 2)
	{
		bkgModelHdf5Debug_region_r = vec1[0];
		bkgModelHdf5Debug_region_c = vec1[1];
	}
	else
	{
        fprintf ( stderr, "Option Error: bkg-debug-region format wrong, not size = 2\n" );
        exit ( EXIT_FAILURE );
	}
	bkgModel_xyflow_fname_in = RetrieveParameterString(opts, json_params, '-', "bkg-debug-trace-sse", "");
	if(bkgModel_xyflow_fname_in.length() > 0)
	{
		bkgModel_xyflow_output = true;
		bkgModel_xyflow_fname_in_type = 1;
	}
	else
	{
		bkgModel_xyflow_fname_in = RetrieveParameterString(opts, json_params, '-', "bkg-debug-trace-rcflow", "");
		if(bkgModel_xyflow_fname_in.length() > 0)
		{
			bkgModel_xyflow_output = true;
			bkgModel_xyflow_fname_in_type = 2;
		}
		else
		{
			bkgModel_xyflow_fname_in = RetrieveParameterString(opts, json_params, '-', "bkg-debug-trace-xyflow", "");
			if(bkgModel_xyflow_fname_in.length() > 0)
			{
				bkgModel_xyflow_output = true;
				bkgModel_xyflow_fname_in_type = 3;
			}
		}
	}
	vector<int> vec2;
	RetrieveParameterVectorInt(opts, json_params, '-', "bkg-dbg-trace", "", vec2);
	if(vec2.size() == 2)
	{
		Region dbg_reg;
		dbg_reg.col = vec2[0];
		dbg_reg.row = vec2[1];
		BkgTraceDebugRegions.push_back ( dbg_reg );
	}
	//jz the following comes from CommandLineOpts::GetOpts
	debug_bead_only = RetrieveParameterBool(opts, json_params, '-', "debug-bead-only", true);
	region_vfrc_debug = RetrieveParameterBool(opts, json_params, '-', "region-vfrc-debug", false);
	bkg_debug_files = RetrieveParameterBool(opts, json_params, '-', "bkg-debug-files", false);
}
