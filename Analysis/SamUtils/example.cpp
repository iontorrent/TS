/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <cassert>
#include <iostream>
#include <iomanip>
#include "BAMReader.h"

using namespace std;

int main(int argc, char* argv[])
{
  // Open a BAM file:
  std::string bamFile  = argv[1];
  //	char* bamIndex = argv[2];
  
  BAMReader reader(bamFile); 
  reader.open();
  assert(reader);
  
  // Print out list of reference sequences, and their lengths:
  cout << "Found " << reader.num_refs() << " reference sequences:" << endl;
  for(int i=0; i<reader.num_refs(); ++i)
    cout << setw(9) << reader.refs()[i] << "    " << reader.lens()[i] << endl;
  
  
  //show ReadGroup functionality
  BAMReader::BAMHeader header = reader.get_header();
  BAMReader::BAMHeader::ReadGroup& rg = header.get_read_group(0);
  cout << rg.to_string() << endl;
  
  // Print out list of reads:
  for (BAMReader::iterator i = reader.get_iterator(); i.good(); i.next()) {
    BAMRead read = i.get();
    cout << read.to_string();
    for (Sequence::iterator s_iter = read.get_seq().get_iterator(); s_iter.good(); s_iter.next())
      cout << s_iter.get(); // nuc from SEQ
    cout << endl;
    
    for (Cigar::iterator c_iter = read.get_cigar().get_iterator(); c_iter.good(); c_iter.next())
      cout << c_iter.len() << ":" << c_iter.op() << "; ";
    cout << endl;
    
    // Don't use MD directly.  Use BAMUtils.
    BAMUtils utils(read);
    cout << utils.get_qdna() << endl << utils.get_matcha() << endl << utils.get_tdna() << endl << endl;
    
    
    //print out flow signals
    std::vector<int> flows = read.get_fz(rg.get_number_of_flows());
    cout << "FS:Z:";
    std::copy(flows.begin(), flows.end(), ostream_iterator<float>(cout, ",") ) ;
    cout << endl;
    
  }
}

