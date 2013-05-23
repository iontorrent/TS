/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "StackEngine.h"



int StackPlus::GrabStack(char *bamFile, string &variant_contig, unsigned int variant_position)
{
// something wonderful happens here
  BamTools::BamReader bamReader;
  OpenMyBam(bamReader,bamFile);
  BamHeaderHelper my_helper;
  my_helper.GetRefID(bamReader);
  my_helper.GetFlowOrder(bamReader);

//  for (unsigned int i=0; i<my_helper.flow_order_set.size(); i++)
//     cout << i << "\t" << my_helper.flow_order_set[i].size() << "\t" << my_helper.flow_order_set[i] << endl;

  int variant_contig_id = my_helper.IdentifyRefID(variant_contig);
  
  if (!bamReader.Jump(variant_contig_id, variant_position)) {
     cerr << "ERROR: Unable to access contig id " << variant_contig_id << " and position = " << variant_position << " within the BAM file provoided " << endl;
      //exit(-1);
   }
   // and now we're at the first read we need
  ExtendedReadInfo current_read;

  while (bamReader.GetNextAlignment(current_read.alignment)) {

    if (!current_read.UnpackThisRead(my_helper, variant_contig, variant_position, 0))
      break;
    read_stack.push_back(current_read);
  }

  bamReader.Close();
  flow_order=my_helper.flow_order_set[0];

  if (my_helper.flow_order_set.size()>0)
    return(my_helper.flow_order_set[0].size());
  else
    return(0);
} 



