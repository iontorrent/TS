/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include "BAMReader.h"

using namespace std;

int main(int argc, char *argv[])
{
  if(5 != argc) {
      cerr << "Usage: BAMFilter <q-score> <q-length> <filter> <in.bam>" << endl;
      cerr << "Filtering options" << endl;
      cerr << "1 : filter on calculated qscore" << endl;
      cerr << "2 : filter on predicted qscore" << endl;
      return 1;
  }

  int filter = atoi(argv[3]);
  int qscore = atoi(argv[1]);
  int qlength = atoi(argv[2]);
  BAMReader reader(argv[4]); 
  reader.open();
  assert(reader);
  BAMReader::iterator iter=reader.get_iterator();
  samfile_t *writer = NULL;

  //writer = samopen("-", "wh", reader.get_header_ptr()); // SAM
  writer = samopen("-", "wb", reader.get_header_ptr()); // BAM
  
  bool pass;
  for(; iter.good(); iter.next()) {
      BAMRead read = iter.get();
      BAMUtils utils(read);
      pass = false;
      switch (filter) {
          case 1:
              if(qlength <= utils.get_phred_len(qscore)) // does it pass the qscore
                  pass = true;
              break;

          case 2:
              {
                  std::string qual = utils.get_qual();
                  int len = utils.get_full_q_length();
                  if (len >= 100) {
                      double avg = 0.0;
                      for(int i=0;i<100;i++) {
                          avg += (qual[i] - 32);
                      }
                      avg /= 100.0;
                      if (avg >= 20.0)
                          pass = true;
                  }
              }
              break;
      }
      if (pass)
          assert(0 < samwrite(writer, read.get_bam_ptr()));
  }

  reader.close();
  samclose(writer);
  return 0;
}
