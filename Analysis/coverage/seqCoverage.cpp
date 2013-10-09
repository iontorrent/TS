/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include "interval_tree.h"
#include "templatestack.h"

#define DEFAULT_N_INTERVALTREES 1000

using namespace std;

struct numcomp 
 {bool operator() (string a, string b) 
  {return (atoi(a.c_str())==atoi(b.c_str()) && a<b) || atoi(a.c_str()) < atoi(b.c_str());}
 };

int main(int argc, char **argv) {
  int printBlocks = 0;
  int reportDepth = 0;
  unsigned int nSeqAlloc = DEFAULT_N_INTERVALTREES;
  
  int c;
  while ((c = getopt(argc, argv, "bdn:")) != -1) {
    switch (c) {
      case 'b':
        printBlocks = 1;
        break;
      case 'd':
        reportDepth = 1;
        break;
      case 'n':
        nSeqAlloc = atoi(optarg);
        if(nSeqAlloc <= 0) {
          fprintf (stderr, "Option n requires a positive integer (value supplied was %s).\n",optarg);
          exit(EXIT_FAILURE);
        }
        break;
      case '?':
        if (optopt == 'n')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        exit(EXIT_FAILURE);
      default:
        exit(EXIT_FAILURE);
    }
  }

  unsigned int nSeq=0;
  IntervalTree *intervalTree = new IntervalTree[nSeqAlloc];

  map<string, unsigned int, numcomp> seqIndex;
  map<string, unsigned int>::iterator seqIndexIter;

  // Read in all intervals and put into sequence-specific interval trees
  string seq;
  int start;
  int stop;
  while(cin >> seq >> start >> stop) {
    // Need some checking on input format for each line
    if(start < 0 || stop < 0) {
      cerr << "ERROR: found sequence entry with invalid negative coordinates [" << start << "," << stop << "]\n";
      exit(EXIT_FAILURE);
    }
    if(start > stop) {
      // Allow for start > stop, in the case of alignments to reverse strand
      int temp = start;
      start = stop;
      stop = temp;
    }
    seqIndexIter = seqIndex.find(seq);
    unsigned int thisIndex;
    if(seqIndexIter != seqIndex.end()) {
      thisIndex = seqIndexIter->second;
    } else {
      if(nSeq >= nSeqAlloc) {
        cerr << "Only have enough space allocated for " << nSeqAlloc << " IntervalTrees\n";
        exit(1);
      }
      seqIndex.insert(pair<string, unsigned int>(seq,nSeq));
      thisIndex = nSeq;
      nSeq++;
    }
    intervalTree[thisIndex].Insert(new IntInterval(start,stop,1));
  }

  if(!reportDepth) {
    // Determine the union for each interval tree and use it to report the coverage
    map<string, unsigned int> seqCoverage;
    for(map<string, unsigned int>::const_iterator it = seqIndex.begin(); it != seqIndex.end(); ++it) {
      int nInterval = 0;
      int *start;
      int *stop;
      int coverage = 0;
      intervalTree[it->second].GetUnion(&start,&stop,&nInterval);
      for(int i=0; i<nInterval; i++) {
        if(printBlocks)
          cout << it->first << "\t" << start[i] << "\t" << stop[i] << "\n";
        coverage += (stop[i]-start[i]+1);
      }
      delete start;
      delete stop;
      seqCoverage.insert(pair<string, unsigned int>(it->first,coverage));
    }
  
    unsigned int coverage=0;
    for(map<string, unsigned int>::const_iterator it = seqCoverage.begin(); it != seqCoverage.end(); ++it) {
      //cout << it->first << "\t" << it->second << "\n";
      coverage += it->second;
    }
    cout << coverage << "\n";
  } else {
    // Go though each tree and report the coverage depth
    for(map<string, unsigned int>::const_iterator it = seqIndex.begin(); it != seqIndex.end(); ++it) {
      intervalTree[it->second].ResolveOverlaps();
      TemplateStack<void *> *intervals = intervalTree[it->second].EnumerateDepthFirst();
      int nIntervals = intervals->Size();
      for(int i=0; i < nIntervals; i++) {
        IntervalTreeNode *thisNode = (IntervalTreeNode *) (*intervals)[i];
        printf("%s\t%d\t%d\t%0.0lf\n",it->first.c_str(),thisNode->GetInterval()->GetLowPoint(), thisNode->GetInterval()->GetHighPoint(), thisNode->GetInterval()->GetValue());
      }
    }
  }

  delete [] intervalTree;

  return 0;
}
