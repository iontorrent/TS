/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * TestNewLayout.cu
 *
 *  Created on: Feb 18, 2014
 *      Author: jakob
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "LayoutTester.h"


using namespace std;

int main(int argc, char *argv[] )
{

  int cacheSetting = 0;
  char * imgfileName = "imgDumpFlow20.dat";
  char * bkgfileName = "bkgDumpFlow20.dat";
  char * resultfileName = "ResultDumpFlow20.dat";
  int blockW = 128;
  int blockH = 1;

  char *cvalue = NULL;
  int index;
  int c;
  float epsilon = 0.001f;
  opterr = 0;

  while ((c = getopt (argc, argv, "c:i:s:r:e:x:y:h")) != -1)
  {
    switch (c)
    {
      case 'c':
        cacheSetting = atoi(optarg);
        break;
      case 'x':
              blockW = atoi(optarg);
              break;
      case 'y':
              blockH = atoi(optarg);
              break;
      case 'e':
        epsilon= atof(optarg);
        break;
      case 'i':
        imgfileName = optarg;
        break;
      case 's':
        bkgfileName = optarg;
        break;
      case 'r':
        resultfileName = optarg;
        break;
      case '?':
        if (optopt == 'c' || optopt == 's')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,
              "Unknown option character `\\x%x'.\n",
              optopt);
        return 1;
      case 'h':
      default:
        cout << endl << "Usage:" <<endl;
        cout << "TestLayout [option]" <<endl;
        cout << "-c\t\tcache setting 0 equal,1 shared prefered, 2 l1 prefered" <<endl;
        cout << "-x <int>\tthreadblock width" <<endl;
        cout << "-y <int>\tthreadblock height" <<endl;
        cout << "-e <float>\tepsilon for result comparison" <<endl;
        cout << "-i <file>\tRaw Image input file name (default: imgDumpFlow20.dat)" <<endl;
        cout << "-s <file>\tBackground Model input file name (default: bkgDumpFlow20.dat)" <<endl;
        cout << "-r <file>\tResult dump file name (default: ResultDumpFlow20.dat)" <<endl;
        cout << "-h\t\thelp, displays this message" << endl;
        //abort();
        exit(0);
    }

  }


  testLayout(imgfileName, bkgfileName, resultfileName, epsilon, cacheSetting, blockW ,blockH );

  return 0;

}






