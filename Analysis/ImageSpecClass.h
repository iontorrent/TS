/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGESPECCLASS_H
#define IMAGESPECCLASS_H

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <limits.h>
#include <signal.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <armadillo>

#include "Image.h"
#include "CommandLineOpts.h"


    class ImageSpecClass{
      public:
        int rows, cols;
        int scale_of_chip;
        unsigned int uncompFrames;
        int *timestamps;
        bool vfr_enabled;
         char *acqPrefix;
        
        ImageSpecClass();
        void DeriveSpecsFromDat(CommandLineOpts &clo, int numFlows, char *experimentName);
        int LeadTimeForChipSize();
        ~ImageSpecClass();
    };
    

#endif // IMAGESPECCLASS_H