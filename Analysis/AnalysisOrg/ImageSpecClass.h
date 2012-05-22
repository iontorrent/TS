/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGESPECCLASS_H
#define IMAGESPECCLASS_H


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
        void DeriveSpecsFromDat(SystemContext &sys_context,ImageControlOpts &img_control, SpatialContext &loc_context, int numFlows, char *experimentName);
        int LeadTimeForChipSize();
        ~ImageSpecClass();
    };
    

#endif // IMAGESPECCLASS_H