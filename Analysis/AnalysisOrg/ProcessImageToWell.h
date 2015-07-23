/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PROCESSIMAGETOWELL_H
#define PROCESSIMAGETOWELL_H

#include <string>
#include <vector>
#include "CommandLineOpts.h"
#include "Mask.h"
#include "RawWells.h"
#include "Region.h"
#include "SeqList.h"
#include "TrackProgress.h"
#include "SlicedPrequel.h"
#include "ImageSpecClass.h"
#include "OptBase.h"

void RealImagesToWells ( 
    OptArgs &opts,
    CommandLineOpts &inception_state,                        
    SeqListClass &my_keys,
    TrackProgress &my_progress, 
    ImageSpecClass &my_image_spec,
    SlicedPrequel &my_prequel_setup);

#endif // PROCESSIMAGETOWELL_H
