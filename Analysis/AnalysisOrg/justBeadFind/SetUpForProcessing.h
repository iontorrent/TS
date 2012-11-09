/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SETUPFORPROCESSING_H
#define SETUPFORPROCESSING_H

#include <stdio.h>
#include <string.h>

#include "CommandLineOpts.h"
#include "SeqList.h"
#include "TrackProgress.h"
#include "ProgramState.h"
#include "CaptureImageState.h"
#include "ImageSpecClass.h"
#include "SlicedPrequel.h"


// Set up functions, shared between Analysis and justBeadFind

void SetUpKeys(SeqListClass &my_keys, KeyContext &key_context, FlowContext &flow_context);
void SetUpToProcessImages ( ImageSpecClass &my_image_spec, CommandLineOpts &inception_state );

void SetUpOrLoadInitialState(CommandLineOpts &inception_state, SeqListClass &my_keys, TrackProgress &my_progress, ImageSpecClass &my_image_spec, SlicedPrequel& my_prequel_setup);
void LoadBeadFindState(CommandLineOpts &inception_state, SeqListClass &my_keys, ImageSpecClass &my_image_spec);

#endif // SETUPFORPROCESSING_H
