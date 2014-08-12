/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ControlSingleFlow.h"


// goal here is to turn this into json-serialized control
// {singleflowfit: {krate_adj_limit: 2.0},{kmult_low_limit:0.65}}
// as part of a huge meta-object for controls.


ControlSingleFlow::ControlSingleFlow(){
krate_adj_limit = 2.0f; // you must be at least this tall to ride the ride
dampen_kmult = 0.0f;
kmult_low_limit = 0.65;
kmult_hi_limit = 1.75f;
}
