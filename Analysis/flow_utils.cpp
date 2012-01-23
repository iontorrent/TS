/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "flow_utils.h"

using namespace std;

// Return flow position of a particular base position doing it the naive way
int getFlowNum(string& seq, string& flow_order, int seq_position)
{
    int flow_position = 0;
    int cur_pos_in_seq = 0;
    for (flowgram_it i(flow_order, seq); i.good() && cur_pos_in_seq < seq_position; i.next(), flow_position++) {
        cur_pos_in_seq += i.hplen();
    }

    return flow_position;
}
