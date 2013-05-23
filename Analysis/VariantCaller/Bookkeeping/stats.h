/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef STATS_H
#define STATS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdbool.h>

double poisson(double, double);
void calc_score_hyp(int num_reads, int num_hyps, float **prob, float *score, int *count, float min_dif, float min_best);


#endif // STATS_H
