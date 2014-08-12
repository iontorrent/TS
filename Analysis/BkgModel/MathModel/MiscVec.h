/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef MISCVEC_H
#define MISCVEC_H

void MultiplyVectorByScalar_Vec(float *my_vec, float my_scalar, int len);

void Dfderr_Step_Vec(int flow_block_size, float** dst, float** et, float** em, int len);
void Dfdgain_Step_Vec(int flow_block_size, float** dst, float** src, float** em, int len, float gain);

#endif // MISCVEC_H

