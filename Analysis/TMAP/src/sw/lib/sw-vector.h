/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SW_VECTOR_H
#define SW_VECTOR_H

int	sw_vector_setup(int, int, int, int, int, int, int, int, int, bool);
int
sw_vector(uint8_t *target, int32_t target_len,
          uint8_t *query, int32_t query_len);
int sw_vector_cleanup(void);
#endif // SW_VECTOR_H
