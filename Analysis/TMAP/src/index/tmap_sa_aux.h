/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TMAP_SA_AUX_H
#define TMAP_SA_AUX_H

/*! 
  returns the suffix array position given the occurrence position
  @param  sa   the suffix array
  @param  bwt  the bwt structure 
  @param  k    the suffix array position
  @return      the pac position
*/
tmap_bwt_int_t 
tmap_sa_pac_pos_aux(const tmap_sa_t *sa, const tmap_bwt_t *bwt, tmap_bwt_int_t k);

#endif // TMAP_SA_AUX_H
