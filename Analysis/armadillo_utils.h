/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ARMADILLO_UTILS_H
#define ARMADILLO_UTILS_H

#include <armadillo>

inline bool is_pos_def(const arma::mat22& M)
{
	// Sylvester's test:
	return M.at(0,0) > 0 and arma::det(M) > 0;
}

#endif // ARMADILLO_UTILS_H

