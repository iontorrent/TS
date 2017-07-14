/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITMATDAT_H
#define BKGFITMATDAT_H

#include <armadillo>

class  BkgFitMatDat {
 public:
  arma::Mat<double> *jtj;
  arma::Col<double> *rhs;
  arma::Col<double> *delta;

  BkgFitMatDat() {
    jtj = new arma::Mat<double>();
    rhs = new arma::Col<double>();
    delta = new arma::Col<double>();
  }

  ~BkgFitMatDat() {
    delete jtj;
    delete rhs;
    delete delta;
  }
};

#endif // BKGFITMATDAT_H
