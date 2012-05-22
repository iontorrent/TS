/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGFITLEVMARDAT_H
#define BKGFITLEVMARDAT_H

#include <armadillo>

class BkgFitLevMarDat {
 public:
  arma::Mat<double> *jac;
  arma::Mat<double> *jtj;
  arma::Mat<double> *lhs;
  arma::Col<double> *rhs;
  arma::Col<double> *delta;

  BkgFitLevMarDat() {
    jac = new arma::Mat<double>();
    jtj = new arma::Mat<double>();
    lhs = new arma::Mat<double>();
    rhs = new arma::Col<double>();
    delta = new arma::Col<double>();
  }

  ~BkgFitLevMarDat() {
    delete jac;
    delete jtj;
    delete lhs;
    delete rhs;
    delete delta;
  }
  double GetDelta(int index) { return delta->at(index); }

};

#endif // BKGFITLEVMARDAT_H
