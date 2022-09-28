/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
/*
Copyright (C) 2009 Affymetrix Inc.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation; either version 2.1 of the License,
or (at your option) any later version.

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
for more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,

59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
*/
#ifndef FITDENSITY_H
#define FITDENSITY_H

#include <cstddef>
#include <vector>
#include <cmath>

namespace FitDensity {
  void XLimits(std::vector<double> const& data, double const bandwidth, std::vector<double>& out, double const range_limit=4.0f, double const bw_increment=3.0f);
  void kdensity(std::vector<double> const &dat, std::vector<double> &density, std::vector<double> &xOut, std::vector<double> const& weight, double const bandWidth, double const from, double const to);
  int findpeaks(std::vector<size_t> &valleyIndex, std::vector<size_t> &peakIndex, std::vector<double> const& y, double const delta, std::vector<double> const& x);
  double findBandWidth(std::vector<double> const& x);
  double SquaredIntegral(std::vector<double> const& x, std::vector<double> const& y);
  double trapzoid(std::vector<double> const& x, std::vector<double> const& y);
  double phi(double const x);
}

/** 
 * Standard normal probability density function.
 * @param x - value of interest.
 * @return - density at value supplied
 */
inline double FitDensity::phi(double const x){
   double one_over_sqrt_2mpi = 0.39894228040143267793994605993438; //1/sqrt(2 * pi)
   return one_over_sqrt_2mpi * exp(-0.5 * x * x);
}

#endif // FITDENSITY_H
