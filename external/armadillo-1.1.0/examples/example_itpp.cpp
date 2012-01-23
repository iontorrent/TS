// Example of how to use Armadillo in conjunction
// with the the IT++ library (also known as ITPP).
// 
// Instead of #include "armadillo", use #include "armadillo_itpp"
//
// Caveat:
// Armadillo is licensed under the LGPL, which is more permissive
// than the GPL. For example, the LGPL doesn't "infect" your code
// with a specific license. In contrast, IT++ is licensed under
// the GPL. As such, using the IT++ library automatically causes
// your code to fall under the GPL license.


#include <iostream>
#include <itpp/itbase.h>

#include "armadillo_itpp"

using namespace std;
using namespace arma;


int main(int argc, char** argv)
  {
  mat A_arma = \
   "\
   0.924005  0.720545  0.355721  0.302286;\
   0.065372  0.153901  0.277607  0.420199;\
   0.059450  0.194317  0.708485  0.998625;\
   0.114934  0.651950  0.667288  0.755914;\
   ";
  
  itpp::mat B_itpp = \
   "\
   0.555950  0.274690  0.540605  0.798938;\
   0.108929  0.830123  0.891726  0.895283;\
   0.948014  0.973234  0.216504  0.883152;\
   0.023787  0.675382  0.231751  0.450332;\
   ";
  
  
  itpp::mat A_itpp = conv_to<itpp::mat>::from(A_arma);
  
  cout << "A_arma = " << endl << A_arma << endl;
  cout << "A_itpp = " << endl << A_itpp << endl;
  
  
  mat B_arma = conv_to<mat>::from(B_itpp);
  
  cout << "B_arma = " << endl << B_arma << endl;
  cout << "B_itpp = " << endl << B_itpp << endl;
  
  
  mat       C_arma = A_arma + B_arma;
  itpp::mat C_itpp = A_itpp + B_itpp;
  
  
  cout << "C_arma = " << endl << C_arma << endl;
  cout << "C_itpp = " << endl << C_itpp << endl;
  
  
  return 0;
  }

