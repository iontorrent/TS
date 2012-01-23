// test case for LA_SYMM_BAND_MAT_DOUBLE_H
// and LA_SYMM_BAND_FACT_DOUBLE_H

#include <stdlib.h>        

#include "lafnames.h"
#include LA_GEN_MAT_DOUBLE_H
#include LA_SYMM_BAND_MAT_DOUBLE_H
#include LA_SYMM_BAND_FACT_DOUBLE_H

int main(void)
{
  int n=5;
  LaSymmBandMatDouble A(n, 2);

  A(0,0)=16;
  A(1,0)=4;
  A(2,0)=4;
  A(1,1)=5;
  A(2,1)=7;
  A(3,1)=-4;
  A(2,2)=19;
  A(3,2)=-18;
  A(4,2)=3;
  A(3,3)=45;
  A(4,3)=6;
  A(4,4)=6;
  
  printf("\nSymmetric 5 x 5 bandes matrix with bandwidth 5.\n");
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      printf("%f \t", A(i,j));
    }
    printf("\n");
  }

  LaGenMatDouble B(5, 2);
  B(0,0)=44;
  B(1,0)=-5;
  B(2,0)=-58;
  B(3,0)=116;
  B(4,0)=0;
  B(0,1)=-8;
  B(1,1)=6;
  B(2,1)=28;
  B(3,1)=33;
  B(4,1)=36;
  printf("\nSolving A*X = B with B := \n");
  for (int i=0; i<n; i++)
  {
    printf("%f \t%f\n", B(i,0), B(i,1));
  }

  LaSymmBandMatDouble AF(A);
  LaSymmBandMatFactorizeIP(AF);

  printf("\nCholesky factorization.\n");
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      printf("%f \t", AF(i,j));
    }
    printf("\n");
  }

  LaLinearSolveIP(A,B);

  printf("\nSolution X:= \n");
  for (int i=0; i<n; i++)
  {
    printf("%f \t%f\n", B(i,0), B(i,1));
  }
  printf("\nShould be <<3,1,-2,2,-1>|<-1,0,2,1,4>>. \n");

  return 0;
}
