#include "armadillo"

#if defined(__GNUG__)
  #if ( (__GNUC__ == 4) && (__GNUC_MINOR__ < 1) )
    #warning "Your compiler is rather old.  Programs using template libraries (such as Armadillo) may not compile correctly."
  #endif
#endif


namespace arma
  {
  namespace junk
    {

    #if defined(ARMA_USE_ATLAS)
      void
      pull_atlas()
        {
        int    x;
        double y;

        arma::atlas::clapack_dgetrf(CblasColMajor, x, x, &y, x, &x);
        arma::atlas::cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, x, x, x, y, &y, x, &y, x, y, &y, x);
        }
    #endif

    #if defined(ARMA_USE_LAPACK)
      void
      pull_lapack()
        {
        int    x;
        double y;

        arma::lapack::dgetrf_(&x, &x, &y, &x, &x, &x);
        }
    #endif

    #if defined(ARMA_USE_BLAS)
      void
      pull_blas()
        {
        char   c;
        int    x;
        double y;

        arma::blas::dgemm_(&c, &c, &x, &x, &x, &y, &y, &x, &y, &x, &y, &y, &x);
        }
    #endif

    }
  }
