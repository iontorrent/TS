/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>

RcppExport SEXP fitBkgTrace(
  SEXP Rsig,
  SEXP RnFrame,
  SEXP RnFlow
) {

  SEXP rl = R_NilValue;
  char *exceptionMesg = NULL;

  try {
    RcppMatrix<double> sig(Rsig);
    int nWell = sig.rows();
    int nCol = sig.cols();
    int nFrame = Rcpp::as<int>(RnFrame);
    int nFlow  = Rcpp::as<int>(RnFlow);

    if(nWell <= 0) {
      std::string exception = "Empty matrix supplied, nothing to fit\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else if(nFlow*nFrame != nCol) {
      std::string exception = "Number of columns in signal matrix should equal nFrame * nFlow\n";
      exceptionMesg = copyMessageToR(exception.c_str());
    } else {
      RcppMatrix<int> bkg(nFrame,nFlow);
      for(int iFlow=0; iFlow<nFlow; iFlow++) {
        for(int iFrame=0, frameIndex=iFlow*nFrame; iFrame<nFrame; iFrame++, frameIndex++) {
          double sum=0;
          for(int iWell=0; iWell<nWell; iWell++)
            sum += sig(iWell,frameIndex);
          sum /= nWell;
          bkg(iFrame,iFlow) = sum;
        }
      }
   
      // Build result set to be returned as a list to R.
      RcppResultSet rs;
      rs.add("bkg", bkg);

      // Set the list to be returned to R.
      rl = rs.getReturnList();

      // Clear allocated memory
    }
  } catch(std::exception& ex) {
    exceptionMesg = copyMessageToR(ex.what());
  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return rl;
}
