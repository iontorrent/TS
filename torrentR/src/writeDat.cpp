/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <string>
#include <iostream>
#include <Rcpp.h>
#include "crop/Acq.h"

using namespace std;

class RawDat : public Acq
{

  public:
    void SetTraceTimestamps( int t ) {
      timestamps = new int[ numFrames ];

      for ( int i = 0; i < numFrames; ++i )
        timestamps[ i ] = i * t;
    };

    virtual ~RawDat() {
      delete[] timestamps;
    };
};

RcppExport SEXP writeDat( SEXP RdatFile, SEXP Rcol, SEXP Rrow, SEXP RWidth, SEXP RHeight, SEXP RSignal )
{

  SEXP ret = R_NilValue;        // Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    char* datFile           = ( char * )Rcpp::as<const char*>( RdatFile );
    int colInt              = Rcpp::as< int >( Rcol );
    int rowInt              = Rcpp::as< int >( Rrow );
    int   width             = Rcpp::as<int>( RWidth );
    int   height            = Rcpp::as<int>( RHeight );
    RcppMatrix<double> signal = Rcpp::as< RcppMatrix<double> >( RSignal );


    //Rprintf("File Name: %s col: %d, row: %d, width: %d, signal: %g, %g, %g\n",datFile, signal.cols(), signal.rows(), width, signal(0,0), signal(1,0), signal(0,1) );

    double* tempTrace = new double[ signal.getDim2()] ;
    int linearIndex = 0;
    RawDat saver;

    saver.SetSize( width, height, signal.getDim2(), signal.getDim2() );
    saver.SetTraceTimestamps( 69 );

    for ( int i = 0; i < height; i++ ) {
      for ( int k = 0; k < width; k++ ) {
        for ( int l = 0; l < signal.getDim2(); l++ ) //copy into raw vector
          tempTrace[l] = signal( linearIndex, l );

        saver.SetWellTrace( tempTrace, k, i );

        linearIndex++;
      }
    }

    saver.Write( datFile, colInt, rowInt, width, height );

    delete[] tempTrace;
    RcppResultSet rs;
    ret = rs.getReturnList();

  }
  catch ( exception& ex ) {
    exceptionMesg = copyMessageToR( ex.what() );
  }
  catch ( ... ) {
    exceptionMesg = copyMessageToR( "unknown reason" );
  }

  if ( exceptionMesg != NULL )
    Rf_error( exceptionMesg );

  return ret;
}
