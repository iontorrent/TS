/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include <string>
#include <vector>
#include <sstream>
#include "SamUtils/BAMReader.h"
#include "SamUtils/BAMUtils.h"

using namespace std;

RcppExport SEXP readBam( SEXP RbamFile )
{

  SEXP ret = R_NilValue;        // Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {

    char* bamFileName         = ( char * )Rcpp::as<const char*>( RbamFile );
    BAMReader bamReader;

    bamReader.open( bamFileName );

    if ( !bamReader.is_open() )
      throw std::string( "Can't open bamFile" );

    int nReadOut = 0;

    for ( BAMReader::iterator it = bamReader.get_iterator(); it.good(); it.next() ) {
      nReadOut++;
    }

    bamReader.close();
    bamReader.open( bamFileName );
    
    vector<int>    out_q7Len( nReadOut );
    vector<int>    out_q10Len( nReadOut );
    vector<int>    out_q17Len( nReadOut );
    vector<int>    out_q20Len( nReadOut );
    vector<int>    out_q47Len( nReadOut );
    std::vector< std::string > out_name( nReadOut );
    vector<int>    out_strand( nReadOut );
    vector<int>    out_tStart( nReadOut );
    vector<int>    out_tLen( nReadOut );
    vector<int>    out_qLen( nReadOut );
    vector<int>    out_match( nReadOut );
    vector<double>  out_percent_id( nReadOut );
    std::vector< std::string > out_rname (nReadOut);
    std::vector< std::string > out_qDNA_a( nReadOut );
    std::vector< std::string > out_match_a( nReadOut );
    std::vector< std::string > out_tDNA_a( nReadOut );
    vector<int>    out_homErrs( nReadOut );
    vector<int>    out_mmErrs( nReadOut );
    vector<int>    out_indelErrs( nReadOut );

    int i = 0;
    for ( BAMReader::iterator it = bamReader.get_iterator(); it.good(); it.next(), ++i ) {
      BAMRead br = it.get();
      BAMUtils bamRead( br );
      bamRead.get_q_length();
      out_q7Len.at( i ) = bamRead.get_phred_len( 7 );
      out_q10Len.at( i ) = bamRead.get_phred_len( 10 );
      out_q17Len.at( i ) = bamRead.get_phred_len( 17 );
      out_q20Len.at( i ) = bamRead.get_phred_len( 20 );
      out_q47Len.at( i ) = bamRead.get_phred_len( 47 );
      out_name.at( i ) = bamRead.get_name();
      out_strand.at( i ) = bamRead.get_strand();
      out_tStart.at( i ) = bamRead.get_t_start();
      out_tLen.at( i ) = bamRead.get_t_length();
      out_qLen.at( i ) = bamRead.get_q_length();
      out_match.at( i ) = bamRead.get_match();
      out_percent_id.at( i ) = bamRead.get_percent_id();
      out_qDNA_a.at( i ) = bamRead.get_qdna();
      out_match_a.at( i ) = bamRead.get_matcha();
      out_tDNA_a.at( i ) = bamRead.get_tdna();
      out_indelErrs.at( i ) = bamRead.get_indel_errors();
      out_mmErrs.at( i ) = bamRead.get_mismatch_errors();
      out_homErrs.at( i ) = bamRead.get_homo_errors();
      out_rname.at (i) = bamRead.get_rname();
    }

    RcppResultSet rs;

    rs.add( "q7Len",            out_q7Len );
    rs.add( "q10Len",            out_q10Len );
    rs.add( "q17Len",            out_q17Len );
    rs.add( "q20Len",            out_q20Len );
    rs.add( "q47Len",            out_q47Len );
    rs.add( "name",             out_name );
    rs.add( "strand",            out_strand );
    rs.add( "tStart",            out_tStart );
    rs.add( "rName",            out_rname );
    rs.add( "tLen",            out_tLen );
    rs.add( "qLen",            out_qLen );
    rs.add( "matchLen",            out_match );
    rs.add( "percent_id",       out_percent_id );
    rs.add( "qDNA",            out_qDNA_a );
    rs.add( "match",            out_match_a );
    rs.add( "tDNA",            out_tDNA_a );
    rs.add( "indelErrs",            out_indelErrs );
    rs.add( "mismatchErrs",            out_mmErrs );
    rs.add( "homopolymerErrs",            out_homErrs );
    ret = rs.getReturnList();
  }
  catch ( std::exception& ex ) {
    exceptionMesg = copyMessageToR( ex.what() );
  }
  catch ( ... ) {
    exceptionMesg = copyMessageToR( "unknown reason" );
  }

  if ( exceptionMesg != NULL )
    Rf_error( exceptionMesg );

  return ret;
}
