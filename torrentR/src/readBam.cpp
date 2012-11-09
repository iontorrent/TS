/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include <string>
#include <vector>
#include <sstream>
#include "../Analysis/file-io/ion_util.h"
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

RcppExport SEXP readBamWithLocationFilter( SEXP RbamFile, SEXP R_seq_location, SEXP R_seq_range )
{

  SEXP ret = R_NilValue;        // Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
		int seq_location          = Rcpp::as<int>(R_seq_location);
		int seq_range           = Rcpp::as<int>(R_seq_range);

    // crude interval - note I'm not even specifying the sequence name!
    int seq_low = seq_location-seq_range;
    int seq_hi = seq_location+seq_range;


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
    
// preallocate way too much memory
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

    // read and filter down
    // there has to be a quicker way
    int i = 0;
    for ( BAMReader::iterator it = bamReader.get_iterator(); it.good(); it.next()) {
      BAMRead br = it.get();
      BAMUtils bamRead( br );
      bamRead.get_q_length();
      int tstart = bamRead.get_t_start();
      // filter reads by sequence location
      if ((seq_low<tstart) & (seq_hi>tstart)){
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
        i+=1;
      }
    }

    // shrink to actual numbers read
    out_q7Len.resize(i);
    out_q10Len.resize(i);
    out_q17Len.resize(i);
    out_q20Len.resize(i);
    out_name.resize(i);
    out_strand.resize(i);
    out_tStart.resize(i);
    out_rname.resize(i);
    out_tLen.resize(i);
    out_qLen.resize(i);
    out_match.resize(i);
    out_percent_id.resize(i);
    out_qDNA_a.resize(i);
    out_match_a.resize(i);
    out_tDNA_a.resize(i);
    out_indelErrs.resize(i);
    out_mmErrs.resize(i);
    out_homErrs.resize(i);
    // so R gets only a short list back

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

// filter to just the guys within a rectangle on the chip
RcppExport SEXP readBamWithSpatialFilter( SEXP RbamFile, SEXP R_col_min, SEXP R_col_max, SEXP R_row_min, SEXP R_row_max )
{

  SEXP ret = R_NilValue;        // Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
		int col_min          = Rcpp::as<int>(R_col_min);
		int col_max          = Rcpp::as<int>(R_col_max);
		int row_min          = Rcpp::as<int>(R_row_min);
		int row_max          = Rcpp::as<int>(R_row_max);

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
    
// preallocate way too much memory
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

    // read and filter down
    // there has to be a quicker way
    int i = 0;
    int mi=0;
    int row,col;
    for ( BAMReader::iterator it = bamReader.get_iterator(); it.good(); it.next()) {
      BAMRead br = it.get();
      BAMUtils bamRead( br );
      bamRead.get_q_length();
      string tname = bamRead.get_name();
      ion_readname_to_rowcol(tname.c_str(),&row, &col);
      //if (mi % 100==0)
        //printf("%s %d %d %d %d %d %d\n",tname.c_str(),row,col,col_min,col_max,row_min,row_max);
      // filter reads by sequence location
      if ((col_min<=col) & (col<=col_max) & (row_min<=row) & (row<=row_max)){
        //if (i % 100==0)
          //printf("IN: %s %d %d %d %d %d %d\n",tname.c_str(),row,col,col_min,col_max,row_min,row_max);
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
        i+=1;
      }
      mi+=1;
    }

    // shrink to actual numbers read
    out_q7Len.resize(i);
    out_q10Len.resize(i);
    out_q17Len.resize(i);
    out_q20Len.resize(i);
    out_name.resize(i);
    out_strand.resize(i);
    out_tStart.resize(i);
    out_rname.resize(i);
    out_tLen.resize(i);
    out_qLen.resize(i);
    out_match.resize(i);
    out_percent_id.resize(i);
    out_qDNA_a.resize(i);
    out_match_a.resize(i);
    out_tDNA_a.resize(i);
    out_indelErrs.resize(i);
    out_mmErrs.resize(i);
    out_homErrs.resize(i);
    // so R gets only a short list back

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

