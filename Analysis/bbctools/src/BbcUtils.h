// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcUtils.h
 *
 *  Created on: Sep 11, 2015
 *      Author: Guy Del Mistro
 */

#ifndef BBCUTILS_H_
#define BBCUTILS_H_

#include <stdexcept>

#include <string>
#include <vector>
using namespace std;

namespace BbcUtils {

string executablePath();

// NOTE: These assume DOS/Linux folder separators are not mixed up in file name
string filePath( const string& str );
string fileName( const string& str );
string fileNameExt( const string& str );

string stringTrim( const string& str );
string collectDigits( const string& str, size_t pos = 0 );
long   stringToInteger( const string& str );
double stringToNumber( const string& str );
string integerToString( long val );
string numberToString( double val, int precision = -1 );
string sigfig( double val );

string replaceAll( const string &str, const string& from, const string& to );

// Return a vector of strings from a single string given a field separator.
// Items between the separator will trimmed of whitespace (other than the specific separator)
// If str is composed of pure whitespace (regardless of separator) an empty vector is returned
vector<string> stringTrimSplit( const string& str, const char sep = ',' );

// Return a vector of integers from a single string given a field separator.
vector<int> stringToIntVector( const string& str, const char sep = ',' );

// Return a pair of <itemsep> (e.g. comma) separated strings of keys and values
// from a list where individual keys were associateed with their values by the <mapsep> characters
// E.g. "a:1,b:2,c:3" -> vector<string>( "a,b,c", "1,2,3" ).
vector<string> mapToPairList( const string& mapstr, const char mapsep = ':', const char itemsep = ',');

class OptParser {
	public:
		OptParser();
		~OptParser(void);

		string PreParseHelpOption( int argc, char* argv[], const string &helpFlags = "-h,?,-?,--help" );
		const string &Parse( int argc, char* argv[], const string &optstr, int minArgs = 0, int maxArgs = 0 );

		string getOptValue( const string &key, const string &defval = "", size_t idx = 0 ) const;
		double getOptNumber( const string &key, double defval = 0, size_t idx = 0 ) const;
		long   getOptInteger( const string &key, long defval = 0, size_t idx = 0 ) const;
		bool   getOptBoolean( const string &key, bool defval = false ) const;
		vector<string> getArgs() const;

	private:
		string m_errMsg;
		vector<string> m_optlist;
		vector<string> m_optvals;
		vector<string> m_argvals;

		static const char m_sep = '\t';
		static const char m_eqv = '=';

		int optionIndex( const string &key ) const;
		string toKey( const string &val ) const;
		vector<string> toKeys( const string &val ) const;
};

} // bbcUtils

#endif /* BBCUTILS_H_ */
