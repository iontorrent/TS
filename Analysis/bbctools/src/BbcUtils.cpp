// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcUtils.cpp
 *
 *  Created on: Sep 2, 2015
 *      Author: Guy Del Mistro
 */
#include "BbcUtils.h"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <ctype.h>

// these just to get executable path!
#include <unistd.h>
#include <libgen.h>

using namespace std;

namespace BbcUtils {

string executablePath()
{
	char result[5000];
	ssize_t len = readlink( "/proc/self/exe", result, 5000 );
	if( len == -1 ) return ".";
	result[len] = 0;
	return string( dirname(result) );
}

string filePath( const string& str )
{
	// assume linux file ext not in windows file name (et visa versa)
	size_t i = str.find_last_of("/\\");
	if( i == string::npos ) return str;
	return str.substr(0,i);
}

string fileName( const string& str )
{
	// assume linux file ext not in windows file name (et visa versa)
	size_t i = str.find_last_of("/\\");
	if( i == string::npos ) return str;
	return str.substr(i+1);
}

string fileNameExt( const string& str )
{
	string fname = fileName(str);
	size_t i = fname.find_last_of('.');
	if( i == string::npos ) return fname;
	return str.substr(i+1);
}

string stringTrim( const string& str )
{
	if( !str.size() ) return "";
    size_t first = str.find_first_not_of(" \t\n\v\r\f");
    if( first == string::npos ) return "";
    size_t last = str.find_last_not_of(" \t\n\v\r\f");
    return str.substr( first, (last-first+1) );
}

string collectDigits( const string& str, size_t pos )
{
	size_t len = str.size(), lpos = pos;
	while( lpos < len ) {
		const char chr = str.at(lpos);
		if( chr < '0' || chr > '9' ) break;
		++lpos;
	}
	return str.substr(pos,lpos-pos);
}

long stringToInteger( const string& str )
{
	string nstr = stringTrim(str);
	char *endptr;
	long i = strtol(str.c_str(),&endptr,10);
	if( *endptr ) throw runtime_error("Failed to parse integer from '"+nstr+"'");
	return i;
}

double stringToNumber( const string& str )
{
	string nstr = stringTrim(str);
	char *endptr;
	double d = strtod(str.c_str(),&endptr);
	if( *endptr ) throw runtime_error("Failed to parse number from '"+nstr+"'");
	return d;
}

string integerToString( long val )
{
	ostringstream oss;
	oss << val;
	return oss.str();
}

string numberToString( double val, int precision )
{
	ostringstream oss;
	if( precision >= 0 ) {
		oss << fixed << setprecision(precision);
	}
	oss << val;
	return oss.str();
}

string sigfig( double val )
{
	double sval = val < 0 ? -val : val;
	if( sval >= 1000 ) return numberToString( val, 0 );
	if( sval >= 100 ) return numberToString( val, 1 );
	if( sval >= 10 ) return numberToString( val, 2 );
	return numberToString( val, 3 );
}

string replaceAll( const string &str, const string& from, const string& to )
{
	string retstr = str;
    size_t pos = 0;
    while( (pos = retstr.find(from,pos)) != string::npos) {
    	retstr.replace( pos, from.length(), to );
        pos += to.length();
    }
	return retstr;
}

vector<string> stringTrimSplit( const string& str, const char sep )
{
	string field;
	vector<string> fields;
	if( !stringTrim(str).empty() ) {
		stringstream ss(str+sep);
		while( getline(ss,field,sep) ) {
			fields.push_back(stringTrim(field));
		}
	}
	return fields;
}

vector<int> stringToIntVector( const string& str, const char sep ) {
	vector<string> fields = stringTrimSplit(str,sep);
    vector<int> retvec;
    for( vector<string>::iterator it = fields.begin(); it != fields.end(); ++it )
{
        retvec.push_back((int)stringToInteger(*it));
    }
 	return retvec;
}

vector<string> mapToPairList( const string& mapstr, const char mapsep, const char itemsep )
{
	string keys, vals;
	vector<string> list = stringTrimSplit( mapstr, itemsep );
	for( size_t i = 0; i < list.size(); ++ i) {
		vector<string> kvp = stringTrimSplit( list[i], mapsep );
		if( kvp.size() == 0 ) continue;
		if( !keys.empty() ) {
			keys += itemsep;
			vals += itemsep;
		}
		keys += kvp[0];
		if( kvp.size() > 1 ) vals += kvp[1];
	}
	vector<string> retvec;
	if( keys.size() ) {
		retvec.push_back(keys);
		retvec.push_back(vals);
	}
	return retvec;
}

//
// --------- OptParser ---------
//

OptParser::OptParser()
{
}

OptParser::~OptParser()
{
}

string OptParser::PreParseHelpOption( int argc, char* argv[], const string &helpFlags )
{
	// call to circumvent full option syntax validation, etc.
	if( helpFlags.empty() ) return "";
	vector<string> helpopts = stringTrimSplit( helpFlags, ',' );
	for( size_t j = 0; j < helpopts.size(); ++j ) {
		const string &hopt = helpopts[j];
		for( int i = 0; i < argc; ++i ) {
			if( hopt == argv[i] ) return hopt;
		}
	}
	return "";
}

const string &OptParser::Parse( int argc, char* argv[], const string &optstr, int minArgs, int maxArgs )
{
	// optstr format examples: "abc:" => as getopt()
	// " a  bc " => any number of spaces used to separate flag tokens
    // "a-about -bc" => alias --about for -a flag and --bc flag (no single base alias)
	// "abc,d;e::f,;" => c takes integer value, d takes number, e takes 2 strings, f takes an integer and a number
	// ":X:Yabc" => 'X' and 'Y' are optional parsing control directives

	// Return an error message if the (user) command line arguments do not parse
	// Throws runtime_error() if the options parsing string is in error

	m_errMsg = "";
	m_optlist.clear();
	m_optvals.clear();
	m_argvals.clear();

	// first build list of string find-able aliases for each option, plus expected value data
	if( optstr.empty() ) {
		m_errMsg = "Invalid empty option parsing string.";
		throw runtime_error(m_errMsg);
	}
	string cpystr = optstr + " ";

	// check for preliminary parsing control directives
	// :A => collect All, :L => keep Last value, :E => Error (default)
	// :S => Suppress warnings, :W show Warnings (default)
	char multiKey =  'L';	// default
	char warnLevel = 'W';	// default
	while( cpystr[0] == ':' && optstr.size() > 1 ) {
		char chr1 = cpystr[1];
		if( chr1 == 'A' || chr1 == 'L' || chr1 == 'E' ) {
			multiKey = chr1;
		} else if( chr1 == 'W' || chr1 == 'S' ) {
			warnLevel = chr1;
		} else {
			m_errMsg = "Error parsing option string: Illegal control option '"+cpystr.substr(0,2)+"'";
			throw runtime_error(m_errMsg);
		}
		cpystr = cpystr.substr(2);
	}

    // parse the options parsing string
	const string sep(1,m_sep);
	const string terms = " ,;:";
	for( size_t i = 0, s = 0; i < cpystr.size(); ++i ) {
		size_t e = terms.find(cpystr[i]);
		if( e != string::npos ) {
			// syntax errors to start with non-option values ?
			if( s == i ) {
				if( !e ) {
					s = i+1;
					continue; // ignore excess spaces
				}
				m_errMsg = "Error parsing option string: Naked value token '"+cpystr.substr(i,1)+"'";
				throw runtime_error(m_errMsg);
			}
			// split by single characters up to before hyphen or term
			string opts;
			while( s < i && cpystr[s] != m_eqv ) {
				opts = cpystr.substr(s,1);
				if( optionIndex(opts) >= 0 ) {
					m_errMsg = "Error parsing option string: Key '"+opts+"' is used more than once";
					throw runtime_error(m_errMsg);
				}
				if( cpystr[s] >= '0' && cpystr[s] <= '9' ) {
					m_errMsg = "Error parsing option string: Single digit key '"+opts+"' not allowed";
					throw runtime_error(m_errMsg);
				}
				++s;
				// hold last in list to share alias variables
				opts = sep + opts;
				if( cpystr[s] != m_eqv && s < i ) {
					opts += sep + terms[0];
					m_optlist.push_back(opts);
					m_optvals.push_back("");
				}
			}
			// add aliases for current key
			while( s < i ) {
				// strip replace first m_eqv
				if( ++s == i ) {
					m_errMsg = "Error parsing option string: Naked long key token '"+string(1,m_eqv)+"'";
					throw runtime_error(m_errMsg);
				}
				// add long key, including extra
				string subkey;
				while( s < i && cpystr[s] == m_eqv ) {
					subkey += cpystr[s++];
				}
				while( s < i && cpystr[s] != m_eqv ) {
					subkey += cpystr[s++];
				}
				if( optionIndex(subkey) >= 0 ) {
					m_errMsg = "Error parsing option string: Key '"+subkey+"' is used more than once";
					throw runtime_error(m_errMsg);
				}
				opts += sep + subkey;
			}
			opts += sep + terms[e];
			// search forward for more values
			while( ++i < cpystr.size() && (e = terms.find(cpystr[i])) != string::npos ) {
				if( !e ) break;
				opts += terms[e];
			}
			s = i--;
			m_optlist.push_back(opts);
			m_optvals.push_back("");
		}
	}
	int lastOptIdx = 0;
	for( ; lastOptIdx < argc; ++lastOptIdx ) {
		// first expand arg to single or list of characer keys
		string opt = argv[lastOptIdx];
		vector<string> keys = toKeys(opt);
		if( keys.empty() ) break;
		for( size_t expKey = 0; expKey < keys.size(); ++expKey ) {
			string key = keys[expKey];
			int optIndex = optionIndex(key);
			if( optIndex < 0 ) {
				m_errMsg = "Invalid option key '"+key+"'";
				if( keys.size() > 1 ) m_errMsg += " (of '"+opt+"')";
				return m_errMsg;
			}
			// check if value(s) have already been set and act according to options
			if( !m_optvals[optIndex].empty() ) {
				if( multiKey == 'E' ) {
					return m_errMsg = "Option key '"+ opt + "' is specified more than once.";
				} else if( multiKey == 'L' ) {
					m_optvals[optIndex].clear();	// overwrite existing value
					if( warnLevel == 'W' ) {
						cerr << "Warning: Option key '"+ opt + "' is specified more than once - previous value ignored." << endl;
					}
				}
			}
			const string &optval = m_optlist[optIndex];
			size_t j = optval.find_last_of(m_sep)+1;
			if( optval[j] == terms[0] ) {
				// only one flag value is ever returned
				m_optvals[optIndex] = "T"+sep;
			} else {
				while( j < optval.size() ) {
					size_t e = terms.find(optval[j++]);
					if( !e || e == string::npos ) break;
					if( ++lastOptIdx >= argc ) {
						return m_errMsg = "No option value available for option '" + opt + "'";
					}
					const string &val = argv[lastOptIdx];
					// assume missing value error if this value is an expected key
					if( optionIndex( toKey(val) ) >= 0 ) {
						return m_errMsg = "Option key '"+opt+"' value is missing or conflicts with key '"+val+"'";
					}
					if( e == 1 ) {
						try { stringToInteger(val); }
						catch( runtime_error &e) {
							return m_errMsg = e.what();
						}
					} else if( e == 2 ) {
						try { stringToNumber(val); }
						catch( runtime_error &e) {
							return m_errMsg = e.what();
						}
					} else if( e == 3 ) {
						if( warnLevel == 'W' && val.size() > 1 && val[0] == '-' && val != "--" ) {
							cerr << "Warning: Option key '"+opt+"' value '"+val+"' maybe invalid." << endl;
						}
					} else {
						break;
					}
					m_optvals[optIndex] += val + sep;
				}
			}
		}
	}
	// finally just validate and collect remaining non-option args
	int numArgs = argc - lastOptIdx;
	if( numArgs < minArgs ) {
		m_errMsg = "Number of non-option arguments (" + integerToString(numArgs)
			+ ") is less than minimum number required (" + integerToString(minArgs)+")";
	} else if( numArgs > maxArgs && maxArgs >= minArgs ) {
		m_errMsg = "Number of non-option arguments (" + integerToString(numArgs)
			+ ") is more than maximum number allowed (" + integerToString(maxArgs)+")";
	} else {
		while( lastOptIdx < argc ) {
			m_argvals.push_back( argv[lastOptIdx++] );
		}
	}
	return m_errMsg;
}

int OptParser::optionIndex( const string &key ) const
{
	if( key.empty() ) return -1;
	string chk = m_sep + key + m_sep;
	for( size_t optIndex = 0; optIndex < m_optlist.size(); ++optIndex ) {
		if( m_optlist[optIndex].find(chk) != string::npos ) {
			return (int)optIndex;
		}
	}
	return -1;
}

string OptParser::toKey( const string &val ) const
{
	// return Y for --Y (0+ chars) or X -X (1 non-digit char), else ""
	size_t len = val.size();
	if( len > 1 && val[0] == '-' ) {
		if( val[1] == '-' ) {
			if( len > 2 ) return val.substr(2);
		} else if( len == 2 ) {
			if( val[1] < '0' || val[1] > '9' ) return val.substr(1);
		}
	}
	return "";
}

vector<string> OptParser::toKeys( const string &val ) const
{
	vector<string> keys;
	size_t len = val.size();
	if( len <= 2 || val[1] == '-' ) {
		string key = toKey(val);
		if( !key.empty() ) keys.push_back(key);
	} else if( val[0] == '-' && val.find_first_of("0123456789") == string::npos ) {
		for( size_t i = 1; i < len; i++ ) {
			keys.push_back( val.substr(i,1) );
		}
	}
	return keys;
}

string OptParser::getOptValue( const string &key, const string &defval, size_t idx ) const
{
	if( !m_errMsg.empty() ) throw runtime_error(m_errMsg);
	int optIndex = optionIndex(key);
	if( optIndex < 0 ) throw runtime_error("getOptValue(key,...): Invalid option key '"+key+"'");
	const string &vallist = m_optvals[optIndex];
	// search for idx value in list
	size_t s = 0, e = 0;
	for( size_t i = 0; i <= idx; ++i ) {
		if( (e = vallist.find(m_sep,s)) == string::npos ) {
			return defval;
		}
		if( i < idx ) s = e+1;
	}
	return vallist.substr(s,e-s);
}

double OptParser::getOptNumber( const string &key, double defval, size_t idx ) const
{
	return stringToNumber( getOptValue( key, numberToString(defval), idx ) );
}

long OptParser::getOptInteger( const string &key, long defval, size_t idx ) const
{
	return stringToInteger( getOptValue( key, integerToString(defval), idx ) );
}

bool OptParser::getOptBoolean( const string &key, bool defval ) const
{
	string test = getOptValue( key, "" );
	return (test == "T" ? !defval : defval);
}

vector<string> OptParser::getArgs() const
{
	return m_argvals;
}

} // bbcUtils
