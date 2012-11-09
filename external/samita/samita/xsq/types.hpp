/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * types.hpp
 *
 *  Created on: Sep 21, 2010
 *      Author: mullermw
 */

#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <boost/algorithm/string/regex.hpp>
#include "samita/common/types.hpp"
#include "exception.hpp"


namespace lifetechnologies {

/*!
 * The 4 Bases of DNA.  If you don't know these you shouldn't be using this library.
 */
//enum DnaBase {
//	DNA_A = 'a',
//	DNA_C = 'c',
//	DNA_G = 'g',
//	DNA_T = 't'
//};

const char NUCLEOTIDE_CODE[] = {  'A', 'C', 'G', 'T' };

enum Instrument {
	INSTRUMENT_SOLID_PI,
	INSTRUMENT_SOLID_HQ,
	INSTRUMENT_UNKNOWN
};
enum Application {
	APPLICATION_RESEQ,
	APPLICATION_TARGETTED_RESEQ,
	APPLICATION_RNASEQ,
	APPLICATION_SMALLRNA,
	APPLICATION_UNKNOWN
};

/*!
  Type of a read
*/
enum XsqReadType
{
    XSQ_READ_TYPE_F3,       /*!< F3 */
    XSQ_READ_TYPE_R3,       /*!< R3 */
    XSQ_READ_TYPE_F5,       /*!< F5 P2 or F5 BC*/
};

inline std::ostream &operator<<(std::ostream &stream, XsqReadType r) {
	switch (r) {
		case XSQ_READ_TYPE_F3:
			stream << "F3";
			break;
		case XSQ_READ_TYPE_R3:
			stream << "R3";
			break;
		case XSQ_READ_TYPE_F5:
			stream << "F5";
			break;
		default:
			stream << "?????";
	}
	return stream;
}

inline XsqReadType toReadType(std::string const& str) {
	if (str.find("F3") != std::string::npos ) return XSQ_READ_TYPE_F3;
	if (str.find("R3") != std::string::npos ) return XSQ_READ_TYPE_R3;
	if (str.find("F5") != std::string::npos ) return XSQ_READ_TYPE_F5;
	std::stringstream stringstream;
	stringstream << str << " is not a valid ReadType.";
	throw illegal_argument_exception(stringstream.str());
}

inline std::vector<XsqReadType> toReadTypes(std::vector<std::string> const strings) {
	std::vector<XsqReadType> vec;
	std::transform(strings.begin(), strings.end(), std::back_inserter(vec), toReadType);
	return vec;
}

inline std::string to_string(XsqReadType const& type) {
	switch (type) {
	case XSQ_READ_TYPE_F3 :
		return "F3";
	case XSQ_READ_TYPE_F5 :
		return "F5";
	case XSQ_READ_TYPE_R3 :
		return "R3";
	}
	throw illegal_argument_exception("unknown ReadType.");
}


/*!
 * Represents a means of encoding bases into a color sequence.
 *
 * Examples: Each Row of
 *
 * Traditional 2 base encoding:
 *  Offset=0, Width=2, Stride=0
 *
 *           0----5----1----1----2----2----3----3----4----4----
 *                     0    5    0    5    0    5    0    5
 * Round 1:  10001100011000110001100011000110001100011000110001
 * Round 2:  11000110001100011000110001100011000110001100011000
 * Round 3:  01100011000110001100011000110001100011000110001100
 * Round 4:  00110001100011000110001100011000110001100011000110
 * Round 5:  00011000110001100011000110001100011000110001100011
 *
 *  Color  Bases
 *  -----  -----
 *    0		0,1
 *    1		1,2
 *    2     2,3
 *    etc.
 *
 * 1 Primer Round of 4 base encoding (hypothetical):
 *
 *  Offset=1, Width=4, Stride=5
 *  0----5----1----1----2----2----3----3----4----4----
 *            0    5    0    5    0    5    0    5
 *  01111011110111101111011110111101111011110111101111
 *  |___||___||___||___||___||___||___||___||___||___|
 *    1    2    3    4    5    6    7    8    9   10
 *
 *  Color  Bases
 *  -----  -----
 *    0		1,2,3,4
 *    1		6,7,8,9
 *    2     11,12,13,14
 *    etc.
 *
 */
class ColorEncoding {
	std::string probeset;
    int8_t offset;
    uint8_t stride;
public:
    ColorEncoding() : probeset("1"), offset(0), stride(1) {}
    ColorEncoding(const std::string & probeset, const int8_t & offset, const uint8_t & stride)
    :probeset(probeset), offset(offset), stride(stride) {}
    bool operator==(ColorEncoding const& other) const {
    	return this->probeset == other.probeset &&
    			this->offset == other.offset &&
    			this->stride == other.stride;
    }
    bool operator<(ColorEncoding const& other) const {
    	if (this->probeset.size() != other.probeset.size()) return this->probeset.size() < other.probeset.size();
    	if (this->probeset != other.probeset) return this->probeset.compare(other.probeset) < 0;
    	if (this->offset != other.offset) return this->offset < other.offset;
    	if (this->stride != other.stride) return this->stride < other.stride;
    	return false;
    }
    int8_t getOffset() const { return offset; }
    std::string getProbeset() const { return probeset; }
    uint8_t getStride() const { return stride; }
};

const ColorEncoding BASE_ENCODING("1", 0, 1);
const ColorEncoding SOLID_ENCODING("11", -1, 1);
const ColorEncoding _5500_PLUS4_ENCODING("1303", 1, 5);

inline std::ostream &operator<<(std::ostream &stream, ColorEncoding const& r) {
	return stream << "ColorEncoding(probeset=" << r.getProbeset() << ", offset=" << static_cast<int>(r.getOffset()) << ", stride=" << static_cast<int>(r.getStride()) << ")";
}

//std::ostream &operator<<(std::ostream& stream, std::vector<size_t> const& vec) {
//	stream << "[";
//	for (std::vector<size_t>::const_iterator it = vec.begin(); it != vec.end(); ++it) {
//		if (it != vec.begin()) stream << ",";
//		stream << *it;
//	}
//	stream << "]";
//	return stream;
//}
//
//std::ostream &operator<<(std::ostream& stream, std::vector<uint32_t> const& vec) {
//	stream << "[";
//	for (std::vector<uint32_t>::const_iterator it = vec.begin(); it != vec.end(); ++it) {
//		if (it != vec.begin()) stream << ",";
//		stream << *it;
//	}
//	stream << "]";
//	return stream;
//}

//From http://www.ietf.org/rfc/rfc2396.txt
const boost::regex URL_PATTERN("^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\\?([^#]*))?(#(.*))?");

class URL {
	typedef std::map<std::string, std::vector<std::string> > queryMap;
	std::vector<std::string> matches;
	queryMap query;
	std::vector<std::string> parameterNames;
public:
	URL(std::string const& string) {
		boost::cmatch matches;
		if (!boost::regex_match(string.c_str(), matches, URL_PATTERN))
			throw parse_exception( "Not a valid URL: " + string);

		for (size_t i = 0; i < matches.size(); i++)
			this->matches.push_back(matches[i].str());
		if (this->getQuery().empty()) return;
		std::vector<std::string> keyValueEntries;
		boost::algorithm::split_regex(keyValueEntries, this->getQuery(), boost::regex("&"));
		for (std::vector<std::string>::const_iterator keyValue=keyValueEntries.begin(); keyValue != keyValueEntries.end(); ++keyValue) {
			if (keyValue->empty()) continue;
			std::vector<std::string> vec;
			boost::algorithm::split_regex(vec, *keyValue, boost::regex("="));
			this->query[vec[0]].push_back( vec.size() < 2 ? "" : vec[1] );
		}
		for (queryMap::const_iterator it=this->query.begin(); it != this->query.end(); ++it)
			this->parameterNames.push_back(it->first);
	}
	URL(URL const& rhs) : matches(rhs.matches), query(rhs.query), parameterNames(rhs.parameterNames) {}
	bool operator==(URL const& rhs) const {
		return this->str() == rhs.str();
	}
	bool operator!=(URL const& rhs) const {
		return this->str() != rhs.str();
	}
	std::string str() const {
		return matches[0];
	}
	std::string getScheme() const {
		return matches[2];
	}
	std::string getAuthority() const {
		return matches[4];
	}
	std::string getPath() const {
		return matches[5];
	}
	std::string getQuery() const {
		return matches[7];
	}
	std::string getFragment() const {
		return matches[9];
	}
	std::vector<std::string> getParameterNames() const {
		return this->parameterNames;
	}
	std::vector<std::string> getParameterValues(std::string const& name) {
		return this->query[name];
	}
	std::string getParameter(std::string const& name) {
		std::vector<std::string> vec = getParameterValues(name);
		return vec.empty() ? "" : vec[0];
	}
};

} // namespace lifetechnologies

#endif /* TYPES_HPP_ */
