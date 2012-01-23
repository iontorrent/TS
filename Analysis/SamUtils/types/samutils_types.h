/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef SAMUTILS_TYPES_H
#define SAMUTILS_TYPES_H

/*
 *  samutils_types.h
 *  SamUtils
 *
 *  Created by Michael Lyons on 12/21/10.
 *  
 *
 */

#include <vector>
#include <cctype>
#include <string>
#include <tr1/memory>
//#include "BAMRead.h"
//intenral *.h's
#include "Cigar.h"

namespace samutils_types {
	

	typedef int64_t						coord_t;
	typedef uint8_t						qual_t;
	typedef std::vector<std::string>	strvec;
	typedef std::vector<coord_t>		lenvec;
	typedef const char*					str_ptr;
	
	//cigar specifics
	typedef std::pair<uint32_t, uint32_t> CigarElement;
	typedef	std::vector<CigarElement>	CigarElementArray;
	
	
	struct bam_cleanup : public std::unary_function<bam1_t *&, void>
	{
		void operator()(bam1_t *& b) const {
			bam_destroy1(b);
		}
	};
	typedef std::tr1::shared_ptr<bam1_t> bam_ptr;
	
		
	class IUPAC{
	public:
		
		static std::vector<char> get_base(char IUPAC) {
			char nuke = toupper(IUPAC);
			std::vector<char> ret_vec;
			switch (nuke) {
				case 'U':
					ret_vec.push_back('T');
					break;
				case 'R':
					ret_vec.push_back('A');
					ret_vec.push_back('G');
					break;
				case 'Y':
					ret_vec.push_back('C');
					ret_vec.push_back('T');
					break;
				case 'S':
					ret_vec.push_back('G');
					ret_vec.push_back('C');
					break;
				case 'W':
					ret_vec.push_back('A');
					ret_vec.push_back('T');
					break;
				case 'K':
					ret_vec.push_back('G');
					ret_vec.push_back('T');
					break;
				case 'M':
					ret_vec.push_back('A');
					ret_vec.push_back('C');
					break;
				case 'B':
					ret_vec.push_back('C');
					ret_vec.push_back('G');
					ret_vec.push_back('T');
					break;
				case 'D':
					ret_vec.push_back('A');
					ret_vec.push_back('G');
					ret_vec.push_back('T');
					break;
				case 'H':
					ret_vec.push_back('C');
					ret_vec.push_back('A');
					ret_vec.push_back('T');
					break;
				case 'V':
					ret_vec.push_back('A');
					ret_vec.push_back('C');
					ret_vec.push_back('G');
					break;
				default:
					ret_vec.push_back(nuke); //default just return what we were given
					break;
			}
			return ret_vec;
		}
	private:
		IUPAC();
		
	};
}

#endif //SAMUTILS_TYPES_H

