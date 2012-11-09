/*
 *  Created on: 04-20-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49984 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:54:43 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef QUALITY_VALUE_HPP_
#define QUALITY_VALUE_HPP_

#include <stdexcept>
#include <string>

#include <lifetech/string/util.hpp>
#include <samita/common/types.hpp>
#include <samita/exception/exception.hpp>

namespace lifetechnologies
{

/*!
  typedef representing an array of PHRED-based quality values
 */
typedef std::vector<uint8_t> QualityValueArray;

/*!
  Convert FASTQ quality values to PHRED quality values.
  \param FASTQ quality values string
  \return a QualityValueArray
 */
inline QualityValueArray asciiToQvs(std::string const& ascii)
{
    QualityValueArray qvs;

    qvs.reserve(ascii.length());
    for (std::string::const_iterator iter = ascii.begin(); iter !=ascii.end(); ++iter)
        qvs.push_back(*iter - 33);
    return qvs;
}

/*!
  Convert PHRED quality values to FASTQ quality values.
  \param PHRED quality values array
  \return a string
 */
inline std::string qvsToAscii(QualityValueArray const& qvs)
{
    std::stringstream qvstrm;

    for (QualityValueArray::const_iterator iter = qvs.begin(); iter!=qvs.end(); ++iter)
    {
        uint8_t qv = *iter;
        qvstrm << (char)(qv+33);
    }
    return qvstrm.str();
}

} //namespace lifetechnologies

#endif //QUALITY_VALUE_HPP_
