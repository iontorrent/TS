/*
 *  Created on: 04-20-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 92486 $
 *  Last changed by:  $Author: zhangzh $
 *  Last change date: $Date: 2011-08-26 15:50:52 -0700 (Fri, 26 Aug 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef CIGAR_HPP_
#define CIGAR_HPP_

#include <stdexcept>
#include <string>

#include <sam.h>
#include <bam.h>
#include <lifetech/string/util.hpp>
#include <samita/common/types.hpp>
#include <samita/exception/exception.hpp>

namespace lifetechnologies
{

/*!
  typedef representing an cigar string element
 */
typedef std::pair<uint32_t, uint32_t> CigarElement;
/*!
  typedef representing an array of cigar string elements
 */
typedef std::vector< CigarElement > CigarElementArray;

/*!
  Class representing a cigar string
 */
class Cigar
{
    public:
        //Cigar() {}
        explicit Cigar(bam1_t const* bt)
        {
            uint32_t *cigar_ptr = bam1_cigar(bt);
            m_elements.reserve(bt->core.n_cigar);
            for (size_t i = 0; i < bt->core.n_cigar; i++)
            {
                uint32_t cigar_i = *cigar_ptr;
                addElement(cigar_i >> BAM_CIGAR_SHIFT, cigar_i & BAM_CIGAR_MASK);
                cigar_ptr++;
            }
        }

        void addElement(int length, int operation) {m_elements.push_back(CigarElement(length, operation));}
        void addElement(CigarElement const& element) {m_elements.push_back(element);}

        size_t getNumElement() const {return m_elements.size();}
        bool isEmpty() const {return (m_elements.size()==0);}
        CigarElementArray const& getElements() const {return m_elements;}
        CigarElement const& getElement(int i) const {return m_elements[i];}

        bool isGapped() const
        {
            for (CigarElementArray::const_iterator iter=m_elements.begin(); iter!=m_elements.end(); ++iter)
            {
                CigarElement const& element = *iter;
                int op = element.second;
                if ((op == BAM_CDEL) || (op == BAM_CINS))
                    return true;
            }
            return false;
        }

        bool isSpliced() const
        {
            for (CigarElementArray::const_iterator iter=m_elements.begin(); iter!=m_elements.end(); ++iter)
            {
                CigarElement const& element = *iter;
                int op = element.second;
                if (op == BAM_CREF_SKIP)
                    return true;
            }
            return false;
        }

        size_t getReadLength() const
        {
            int length = 0;
            for (CigarElementArray::const_iterator iter=m_elements.begin(); iter!=m_elements.end(); ++iter)
            {
                CigarElement const& element = *iter;
                int op = element.second;
                if ((op == BAM_CMATCH) || (op == BAM_CINS) || (op == BAM_CSOFT_CLIP))
                    length += element.first;
            }
            return length;
        }
        size_t getReferenceLength() const
        {
            int length = 0;
            for (CigarElementArray::const_iterator iter=m_elements.begin(); iter!=m_elements.end(); ++iter)
            {
                CigarElement const& element = *iter;
                int op = element.second;
                if ((op == BAM_CMATCH) || (op == BAM_CDEL) || (op == BAM_CREF_SKIP) || (op == BAM_CPAD))
                    length += element.first;
            }
            return length;
        }

        /*! \brief Compute effect of soft clipping on coordinates
         * (soft clip doesn't affect start/end, but does appear in query string.)
         * Valid operations are [[H] [S]] M (Hard must be outside soft, which must be outside match)
         * @return bases between start and position 0 of query string or end and end of query string.
         */
        int getOffsetForward() const
        {
            int offset = 0;
                for (CigarElementArray::const_iterator iter=m_elements.begin(); iter!=m_elements.end(); ++iter)
                {
                    if (iter->second == BAM_CMATCH)
                        break;
                    else if (iter->second == BAM_CHARD_CLIP)
                        continue;
                    offset += iter->first;
                }
                return offset;
        }
        int getOffsetReverse() const
        {
            int offset = 0;
            for (CigarElementArray::const_reverse_iterator iter=m_elements.rbegin(); iter < m_elements.rend(); ++iter)
            {
                if (iter->second == BAM_CMATCH)
                    break;
                else if (iter->second == BAM_CHARD_CLIP)
                    continue;
                offset += iter->first;
            }
            return offset;
        }

        /*! \brief Compute effect of soft and hard clipping on coordinates
         * When offset is applied to either reference position or query position
         * the origin of sequencing can be identified.
         * @return offset of origin of sequencing from start or end
         */
        int getOriginOffsetForward() const
        {
            int originOffset = 0;
            for (CigarElementArray::const_iterator iter=m_elements.begin(); iter!=m_elements.end(); ++iter)
            {
                if (iter->second == BAM_CHARD_CLIP || iter->second == BAM_CSOFT_CLIP)
                {
                    originOffset -= iter->first;
                }
                else
                {
		    return originOffset;
                }
            }
            return originOffset;
        }
        /*! \brief Compute effect of soft and hard clipping on coordinates
         * When offset is applied to either reference position or query position
         * the origin of sequencing can be identified.
         * @return offset of origin of sequencing from start or end
         */
        int getOriginOffsetReverse() const
        {
            int originOffset = 0;
            //originOffset = getReadLength(); // Convert to get offset from read start
            // Walk CIGAR from end, add soft and hard clip
            for (CigarElementArray::const_reverse_iterator iter=m_elements.rbegin(); iter < m_elements.rend(); ++iter)
            {
                if (iter->second == BAM_CHARD_CLIP || iter->second == BAM_CSOFT_CLIP)
                {
                    // Contribute read length
                    originOffset += iter->first;
                }
                else
                {
		    return originOffset;
                }
            }
            return originOffset;
        }

        /*!
          \exception invalid_cigar_operation
         */
        std::string toString() const
        {
            std::ostringstream strm;
            for (CigarElementArray::const_iterator iter=m_elements.begin(); iter!=m_elements.end(); ++iter)
            {
                CigarElement const& element = *iter;
                strm << element.first;
                switch (element.second)
                {
                    case BAM_CMATCH:
                        strm << "M";
                        break;
                    case BAM_CINS:
                        strm << "I";
                        break;
                    case BAM_CDEL:
                        strm << "D";
                        break;
                    case BAM_CREF_SKIP:
                        strm << "N";
                        break;
                    case BAM_CSOFT_CLIP:
                        strm << "S";
                        break;
                    case BAM_CPAD:
                        strm << "P";
                        break;
                    case BAM_CHARD_CLIP:
                        strm << "H";
                        break;
                    default:
                        throw lifetechnologies::invalid_cigar_operation(element.second);
                }
            }
            return strm.str();
        }
    private:
        CigarElementArray m_elements;
};


} //namespace lifetechnologies

#endif //CIGAR_HPP_
