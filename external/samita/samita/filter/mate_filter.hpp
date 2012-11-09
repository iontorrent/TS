/*
 *  Created on: 8-24-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 77374 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-02-17 13:35:12 -0800 (Thu, 17 Feb 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef MATE_FILTER_HPP_
#define MATE_FILTER_HPP_

#include <boost/unordered_map.hpp>
#include <samita/filter/filter.hpp>
#include <samita/sam/bam.hpp>

/*!
    lifetechnologies namespace
*/
namespace lifetechnologies {

/*!
    Class representing two mate alignments
*/
class AlignMates
{
public:
    Align first;
    Align second;
};

/*!
    Predicate class to find mates
*/
class MateFilter
{
    public:
        MateFilter(AlignMates *mates) : m_matesPtr(mates) {assert(m_matesPtr);}
        ~MateFilter()
        {
            // delete the cached bam records
            for (BamRecordPtrHash::iterator iter=m_bamRecordCache.begin(); iter!=m_bamRecordCache.end(); ++iter)
                bam_destroy1(iter->second);
        }
        bool operator() (Align const &a)
        {
            //std::cout << "m_bamRecordCache.size() : " << m_bamRecordCache.size() << std::endl;
            bam1_t const* b = a.getBamPtr();
            if ((b->core.flag & BAM_FUNMAP) ||  (b->core.flag & BAM_FMUNMAP))
            {
                return false;
            }
            if (!(b->core.flag & BAM_FPAIRED))
            {
                return false;
            }
            std::string name = bam1_qname(b);
            BamRecordPtrHash::iterator iter = m_bamRecordCache.find(name);
            if (iter != m_bamRecordCache.end())
            {
                bam1_t *m = iter->second;
                if (m->core.pos != b->core.mpos || b->core.pos != m->core.mpos)
                    return false;
                if (m->core.flag & BAM_FREAD1)
                {
                    // FIXME - may be broken in shared_ptr implementation
                    m_matesPtr->first.setDataPtr(m);
                    m_matesPtr->second.setDataPtr(bam_dup1(b));
                    m_bamRecordCache.erase(iter);
                    return true;
                }
                else if (m->core.flag & BAM_FREAD2)
                {
                    // FIXME - may be broken in shared_ptr implementation
                    m_matesPtr->first.setDataPtr(bam_dup1(b));
                    m_matesPtr->second.setDataPtr(m);
                    m_bamRecordCache.erase(iter);
                    return true;
                }
            }
            else
            {
                m_bamRecordCache[name] = bam_dup1(b);
                return false;
            }
            return false;
        }

    private:
        typedef  boost::unordered_map<std::string, bam1_t *> BamRecordPtrHash;

        AlignMates *m_matesPtr;
        BamRecordPtrHash m_bamRecordCache;
};


} //namespace lifetechnologies

#endif //MATE_FILTER_HPP_
