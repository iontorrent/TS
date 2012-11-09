/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4 -*-   vi:set expandtab ts=4 sw=4:
 *
 *  Created on: 10-25-2010
 *      Author: Jonathan Manning
 *
 *  Latest revision:  $Revision: 78917 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date$
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

/*
#include <log4cxx/logger.h>
#include <log4cxx/log4cxx.h>

static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.samita.pileup");
*/

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cassert>

#include <boost/tr1/functional.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "samita/pileup/pileup_builder.hpp"
#include "samita/pileup/pileup.hpp"

namespace lifetechnologies
{
    using namespace std;
    using namespace boost;

    struct FastqTransform: public std::unary_function<int, char> {
        // Combine operations to correct 255 to 0 and transform to character
        // 041 (octal) == 33 (decimal) == '!' (ascii)
        static const int bang = 041;
        char operator() (const int& x) const {
            // Transform -1 to zero
            //if (x < 0 || x == 255) return bang; // -1 (aka 255 unsigned) converted to 0
            //if (x > 93) return (93 + bang); // maximum representable in ascii
            if(x > 93) return 126;
            return (x + bang);
        }
    };

    /* Class PileupAlign */

    void PileupAlign::getSeq( std::vector<char> &seq ) const
    {
        // FIXME - use reference sequence for deletions and to print "." or "," instead of bases
        std::string src(align->getSeq());
        char base = src[qpos];
        if(align->getStrand() == REVERSE)
        {
            base = tolower(base); // Lowercase reverse strand, consistent with samtools pileup convention
        }

        if(is_head)
        {
            seq.push_back('^');
            seq.push_back(FastqTransform()(align->getQual())); // Mapping quality, not SM, not base quality.
            seq.push_back(base);
        }
        else if(indel != 0)
        {
            // Indels prefixed with sign and number, eg. +nACTG or -nACTG
            char c[6];
            int n = snprintf(c, 6, "%+d", indel); // convert integer to number
            seq.insert(seq.end(), c, c+n); // And append to sequence

            if(is_del)
            {
                // Deletion
                for(int i = -indel; i > 0; i--)
                {
                    if(align->getStrand() == REVERSE)
                        seq.push_back('n'); // Lowercase reverse strand
                    else
                        seq.push_back('N');
                }
                // FIXME - deletions need reference sequence!
                //std::vector<char>::iterator b = align->getSeq().begin();
                //seq.insert(seq.end(), b + qpos, b + qpos + indel);
            }
            else
            {
                // Insertion - show inserted bases
                seq.insert(seq.end(), src.begin() + qpos, src.begin() + qpos + indel);
            }
        }
        else if(is_tail)
        {
            seq.push_back(base);
            seq.push_back('$');
        }
        else
        {
            seq.push_back(base);
        }
        return;
    }

    void PileupAlign::getQual(std::vector<char> & qual) const
    {
        QualityValueArray src(align->getBaseQvs());
        if(!indel)
        {
            qual.push_back(src[qpos]);
        }
        // ?? No handling of insertions/deletions?
        return;
    }

    /* Class Pileup */

    // Sequence and Quality Pileup Strings
    std::vector<char> Pileup::getSeqStack() const
    {
        std::vector<char> seq;
        const_iterator pai(m_alignments.begin()), end(m_alignments.end());
        while(pai != end) {
            (*pai)->getSeq(seq);
            pai++;
        }
        return seq;
    }

    QualityValueArray Pileup::getQualStack() const
    {
        QualityValueArray qual;
        const_iterator pai(m_alignments.begin()), end(m_alignments.end());
        while(pai != end) {
            // FIXME - watch out for indels and tail
            qual.push_back( ((*pai)->align->getBaseQvs())[(*pai)->qpos] );
            pai++;
        }
        return qual;
    }


    /** Generate Pileup Line
     *  
     * Pileup Line:
     * chromosome, 
     * 1-based coordinate,
     * reference base,
     * the number of reads covering the site,
     * read bases and base qualities. 

     * Extended consensus pileup:
     * chromosome,
     * 1-based coordinate,
     * reference base,
     * consensus base,
     * consensus quality,
     * SNP quality and maximum mapping quality,
     * the number of reads covering the site,
     * read bases and base qualities. 
     */
    std::string Pileup::getPileupStr(BamHeader &header)
    {
        std::ostringstream ss;

        int tid = this->chr();
        int32_t pos = this->pos();
        std::vector<char> seq = getSeqStack();
        std::string seqstr(seq.begin(), seq.end());
        QualityValueArray qv = getQualStack();

        // FIXME - if we had SN, this could be operator<<
        // Pileup Like output
        ss << header.getSequence(tid).SN << "\t" << pos << "\t"  << "N" << "\t" << getPileupCoverage(*this) << "\t";
        copy(seq.begin(), seq.end(), std::ostream_iterator<char>(ss));
        ss << "\t";
        transform(qv.begin(), qv.end(), std::ostream_iterator<char>(ss), FastqTransform());
        // ss << qvsToAscii(qv);
        return ss.str();
    }

    int32_t Pileup::countStartPoints() const
    {
        const_iterator pai(m_alignments.begin()), end(m_alignments.end());
        int32_t starts = 0;
        while(pai != end) {
            if( (*pai++)->is_head ) starts++;
            pai++;
        }
        return starts;
    }

    int32_t Pileup::countEndPoints() const
    {
        const_iterator pai(m_alignments.begin()), end(m_alignments.end());
        int32_t tails = 0;
        while(pai != end) {
            if( (*pai++)->is_tail ) tails++;
            pai++;
        }
        return tails;
    }

    Pileup::~Pileup()
    {
        iterator pai(m_alignments.begin()), end(m_alignments.end());
        while(pai != end) {
            // WARNING: Only valid for PileupAlignType == PileupAlign *,
            // but not value for shared_ptr<PileupAlign>;
            delete *pai;
            pai++;
        }
    }

    /* Template specialization with custom constructor for AlignReader */
    /* NB: Must be defined in cpp, not header */
    template <>
    PileupBuilder<AlignReader::iterator>::PileupBuilder(AlignReader & ar)
        : m_buf(), m_begin(ar.begin()), m_end(ar.end()), m_current(ar.begin()) {;}
    

    // coverage / throughput
    int32_t getPileupCoverage(Pileup const & p)
    {
        return std::accumulate(
                boost::make_transform_iterator(p.begin(),
                        boost::mem_fn(&PileupAlign::coverage)),
                boost::make_transform_iterator(p.end(),
                        boost::mem_fn(&PileupAlign::coverage)),
                static_cast<int32_t> (0));
    }

    int32_t getPileupThroughput(Pileup const & p)
    {
        return std::accumulate(
                boost::make_transform_iterator(p.begin(),
                        boost::mem_fn(&PileupAlign::throughput)),
                boost::make_transform_iterator(p.end(),
                        boost::mem_fn(&PileupAlign::throughput)),
                static_cast<int32_t> (0));
    }


} // end namespace lifetech

