/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4 -*-   vi:set expandtab ts=4 sw=4: */
/* Copyright 2010 Life Technologies Corporation. All rights reserved. */

/** Samita Pileup module
 *  Pileup Iterator for BAM files
 */

#ifndef PILEUP_CPP_H
#define PILEUP_CPP_H

#include <vector>
#include <string>
#include <functional>
#include <stdint.h>

#include <boost/shared_ptr.hpp>

#include <samita/common/types.hpp>
#include <samita/common/interval.hpp>
#include <samita/align/align_reader.hpp>

namespace lifetechnologies {

    typedef boost::shared_ptr<Align> AlignPtr;

    /** \class PileupAlign
     *  \brief One read in the pileup stack.
     * An alignment with an individual position and context called out.
     * The same Align record is shared between multiple PileupAligns
     */
    class PileupAlign
    {
    public:
        const AlignPtr align; // Alignment Record
        const int32_t qpos; // Query Position 0-based
        int indel; // size of indel; pos for ins, neg for del
        bool is_del, is_head, is_tail, is_origin; // Meaningful for pileup annotations

        PileupAlign(const AlignPtr &a, const int32_t qoffset) :
            align(a),
            qpos(qoffset),
            indel(0), // Need to compute this?
            is_del(false), // (indel < 0),
            is_head(qpos == 0),
            is_tail(qpos == static_cast<int32_t>(align->getCigar().getReadLength() - 1)),
            is_origin(false)
        {
            // Strand sensitive operations
            if((align->getStrand() == FORWARD)
              && (qpos == align->getOriginOffset()))
            {
                is_origin = true;
              // if is_head, is_start.
            } else if ((align->getStrand() == REVERSE)
                    && (qpos == static_cast<int32_t>(align->getCigar().getReadLength()) + align->getOriginOffset()) )
            {
               is_origin = true;
            }
        };

        const int setIndel(const int _indel)
        {
            indel = _indel;
            is_del = (indel < 0);
            return indel;
        }

        void getSeq(std::vector<char> & src) const;
        void getQual(std::vector<char> & src) const;

        int32_t coverage() const { return ((indel == 0) ? 1 : 0); }
        int32_t throughput() const { return (is_del ? 0 : indel + (is_tail ? 0 : 1)); }
    };

    // One position in the genome - an array of all Pileup Reads.
    class Pileup {
    public:
        typedef PileupAlign * PileupAlignPtr;
        typedef std::vector< PileupAlignPtr > PileupArray;
        typedef std::unary_function< PileupAlignPtr , bool> PileupPred;

        typedef PileupArray::iterator iterator;
        typedef PileupArray::const_iterator const_iterator;

    private:
        PileupArray m_alignments;
        const int m_tid; // Chromosome ID
        const int32_t m_position;
        // max_tid, max_pos
    public:

        // New Empty Pileup
        Pileup(int const tid, int32_t const position): m_alignments(), m_tid(tid), m_position(position) {;}

        // NULL pileup??!
        Pileup(): m_alignments(), m_tid(), m_position() {;}

        ~Pileup();

        // Proxy internal array iterator
        iterator begin() { return m_alignments.begin(); }
        iterator end() { return m_alignments.end(); }
        const_iterator begin() const { return m_alignments.begin(); }
        const_iterator end() const { return m_alignments.end(); }

        void push_back(PileupAlignPtr p) { return m_alignments.push_back(p); }

        int chr() const { return m_tid; }
        int32_t pos() const { return m_position; }

        const int32_t count() const { return m_alignments.size(); }

        // Sequence and Quality Pileup Strings
        std::string getPileupStr(BamHeader & header);
        std::vector< char > getSeqStack() const;
        QualityValueArray getQualStack() const;
        int32_t countStartPoints() const;
        int32_t countEndPoints() const;

    };

    // coverage / throughput
    int32_t getPileupCoverage(Pileup const & p);
    int32_t getPileupThroughput(Pileup const & p);

    inline std::ostream & operator<<(std::ostream & out, Pileup const & plp)
    {
        int     tid = plp.chr();
        int32_t pos = plp.pos();
        std::vector<char>   seq = plp.getSeqStack();
        std::string seqstr(seq.begin(), seq.end());
        QualityValueArray qv = plp.getQualStack();

        /* Pileup Like output */
        out <<
                tid << //header.getSequence(tid).SN <<
                "\t" <<
                pos <<
                "\t" <<
                "N" << // reference base
                "\t" <<
                getPileupCoverage(plp) <<
                "\t" <<
                seqstr <<
                "\t" <<
                qvsToAscii(qv);
        // No endl?
        return out;
    }

    // Debug output - each alignment in pileup with annotations
    inline std::ostream & operator<<(std::ostream & out, PileupAlign const & pa)
    {
        out << *pa.align << std::endl;
        out << "qpos: " << pa.qpos;
        std::ostringstream oss;
        oss << std::boolalpha;
        oss << " Origin: "<< pa.is_origin;
        oss << " Head: "<< pa.is_head;
        oss << " Tail: "<< pa.is_tail;
        out << oss.str();
        return out;
    }

} // end namespace lifetechnologies

#endif // def PILEUP_CPP_H
