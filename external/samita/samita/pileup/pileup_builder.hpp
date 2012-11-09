/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4 -*-   vi:set expandtab ts=4 sw=4: */
/* Copyright 2010 Life Technologies Corporation. All rights reserved. */

/** Samita Pileup module
 *  Pileup Iterator for BAM files
 */

#ifndef PILEUP_BUILDER_CPP_H
#define PILEUP_BUILDER_CPP_H

#include <boost/tr1/memory.hpp> // shared_ptr

#include <boost/circular_buffer.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <stdint.h>
#include <functional>

#include "samita/pileup/pileup.hpp"
#include "samita/align/align_reader.hpp"

namespace lifetechnologies {

    // Shortcut to a Pileup shared Pointer.
    // Use this instead of Pileup * or explicit calls to shared_ptr.
    typedef boost::shared_ptr<Pileup> PileupPtr;


    // DEBUG printer
    inline std::ostream & operator<<(std::ostream & out, boost::circular_buffer< PileupPtr > const & buf)
    {
        out << "Circular Buffer of size: " << buf.size() << " and capacity " << buf.capacity() << std::endl;
        for(size_t i = 0; i < buf.size(); ++i)
        {
            out << "buf[" << i <<"] = " << buf[i]->pos() << std::endl;
            for (Pileup::const_iterator plp = buf[i]->begin(); plp != buf[i]->end(); ++plp)
            {
                PileupAlign * pa = *plp;
                out << *pa << std::endl;
                //AlignPtr a = (*plp)->align;
                //out << "\t" << (*a) << std::endl;
            }
        }
        return out;
    }

    /** \class PileupBuilder
     *  \brief Generate pileup given a Iterator
     *   NOTE: Currently debating making this class the iterator...
     */
    template <typename Iterator = AlignReader::iterator>
    class PileupBuilder
    {
    protected:
        typedef boost::circular_buffer< PileupPtr > pileupbuf;
    private:
        pileupbuf m_buf;
        int m_tid; // Chromosome
        Iterator m_begin, m_end, m_current;
        //bool m_startsOnly, m_originOfSequencing;
    public:
        PileupBuilder(AlignReader & ar);
        PileupBuilder(const Iterator &start, const Iterator &end = Iterator())
            : m_buf(), m_tid(-1), m_begin(start), m_end(end), m_current(start) /*, m_startsOnly(false), m_originOfSequencing(false) */ {;}
        virtual ~PileupBuilder() {;}

        //void setStartsOnly(bool startsOnly) { m_startsOnly = startsOnly; }
        //void setOriginOfSequencing(bool originOfSequencing) { m_originOfSequencing = originOfSequencing; }

        uint32_t position() { return m_buf[0]->pos(); };
        PileupPtr pileup() { return m_buf[0]; }

        /** \class pileup_iterator
         *  Nested iterator type, returned by begin/end
         */
        class pileup_iterator
            : public boost::iterator_facade<
            pileup_iterator, // Self - CRTP
            PileupBuilder, // Base Class
            std::input_iterator_tag, // Iterator category - Single Pass
            PileupPtr > // Value Type
        {
        private:
            PileupBuilder * m_parent;
            PileupPtr m_pileup;
        public:
            pileup_iterator() : m_parent(NULL), m_pileup() {};
            pileup_iterator(PileupBuilder * parent) : m_parent(parent), m_pileup()
            {
                if(parent)
                {
                    increment();
                }
            };
        private:
            friend class boost::iterator_core_access;

            const PileupPtr dereference() const { return m_pileup; };

            void increment() {
                do {
                    // Remove first item
                    if(!m_parent->empty())
                    {
                        m_parent->pop_front();
                    }
                    // Ensure new first item is fully populated
                    // If update fails, and no more reads are in buffer, stop iteration
                    if((! m_parent->update()) && m_parent->empty()) {
                        m_pileup.reset(); // Clear shared_ptr - iterator is now == end()
                        break;
                    }
                    m_pileup = m_parent->pileup();
                } while (m_pileup->count() == 0);
                // FIXME Don't return empty pileups...
            }

            bool equal(pileup_iterator const & other) const
            {
                return (m_pileup == other.m_pileup);
            }
        }; // End pileup_iterator nested class

        typedef pileup_iterator iterator;
        iterator begin() { return iterator(this); }
        iterator end() { return iterator(NULL); }

    private:
        PileupBuilder(): m_buf(0), m_begin(NULL), m_end(NULL), m_current(NULL) {};

        // Copy constructor and assignment operator -- private - prevents copying
        PileupBuilder(const PileupBuilder & pb);//
        // : m_buf(pb.m_buf), m_begin(pb.m_begin), m_end(pb.m_end), m_current(pb.m_current) {};
        PileupBuilder & operator=(PileupBuilder const & other);

        /** Ensure first position in pileup is fully populated.
         *  Rest of reads are inserted into internal buffer for future use
         */
        virtual bool update()
        {
            // Read alignments until we get to a new position
            while(m_current != m_end) {
                // try to insert into buffer
                // assert Iterator::value_type == Align
                if(!(*m_current).isMapped() )
                {
                    m_current++; // skip unaligned read
                    continue;
                }
                if( this->insert(*m_current) ) {
                    m_current++;
                } else {
                    // Returns false when buffer isn't ready for next record yet
                    // m_current is kept as-is for next pass
                    break;
                }
            }
            // Return false at EOF
            return (m_current != m_end);
        };

        void pop_front() { m_buf.pop_front(); };

        bool equal( PileupBuilder const & other ) const
        {
            return (m_buf != other.m_buf);
        }

        virtual bool insert(Align &input_align)
        {
            int32_t pos = input_align.getStart();
            if( ! this->checkAndIncrementChromosome(input_align.getRefId()) ) return false;
            // If position is past current leftmost position, defer until later
            if( !empty() && m_buf[0]->pos() != pos) return false;

            // OK, we want it. Now create copy. Align copy constructor does deep copy.
            AlignPtr align(new Align(input_align));

            const Cigar cigar = align->getCigar();
            const CigarElementArray ce = cigar.getElements();

            // Adjust capacity to be enough to hold reference footprint of entire read we are inserting.
            this->prepareBuffer(pos, cigar.getReferenceLength());

            // ************************************
            // Walk cigar string
            int poffset = m_buf.empty() ? 0 : pos - m_buf[0]->pos(); // Position offset - index in m_buf
            assert(poffset == 0); // No lookahead yet, will add later...

            int qoffset = 0; // Query offset - index in alignment

            CigarElementArray::const_iterator c = ce.begin();
            int cigar_op = 0; // 4-bits
            uint32_t cigar_len = 0; // max 28-bits
            bool is_head = true;
            if(c != ce.end()) {
                cigar_len = (*c).first;
                cigar_op = (*c).second;
            }

            while(c != ce.end())
            {
                // Create a new PileupAlign element (with shared align) for each position
                // Note - delete is called from within Pileup's destructor,
                // so 'new PileupAlign' must be consumed with a push() call (or delete);
                Pileup::PileupAlignPtr p;
                switch (cigar_op) {
                case BAM_CMATCH:
                    p = new PileupAlign(align, qoffset);
                    if(is_head) { p->is_head = true; is_head = false; }
                    assert(static_cast<pileupbuf::size_type>(poffset) < m_buf.size());
                    push(poffset,p);
                    poffset++;
                    qoffset++;
                    cigar_len--;
                    // Matches consumed one base at a time.
                    // One PileupAlign generated per matching location.
                    break;

                case BAM_CHARD_CLIP:
                    // Ignore.
                    // qoffset - unchanged
                    // poffset - unchanged
                    cigar_len = 0;
                    break;

                case BAM_CSOFT_CLIP:
                    // Does not appear in pileup, but does increment offset.
                    // Do it all at once, instead of bit by bit like a match
                    // FIXME check leading soft clip
                    //poffset += cigar_len; // Position not advanced by soft clip
                    qoffset += cigar_len;
                    cigar_len = 0;
                    break;

                case BAM_CDEL:
                    // Every position of a deletion is marked in the reference pileup
                    // FIXME - not compatible with samtools pileup?
                    p = new PileupAlign(align, qoffset);
                    p->setIndel(-cigar_len); // Use total size of deletion, not remaining.
                    // Account for deletion at every deleted position
                    assert(static_cast<pileupbuf::size_type>(poffset) < m_buf.size());
                    push(poffset,p);
                    poffset++;
                    // A deletion does not consume alignment sequence (qoffset)
                    cigar_len--; // Entire deletion accounted for?
                    break;

                case BAM_CINS:
                    // Insertions land at prior location, but as separate event
                    // Really should modify prior sequence?
                    p = new PileupAlign(align, qoffset);
                    if(is_head) { p->is_head = true; is_head = false; }
                    p->setIndel(cigar_len);
                    if(poffset > 0) {
                        push(poffset-1,p);
                    } else {
                        // This is the best we can do for a leading indel
                        // FIXME - inconsistent placement of insertions!!!
                        push(poffset,p); // poffset == 0
                    }
                    // poffset is not incremented
                    // The entire indel appears at one position in reference, but consumes multiple qpos.
                    // The entire insertion appears at one position in reference, but consumes multiple qpos.
                    qoffset+=cigar_len;
                    cigar_len = 0; // Entire insertion accounted for
                    break;

                case BAM_CPAD:
                    // An insert that doesn't consume alignment sequence
                    p = new PileupAlign(align, qoffset);
                    p->setIndel((*c).first); // Use total size of insertion
                    assert(static_cast<pileupbuf::size_type>(poffset) < m_buf.size());
                    push(poffset, p);
                    poffset += cigar_len;
                    cigar_len = 0;
                    break;

                case BAM_CREF_SKIP:
                default:
                    std::cerr << "Unhandled cigar_op " << cigar_op << std::endl;
                    poffset += cigar_len;
                    cigar_len = 0;
                };

                // Advance to next cigar
                if(cigar_len == 0) {
                    c++;
                    if(c != ce.end()) {
                        cigar_len = (*c).first;
                        cigar_op = (*c).second;
                    } else {
                        //p->is_tail = true;
                        cigar_op = 254;
                    }
                }
            }
            return true;
        }

    public:
        // DEBUG ONLY
        inline pileupbuf const & getBuffer() const { return m_buf; }
    protected:

        // Encapsulate and expose utility functions for working with private buf
        inline bool empty() const { return m_buf.empty(); }

        /// Returns false when current buffer contains different chromosome, otherwise increments chromosome
        bool checkAndIncrementChromosome(const int32_t tid) {
            if(empty())
            {
                // Not just check - If buffer is empty
                m_tid = tid;
                return true;
            }
            return (m_tid == tid);
        }
        void push(const int32_t offset, Pileup::PileupAlignPtr const & p)
        {
            assert(offset >= 0);
            assert(offset < static_cast<int32_t>(m_buf.size()));
            m_buf[offset]->push_back(p);
        }

        int32_t leftmost() const
        {
            assert(!empty());
            return m_buf[0]->pos();
        }
        int32_t offset(int32_t pos) const 
        {
            assert(pos >= leftmost());
            return pos - leftmost();
        }

        //int32_t getTid() const { return m_tid; } 

        /* Prepare Circular buffer with at least needed positions starting at pos
         * Circular buffer will overwrite existing content,
         * so we need to ensure capacity to store data in advance.
         * Buffer must have contiguous positions
         */
        size_t prepareBuffer(const int32_t pos, const pileupbuf::size_type len) {
            // Set bounds
            int32_t start = (m_buf.empty()) ? pos : m_buf[0]->pos();
            int32_t minpos = std::min(start, pos);
            int32_t maxpos = std::max(start + m_buf.size(), pos + len);
            assert(maxpos > minpos);
            uint32_t needed = static_cast<uint32_t>(maxpos - minpos);
            assert(needed > 0);
            // Never shrink capacity here.
            // pop() shrinks buffer (size and capacity) during processing
            if(m_buf.capacity() < needed)
            {
                m_buf.set_capacity(needed);
            }

            // Insert new, empty pileups with correct positions.

            // Insert at beginning up to start.
            int32_t cpos;
            if(! m_buf.empty() )
            {
                // Start is m_buf[0] and already exists
                cpos = start - 1;
                // Must iterate backwards from start-1 to minpos
                while(cpos >= minpos)
                {
                    // WARNING: Inserting beyond capacity will cause circular buffer to drop sequence.
                    assert(m_buf.size() < m_buf.capacity());

                    PileupPtr tmp(new Pileup(m_tid, cpos--));
                    m_buf.push_front(tmp);
                }
                assert(minpos = m_buf[0]->pos()); // Starts where expected
            }

            // Insert at end for any remaining capacity
            // NB - if buf was empty, cpos == start.
            cpos = pos + m_buf.size(); // skip ahead
            while(m_buf.size() < needed)
            {
                // WARNING: Inserting beyond capacity will cause circular buffer to drop sequence.
                assert(m_buf.size() < m_buf.capacity());

                PileupPtr tmp(new Pileup(m_tid, cpos++));
                m_buf.push_back(tmp);
            }

            if( (pos < m_buf[0]->pos()) || (minpos != m_buf[0]->pos()) )
            {
                // DEBUG
                std::cerr << "Minpos: " << minpos << " Leftmost: " << leftmost() << std::endl;
                std::cerr << m_buf << std::endl;
            }
            assert(minpos == m_buf[0]->pos()); // Starts where expected
            assert(pos >= m_buf[0]->pos());
            assert(cpos == (pos + static_cast<int32_t>(needed)));
            assert(m_buf.size() == needed);
            //std::cerr << "DEBUG: " << std::endl << m_buf;
            return m_buf.size();
        };


    };

} // namespace lifetechnologies
#endif // def PILEUP_BUILDER_CPP_H
