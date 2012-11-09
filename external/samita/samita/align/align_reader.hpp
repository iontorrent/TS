/*
 *  Created on: 12-21-2009
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 87775 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-04-28 10:46:53 -0700 (Thu, 28 Apr 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef ALIGN_READER_HPP_
#define ALIGN_READER_HPP_

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <iostream>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <queue>

#include <boost/cstdint.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/shared_ptr.hpp>

#include <samita/common/types.hpp>
#include <samita/sam/bam.hpp>

/*!
        lifetechnologies namespace
*/
namespace lifetechnologies {

/*!
   Class used to open and iterate over alignments in BAM files.

   \par Iterate Example
   The following example opens a bam file and iterates over all records in the file
   \code
        AlignReader sam("my_file.bam");
        AlignReader::const_iterator iter = sam.begin();
        AlignReader::const_iterator end = sam.end();
        while (iter != end)
        {
            Align const& ac = *iter;
            // do some work with ac...
            cout << ac << endl;
            ++iter;
        }
   \endcode

   \par Selection Example
   The following example opens a bam file and iterates over all records in a specified range
   \code
        AlignReader sam("my_file.bam");
        sam.select("1:5000-9000");
        AlignReader::iterator iter = sam.begin();
        AlignReader::iterator end = sam.end();
        while (iter != end)
        {
            Align const& ac = *iter;
            // do some work with ac...
            ++iter;
        }
   \endcode

   \par Filter Example
   The following example opens a bam file and iterates over all records with a user specified filter
   NOTE: The filter may be called with a different alignment record order then your iterator is called.
         So, do not do much work if any in your filter except for determining filter return value.
   \code
        class ExampleFilter
        {
            public:
                ExampleFilter(int mq) : m_mq(mq) {}
                bool operator() (Align const &a) const
                {
                    if (a.getQual() >= m_mq)
                        return true;  // return true if you want the record to be iterated
                    else
                        return false;  // return false to filter out
                }
            private:
                int m_mq;
        };

        ExampleFilter filter(25);  // filter out records with mapq < 25
        AlignReader sam("my_file.bam", filter);
        AlignReader::filter_iterator<ExampleFilter> iter(filter, sam.begin(), sam.end());
        AlignReader::filter_iterator<ExampleFilter> end(filter, sam.end(), sam.end());

        while (iter != end)
        {
            Align const& ac = *iter;
            // do some work with ac...
            ++iter;
        }
   \endcode
*/

class AlignReader
{
    private: // typedefs
        typedef std::vector< std::vector<uint32_t> > TidLUT;
        typedef std::vector<BamReader*> BamReaderArray;
        typedef std::map<std::string, BamReader*> BamReaderMap;

    public:
        AlignReader() :  m_indexesLoaded(false), m_errorState(0) {}
        AlignReader(char const* filename) :  m_indexesLoaded(false), m_errorState(0)
        {
            open(filename);
        }

        ~AlignReader()
        {
            close();
        }

        /*!
          Clear error state.
        */
        void clear()
        {
                m_errorState = 0;
        }

        /*!
          Close all open BAM files.
        */
        void close();

        /*!
          Close the specified BAM file.
          \param filename a constant character pointer
          \return true if successful; otherwise false
        */
        bool close(char const* filename);

        /*!
          Open a BAM file for iterating or reading.
          \param filename a constant character pointer for the BAM filename
          \param index_filename an optional constant character pointer.
          \return true if successful; otherwise false
          \note
          If the index_filename parameter is not specified then the default index
          file name is used [filename+.bai]
          \par Multiple BAM Files
          Multiple BAM files can be opened at the same time for iterating.
          When iterating multiple opened BAM files, the records are merged
          on-the-fly according to coordinate sort order.  Name order may be
          supported in the future but, for the current revision, only
          coordinate oder is preserved when merging.
          \par Header Merging
          When opening multiple BAM files, the headers from all files will
          be merged into a single header.
          \li Sequence dictionary - all unique comments from all headers will appear in the merged header.
          Note, if multiple files have identical \@SQ SN values but differ in another tag value then
          the headers will not be merged and open will return false.  Alignment reference ID values
          will be re-mapped during iteration to be consistent with the merged header.
          \li Read groups - All read groups from the multiple BAMs must be unique.  If not then the header
          merge will fail and open returns false.\n
          <I> Future revisions may contain read group re-mapping as a feature.</I>
          \li Program - all unique comments from all headers will appear in the merged header
          \li Comments - all unique comments from all headers will appear in the merged header
        */
        bool open(char const* filename, char const* index_filename=NULL);

        /*!
          Select alignments from the specified region string.
          \param region a chromosome and optional range of the form "chr1:1-1000"
          \return true if successful; otherwise false
                  \note
          The index will be loaded on the first call to select.  If the index file does
          not exist then it will be created on the first call to select.
        */
        bool select(char const* region);

        /*!
          Select alignments from the specified sequence interval.
          \return true if successful; otherwise false
          \note
          The index will be loaded on the first call to select.  If the index file does
          not exist then it will be created on the first call to select.
        */
        bool select(SequenceInterval const& interval);

        /*!
          Select alignments from the specified region.
          \param tid id of the chromosome
          \param start start coordinate
          \param end end coordinate
          \return true if successful; otherwise false
          \see select(char const* region)
        */
        bool select(int tid, int start, int end);

        /*!
          Builds the index for each open bam file if it does not exist already
        */
        void buildIndexes();

        /*!
          Get Number of Bam readers (for use in iterating through using getBamReader)
        */
        size_t getNumBamReaders() const
        {
            return m_bamReaderArray.size();
        }

        /*!
          Return a BamReader reference given the file ID from an Align record
        */
        BamReader const& getBamReader(int32_t id) const
        {
            assert(static_cast<BamReaderArray::size_type>(id) < m_bamReaderArray.size());
            return *m_bamReaderArray[id];
        }


        // These should not be templated on class Value. Moved out of
        // align_iterator so tha tBamIteratorStatePtrQueue may be
        // copy-constructed between iterator and const_iterator
        struct bam_iterator_state_t {
            uint32_t index;
            BamReader::const_iterator iter;
            BamReader::const_iterator end;
        };
        typedef boost::shared_ptr< bam_iterator_state_t > bam_iterator_state_ptr;

        // NB: Comparator for priority queue is ordered by reverse start position. Lowest position in genome has highest priority.
        // So, Tests for "less" are reversed below.
        struct bam_iterator_state_compare : public std::binary_function< bam_iterator_state_ptr, bam_iterator_state_ptr, bool >
        {
            bool operator()(const bam_iterator_state_ptr & s1, const bam_iterator_state_ptr & s2) const
            {
                const int32_t t1 = (*s1->iter)->getRefId();
                const int32_t t2 = (*s2->iter)->getRefId();
                if(t1 != t2)
                {
                    // Handle -1 - unmapped reads go at end, and are always less priority
                    if(t1 < 0) return true; // Use s2
                    if(t2 < 0) return false; // Use s1
                    return t1 > t2;
                }

                const int32_t a1 = (*s1->iter)->getStart();
                const int32_t a2 = (*s2->iter)->getStart();
                if (a1 == a2)
                    return (s1->iter->get() > s2->iter->get());  // just arbitrarily pick the one with higher pointer
                else
                    return a1 > a2; // NB: Ordered by reverse position
            }
        };
        typedef std::priority_queue<bam_iterator_state_ptr, std::vector<bam_iterator_state_ptr>, bam_iterator_state_compare > BamIteratorStatePtrQueue;


        template <class Value>
        class align_iterator
            : public boost::iterator_facade<align_iterator<Value>, Value, boost::forward_traversal_tag>
        {
        private:
            /// \cond DEV
            friend class boost::iterator_core_access;
            template <class> friend class align_iterator; // allows access to private m_align

            struct enabler {};
            /// \endcond

        public:
            typedef typename align_iterator::value_type value_type;

            align_iterator() : m_align(), m_tidLUT(), m_statePtrQueue(), m_state() {}

            explicit align_iterator(const AlignReader * parent)
                : m_align(new value_type), m_tidLUT(parent->m_tidLUT), m_statePtrQueue(), m_state()
            {
                // create the iterators
                uint16_t index = 0;
                for (BamReaderArray::const_iterator braIter=parent->m_bamReaderArray.begin(); braIter!=parent->m_bamReaderArray.end(); ++braIter, index++)
                {
                    BamReader *bam = *braIter;
                    bam_iterator_state_ptr state(new bam_iterator_state_t);
                    state->index = index;
                    state->iter = bam->begin();
                    state->end = bam->end();
                    if (state->iter != state->end)
                        m_statePtrQueue.push(state);
                }
                increment();
            }

            template <class OtherValue>
            align_iterator(align_iterator<OtherValue> const& other, typename boost::enable_if<boost::is_convertible<OtherValue*,Value*>, enabler>::type = enabler())
                :  m_align(other.m_align), m_tidLUT(other.m_tidLUT), m_statePtrQueue(other.m_statePtrQueue), m_state(other.m_state)
            {}

         private:
            template <class OtherValue>
            bool equal(align_iterator<OtherValue> const& other) const
            {
                return (m_align == other.m_align);
            }

            void increment()
            {
                if(m_state.get())
                {
                    // Saved m_state from previous pass
                    ++m_state->iter;
                    if (m_state->iter == m_state->end)
                    {
                        m_state.reset();
                    }
                    else
                    {
                        m_statePtrQueue.push(m_state); // re-insert onto queue
                    }
                }
                if (!m_statePtrQueue.empty())
                {
                    m_state = m_statePtrQueue.top();
                    m_statePtrQueue.pop();

                    assert(m_state.get());
                    assert(m_state->iter != m_state->end);
                    //assert(m_align.get());
                    // Copy the value from BamReader iterator into this m_align.
                    // Calls Align operator=
                    // Copy of one align object into new align object
                    // bam1_t is re-used
                    //(*m_align) = **(m_state->iter);

                    //boost::shared_ptr<value_type>  align = *(m_state->iter);
                    m_align = boost::const_pointer_cast<value_type>(*(m_state->iter));
                    //m_align = *(m_state->iter);

                    m_align->setFileId(m_state->index);

                    if (!m_tidLUT.empty())
                    {
                        // re-map the ref id
                        std::vector<uint32_t> const& tbl = m_tidLUT[m_state->index];
                        if (m_align->isMapped())
                            m_align->setRefId(tbl[m_align->getRefId()-1]);
                        // Mate can be mapped if this read isn't (and usually is)
                        if (m_align->shouldHaveMate() && m_align->isMateMapped())
                            m_align->setMateRefId(tbl[m_align->getMateRefId()-1]);

                    }
                }
                else
                {
                    // we are at the end
                    m_align.reset();
                }
            }

            Value& dereference() const
            {
                assert(m_align.get()); // Not null
                return *(m_align.get());
            }
            
            boost::shared_ptr<value_type> m_align; // Non-const - internal state re-used
            TidLUT m_tidLUT;
            BamIteratorStatePtrQueue m_statePtrQueue;
            bam_iterator_state_ptr m_state;
        };

        typedef align_iterator<Align> iterator;
        typedef align_iterator<Align const> const_iterator;

        template<class Predicate>
        struct filter_iterator : public boost::filter_iterator<Predicate, AlignReader::iterator >
        {
            filter_iterator(Predicate &predicate, AlignReader::iterator const& begin, AlignReader::iterator const& end) :
                boost::filter_iterator<Predicate,  AlignReader::iterator >(predicate, begin, end)
            {}
        }; // struct filter_iterator

        /*!
          Begin iterator
          \return iterator to the first alignment record
        */
        iterator begin()
        {
            if ((m_bamReaderMap.size() > 0) && (m_errorState == 0))
                return iterator(this);
            return iterator();
        }
        /*!
          End iterator
          \return iterator to the end of the alignment records
        */
        iterator end()
        {
            return iterator();
        }

        BamHeader & getHeader() {return m_header;}

        //bool good() { return m_errorState == 0; }
        //bool operator!() { return (! good()); }

    private:  // methods
        // build the reference id lookup table.
        //   the resulting lookup table is a mapping from
        //     bam object (aka file) => previous reference id (aka tid) => new reference id
        void buildTidLUT();

        // clear and build the header from all open bam files
        void buildHeader();

        // merge the header from the specified bam file with the header member variable
        //     returns true if the header could be merged without conflict, otherwise returns false
        bool mergeHeader(BamReader const* bam);

        // loads the indexes for each open bam file
        void loadIndexes();
    private:  // members
        // NOTE: copy constructor not allowed - want compile error if used by self
        AlignReader(AlignReader const&);

        BamReaderMap m_bamReaderMap;
        BamReaderArray m_bamReaderArray;
        bool m_indexesLoaded;
        int m_errorState;
        BamHeader m_header;

        TidLUT m_tidLUT;
};


} //namespace lifetechnologies

#endif //ALIGN_READER_HPP_
