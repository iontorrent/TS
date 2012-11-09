/*
 *  Created on: 12-21-2009
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 89253 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-05-12 13:42:34 -0700 (Thu, 12 May 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef BAM_HPP_
#define BAM_HPP_

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <cstring> // memset

#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>
#include <samita/common/types.hpp>
#include <samita/common/interval.hpp>
#include <samita/exception/exception.hpp>
#include <lifetech/string/util.hpp>
//#include <log4cxx/logger.h>

namespace lifetechnologies
{
/*!
 Sort order of a BAM file
 */
typedef enum
{
    BAM_SO_UNSORTED, /*!< Unsorted */
    BAM_SO_QUERYNAME, /*!< Sorted by query name */
    BAM_SO_COORDINATE
/*!< Sorted by coordinate */
} BAM_SO;

/*!
 Group order of a BAM file
 */
typedef enum
{
    BAM_GO_NONE, /*!< Unsorted */
    BAM_GO_QUERY, /*!< Sorted by query */
    BAM_GO_REFERENCE
/*!< Sorted by reference */
} BAM_GO;

typedef std::pair<uint64_t, uint64_t> BamIndexChunk;
typedef std::vector<BamIndexChunk> BamIndexChunkList;
typedef boost::shared_ptr<bam_header_t> BamHeaderPtr;

//static log4cxx::LoggerPtr bam_log = log4cxx::Logger::getLogger("lifetechnologies.samita.BamHeader");

/*!
 Class for a header tag
 */
class BamHeaderTag
{
public:
    BamHeaderTag(std::string n, std::string v) :
        name(n), value(v)
    {
    }
    BamHeaderTag(std::string n, uint32_t v) :
        name(n)
    {
        std::ostringstream sstrm;
        sstrm << v;
        value = sstrm.str();
    }
    std::string name;
    std::string value;
};

/*!
 Class BamHeader
 */
class BamHeader
{
public:
    BamHeader()
    : m_rg(), m_sq(), m_pg(), m_version(),
      m_sortOrder(BAM_SO_UNSORTED), m_groupOrder(BAM_GO_NONE),
      m_comments(), m_rgid2rg(),
      m_bam_header()
    {
    }

    ~BamHeader();

    /*!
     \exception read_group_not_found
     */
    RG const& getReadGroup(std::string const &id) const
    {
        if (id.empty() && m_rg.size() == 1)
        {
            // no RG tag and only one group in header
            //   so just assume it comes from the single group
            return m_rg[0];
        }
        // try to find it
        std::map<std::string, RG>::const_iterator iter = m_rgid2rg.find(id);
        if (iter != m_rgid2rg.end())
            return iter->second;
        else
            throw read_group_not_found(id);
    }

    bool getSequenceRegion(char const* region, int& tid, int& beg, int& end) const
    {
        std::string seq;
        tid = -1;
        beg = 0;
        end = -1;

        SequenceInterval::parse(region, seq, beg, end);
        SQ const& sq = getSequence(seq, &tid);
        if(tid < 0) return false;
        if (beg < 0) beg = 0;
        if (end < 0 || end > sq.LN) end = sq.LN;
        return true;
    }
    int getSequenceLength( char const* region ) const
    {
        std::string seq;
        int tid = -1;
        int beg;
        int end;

        SequenceInterval::parse(region, seq, beg, end);
        SQ const& sq = getSequence(seq, &tid);
        if( tid < 0 ) return -1;	// sequence name not found?
        return sq.LN;
    }
    SequenceInterval getInterval(std::string const& name, int32_t *id = NULL) const
    {
        SQ const& sq = getSequence(name, id);
        return sq.getInterval();
    }
    SequenceIntervalArray getIntervals() const
    {
        SequenceIntervalArray intervals;

        for (std::vector<SQ>::const_iterator iter = m_sq.begin(); iter != m_sq.end(); ++iter)
        {
            SQ const& sq = *iter;
            intervals.push_back(sq.getInterval());
        }
        return intervals;
    }

    /*!
     \exception read_group_not_found
     */
    RG const& getReadGroup(int32_t id) const
    {
        size_t correctedId = id - 1;
        if (m_rg.size() && correctedId < m_rg.size())
            return m_rg[correctedId];
        else
            throw read_group_not_found(id);
    }
    /*!
     \exception reference_sequence_not_found
     */
    SQ const& getSequence(int32_t id) const
    {
        size_t correctedId = id - 1;
        if (m_sq.size() && correctedId < m_sq.size())
            return m_sq[correctedId];
        else
            throw reference_sequence_not_found(id);
    }

    /*!
     \exception reference_sequence_not_found
     */
    SQ const& getSequence(std::string const& name, int32_t *id = NULL) const
    {
        int i = 0;
        for (std::vector<SQ>::const_iterator iter = m_sq.begin(); iter != m_sq.end(); ++iter, i++)
        {
            SQ const& sq = *iter;
            if (sq.SN == name)
            {
                if (id)
                    *id = i;
                return sq;
            }
        }
        throw reference_sequence_not_found(name);
    }

    void addBamStats(const char* filename);

    /*!
     \exception invalid_input_record
     \exception boost::bad_lexical_cast
     */
    void initialize(bam_header_t *header);

    void clear();

    bool empty() const
    {
        return (m_rg.size() == 0);
    }

    std::string const& getVersion() const
    {
        return m_version;
    }
    void setVersion(std::string const& version)
    {
        m_version = version;
    }
    BAM_SO getSortOrder() const
    {
        return m_sortOrder;
    }
    void setSortOrder(BAM_SO so)
    {
        m_sortOrder = so;
    }
    BAM_GO getGroupOrder() const
    {
        return m_groupOrder;
    }
    void setGroupOrder(BAM_GO go)
    {
        m_groupOrder = go;
    }

    std::vector<RG> const& getReadGroups() const
    {
        return m_rg;
    }
    void addReadGroup(RG const& rg)
    {
        // FIXME - push_back and assignment both copy their inputs
        // The vector and map hold *different* copies of the RG!
        m_rg.push_back(rg);
        m_rgid2rg[rg.ID] = rg;
    }
    std::vector<SQ> const& getSequenceDictionary() const
    {
        return m_sq;
    }
    void addSequence(SQ const& sq)
    {
        m_sq.push_back(sq);
    }
    std::vector<PG> const& getPrograms() const
    {
        return m_pg;
    }
    void addProgram(PG const& pg)
    {
        m_pg.push_back(pg);
    }
    std::vector<std::string> const& getComments() const
    {
        return m_comments;
    }
    void addComment(std::string const& co)
    {
        m_comments.push_back(co);
    }

    void setRGStats(std::string const& id, RGStats const&stats)
    {
        // try to find it
        std::map<std::string, RG>::iterator iter = m_rgid2rg.find(id);
        if (iter != m_rgid2rg.end())
        {
            RG &rg = iter->second;
            rg.Stats = stats;
        }
        else
            throw read_group_not_found(id);
    }

    bam_header_t * getRawHeader() const { return m_bam_header.get(); }
    //BamHeaderPtr getRawHeaderPtr() const { return m_bam_header; }

    bool isValidSequenceInterval( const SQ& sq, int beg, int end ) {
    	bool isValid = true;
    	std::stringstream chromInfo;
    	chromInfo << "chromName: " << sq.SN << "; chromLength: " << sq.LN << "; start: " << beg << "; end: " << end;
    	if( beg < 1 ) {
    		std::string msg = "interval start cannot be less than 1. " + chromInfo.str();
			std::cerr << "ERROR: " << msg << std::endl;
//			LOG4CXX_ERROR( bam_log, msg );
			isValid = false;
    	}
    	if( beg > sq.LN ) {
    		std::string msg = "interval start cannot be greater than chromLength. " + chromInfo.str();
			std::cerr << "ERROR: " << msg << std::endl;
//			LOG4CXX_ERROR( bam_log, msg );
			isValid = false;
    	}
    	if( beg > end ) {
    		std::string msg = "interval start cannot be greater than interval end. " + chromInfo.str();
			std::cerr << "ERROR: " << msg << std::endl;
//			LOG4CXX_ERROR( bam_log, msg );
			isValid = false;
    	}
    	if( end > sq.LN ) {
    		std::string msg = "interval end cannot be greater than chromLength. " + chromInfo.str();
			std::cerr << "ERROR: " << msg << std::endl;
//			LOG4CXX_ERROR( bam_log, msg );
			isValid = false;
    	}
    	return isValid;
    }

    bool isValidSequenceInterval( int tid, int beg, int end ) {
    	SQ sq;
    	try {
    		sq = getSequence( tid ); }	// fetch SQ for this tid
    	catch (int e) {
			std::stringstream msg;
			msg << "cannot find tid in BamHeader. tid: " << tid << "; chromStart: " << beg << "; chromEnd: " << end;
			std::cerr << "ERROR: " << msg.str() << std::endl;
//			LOG4CXX_ERROR( bam_log, msg.str() );
    		return false; }
    	return isValidSequenceInterval( sq, beg, end );
    }

    bool isValidSequenceInterval( const std::string seq, int beg, int end ) {
    	SQ sq;
    	try {
    		sq = getSequence( seq ); }	// fetch SQ for this name
    	catch (int e) {
			std::stringstream msg;
			msg << "invalid chromName. chromName: " << seq << "; chromStart: " << beg << "; chromEnd: " << end;
			std::cerr << "ERROR: " << msg.str() << std::endl;
//			LOG4CXX_ERROR( bam_log, msg.str() );
    		return false; }
    	if( beg == -1 )		// begin not defined
    		beg = 1;		// default to start of chrom
    	if( end == -1 )		// end not defined
    		end = sq.LN;	// default to end of chrom
    	return isValidSequenceInterval( sq, beg, end );
    }

    bool isValidSequenceInterval( const char* region ) {
    	std::string seq;
    	int beg, end;
    	SequenceInterval::parse( region, seq, beg, end );
    	return isValidSequenceInterval( seq, beg, end );
    }

    bool isValidSequenceInterval( SequenceInterval &interval ) {
    	return isValidSequenceInterval( interval.getName(), interval.getStart(), interval.getEnd() );
    }

private:
    std::vector<RG> m_rg;
    std::vector<SQ> m_sq;
    std::vector<PG> m_pg;
    std::string m_version;
    BAM_SO m_sortOrder;
    BAM_GO m_groupOrder;
    std::vector<std::string> m_comments;

    std::map<std::string, RG> m_rgid2rg; // dictionary to make ReadGroup lookups faster
    BamHeaderPtr m_bam_header; // Original header
};

/*!
 Class representing a BAM file
 */
class BamReader
{
public:
    BamReader();
    BamReader(const char* filename, const char* index_filename = NULL);

    virtual ~BamReader()
    {
        close();
    }

    bool open(const char* filename, const char* index_filename = NULL);
    void close();

    bool isOpen() const
    {
        return (m_file != NULL);
    }
    bool isIndexLoaded() const
    {
        return (m_index != NULL);
    }

    std::string getFilename() const {return m_filename;}

    /*!
     \exception index_creation_exception
     */
    void buildIndex();
    bool loadIndex();

    bool select(SequenceInterval const& interval);
    bool select(const char* region);
    bool select(int tid, int beg, int end);

    void clear()
    {
        m_errorState = 0;
        m_selectTid = -1;
        m_selectBegin = -1;
        m_selectEnd = -1;
    }

    BamHeader const& getHeader() const
    {
        return m_header;
    }

    bam_header_t* getRawHeader() const;

    /*!
     Input iterator class used to iterate over records in the BAM file.
     */
    template <class Value>
    class bam_iterator
        : public boost::iterator_facade<bam_iterator<Value>, Value, boost::forward_traversal_tag>
    {
    private:
        /// \cond DEV
        friend class boost::iterator_core_access;
        template <class> friend class bam_iterator;
        struct enabler {};
        /// \endcond

    public:
        typedef typename bam_iterator::value_type value_type; // import from parent
        bam_iterator()
            : m_bam1Ptr(), m_align(),
              m_chunks(NULL), m_offset(0), m_chunkIter(),
              m_selectTid(-1), m_selectBegin(0), m_selectEnd(0),
              m_selectMode(), m_bamFile()
        {}
        
        explicit bam_iterator(const BamReader * parent)
            : m_bam1Ptr(bam_init1(), BamCleanup()),
              m_align(new typename value_type::element_type(m_bam1Ptr)),
              m_chunks(&parent->m_chunks),
              m_offset(0),
              m_chunkIter(m_chunks->begin()),
              m_selectTid(parent->m_selectTid),
              m_selectBegin(parent->m_selectBegin),
              m_selectEnd(parent->m_selectEnd),
              m_selectMode(parent->m_selectMode),
              m_bamFile(parent->m_file->x.bam)
        {
            if(m_chunkIter != m_chunks->end())
                increment();
        }

        template <class OtherValue>
        bam_iterator(bam_iterator<OtherValue> const& other, typename boost::enable_if<boost::is_convertible<OtherValue,Value>, enabler>::type = enabler())
            : m_bam1Ptr(other.m_bam1Ptr),
              m_align(boost::const_pointer_cast<typename OtherValue::element_type>(other.m_align)),
              m_chunks(other.m_chunks),
              m_offset(other.m_offset),
              m_chunkIter(other.m_chunkIter),
              m_selectTid(other.m_selectTid),
              m_selectBegin(other.m_selectBegin),
              m_selectEnd(other.m_selectEnd),
              m_selectMode(other.m_selectMode),
              m_bamFile(other.m_bamFile)
        {
        }

     private:
        template <class OtherValue>
        bool equal(bam_iterator<OtherValue> const& other) const
        {
            return (m_bam1Ptr == other.m_bam1Ptr && m_align == other.m_align);
        }

        void increment()
        {
            // Failed previous increment, or have no chunks to query
            if (m_chunkIter == m_chunks->end()) return;

            bool atEnd = false;
            while (!atEnd)
            {
                if (m_offset == 0)
                {
                        m_offset = m_chunkIter->first;
                        //std::cout << "seeking 0 : " << m_offset << " (" << m_chunkIter->first << ", " << m_chunkIter->second << ")" << std::endl;
                        if(bam_seek(m_bamFile, m_offset, SEEK_SET) != 0)
                        {
                            atEnd = true; // Not at end, failed!
                        }
                }
                else if (m_offset >= m_chunkIter->second)
                {
                    ++m_chunkIter;
                    if (m_chunkIter != m_chunks->end())
                    {
                        // goto next chunk
                        if (m_offset != m_chunkIter->first) // check for adjacency before seeking
                        {
                            m_offset = m_chunkIter->first;
                            //std::cout << "seeking 1 : " << m_offset << " (" << m_chunkIter->first << ", " << m_chunkIter->second << ")" << std::endl;
                            if( bam_seek(m_bamFile, m_offset, SEEK_SET) != 0 )
                            {
                                atEnd = true; // Not at end, failed!
                            }
                        }
                    }
                    else
                        atEnd = true; // no more chunks
                }

                if (m_chunkIter == m_chunks->end())
                {
                    atEnd = true;
                    break;
                }

                if (!atEnd)
                {
                    // Zero out the current record
                    if (m_bam1Ptr->data)
                        free(m_bam1Ptr->data);
                    memset(m_bam1Ptr.get(), 0, sizeof(bam1_t));

                    // get the next alignment (if possible)
                    int bytesRead;
                    if ((bytesRead = bam_read1(m_bamFile, m_bam1Ptr.get())) > 0)
                    {
                        m_offset += bytesRead; // get the current offset for the next time 'round on the guitar

                        if (m_selectMode) // are we looking for a range?
                        {
                            // indeed we are.  Now see if we have the right ref id and are not already beyond the range
                            if ((m_bam1Ptr->core.tid != m_selectTid) || (m_bam1Ptr->core.pos >= m_selectEnd))
                                atEnd = true; // if so then no need to proceed
                            else if (!overlaps(m_selectBegin, m_selectEnd, m_bam1Ptr.get()))
                                continue; // if this record does not overlap the select range then nothing to see here...
                        }
                        break;
                    }
                    else
                    {
                        atEnd = true;
                    }
                }
            }

            if (!atEnd)
            {
                assert(m_align.get());
                assert(m_bam1Ptr.get());
                m_align->setDataPtr(m_bam1Ptr);
            }
            else
            {
                m_bam1Ptr.reset();
                m_align.reset();
            }
        }

        Value & dereference() const
        {
            assert(m_align.get());
            //return *(m_align.get());  // FIXME - last peek inside shared_ptr
            return m_align;
        }

        static bool overlaps(uint32_t beg, uint32_t end, bam1_t * bt)
        {
            uint32_t rbeg = bt->core.pos;
            uint32_t rend = bt->core.n_cigar ? bam_calend(&bt->core, bam1_cigar(bt)) : bt->core.pos + 1;
            return ((rend > beg) && (rbeg < end));
        }

        Bam1Ptr m_bam1Ptr;
        value_type m_align; // Assuming value_type isa shared_ptr to Align [const] // m_align MUST be non-const
        BamIndexChunkList const* m_chunks; // Not owned, borrowed from parent
        uint64_t m_offset;
        BamIndexChunkList::const_iterator m_chunkIter;
        int m_selectTid, m_selectBegin, m_selectEnd; // select region
        bool m_selectMode;
        bamFile m_bamFile;
    };

    typedef bam_iterator< boost::shared_ptr<Align> > iterator;
    typedef bam_iterator< const boost::shared_ptr<Align> > const_iterator; // Can't make Align const with shared_ptr?


    iterator begin() const
    {
        if (m_file && (m_errorState == 0))
            return iterator(this);
        return iterator();
    }
    iterator end() const
    {
        return iterator();
    }

    /// bam_seek returns non-zero when seek fails.
    int64_t seek(uint64_t offset)
    {
        return bam_seek(m_file->x.bam, offset, SEEK_SET);
    }
    uint64_t tell()
    {
        return bam_tell(m_file->x.bam);
    }

    bool read(bam1_t &bam)
    {
        memset(&bam, 0, sizeof(bam1_t));
        return (bam_read1(m_file->x.bam, &bam) == 0);
    }

    BamIndexChunkList const* getChunks() const
    {
        return &m_chunks;
    }

private:
    // methods
    // NOTE: copy constructor not allowed - want compile error if used by self
    BamReader(BamReader const&);

    bool loadChunkCoordinates(bam_index_t *index, int tid, int begin, int end);

private:
    // members
    std::string m_filename;
    std::string m_indexFilename;
    samfile_t *m_file;
    bam_index_t *m_index;
    bool m_selectMode;
    int m_errorState;
    int m_selectTid, m_selectBegin, m_selectEnd; // select region
    BamIndexChunkList m_chunks;
    BamHeader m_header;
};

/*!
 << operator for outputting a formatted BAM_SO to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, BAM_SO const& so)
{
    switch (so)
    {
    case BAM_SO_UNSORTED:
        out << "unsorted";
        break;
    case BAM_SO_QUERYNAME:
        out << "queryname";
        break;
    case BAM_SO_COORDINATE:
        out << "coordinate";
        break;
    }
    return out;
}

/*!
 << operator for outputting a formatted BAM_GO to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, BAM_GO const& go)
{
    switch (go)
    {
    case BAM_GO_NONE:
        out << "none";
        break;
    case BAM_GO_QUERY:
        out << "query";
        break;
    case BAM_GO_REFERENCE:
        out << "reference";
        break;
    }
    return out;
}

//TODO: implement me someday
/*!
 function for formatting a bam1_t to a string (just like samtools -view)
 */
//inline std::string toString(bam_header_t const* header, bam1_t const* b)
//{
//    char *s = bam_format1(header, b);
//    std::string str = s;
//    free(s);
//    return s;
//}

/*!
 << operator for outputting a formatted BamHeaderTag to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, BamHeaderTag const& tag)
{
    if (!tag.value.empty())
        out << tag.name << ":" << tag.value;
    return out;
}

/*!
 << operator for outputting a formatted SQ to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, SQ const& sq)
{
    out << "@SQ";
    out << "\t" << BamHeaderTag("SN", sq.SN) << "\t" << BamHeaderTag("LN", sq.LN);
    out << "\t" << BamHeaderTag("AS", sq.AS) << "\t" << BamHeaderTag("M5", sq.M5);
    out << "\t" << BamHeaderTag("UR", sq.UR) << "\t" << BamHeaderTag("SP", sq.SP);
	out << std::endl; // Newline is significant
    return out;
}

/*!
 << operator for outputting a formatted RG to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, RG const& rg)
{
    out << "@RG";
    out << "\t" << BamHeaderTag("ID", rg.ID) << "\t" << BamHeaderTag("SM", rg.SM);
    out << "\t" << BamHeaderTag("LB", rg.LB) << "\t" << BamHeaderTag("DS", rg.DS);
    out << "\t" << BamHeaderTag("PU", rg.PU) << "\t" << BamHeaderTag("PI", rg.PI);
    out << "\t" << BamHeaderTag("CN", rg.CN) << "\t" << BamHeaderTag("DT", rg.DT);
    out << "\t" << BamHeaderTag("PL", rg.PL);
	out << std::endl; // Newline is significant
    return out;
}

/*!
 << operator for outputting a formatted PG to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, PG const& pg)
{
    out << "@PG";
    out << "\t" << BamHeaderTag("ID", pg.ID) << "\t" << BamHeaderTag("VN", pg.VN);
    out << "\t" << BamHeaderTag("CL", pg.CL);
	out << std::endl; // Newline is significant
    return out;
}

/*!
 << operator for outputting a formatted Header to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, BamHeader const& hdr)
{

    // FIXME - distinguish between required fields and optional fields.
    // Suppress optional fields on output if empty.

    out << "@HD\tVN:" << hdr.getVersion() << "\tSO:" << hdr.getSortOrder() << "\tGO:" << hdr.getGroupOrder()
            << std::endl;

    std::vector<SQ> const& seqs = hdr.getSequenceDictionary();
    for (std::vector<SQ>::const_iterator iter = seqs.begin(); iter != seqs.end(); ++iter)
        out << *iter;// << std::endl;

    std::vector<RG> const& rgs = hdr.getReadGroups();
    for (std::vector<RG>::const_iterator iter = rgs.begin(); iter != rgs.end(); ++iter)
        out << *iter;// << std::endl;

    std::vector<PG> const& pgs = hdr.getPrograms();
    for (std::vector<PG>::const_iterator iter = pgs.begin(); iter != pgs.end(); ++iter)
        out << *iter;// << std::endl;

    std::vector<std::string> const& comments = hdr.getComments();
    for (std::vector<std::string>::const_iterator iter = comments.begin(); iter != comments.end(); ++iter)
    {
        // Array of strings - do not have embedded newlines
        out << *iter << std::endl;
    }
    return out;
}

} //namespace lifetechnologies

#endif //BAM_HPP_
