/*
 *  Created on: 8-19-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 98825 $
 *  Last changed by:  $Author: utirams1 $
 *  Last change date: $Date: 2011-12-19 15:33:20 -0800 (Mon, 19 Dec 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */
#include <sam.h>
#include <bam.h>
#include <log4cxx/logger.h>
#include <samita/sam/bam.hpp>
#include <samita/sam/bas.hpp>

static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.samita.sam");

#define MAX_BLOCK_OFFSET  ULLONG_MAX

// BEGIN: forward decls from "bam_index.c"
//    NOTE: it would be better for these to be exposed in a file (like bam_index.h)
//          instead of being hidden in bam_index.c.  For now declaring them here is OK
//          but, in the future, work with the samtools devel team to get them promoted
//          to a header file.
extern "C"
{

typedef struct
{
    uint64_t u, v;
} pair64_t;

struct __bam_iter_t {
        int from_first; // read from the first record; no random access
        int tid, beg, end, n_off, i, finished;
        uint64_t curr_off;
        pair64_t *off;
};

//pair64_t * get_chunk_coordinates(const bam_index_t *idx, int tid, int beg, int end, int* cnt_off);

// Forward decl from sam.c, exported properly in 0.1.12
extern bam_header_t *bam_header_dup(const bam_header_t *h0);

} // extern "C"
// END: forward decls from "bam_index.c"

// SIMILAR for extern from bam_maqcns.c, which is not linked in libbam.so
char bam_nt16_nt4_table[] = { 4, 0, 1, 4, 2, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4 };

namespace lifetechnologies
{

void BamHeader::initialize(bam_header_t *header)
{
    std::vector<std::string> lines;

    clear();
    assert(header != NULL);
    m_bam_header.reset(bam_header_dup(header), std::ptr_fun(::bam_header_destroy));

    std::string text(m_bam_header->text);
    string_util::tokenize(text, "\n", lines);

    for (std::vector<std::string>::const_iterator iter = lines.begin(); iter != lines.end(); ++iter)
    {
        std::string const& line = *iter;
        std::vector<std::string> fields;

        string_util::tokenize(line, "\t", fields);
        size_t nFields = fields.size();
        if (nFields)
        {
            std::string const& type = fields[0];

            if (type == "@HD")
            {
                for (size_t i = 1; i < nFields; i++)
                {
                    std::string const& tag = fields[i].substr(0, 2);
                    std::string value = fields[i].substr(3);

                    if (tag == "VN")
                        m_version = value;
                    else if (tag == "SO")
                    {
                        if (value == "unsorted")
                            m_sortOrder = BAM_SO_UNSORTED;
                        else if (value == "queryname")
                            m_sortOrder = BAM_SO_QUERYNAME;
                        else if (value == "coordinate")
                            m_sortOrder = BAM_SO_COORDINATE;
                        else
                            throw invalid_input_record("bam header unknown SO value", line);
                    }
                    else if (tag == "GO")
                    {
                        if (value == "none")
                            m_groupOrder = BAM_GO_NONE;
                        else if (value == "queryname")
                            m_groupOrder = BAM_GO_QUERY;
                        else if (value == "coordinate")
                            m_groupOrder = BAM_GO_REFERENCE;
                        else
                            throw invalid_input_record("bam header unknown GO value", line);
                    }
                    else
                        throw invalid_input_record("bam header unknown @HD attribute: " + tag, line);
                }
            }
            else if (type == "@SQ")
            {
                SQ sq;
                for (size_t i = 1; i < nFields; i++)
                {
                    std::string const& tag = fields[i].substr(0, 2);
                    std::string value = fields[i].substr(3);
                    if (tag == "SN")
                        sq.SN = value;
                    else if (tag == "LN")
                        sq.LN = boost::lexical_cast<int>(value);
                    else if (tag == "AS")
                        sq.AS = value;
                    else if (tag == "M5")
                        sq.M5 = value;
                    else if (tag == "MD")
                        sq.M5 = value;
                    else if (tag == "UR")
                        sq.UR = value;
                    else if (tag == "SP")
                        sq.SP = value;
                    else
                        LOG4CXX_WARN(g_log, "bam header unknown @SQ attribute: " + tag);
                        //throw invalid_input_record("bam header unknown @SQ attribute: " + tag, line);
                }
                addSequence(sq);
            }
            else if (type == "@RG")
            {
                RG rg;
                for (size_t i = 1; i < nFields; i++)
                {
                    std::string const& tag = fields[i].substr(0, 2);
                    std::string value = fields[i].substr(3);
                    if (tag == "ID")
                        rg.ID = value;
                    else if (tag == "SM")
                        rg.SM = value;
                    else if (tag == "LB")
                        rg.LB = value;
                    else if (tag == "DS")
                        rg.DS = value;
                    else if (tag == "PU")
                        rg.PU = value;
                    else if (tag == "PI")
                        rg.PI = value;
                    else if (tag == "CN")
                        rg.CN = value;
                    else if (tag == "DT")
                        rg.DT = value;
                    else if (tag == "PL")
                        rg.PL = value;
                    else if (tag == "PG")
                        rg.PG = value; // new in SAM 1.3 spec
                    else if (tag == "FO")
			rg.FO = value;
		    else if (tag == "KS")
			rg.KS = value;
		    
                        //throw invalid_input_record("bam header unknown @RG attribute: " + tag, line);
                }
                addReadGroup(rg);
            }
            else if (type == "@PG")
            {
                PG pg;
                for (size_t i = 1; i < nFields; i++)
                {
                    std::string const& tag = fields[i].substr(0, 2);
                    std::string value = fields[i].substr(3);
                    if (tag == "ID")
                        pg.ID = value;
                    else if (tag == "VN")
                        pg.VN = value;
                    else if (tag == "CL")
                        pg.CL = value;
                    else if (tag == "PP")
                        ;// untracked
                    else if (tag == "PN")
                        pg.PN = value;
                    else
                        LOG4CXX_WARN(g_log, "bam header unknown @PG attribute: " + tag);
                        //throw invalid_input_record("bam header unknown @PG attribute: " + tag, line);
                }
                addProgram(pg);
            }
            else if (type == "@CO")
            {
                //if (nFields > 1)
                //    addComment(fields[1]);
                addComment(line);
            }
        }
    }
}

BamHeader::~BamHeader() { clear(); }

void BamHeader::clear()
{
    m_rg.clear();
    m_sq.clear();
    m_pg.clear();
    m_version.clear();
    m_sortOrder = BAM_SO_UNSORTED;
    m_groupOrder = BAM_GO_NONE;
    m_comments.clear();
    //m_bam_header.reset();
}


BamReader::BamReader() :
    m_filename(""), m_indexFilename(""), m_file(NULL), m_index(NULL), m_selectMode(false), m_errorState(0)
{
}

BamReader::BamReader(const char* filename, const char* index_filename) :
    m_file(NULL), m_index(NULL), m_selectMode(false), m_errorState(0)
{
    open(filename, index_filename);
}

bool BamReader::open(const char* filename, const char* index_filename)
{
    close(); // just in case the previous open call was not paired with a close
    m_selectTid = -1;
    m_selectBegin = -1;
    m_selectEnd = -1;
    m_file = samopen(filename, "rb", 0);
    if (!m_file)
    {
        LOG4CXX_ERROR(g_log, "unable to open file " << filename);
        return false;
    }
    else if (((m_file->type & 1) != 1) || (m_file->header == 0))
    {
        LOG4CXX_ERROR(g_log, "invalid format for file " << filename);
        close();
        return false;
    }
    m_filename = filename;
    if (index_filename)
        m_indexFilename = index_filename;
    else
        m_indexFilename = m_filename;
    m_chunks.clear();
    m_chunks.push_back(std::make_pair(bam_tell(m_file->x.bam), MAX_BLOCK_OFFSET)); // by default, add the first chunk to the chunk offsets
    m_selectMode = false;
    m_header.initialize(m_file->header);
    return true;
}

void BamReader::close()
{
    if (m_index)
        bam_index_destroy(m_index);
    m_index = NULL;
    if (m_file)
        samclose(m_file);
    m_file = NULL;
    m_header.clear();
}

void BamReader::buildIndex()
{
    // build index if it does not exist
    std::string realIndexFilename = m_indexFilename + ".bai";

    std::ifstream file(realIndexFilename.c_str(), std::ios::in | std::ios::binary);
    if (file.fail())
    {
        LOG4CXX_INFO(g_log, "building index : " << m_indexFilename);
        if (bam_index_build(m_indexFilename.c_str()) != 0)
            throw index_creation_exception(m_indexFilename);
        LOG4CXX_INFO(g_log, "finished building index : " << m_indexFilename);
    }
    else
        file.close();
}
bool BamReader::loadIndex()
{
    if (!isIndexLoaded())
    {
        buildIndex();

        // load index
        LOG4CXX_INFO(g_log, "loading index : " << m_indexFilename);

        m_index = bam_index_load(m_indexFilename.c_str());
        if (!m_index)
        {
            LOG4CXX_ERROR(g_log, "unable to open index " << m_indexFilename);
            return false;
        }
    }
    return true;
}

bool BamReader::select(SequenceInterval const& interval)
{
    int tid = -1;

    m_header.getSequence(interval.getSequence(), &tid);
    if (tid >= 0)
        return select(tid, interval.getStart(), interval.getEnd());
    //else
    //    throw reference_sequence_not_found(interval.getSequence(), m_filename);
    // Setup invalid iterator state. Not everyone checks return values
    m_errorState = 1;
    m_selectTid = -1;
    m_selectBegin = -1;
    m_selectEnd = -1;
    m_chunks.clear();
    return false;
}

bool BamReader::select(const char* region)
{
    if (!m_file)
    {
        LOG4CXX_ERROR(g_log, "bam file not open");
        return false;
    }
    m_selectTid = -1;
    if ((bam_parse_region(m_file->header, region, &m_selectTid, &m_selectBegin, &m_selectEnd) == 0) && (m_selectTid >= 0))
        return select(m_selectTid, m_selectBegin, m_selectEnd);
    else
    {
        std::ostringstream oss;
        oss << "region '" << region << "'";
        LOG4CXX_ERROR(g_log, "unable to parse " << oss.str() <<  " from file " << m_filename);
        m_errorState = 1;
        m_selectTid = -1;
        m_selectBegin = -1;
        m_selectEnd = -1;
        m_chunks.clear();
        //throw reference_sequence_not_found(oss.str(), m_filename);
        return false;
    }
}

bool BamReader::select(int tid, int beg, int end)
{
    //std::cout << tid << ", " << beg << ", " << end << std::endl;
    //assert(beg >= 0); // beg could be -1 if only chr was specified
    assert(tid >= 0);
    // if(beg >=0) assert(end > beg); // Now handled later on.
    m_selectMode = true;
    m_chunks.clear();
    m_selectTid = tid;
    m_selectBegin = (beg >= 0) ? beg : 0;
    m_selectEnd = (end >= 0) ? end : 1<<29; // Convert end of -1 to maximum value
    if (m_index && loadChunkCoordinates(m_index, m_selectTid, m_selectBegin, m_selectEnd))
    {
        m_errorState = 0;
        return true;
    }
    else
    {
        // Fairly common - no reads in chunk, so no chunks returned.
        m_errorState = 1;
        m_selectTid = -1;
        m_selectBegin = -1;
        m_selectEnd = -1;
        return false;
    }
}

bool BamReader::loadChunkCoordinates(bam_index_t *index, int tid, int begin, int end)
{
    //pair64_t *off = get_chunk_coordinates(index, tid, begin, end, &nOff);
    if(begin < 0) begin = 0;
    if(end < 0) end = 0; // Avoid early return if begin and end are -1.
    bam_iter_t iter = bam_iter_query(index, tid, begin, end);
    // Iterator will be NULL if end < beg.
    if(iter == NULL) {
        LOG4CXX_DEBUG(g_log, "Iterator Query for " << tid << ":" << begin << "-" << end << " failed.");
        return false;
    }
    int nOff = iter->n_off;
    if(nOff == 0) {
        bam_iter_destroy(iter);
        return false; // No reads in index
    }
    m_chunks.reserve(nOff);
    for (int i = 0; i < nOff; i++)
    {
        m_chunks.push_back(BamIndexChunk(iter->off[i].u, iter->off[i].v));
    }
    bam_iter_destroy(iter);
    return true;
}

bam_header_t * BamReader::getRawHeader() const
{
    return bam_header_dup(m_file->header);
}

void BamHeader::addBamStats(const char* filename)
{
    BasReader bas(filename);
    BasReader::iterator iter = bas.begin();
    BasReader::iterator end = bas.end();
    while (iter != end)
    {
        BasRecord const& br = *iter;
        setRGStats(br.getReadGroup(), br);
        ++iter;
    }
}

} //namespace lifetechnologies

