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
#include "samita/align/align_reader.hpp"

#include <algorithm>
#include <functional>
#include <log4cxx/logger.h>
#include <boost/regex.hpp>

namespace lifetechnologies
{

static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.samita.align_reader");

LibraryType getLibType(std::string const& library)
{
               // For ION reads library type not well defined therefore we default it to FRAG
               // IMPORTANT!!!!
               return LIBRARY_TYPE_FRAG;
               
	//LOG4CXX_DEBUG(g_log, "library is " << library);
	LOG4CXX_DEBUG(g_log, "Deprecated: Using LibraryType from Legacy LB or DS record. "
	        << "Please use new 2.0 comment metadata instead.")

	size_t len = library.size();
	if (len == 0) {
		return LIBRARY_TYPE_NA;
	}
	char x = library[len -1];
	if (x == 'F') {
		return LIBRARY_TYPE_FRAG;
	}
	else if (x == 'R'){
		if (len > 2 && library[len - 2] == 'R') {
			return LIBRARY_TYPE_RR;
		}
	}
	else if (x == 'P'){
		if (len > 2 && library[len - 2] == 'M'){
			return LIBRARY_TYPE_MP;
		}
	}
	else if (x == 'C'){
		if (len > 2 && library[len - 2] == 'B'){
			return LIBRARY_TYPE_RRBC;
		}
	}
	LOG4CXX_DEBUG(g_log, "Unable to identify library type from: '" << library << "'");
	return LIBRARY_TYPE_NA;
}

LibraryType getLibType(RG const& rg)
{
    LibraryType lib = getLibType(rg.LB);
    if(lib == LIBRARY_TYPE_NA)
    {
        LOG4CXX_DEBUG(g_log, "Unable to get LibraryType from LB: '" << rg.LB
                <<  "', trying DS: '" << rg.DS << "'");
        lib = getLibType(rg.DS);
    }
    return lib;

}

std::string getCategory(Align const &ac, LibraryType lib_type, int32_t max, int32_t min)
{
    std::ostringstream strm;

    int32_t insertSize = ac.getInsertSize();
    int32_t start = ac.getStart();
    int32_t mateStart = ac.shouldHaveMate() ? ac.getMateStart() : -1;
    int32_t refId = ac.getRefId();
    int32_t mateRefId = ac.shouldHaveMate() ? ac.getMateRefId() : 0;
    Strand strand = ac.getStrand();
    Strand mateStrand = ac.shouldHaveMate() ? ac.getMateStrand() : STRAND_NA;
    bool isFirst = ac.shouldHaveMate() ? ac.isFirstRead() : true; // FIXME? true?
    bool isProperPair = ac.shouldHaveMate() ? ac.isProperPair() : false;
    bool isMateUnmapped = ac.shouldHaveMate() ? ac.isMateUnmapped() : false;

    if (isProperPair) {
        return "AAA";
    }

    if (refId != mateRefId)
    {
        return "C**";
    }
    else if (isMateUnmapped)
    {
        return "D**";
    }
    if (lib_type == LIBRARY_TYPE_NA){
        return strm.str();
    }

    if (lib_type == LIBRARY_TYPE_MP){
        if (strand == mateStrand)
            strm << "A";
        else
            strm << "B";
        // if it's F3
        if (isFirst)
        {
            if (strand == FORWARD)
            {
                if (start > mateStart)
                    strm << "A";
                else
                    strm << "B";
            }
            else if (strand == REVERSE)
            {
                if (start < mateStart)
                    strm << "A";
                else
                    strm << "B";
            }
        }
        else
        {
            if (mateStrand == FORWARD)
            {
                if (mateStart > start)
                    strm << "A";
                else
                    strm << "B";
            }
            else if (mateStrand == REVERSE)
            {
                if (mateStart < start)
                    strm << "A";
                else
                    strm << "B";
            }
        }
        if (max == 0 && min == 0)
            strm << "C";
        else if (abs(insertSize) <= max )
        {
            if (abs(insertSize) >= min)
                strm << "A";
            // too small
            else
                strm << "B";
        }
        else
            strm << "C";
    }
    else if (lib_type == LIBRARY_TYPE_RR || lib_type == LIBRARY_TYPE_RRBC)
    {
        if (strand != mateStrand)
            strm << "A";
        else
            strm << "B";
        // if it's F3
        if (isFirst)
        {
            if (strand == FORWARD)
            {
                if (start < mateStart)
                    strm << "A";
                else
                    strm << "B";
            }
            else if (strand == REVERSE)
            {
                if (start > mateStart)
                    strm << "A";
                else
                    strm << "B";
            }
        }
        else
        {
            // this is an F5
            if (mateStrand == FORWARD)
            {
                if (mateStart < start)
                    strm << "A";
                else
                    strm << "B";
            }
            else if (mateStrand == REVERSE)
            {
                if (mateStart > start)
                    strm << "A";
                else
                    strm << "B";
            }
        }
        if (max == 0 && min == 0)
            strm << "C";
        else if (abs(insertSize) <= max )
        {
            if (abs(insertSize) >= min)
                strm << "A";
            else
                strm << "B";
        }
        else
            strm << "C";
    }
    return strm.str();
}

//***********************************
//***********************************
//***********************************
// AlignReader class
//***********************************
//***********************************
//***********************************
void AlignReader::close()
{
    for (BamReaderArray::iterator iter=m_bamReaderArray.begin(); iter!=m_bamReaderArray.end(); ++iter)
    {
        BamReader *bam = *iter;
        delete bam;
    }
    m_bamReaderMap.clear();
    m_bamReaderArray.clear();
}

bool AlignReader::close(char const* filename)
{
    BamReaderMap::iterator iter = m_bamReaderMap.find(filename);
    if (iter != m_bamReaderMap.end())
    {
        BamReader *bam = iter->second;
        delete bam;
        m_bamReaderMap.erase(iter);

        BamReaderArray::iterator iter2 = std::find(m_bamReaderArray.begin(), m_bamReaderArray.end(), bam);
        m_bamReaderArray.erase(iter2);

        buildHeader(); // need to rebuild the header
        buildTidLUT(); // may need to rebuild the ref id mapping
        return true;
    }
    return false;
}

bool AlignReader::open(char const* filename, char const* index_filename)
{
    BamReader *bam = new BamReader();
    if (bam->open(filename, index_filename))
    {
        m_bamReaderMap[filename] = bam;
        m_bamReaderArray.push_back(bam);

        // merge the header
        if (mergeHeader(bam))
        {
            buildTidLUT(); // may need to build the ref id mapping
            return true;
        }
        else
        {
            close(filename);
            return false;
        }
    }
    return false;
}

bool AlignReader::select(char const* region)
{
    if (!m_indexesLoaded)
        loadIndexes();

    if (!m_indexesLoaded)
        return false;

    clear(); // Clear any errors
    bool ok = false;
    for (BamReaderArray::iterator iter=m_bamReaderArray.begin(); iter!=m_bamReaderArray.end(); ++iter)
    {
        BamReader *bam = *iter;
        if (bam->isOpen())
        {
            // try {
            ok |= bam->select(region);
            //} catch (reference_sequence_exception & e) { throw; }
        }
    }
    if (!ok)
        m_errorState = 1;
    return ok;
}

bool AlignReader::select(SequenceInterval const& interval)
{
    int tid = -1;
    m_header.getSequence(interval.getSequence(), &tid);
    if (tid >= 0)
    {
        return select(tid, interval.getStart(), interval.getEnd());
    }
    return false;
}

bool AlignReader::select(int tid, int start, int end)
{
    if (!m_indexesLoaded)
        loadIndexes();

    if (!m_indexesLoaded)
        return false;

    clear(); // Clear any errors
    bool ok = false;
    for (BamReaderArray::iterator iter=m_bamReaderArray.begin(); iter!=m_bamReaderArray.end(); ++iter)
    {
        BamReader *bam = *iter;
        if (bam->isOpen())
        {
            ok |= bam->select(tid, start, end);
        }
    }
    if (!ok)
        m_errorState = 1;
    return ok;
}

void AlignReader::buildIndexes()
{
    for (BamReaderArray::iterator iter=m_bamReaderArray.begin(); iter!=m_bamReaderArray.end(); ++iter)
    {
        BamReader *bam = *iter;
        if (bam->isOpen())
        {
            bam->buildIndex();
        }
    }
}

void AlignReader::buildTidLUT()
{
    m_tidLUT.clear();
    int nBams = m_bamReaderArray.size();

    if (nBams > 1) // no need to make LUT if < 2 bams open
    {
        std::vector<SQ> const& mergedSeqs = m_header.getSequenceDictionary();

        // set first table dimension to the number of bam objects
        m_tidLUT.resize(nBams);
        int b_index = 0;
        for (BamReaderArray::iterator b_iter=m_bamReaderArray.begin(); b_iter!=m_bamReaderArray.end(); ++b_iter, b_index++)
        {
            BamReader *bam = *b_iter;
            BamHeader const& hdr = bam->getHeader();
            std::vector<SQ> const& seqs = hdr.getSequenceDictionary();
            uint32_t old_id = 0;
            // get a slice of the table for the current bam object
            std::vector<uint32_t> &slice = m_tidLUT[b_index];
            // set the second table dimension to the size of the sequence dictionary
            slice.resize(seqs.size());
            // iterator over each sequence in sequence dictionary
            for (std::vector<SQ>::const_iterator s_iter = seqs.begin(); s_iter!=seqs.end(); ++s_iter, old_id++)
            {
                SQ const& sq = *s_iter;
                // try to find this sequence in the previously merged sequence dictionary
                std::vector<SQ>::const_iterator findIter = std::find(mergedSeqs.begin(), mergedSeqs.end(), sq);
                // NB: Moved assertion inside test. Always an error
                // - assert just provides more detail during debugging.
                if(findIter == mergedSeqs.end()) // we'd better find it
                {
                    int count = 0;
                    for (std::vector<SQ>::const_iterator i = mergedSeqs.begin(); i != mergedSeqs.end(); ++i, ++count) {
                        LOG4CXX_DEBUG(g_log, "Merged Sequences [ " << count << " ]: " << (*i));
                    }
                    assert(!mergedSeqs.empty());
                    assert(findIter != mergedSeqs.end()); // DEBUG MODE
                    LOG4CXX_FATAL(g_log, "Sequence " << sq.SN << " not found in merged dictionary.");
                    abort(); // Production mode
                }
                // compute and set it's new id
                uint32_t new_id = findIter - mergedSeqs.begin();
                LOG4CXX_DEBUG(g_log, "mapping index : " << b_index << " : " << old_id << " to " << new_id);
                slice[old_id] = new_id;
            }
        }
    }
}

void AlignReader::buildHeader()
{
    m_header.clear();
    for (BamReaderArray::iterator b_iter=m_bamReaderArray.begin(); b_iter!=m_bamReaderArray.end(); ++b_iter)
    {
        BamReader *bam = *b_iter;
        mergeHeader(bam); // Merge handles the empty case fine...
    }
}

struct RGByName : std::unary_function<RG, bool>
{
  private:
    std::string const & m_id;
  public:
    RGByName(std::string const & id) : m_id(id) {;}
    bool operator() (RG const & rg) { return rg.ID == m_id; }
};

struct ReplaceRGLUT : public std::unary_function<std::string, boost::smatch>
{
    typedef std::map<std::string, std::string> LUTMap;
    LUTMap lut;

    template <typename Out>
    Out & operator()(boost::smatch const & what, Out & out) const
    {
        out << "@CO\tRG:";
        LUTMap::const_iterator where = lut.find(what[1]);
        if(where != lut.end())
        {
            out << where->second;
        } else
        {
            out << what[1].str(); // original content
        }
        out << "\t";
        return out;
    }

    std::string replace(std::string const & input) const
    {
        //boost::smatch m;
        boost::match_results<std::string::const_iterator> m;
        static const boost::regex hasRGID("^@CO\tRG:(.*?)\t");
        if(boost::regex_search(input, m, hasRGID, boost::match_default | boost::match_continuous))
        {
            std::ostringstream out;
            (*this)(m, out); // Output replaced string
            out << m.suffix();
            //std::copy(m.suffix().first, m.suffix().second, out); // Output remainder of line
            return out.str();
        } else {
            return input;
        }
    }
};

bool AlignReader::mergeHeader(BamReader const* bam)
{
    BamHeader const& hdrIn = bam->getHeader();
    if (m_header.empty())
    {
        // Handle first header correctly
        m_header = hdrIn;
        return true;
    }
    // check that the header line values are consistent
    else if ((hdrIn.getVersion() != m_header.getVersion()) ||
             (hdrIn.getSortOrder() != m_header.getSortOrder()) ||
             (hdrIn.getGroupOrder() != m_header.getGroupOrder()))
    {
        LOG4CXX_WARN(g_log, "BAM Header validation @HD mismatch while merging: " << bam->getFilename());
        LOG4CXX_INFO(g_log, "Existing: @HD\tVN:" << m_header.getVersion() << "\tGO:" << m_header.getGroupOrder() << "\tSO:" << m_header.getSortOrder());
        LOG4CXX_INFO(g_log, "     New: @HD\tVN:" <<  hdrIn.getVersion() << "\tGO:" <<  hdrIn.getGroupOrder() << "\tSO:" <<  hdrIn.getSortOrder());
        return false;
    }

    // Create temporary copy of existing header, so that we can attempt to
    // modify it without disrupting internal state until succesful
    BamHeader hdrOut(m_header);

    // merge the sequence dictionary
    bool merge = true; // until proven otherwise
    std::vector<SQ> const& seqsIn = hdrIn.getSequenceDictionary();
    std::vector<SQ> const& seqsOut = hdrOut.getSequenceDictionary();

    for (std::vector<SQ>::const_iterator iterIn = seqsIn.begin(); iterIn!=seqsIn.end(); ++iterIn)
    {
        SQ const& sqIn = *iterIn;

        for (std::vector<SQ>::const_iterator iterOut = seqsOut.begin(); iterOut!=seqsOut.end(); ++iterOut)
        {
            SQ const& sqOut = *iterOut;
            if (sqIn == sqOut)
            {
                merge = false;
                //sqOut.merge(sqIN); // Take superset of fields.
            }
            else if(sqIn.SN == sqOut.SN)
            {
                LOG4CXX_ERROR(g_log, "BAM Header Merge - SQ records with same name but different content while merging: " << bam->getFilename());
                LOG4CXX_INFO(g_log, "Existing: " << sqOut);
                LOG4CXX_INFO(g_log, "     New: " << sqIn);
                return false;
            }
        }
        if (merge)
            hdrOut.addSequence(sqIn); // we have a new seq so add to merged header
    }

    // merge the read groups
    merge = true; // until proven otherwise
    std::vector<RG> const& rgsIn = hdrIn.getReadGroups();
    std::vector<RG> const& rgsOut = hdrOut.getReadGroups();
    ReplaceRGLUT rgIdLUT;
    for (std::vector<RG>::const_iterator iterIn = rgsIn.begin(); iterIn!=rgsIn.end(); ++iterIn)
    {
        RG rgIn = *iterIn; // Copy, not a reference

        // A for loop? Surely we can lookup in existing header by RGID?
        int suffix = 0;
        std::vector<RG>::const_iterator ret = std::find_if(rgsOut.begin(), rgsOut.end(), RGByName(rgIn.ID));
        while(ret != rgsOut.end())
        {
            RG const & rgOut = *ret;
            if (rgIn == rgOut)
            {
                merge = false;
                break; // Found identical RG. Stop.
            }
            // else
            LOG4CXX_DEBUG(g_log, "BAM Header Merge - RG records with same name but different content while merging: " << bam->getFilename());
            LOG4CXX_DEBUG(g_log, "Existing: " << rgOut);

            std::ostringstream oss;
            oss << (*iterIn).ID << "_" << ++suffix;
            rgIn.ID = oss.str();

            LOG4CXX_DEBUG(g_log, "     New: " << rgIn);

            // Save the mapping so we can update CO RG IDs.
            rgIdLUT.lut[(*iterIn).ID] = rgIn.ID;

            ret = std::find_if(rgsOut.begin(), rgsOut.end(), RGByName(rgIn.ID));
        }
        if (merge)
            hdrOut.addReadGroup(rgIn);  // we have a new rg so add to merged header
    }

    // merge the programs
    std::vector<PG> const& pgsIn = hdrIn.getPrograms();
    std::vector<PG> const& pgsOut = hdrOut.getPrograms();
    for (std::vector<PG>::const_iterator iter = pgsIn.begin(); iter!=pgsIn.end(); ++iter)
    {
        PG const& pg = *iter;
        std::vector<PG>::const_iterator findIter = std::find(pgsOut.begin(), pgsOut.end(), pg);
        if (findIter == pgsOut.end())
            hdrOut.addProgram(pg);
    }

    // merge the comments
    // FIXME - properly merge metadata comments!
    std::vector<std::string> const& cosIn = hdrIn.getComments();
    std::vector<std::string> const& cosOut = hdrOut.getComments();
    for (std::vector<std::string>::const_iterator iter = cosIn.begin(); iter!=cosIn.end(); ++iter)
    {
        std::string const & co = rgIdLUT.replace(*iter);
        // Avoid adding exact duplicate comments.
        std::vector<std::string>::const_iterator findIter = std::find(cosOut.begin(), cosOut.end(), co);
        if (findIter == cosOut.end())
            hdrOut.addComment(co);
    }

    // FIXME - build bam_header_t from hdrOut!

    // finally we can update the header member variable with the temporary merged header
    m_header = hdrOut;
    return true;
}

void AlignReader::loadIndexes()
{
    m_indexesLoaded = true;
    for (BamReaderArray::iterator iter=m_bamReaderArray.begin(); iter!=m_bamReaderArray.end(); ++iter)
    {
        BamReader *bam = *iter;
        if (bam->isOpen())
        {
            m_indexesLoaded &= bam->loadIndex();
        }
    }
}


} // lifetechnologies namespace
