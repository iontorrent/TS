/*
 *  Created on: 04-12-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 88236 $
 *  Last changed by:  $Author: kerrs1 $
 *  Last change date: $Date: 2011-05-04 16:53:53 -0700 (Wed, 04 May 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <samita/reference/reference.hpp>
#include <faidx.h>
#include <lifetech/string/util.hpp>

namespace lifetechnologies
{

std::string ReferenceSequence::get(size_t start, size_t end) const
{
    if (start > end)
    {
        std::stringstream sstrm;
        sstrm << "end: " << end << " > "  << "start: " << start;
        throw std::invalid_argument(sstrm.str());
    }
    if (start < 1)
        throw reference_sequence_index_out_of_bounds(m_name, m_bases.length(), start);
    else if (end > m_bases.length())
        throw reference_sequence_index_out_of_bounds(m_name, m_bases.length(), end);
    return m_bases.substr(start-1, (end-start));
}

bool ReferenceSequenceReader::open(const char* filename, const char* index_filename)
{
    close(); // just in case the previous open call was not paired with a close
    m_filename = filename;
    if (index_filename)
        m_indexFilename = index_filename;
    else
        m_indexFilename = m_filename;

    // do some basic checks on the FASTA file; these may not matter if a valid index file exists
    std::ifstream refFile( filename, std::ios::in );
    bool isBadRefFile = refFile.fail();		// file does not exist?
    if ( !isBadRefFile ) {
    	refFile.peek();						// force disk read to ensure valid EOF test?
    	isBadRefFile = refFile.eof();		// already at EOF? then file is empty
    }
    refFile.close();

    // check for index file, if it does not exist then create it
    std::string realIndexFilename = m_indexFilename + ".fai";
    std::ifstream file(realIndexFilename.c_str(), std::ios::in | std::ios::binary);
    bool indexFileMissing = file.fail();
    if ( indexFileMissing ) {
    	if( isBadRefFile )	{	// reference FASTA file is missing or empty; can't index it
    		file.close();
    		throw index_creation_exception( filename );
    	}
        buildIndex();
    }
    file.close();

    // load the dictionary
    std::ifstream indexFile(realIndexFilename.c_str(), std::ios::in);
    std::string line;
    while (!indexFile.eof())
    {
        std::getline(indexFile, line, '\n');
        if (!line.empty())
        {
            std::vector<std::string> tokens;
            string_util::tokenize(line, " \t", tokens);
            if (tokens.size() < 5)
                throw invalid_input_record("fai", line);

            m_dictionary.push_back(ReferenceDictionaryEntry(tokens[0], atoi(tokens[1].c_str())));
        }
    }

    // load the index
    m_indexPtr = fai_load(m_filename.c_str());

    return ((m_dictionary.size() > 0) && (m_indexPtr));
}

void ReferenceSequenceReader::close()
{
    // close index
    if (m_indexPtr)
    {
        fai_destroy(m_indexPtr);
        m_indexPtr = NULL;
    }
    m_dictionary.clear();
}

void ReferenceSequenceReader::buildIndex()
{
    if (fai_build(m_indexFilename.c_str()) != 0)
        throw index_creation_exception(m_indexFilename);
}

ReferenceSequence const& ReferenceSequenceReader::getSequence(const char* range)
{
    if (m_indexPtr)
    {
        std::string name;
        int start;
        int end;
        SequenceInterval::parse(range, name, start, end);
        size_t contigIndex = findReferenceSequenceContig(name);
        if (contigIndex > 0)
        {
            int len;
            char *bases = fai_fetch(m_indexPtr, range, &len);
            m_refseq.setName(name);
            m_refseq.setContigIndex(contigIndex);
            if (bases)
                m_refseq.setBases(bases);
            free(bases);
            return m_refseq;
        }
        throw reference_sequence_not_found(name, m_filename);
    }
    throw reference_sequence_not_found(range, m_filename);
}

ReferenceSequence const& ReferenceSequenceReader::getSequence(const char* sequence, int start, int end)
{
    if (m_indexPtr)
    {
        size_t contigIndex = findReferenceSequenceContig(sequence);
        if (contigIndex > 0)
        {
            int len;

            char *bases = faidx_fetch_seq(m_indexPtr, const_cast<char *> (sequence), start - 1, end - 1, &len);
            m_refseq.setName(sequence);
            m_refseq.setContigIndex(contigIndex);
            if (bases)
                m_refseq.setBases(bases);
            free(bases);
            return m_refseq;
        }
        throw reference_sequence_not_found(sequence, m_filename);
    }
    throw reference_sequence_not_found(sequence, m_filename);
}

size_t ReferenceSequenceReader::findReferenceSequenceContig(std::string const& name)
{
    int contigIndex = 0;
    for (std::vector< ReferenceDictionaryEntry >::const_iterator iter = m_dictionary.begin(); iter != m_dictionary.end(); ++iter, contigIndex++)
    {
        ReferenceDictionaryEntry const& entry = *iter;
        if (entry.first == name)
        {
            return (contigIndex + 1);
        }
    }
    return 0;
}

ReferenceSequenceReader::iterator::iterator(ReferenceSequenceReader *parent) :
    m_indexPtr(NULL), m_dictionaryPtr(NULL)
{
    if (parent)
    {
        m_indexPtr = parent->m_indexPtr;
        m_dictionaryPtr = &parent->m_dictionary;
        m_dictionaryIter = m_dictionaryPtr->begin();
        m_dictionaryEnd = m_dictionaryPtr->end();
        m_dictionarySize = m_dictionaryPtr->size();
        // queue up the first record
        operator++();
    }
}


ReferenceSequenceReader::iterator& ReferenceSequenceReader::iterator::operator++()
{
    if (m_dictionaryIter != m_dictionaryEnd)
    {
        // get the next record
        std::string name = m_dictionaryIter->first;
        int len;
        char *bases = fai_fetch(m_indexPtr, name.c_str(), &len);
        m_refseq.setName(name);
        m_refseq.setContigIndex(m_dictionarySize - (m_dictionaryEnd - m_dictionaryIter) + 1);
        if (bases)
            m_refseq.setBases(bases);
        free(bases);
        ++m_dictionaryIter;
    }
    else
        *this = iterator(NULL);
    return *this;
}


} //namespace lifetechnologies
