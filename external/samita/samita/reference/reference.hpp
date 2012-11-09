/*
 *  Created on: 04-12-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 73500 $
 *  Last changed by:  $Author: kerrs1 $
 *  Last change date: $Date: 2011-02-09 11:03:44 -0800 (Wed, 09 Feb 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef REFERENCE_HPP_
#define REFERENCE_HPP_

#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <log4cxx/logger.h>
#include <samita/common/types.hpp>

//forward decls
struct __faidx_t;
typedef struct __faidx_t faidx_t;

namespace lifetechnologies
{

/*!
 Class representing a record from a reference file
 */
class ReferenceSequence
{
public:
    ReferenceSequence() :
        m_contigIndex(0)
    {
    }
    ReferenceSequence(std::string name, const char* bases, size_t contigIndex) :
        m_name(name), m_bases(bases), m_contigIndex(contigIndex)
    {
    }

    void setName(std::string const& name)
    {
        m_name = name;
    }
    std::string const& getName() const
    {
        return m_name;
    }

    void setBases(const char* bases)
    {
        assert(bases);
        m_bases = bases;
    }
    const char *getBases() const
    {
        return m_bases.c_str();
    }

    /*!
     [] operator for indexing into bases (1-based)
     \exception reference_sequence_index_out_of_bounds
     */
    char operator[](size_t i) const
    {
        if ((i < 1) || (i > m_bases.length()))
            throw reference_sequence_index_out_of_bounds(m_name, m_bases.length(), i);
        return m_bases[i - 1];
    }

    /*!
     Method for extracting a sub-sequence (1-based)
     \exception reference_sequence_index_out_of_bounds
     */
    std::string get(size_t start, size_t end) const;

    size_t getLength() const
    {
        return m_bases.length();
    }

    void setContigIndex(size_t index)
    {
        m_contigIndex = index;
    }

    /*!
     Returns the contig index (1-based)
     */
    size_t getContigIndex() const
    {
        return m_contigIndex;
    }

    void clear()
    {
        m_name = "";
        m_bases = "";
        m_contigIndex = 0;
    }
private:
    std::string m_name;
    std::string m_bases;
    size_t m_contigIndex;
};

/*!
 Class used to open, query and iterate over sequences in reference FASTA files.
 \note
 The ReferenceSequenceReader supports standard FASTA files and FASTA files that have been
 compressed in the RAZF format.  Samtools comes with a tool, razip, that will compress FASTA
 references.
 \par Example
 The following example iterated over the reference sequences in a FASTA file
 \code
 ReferenceSequenceReader reader("my_reference.fasta");

 ReferenceSequenceReader::iterator iter = reader.begin();
 ReferenceSequenceReader::iterator end = reader.end();

 while (iter != end)
 {
     ReferenceSequence const& refseq = *iter;
     // do something useful with the ReferenceSequence
     ++iter;
 }
 reader.close();

 \endcode
 \par Example
 The following example queries a sub-sequence of a reference sequences in a FASTA file
 \code
 ReferenceSequenceReader reader("my_reference.fasta");

 ReferenceSequence const& refseq = reader.getSequence("chr2:10-1000");
 // do something useful with the ReferenceSequence

 reader.close();

 \endcode
 */
class ReferenceSequenceReader
{
private:
    typedef std::pair<std::string, size_t> ReferenceDictionaryEntry;
    typedef std::vector< ReferenceDictionaryEntry > ReferenceDictionary;
public:
    ReferenceSequenceReader() :
        m_log(log4cxx::Logger::getLogger("lifetechnologies.samita.ReferenceSequenceReader")),
        m_filename(""), m_indexFilename(""), m_indexPtr(NULL) {}

    ReferenceSequenceReader(char const* filename) :
        m_indexPtr(NULL)
    {
        open(filename);
    }
    ~ReferenceSequenceReader()
    {
        close();
    }

    /*!
     Open a FASTA reference file for iterating or reading.
     \param filename a constant character pointer for the BAM filename
     \param index_filename an optional constant character pointer.
     \return true if successful; otherwise false
     \note
     If the index_filename parameter is not specified then the default index
     file name is used [filename+.fai].  The index will be loaded when open
     is called. If the index file does not exist then it will be created.
     */
    bool open(const char* filename, const char* index_filename = NULL);

    bool isOpen()
    {
        return (m_indexPtr != NULL);
    }

    void close();

    /*!
     \exception index_creation_exception
     */
    void buildIndex();

    /*!
     Query a reference subsequence from the specified region string.
     \param region a chromosome and optional range of the form "chr1:1-1000"
     \exception reference_sequence_not_found
     */
    ReferenceSequence const& getSequence(const char* range);

    /*!
     Query a reference subsequence from the specified sequence interval.
     */
    ReferenceSequence const& getSequence(SequenceInterval const& interval)
    {
        return getSequence(interval.getSequence().c_str(), interval.getStart(), interval.getEnd());
    }

    /*!
     Query a reference subsequence from the specified sequence and interval.
     \exception reference_sequence_not_found
     */
    ReferenceSequence const& getSequence(const char* sequence, int start, int end);

	#define NO_SEQ_LENGTH -1
    // must support signed value on return
    long getLength( std::string refName ) const {
    	// look for the name in the m_dictionary; return length when found
    	for (std::vector< ReferenceDictionaryEntry >::const_iterator iter = m_dictionary.begin();
    			iter != m_dictionary.end(); ++iter) {
	        // iter resolves to a ReferenceDictionaryEntry type (std::pair<std::string,size_t>)
            if ( iter->first.compare(refName) == 0 )
            	return iter->second;
        }
    	// name was not found; return error code for length
    	return NO_SEQ_LENGTH;
    }

    /*!
     Input iterator class used to iterate over sequence records in opened reference file.
     */
    struct iterator: public std::iterator<std::input_iterator_tag, ReferenceSequence>
    {
    public:
        iterator(ReferenceSequenceReader *parent = NULL);
        ~iterator() {}

        iterator& operator=(iterator const& other)
        {
            if (this != &other) // protect against invalid self-assignment
            {
                m_dictionaryPtr = other.m_dictionaryPtr;
            }
            return *this;
        }
        bool operator!=(iterator const& right)
        {
            return (m_dictionaryPtr != right.m_dictionaryPtr);
        }
        /*!
         Increment the iterator.
         \return reference to the iterator
         */
        iterator& operator++();

        /*!
         Dereference the iterator
         \return reference to current ReferenceSequence
         \note
         The return type is a reference.  You should not store a pointer to it.
         because that pointer will not point to the same record once ++ is called.
         If you want to store a record then make a copy.  But, beware, ReferenceSequence
         objects can be large.
         */
        ReferenceSequence const& operator*()
        {
            return m_refseq;
        }
    private:
        faidx_t *m_indexPtr;
        ReferenceDictionary *m_dictionaryPtr;
        ReferenceDictionary::const_iterator m_dictionaryIter;
        ReferenceDictionary::const_iterator m_dictionaryEnd;
        size_t m_dictionarySize;
        ReferenceSequence m_refseq;
    }; //struct iterator

    /*!
     Begin iterator
     \return iterator to the first gff record
     */
    iterator begin()
    {
        if (m_dictionary.size() > 0)
            return iterator(this);
        return iterator(NULL);
    }
    /*!
     End iterator
     \return iterator to the end of the gff records
     */
    iterator end()
    {
        return iterator(NULL);
    }
private:
    //methods
    size_t findReferenceSequenceContig(std::string const& name);

private:
    //members
    log4cxx::LoggerPtr m_log;
    std::string m_filename;
    std::string m_indexFilename;
    faidx_t *m_indexPtr;
    ReferenceDictionary m_dictionary;
    ReferenceSequence m_refseq;
};

} //namespace lifetechnologies

#endif //REFERENCE_HPP_
