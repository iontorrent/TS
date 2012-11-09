/*
 *  Created on: 04-20-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 47952 $
 *  Last changed by:  $Author: moultoka $
 *  Last change date: $Date: 2010-07-08 11:05:34 -0400 (Thu, 08 Jul 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef RECORD_STREAM_HPP_
#define RECORD_STREAM_HPP_

#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/shared_ptr.hpp>
#include <lifetech/string/util.hpp>
#include <samita/exception/exception.hpp>

namespace lifetechnologies
{

/*!
 Class used to open and iterate over records of the type DataType in files.
 The >> operator must be implemented for DataType
 \note
 The RecordReader does support standard Gff files that have been compressed with gzip.
 Compressed files should have a ".gz" extension.
 */
template<class DataType>
class RecordReader
{
public:
    RecordReader(size_t buffer_size = 512000)
    {
        setBufferSize(buffer_size);
    }
    RecordReader(char const* filename, size_t buffer_size = 512000)
    {
        setBufferSize(buffer_size);
        open(filename);
    }
    ~RecordReader()
    {
        close();
    }
    bool open(char const* filename)
    {
        m_file.open(filename, std::ios_base::in | std::ios_base::binary);
        if (m_file.good())
        {
            if (string_util::ends_with(filename, ".gz"))
                m_stream.push(boost::iostreams::gzip_decompressor());
            m_stream.push(m_file);
            return true;
        }
        return false;
    }
    void close()
    {
        m_stream.reset();
        m_file.close();
    }

    void setBufferSize(size_t size)
    {
        if (size != m_buffer.size())
        {
            // resize io buffer
            m_buffer.resize(size);
            // set the io buffer
            m_file.rdbuf()->pubsetbuf(&m_buffer[0], size);
        }
    }
    /*!
       Input iterator class used to iterate over records in opened file.

       Re-uses the same block of memory for each record, so be sure to copy result
       if you need it to psersist.
    */
    template <class Value>
    class record_stream_iterator
        : public boost::iterator_facade<record_stream_iterator<Value>, Value, boost::forward_traversal_tag>
    {
    private:
        /// \cond DEV
        friend class boost::iterator_core_access;
        template <class> friend class record_stream_iterator;
        typedef boost::shared_ptr<DataType> DataTypePtr;
       struct enabler {};
       /// \endcond

    public:
        record_stream_iterator() : m_streamPtr(NULL) {}

        explicit record_stream_iterator(RecordReader * const parent)
          : m_streamPtr(), m_recordPtr(new Value())
        {
            if (parent)
            {
                m_streamPtr = &parent->m_stream;
                // queue up the first record
                increment();
            }
        }

        template <class OtherValue>
        record_stream_iterator(record_stream_iterator<OtherValue> const& other, typename boost::enable_if<boost::is_convertible<OtherValue*,Value*>, enabler>::type = enabler())
            : m_streamPtr(other.m_streamPtr), m_recordPtr(other.m_recordPtr)
        {}


     private:
        template <class OtherValue>
        bool equal(record_stream_iterator<OtherValue> const& other) const
        {
            return (m_streamPtr == other.m_streamPtr);
        }

        void increment()
        {
            // get the next record
            (*m_streamPtr) >> (*m_recordPtr);
            if (m_streamPtr->eof())
                m_streamPtr = NULL;
        }

        Value& dereference() const
        {
            return *m_recordPtr;
        }

        boost::iostreams::filtering_istream *m_streamPtr;
        DataTypePtr m_recordPtr; // Must be non-const ptr
    };

    typedef record_stream_iterator<DataType> iterator;
    typedef record_stream_iterator<DataType const> const_iterator;

    /*!
     Begin iterator
     \return iterator to the first record
     */
    iterator begin()
    {
        m_file.seekg(0);
        if (m_file.good())
            return iterator(this);
        return iterator(NULL);
    }
    /*!
     End iterator
     \return iterator to the end of the records
     */
    iterator end()
    {
        return iterator(NULL);
    }
protected:
    std::ifstream m_file;
    boost::iostreams::filtering_istream m_stream;
    std::vector<char> m_buffer;
private:
    // NOTE: copy constructor not allowed - want compile error if used by self
    RecordReader(RecordReader const&);
};

/*!
 Class used to output records to a file.
 The << operator must be implemented for DataType
 \note
 The RecordWriter does support outputting to a gzip compressed file.
 */
template<class DataType>
class RecordWriter: public boost::iostreams::filtering_ostream
{
public:
    RecordWriter(size_t buffer_size = 512000)
    {
        setBufferSize(buffer_size);
    }
    RecordWriter(const char* filename, bool compressed = false, size_t buffer_size = 512000, boost::iostreams::gzip_params const& params =
            boost::iostreams::zlib::default_compression)
    {
        setBufferSize(buffer_size);
        open(filename, compressed, params);
    }
    virtual ~RecordWriter()
    {
        close();
    }
    bool open(const char* filename, bool compressed = false, boost::iostreams::gzip_params const& params =
            boost::iostreams::gzip_params())
    {
        std::string modified_filename = filename;
        if (compressed)
        {
            if (!string_util::ends_with(modified_filename, ".gz"))
                modified_filename += ".gz";
            push(boost::iostreams::gzip_compressor(params));
        }
        m_file.open(modified_filename.c_str());
        push(m_file);
        return true;
    }
    void close()
    {
        reset();
        m_file.close();
    }
    void setBufferSize(size_t size)
    {
        if (size != m_buffer.size())
        {
            // resize io buffer
            m_buffer.resize(size);
            // set the io buffer
            m_file.rdbuf()->pubsetbuf(&m_buffer[0], size);
        }
    }

protected:
    std::ofstream m_file;
    std::vector<char> m_buffer;
private:
    // NOTE: copy constructor not allowed - want compile error if used by self
    RecordWriter(RecordWriter const&);
};


} //namespace lifetechnologies

#endif //RECORD_STREAM_HPP_
