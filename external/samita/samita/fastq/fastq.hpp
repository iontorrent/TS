/*
 *  Created on: 04-19-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49984 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:54:43 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef FASTQ_HPP_
#define FASTQ_HPP_

#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <lifetech/string/util.hpp>
#include <samita/common/record_stream.hpp>
#include <samita/common/types.hpp>
#include <samita/common/quality_value.hpp>
#include <samita/exception/exception.hpp>

#define FASTQ_COMMENT_CHAR      '#'
#define FASTQ_SEQUENCE_HEADER   '@'
#define FASTQ_QUALITY_HEADER    '+'

namespace lifetechnologies
{

/*!
 Class representing a record from a FASTQ file
 \sa http://maq.sourceforge.net/fastq.shtml
 */
class FastqRecord
{
public:
    FastqRecord() {}
    FastqRecord(std::string const& id, std::string const& sequence, std::string const& qvs, std::string const& description = "") :
        m_id(id), m_sequence(sequence), m_qvs(qvs), m_description(description)
    {
    }
    void setId(std::string const& id)
    {
        m_id = id;
    }
    std::string const& getId() const
    {
        return m_id;
    }

    void setDescription(std::string const& description)
    {
        m_description = description;
    }
    std::string const& getDescription() const
    {
        return m_description;
    }

    void setSequence(std::string const& sequence)
    {
        m_sequence = sequence;
    }
    std::string const& getSequence() const
    {
        return m_sequence;
    }

    void setQvs(std::string const& qvs)
    {
        m_qvs = qvs;
    }
    std::string const& getQvs() const
    {
        return m_qvs;
    }
    void setQvs(QualityValueArray const& qvs)
    {
        m_qvs = qvsToAscii(qvs);
    }
    QualityValueArray getQvArray() const
    {
        return asciiToQvs(m_qvs);
    }

    friend std::istream &operator>>(std::istream &in, FastqRecord &record);

protected:
    std::string m_id;
    std::string m_sequence;
    std::string m_qvs;
    std::string m_description;
};

/*!
 \class FastqReader
 Class used to open and iterate over records in FASTQ files.
 \sa http://maq.sourceforge.net/fastq.shtml
 \note
 The FastqReader does support standard FASTQ files that have been compressed with gzip.
 Compressed files should have a ".gz" extension.
 \see RecordReader
 \par Example
 The following example opens a GFF file and iterates over all records in the file
 \code
     FastqReader reader("my.fastq");
     FastqReader::iterator iter = reader.begin();
     FastqReader::iterator end = reader.end();
     while (iter != end)
     {
         FastqRecord const& record = *iter;
         // do some something useful with record
         ++iter;
     }
     reader.close();
 \endcode
 */
typedef RecordReader<FastqRecord> FastqReader;


/*!
 \class FastqWriter
 Class used to output FASTQ records to a file.
 \sa http://maq.sourceforge.net/fastq.shtml
 \note
 The FastqWriter does support outputting to a gzip compressed file.
 \see RecordWriter
 \par Uncompressed Example
 The following example opens a FASTQ file and writes a record
 \code
FastqWriter writer("my.fastq");
FastqRecord record;
// set up the record with appropriate data values
record.setId("1_1_1");
record.setDescription("test read");
record.setSequence("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
record.setQvs("!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`");

// write a comment
writer << fastq_comment << "Test comment" << std::endl;
// write the record
writer << record << std::endl;
writer.close();
\endcode
 \par Compressed Example
 The following example opens a compressed GFF file and writes a record
 \code
FastqWriter writer("my.fastq", true);
FastqRecord record;
// set up the record with appropriate data values
record.setId("1_1_1");
record.setDescription("test read");
record.setSequence("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
record.setQvs("!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`");

// write a comment
writer << fastq_comment << "Test comment" << std::endl;
// write the record
writer << record << std::endl;
writer.close();
\endcode
 */
typedef RecordWriter<FastqRecord> FastqWriter;

/*!
 >> operator for reading a formatted FastqRecord from an input stream
 \exception invalid_input_record
 */
inline std::istream &operator>>(std::istream &in, FastqRecord &record)
{
    std::string line;

    while (!in.eof())
    {
        std::getline(in, line, '\n');
        if (line[0] != FASTQ_COMMENT_CHAR)
            break;
    }
    if (!line.empty())
    {
        if (line[0] != FASTQ_SEQUENCE_HEADER)
            throw invalid_input_record("fastq", line);
        std::vector<std::string> tokens;
        string_util::tokenize(line, " \t", tokens);
        record.setId(tokens[0]);
        if (tokens.size() > 1)
            record.setDescription(tokens[1]);

        std::getline(in, line, '\n');
        record.setSequence(line);

        std::getline(in, line, '\n');
        if (line[0] != FASTQ_QUALITY_HEADER)
            throw invalid_input_record("fastq", line);

        std::getline(in, line, '\n');
        record.setQvs(line);
    }
    return in;
}

/*!
 ostream manipulator for outputting a string as a FASTQ comment line
 */
inline std::ostream &fastq_comment(std::ostream &out)
{
    out << FASTQ_COMMENT_CHAR;
    return out;
}

/*!
 << operator for outputting a formatted FastqRecord to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, FastqRecord const& record)
{
    out << FASTQ_SEQUENCE_HEADER << record.getId();
    std::string const& description = record.getDescription();
    if (!description.empty())
        out << "\t" << description;
    out << "\n";
    out << record.getSequence() << "\n";
    out << FASTQ_QUALITY_HEADER << "\n";
    out << record.getQvs();
    return out;
}

} //namespace lifetechnologies

#endif //FASTQ_HPP_
