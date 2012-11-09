/*
 *  Created on: 04-12-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49984 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:54:43 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef GFF_HPP_
#define GFF_HPP_

#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <samita/common/feature.hpp>
#include <samita/common/record_stream.hpp>
#include <samita/common/types.hpp>
#include <samita/exception/exception.hpp>
#include <lifetech/string/util.hpp>

#define GFF_COMMENT_CHAR      '#'

namespace lifetechnologies
{

typedef Feature GffFeature;

/*!
 \class GffReader
 Class used to open and iterate over records in GFF files.
 \sa http://www.sanger.ac.uk/resources/software/gff/spec.html
 \note
 The GffReader does support standard Gff files that have been compressed with gzip.
 Compressed files should have a ".gz" extension.
 \see RecordReader
 \par Example
 The following example opens a GFF file and iterates over all records in the file
 \code
 GffReader reader("my.gff");
 GffReader::iterator iter = reader.begin();
 GffReader::iterator end = reader.end();
 while (iter != end)
 {
     GffFeature const& record = *iter;
     // do some something useful with record
     ++iter;
 }
 reader.close();
 \endcode
 */
typedef RecordReader<GffFeature> GffReader;

/*!
 \class GffWriter
 Class used to output GFF records to a file.
 \sa http://www.sanger.ac.uk/resources/software/gff/spec.html
 \note
 The GffWriter does support outputting to a gzip compressed file.
 \see RecordWriter
 \par Uncompressed Example
 The following example opens a GFF file and writes a record
 \code
GffWriter writer("my.gff");
GffFeature record;
// set up the record with appropriate data values
record.setSequence("foo");
record.setSource("src");
record.setFeature("feat1");
record.setStart(1);
record.setEnd(100);
record.setScore(3.1415);

// write a comment
writer << gff_comment << "Test comment" << std::endl;
// write the record
writer << record << std::endl;
writer.close();
\endcode
 \par Compressed Example
 The following example opens a compressed GFF file and writes a record
 \code
GffWriter writer("my.gff", true);
GffFeature record;
// set up the record with appropriate data values
record.setSequence("foo");
record.setSource("src");
record.setFeature("feat1");
record.setStart(1);
record.setEnd(100);
record.setScore(3.1415);

// write a comment
writer << gff_comment << "Test comment" << std::endl;
// write the record
writer << record << std::endl;
writer.close();
\endcode
 */
typedef RecordWriter<GffFeature> GffWriter;

/*!
 >> operator for reading a formatted GffFeature from an input stream
 \exception invalid_input_record
 */
inline std::istream &operator>>(std::istream &in, GffFeature &record)
{
    std::string line;

    while (!in.eof())
    {
        std::getline(in, line, '\n');
        if (line[0] != GFF_COMMENT_CHAR)
            break;
    }
    if (!line.empty())
    {
        std::vector<std::string> tokens;
        string_util::tokenize(line, "\t", tokens);
        if (tokens.size() < 8)
            throw invalid_input_record("gff", line);

        record.setSequence(tokens[0]);
        record.setSource(tokens[1]);
        record.setType(tokens[2]);
        record.setStart(boost::lexical_cast<int>(tokens[3]));
        record.setEnd(boost::lexical_cast<int>(tokens[4]));
        if (tokens[5] == ".")
            record.setScore(0);
        else
            record.setScore(atof(tokens[5].c_str()));
        record.setStrand(tokens[6]);
        record.setFrame(tokens[7]);

        if (tokens.size() > 8)
        {
            // parse the attributes
            std::vector<std::string> attrs;
            string_util::tokenize(tokens[8], ";", attrs);
            for (std::vector<std::string>::const_iterator iter = attrs.begin(); iter != attrs.end(); ++iter)
            {
                std::string const& attr = *iter;
                std::vector<std::string> attr_data;
                string_util::tokenize(attr, "=", attr_data);
                if (attr_data.size() != 2)
                    throw invalid_input_record("gff", line);
                record.setAttribute(attr_data[0], attr_data[1]);
            }
        }
    }
    return in;
}

/*!
 ostream manipulator for outputting a string as a gff comment line
 */
inline std::ostream &gff_comment(std::ostream &out)
{
    out << GFF_COMMENT_CHAR;
    return out;
}

/*!
 << operator for outputting a formatted GffFeature to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, GffFeature const& record)
{
    out << record.getSequence() << "\t";
    out << record.getSource() << "\t";
    out << record.getType() << "\t";
    out << record.getStart() << "\t";
    out << record.getEnd() << "\t";
    if (record.getScore() > 0)
        out << record.getScore() << "\t";
    else
        out << "." << "\t";
    out << record.getStrand() << "\t";
    out << record.getFrame() << "\t";

    FeatureAttributeMap const& attrs = record.getAttributes();
    FeatureAttributeMap::const_iterator attrs_iter = attrs.begin();
    FeatureAttributeMap::const_iterator attrs_end = attrs.end();
    size_t nAttr = attrs.size();
    size_t iAttr = 0;
    while (attrs_iter != attrs_end)
    {
        std::string const& name = attrs_iter->first;
        FeatureAttributeValues const& values = attrs_iter->second;
        FeatureAttributeValues::const_iterator values_iter = values.begin();
        FeatureAttributeValues::const_iterator values_end = values.end();
        size_t nValue = values.size();
        size_t iValue = 0;
        while (values_iter != values_end)
        {
            std::string value = *values_iter;
            out << name << "=" << value;
            ++values_iter;
            iValue++;
            if (iValue != nValue)
                out << ";";
        }
        ++attrs_iter;
        iAttr++;
        if (iAttr != nAttr)
            out << ";";
    }
    return out;
}

} //namespace lifetechnologies

#endif //GFF_HPP_
