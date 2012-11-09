/*
 *  Created on: 06-16-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: $
 *  Last changed by:  $Author: $
 *  Last change date: $Date: $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef BAS_HPP_
#define BAS_HPP_

#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <lifetech/string/util.hpp>
#include <samita/common/types.hpp>
#include <samita/common/record_stream.hpp>
#include <samita/exception/exception.hpp>

#define BAS_COMMENT_CHAR      '#'

namespace lifetechnologies
{

/*!
 Class representing a BAS file record
 \sa ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/pilot_data/README.bas
 */
class BasRecord : public RGStats
{
public:
    BasRecord() : RGStats() {}

    void setFilename(std::string const& str) {m_filename = str;}
    std::string const& getFilename() const {return m_filename;}

    void setMD5(std::string const& str) {m_md5 = str;}
    std::string const& getMD5() const {return m_md5;}

    void setStudy(std::string const& str) {m_study = str;}
    std::string const& getStudy() const {return m_study;}

    void setSample(std::string const& str) {m_sample = str;}
    std::string const& getSample() const {return m_sample;}

    void setPlatform(std::string const& str) {m_platform = str;}
    std::string const& getPlatform() const {return m_platform;}

    void setLibrary(std::string const& str) {m_library = str;}
    std::string const& getLibrary() const {return m_library;}

    void setReadGroup(std::string const& str) {m_readgroup = str;}
    std::string const& getReadGroup() const {return m_readgroup;}
private:
    std::string m_filename;
    std::string m_md5;
    std::string m_study;
    std::string m_sample;
    std::string m_platform;
    std::string m_library;
    std::string m_readgroup;
};

/*!
 \class BasReader
 Class used to open and iterate over records in BAS files.
 \sa ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/pilot_data/README.bas
 \see RecordReader
 \par Example
 The following example opens a BAS file and iterates over all records in the file
 \code
 BasReader reader("my.bas");
 BasReader::iterator iter = reader.begin();
 BasReader::iterator end = reader.end();
 while (iter != end)
 {
     BasRecord const& record = *iter;
     // do some something useful with record
     ++iter;
 }
 reader.close();
 \endcode
 */
typedef RecordReader<BasRecord> BasReader;

/*!
 \class BasWriter
 Class used to output BamStats records to a file.
 \sa ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/pilot_data/README.bas
 \see RecordWriter
 \par Example
 The following example opens a BAS file and writes a record
 \code
BasWriter writer("my.bas");
BasRecord record;
// set up the record with appropriate data values
record.setInsertSizeMean(12345);
record.setInsertSizeSd(123);

// write the record
writer << record << std::endl;
writer.close();
\endcode
 */
typedef RecordWriter<BasRecord> BasWriter;

/*!
 >> operator for reading a formatted GffFeature from an input stream
 \exception invalid_input_record
 */
inline std::istream &operator>>(std::istream &in, BasRecord &record)
{
    std::string line;

    while (!in.eof())
    {
        std::getline(in, line, '\n');
        if (line[0] != BAS_COMMENT_CHAR)
            break;
    }
    if (!line.empty())
    {
        std::vector<std::string> tokens;
        string_util::tokenize(line, "\t", tokens);
        if (tokens.size() < 19)
            throw invalid_input_record("bas", line);

        record.setFilename(tokens[0]);
        record.setMD5(tokens[1]);
        record.setStudy(tokens[2]);
        record.setSample(tokens[3]);
        record.setPlatform(tokens[4]);
        record.setLibrary(tokens[5]);
        record.setReadGroup(tokens[6]);

        if ((tokens[7] != ".") && (!tokens[7].empty()))
            record.setTotalBases(boost::lexical_cast<size_t>(tokens[7]));
        else
            record.setTotalBases(0);

        if ((tokens[8] != ".") && (!tokens[8].empty()))
            record.setMappedBases(boost::lexical_cast<size_t>(tokens[8]));
        else
            record.setMappedBases(0);
        if ((tokens[9] != ".") && (!tokens[9].empty()))
            record.setTotalReads(boost::lexical_cast<size_t>(tokens[9]));
        else
            record.setTotalReads(0);
        if ((tokens[10] != ".") && (!tokens[10].empty()))
            record.setMappedReads(boost::lexical_cast<size_t>(tokens[10]));
        else
            record.setMappedReads(0);
        if ((tokens[11] != ".") && (!tokens[11].empty()))
            record.setMappedReadsPairedInSequencing(boost::lexical_cast<size_t>(tokens[11]));
        else
            record.setMappedReadsPairedInSequencing(0);
        if ((tokens[12] != ".") && (!tokens[12].empty()))
            record.setMappedReadsProperlyPaired(boost::lexical_cast<size_t>(tokens[12]));
        else
            record.setMappedReadsProperlyPaired(0);

        if ((tokens[13] != ".") && (!tokens[13].empty()))
            record.setPctMismatchedBases(boost::lexical_cast<double>(tokens[13]));
        else
            record.setAvgQualityMappedBases(0.0);
        if ((tokens[14] != ".") && (!tokens[14].empty()))
            record.setAvgQualityMappedBases(boost::lexical_cast<double>(tokens[14]));
        else
            record.setAvgQualityMappedBases(0.0);
        if ((tokens[15] != ".") && (!tokens[15].empty()))
            record.setMeanInsertSize(boost::lexical_cast<double>(tokens[15]));
        else
            record.setMeanInsertSize(0.0);
        if ((tokens[16] != ".") && (!tokens[16].empty()))
            record.setSdInsertSize(boost::lexical_cast<double>(tokens[16]));
        else
            record.setSdInsertSize(0.0);
        if ((tokens[17] != ".") && (!tokens[17].empty()))
            record.setMedianInsertSize(boost::lexical_cast<double>(tokens[17]));
        else
            record.setMedianInsertSize(0.0);
        if ((tokens[18] != ".") && (!tokens[18].empty()))
            record.setAdMedianInsertSize(boost::lexical_cast<double>(tokens[18]));
        else
            record.setAdMedianInsertSize(0.0);
    }
    return in;
}

/*!
 ostream manipulator for outputting a string as a gff comment line
 */
inline std::ostream &bas_comment(std::ostream &out)
{
    out << BAS_COMMENT_CHAR;
    return out;
}

/*!
 << operator for outputting a formatted GffFeature to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, BasRecord const& record)
{
    out << record.getFilename() << "\t";
    out << record.getMD5() << "\t";
    out << record.getStudy() << "\t";
    out << record.getSample() << "\t";
    out << record.getPlatform() << "\t";
    out << record.getLibrary() << "\t";
    out << record.getReadGroup() << "\t";

    out << record.getTotalBases() << "\t";
    out << record.getMappedBases() << "\t";
    out << record.getTotalReads() << "\t";
    out << record.getMappedReads() << "\t";
    out << record.getMappedReadsPairedInSequencing() << "\t";
    out << record.getMappedReadsProperlyPaired() << "\t";

    out << record.getPctMismatchedBases() << "\t";
    out << record.getAvgQualityMappedBases() << "\t";
    out << record.getMeanInsertSize() << "\t";
    out << record.getSdInsertSize() << "\t";
    out << record.getMedianInsertSize() << "\t";
    out << record.getAdMedianInsertSize() << "\t";
    return out;
}

} //namespace lifetechnologies

#endif //BAS_HPP_
