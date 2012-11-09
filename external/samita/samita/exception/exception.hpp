/*
 *  Created on: 04-15-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49984 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:54:43 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef SAMITA_EXCEPTION_HPP_
#define SAMITA_EXCEPTION_HPP_

#include <boost/cstdint.hpp>
#include <stdexcept>
#include <string>

/*!
 lifetechnologies namespace
 */
namespace lifetechnologies
{

/*!
 Exception thrown when an index cannot be created.
 */
class index_creation_exception: public std::exception
{
public:
    index_creation_exception(std::string const& file) :
        m_file(file) {}
    ~index_creation_exception() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << "unable to build index for file : " << m_file;
        return sstrm.str().c_str();
    }
private:
    std::string m_file;
};

/*!
 Exception thrown when an invalid cigar operator is used.
 */
class invalid_cigar_operation: public std::exception
{
public:
    invalid_cigar_operation(uint32_t op) :
        m_op(op) {}
    ~invalid_cigar_operation() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << "invalid cigar operation : " << m_op;
        return sstrm.str().c_str();
    }
private:
    uint32_t m_op;
};

/*!
 Exception thrown when an invalid record is parsed.
 */
class invalid_input_record: public std::exception
{
public:
    invalid_input_record(std::string const& filename, std::string const& record) :
        m_name(filename), m_record(record) {}
    ~invalid_input_record() throw () {}
    const char* what() const throw ()
    {
        std::stringstream sstrm;
        sstrm << m_name << " : " << m_record;
        return sstrm.str().c_str();
    }
private:
    std::string m_name;
    std::string m_record;
};

/*!
 Exception thrown when a read group can not be found.
 */
class read_group_not_found: public std::exception
{
public:
    read_group_not_found(std::string const& id) :
        m_id(id) {}
    read_group_not_found(int32_t id)
    {
        std::ostringstream sstrm;
        sstrm << id;
        m_id = sstrm.str();
    }
    ~read_group_not_found() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << "read group not found : " << m_id;
        return sstrm.str().c_str();
    }
private:
    std::string m_id;
};

/*!
 Exception thrown when a reference sequence can not be found.
 */
class reference_sequence_not_found: public std::exception
{
public:
    reference_sequence_not_found(std::string const& id, std::string const& filename = "") :
        m_id(id), m_filename(filename) {}
    reference_sequence_not_found(int32_t id)
    {
        std::ostringstream sstrm;
        sstrm << id;
        m_id = sstrm.str();
    }
    ~reference_sequence_not_found() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << "reference sequence '" << m_id << "' not found";
        if (!m_filename.empty())
            sstrm << " in file '" << m_filename << "'";
        return sstrm.str().c_str();
    }
private:
    std::string m_id;
    std::string m_filename;
};

/*!
 Exception thrown when reference sequence is queried out of bounds.
 */
class reference_sequence_index_out_of_bounds: public std::exception
{
public:
    reference_sequence_index_out_of_bounds(std::string const& name, size_t length, size_t index) :
        m_name(name), m_length(length), m_index(index) {}
    ~reference_sequence_index_out_of_bounds() throw () {}
    const char* what() const throw ()
    {
        std::ostringstream sstrm;
        sstrm << m_name << " : length is  " <<  m_length << ", can not query with index " << m_index;
        return sstrm.str().c_str();
    }
private:
    std::string m_name;
    size_t m_length;
    size_t m_index;
};

} //namespace lifetechnologies

#endif //SAMITA_EXCEPTION_HPP_
