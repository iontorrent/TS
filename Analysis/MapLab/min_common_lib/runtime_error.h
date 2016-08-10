/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#ifndef __runtime_error_h__
#define __runtime_error_h__

#include <string>
#include <sstream>
#include <iostream>

#include "common_str.h"

class RunTimeError
{
protected:
    std::string msg_;

public:
    RunTimeError (const char* s = EMPTY_STR) : msg_ ((const char*) s) {}
    operator const char* () const {return msg_.c_str ();}
friend class RunTimeErrorStream;
};

template <typename ExceptionType = RunTimeError>
class RaiseAction
{
public:
    int line_number_;
    const char* file_name_;
    ExceptionType exception_;
    RaiseAction (ExceptionType exception, int line_number = 0, const char* file_name = NULL)
    :
    line_number_ (line_number),
    file_name_ (file_name),
    exception_ (exception)
    {
    }
};

class RunTimeErrorStream : public std::ostringstream
{
public:
    template <typename ExceptionType>
    void raise (ExceptionType exception)
    {
        if (exception.msg_.length ())
            exception.msg_ += " : ";
        exception.msg_ += str ();
        str (EMPTY_STR);
        throw exception;
    }
};

template <typename ExceptionType>
inline RunTimeErrorStream& operator << (RunTimeErrorStream& error_stream, RaiseAction <ExceptionType> raise_action)
{
    struct tm* tinfo;
    time_t cur_time;
    static size_t BUFSZ = 32;
    char time_buffer [BUFSZ];

    time (&cur_time);
    tinfo = localtime (&cur_time);
    strftime (time_buffer, sizeof (time_buffer), "[%x:%X:%Z]", tinfo);

    if (raise_action.file_name_) error_stream << " (module " << raise_action.file_name_ << ", line " << raise_action.line_number_ << ")";
    error_stream << time_buffer;
    error_stream.raise (raise_action.exception_);
    return error_stream;
}

template <typename OperationType>
inline RunTimeErrorStream& operator << (RunTimeErrorStream& error_stream, const OperationType& operation_type)
{
    ((std::ostringstream&) error_stream) << operation_type;
    return error_stream;
}

// ostream manipulators support
inline RunTimeErrorStream& operator << (RunTimeErrorStream& error_stream, std::ostream& (*manipulator) (std::ostream&))
{
    ((std::ostringstream&) error_stream) << manipulator;
    return error_stream;
}

std::ostream& operator << (std::ostream&, const RunTimeError& run_time_error);

#ifndef __runtime_error_cpp__
extern RunTimeErrorStream ers;
#endif

#define Throw RaiseAction <RunTimeError> (RunTimeError (), __LINE__, __FILE__)
#define ThrowEx(RunTimeErrorSubclass) RaiseAction <RunTimeErrorSubclass> (RunTimeErrorSubclass (), __LINE__, __FILE__)
#define ERROR_LOCATOR " (module " << __FILE__ << ", line " << __LINE__ << ") "
#define ERROR(x) ers << x << Throw
#define Error(C) ers << ThrowEx (C)
#if !defined (NOCONSTEST)
    #define CONSISTENCY_TEST(x) { if (!(x)) ers << "Consistency test failed for (" #x ")" << Throw; }
#else
    #define CONSISTENCY_TEST(x)
#endif


// standard error strings
#include "common_errors.h"


#endif // __runtime_error_h__
