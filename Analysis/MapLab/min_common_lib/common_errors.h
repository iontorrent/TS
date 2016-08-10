/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#ifndef __common_errors_h__
#define __common_errors_h__

#ifndef __common_errors_cpp__

extern const char* ERR_NoMemory;
extern const char* ERR_Internal;
extern const char* ERR_FileNotFound;
extern const char* ERR_OSError;
extern const char* ERR_OutOfBounds;


// synonyms
#define NOEMEM ERR_NoMemory
#define INTERNAL ERR_Internal

#define MAKE_ERROR_TYPE(C,N) class C : public RunTimeError\
{\
public:\
    C (const char* s = "")\
    : RunTimeError (N)\
    {\
        if (*s)\
        {\
            msg_ += ": ";\
            msg_ += s;\
        }\
    }\
};

MAKE_ERROR_TYPE (MemoryError, ERR_NoMemory);
MAKE_ERROR_TYPE (InternalError, ERR_Internal);
MAKE_ERROR_TYPE (FileNotFoundError, ERR_FileNotFound);
MAKE_ERROR_TYPE (OutOfBoundsError, ERR_OutOfBounds);

#define INTERNAL_ERROR(x) ers << "Internal error: " << x << ERROR_LOCATOR << ThrowEx(InternalError)
#define ERR_IFACE_CALL ers << "Error: Direct call of interface definition placeholder function" << ERROR_LOCATOR << ThrowEx(InternalError)

#endif

class OSError : public RunTimeError
{
    const char* get_err_str () const;
    const char* get_errno_str () const;
public:
    OSError (const char* s = "")
    : RunTimeError (ERR_OSError)
    {
        msg_ += ": errno ";
        msg_ += get_errno_str ();
        msg_ += ": ";
        msg_ += get_err_str ();
        if (*s)
        {
            msg_ += ": ";
            msg_ += s;
        }
    }
};


#endif // __common_errors_h__
