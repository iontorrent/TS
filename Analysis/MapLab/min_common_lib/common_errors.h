/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
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

#define MAKE_ERROR_TYPE(C,N) class C : public Rerror\
{\
public:\
    C (const char* s = "")\
    : Rerror (N)\
    {\
        if (*s)\
        {\
            msg_ += ": ";\
            msg_ += s;\
        }\
    }\
};

MAKE_ERROR_TYPE (MemoryRerror, ERR_NoMemory);
MAKE_ERROR_TYPE (InternalRerror, ERR_Internal);
MAKE_ERROR_TYPE (FileNotFoundRerror, ERR_FileNotFound);
MAKE_ERROR_TYPE (OutOfBoundsRerror, ERR_OutOfBounds);

#define INTERNAL_ERROR(x) ers << "Internal error: " << x << ERRINFO << ThrowEx(InternalRerror)
#define ERR_IFACE_CALL ers << "Error: Direct call of interface definition " << ERRINFO << ThrowEx(InternalRerror)

#endif

class OSRerror : public Rerror
{
    const char* get_err_str () const;
    const char* get_errno_str () const;
public:
    OSRerror (const char* s = "")
    : Rerror (ERR_OSError)
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
