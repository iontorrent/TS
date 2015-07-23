/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __RESOURCE_H__
#define __RESOURCE_H__

#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include "portability.h"
#include "rerror.h"
#include "tracer.h"

class FileWrapper
{
    FILE* f_;
    mutable bool controlled_;
public:

    FileWrapper (const char* fname, const char* mode, bool control = true)
    {
        f_ = fopen (fname, mode);
        controlled_ = control;
    }
    FileWrapper (FILE* f, bool control = true)
    :
    f_ (f),
    controlled_ (control)
    {
    }
    FileWrapper ()
    :
    f_ (NULL),
    controlled_ (true)
    {
    }
    FileWrapper (const FileWrapper& from)
    :
    f_ (from.f_),
    controlled_ (from.controlled_)
    {
        from.controlled_ = false;
    }
    ~FileWrapper ()
    {
        free ();
    }
    FileWrapper& operator = (const FileWrapper& from)
    {
        free ();
        f_ = from.f_;
        controlled_ = from.controlled_;
        from.controlled_ = false;
        return *this;
    }
    FileWrapper& operator = (FILE* nf)
    {
        free ();
        f_ = nf;
        controlled_ = true;
        return *this;
    }
    bool operator ! () const
    {
        return (f_ == NULL);
    }
    void free ()
    {
        if (controlled_ && f_)
        {
            fclose (f_);
            f_ = NULL;
        }
    }
    FILE* release ()
    {
        controlled_ = false;
        return f_;
    }
    FILE* operator* ()
    {
        return f_;
    }
};
class FhandleWrapper
{
    int f_;
    mutable bool controlled_;
public:

    FhandleWrapper (const char* fname, int oflags, mode_t omode = 0, bool control = true)
    {
        f_ = ::sci_open (fname, oflags, omode);
        controlled_ = control;
    }
    FhandleWrapper (int f, bool control = true)
    :
    f_ (f),
    controlled_ (control)
    {
    }
    FhandleWrapper ()
    :
    f_ (-1),
    controlled_ (true)
    {
    }
    FhandleWrapper (const FhandleWrapper& from)
    :
    f_ (from.f_),
    controlled_ (from.controlled_)
    {
        from.controlled_ = false;
    }
    ~FhandleWrapper ()
    {
        free ();
    }
    FhandleWrapper& operator = (const FhandleWrapper& from)
    {
        free ();
        f_ = from.f_;
        controlled_ = from.controlled_;
        from.controlled_ = false;
        return *this;
    }
    FhandleWrapper& operator = (int nf)
    {
        free ();
        f_ = nf;
        controlled_ = true;
        return *this;
    }
    bool operator ! () const
    {
        return (f_ == -1);
    }
    void free ()
    {
        if (controlled_ && f_)
        {
            ::sci_close (f_);
            f_ = -1;
        }
    }
    int release ()
    {
        controlled_ = false;
        return f_;
    }
    int operator* ()
    {
        return f_;
    }
};

template <class ValueType> class MemWrapper
{
    ValueType* ptr_;
    mutable bool controlled_;
    void alloc (size_t size = 0)
    {
        if (size)
        {
            ptr_ = NULL;
            try
            {
                ptr_ = new ValueType [size];
            }
            catch (std::bad_alloc&)
            {
            }
            if (!ptr_) 
            {
                ers << "Unable to satisfy allocation request for " << size << " objects of " << sizeof (ValueType) << " bytes each" << ThrowEx(MemoryRerror);
            }
        }
    }
public:
    MemWrapper (size_t size, bool control = true)
    :
    ptr_ (NULL),
    controlled_ (control)
    {
        alloc (size);
    }
    MemWrapper (ValueType* ptr, bool control = true)
    :
    ptr_ (ptr),
    controlled_ (control)
    {
    }
    MemWrapper ()
    :
    ptr_ (NULL),
    controlled_ (false)
    {
    }
    MemWrapper (const MemWrapper& from)
    :
    ptr_ (from.ptr_),
    controlled_ (from.controlled_)
    {
        from.controlled_ = false;
    }
    ~MemWrapper ()
    {
        free ();
    }
    MemWrapper& operator = (const MemWrapper& from)
    {
        free ();
        ptr_ = from.ptr_;
        controlled_ = from.controlled_;
        from.controlled_ = false;
        return *this;
    }
    MemWrapper& operator = (ValueType* np)
    {
        free ();
        ptr_ = np;
        controlled_ = true;
        return *this;
    }
    void free ()
    {
        if (controlled_ && ptr_)
        {
            delete [] ptr_;
            ptr_ = NULL;
        }
    }
    void reset (size_t size = 0, bool control = true)
    {
        free ();
        controlled_ = control;
        alloc (size);
    }
    bool operator! ()
    {
        return ptr_ == NULL;
    }
    ValueType* release ()
    {
        controlled_ = false;
        return ptr_;
    }
    ValueType& operator* ()
    {
        return *ptr_;
    }
    ValueType* operator-> ()
    {
        return ptr_;
    }
//    ValueType& operator [] (size_t idx)
//    {
//        return ptr_ [idx];
//    }
//    ValueType& operator [] (int idx)
//    {
//        return ptr_ [(size_t) idx];
//    }
//    ValueType operator [] (size_t idx) const
//    {
//        return ptr_ [idx];
//    }
//    ValueType operator [] (int idx) const
//    {
//        return ptr_ [(size_t) idx];
//    }
    operator ValueType* ()
    {
        return ptr_;
    }
    operator const ValueType* () const
    {
        return ptr_;
    }
};

template <class ValueType> class ConstMemWrapper
{
    const ValueType* ptr_;
    mutable bool controlled_;
    void alloc (size_t size)
    {
        if (size)
        {
            try
            {
                ptr_ = new ValueType [size];
            }
            catch (std::bad_alloc&)
            {
            }
            if (!ptr_) 
                Error (MemoryRerror);
        }
    }
public:
    ConstMemWrapper (size_t size, bool control = true)
    :
    ptr_ (NULL),
    controlled_ (control)
    {
        alloc (size);
    }
    ConstMemWrapper (const ValueType* ptr, bool control = true)
    :
    ptr_ (ptr),
    controlled_ (control)
    {
    }
    ConstMemWrapper ()
    :
    ptr_ (NULL),
    controlled_ (false)
    {
    }
    ConstMemWrapper (const ConstMemWrapper& from)
    :
    ptr_ (from.ptr_),
    controlled_ (from.controlled_)
    {
        from.controlled_ = false;
    }
    ~ConstMemWrapper ()
    {
        free ();
    }
    ConstMemWrapper& operator = (const ConstMemWrapper& from)
    {
        free ();
        ptr_ = from.ptr_;
        controlled_ = from.controlled_;
        from.controlled_ = false;
        return *this;
    }
    ConstMemWrapper& operator = (const ValueType* np)
    {
        free ();
        ptr_ = np;
        controlled_ = true;
        return *this;
    }
    void free ()
    {
        if (controlled_ && ptr_)
        {
            delete [] (ValueType*) ptr_;
            ptr_ = NULL;
        }
    }
    void reset (size_t size, bool controlled)
    {
        free ();
        controlled_ = controlled;
        alloc (size);
    }
    bool operator! ()
    {
        return ptr_ == NULL;
    }
    const ValueType* release ()
    {
        controlled_ = false;
        return ptr_;
    }
    const ValueType& operator* ()
    {
        return *ptr_;
    }
    const ValueType* operator-> ()
    {
        return ptr_;
    }
    const ValueType& operator [] (size_t idx)
    {
        return ptr_ [idx];
    }
//    const ValueType& operator [] (int idx)
//    {
//        return ptr_ [(size_t) idx];
//    }
    ValueType operator [] (size_t idx) const
    {
        return ptr_ [idx];
    }
//    ValueType operator [] (int idx) const
//    {
//        return ptr_ [(size_t) idx];
//    }
    operator const ValueType* ()
    {
        return ptr_;
    }
    operator const ValueType* () const
    {
        return ptr_;
    }
};

template <class ValueType> class ObjWrapper
{
    ValueType* ptr_;
    mutable bool controlled_;
public:
    ObjWrapper (ValueType* ptr, bool control = true)
    :
    ptr_ (ptr),
    controlled_ (control)
    {
    }
    ObjWrapper ()
    :
    ptr_ (NULL),
    controlled_ (false)
    {
    }
    ObjWrapper (const ObjWrapper& from)
    :
    ptr_ (from.ptr_),
    controlled_ (from.controlled_)
    {
        from.controlled_ = false;
    }
    ~ObjWrapper ()
    {
        free ();
    }
    ObjWrapper &operator = (const ObjWrapper& from)
    {
        free ();
        ptr_ = from.ptr_;
        controlled_ = from.controlled_;
        from.controlled_ = false;
        return *this;
    }
    ObjWrapper &operator = (ValueType* np)
    {
        free ();
        ptr_ = np;
        controlled_ = true;
        return *this;
    }
    void free ()
    {
        if (controlled_ && ptr_)
        {
            delete ptr_;
            ptr_ = NULL;
        }
    }
    bool operator! ()
    {
        return ptr_ == NULL;
    }
    ValueType* release ()
    {
        controlled_ = false;
        return ptr_;
    }
    ValueType& operator* ()
    {
        return *ptr_;
    }
    ValueType* operator-> ()
    {
        return ptr_;
    }
    operator ValueType* ()
    {
        return ptr_;
    }
};


#endif
