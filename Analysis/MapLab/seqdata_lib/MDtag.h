/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __mdtag_h__
#define __mdtag_h__

#include <cstddef>
#include <vector>
#include <cctype>

#include <platform.h>
#include <myassert.h>
#include <tracer.h>

// instruments for operating with BAM file MD tags

class MD
{
public:
    enum Operation
    {
        match = 0,
        replacement = 1
    };
    class Component
    {
    public:
        Operation type_  :  1;  // substitution or match
        bool      pastd_ :  1;  // substitution is past deletion mark ('^')
        BYTE      chr_   :  5;  // the replacement character value (less ('A' - 1))
        size_t    count_ :  25;
        Component (size_t count)
        :
        type_   (match),
        pastd_  (false),
        chr_    (0),
        count_  (count)
        {
            myassert (count < (1<<25));
        }
        Component (char chr, bool pasted)
        :
        type_   (replacement),
        pastd_  (pasted),
        chr_    (toupper (chr) - ('A' - 1)),
        count_  (1)
        {
            // myassert (toupper (chr) < ((1<<5) + ('A' - 1)));
            myassert (toupper (chr) < ((1<<5) + ('A' - 1)));
        }
#if 0        
        Component (const Component& oth)
        :
        type_  (oth.type_),
        pastd_ (oth.pastd_),
        chr_   (oth.chr_),
        count_ (oth.count_)
        {
            // memcpy (this, &oth, sizeof (Component));
            // operator = (oth);
        }
        Component& operator = (const Component& oth)
        {
            type_  = oth.type_,
            pastd_ = oth.pastd_,
            chr_   = oth.chr_,
            count_ = oth.count_;
            // memcpy (this, &oth, sizeof (Component));
        }
#endif
        char chr (char orig = -1) const
        {
            myassert (orig >= 0 || type_ == replacement);
            return (type_ == match)?orig:(chr_ + ('A' - 1));
        }
        bool pasted () const
        {
            return pastd_;
        }
        friend std::ostream& operator << (std::ostream&, const Component&);
    };
    Component operator [] (size_t i) const
    {
        return components_ [i];
    }
    size_t size () const 
    { 
        return components_.size (); 
    }
    void parse (const char* mdstring);
    MD ()
    {
    }
    MD (const char* mdstring)
    {
        parse (mdstring);
    }
protected:
    std::vector<Component> components_;
friend class MDIterator;
friend std::ostream& operator << (std::ostream&, const MD&);
friend Logger& operator << (Logger&, const MD&);
};

std::ostream& operator << (std::ostream&, const MD::Component&);
std::ostream& operator << (std::ostream&, const MD&);

inline Logger& operator << (Logger& l, const MD& md)
{
    if (l.enabled ())
        l.o_ << md;
    return l;
}

class MDIterator
{
    const MD* md_;
    size_t pos_;   // position in reference sequence 
    size_t idx_;   // index of the MD element in MD tag
    size_t off_;   // offset in current MD element (can be non-zero only for 'match' elements longer then 1 
    bool advance ()
    {
        size_t curcount = 0;
        size_t mdsize = md_->size ();
        myassert (idx_ <= mdsize);
        if (idx_ != mdsize) 
        {
            curcount = (*md_)[idx_].count_;
            do 
            {
                myassert (off_ <= curcount)
                if (off_ == curcount)
                {
                    off_ = 0;
                    ++ idx_;
                }
                else
                    ++ off_;
            }
            while (idx_ != mdsize && off_ == (curcount = (*md_)[idx_].count_));
        }
        ++ pos_;
        return (idx_ != mdsize);
    }
public:
    MDIterator (const MD& md)
    :
    md_ (&md),
    pos_ (0),
    idx_ (0),
    off_ (0)
    {
    }
    MDIterator (const MDIterator& copy)
    :
    md_ (copy.md_),
    pos_ (copy.pos_),
    idx_ (copy.idx_),
    off_ (copy.off_)
    {
    }
    MDIterator ()
    :
    md_ (NULL),
    pos_ (0),
    idx_ (0),
    off_ (0)
    {
    }
    MDIterator& operator = (const MDIterator& copy)
    {
        md_ = copy.md_;
        pos_ = copy.pos_;
        idx_ = copy.idx_;
        off_ = copy.off_;
        return *this;
    }
    MD::Component operator * () const
    {
        if ((*md_) [idx_].type_ == MD::match)
            return MD::Component ((size_t) 1);
        else
            return (*md_) [idx_];
    }
    MDIterator operator ++ (int)
    {
        MDIterator retcopy = *this;
        advance ();
        return retcopy;
    }
    MDIterator& operator ++ ()
    {
        advance ();
        return *this;
    }
    bool done () const
    {
        return (!md_) || idx_ == (md_->size ());
    }
    size_t pos () const
    {
        return pos_;
    }
};


#endif // __mdtag_h__
