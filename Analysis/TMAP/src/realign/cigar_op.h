/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CIGAR_OP_H
#define CIGAR_OP_H

struct CigarOp 
{

    char     Type;   //!< CIGAR operation type (MIDNSHPX=)
    unsigned Length; //!< CIGAR operation length (number of bases)

    //! constructor
    CigarOp (const char type = '\0', 
            const unsigned& length = 0)
    : 
    Type(type),
    Length(length) 
    {
    }
};


#endif // CIGAR_OP_H
