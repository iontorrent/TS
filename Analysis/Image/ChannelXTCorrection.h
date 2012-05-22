/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CHANNELXTCORRECTION_H
#define CHANNELXTCORRECTION_H

#include <string.h>

struct ChannelXTCorrectionDescriptor {
    float **xt_vector_ptrs; // array of pointers to cross-talk correction vectors
    int num_vectors;        // number of vectors in xt_vector_ptrs
    int vector_len;         // length of each correction vector
    int *vector_indicies;   // relative indices for the application of each vector
};


/* class that describes a set of vectors used to correct for in-channel cross talk */
class ChannelXTCorrection {

public:

    ChannelXTCorrection(void)
    {
        xt_vector_storage = NULL;
        memset(&descr,0,sizeof(descr));
    }

    struct ChannelXTCorrectionDescriptor GetCorrectionDescriptor(void)
    {
        return descr;
    }

    float *AllocateVectorStorage(int num_vect,int vect_length)
    {
        xt_vector_storage = new float[num_vect * vect_length];
        descr.vector_len = vect_length;

        return(xt_vector_storage);
    }

    float **AllocateVectorPointerStorage(int num_ptrs)
    {
        descr.xt_vector_ptrs = new float *[num_ptrs];
        descr.num_vectors = num_ptrs;
        return(descr.xt_vector_ptrs);
    }

    void SetVectorIndicies(int *indicies,int vect_length)
    {
        descr.vector_len = vect_length;
        descr.vector_indicies = new int[vect_length];
        memcpy(descr.vector_indicies,indicies,sizeof(int[vect_length]));
    }

    ~ChannelXTCorrection(void)
    {
        if (xt_vector_storage != NULL)
            delete [] xt_vector_storage;
        
        if (descr.xt_vector_ptrs != NULL)
            delete [] descr.xt_vector_ptrs;

        if (descr.vector_indicies != NULL)
            delete [] descr.vector_indicies;
    }

private:
    float *xt_vector_storage;   // internal storage for vector data which may or may not be used
    struct ChannelXTCorrectionDescriptor descr;
};

#endif // CHANNELXTCORRECTION_H 

