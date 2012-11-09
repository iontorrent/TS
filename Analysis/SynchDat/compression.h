/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * compression.h
 * Exported: filterThroughProcess/BytePacker/ByteUnpacker
 *  
 * @author Magnus Jedvert
 * @version 1.1 april 2012
*/

#pragma once
#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <vector>
#include <armadillo>
#include <stdint.h>
using namespace std;
using namespace arma;

/**
 * filterThroughProcess - Send stdin_ to command and store output in stdout_.
 * Will be used for zipping/unzipping. 
 * Can probably be made with pipes instead to avoid I/O.
 */
void filterThroughProcess(const string &command, const vector<u8> &stdin_, vector<u8> &stdout_);


/**
 * BytePacker - Constructor takes a reference to a vector where the 
 * concatenated data will be stored. Only possible operations is to push data.
 * When done call finalize().
 */
class BytePacker {
public:
    BytePacker(vector<u8> &compressed_);

    // push len bytes pointed to by data
    void push(const char *data, int len);

    // wrapper for std::vectors
    template<typename T>
    void push(const vector<T> &data);

    // wrapper for armadillos Mat
    template<typename T>
    void push(const arma::Mat<T> &data);

    void finalize();  // call when done to finalize vector compressed

private:
    vector<u8> &compressed;
    size_t byteIdx;
};

/**
 * ByteUnpacker - Constructor takes a pointer to concatenated data.
 * Only operation possible is to pop data.
 */
class ByteUnpacker {
public:
    ByteUnpacker(const char* compressed);

    // wrapper to decompress std::vectors
    template<typename T>
    vector<T> pop(int n);

    // wrapper to decompress armadillos Mat, returns a pointer and updates internal ptr
    template<typename T>
    T *popPtr(int n);

private:
    const char* ptr;
};


// ====================== template implementation:
template<typename T>
void BytePacker::push(const vector<T> &data) {
    push( (char*)data.data(), sizeof(T) * data.size() );
}

template<typename T>
void BytePacker::push(const Mat<T> &data) {
    push( (char*)data.colptr(0), sizeof(T) * data.n_elem );
}


template<typename T>
vector<T> ByteUnpacker::pop(int n) {
    vector<T> res(n);

    const int len = sizeof(T) * n;
    memcpy( res.data(), ptr, len );
    ptr += len;

    return res;
}

template<typename T>
T* ByteUnpacker::popPtr(int n) {
    T *res = (T*)ptr;
    ptr += sizeof(T) * n;
    return res;
}

#endif // COMPRESSION_H

