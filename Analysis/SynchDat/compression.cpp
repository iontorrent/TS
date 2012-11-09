/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * compression.cpp
 * Exported: filterThroughProcess/BytePacker/ByteUnpacker
 *  
 * @author Magnus Jedvert
 * @version 1.1 april 2012
*/

#include "compression.h"
#include <cassert>
#include <string>
#include <vector>
#include <sys/stat.h>
using namespace std;

/**
 * filterThroughProcess - Send stdin_ to command and store output in stdout_.
 * Will be used for zipping/unzipping. 
 * Can probably be made with pipes instead to avoid I/O.
 */
void filterThroughProcess(const string &command, const vector<u8> &stdin_, vector<u8> &stdout_) {
  //  char buff[2038];
  // Don't use this code until the temporary file creation is worked out
  assert(1==2);
  const string fileName = "hello"; // tmpnam(buff);  // get a random temporary file name
    
    // call command and store in temporary file:
    FILE *pstdin = popen( (command + " > " + fileName).c_str(), "w" );
    fwrite ( stdin_.data(), 1, stdin_.size(), pstdin );
    pclose(pstdin);

    // get number of bytes in file:
    struct stat statbuf;
    assert( stat(fileName.c_str(), &statbuf) != -1 );
    const size_t nBytes = statbuf.st_size;

    // read the file back:
    stdout_.resize( nBytes );
    FILE *fin = fopen(fileName.c_str(), "rb");
    assert( fread( stdout_.data(), 1, nBytes, fin ) == nBytes );
    fclose(fin);
    unlink( fileName.c_str() );
    remove(fileName.c_str());
}


BytePacker::BytePacker(vector<u8> &compressed_): compressed( compressed_ ), byteIdx(0) {
    compressed.resize(30 * 1000000); // reserve 30MB to start with
}

void BytePacker::push(const char *data, int len) {
    while ( byteIdx + len > compressed.size() ) {
        compressed.resize( compressed.size() * 2 );
    }
    memcpy( &compressed[byteIdx], data, len );
    byteIdx += len;
}

void BytePacker::finalize() {
    compressed.resize( byteIdx );  // shrink to fit
}


ByteUnpacker::ByteUnpacker(const char* compressed): ptr(compressed) {}

