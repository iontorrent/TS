/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef DUMPER_H
#define DUMPER_H

#include <iostream>
#include <cstring>
#include <vector>


//#define DO_CORRECTNESS_CHECK 

#define DUMPNAMELEN 128


using namespace std;

enum OutputFormat {
    MIN,
    FAIL,
    ALL
};


class DumpBuffer
{

char _name[DUMPNAMELEN];
char * _buffer;
size_t _sizeBytes;
size_t _writeOffset;  



public:

DumpBuffer(size_t sizeBytes, const char * name = NULL);
DumpBuffer(ifstream& myFile);

~DumpBuffer();

void cleanup();

size_t addData(void * data, size_t bytes);
size_t addCudaData(void * devData, size_t bytes);

void writeToFile(ofstream& oFile);
void readFromFile(ifstream& iFile);

void * getData();
size_t getSize();
const char * getName();

bool CompareName(DumpBuffer& buffer);
bool CompareSize(DumpBuffer& buffer);

// compares twe buffer objects for name, size and data
bool Compare(DumpBuffer& buffer, float threshold, OutputFormat output =MIN,
size_t length = 0, size_t stride = 1,size_t start = 0);  // compares twe buffer objects for name, size and data

// compares the data of two buffer objects (size not checked)
bool CompareData(DumpBuffer& buffer, float threshold, OutputFormat output
=MIN, size_t length = 0, size_t stride = 1,size_t start = 0);

// compares data in buffer object with data at pointer (size used from buffer object)
bool CompareData(float * data, float threshold, OutputFormat output =MIN, size_t
length = 0, size_t stride = 1,size_t start = 0); 

bool CompareData(short int * data, float threshold = 0, OutputFormat output =MIN, size_t
length = 0, size_t stride = 1,size_t start = 0); 

bool CompareData(int * data, float threshold =0 , OutputFormat output =MIN, size_t
length = 0, size_t stride = 1,size_t start = 0); 



// compares data in buffer object with data on device (size from data object)
bool CompareCuda(float * devData, float threshold, OutputFormat output =MIN); 

static bool CompareCuda(float * devData, float * hostData, size_t size, float threshold, OutputFormat output =MIN);

size_t PrintHeader();
size_t Print();
size_t Print(size_t length , size_t stride,size_t start=0, size_t max=0);


};
///////////////////////////////////////////////////////////////////////////////////

class DumpFile
{

    
 vector<DumpBuffer*> _buffers;
    

 public: 
 
  DumpFile() {};
  ~DumpFile();

  size_t addBuffer(DumpBuffer * buffer);
  size_t writeToFile(const char * filename);
  size_t readFromFile(const char * filename);  
  size_t getNumBuffers();
  
  DumpBuffer* getBuffer(int id);
  DumpBuffer* getBuffer(const char * buffername);

  void Compare(DumpFile& DF, float threshold, OutputFormat output = MIN );
  void Compare(DumpBuffer& buffer, float threshold,OutputFormat output = MIN );
 
  void printContent(); 
  void printContent(int id, size_t length=0 , size_t stride=1 ,size_t start=0);
  void printContent(const char * buffername, size_t length=0 , size_t stride=1 ,size_t start=0); 

};


#endif // DUMPER_H
