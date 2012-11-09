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
unsigned int _sizeBytes;
unsigned int _writeOffset;  



public:

DumpBuffer(unsigned int sizeBytes, const char * name = NULL);
DumpBuffer(ifstream& myFile);

~DumpBuffer();

void cleanup();

unsigned int addData(void * data, unsigned int bytes);
unsigned int addCudaData(void * devData, unsigned int bytes);

void writeToFile(ofstream& oFile);
void readFromFile(ifstream& iFile);

void * getData();
unsigned int getSize();
const char * getName();

bool CompareName(DumpBuffer& buffer);
bool CompareSize(DumpBuffer& buffer);

bool Compare(DumpBuffer& buffer, float threshold, OutputFormat output =MIN,
int length = 0, int stride = 1,int start = 0);  // compares toe buffer objects for name, size and data
bool CompareData(DumpBuffer& buffer, float threshold, OutputFormat output
=MIN, int length = 0, int stride = 1,int start = 0); // compares the data of two buffer objects (size not checked)
bool CompareData(float * data, float threshold, OutputFormat output =MIN, int
length = 0, int stride = 1,int start = 0); // compares data in buffer object with data at pointer (size used from buffer object)

bool CompareCuda(float * devData, float threshold, OutputFormat output =MIN); // compares data in buffer object with data on device (size from data object)

static bool CompareCuda(float * devData, float * hostData, unsigned int size, float threshold, OutputFormat output =MIN);

unsigned int PrintHeader();
unsigned int Print();


};
///////////////////////////////////////////////////////////////////////////////////

class DumpFile
{

    
 vector<DumpBuffer*> _buffers;
    

 public: 
 
  DumpFile() {};
  ~DumpFile();

  int addBuffer(DumpBuffer * buffer);
  int writeToFile(const char * filename);
  int readFromFile(const char * filename);  
  int getNumBuffers();
  
  DumpBuffer* getBuffer(int id);
  DumpBuffer* getBuffer(const char * buffername);

  void Compare(DumpFile& DF, float threshold, OutputFormat output = MIN );
  void Compare(DumpBuffer& buffer, float threshold,OutputFormat output = MIN );
 
  void printContent(); 
  void printContent(int id);
  void printContent(const char * buffername); 

};


#endif // DUMPER_H
