/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef DUMPER_H
#define DUMPER_H

#include <iostream>
#include <cstring>
#include <vector>
#include <map>
#include <typeinfo>

#define DUMP_VERSION 2

#define DUMPNAMELEN 128





using namespace std;

enum OutputFormat {
    MIN,
    FAIL,
    ALL
};

enum DataType {
  T_UNDEFINED = 0,
  T_INT,
  T_UINT,
  T_SHORT,
  T_USHORT,
  T_FLOAT,
  T_DOUBLE,
  T_CHAR
};

enum WrapType {
    COPY, // deep copy
    LINK, // only link data and mark locally as link
    LOCAL // take responsibility for data and make original buffer link to here.
};

class DumpBuffer
{

protected:
static map<DataType, string> typeNames;

char _name[DUMPNAMELEN];
char * _buffer;
size_t _sizeBytes;
size_t _writeOffset;  

bool _externData;

DataType _dtype;

virtual DataType determineType();

public:
static map<DataType, string> creatTypeMap()
{
    map<DataType,string> m;
    m[T_UNDEFINED] = "undefined";
    m[T_INT] = "int";
    m[T_UINT] = "unsigned int";
    m[T_SHORT] = "short";
    m[T_USHORT] = "unsigned short";
    m[T_FLOAT] = "float";
    m[T_DOUBLE] = "double";
    m[T_CHAR] = "char";
    return m;
};

DumpBuffer();
DumpBuffer(size_t sizeBytes, const char * name = NULL);
DumpBuffer(ifstream& myFile);
DumpBuffer(DumpBuffer& other, WrapType type = COPY);

virtual ~DumpBuffer();

void cleanup();

size_t addData(void * data, size_t bytes);
//size_t addCudaData(void * devData, size_t bytes);

void writeToFile(ofstream& oFile);
void readFromFile(ifstream& iFile);

void * getData();
size_t getSize();
const char * getName();
DataType getTypeInfo();
string getTypeString();

bool CompareName(DumpBuffer& buffer);
bool CompareSize(DumpBuffer& buffer);

virtual bool Compare(DumpBuffer& buffer, float threshold, OutputFormat output =MIN, size_t length = 0, size_t stride = 1,size_t start = 0);  // compares twe buffer objects for name, size and data
// compares the data of two buffer objects (size not checked) 
virtual bool CompareData(DumpBuffer& buffer, float threshold, OutputFormat output=MIN, size_t length = 0, size_t stride = 1,size_t start = 0);
virtual bool CompareData(void * data, float threshold, OutputFormat output =MIN, size_t length = 0, size_t stride = 1,size_t start = 0); 

//deep copy
virtual DumpBuffer & operator=(const DumpBuffer & other);

virtual size_t PrintHeader(); 
virtual size_t Print(); 
virtual size_t Print(size_t length , size_t stride,size_t start=0, size_t max=0);

void moveToOtherAndClear(DumpBuffer & other);

void externData(bool b);

};



template<typename T>
class DumpWrapper : public DumpBuffer 
{


protected: 
DataType determineType();

public:


DumpWrapper();
DumpWrapper(size_t sizeBytes, const char * name = NULL);
DumpWrapper(ifstream& myFile);
DumpWrapper( DumpBuffer& other, WrapType type = COPY ); //(low level copy constructor creats compelte copy of other

// not good practice but since I am only using it ;)
//DumpWrapper( DumpBuffer ** other); // copies the fields of other into the wrapper and repalces *other with a handle to the tempalted object  ;


size_t addData(T * data, size_t bytes);

// compares twe buffer objects for name, size and data
bool Compare(DumpBuffer& buffer, float threshold, OutputFormat output =MIN,
size_t length = 0, size_t stride = 1,size_t start = 0);  // compares twe buffer objects for name, size and data

// compares the data of two buffer objects (size not checked) 
bool CompareData(DumpBuffer& buffer, float threshold, OutputFormat output
=MIN, size_t length = 0, size_t stride = 1,size_t start = 0);

bool CompareData(void * data, float threshold, OutputFormat output =MIN, size_t
length = 0, size_t stride = 1,size_t start = 0); 

// compares data in buffer object with data on device (size from data object) 
//bool CompareCuda(float * devData, float threshold, OutputFormat output =MIN); 
//static bool CompareCuda(float * devData, float * hostData, size_t size, float threshold, OutputFormat output =MIN);
 
size_t PrintHeader(); 
size_t Print(); 
size_t Print(size_t length , size_t stride,size_t start=0, size_t max=0);
/*
DumpWrapper & operator=(const DumpBuffer & other);
DumpWrapper & operator=(const DumpWrapper & other);
*/
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
  
  DumpBuffer* getBuffer(unsigned int id);
  DumpBuffer* getBuffer(const char * buffername);

  void Compare(DumpFile& DF, float threshold, OutputFormat output = MIN, size_t length=0 , size_t stride=1 ,size_t start=0);
  void Compare(DumpBuffer& buffer, float threshold,OutputFormat output = MIN, size_t length=0 , size_t stride=1 ,size_t start=0);
 
  void printContent(); 
  void printContent(int id, size_t length=0 , size_t stride=1 ,size_t start=0);
  void printContent(const char * buffername, size_t length=0 , size_t stride=1 ,size_t start=0); 

};




#endif // DUMPER_H
