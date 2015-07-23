/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDAEXCEPTION_H 
#define CUDAEXCEPTION_H

#include <iostream>
#include <exception>
#include "CudaDefines.h"


using namespace std;


class cudaException: public exception
{

protected:

  cudaError_t err;

public:

  cudaError_t getCudaError() { return err; };

  virtual const char* what() const throw()
  {
    Print();
    return "CUDA EXCEPTION: an Exception occured" ;
  }

    
  cudaException(cudaError_t err):err(err)
  {};

  virtual void Print() const throw() 
  {
    
   cout << " +----------------------------------------" << endl
        << " | ** CUDA ERROR! ** " << endl                               
        << " | Error: " << err << endl                             
        << " | Msg: " << cudaGetErrorString(err) << endl           
        << " +----------------------------------------" << endl << flush;  
  }

}; 




class cudaExceptionDebug: public cudaException
{

  const char * file;
  int line;

public:


  virtual const char* what() const throw()
  {
    Print();
    return "CUDA EXCEPTION: an Exception occured";
  }

    
  cudaExceptionDebug(cudaError_t err,  const char * file, int line):cudaException(err),file(file),line(line)
  {};

  virtual void Print() const throw() 
  {
    
   cout << " +----------------------------------------" << endl
        << " | ** CUDA ERROR! ** " << endl                               
        << " | Error: " << err << endl                             
        << " | Msg: " << cudaGetErrorString(err) << endl           
        << " | File: " << file << endl                         
        << " | Line: " << line << endl                         
        << " +----------------------------------------" << endl << flush;  
  }

}; 


class cudaStreamCreationError: public cudaExceptionDebug
{

  public:

  virtual const char* what() const throw()
  {
    //Print();
    return "CUDA EXCEPTION: could not acquire stream resources";
  }

  cudaStreamCreationError( const char * file, int line):cudaExceptionDebug(cudaErrorUnknown,file,line) {};

};


class cudaAllocationError: public cudaExceptionDebug
{

  public:

  virtual const char* what() const throw()
  {
    Print();
    return "CUDA EXCEPTION: could not allocate memory";
  }

  cudaAllocationError(cudaError_t err, const char * file, int line):cudaExceptionDebug(err,file,line) {};

};

class cudaNotEnoughMemForStream: public cudaExceptionDebug
{
public:
  virtual const char* what() const throw()
  {
    Print();
    return "CUDA EXCEPTION: Not enough memory for context and at least one stream!";
  }

  cudaNotEnoughMemForStream( const char * file, int line):cudaExceptionDebug(cudaErrorMemoryValueTooLarge,file,line) {};

};


class cudaExecutionException: public cudaExceptionDebug
{

  virtual const char* what() const throw()
  {
    Print();
    return "CUDA EXCEPTION: Error occured during job Execution!";
  }

 public:
  cudaExecutionException( cudaError_t err,  const char * file, int line):cudaExceptionDebug(err,file,line) {};

};

#endif  // CUDAEXCEPTION_H

