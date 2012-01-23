/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TEMPLATESTACK_H
#define TEMPLATESTACK_H

#include <stdlib.h>
#include <stdio.h>

#ifndef DEFAULT_INITIAL_SIZE
#define DEFAULT_INITIAL_SIZE 16
#endif

//  TemplateStack is an array based stack which grows to
//  the necessary size.  The standard stack operations
//  are supported for Type.  Caution:: Nothing should
//  point to something you put in a TemplateStack if
//  you let the stack grow dynamically because
//  the dynamic sizing can cause something pointing
//  to an element of a TemplateStack to become invalid.
//  If you want to keep pointers to things in the stack,
//  make sure that resizing doesn't happen or keep indexes
//  into the stack instead of pointers.
//
//
// ********************************************************
//
//  See the html documentation in gui.html for detailed 
//  instructions on how to use this template.
//
// ********************************************************

template <class Type>
class TemplateStack {
 public:
  TemplateStack(int size = DEFAULT_INITIAL_SIZE)
    : currentSize(0) ,
      maxSize(size) ,
      arrayData ( (Type *) malloc(sizeof(Type) * maxSize) ) {
  }
  
  TemplateStack(TemplateStack const & originalStack)
    : currentSize(originalStack.currentSize) ,
      maxSize(originalStack.maxSize),
      arrayData( (Type *) malloc(sizeof(Type) * maxSize) ) 
  {
    int k = 0;
    while (k < currentSize)
      arrayData[k] = originalStack.arrayData[k++];
  }

  TemplateStack const & operator=(TemplateStack const & stackToCopy)
  {
    currentSize = stackToCopy.currentSize;
    maxSize = stackToCopy.maxSize;
    if (arrayData != NULL)
      free(arrayData);
    arrayData = (Type *) malloc(sizeof(Type) * maxSize);
    int  i;
    for (i = 0; i < stackToCopy.currentSize; i++) {
      arrayData[i] = stackToCopy.arrayData[i];
    }
    return *this;
  }
   
  ~TemplateStack()
  { // if arrayData hasn't been freed we must free it
    if (arrayData != NULL)
      free(arrayData);
  }

  void Destroy(void (*DestFunc)(Type) )
  //  DestFunc is called on each item in the stack, then arrayData is freed
  {
    int k=0;
    int j=currentSize;
    
    while ( k < j) 
      DestFunc(arrayData[k++]);
    free(arrayData);
    arrayData = NULL;
    currentSize = 0;
  }

  inline Type & operator[](int k) const
  {
    return arrayData[k];
  }
  
  void Push(Type newInfo)
  {
    if (currentSize == maxSize) {
      maxSize *= 2;
      arrayData = SpecialRealloc(arrayData,
				 sizeof(Type)*maxSize);
    }
    arrayData[currentSize] = newInfo;
    currentSize += 1;
  }    
  
  void Push(const TemplateStack<Type> & stackToAdd)
  {
    if ( currentSize + stackToAdd.Size() > maxSize ) {
      maxSize += stackToAdd.Size();
      arrayData = SpecialRealloc(arrayData,
				 sizeof(Type)*maxSize);
    }
    for (int i = 0, k = stackToAdd.Size(); i < k ;)
      arrayData[currentSize++] = stackToAdd[i++];
      
  }

  void Push(const TemplateStack<Type> * const stackToAdd)
  {
    if ( currentSize + stackToAdd->Size() > maxSize ) {
      maxSize += stackToAdd->Size();
      arrayData = SpecialRealloc(arrayData,
				 sizeof(Type)*maxSize);
    }
    Type * dataToAdd = stackToAdd->GetAddressOfArray();
    for (int i = 0, k = stackToAdd->Size(); i < k ;)
      arrayData[currentSize++] = dataToAdd[i++];
      
  }
  
  inline Type * Member(Type const & possibleMemberOfStack)
  {
    for (int k=0; k < currentSize; k++) 
      if (possibleMemberOfStack == arrayData[k])
	return &(arrayData[k]);
    return NULL;
  }

  inline int GetIndexOf(Type const & possibleMemberOfStack) const
  {
    for (int k=0; k < currentSize; k++) 
      if (possibleMemberOfStack == arrayData[k])
	return k;
    return -1;
  }
  
  inline Type Pop()
  {
    return(arrayData[--currentSize]);
  }
  
  inline Type * Top() const
  {
    return((&arrayData[(currentSize - 1)]));
  }
  
  inline Type * Bottom() const
  {
    return(&(arrayData[0]));
  }
  
  inline void DeleteTop()
  {
    currentSize--;
  }
  
  inline int Empty() const
  {
    return(currentSize == 0);
  }
  inline int NotEmpty() const
  {
    return(currentSize);
  }
  
  inline int Size() const 
  {
    return currentSize;
  }
  
  inline int IndexOfTop() const
  {
    return (currentSize-1);
  }

  inline Type & ItemAtTop() const 
  {
    return arrayData[currentSize-1];
  }
  
  inline int Capacity() const 
  {
    return maxSize;
  }
  
  inline void SwapTwoElements(int first , int second)
  {
    Type temp = arrayData[first];
    arrayData[first] = arrayData[second];
    arrayData[second] = temp;
  }
  
  inline Type * GetAddressOfArray() const { return arrayData; }
  
  inline void Clear() { currentSize = 0;} // clears the TemplateStack
  
  inline void Clear(void (*DestFunc)(Type ) ) 
  { // calls DestFunc on everything in the array and then clears it
    for (int k=0; k < currentSize; k++)
      DestFunc(arrayData[k]);
    currentSize = 0;
  }
  
  void InsertAtPosition(const  int position,
			const  int size,
			const Type * arrayToInsert)
  {
    if ((size + currentSize) >= maxSize) {
      maxSize += size;
      arrayData = SpecialRealloc(arrayData,
				 sizeof(Type)*maxSize);
    }
    for ( int k = currentSize+size-1, stoppingPoint=position+size-1 ;
	  k > stoppingPoint;k--)
      arrayData[k] = arrayData[(k-size)];
    for ( int k = position, j = 0; j < size;)
      arrayData[k++]=arrayToInsert[j++];
    currentSize+=size;
  }
  
  
  void InsertAtPosition(const int position, const Type itemToInsert)
  {
    if ((1+currentSize) == maxSize) {
      maxSize *= 2;
      arrayData = SpecialRealloc(arrayData,
				 sizeof(Type)*maxSize);
    } 
    for (int k = currentSize; k > position; ) 
      arrayData[k] = arrayData[--k];
    arrayData[position]=itemToInsert;
    currentSize++;
  }
  
  inline void RemoveAtIndex(int position)
  {
    if (position != --currentSize)
      arrayData[position] = arrayData[currentSize];
  }


  void DeleteAtPosition(int position) 
  {
    for (int k = position; k < (currentSize-1);)
      arrayData[k] = arrayData[++k];
    currentSize--;
  }
  
  void DeleteAtPosition(const int position, const int size)
  {
    for ( int k = position; k < currentSize - size + 1;)
      arrayData[k] = arrayData[k+=size];
    currentSize-=size;
  }
  
  inline void SetCurrentSize(const int newSize)
  {
    currentSize = newSize;
  }

  inline void ForEachItemDo(void (*function)(Type)) const
  {
    for (int i = 0; i < currentSize; i++)
      function(arrayData[i]);
  }

  inline void ForEachItemDo(void (*function)(Type, void * controller),
			    void *controller) const 
  {
    for (int i = 0; i < currentSize; i++)
      function(arrayData[i],controller);
  }
    
protected:
  int currentSize;
  int maxSize;
  Type * arrayData;

private:
  
  inline Type * SpecialRealloc(Type * arrayData, int size) 
  {
    if (NULL == (arrayData = (Type *) 
		 realloc(arrayData,size)))
      { 
      printf("realloc failed for TemplateStack from %s\n",__FILE__);
      printf("exiting\n"); 
	  exit(1);
    } 
    return(arrayData);
  }
};

#endif // TEMPLATESTACK_H
