/* $Id: AeContainers.h,v 1.1.2.5 2009/12/01 18:03:03 hfeng Exp $ */
/**************************************************************************
 *
 *  Copyright (c) 1999-2009 Axeda Systems. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeContainers.h
 *  
 *  Subsystem  :  Axeda Agent Embedded
 *  
 *  Description:  Collections function declarations
 *
 **************************************************************************/

#ifndef _AE_CONTAINERS_H_
#define _AE_CONTAINERS_H_


#define AE_ARRAY_MIN_SIZE   16

/* Forward declarations for some structures */
typedef struct _AeList      AeList;
typedef struct _AeListItem  AeListItem;
typedef struct _AeArray     AeArray;
typedef struct _AePtrArray  AePtrArray;
typedef struct _AeBuffer    AeBuffer;

/* Compare function called by Lists and Arrays when item comparison is needed */
/* pData1 and pData2 are pointers of application added list items */
/* If NULL is provided for this callback function, no item comparison is performed */
typedef AeInt (*AeCompareFunc)(const AePointer pData1, const AePointer pData2);

/* Release function called by Lists and Arrays when item is to be freed */
/* pObject is the application added list item */
/* If NULL is provided for this callback function, item will not be properly freed and memory leak will happen */
typedef void (*AeReleaseFunc)(AePointer pObject);

/* AeListItem: item held by Lists */
struct _AeListItem
{
    AePointer   pData;
    AeListItem  *pNext;
    AeListItem  *pPrev;
};

/* AeList: for List */
struct _AeList
{
    AeCompareFunc   pCompareFunc;
    AeReleaseFunc   pReleaseFunc;
    AeListItem      *pFirstItem;
    AeListItem      *pLastItem;
    AeInt32         iCount;
};

/* AeArray: for Array */
struct _AeArray
{
    AeChar  *pData;
    AeInt32 iCount;
    AeInt32 iAllocSize;
    AeInt32 iElementSize;
};

/* AePtrArray: for Array of pointers */
struct _AePtrArray
{
    AeCompareFunc   pCompareFunc;
    AeReleaseFunc   pReleaseFunc;
    AePointer       *pData;
    AeInt32         iCount;
    AeInt32         iAllocCount;
};

/* AeBuffer: for Buffer */
struct _AeBuffer
{
    AeChar      *pData;
    AeInt32     iSize;
    AeInt32     iWritePos;
    AeInt32     iReadPos;
    AeBool      bExternalData;
    AeInt32     iChunkSize;
};

#ifdef __cplusplus
extern "C" {
#endif

/* Creates a new list to perform list manipulations */
/* pcompareFunc is callback function to perform item comparison. If NULL, no item comparison is performed */
/* pReleaseFunc is callback function to free the application added items */
/* If NULL, application added items will not be freed. Memory leaks! */
AeList      *AeListNew(AeCompareFunc pCompareFunc, AeReleaseFunc pReleaseFunc);

/* Destroys the list. It is the application's responsibility to free any created list by calling this function */
/* Once this call is executed, application callbacks will be called to free list items */
void        AeListDestroy(AeList *pList);

/* Applications should use AeListDestroy */
void        AeListFree(AeList *pList);

/* Appends the item to the end of the list. Returns the first item in the list */
AeListItem  *AeListAppend(AeList *pList, AePointer pData);

/* Prepends the item to the header of the list. Returns the first item in the list */
AeListItem  *AeListPrepend(AeList *pList, AePointer pData);

/* Inserts pData as a list item after the specified pItem. Returns the first item in the list */
AeListItem  *AeListInsertAfter(AeList *pList, AeListItem *pItem, AePointer pData);

/* Inserts pData as a list item before the specified pItem. Returns the first item in the list */
AeListItem  *AeListInsertBefore(AeList *pList, AeListItem *pItem, AePointer pData);

/* Removes the item from the list. Returns the first item in the list after removal is performed */
AeListItem  *AeListRemove(AeList *pList, AePointer pData);

/* Removes the item from the list. Returns the first item in the list after removal is performed */
AeListItem  *AeListRemoveItem(AeList *pList, AeListItem *pItem);

/* Finds the item from the list. Returns the item found or NULL */
AeListItem  *AeListFind(AeList *pList, AePointer pData);

/* Inserts pData as list item into the list sorted and returns the firt item in the list. If pcompareFunc is NULL, this degrades to Append */
AeListItem  *AeListInsertSorted(AeList *pList, AePointer pData, AeBool bAscending);

/* Accesses the application data from the list item. p is AeListItem pointer, t is application defined data type */
#define     AeListData(p, t) ((t *) (p)->pData)

/* Returns the first item in the list or NULL. p is the AeList pointer returned by AeListNew(), etc */
#define     AeListFirst(p) (p)->pFirstItem

/* Returns the last item in the list or NULL. p is the AeList pointer returned by AeListNew(), etc */
#define     AeListLast(p) (p)->pLastItem

/* Returns the next item in the list or NULL. p is the AeListItem pointer returned by ListLast, ListFirst, etc */
#define     AeListNext(p) (p)->pNext

/* Returns the previous item in the list or NULL. p is the AeListItem pointer returned by ListLast, ListFirst, etc */
#define     AeListPrev(p) (p)->pPrev

/* Returns the number of items in the list */
#define     AeListCount(p) (p)->iCount

/* Compares two pointers. Returns 0 if equal, 1 if not equal */
AeInt       AePointerCompare(AePointer pData1, AePointer pData2);

/*  The following creates a list and adds an item to it          */
/*  AeList  *pList = AeListNew(NULL, (AeReleaseFunc) MyDestroy); */
/*	AppData *pData = (AppData *)AeAlloc(sizeof(AppData));        */
/*	AeListAppend(pList, pData);                                  */


/*  The following is a typical list walker             */
/*  AeList      *pList                                 */
/*  AeListItem  *pItem                                 */
/*  AppData     *pData                                 */
/*  pItem = AeListFirst(pList);                        */
/*  while(pItem)                                       */
/*  {                                                  */
/*    pData = AeListData(pItem, AppData);              */
/*    if(pData)                                        */
/*    {                                                */
/*      do something here with pData                   */
/*    }                                                */
/*    pItem = AeListNext(pItem);                       */
/*  }                                                  */


/*  The following destroys a list and all the application items in it     */
/*  AeListDestroy(pList);                                                 */

/*	This is how the application frees its list items                      */
/*  void MyDestroy(AppData *pData)                                        */
/*  {                                                                     */ 
/*	  if(pData->pMember)                                                  */
/*		AeFree(pData->pMember);                                           */
/*    AeFree(pData);                                                      */
/*  }                                                                     */  


/* Creates a new array to perform array manipulations.  */
/* iElementSize is array element size in bytes. Use 1 for character arrays  */
AeArray     *AeArrayNew(AeInt32 iElementSize);

/* Destroys the array. It is the application's responsibility to free any created array by calling this function */
/* AeArrayDestroy has no callback mechanism to free application created data. Use PtrArray for that purpose */
void        AeArrayDestroy(AeArray *pArray);

/* Applications should use AeArrayGet() */
#define     AeArrayElementPtr(p, i) ((p)->pData + (i) * (p)->iElementSize)

/* Access the array item at specified index. p is AeArray pointer, i is the index */
#define     AeArrayGet(p, i) ((i) >= (p)->iCount ? NULL : AeArrayElementPtr((p), (i)))

/* Set item or items value at the index. pArray is AeArray pointer, iIndex is starting position */
/* pData is any application data. iCount is the number of elements to set */
/* This function will overwrite the original values or leave gaps between original and new values */
/* It is the application's responsibility to keep data integrity. */
void        AeArraySet(AeArray *pArray, AeInt32 iIndex, AePointer pData, AeInt32 iCount);

/* Insert item or items value at the index. pArray is AeArray pointer, iIndex is starting position */
/* pData is any application data. iCount is the number of elements to set */
/* This function expands the array to make room for the new items. It is possible to create gaps between original and new items */
void        AeArrayInsert(AeArray *pArray, AeInt32 iIndex, AePointer pData, AeInt32 iCount);

/* Append item or items value at end of array. pArray is AeArray pointer */
/* pData is any application data. iCount is the number of elements to set */
void        AeArrayAppend(AeArray *pArray, AePointer pData, AeInt32 iCount);

/* Returns the array element size */
#define     AeArrayElementSize(p, i) ((p)->iElementSize * (i))

/* Returns the items currently in the array */
#define     AeArrayCount(p) (p)->iCount

/* Sets the array to be empty. Note: AeArray does not free any application data. Use PtrArray for that purpose */
void        AeArrayEmpty(AeArray *pArray);

/* Creates a new pointer array to perform array manipulations.  */
/* AePtrArray is intended to hold any data pointers created by applications  */
/* pcompareFunc is callback function to perform item comparison. If NULL, no item comparison is performed */
/* pReleaseFunc is callback function to free the application added items */
/* If NULL, application added items will not be freed. Memory leaks! */
AePtrArray  *AePtrArrayNew(AeCompareFunc pCompareFunc, AeReleaseFunc pReleaseFunc);

/* Destroys the array. It is the application's responsibility to free any created array by calling this function */
/* Once this call is executed, application callbacks will be called to free array items */
void        AePtrArrayDestroy(AePtrArray *pArray);

/* Applications should use AePtrArrayDestroy */
void        AePtrArrayFree(AePtrArray *pArray);

/* Access the array item at specified index. p is AePtrArray pointer, i is the index */
#define     AePtrArrayGet(p, i) ((i) >= (p)->iCount ? NULL : (p)->pData[(i)])

/* Set item value at the index. pArray is AePtrArray pointer, iIndex is position, pData is application data. */
/* WARNING: This function will overwrite the original value or leave gaps between original and new values */
/*          This will either cause memory leaks or invalid data pointers. */
/*          It is the application's responsibility to keep data integrity. */
void        AePtrArraySet(AePtrArray *pArray, AeInt32 iIndex, AePointer pData);

/* Insert item value at the index. pArray is AePtrArray pointer, iIndex is position, pData is application data */
/* This function expands the array to make room for the new item if needed */
/* WARNING: It is possible that this function will create gaps between original and new items */
/*          It is the application's responsibility to keep data integrity. */
void        AePtrArrayInsert(AePtrArray *pArray, AeInt32 iIndex, AePointer pData);

/* Append item value at end of array. pArray is AePtrArray pointer, pData is any application data. */
/* This function will not cause gaps to happen, and is safe to use. */
void        AePtrArrayAppend(AePtrArray *pArray, AePointer pData);

/* Finds the item from the array. Returns the item found or NULL */
AePointer   AePtrArrayFind(AePtrArray *pArray, AePointer pData);

/* Returns the item count of the array */
#define     AePtrArrayCount(p) (p)->iCount


/* Creates a new AeBuffer to perform buffer manipulations.  */
/* AeBuffer is intended to perform read and write operations on the buffer  */
/* Like a file descriptor, it has read and write positions, both set at 0 to start with  */
/* At anytime, (Write-pos - Read-pos) bytes of data are considered pending and thus readable  */
/* pContent is data, iSize is data size, iChunkSize is growth size of data butter */
/* If pContent is not NULL AND iSize is non-zero, AeBuffer uses the application provided data directly */
/* WARNING: Application provided data will be freed if AeBuffer needs to re-alloc the data buffer,  */
/*          and there is no way for agentembedded to notify the application of this fact  */
/*          It is therefore recomnended that applications pass NULL for pContent and 0 for iSize when creating the AeBuffer */
AeBuffer    *AeBufferNew(AeChar *pContent, AeInt32 iSize, AeInt32 iChunkSize);

/* Destroys the AeBuffer and frees the data buffer.  */
/* It will not free application provided data. However, see WARNING above  */
/* It is the application's responsibility to free any data by calling this function  */
void        AeBufferDestroy(AeBuffer *pBuffer);

/* Writes the data to the buffer at current write position, then advances the write position accordingly  */
/* Returns the number of bytes written  */
AeInt32     AeBufferWrite(AeBuffer *pBuffer, AeChar *pData, AeInt32 iSize);

/* Performs AeBufferWrite() first, then writes \r\n, and advances the write position  */
/* Returns the total number of bytes written  */
AeInt32     AeBufferWriteLn(AeBuffer *pBuffer, AeChar *pData, AeInt32 iSize);

/* Read the specified number of bytes from the buffer at the current read position, then advances the read position */
/* If iSize is -1 or larger than pending size, all pending data is read out  */
/* Returns the number of bytes read. ppData is only set to the data pointer, applications MUST not free returned data  */
AeInt32     AeBufferRead(AeBuffer *pBuffer, AeChar **ppData, AeInt32 iSize);

/* Read a line from the data buffer, then advances the read position past the LF marker */
/* iSize is ignored, that is, the whole pending data is searched for the LF marker  */
/* Returns the number of bytes read. ppData is only set to the data pointer, applications MUST not free returned data  */
AeInt32     AeBufferReadLn(AeBuffer *pBuffer, AeChar **ppData, AeInt32 iSize);

/* Moves the write position in the data buffer, iOffset can be positive or negative. */
/* If bIsRelative is true, move point is the current write position. Else move point is the start position */
/* The write position will be co-erced into valid position after the move, that is, between 0 and buffer-size */
void        AeBufferSeekWritePos(AeBuffer *pBuffer, AeInt32 iOffset, AeBool bIsRelative);

/* Moves the read position in the data buffer, iOffset can be positive or negative. */
/* If bIsRelative is true, move point is the current read position. Else move point is the start position */
/* The read position will be co-erced into valid position after the move, that is, between 0 and write position */
/* In the current implementation, seeking read beyond the write position behaves save as AeBufferReset() */
void        AeBufferSeekReadPos(AeBuffer *pBuffer, AeInt32 iOffset, AeBool bIsRelative);

/* Returns the current read position and pending size. Applications MUST not free returned data */
void        AeBufferGetReadArea(AeBuffer *pBuffer, AeChar **ppData, AeInt32 *piSize);

/* Returns the current write position and write-able size. Applications MUST not free returned data */
void        AeBufferGetWriteArea(AeBuffer *pBuffer, AeChar **ppData, AeInt32 *piSize);

/* Expands the buffer size if needed. Applications should not call this function. AeBuffer manages this itself */
AeBool      AeBufferExpand(AeBuffer *pBuffer, AeInt32 iSize);

/* Resets the write position and read position back to 0 */
void        AeBufferReset(AeBuffer *pBuffer);

#ifdef __cplusplus
}
#endif

#endif
