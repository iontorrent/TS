/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifdef _DEBUG

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <semaphore.h>
#include <inttypes.h>
#include "memsl.h"

MemSkipList memlist;
static const char *curDeleteFile = NULL;
static int curDeleteLine = 0;
static uint32_t bytesInUse = 0;
static sem_t dbgsem;
static bool dbgmemReady = false;
static bool dbgmemWantCheck = true;

#define GUARDVAL 0xdeadbeef

struct memitem {
	void *ptr; // pointer to the raw allocated ptr, user returned ptr is +4, this ptr needs to be first
	uint32_t sz;
	const char *file;
	int line;
};

bool dbginfo(const char *file, int line)
{
	curDeleteFile = file;
	curDeleteLine = line;
	return true;
}

static inline void checkGuard(memitem *item, const char *file, int line)
{
	uint32_t *guardStart = (uint32_t *)item->ptr;
	uint32_t *guardEnd = (uint32_t *)((unsigned char *)(item->ptr)+item->sz+4);
	if (*guardStart != GUARDVAL)
		printf("WARNING!  memory underrun from %s:%d found at %s:%d\n", item->file, item->line, file, line);
	if (*guardEnd != GUARDVAL)
		printf("WARNING!  memory overrun from %s:%d found at %s:%d\n", item->file, item->line, file, line);
}

void *dbgmalloc(size_t sz, const char *file, int line)
{
	if (dbgmemReady)
		sem_wait(&dbgsem);
	memitem *item = (memitem *)malloc(sizeof(memitem));
	item->sz = sz;
	item->file = file;
	item->line = line;
	item->ptr = malloc(item->sz+8);
	uint32_t *guardStart = (uint32_t *)item->ptr;
	uint32_t *guardEnd = (uint32_t *)((unsigned char *)(item->ptr)+sz+4);
	*guardStart = GUARDVAL;
	*guardEnd = GUARDVAL;
	memlist.insert(item, item->ptr);
	bytesInUse += sz;
	if (dbgmemReady)
		sem_post(&dbgsem);
	return (((unsigned char *)item->ptr)+4);
}

void dbgfree(void *ptr, const char *file, int line)
{
	if (dbgmemReady)
		sem_wait(&dbgsem);
	void *ptr2 = ((unsigned char *)ptr)-4;
	memitem *item = (memitem *)memlist.del(ptr2);
	if (item) {
		// printf("Deleted mem size: %d\n", item->sz);
		// check guard areas
		if (dbgmemWantCheck)
			checkGuard(item, file, line);
		bytesInUse -= item->sz;
		free(item->ptr);
		free(item);
	} else
		printf("ERROR!  mem del request not found from %s:%d\n", file, line);
	if (dbgmemReady)
		sem_post(&dbgsem);
}

void *dbgrealloc(void *ptr, size_t sz, const char *file, int line)
{
	if (ptr == NULL)
		return dbgmalloc(sz, file, line);

	if (dbgmemReady)
		sem_wait(&dbgsem);
	void *reallocptr = ptr;
	// get current item
	void *ptr2 = ((unsigned char *)ptr)-4; // true ptr is 4 bytes below user-level ptr
	memitem *item = (memitem *)memlist.find(ptr2);
	if (item) {
		if (item->sz < sz) { // new size request is too big, allocate a new block, copy, then delete old block
			// get a new mem block large enough for the realloc
			reallocptr = dbgmalloc(sz, item->file, item->line);
			// copy in old contents to new block
			memcpy(reallocptr, ptr, item->sz);
			// get rid of old memory block
			dbgfree(ptr, file, line);
		}
	}
	if (dbgmemReady)
		sem_post(&dbgsem);
	return reallocptr;
}

char *dbgstrdup(const char *s1, const char *file, int line)
{
	int len = strlen(s1) + 1;
	char *s2 = (char *)dbgmalloc(len, file, line);
	strcpy(s2, s1);
	return s2;
}

void* operator new(size_t sz)
{
	printf("Un-macro-ed debug new called?\n");
	// return dbgmalloc(sz, "", 0);
	return malloc(sz);
}

void* operator new[] (size_t sz)
{
	printf("Un-macro-ed debug new[] called?\n");
	// return dbgmalloc(sz, "", 0);
	return malloc(sz);
}

void* operator new(size_t sz, const char *file, int line)
{
	// printf("Here: new\n");
	return dbgmalloc(sz, file, line);
}

void* operator new[] (size_t sz, const char *file, int line)
{
	// printf("Here: new[]\n");
	return dbgmalloc(sz, file, line);
}

void operator delete (void *ptr)
{
	// printf("Here: delete\n");
	if (dbgmemReady)
		dbgfree(ptr, curDeleteFile, curDeleteLine);
	else
		free(ptr);
}

void operator delete[] (void *ptr)
{
	// printf("Here: delete[]\n");
	if (dbgmemReady)
		dbgfree(ptr, curDeleteFile, curDeleteLine);
	else
		free(ptr);
}

void dbgmemDump(const char *file, int line)
{
	if (dbgmemReady)
		sem_wait(&dbgsem);
	printf("%u bytes in use:\n", bytesInUse);
	memitem *item;
	for(memlist.toStart();(item = (memitem *)memlist.getNext()) != NULL;) {
		printf("%u bytes from %s:%d\n", item->sz, item->file, item->line);
		checkGuard(item, file, line);
	}
	if (dbgmemReady)
		sem_post(&dbgsem);
}

void dbgmemInit()
{
	printf("dbgmem init\n");

	sem_init(&dbgsem, 0, 1);
	dbgmemReady = true;
}

void dbgmemClose()
{
	dbgmemReady = false;
	sem_close(&dbgsem);
}

void dbgmemCheck(const char *file, int line)
{
	if (dbgmemReady)
		sem_wait(&dbgsem);
	memitem *item;
	for(memlist.toStart();(item = (memitem *)memlist.getNext()) != NULL;) {
		checkGuard(item, file, line);
	}
	if (dbgmemReady)
		sem_post(&dbgsem);
}

void dbgmemDisableCheck()
{
	dbgmemWantCheck = false;
}

void dbgmemEnableCheck()
{
	dbgmemWantCheck = true;
}

#endif /* _DEBUG */

#ifdef STANDALONE_TEST
struct AB {
	int a;
	int b;
};

int main(int argc, char *argv[])
{
	AB *ab = new AB;
	ab->a = 1;
	ab->b = 2;
	delete ab;
	delete ab;

	char *str = new char[5];
	str[0] = 0;
	str[5] = 0; // memory overrun
	str[-1] = 0; // memory underrun

	// see what mem is in use
	dumpInUse();

	delete [] str;

	// see what mem is in use
	dumpInUse();

	return 0;
}
#endif /* STANDALONE_TEST */

