/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DBGMEM_H
#define DBGMEM_H

#ifdef _DEBUG

#define malloc(a) dbgmalloc((a),__FILE__,__LINE__)
#define free(a) dbgfree((a),__FILE__,__LINE__)
#define realloc(ptr,a) dbgrealloc(ptr,(a),__FILE__,__LINE__)
#define strdup(a) dbgstrdup((a),__FILE__,__LINE__)

void* operator new(size_t sz, const char *file, int line);
void* operator new[] (size_t sz, const char *file, int line);

#define new new(__FILE__,__LINE__)
#define delete if(dbginfo(__FILE__,__LINE__)) delete

#define memdump() dbgmemDump(__FILE__,__LINE__)
#define memcheck() dbgmemCheck(__FILE__,__LINE__)

bool dbginfo(const char *file, int line);
void *dbgmalloc(size_t sz, const char *file, int line);
void dbgfree(void *ptr, const char *file, int line);
void *dbgrealloc(void *ptr, size_t sz, const char *file, int line);
char *dbgstrdup(const char *s1, const char *file, int line);
void dbgmemDump(const char *file, int line);
void dbgmemCheck(const char *file, int line);
void dbgmemInit();
void dbgmemClose();
void dbgmemDisableCheck();
void dbgmemEnableCheck();

#else

#define dbgmemInit()
#define memdump()
#define memcheck()

#endif /* _DEBUG */

#endif // DBGMEM_H

