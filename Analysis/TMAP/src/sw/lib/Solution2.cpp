/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdlib.h>
#include <cstring>
#include <sstream>
#include "../../util/tmap_alloc.h"
#include "Solution2.h"
#include <iostream>

// Ivan's Solution

using namespace std;

#define max(a, b) ((a)>(b)?a:b)

typedef struct Sol2_CacheEntry{
  int m;
  int n;
  Sol2_CacheEntry *next;
}Sol2_CacheEntry_t;
static Sol2_CacheEntry_t *Sol2_cachePtr=NULL;
pthread_mutex_t cacheMutex=PTHREAD_MUTEX_INITIALIZER;

static void FreeCachePtr(Sol2_CacheEntry_t *entry, int m, int n)
{
  pthread_mutex_lock(&cacheMutex);
  entry->m=m;
  entry->n=n;
  entry->next=Sol2_cachePtr;
  Sol2_cachePtr=entry;
  pthread_mutex_unlock(&cacheMutex);
}

static void *findCachePtr(int m, int n)
{
  void *rc=NULL;
  pthread_mutex_lock(&cacheMutex);
  Sol2_CacheEntry_t *entry=Sol2_cachePtr;
  Sol2_CacheEntry_t *prevEntry=NULL;
  while(entry){
    if(entry->m >= m && entry->n >= n){
      rc=entry;
      if(prevEntry)
        prevEntry->next=entry->next;
      else
        Sol2_cachePtr=entry->next;
      break;
    }
    prevEntry=entry;
    entry=entry->next;
  }
  pthread_mutex_unlock(&cacheMutex);
  return rc;
}

void Solution2::resize(int m, int n)
{
  _n = n;
  _m = m;
  if(n <= _alloc_n && m <= _alloc_m) return;

  int i;
  int old_n=_n;
  int old_m=_m;
  void *old_Malloc_Ptr=_Malloc_Ptr;
  unsigned int old_Malloc_m=_alloc_m;
  unsigned int old_Malloc_n=_alloc_n;


  _alloc_n = _n;
  _alloc_m = _m;
  _Malloc_Ptr = findCachePtr(_alloc_m,_alloc_n);
  if(_Malloc_Ptr==NULL){
    unsigned int newLen = _m*3*(sizeof(int*) + _n*sizeof(int));
    _Malloc_Ptr=malloc(newLen);
  }
  char *RPtr=(char *)_Malloc_Ptr;

  int **NM=(int **)RPtr; RPtr+= _m*sizeof(int*);
  int **NH=(int **)RPtr; RPtr+= _m*sizeof(int*);
  int **NV=(int **)RPtr; RPtr+= _m*sizeof(int*);
  for(i=0;i<_m;i++) {
	  NM[i]=(int *)RPtr; RPtr+= _n*sizeof(int);
	  NH[i]=(int *)RPtr; RPtr+= _n*sizeof(int);
	  NV[i]=(int *)RPtr; RPtr+= _n*sizeof(int);
  }
  if(old_Malloc_Ptr){
    FreeCachePtr((Sol2_CacheEntry_t*)old_Malloc_Ptr,old_Malloc_m,old_Malloc_n);
  }
  M=NM;
  H=NH;
  V=NV;
}

Solution2::Solution2()
{
  _m=0;
  _n=0;
  _alloc_m=0;
  _alloc_n=0;
  _Malloc_Ptr=NULL;
  M=V=H=NULL;
  max_qlen = 512;
  max_tlen = 1024;
}

Solution2::~Solution2()
{
  if(_Malloc_Ptr){
    FreeCachePtr((Sol2_CacheEntry_t*)_Malloc_Ptr,_alloc_m,_alloc_n);
    _Malloc_Ptr=NULL;
  }
  M=H=V=NULL;
}

int Solution2::process(const string& b, const string& a, int qsc, int qec,
                                     int mm, int mi, int o, int e, int dir,
                                     int *_opt, int *_te, int *_qe, int *_n_best, int* fitflag) {
    int n = b.size(), m = a.size();

    int id = -1;
    if (qsc == 1 && qec == 1) id = 1;
    if (qsc == 0 && qec == 1) id = 2;
    if (qsc == 1 && qec == 0) id = 3;
    if (qsc == 0 && qec == 0) id = 4;

    resize(m+1,n+1);

    if (id == 1 || id == 3) {
        for (int i=0; i <= m; i++){
        	M[i][0] = 0;
        	H[i][0] = V[i][0] = -INF;
        }
        for (int j=0; j <= n; j++){
        	M[0][j] = 0;
        	H[0][j] = V[0][j] = -INF;
        }

        for (int i=1; i <= m; i++)
          for (int j=1; j <= n; j++) {
              V[i][j] = max(M[i-1][j] + o + e, V[i-1][j] + e);
              H[i][j] = max(M[i][j-1] + o + e, H[i][j-1] + e);
              int mx = max(max(M[i-1][j-1], V[i-1][j-1]), H[i-1][j-1]);
              M[i][j] = max(0, mx + (a[i-1] == b[j-1] ? mm : mi));
          }
    } else {                                      
        for (int j=0; j <= n; j++) M[0][j] = 0;
        for (int i=1; i <= m; i++) M[i][0] = -INF;
        for (int j=0; j <= n; j++) H[0][j] = V[0][j] = -INF;
        for (int i=1; i <= m; i++) {
            V[i][0] = -INF;
            H[i][0] = o + e * i;
        }

        for (int i=1; i <= m; i++)
          for (int j=1; j <= n; j++) {
              V[i][j] = max(M[i-1][j] + o + e, V[i-1][j] + e);
              H[i][j] = max(M[i][j-1] + o + e, H[i][j-1] + e);
              int mx = max(max(M[i-1][j-1], V[i-1][j-1]), H[i-1][j-1]);
              M[i][j] = mx + (a[i-1] == b[j-1] ? mm : mi);
          }
    }

    int minI = 1, maxI = m, minJ = 1, maxJ = n;
    if (id == 3 || id == 4) minI = m;

    int opt = -INF, query_end = -1, target_end = -1, n_best = 0;

    for (int i=minI; i <= maxI; i++)
      for (int j=minJ; j <= maxJ; j++) {
          opt = max(opt, M[i][j]);
          opt = max(opt, V[i][j]);
          opt = max(opt, H[i][j]);
      }

    n_best = 0;
    for (int i=minI; i <= maxI; i++)
      for (int j=minJ; j <= maxJ; j++)
        if (M[i][j] == opt || V[i][j] == opt || H[i][j] == opt)
          n_best++;

    if (dir == 0) {
        for (int i=minI; i <= maxI && query_end == -1; i++)
          for (int j=maxJ; j >= minJ && query_end == -1; j--) // maximize target end
            if (M[i][j] == opt || V[i][j] == opt || H[i][j] == opt) {
                query_end = i-1;
                target_end = j-1;
            }
    } else {
        for (int i=maxI; i >= minI && query_end == -1; i--)
          for (int j=maxJ; j >= minJ && query_end == -1; j--)
            if (M[i][j] == opt || V[i][j] == opt || H[i][j] == opt) {
                query_end = i-1;
                target_end = j-1;
            }
    }

    (*_opt) = opt;
    (*_te) = target_end;
    (*_qe) = query_end;
    (*_n_best) = n_best;

    return opt;
}
