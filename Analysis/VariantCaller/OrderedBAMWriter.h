/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

/* Author: Alex Artyomenko <aartyomenko@cs.gsu.edu> */

#ifndef ION_ANALYSIS_ORDEREDBAMWRITER_H
#define ION_ANALYSIS_ORDEREDBAMWRITER_H

#include <deque>
#include <queue>
#include <algorithm>
#include "BAMWalkerEngine.h"

using namespace std;

class Alignment;

template<typename T>
int cmp(T p1, T p2) { return p1 < p2 ? 1 : (p1 > p2 ? -1 : 0); }

template<typename T1, typename T2>
int cmp_pairs(T1 p11, T2 p12, T1 p21, T2 p22) { int lvl1 = cmp(p11, p21); if (lvl1) return lvl1; return cmp(p12, p22); }

struct AlignmentComporator {
  bool operator()(Alignment* lhs, Alignment* rhs);
};

typedef priority_queue<Alignment*, deque<Alignment*>, AlignmentComporator> alignment_queue;

class OrderedBAMWriter : protected alignment_queue {
  AlignmentComporator cmp;
public:
  OrderedBAMWriter() : alignment_queue(cmp) { }
  Alignment* process_new_etries(Alignment* list);
  Alignment* flush();
private:
  Alignment* wrap_items_in_linkedlist(Alignment * current);
  bool cond(Alignment* current);
};

#endif //ION_ANALYSIS_ORDEREDBAMWRITER_H
