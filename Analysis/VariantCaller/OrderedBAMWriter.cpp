/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

/* Author: Alex Artyomenko <aartyomenko@cs.gsu.edu> */

#include "OrderedBAMWriter.h"

bool AlignmentComporator::operator()(Alignment *lhs, Alignment *rhs) {
  return cmp_pairs(lhs->alignment.RefID, lhs->alignment.Position, rhs->alignment.RefID, rhs->alignment.Position) == -1;
}

bool OrderedBAMWriter::cond(Alignment *current) {
  return current ? 1 == cmp_pairs(top()->alignment.RefID, top()->alignment.Position,
                                  current->alignment.RefID, current->original_position) : !empty();
}

Alignment* OrderedBAMWriter::process_new_etries(Alignment *list) {
  if (!list) return NULL;
  Alignment * last = list;
  for (Alignment* current = list;current; last = current, current = current->next) {
    push(current);
  }
  return wrap_items_in_linkedlist(last);
}

Alignment* OrderedBAMWriter::flush() {
  return wrap_items_in_linkedlist(NULL);
}

Alignment* OrderedBAMWriter::wrap_items_in_linkedlist(Alignment * current) {
  if (empty()) return NULL;
  Alignment* result = NULL, *c = top();
  while (cond(current)) {
    if (!result) result = top();
    c = top();
    pop();
    c->next = top();
  }
  if (result) c->next = NULL;
  return result;
}