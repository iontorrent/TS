/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

/* Author: Alex Artyomenko <aartyomenko@cs.gsu.edu> */

#include "OrderedBAMWriter.h"


bool alignment_left_after_right (Alignment *lhs, Alignment *rhs)
{
  if (lhs == rhs)
    return false;

  if (!rhs->processed)
    return false;
  if (!lhs->processed)
    return true;

  if (!rhs->alignment.IsMapped())
    return false;
  if (!lhs->alignment.IsMapped())
    return true;

  if (lhs->alignment.RefID > rhs->alignment.RefID)
    return true;
  if (lhs->alignment.RefID < rhs->alignment.RefID)
    return false;

  return lhs->alignment.Position > rhs->alignment.Position;
}


bool AlignmentComporator::operator()(Alignment *lhs, Alignment *rhs) {
  return alignment_left_after_right(lhs, rhs);
}

bool OrderedBAMWriter::cond(Alignment *current) {
  if (!current)
    return !empty();

  if (current == top())
    return false;

  if (!top()->processed)
    return false;
  if (!current->processed)
    return true;

  if (!top()->alignment.IsMapped())
    return false;
  if (!current->alignment.IsMapped())
    return true;

  if (current->alignment.RefID > top()->alignment.RefID)
    return true;
  if (current->alignment.RefID < top()->alignment.RefID)
    return false;

  return current->original_position > top()->alignment.Position;

  /*
  return current ? 1 == cmp_pairs(top()->alignment.RefID, top()->alignment.Position,
                                  current->alignment.RefID, current->original_position) : !empty();
                                  */
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
