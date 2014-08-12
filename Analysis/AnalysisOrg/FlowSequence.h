/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOWSEQUENCE_H
#define FLOWSEQUENCE_H

#include <map>
#include <string>

// Here's where we keep track of which flows we want to be working with.

// We'll be dealing with a block of flows.
// Such a block must be part of a sequence, so you can't really create one directly.
// In order to keep C++ passing all the data properly,
// the structure here is going to be a bit unusual.
// The basic idea, though, is that you get a FlowBlockSequenceData by pulling it out of a
// FlowBlockSequence's const_iterator.
class FlowBlockSequenceData {
protected:
    // We'll need an iterator into the parent's map. This gives us our base and repetition pattern.
    std::map< int, int >::const_iterator pattern;

    // We also need a counter to know how many repetitions we are into that pattern.
    int repetitions;

    // Our constructor is protected; only const_iterator can build one this way.
    FlowBlockSequenceData( const std::map< int, int >::const_iterator & p, int r );

public:
    // Which real flow # do we begin and end with?
    int begin() const;
    int end() const;        // Follows C++ convention; is the first flow *after* the block.

    // If you want to loop over all flows in a flow block, you probably want to go from 0 to this.
    size_t size() const;
};

// There's some sort of FlowBlockSequence object, which takes the user-defined input specification,
// and figures out what set of flows to work with.

class FlowBlockSequence {
  // At our simplest level, we have a bunch of flow blocks.
  // The number of blocks in a flow tends to be the same,
  // so we'll store an inherently ordered list of the starting flow and the number of flows in
  // each flow block.
  std::map< int, int >  flowBlocks;
public:
  // We have an iterator, which we'll be officially declaring in a moment.
  class const_iterator;

  // Restore defaults--the historic state of an infinite number of 20-flow blocks.
  void Defaults();

  // Try to parse a flow input string.
  // The format is [X:]Y,[Z:W,]...
  // First number in each pair is the starting flow.
  // Second number in each pair is the number of flows in each flow block.
  // This repeats until infinity, or until block "Z".
  // X must be 0.
  // Return true if we parsed it, false if we didn't.
  bool Set( const char * arg );

  // Convert our internal representation into a string.
  std::string ToString() const;

  // The first flow block is special. We often want to know whether or not
  // we should do something special for a flow, depending on whether or not
  // it's in the first block.
  bool HasFlowInFirstFlowBlock( int flow ) const;

  // Sometimes you allocate data based on the size of a flow block,
  // and you want to be sure you have a *global* maximum number of flows.
  int MaxFlowsInAnyFlowBlock() const;

  // How many different flow blocks are included from [begin,end)?
  int FlowBlockCount( int begin, int end ) const;

  // What's the 0-based numerical index of the flow block containing this particular flow?
  int FlowBlockIndex( int flow ) const;

  // We have some iterators, so that you can loop over our blocks.
  // Instead of begin/end, we have BlockAtFlow().
  // begin() would be BlockAtFlow(0), and there is no end().
  const_iterator BlockAtFlow( int flow ) const;

  // We need a structure to iterate over the flow blocks in a flow block sequence.
  // The FlowBlock parent is "protected", so that its methods don't appear in the const_iterator,
  // even though it's using the same data.
  class const_iterator : protected FlowBlockSequenceData {
    friend const_iterator FlowBlockSequence::BlockAtFlow( int ) const;

    // Only our friends can actually make one of us.
    const_iterator( const std::map< int, int >::const_iterator & p, int r );
  public:

    // Do we point to the same block?
    bool operator==( const const_iterator & that ) const;

    // Do we point to different blocks?
    bool operator!=( const const_iterator & that ) const;

    // Advance to the next block.
    const_iterator & operator++();

    // Access the underlying FlowBlock as if it were a separate object.
    const FlowBlockSequenceData * operator -> () const;
    const FlowBlockSequenceData * operator *  () const;
  };

};

#endif // FLOWSEQUENCE_H
