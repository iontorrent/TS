/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FlowSequence.h"
#include <iostream>
#include <sstream>
#include <stdio.h>

using namespace std;

void FlowBlockSequence::Defaults()
{
  // We just want a single entry in the map, with an infinite number of 20-flow blocks.
  flowBlocks.clear();
  flowBlocks[0] = 20;
  flowBlocks[ 0x7fffffff ] = 1;
}

bool FlowBlockSequence::Set( const char *arg )
{
  // Let's see what we can find...
  // First, we should have ###:#### or just ###
  int start = 0;
  int number = 0;
  bool firstWord = true;
  bool buildingStart = true;

  flowBlocks.clear();

  while( true ) {
    // Digits are good. They make the number bigger.
    if ( *arg >= '0' && *arg <= '9' ) {
      number = number * 10 + *arg - '0';
      ++arg;
      continue;
    }

    // Colon means that we save the start and move on to spacing.
    if ( *arg == ':' ) {
      // Only one colon per pair.
      if ( ! buildingStart ) return false;

      start = number;
      number = 0;
      buildingStart = false;
      ++arg;
      continue;
    }

    // Comma or end means that we need to save the number.
    if ( *arg == 0 || *arg == ',' ) {
      // Special case for first word. Add the missing 0:.
      if ( firstWord && buildingStart ) {
        start = 0;
        buildingStart = false;
      }

      // If we don't have a second number, that won't work.
      if ( buildingStart || number == 0 ) return false;

      // Does this sequence make sense?
      if ( firstWord ) {
        if ( start != 0 ) return false;
      }
      else {
        int prevStart = flowBlocks.rbegin()->first;
        int prevNumber = flowBlocks.rbegin()->second;

        // Make sure that this is bigger than the previous start.
        if ( start <= prevStart ) return false;

        // Make sure that the block starts at the end of some previous block.
        if ( (start - prevStart) % prevNumber != 0 ) return false;
      }

      // Save it.
      flowBlocks[ start ] = number;

      // See if we're done.
      if ( *arg == 0 ) break;

      // Reset for the next pass.
      start = number = 0;
      firstWord = false;
      buildingStart = true;
      ++arg;
      continue;
    }

    // Bad user. No cookie.
    return false;
  }

  // Add a tail, so that we never advance past the end.
  flowBlocks[ 0x7fffffff ] = 1;

  // Print out the sequence.
  cout << "Flow block sequence: " << ToString().c_str() << endl;

  return true;
}

string FlowBlockSequence::ToString() const
{
  stringstream str;
  for( map<int,int>::const_iterator it = flowBlocks.begin() ; it != flowBlocks.end() ; ++it ) {
    if ( it->first == 0x7fffffff ) break;
    if ( it != flowBlocks.begin() ) str << ',';
    str << it->first << ':' << it->second;
  }

  return str.str();
}

bool FlowBlockSequence::HasFlowInFirstFlowBlock( int flow ) const
{
  pair< int, int > firstFlowBlock = *flowBlocks.begin();

  return flow >= firstFlowBlock.first && flow < firstFlowBlock.first + firstFlowBlock.second;
}

int FlowBlockSequence::MaxFlowsInAnyFlowBlock() const
{
  // This is pretty easy... just accumulate the largest block length.
  int answer = 0;
  for( map<int,int>::const_iterator it = flowBlocks.begin() ; it != flowBlocks.end() ; ++it ) {
    answer = max( answer, it->second );
  }

  return answer;
}

int FlowBlockSequence::FlowBlockCount( int begin, int end ) const
{
  if ( begin >= end ) return 1;

  // Just count how many blocks we go through. Include the starting block.
  int count = 1;
  const_iterator endBlock = BlockAtFlow( end - 1 );
  for( const_iterator block = BlockAtFlow( begin ) ; block != endBlock ; ++block )
  {
    ++count;
  }

  return count;
}

int FlowBlockSequence::FlowBlockIndex( int flow ) const
{
  // We want to include the ending block, so we add one to the input... 
  // but we want a zero-based count, so we subtract one from the output.
  return FlowBlockCount( 0, flow + 1 ) - 1;
}

FlowBlockSequence::const_iterator FlowBlockSequence::BlockAtFlow( int flow ) const
{
  map< int, int >::const_iterator pattern = flowBlocks.upper_bound( flow );
  --pattern;
  int repetitions = ( flow - pattern->first ) / pattern->second;

  return FlowBlockSequence::const_iterator( pattern, repetitions );
}

FlowBlockSequenceData::FlowBlockSequenceData(
    const map< int, int >::const_iterator & p,
    int r
  ) :
  pattern( p ),
  repetitions( r )
{
}

FlowBlockSequence::const_iterator::const_iterator(
    const map< int, int >::const_iterator & p,
    int r
  ) :
  FlowBlockSequenceData( p, r )
{
}

bool FlowBlockSequence::const_iterator::operator==( 
    const FlowBlockSequence::const_iterator & that 
  ) const
{
  return this->pattern == that.pattern && this->repetitions == that.repetitions;
}

bool FlowBlockSequence::const_iterator::operator!=( 
    const FlowBlockSequence::const_iterator & that 
  ) const
{
  return this->pattern != that.pattern || this->repetitions != that.repetitions;
}

FlowBlockSequence::const_iterator & FlowBlockSequence::const_iterator::operator++()
{
  // Add another repetition.
  ++repetitions;

  // If we've reached the next pattern, advance to it.
  map< int, int >::const_iterator nextPattern = pattern;
  ++nextPattern;

  if ( pattern->first + repetitions * pattern->second == nextPattern->first ) {
    pattern = nextPattern;
    repetitions = 0;
  }

  return *this;
}

int FlowBlockSequenceData::begin() const
{
  return pattern->first + repetitions * pattern->second;
}

int FlowBlockSequenceData::end() const
{
  return pattern->first + (repetitions + 1) * pattern->second;
}

size_t FlowBlockSequenceData::size() const
{
  return pattern->second;
}

const FlowBlockSequenceData * FlowBlockSequence::const_iterator::operator -> () const
{
  return this;
}

const FlowBlockSequenceData * FlowBlockSequence::const_iterator::operator *  () const
{
  return this;
}
