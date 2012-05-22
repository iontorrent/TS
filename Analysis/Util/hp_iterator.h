/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef HP_ITERATOR_H
#define HP_ITERATOR_H

// iterate over all the homopolymer runs in a sequence

template <class For>
class hp_iterator {
public:
	hp_iterator(For first, For last) : _beg(first), _end(first), _last(last) {next();}

	inline bool good() const {return _beg < _last;}
	inline void next()       {_beg=_end; while(*++_end == *_beg) ;}
	inline char base() const {return *_beg;}
	inline int  len()  const {return _end - _beg;}

private:
	hp_iterator();
	hp_iterator(const hp_iterator&);
	hp_iterator& operator=(const hp_iterator&);

	For _beg;   // begining of current homopolymer run
	For _end;   // end of current homopolymer
	For _last;  // end of sequence
};

#endif //HP_ITERATOR_H

