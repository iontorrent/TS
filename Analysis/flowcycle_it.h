/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOWCYCLE_IT_H
#define FLOWCYCLE_IT_H

#include <string>

class flowcycle_it {
public:
	// cycle without end over letters in seq:
	flowcycle_it(const std::string& seq) : _seq(seq), _curr(_seq.begin()), _flow(0) {}
	
	inline long len()  const {return _seq.length();}
	inline long flow() const {return _flow;}
	inline char base() const {return *_curr;}
	inline void next()       {++_curr; if(_curr==_seq.end()) _curr=_seq.begin(); ++_flow;}
	
private:
	const std::string           _seq;
	std::string::const_iterator _curr;
	long                        _flow;
};

#endif //FLOWCYCLE_IT_H

