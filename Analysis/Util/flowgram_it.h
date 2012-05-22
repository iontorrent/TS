/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOWGRAM_IT_H
#define FLOWGRAM_IT_H

#include <cctype>
#include <string>
#include "hp_iterator.h"
#include "flowcycle_it.h"

// generate an ideal (no noise, cafie or droop) ionogram from a sequence.

class flowgram_it {
public:
	typedef std::string::const_iterator seq_it;
	
	flowgram_it(const std::string& flow, const std::string& seq)
	: _flow(flow)
	, _hpit(seq.begin(), seq.end())
	, _base('!')
	, _hplen(0)
	, _good(_hpit.good())
	{
		next();
	}
	
	inline long flow()  const {return _flow.flow();}
	inline long hplen() const {return _hplen;}
	inline char base()  const {return _base;}
	inline bool good()  const {return _good;}
	
	inline void next()
	{
		if(std::tolower(_hpit.base()) == std::tolower(_flow.base())){
			_hplen = _hpit.len();
			_good  = _hpit.good();
			_hpit.next();
		}else{
			_hplen = 0;
			_good  = _hpit.good();
		}
		_base = _flow.base();
		_flow.next();
	}
	
private:
	typedef hp_iterator<std::string::const_iterator> hp_it;
	
	flowcycle_it _flow;
	hp_it        _hpit;
	char         _base;
	long         _hplen;
	bool         _good;
};

#endif //FLOWGRAM_IT_H

