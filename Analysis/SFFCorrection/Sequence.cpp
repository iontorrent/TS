/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

 #include <string>
 #include "Sequence.h"
 #include <stdio.h>
 namespace ion {

	Sequence::Sequence(){
	    type = 0;
	    stringset = 0;
	}

	Sequence::Sequence(std::string seq) {
	   sequence = seq;
	   type = 0;
	   stringset = 0;
	}

        Sequence::~Sequence() {
	}

	std::string Sequence::getSequence() {
		return sequence;
	}

	void Sequence::setSequence(std::string seq) {
		sequence = seq;
		fprintf(stdout, "setSequence seq = %s, sequence = %s \n", seq.c_str(), sequence.c_str());

	}

	int Sequence::getType() {
		return type;
	}

	int Sequence::getSequenceLength() {
		return sequence.length();
	}


}
