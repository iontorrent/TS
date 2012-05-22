/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SEQUENCE_H
#define SEQUENCE_H

 #include <string>

 namespace ion {
	class Sequence {
	public:
	Sequence();
	Sequence(std::string sequence);
	~Sequence();


	std::string getSequence();

	void setSequence(std::string seq);

	int getType();

	int getSequenceLength();


	private:
	std::string sequence;
	int type;
	int stringset;

       };

}

#endif // SEQUENCE_H
