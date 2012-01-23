/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Created on: 10/13/2010
//     Author: Keith Moulton
//
// Latest revision:   $Revision: 6876 $
// Last changed by:   $Author: keith.moulton@lifetech.com $
// Last changed date: $Date: 2010-10-15 06:54:32 -0700 (Fri, 15 Oct 2010) $

#ifndef TRIMMER_H
#define TRIMMER_H

#include <inttypes.h>

/*!
 Class used to trim low quality flows and adaptor sequence from SFF records
 */
class Trimmer
{
public:
	Trimmer() : qualThreshold(0) {}
	virtual ~Trimmer() {}

	void setQualThreshold(uint8_t threshold);
	uint8_t getQualThreshold() const {return qualThreshold;}

	void calculateQualClip(uint16_t &left, uint16_t &right);
	void calculateAdapterClip(uint16_t &left, uint16_t &right);

protected:
	uint8_t qualThreshold;
private:

};

#endif // TRIMMER_H
