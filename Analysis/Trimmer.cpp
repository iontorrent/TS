/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Created on: 10/13/2010
//     Author: Keith Moulton
//
// Latest revision:   $Revision: 6876 $
// Last changed by:   $Author: keith.moulton@lifetech.com $
// Last changed date: $Date: 2010-10-15 06:54:32 -0700 (Fri, 15 Oct 2010) $
//

#include <iostream>
#include "Trimmer.h"

void Trimmer::setQualThreshold(uint8_t threshold)
{
	std::cout << "Trimmer: set quality threshold = " << threshold << std::endl;
	qualThreshold = threshold;
}

void Trimmer::calculateQualClip(uint16_t &left, uint16_t &right)
{
	left = 0;
	right = 0;
}

void Trimmer::calculateAdapterClip(uint16_t &left, uint16_t &right)
{
	left = 0;
	right = 0;
}
