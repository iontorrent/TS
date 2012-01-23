/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MASKSAMPLE_H
#define MASKSAMPLE_H

#include <cstdlib>
#include <vector>
#include "Mask.h"
#include "ReservoirSample.h"

template <class T>
class MaskSample {
public:
    // Given a Mask and a MaskType, generate a random sample chosen
    // from all wells of that type.
    MaskSample(Mask& mask, MaskType type, std::size_t sampleSize)
    : mSample(sampleSize)
	{
        T numWells = mask.W() * mask.H();
	    for(T well=0; well<numWells; ++well){
	        if(mask.Match(well, type))
	            mSample.Add(well);
	    }
	    mSample.Finished();
	}

    std::size_t     SampleSize() const {return mSample.GetCount();}
    std::vector<T>& Sample() {return mSample.GetData();}

private:
    MaskSample();
    MaskSample(const MaskSample&);
    MaskSample& operator=(const MaskSample&);

    ReservoirSample<T> mSample;
};

#endif // MASKSAMPLE_H

