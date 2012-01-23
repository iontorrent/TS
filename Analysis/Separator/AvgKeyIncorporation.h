/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef AVGKEYINCORPORATION_H
#define AVGKEYINCORPORATION_H

class AvgKeyIncorporation {
 public:
  virtual ~AvgKeyIncorporation() {}
  virtual float  *GetAvgKeySig (int region_num, int rStart, int rEnd, int cStart, int cEnd) = 0;
  virtual double  GetAvgKeySigLen () = 0;
  virtual int     GetStart(int region_num, int rStart, int rEnd, int cStart, int cEnd) = 0;
};

#endif // AVGKEYINCORPORATION_H
