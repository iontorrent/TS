/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef LOADTRACESJOB_H
#define LOADTRACESJOB_H
#include <vector>
#include <string>
#include <stdio.h>
#include "PJob.h"
#include "Traces.h"
#include "Mask.h"
#include "Image.h"
#include "ReportSet.h"
#include "IonErr.h"

class LoadTracesJob : public PJob
{

  public:

    LoadTracesJob()
    {
      mMask = NULL;
      mTrace = NULL;
      mPrefix = NULL;
      mSuffix = NULL;
      mReportSet = NULL;
      mFrameZero  = -1;
      mFrameLast  = -1;
      mFirstDCFrame  = -1;
      mLastDCFrame  = -1;
      mIndex = -1;
      mIgnoreChecksumErrors = false;
    }

    void Init (Mask *mask, Traces *trace, std::string *prefix,
               std::string *suffix, ReportSet *reportSet,
               int frameZero, int frameLast,
               int firstDcFrame, int lastDcFrame,
               int index, int ignoreChecksumErrors)
    {
      mMask = mask;
      mTrace = trace;
      mPrefix = prefix;
      mSuffix = suffix;
      mReportSet = reportSet;
      mFrameZero = frameZero;
      mFrameLast = frameLast;
      mFirstDCFrame = firstDcFrame;
      mLastDCFrame = lastDcFrame;
      mIndex = index;
      mIgnoreChecksumErrors = ignoreChecksumErrors;
    }

    /** Process work. */
    virtual void Run()
    {
      char buff[mPrefix->size() + mSuffix->size() + 20];
      const char *p = mPrefix->c_str();
      const char *s = mSuffix->c_str();
      snprintf (buff, sizeof (buff), "%s%.4d%s", p, (int) mIndex, s);
      Image img;
      img.SetImgLoadImmediate (false);
      img.SetIgnoreChecksumErrors (mIgnoreChecksumErrors);
      bool loaded = img.LoadRaw (buff);
      if (!loaded)
      {
        ION_ABORT ("Couldn't load file: " + ToStr (buff));
      }
      mTrace->Init (&img, mMask, mFrameZero, mFrameLast, mFirstDCFrame, mLastDCFrame); //frames 0-75, dc offset using 3-12
      img.Close();
      mTrace->SetReportSampling (*mReportSet, false);
      /* // @todoo - are we using a common t0? */
      mTrace->CalcT0 (true);
      img.Close();
    }

    /** Cleanup any resources. */
    virtual void TearDown() {}

    /** Exit this pthread (killing thread) */
    void Exit()
    {
      pthread_exit (NULL);
    }

  private:
    Mask *mMask;
    Traces *mTrace;
    std::string *mPrefix;
    std::string *mSuffix;
    ReportSet *mReportSet;
    int mFrameZero, mFrameLast;
    int mFirstDCFrame, mLastDCFrame;
    int mIndex;
    int mIgnoreChecksumErrors;

};

#endif // LOADTRACESJOB_H
