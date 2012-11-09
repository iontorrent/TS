/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include "Mask.h"
#include "Image.h"
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "Utils.h"
#include "HandleExpLog.h"
#include "OptArgs.h"
#include "IonErr.h"
#include "ChipIdDecoder.h"

using namespace std;

int hasElementOverThreshold ( float threshold, vector<float> &v )
{
  for ( size_t i = 0; i < v.size(); i++ ) {
    if ( v[i] >= threshold ) {
      return 1;
    } else if ( v[i] <= -1 * threshold ) {
      return -1;
    }
  }
  return 0;
}

int findNextNucFlow ( int nucIx, const string &flowOrder )
{
  size_t n = 100000;
  for ( n = nucIx+1; n < flowOrder.length(); n++ ) {
    if ( flowOrder[n] == flowOrder[nucIx] ) {
      break;
    }
  }
  return n;
}

class WellSummary
{

  public:

    WellSummary() {
      mMinMatch = 0;
      mMaxMatch = 0;
    }

    WellSummary ( const char *_match6, const char *_match8,
                  int minMatch, int maxMatch ) {
      mMatch6 = char2Vec<int> ( _match6, '*' );
      mMatch8 = char2Vec<int> ( _match8, '*' );
      mMinMatch = minMatch;
      mMaxMatch = maxMatch;
    }


    void AddData ( const std::vector<float> &data, int mult ) {
      double peakLoc = 0;
      float intLoc=0;
      if ( data.size() > frames.size() ) {
        frames.resize ( data.size() );
      }
      for ( size_t frameIx = 0; frameIx < data.size(); frameIx++ ) {
        peakLoc = max ( peakLoc, static_cast<double>(fabs ( mult * data[frameIx] ) ));
        intLoc += mult * data[frameIx];
        frames[frameIx].AddValue ( mult * data[frameIx] );
      }
      integral.AddValue ( intLoc );
      peak.AddValue ( peakLoc );
    }

    void MatchVectors ( int &mult, int &match, int &negMatch,
                        vector<int> &multVec,
                        Mask &mask,
                        vector<int> &matchVec,
                        const vector<int> &neighbors,
                        const vector<int> &aggressors,
                        int minMatch, int maxMatch ) {
      int matches = -1;
      mult = 0;
      match  = -1;
      negMatch = 0;
      multVec.resize ( 2 );
      multVec[0] = 0;
      multVec[1] = 0;
      for ( size_t i = 0; i < neighbors.size(); i++ ) {
        if ( mask[neighbors[i]] & MaskPinned ) {
          match = -1;
          mult = 0;
          return;
        }
        if ( aggressors[neighbors[i]] == -1 ) {
          match = -1;
          mult = 0;
          return;
        }
        if ( ( matchVec[i] != 0 && aggressors[neighbors[i]] != 0 ) ) {
          matches++;
          if ( aggressors[neighbors[i]] == 1 ) {
            multVec[0]++;
          } else {
            multVec[1]++;
          }
          // else (aggressors[neighbors[i]] == -1) {
          //     multVec[1]++;
          //   }
        } else if ( matchVec[i] == 0 && aggressors[neighbors[i]] == 0 ) {
          negMatch++;
        }
      }
      matches += negMatch;
      if ( ( matches >= minMatch && matches <= maxMatch ) &&
           ( multVec[0] == 0 || multVec[1] == 0 ) ) {
        if ( multVec[0] != 0 ) {
          mult = 1;
        } else {
          mult = -1;
        }
        match = matches;
      }

    }

    virtual bool PatternMatch ( int &matches,
                                int &mult,
                                Mask &mask,
                                vector<int> &match6,
                                vector<int> &match8,
                                const vector<int> &neighbors,
                                const vector<int> &aggressors ) {
      vector<int> multVec ( 2 );
      int negMatch = 0;
      if ( neighbors.size() == 6 ) {
        MatchVectors ( mult, matches, negMatch, multVec, mask, match6, neighbors, aggressors,mMinMatch,mMaxMatch );
      } else if ( neighbors.size() == 8 ) {
        MatchVectors ( mult, matches, negMatch, multVec, mask, match8, neighbors, aggressors, mMinMatch, mMaxMatch );
      } else {
        ION_ABORT ( "Can't have neighbor length: " + ToStr ( neighbors.size() ) );
      }
      return matches >= 0;
    }

    virtual bool Match ( int &mult, Mask &mask,
                         const vector<int> &neighbors,
                         const vector<int> &aggressors ) {
      int matches = 0;
      return PatternMatch ( matches, mult, mask, mMatch6, mMatch8, neighbors, aggressors );
    }

    void Print ( const std::string &prefix, ostream &out ) {
      out << prefix << "\t" << GetCount();
      for ( size_t i = 0; i < frames.size(); i++ ) {
        out << "\t" << frames[i].GetMean();
      }
      out << endl;
    }

    int GetCount() {
      if ( frames.size() == 0 ) {
        return 0;
      }
      return frames[0].GetCount();
    }

    int mMinMatch, mMaxMatch;
    vector<int> mMatch6;
    vector<int> mMatch8;
    vector<int> emptyNeighbors;
    SampleStats<float> integral;
    vector<SampleStats<float> > frames;
    SampleStats<float> peak;
};

void usage()
{
  cout << "MeasureXTalk - Use matching flows with same nucs to look at the crosstalk" << endl;
  cout << "between wells with signal and empty wells." << endl;
  cout << "" << endl;
  cout << "usage:" << endl;
  cout << "  MeasureXTalk --mask-file mask.bin --results-dir /path/to/dats --out-prefix out \\" << endl;
  cout << "     --start-row 500 --start-col 500 --width 100 --height 100" << endl;
  exit ( 1 );
}

int main ( int argc, const char *argv[] )
{

  OptArgs opts;
  string rawData;
  string maskFile;
  string output;
  bool help;
  string flowOrder = "TACGTACGTCTGAGCATCGATCGATGTACAGCTACGTACGTCTGAGCATCGATCGATGTACAGCTACGTACGTCTGAGCATCGATCGATGTACAGC";
  int startFlow  = 9;
  int endFlow = 24;
  int startRow, startCol, width, height;
  double threshold = 50;
  opts.ParseCmdLine ( argc, argv );
  opts.GetOption ( rawData, "", '-', "results-dir" );
  opts.GetOption ( maskFile, "", '-', "mask-file" );
  opts.GetOption ( output, "", '-', "out-prefix" );
  //                         01234567890123456789
  opts.GetOption ( flowOrder, "TACGTACGTCTGAGCATCGATCGATGTACAGCTACGTACGTCTGAGCATCGATCGATGTACAGCTACGTACGTCTGAGCATCGATCGATGTACAGC", '-', "flow-order" );
  opts.GetOption ( help, "false", 'h', "help" );
  opts.GetOption ( startRow, "0", '-', "start-row" );
  opts.GetOption ( startCol, "0", '-', "start-col" );
  opts.GetOption ( height, "-1", '-', "height" );
  opts.GetOption ( width, "-1", '-', "width" );
  opts.GetOption ( threshold, "50", '-', "theshold" );
  opts.CheckNoLeftovers();
  if ( rawData.empty() || output.empty() || help ) {
    usage();
  }
  vector<WellSummary> victims ( 8 );
  vector<WellSummary> aggNeighbors ( 9 );
  WellSummary aggressor;
  WellSummary random ( "000000","00000000", 0, 0 );

  WellSummary upstreamOnly ( "000100","00001000", 1, 1 );
  WellSummary downstreamOnly; //("100000","10000000", 1, 1);
  //  WellSummary downstreamOnlyTest("100000","10000000", 8, 8);
  WellSummary twoUpOnly ( "001110", "00011100", 2, 2 );
  Mask mask ( maskFile.c_str() );
  if ( width == -1 ) {
    width = mask.W();
  }
  if ( height == -1 ) {
    height = mask.H();
  }
  int numWells = mask.W() * mask.H();
  vector<vector<float> > diff ( numWells );
  char *explog_path = MakeExpLogPathFromDatDir(rawData.c_str());
  char *chipType = GetChipId ( explog_path );
  if (explog_path) free (explog_path);
  ChipIdDecoder::SetGlobalChipId ( chipType );
  size_t numFrames = 80;
  ImageTransformer::CalibrateChannelXTCorrection ( rawData.c_str(),"lsrowimage.dat" );

  vector<int> empties;
  for ( int rowIx  = startRow; rowIx < startRow + height; rowIx++ ) {
    for ( int colIx = startCol; colIx < startCol + width; colIx++ ) {
      int wellIx = mask.ToIndex ( rowIx, colIx );
      if ( mask[wellIx]  == MaskEmpty ) {
        empties.push_back ( wellIx );
      }
    }
  }

  vector<int> numAggressors ( ( endFlow - startFlow ), 0 );

  for ( int flowIx = startFlow; flowIx < endFlow; flowIx++ ) {
    int nextFlow = findNextNucFlow ( flowIx, flowOrder );
    char buff[2048];
    snprintf ( buff, sizeof ( buff ), "%s/acq_%04d.dat", rawData.c_str(), flowIx );
    Image img1;
    img1.LoadRaw ( buff, 0, true, false );
    int normStart = img1.GetFrame ( -663 ); //5
    int normEnd = img1.GetFrame ( 350 ); //20
    img1.FilterForPinned ( &mask, MaskEmpty );
    img1.SetMeanOfFramesToZero ( normStart, normEnd );
    // correct in-channel electrical cross-talk
    // img1.XTChannelCorrect(&mask);
    ImageTransformer::XTChannelCorrect ( img1.raw,img1.results_folder );

    numFrames  = img1.GetUnCompFrames();
    //mean += img->GetInterpolatedValue(frameIx,colIx,rowIx);

    snprintf ( buff, sizeof ( buff ), "%s/acq_%04d.dat", rawData.c_str(), nextFlow );
    Image img2;
    img2.LoadRaw ( buff, 0, true, false );
    img2.FilterForPinned ( &mask, MaskEmpty );
    img2.SetMeanOfFramesToZero ( normStart, normEnd );
    // correct in-channel electrical cross-talk
    // img2.XTChannelCorrect(&mask);
    ImageTransformer::XTChannelCorrect ( img2.raw,img2.results_folder );

    //    int numFrames = img1.GetUnCompFrames();
    //    numFrames = 80;
    int row, col;

    // Get the differences between two frames
    for ( int wellIx = 0; wellIx < numWells; wellIx++ ) {
      if ( mask[wellIx] & MaskPinned ) {
        continue;
      }
      diff[wellIx].resize ( numFrames );
      mask.IndexToRowCol ( wellIx, row, col );
      for ( size_t frameIx = 0; frameIx < numFrames; frameIx++ ) {
        float v1 = img1.GetInterpolatedValue ( frameIx,col,row );
        float v2 = img2.GetInterpolatedValue ( frameIx,col,row );
        diff[wellIx][frameIx] = v1 - v2;
      }
    }


    // Add up the aggressor wells

    vector<int> neighbors;
    vector<int> aggressors ( numWells, 0 );
    /* Find the aggressors and mark them. */
    for ( int rowIx  = startRow; rowIx < startRow + height; rowIx++ ) {
      for ( int colIx = startCol; colIx < startCol + width; colIx++ ) {
        int wellIx = mask.ToIndex ( rowIx, colIx );
        //    for (int wellIx = 0; wellIx < numWells; wellIx++) {
        if ( mask[wellIx] & MaskPinned ) {
          continue;
        }
        mask.IndexToRowCol ( wellIx, row, col );
        aggressors[wellIx] = hasElementOverThreshold ( threshold, diff[wellIx] );
      }
    }
    /* Loop through all the wells around the aggressors. */
    for ( int rowIx  = startRow; rowIx < startRow + height; rowIx++ ) {
      for ( int colIx = startCol; colIx < startCol + width; colIx++ ) {
        int wellIx = mask.ToIndex ( rowIx, colIx );
        //    for (int wellIx = 0; wellIx < numWells; wellIx++) {
        if ( mask[wellIx] & MaskPinned || mask[wellIx] & MaskExclude ) {
          continue;
        }
        mask.IndexToRowCol ( wellIx, row, col );
        int mult = 1;
        mult = hasElementOverThreshold ( threshold, diff[wellIx] );
        if ( mult != 0 ) {
          numAggressors[flowIx - startFlow]++;
          mask.GetNeighbors ( row, col, neighbors );

          /* Agressor well */
          aggressor.AddData ( diff[wellIx], mult );
          for ( size_t i = 0; i < neighbors.size(); i++ ) {
            if ( neighbors[i] < 0  || ! ( mask[neighbors[i]] & MaskEmpty ) || ( mask[neighbors[i]] & MaskPinned ) ) {
              continue;
            }
            victims[i].AddData ( diff[neighbors[i]], mult );
          }
        }
      }
    }

    vector<int> emptyNeighbors;
    int emptyRow, emptyCol;
    for ( size_t emptyIx = 0; emptyIx < empties.size(); emptyIx++ ) {
      if ( mask[empties[emptyIx]] & MaskPinned ) {
        continue;
      }
      mask.IndexToRowCol ( empties[emptyIx], emptyRow, emptyCol );
      mask.GetNeighbors ( emptyRow, emptyCol, emptyNeighbors );
      int nCount = 0;
      int aggMult = 0;
      for ( size_t n = 0; n < emptyNeighbors.size(); n++ ) {
        if ( emptyNeighbors[n] == -1 || mask[emptyNeighbors[n]] & MaskPinned ) {
          nCount = -1;
          break;
        }
        if ( emptyNeighbors[n] >= 0 && aggressors[emptyNeighbors[n]] != 0 ) {
          if ( aggMult == 0 ) {
            aggMult = aggressors[emptyNeighbors[n]];
          }
          if ( aggressors[emptyNeighbors[n]] == aggMult ) {
            nCount++;
            if ( nCount >= ( int ) aggNeighbors.size() ) {
              cout << "Why?" << endl;
            }
          } else {
            nCount = -1;
            n = emptyNeighbors.size();
            break;
          }
        }
      }
      if ( nCount == 0 ) {
        aggNeighbors[nCount].AddData ( diff[empties[emptyIx]], 1 );
      }
      if ( nCount > 0 ) {
        assert ( nCount < ( int ) aggNeighbors.size() && aggMult != 0 );
        aggNeighbors[nCount].AddData ( diff[empties[emptyIx]], aggMult );
      }
    }


    /* Single upstream */
    int upstreamOnlyCount = 0;
    for ( size_t emptyIx = 0; emptyIx < empties.size(); emptyIx++ ) {
      if ( mask[empties[emptyIx]] & MaskPinned ) {
        continue;
      }
      mask.IndexToRowCol ( empties[emptyIx], emptyRow, emptyCol );
      mask.GetNeighbors ( emptyRow, emptyCol, emptyNeighbors );
      int nCount = 0;
      if ( emptyNeighbors[3] < 0 ) {
        continue;
      }
      for ( size_t n = 0; n < emptyNeighbors.size(); n++ ) {
        if ( emptyNeighbors[n] >= 0 && aggressors[emptyNeighbors[n]] != 0 ) {
          nCount++;
        }
      }
      if ( nCount != 1 || ! ( aggressors[emptyNeighbors[3]] != 0 ) ) {
        continue;
      }
      int mult = aggressors[emptyNeighbors[3]];
      assert ( mult != 0 );
      upstreamOnlyCount++;
      upstreamOnly.AddData ( diff[empties[emptyIx]], mult );

    }
    cout << "Num upstream only: " << upstreamOnlyCount << endl;

    int downstreamOnlyCount = 0;
    for ( size_t emptyIx = 0; emptyIx < empties.size(); emptyIx++ ) {
      if ( mask[empties[emptyIx]] & MaskPinned ) {
        continue;
      }
      mask.IndexToRowCol ( empties[emptyIx], emptyRow, emptyCol );
      mask.GetNeighbors ( emptyRow, emptyCol, emptyNeighbors );
      int nCount = 0;
      if ( emptyNeighbors[0] < 0 ) {
        continue;
      }
      for ( size_t n = 0; n < emptyNeighbors.size(); n++ ) {
        if ( emptyNeighbors[n] >= 0 && aggressors[emptyNeighbors[n]] != 0 ) {
          nCount++;
        }
      }
      if ( nCount != 1 || ! ( aggressors[emptyNeighbors[0]] != 0 ) ) {
        continue;
      }
      int mult = hasElementOverThreshold ( threshold, diff[emptyNeighbors[0]] );
      assert ( mult != 0 );
      downstreamOnlyCount++;
      downstreamOnly.AddData ( diff[empties[emptyIx]], mult );
    }
    cout << "Num downstream only: " << downstreamOnlyCount << endl;

    // for (size_t emptyIx = 0; emptyIx < empties.size(); emptyIx++) {
    //  if (emptyIx == 30) {
    //    cout << "here we come." << endl;
    //  }
    //  mask.IndexToRowCol(empties[emptyIx], emptyRow, emptyCol);
    //  mask.GetNeighbors(emptyRow, emptyCol, emptyNeighbors);
    //  int mult = 0;
    //  if (downstreamOnlyTest.Match(mult, mask, neighbors, aggressors)) {
    //    downstreamOnlyTest.AddData(diff[empties[emptyIx]], mult);
    //  }
    // }

    int twoUpCount = 0;
    vector<int> multVec;
    for ( size_t emptyIx = 0; emptyIx < empties.size(); emptyIx++ ) {
      if ( mask[empties[emptyIx]] & MaskPinned ) {
        continue;
      }
      multVec.clear();
      mask.IndexToRowCol ( empties[emptyIx], emptyRow, emptyCol );
      mask.GetNeighbors ( emptyRow, emptyCol, emptyNeighbors );
      int nCount = 0;
      // if (emptyNeighbors[2] < 0 && emptyNeighbors[3] < 0 && emptyNeighbors[4] < 0) {
      //  continue;
      // }
      if ( emptyNeighbors[3] < 0 || emptyNeighbors[4] < 0 ) {
        continue;
      }
      for ( size_t n = 0; n < emptyNeighbors.size(); n++ ) {
        if ( emptyNeighbors[n] >= 0 && aggressors[emptyNeighbors[n]] != 0 ) {
          nCount++;
          multVec.push_back ( aggressors[emptyNeighbors[n]] );
        }
      }
      if ( nCount != 2 // || !(aggressors[emptyNeighbors[2]] != 0)
           || ! ( aggressors[emptyNeighbors[3]] != 0 )
           || ! ( aggressors[emptyNeighbors[4]] != 0 ) ) {
        continue;
      }
      int mult = multVec[0];
      int match = 0;
      for ( size_t m = 0; m < multVec.size(); m++ ) {
        if ( multVec[m] == mult ) {
          match++;
        }
      }
      if ( match != 2 ) {
        continue;
      }
      assert ( mult != 0 );
      twoUpCount++;
      twoUpOnly.AddData ( diff[empties[emptyIx]], mult );
    }
    cout << "Num 2 up only: " << twoUpCount << endl;


    int controlWells = 0;
    /* Random well. */
    for ( size_t emptyIx = 0; emptyIx < empties.size(); emptyIx++ ) {
      if ( mask[empties[emptyIx]] & MaskPinned ) {
        continue;
      }
      mask.IndexToRowCol ( empties[emptyIx], emptyRow, emptyCol );
      mask.GetNeighbors ( emptyRow, emptyCol, emptyNeighbors );
      int nCount = 0;
      for ( size_t n = 0; n < emptyNeighbors.size(); n++ ) {
        if ( emptyNeighbors[n] >= 0 && aggressors[emptyNeighbors[n]] ) {
          nCount++;
        }
      }
      if ( nCount > 0 ) {
        continue;
      }
      controlWells++;
      random.AddData ( diff[empties[emptyIx]], 1.0 );
    }
    cout << "Got: " << controlWells << " control wells." << endl;
  }
  cout << victims[2].integral.GetMean() << "\t" << victims[3].integral.GetMean() << endl;
  cout << victims[1].integral.GetMean() << "\t" << aggressor.integral.GetMean() << "\t" << victims[4].integral.GetMean() << endl;
  cout << victims[0].integral.GetMean() << "\t" << victims[5].integral.GetMean() << endl;

  cout << endl;
  cout << victims[2].peak.GetMean() << "\t" << victims[3].peak.GetMean() << endl;
  cout << victims[1].peak.GetMean() << "\t" << aggressor.peak.GetMean() << "\t" << victims[4].peak.GetMean() << endl;
  cout << victims[0].peak.GetMean() << "\t" << victims[5].peak.GetMean() << endl;

  cout << "Rand Peak and Int:\t" << random.peak.GetMean() << "\t" << random.integral.GetMean() << endl;
  cout << endl;
  string fileOut = output + ".summary-traces.txt";
  ofstream out;
  out.open ( fileOut.c_str() );
  aggressor.Print ( "aggr", out );
  for ( size_t v = 0; v < victims.size(); v++ ) {
    if ( victims[v].GetCount() > 50 ) {
      victims[v].Print ( ToStr ( "vic" ) + ToStr ( v ), out );
    }
  }

  random.Print ( "rand", out );
  upstreamOnly.Print ( "upst", out );
  downstreamOnly.Print ( "dnst", out );
  //  downstreamOnlyTest.Print("test:", cout);
  twoUpOnly.Print ( "2up", out );
  for ( size_t n = 0; n < aggNeighbors.size(); n++ ) {
    if ( aggNeighbors[n].GetCount() > 50 ) {
      aggNeighbors[n].Print ( "nbr" + ToStr ( n ), out );
    }
  }
  out.close();
  cout  << "Number:";
  for ( size_t x = 0; x < numAggressors.size(); x++ ) {
    cout <<  "\t" << numAggressors[x];
  }
  cout << endl;
}
