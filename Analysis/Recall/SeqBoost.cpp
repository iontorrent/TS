/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <cassert>
#include <string>
#include <stdio.h>
#include <libgen.h>
#include <cmath>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

// Analysis Utils
#include "OptArgs.h"
#include "IonErr.h"

// file-io tools
#include "ion_alloc.h"
#include "ion_util.h"

// file-io sff
#include "sff.h"
#include "sff_file.h"
#include "sff_header.h"

//unmapped bam support
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/BamWriter.h"
#include "boost/scoped_array.hpp"

#include "json/json.h"

#include "IonVersion.h"

#include <sys/time.h>
#include <pthread.h>
#include </usr/include/semaphore.h>

#define DEBUG 0
#define debug_print(fmt, ...) \
            do { if (DEBUG) fprintf(stdout, fmt, __VA_ARGS__); } while (0)

#define SEQBOOST_VERSION "1.0.0"
#define DEAFAUL_QUALITY 30

using namespace std;
using namespace BamTools;

uint toInt(char nuc) {
  switch(nuc) {
  case 'A': return 0;
  case 'C': return 1;
  case 'G': return 2;
  default: return 3;
  }
}

uint toBase(int nuc) {
  switch(nuc) {
  case 1: return 'A';
  case 2: return 'C';
  case 3: return 'G';
  default: return 'T';
  }
}

struct Partition {
  string nuc;
  int flowStart;
  int flowEnd;
  int xMin;
  int xMax;
  int yMin;
  int yMax;
  Partition(string n, int fStart, int fEnd, int xi, int xx, int yi, int yx): nuc(n), flowStart(fStart), flowEnd(fEnd), xMin(xi), xMax(xx), yMin(yi), yMax(yx){}
};

struct Stratification{
  int flowStart;
  int flowEnd;
  int flowSpan;
  int flowCuts;
  int xMin;
  int xMax;
  int xSpan;
  int xCuts;
  int yMin;
  int yMax;
  int ySpan;
  int yCuts;
  Stratification(int fs, int fe, int f, int xi, int xx, int xs, int yi, int yx, int ys):flowStart(fs), flowEnd(fe), flowSpan(f), xMin(xi), xMax(xx), xSpan(xs), yMin(yi), yMax(yx), ySpan(ys){
      flowCuts = (flowEnd - flowStart + 1) / flowSpan;
      xCuts = (xMax - xMin + 1) / xSpan;
      yCuts = (yMax - yMin + 1) / ySpan;
  }
};

void saveJson(const Json::Value & json, const string& filename_json)
{
  ofstream out(filename_json.c_str(), ios::out);
  if (out.good())
    out << json.toStyledString();
  else
    cout << "Error: unable to write JSON file " << filename_json << endl;
}

const uint32_t MAX_BASES = 10000;
const int NUMTHREADS = 8;
sem_t semLoadNull[NUMTHREADS];
sem_t semLoadFull[NUMTHREADS];
sem_t semSaveNull[NUMTHREADS];
sem_t semSaveFull[NUMTHREADS];
pthread_mutex_t mutexQuit;
pthread_mutex_t mutexSave;
bool quit[NUMTHREADS];
int num2save = 0;
BamAlignment bamAlignmentIn[NUMTHREADS];
BamAlignment bamAlignmentOut[NUMTHREADS];

vector< vector<int> > hpsList;
vector< vector<double> > qvsList;
vector< vector<double> > maxQVs;
vector< vector<double> > minQVs;
vector< vector<int> > leftBoundaryLists;
vector< vector<int> > rightBoundaryLists;
vector<Partition> partitionList;
Stratification *stratification = NULL;

// flow QV Table
// First line contains flow signal
// Each line starts with nuc Type (A, C, G, T)
// Every nuc contains two lines, first of which corresponds to tailed homo-polymer call (left tailed is negative) and second of which corresponds to un-rounded QVs
// assume flow signal starts from 0 and increments by 1 for now; TODO: parse flow signal explicitly to create map
void parseFlowQVTable(string flowQVTableFile, bool rb) {
  ifstream flowQVReader;
  flowQVReader.open(flowQVTableFile.c_str());
  if(!flowQVReader){
      cout << "Error: file could not be opened; " << flowQVTableFile << endl;
      exit(1);
  }
  int MAX_HP = 20;
  string line;
  getline(flowQVReader,line); //skip first line f
  int offset = 1;
  if (line.compare(0, 9, "flowStart") == 0){
      offset = 7;
      getline(flowQVReader, line); //parse into stratification
      std::vector<std::string> sList;
      boost::split(sList, line, boost::is_any_of(","));

      stratification = new Stratification(boost::lexical_cast<int>(sList[0]),//flowStart
                                          boost::lexical_cast<int>(sList[1]),//flowEnd
                                          boost::lexical_cast<int>(sList[2]),//flowSpan
                                          boost::lexical_cast<int>(sList[3]),//xMin
                                          boost::lexical_cast<int>(sList[4]),//xMax
                                          boost::lexical_cast<int>(sList[5]),//xSpan
                                          boost::lexical_cast<int>(sList[6]),//yMin
                                          boost::lexical_cast<int>(sList[7]),//yMax
                                          boost::lexical_cast<int>(sList[8]));//ySpan

      getline(flowQVReader,line); //skip third line of header for QV table
  }

  //cout << strs.size() << "; " << strs[1] << endl;
  while ( getline(flowQVReader,line) ){
    std::vector<std::string> strs;
    boost::split(strs, line, boost::is_any_of(","));
    //partitionList
    if(offset == 7){
        Partition p(strs[0], boost::lexical_cast<int>(strs[1]), boost::lexical_cast<int>(strs[2]), boost::lexical_cast<int>(strs[3]), boost::lexical_cast<int>(strs[4]), boost::lexical_cast<int>(strs[5]), boost::lexical_cast<int>(strs[6]));
        partitionList.push_back(p);
    }

    //hp
    vector<int> hps(strs.size() - offset);
    for(unsigned int ind = offset; ind < strs.size(); ++ind){
        hps[ind - offset] = boost::lexical_cast<int>(strs[ind]);
    }
//    hpsList.push_back(hps);

    //QV
    if(!getline(flowQVReader, line)){
        cout << "Error: incomplete file (HP followed by QV); " << flowQVTableFile << endl;
        exit(1);
    }
    boost::split(strs, line, boost::is_any_of(","));
    vector<double> qvs(strs.size() - offset);
    for(unsigned int ind = offset; ind < strs.size(); ++ind){
        qvs[ind - offset] = boost::lexical_cast<double>(strs[ind]);
    }
    qvsList.push_back(qvs);

    //produce maxQVs and minQVs

    vector<double> maxQVList(MAX_HP, 0);
    vector<double> minQVList(MAX_HP, 100);

    int firstMissingInd = -1;

    for(unsigned int ind = 0; ind < qvs.size(); ++ind){
        if(qvs[ind] > 0){
            unsigned int hpTemp = hps[ind]>0?hps[ind]:(-hps[ind]);
            maxQVList[hpTemp] = maxQVList[hpTemp]<qvs[ind]?qvs[ind]:maxQVList[hpTemp];
            minQVList[hpTemp] = minQVList[hpTemp]>qvs[ind]?qvs[ind]:minQVList[hpTemp];
        } else if (firstMissingInd == -1){
            firstMissingInd = ind;
        }
    }

    maxQVs.push_back(maxQVList);
    minQVs.push_back(minQVList);
    vector<int> leftBoundaryList(MAX_HP, 2000);
    vector<int> rightBoundaryList(MAX_HP, 0);
    int totalSize = hps.size();
    for(int ind = 0; ind < totalSize; ++ind){
        if(qvs[ind] > 0){
            unsigned int hpTemp = hps[ind]>0?hps[ind]:(-hps[ind]);
            leftBoundaryList[hpTemp] = leftBoundaryList[hpTemp] > ind ? ind : leftBoundaryList[hpTemp];
            rightBoundaryList[hpTemp] = rightBoundaryList[hpTemp] < ind ? ind : rightBoundaryList[hpTemp];
        }
    }
    leftBoundaryList[0] = -49;

    //imputate
    bool startExtrapolation = false;
    for(int hp = 0; hp < MAX_HP; ++hp){
        if(hp > 7) //skip any correction for hp > 7
            startExtrapolation = true;
        if (startExtrapolation) {
            if(hp > 0){
                leftBoundaryList[hp] = 100 * hp - 49;
                rightBoundaryList[hp] = 100 * hp + 49;
            }
        } else if (firstMissingInd >= 100*hp - 49 && firstMissingInd <= 100 * hp + 49) {
            if(hp > 0)
                leftBoundaryList[hp] = rightBoundaryList[hp - 1] + 1;
            rightBoundaryList[hp] = 100 * hp + 49;
            startExtrapolation = true;
        }
    }

    //make sure monotonicity satisfication
    for(int hp = 0; hp < 8; ++hp){
        if(leftBoundaryList[hp+1] <= rightBoundaryList[hp] || leftBoundaryList[hp+1] - rightBoundaryList[hp] > 2 || rightBoundaryList[hp] - leftBoundaryList[hp] < 80){
            rightBoundaryList[hp] = (rightBoundaryList[hp+1] - leftBoundaryList[hp])/2 + leftBoundaryList[hp];
            leftBoundaryList[hp+1] = rightBoundaryList[hp] + 1;
        }
    }

    leftBoundaryLists.push_back(leftBoundaryList);
    rightBoundaryLists.push_back(rightBoundaryList);

    //update hps
    int hpTemp = 0;
    for(int ind = 0; ind < totalSize; ++ind){
        if(ind <= rightBoundaryList[hpTemp]){
          hps[ind] = hpTemp;
        } else {
          hpTemp++;
          hps[ind] = hpTemp;
        }
    }
    hpsList.push_back(hps);

	if(rb)
	{
		Json::Value json(Json::objectValue);
		char buf1[100];
		char buf2[100];
		for(int hp = 0; hp < 12; ++hp)
		{
			  sprintf(buf1, "%d", hp);
			  sprintf(buf2, "%d", rightBoundaryList[hp]);
			  json["RightBoundary"][buf1] = buf2;
		} 

		string filename_json("all_flow_whole_chip_");
		
		if(offset == 7)
		{		
			filename_json = "flow_";
			filename_json += strs[1];
			filename_json += "-";
			filename_json += strs[2];
			filename_json += "_y_";
			filename_json += strs[5];	
			filename_json += "-";
			filename_json += strs[6];	
			filename_json += "_x_";
			filename_json += strs[3];
			filename_json += "-";
			filename_json += strs[4];
			filename_json += "_";
		}

		filename_json += strs[0];
		filename_json += "_right_boundary.json";

		saveJson(json, filename_json);
	}
  }

//  for(unsigned int ind = 0; ind < std::min((size_t)4, leftBoundaryLists.size()); ++ind){
//      for(int hp = 0; hp < MAX_HP; ++hp){
//          fprintf(stdout, "%d  ", leftBoundaryLists[ind][hp]);
//      }
//      fprintf(stdout, "\n");

//          for(int hp = 0; hp < MAX_HP; ++hp){
//              fprintf(stdout, "%d  ", rightBoundaryLists[ind][hp]);
//          }
//          fprintf(stdout, "\n");
//  }

  if(stratification != NULL){
      if ((int)hpsList.size() != 4*stratification->xCuts * stratification->yCuts * stratification->flowCuts){
          cout << "Abort recalibration due to insufficient flow QV table " << flowQVTableFile << endl;
          exit(1);
      }
  }

  flowQVReader.close();
}

int recalibrate( bool flowOrderCheck, int maxFlowSignal, uint32_t &shorterReadCount, uint32_t &ignoreCorrectionCount, sff_t * sff, bool allocateNewMemory=true){

    uint16_t* flowgram = sff->read->flowgram;
    vector<int> newFlowIndices;
    vector<int> newFlowSignal(flowgram, flowgram + sff->gheader->flow_length); //fixed size
    string newBases;
    string newQualities;

    //identify flows that are not corrected, TODO: still need to handle flow signal
    //identify candidates of possible violation
    std::vector<uint32_t> ignoreCorrection(sff->gheader->flow_length, 0);
    int offsetRegion = 0;
    if (stratification != NULL){
      //get region index
      std::vector<std::string> strs;
      boost::split(strs, sff->rheader->name->s, boost::is_any_of(":"));
      offsetRegion = (boost::lexical_cast<int>(strs[1]) - stratification->yMin)/stratification->ySpan + (boost::lexical_cast<int>(strs[2]) - stratification->xMin)/stratification->xSpan * stratification->yCuts;
      //TODO: default one
      if(offsetRegion >= 0 && offsetRegion < stratification->xCuts * stratification->yCuts){
        offsetRegion = offsetRegion * stratification->flowCuts * 4;
      } else {
        offsetRegion = 0;
      }
    }

    if(flowOrderCheck){
        std::vector<uint32_t> illegalCandidates;
        illegalCandidates.reserve(sff->gheader->flow_length);
        //group into regions
        std::vector<uint32_t> regionLeft;
        std::vector<uint32_t> regionRight;
        regionLeft.reserve(illegalCandidates.size());
        regionRight.reserve(illegalCandidates.size());
        uint32_t MAGICNUMBER = 888888; //never to be a flow index, to avoid -1 for uint32_t
        uint32_t leftInd = MAGICNUMBER;
        uint32_t rightInd = MAGICNUMBER;
        for (uint32_t i=0; i < sff->gheader->flow_length; i++) {
            char flowBase = sff->gheader->flow->s[i];
            int flowBaseInt = toInt(flowBase);
            if(stratification != NULL){
                int offsetFlow = (i-stratification->flowStart)/stratification->flowSpan;
                if(offsetFlow < 0 || offsetFlow >= stratification->flowCuts){
                    offsetFlow = 0;
                } else {
                    offsetFlow = offsetFlow * 4;
                }
                flowBaseInt += offsetRegion + offsetFlow;
            }
            int flowSignal = flowgram[i];
            int oldHP = (flowSignal+50) / 100; // by convention.  flowSignals are fixed so this always works.
            int newHP = std::abs(hpsList[flowBaseInt][flowSignal]);
            if( (oldHP == 1 && newHP == 0) || (oldHP == 0 && newHP == 1) ){
                if(leftInd == MAGICNUMBER){
                  leftInd = i;
                }
                rightInd = i;
            } else if (oldHP >= 1 && newHP >= 1) {
              if(leftInd != MAGICNUMBER){
                  regionLeft.push_back(leftInd);
                  regionRight.push_back(rightInd);
                  leftInd = MAGICNUMBER;
                  rightInd = MAGICNUMBER;
              }
            }
        }
        //push the last one into queue
        if(leftInd != MAGICNUMBER){
            regionLeft.push_back(leftInd);
            regionRight.push_back(rightInd);
        }

        if(regionLeft.size() != 0){
          for(uint32_t i = 0; i < regionLeft.size(); ++i){
              debug_print("region left: %d; right: %d\n", regionLeft[i], regionRight[i]);
          }

          //evaluate all regions and report places causing illegal flow
          uint32_t tempIndex = 0;
          uint32_t tempFlowIndex = 0;// 1-based -> 0-based
          uint32_t numBases = sff->rheader->n_bases;


          for(uint32_t i=0; i < regionLeft.size(); ++i){
            uint32_t left = regionLeft[i];
            uint32_t right = regionRight[i];
            uint32_t tempLeft = left;
            uint32_t tempRight = right;
            string tempFragment;
            //get nearest left basecall
            if(left != 0){
              while(tempIndex < numBases){
                  if(sff->read->flow_index[tempIndex] != 0){ //skip filler
                    if(tempFlowIndex + sff->read->flow_index[tempIndex] >= left + 1){ //get
                      if (tempFlowIndex >= 1) {
                          debug_print("nearest left index: %d;", tempFlowIndex-1);
                          tempFragment += sff->gheader->flow->s[tempFlowIndex-1];
                          tempLeft = tempFlowIndex - 1;
                      } else {
                        tempLeft = 0; //consider the beginning of read
                      }
                      break;
                    } else {
                      tempFlowIndex += sff->read->flow_index[tempIndex];
                    }
                  }
                  tempIndex++;
              }
            }


            //get the internal segment
            for(uint32_t pos = left; pos <= right; ++pos){
              char flowBase = sff->gheader->flow->s[pos];
              int flowBaseInt = toInt(flowBase);
              if(stratification != NULL){
                  int offsetFlow = (pos-stratification->flowStart)/stratification->flowSpan;
                  if(offsetFlow < 0 || offsetFlow >= stratification->flowCuts){
                      offsetFlow = 0;
                  } else {
                      offsetFlow = offsetFlow * 4;
                  }
                  flowBaseInt += offsetRegion + offsetFlow;
//                  flowBaseInt += offsetRegion + (pos-stratification->flowStart)/stratification->flowSpan*4; //4 nuc tytpes
              }
              int flowSignal = flowgram[pos];
              int newHP = std::abs(hpsList[flowBaseInt][flowSignal]);
              if(newHP == 1){
                tempFragment += sff->gheader->flow->s[pos];
              }
            }

            //get nearest right basecall
            if(right + 1 != sff->gheader->flow_length ){
              while(tempIndex < numBases){
                  if(sff->read->flow_index[tempIndex] != 0){ //skip filler
                    tempFlowIndex += sff->read->flow_index[tempIndex];
                    tempIndex++;//increment
                    if(tempFlowIndex > right+1){
                      if (tempFlowIndex <= sff->gheader->flow_length) {
                          debug_print("nearest right index: %d", tempFlowIndex-1);
                          tempFragment += sff->gheader->flow->s[tempFlowIndex-1];
                          tempRight = tempFlowIndex-1;
                      }
                      break;
                    }
                  } else {
                    tempIndex++;
                  }
              }
            }
            debug_print("; fragment: %s\n", tempFragment.c_str());

            //check wether there is any 2 mer: if exists, discard the region
            int isIllegal = 0;

            if(isIllegal == 0){
                for(uint32_t sInd=0; sInd < tempFragment.size() - 1; ++sInd){
                  if(tempFragment[sInd] == tempFragment[sInd+1]){
                    isIllegal = 1;
                    break;
                  }
                }
            }

            //check whether nuc got incorporated at later flow while empty in same earlier flow and there is no positive nuc flow in between
            //create nucFragment
            debug_print("tempLeft: %d; tempRight: %d\n", tempLeft, tempRight);
            if(isIllegal == 0){
               uint32_t pos;
               pos = tempLeft==0 ? 0 : (tempLeft + 1);

               while (pos < tempRight){
                   char flowBase = sff->gheader->flow->s[pos];
                   int flowBaseInt = toInt(flowBase);
                   if(stratification != NULL){
                       int offsetFlow = (pos-stratification->flowStart)/stratification->flowSpan;
                       if(offsetFlow < 0 || offsetFlow >= stratification->flowCuts){
                           offsetFlow = 0;
                       } else {
                           offsetFlow = offsetFlow * 4;
                       }
                       flowBaseInt += offsetRegion + offsetFlow;
//                       flowBaseInt += offsetRegion + (pos-stratification->flowStart)/stratification->flowSpan*4; //4 nuc tytpes
                   }
                   int flowSignal = flowgram[pos];
                   int newHP = std::abs(hpsList[flowBaseInt][flowSignal]);
    //               fprintf(stdout, "pos: %d; newHP: %d\n", pos, newHP);
                   if(newHP == 1){
                     pos++;
                   } else {
                     uint32_t negativeNucInd = pos;
                     uint32_t positiveNucInd;
                     //identify positiveNucInd
                     uint32_t temp = pos+1;
    //                 fprintf(stdout, "temp: %d\n", temp);
                     while(temp <= tempRight){
                         char flowBase = sff->gheader->flow->s[temp];
                         int flowBaseInt = toInt(flowBase);
                         if(stratification != NULL){
                             int offsetFlow = (temp-stratification->flowStart)/stratification->flowSpan;
                             if(offsetFlow < 0 || offsetFlow >= stratification->flowCuts){
                                 offsetFlow = 0;
                             } else {
                                 offsetFlow = offsetFlow * 4;
                             }
                             flowBaseInt += offsetRegion + offsetFlow;
//                             flowBaseInt += offsetRegion + (temp-stratification->flowStart)/stratification->flowSpan*4; //4 nuc tytpes
                         }
                         int flowSignal = flowgram[temp];
//                         std::cout << "flowBaseInt: " << flowBaseInt << std::endl;
                         int newHP = std::abs(hpsList[flowBaseInt][flowSignal]);
                         if(newHP == 0){
                           temp++;
                         } else {
                           positiveNucInd = temp;
                           while(negativeNucInd < positiveNucInd){
                               if(sff->gheader->flow->s[negativeNucInd] ==  sff->gheader->flow->s[positiveNucInd]){
                                 isIllegal = 2;
                                 break;
                               } else{
                                   negativeNucInd++;
                               }
                               debug_print("negativeNucInd: %d; positiveNucInd: %d\n", negativeNucInd, positiveNucInd);
                           }
                           if(negativeNucInd == positiveNucInd){
                             pos = positiveNucInd+1;
                             break;
                           }
                         }
                         if(temp >= tempRight){
                           pos = tempRight + 1;//break out of the loop
                         }
                         if(isIllegal != 0){
                           break;
                         }
                     }
                     if(isIllegal != 0){
                       break;
                     }
                   }
               }
            }

            if(isIllegal != 0) {
              debug_print("fragment: %s is illegal; type: %d!\n", tempFragment.c_str(), isIllegal);
            } else {
              debug_print("fragment: %s is legal!\n", tempFragment.c_str());
            }

            if(isIllegal != 0) {
                for(uint32_t pos = left; pos <= right; ++pos){
                    ignoreCorrection[pos] = 1;
                }
            }
          }
        }
    }

    int flowIdx = 0;		// flow_idx is the number of flows to the next non-zero HP
    uint32_t oldBaseIdx = 0;	// index into sff->read->bases and quality

    int leftOffset = sff->rheader->clip_adapter_left > sff->rheader->clip_qual_left?sff->rheader->clip_adapter_left:sff->rheader->clip_qual_left;
    for (uint32_t i=0; i < sff->gheader->flow_length && oldBaseIdx < sff->rheader->n_bases; i++) {
      //calculate offset due to flow
      flowIdx++;
      char flowBase = sff->gheader->flow->s[i];
      int flowBaseInt = toInt(flowBase);
      if(stratification != NULL){
          int offsetFlow = (i-stratification->flowStart)/stratification->flowSpan;
          if(offsetFlow < 0 || offsetFlow >= stratification->flowCuts){
              offsetFlow = 0;
          } else {
              offsetFlow = offsetFlow * 4;
          }
          flowBaseInt += offsetRegion + offsetFlow;
//          flowBaseInt += offsetRegion + (i-stratification->flowStart)/stratification->flowSpan*4; //4 nuc tytpes
      }

      int flowSignal = flowgram[i];
      int oldHP = (flowSignal+50) / 100; // by convention.  flowSignals are fixed so this always works.

      //handle case oldBaseIdx + oldHP > (sff->rheader->n_bases - 1)
      if(oldBaseIdx + oldHP > sff->rheader->n_bases){
//          std::cout << "oldBaseIdx: " << oldBaseIdx << "; oldHP: " << oldHP << "; n_bases: " << sff->rheader->n_bases << std::endl;
          newBases += string(sff->rheader->n_bases - oldBaseIdx, flowBase);
          newQualities.append(sff->read->quality->s+oldBaseIdx, sff->rheader->n_bases - oldBaseIdx);
          newFlowIndices.push_back(flowIdx);
          if (sff->rheader->n_bases - oldBaseIdx > 1){
              newFlowIndices.insert(newFlowIndices.end(), sff->rheader->n_bases - oldBaseIdx - 1, 0);
          }
          newFlowSignal[i] = flowgram[i];
          break;
      }

      //handle barcode case where some mismatches are allowed in barcode adapter matching
      if(oldHP && sff->read->bases->s[oldBaseIdx] != flowBase){
          ignoreCorrectionCount++;
          return -1;
      }

      //assert(!oldHP || sff->read->bases->s[oldBaseIdx] == flowBase); // make sure we're tracking the old read

      // truncate observed signal to max
      if (flowSignal >= maxFlowSignal)
        flowSignal = maxFlowSignal - 1;

      // the new predicted HP as a function of flow index (bin), base and signal
      int newHP = hpsList[flowBaseInt][flowSignal];
      double qv = qvsList[flowBaseInt][flowSignal];

      //ignore any correction that could instroduce a new base or remove a single base
//      if( (oldHP == 1 && newHP == 0) || (oldHP == 0 && newHP == 1) ){
//        ignoreCorrection[i] = 1;
//      }

      if ((int)oldBaseIdx + oldHP < leftOffset || ignoreCorrection[i] == 1) // || i <= leftBound-1 || i >= righ ) // use oldHP instead
      {
        newHP = oldHP;
        newFlowSignal[i] = flowgram[i];
      } else {//update flowsignal
          int signal = 0;
          //no scaling & shifting
          if(rightBoundaryLists[flowBaseInt][newHP] <= newHP*100 + 49 && leftBoundaryLists[flowBaseInt][newHP] >= newHP*100 - 49){
            signal = flowgram[i];
          }else {
            signal =(int) std::floor( 98.0 * (flowgram[i] - leftBoundaryLists[flowBaseInt][newHP])/ (rightBoundaryLists[flowBaseInt][newHP] - leftBoundaryLists[flowBaseInt][newHP]) + newHP * 100 - 49 + 0.5 );
          }

          if(signal < 0)
              signal = 0;
          else if(signal < newHP * 100 - 49)
              signal = newHP * 100 - 49;
          else if(signal > newHP * 100 + 49)
              signal = newHP * 100 + 49;
          newFlowSignal[i] = signal;
      }

      if (newHP > 0) {
        // add newHP bases to sequence
        newBases += string(newHP, flowBase);

        // Copy the old quality scores.
        // newHP > oldHP: assign QV of first base of HP to newly inserted bases
        // newHP < oldHP then drop the extra leading QVs.
        if (newHP > oldHP) {
          if (oldHP > 0) {
            newQualities += string(newHP-oldHP, sff->read->quality->s[oldBaseIdx]);
            newQualities.append(sff->read->quality->s+oldBaseIdx, oldHP);

          }
          else {
            // there are no qualities to copy. using flow signal qv
              newQualities += string(newHP, (int)qv);
          }
        }

        else {
          newQualities.append(sff->read->quality->s+oldBaseIdx + oldHP - newHP, newHP);
        }

        // update flow index.  first position is number of skipped flows.
        // remaining bases are zero because we're already at the designated flow.
        newFlowIndices.push_back(flowIdx);
        newFlowIndices.insert(newFlowIndices.end(), newHP-1, 0);
        flowIdx=0;		// reset for next base
      }

      // update our offset in sff->read->bases
      oldBaseIdx += oldHP;
    }

    //only replace it if numBases is larger than qual clip
    if(newBases.size() >= sff->rheader->clip_adapter_right && newBases.size() >= sff->rheader->clip_qual_right){

        // update the number of bases in the read header.  the rest of the read header is unchanged.
        sff->rheader->n_bases = newBases.size();
        // update sff with new values using ion_string
        if (allocateNewMemory){
            ion_string_destroy(sff->read->bases);
            sff->read->bases = ion_string_init(newBases.size()+1);
        }
        copy(newBases.begin(), newBases.end(), sff->read->bases->s);
        sff->read->bases->l = newBases.size();
        sff->read->bases->s[sff->read->bases->l]='\0';

        //reset qv for regions preceding to leftOffset
        for(uint32_t ind = 0; ind+1 < (uint32_t)leftOffset; ++ind)
        {
            newQualities[ind] = sff->read->quality->s[ind];
        }
        if (allocateNewMemory){
          ion_string_destroy(sff->read->quality);
          sff->read->quality = ion_string_init(newQualities.size()+1);
        }
        copy(newQualities.begin(), newQualities.end(), sff->read->quality->s);
        sff->read->quality->l = newQualities.size();
        sff->read->quality->s[sff->read->quality->l]='\0';


        if (allocateNewMemory){
          free(sff->read->flow_index);
          sff->read->flow_index = (uint8_t*)ion_malloc(sizeof(uint8_t)*newFlowIndices.size(), __func__, "sff->read->flow_index");
        }
        copy(newFlowIndices.begin(), newFlowIndices.end(), sff->read->flow_index);

        copy(newFlowSignal.begin(), newFlowSignal.end(), sff->read->flowgram);
   }
    else {
        shorterReadCount++;
    }
    return 0;
}




typedef struct loadArg
{
	BamReader* bamReader;
}loadArg;

void* LoadFunc(void* arg0)
{
    loadArg* arg = (loadArg*)arg0;
	int threadIndex = 0;
	if(arg)
	{
		while(arg->bamReader)
		{				
			sem_wait(&semLoadNull[threadIndex]);

			bamAlignmentIn[threadIndex].QueryBases.clear();
			bamAlignmentIn[threadIndex].Qualities.clear();
			bamAlignmentIn[threadIndex].RemoveTag("RG");
			bamAlignmentIn[threadIndex].RemoveTag("PG");
			bamAlignmentIn[threadIndex].RemoveTag("ZF"); 	
			bamAlignmentIn[threadIndex].RemoveTag("FZ");			
			if(bamAlignmentIn[threadIndex].HasTag("ZA"))
			{
				bamAlignmentIn[threadIndex].RemoveTag("ZA");
			}
            if(bamAlignmentIn[threadIndex].HasTag("ZG"))
			{
				bamAlignmentIn[threadIndex].RemoveTag("ZG");
			}

			if((arg->bamReader)->GetNextAlignment(bamAlignmentIn[threadIndex]))
			{
				sem_post(&semLoadFull[threadIndex]);					
                threadIndex = (threadIndex + 1) % NUMTHREADS;
			}
			else
			{					
				break;
			}
		}		
	}

    for(int i = 0; i < NUMTHREADS; ++i)
    {
		pthread_mutex_lock(&mutexQuit);
        quit[threadIndex] = true;
		pthread_mutex_unlock(&mutexQuit);
        sem_post(&semLoadFull[threadIndex]);
		threadIndex = (threadIndex + 1) % NUMTHREADS;
    }

	cout << "Loading thread exits." << endl;

	return NULL;
}

typedef struct recallArg
{
	int threadIndex;
    string rgname;
	bool flowOrderCheck;
	int maxFlowSignal;
	string flow_order;
    string key;
	uint32_t* nReads;
	uint32_t* shorterReadCount;
    uint32_t* ignoreCorrection;
}recallArg;

void* RecallFunc(void* arg0)
{
    recallArg* arg = (recallArg*)arg0;
	if(arg)
	{
		int nZA = -1;
		int nZG = -1;
		int recal_result = 0;
		int clip_flow0 = 0;

		uint16_t nFlows = arg->flow_order.length();
        uint16_t nKeys = arg->key.length();

		boost::scoped_array<uint16_t> flowgram(new uint16_t[nFlows]);      
		boost::scoped_array<char> bases(new char[MAX_BASES]);
		boost::scoped_array<char> qalities(new char[MAX_BASES]);
		boost::scoped_array<uint8_t> flow_index(new uint8_t[MAX_BASES]);
		vector<uint16_t> flowInt(nFlows);

		sff_t* sff = sff_init1();
		sff->gheader = sff_header_init1(0, nFlows, arg->flow_order.c_str(), arg->key.c_str());;
		sff->rheader->name = ion_string_init(0);
		sff->read->bases = ion_string_init(0);
		sff->read->quality = ion_string_init(0);
		sff->rheader->clip_adapter_left = nKeys + 1; //only processing inserts
		sff->rheader->clip_qual_left = 0;
		sff->rheader->clip_qual_right = 0;//sff->rheader->n_bases + 1;
		sff->rheader->clip_adapter_right = 0;//sff->rheader->n_bases + 1;

		bool myQuit = false;
        sem_wait(&semLoadFull[arg->threadIndex]);
		pthread_mutex_lock(&mutexQuit);
		myQuit = quit[arg->threadIndex];
		pthread_mutex_unlock(&mutexQuit);
		while(!myQuit)
        {
			string rname = bamAlignmentIn[arg->threadIndex].Name;
			sff->rheader->name_length = bamAlignmentIn[arg->threadIndex].Name.length();
			sff->rheader->name->s = (char*)bamAlignmentIn[arg->threadIndex].Name.c_str();
			sff->rheader->n_bases = nKeys + bamAlignmentIn[arg->threadIndex].Length;

			if(!bamAlignmentIn[arg->threadIndex].GetTag("FZ", flowInt))
			{
				sem_post(&semLoadNull[arg->threadIndex]);
				continue;
			}

			char zfType = ' ';
            bool hasZFTag = bamAlignmentIn[arg->threadIndex].GetTagType("ZF", zfType);
            if(hasZFTag)
			{
                  switch(zfType)
				  {
                  case Constants::BAM_TAG_TYPE_INT8:
                      {
                          int8_t zf_int8 = 0;
                          bamAlignmentIn[arg->threadIndex].GetTag("ZF", zf_int8);
                          clip_flow0 = zf_int8;
                      }
                      break;
                  case Constants::BAM_TAG_TYPE_UINT8:
                      {
                          uint8_t zf_uint8 = 0;
                          bamAlignmentIn[arg->threadIndex].GetTag("ZF", zf_uint8);
                          clip_flow0 = zf_uint8;
                      }
                      break;
                  case Constants::BAM_TAG_TYPE_INT16:
                      {   int16_t zf_int16 = 0;
                          bamAlignmentIn[arg->threadIndex].GetTag("ZF", zf_int16);
                          clip_flow0 = zf_int16;
                      }
                      break;
                  case Constants::BAM_TAG_TYPE_UINT16:
                      {
                          uint16_t zf_uint16 = 0;
                          bamAlignmentIn[arg->threadIndex].GetTag("ZF", zf_uint16);
                          clip_flow0 = zf_uint16;
                      }
                      break;
                  case Constants::BAM_TAG_TYPE_INT32:
                      {
                          int32_t zf_int32 = 0;
                          bamAlignmentIn[arg->threadIndex].GetTag("ZF", zf_int32);
                          clip_flow0 = zf_int32;
                      }
                      break;
                  case Constants::BAM_TAG_TYPE_UINT32:
                      {
                          uint32_t zf_uint32 = 0;
                          bamAlignmentIn[arg->threadIndex].GetTag("ZF", zf_uint32);
                          clip_flow0 = zf_uint32;
                      }
                      break;
                  default:
					  sem_post(&semLoadNull[arg->threadIndex]);
                      continue;
                  }
            } 
			else 
			{
                //ignore the record instead of contaminating the bam file as it has problem with parsing zf tag
				sem_post(&semLoadNull[arg->threadIndex]);
                continue;
            }

			pthread_mutex_lock(&mutexSave);
			++num2save;
			pthread_mutex_unlock(&mutexSave);

			copy(flowInt.begin(), flowInt.end(), flowgram.get());
			sff->read->flowgram = flowgram.get(); 

			char* pBases = bases.get();
			char* pQualities = qalities.get();
            uint32_t nBases0 = sff->rheader->n_bases;
			uint32_t nBase = 0;
			int index = 1;
			//reconstruct bases, quality and flow_index from flowInt such that all reads should be able to be recalibrated
            for(uint32_t nFlow = 0; nBase < nBases0 && nFlow < nFlows; ++nFlow){
			  int nHp = ((flowInt[nFlow]) + 50) / 100;
			  if(nHp > 0)
			  {
				  flow_index[nBase] = index;
                  pBases[nBase] = arg->flow_order[nFlow];
				  if(nBase < nKeys){
					//use default quality
					  pQualities[nBase] = DEAFAUL_QUALITY;
				  } else {
                      pQualities[nBase] = bamAlignmentIn[arg->threadIndex].Qualities[nBase - nKeys] - 33;
				  }
				  ++nBase;
				  --nHp;
                  while(nHp > 0 && nBase < nBases0)
				  {
					  flow_index[nBase] = 0;
                      pBases[nBase] = arg->flow_order[nFlow];
					  if(nBase < nKeys){
						//use default quality
						  pQualities[nBase] = DEAFAUL_QUALITY;
					  } else {
                          pQualities[nBase] = bamAlignmentIn[arg->threadIndex].Qualities[nBase - nKeys] - 33;
					  }
					  ++nBase;
					  --nHp;
				  }
				  index = 1;
			  }
			  else
			  {
				  ++index;
			  }
			}
			sff->read->bases->s = bases.get();
			sff->read->quality->s = qalities.get();
			sff->read->flow_index = flow_index.get();
         
			recal_result = recalibrate(arg->flowOrderCheck, arg->maxFlowSignal, *(arg->shorterReadCount), *(arg->ignoreCorrection), sff, false);

			nZA = -1;
			nZG = -1;

            if(bamAlignmentIn[arg->threadIndex].HasTag("ZA"))
			{
                bamAlignmentIn[arg->threadIndex].GetTag("ZA", nZA);
			}
            if(bamAlignmentIn[arg->threadIndex].HasTag("ZG"))
			{
                bamAlignmentIn[arg->threadIndex].GetTag("ZG", nZG);
			}

            std::vector<float> fieldZM;
            if (bamAlignmentIn[arg->threadIndex].HasTag("ZM"))
            {
                bamAlignmentIn[arg->threadIndex].GetTag("ZM", fieldZM);
            }


            std::vector<float> fieldZP;
            if (bamAlignmentIn[arg->threadIndex].HasTag("ZP"))
            {
                bamAlignmentIn[arg->threadIndex].GetTag("ZP", fieldZP);
            }

			sem_post(&semLoadNull[arg->threadIndex]);

			sem_wait(&semSaveNull[arg->threadIndex]);
			
			  bamAlignmentOut[arg->threadIndex].QueryBases.clear();
			  bamAlignmentOut[arg->threadIndex].Qualities.clear();
			  bamAlignmentOut[arg->threadIndex].RemoveTag("RG");
			  bamAlignmentOut[arg->threadIndex].RemoveTag("PG");
			  bamAlignmentOut[arg->threadIndex].RemoveTag("ZF"); 	
			  bamAlignmentOut[arg->threadIndex].RemoveTag("FZ");
			 
			  if(bamAlignmentOut[arg->threadIndex].HasTag("ZA"))
			  {
				  bamAlignmentOut[arg->threadIndex].RemoveTag("ZA");
			  }

			  if(bamAlignmentOut[arg->threadIndex].HasTag("ZG"))
			  {		  
				  bamAlignmentOut[arg->threadIndex].RemoveTag("ZG");
			  }

              if(bamAlignmentOut[arg->threadIndex].HasTag("ZM"))
              {
                  bamAlignmentOut[arg->threadIndex].RemoveTag("ZM");
              }

              if(bamAlignmentOut[arg->threadIndex].HasTag("ZP"))
              {
                  bamAlignmentOut[arg->threadIndex].RemoveTag("ZP");
              }

			bamAlignmentOut[arg->threadIndex].SetIsMapped(false);
			bamAlignmentOut[arg->threadIndex].Name = rname;

			uint32_t nBases = sff->rheader->n_bases - sff->rheader->clip_adapter_left + 1;
			bamAlignmentOut[arg->threadIndex].QueryBases.reserve(nBases);
			bamAlignmentOut[arg->threadIndex].Qualities.reserve(nBases);
			for (uint32_t base = sff->rheader->clip_adapter_left - 1; base < sff->rheader->n_bases; ++base)
			{
				bamAlignmentOut[arg->threadIndex].QueryBases.push_back(sff->read->bases->s[base]);
				bamAlignmentOut[arg->threadIndex].Qualities.push_back(sff->read->quality->s[base] + 33);
			}

            int clip_flow = 0;

			if(recal_result == 0)
			{
				for (unsigned int base = 0; base < sff->rheader->clip_adapter_left && base < sff->rheader->n_bases; ++base)
				{
					clip_flow += sff->read->flow_index[base]; //TODO: need to take care of case of hp > 1
				}
				if (clip_flow > 0)
				{
					clip_flow--;
				}
			} 
			else 
			{
				clip_flow = clip_flow0;
			}

            bamAlignmentOut[arg->threadIndex].AddTag("RG","Z", arg->rgname);
			bamAlignmentOut[arg->threadIndex].AddTag("PG","Z", string("Recalibration"));
			bamAlignmentOut[arg->threadIndex].AddTag("ZF","i", clip_flow); // TODO: trim flow
			vector<uint16_t> flowgram(sff->gheader->flow_length);
			//flowgram is clipped at right end of read
			copy(sff->read->flowgram, sff->read->flowgram + sff->gheader->flow_length, flowgram.begin());
			bamAlignmentOut[arg->threadIndex].AddTag("FZ", flowgram);			     

			if(nZG > 0)
			{
				//calculate nZA
				int nZA = 0;
				for(int nFlow = 0; nFlow <= nZG; ++nFlow)
				{
					nZA += (flowgram[nFlow] + 50) / 100;
				}
				nZA = nZA - nKeys - 1; //allocate one for adapter
				if(nZA > 0)
				{ //skip writing any value that might be wrong
                    bamAlignmentOut[arg->threadIndex].AddTag("ZA","i", nZA);
				}

                bamAlignmentOut[arg->threadIndex].AddTag("ZG","i", nZG);
			}

            if (fieldZP.size() > 0)
            {
                bamAlignmentOut[arg->threadIndex].AddTag("ZP", fieldZP);
            }

            if (fieldZM.size() > 0)
            {
                bamAlignmentOut[arg->threadIndex].AddTag("ZM", fieldZM);
            }

			++(*(arg->nReads)); 

			sem_post(&semSaveFull[arg->threadIndex]);

            sem_wait(&semLoadFull[arg->threadIndex]);
			pthread_mutex_lock(&mutexQuit);
			myQuit = quit[arg->threadIndex];
			pthread_mutex_unlock(&mutexQuit);
		}
	}

	sem_post(&semSaveFull[arg->threadIndex]);

	cout << "Recall thread " << arg->threadIndex << " exits." << endl;

	return NULL;
}



void usage() {
  cout << "SeqBoost - perform flow signal recalibration in an SFF/BAM with flow QV table (with gentler flow signal adjustment)." << endl;
  fprintf (stdout, "Version = %s-%s (%s) (%s)\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(),
      IonVersion::GetSvnRev().c_str(), IonVersion::GetBuildNum().c_str());
  cout << "Usage: " << endl
       << "  SeqBoost -t flowQVTable.txt rawlib.sff/rawlib.basecaler.bam " << endl << endl
       << "Options:" << endl
       << "  -f,--flow-order-check  check whether flow order is legal" << endl
       << "  -i,--input-type        input file type (sff or bam), which overrides the implicit naming convention" << endl
       << "  -h,--help              this message" << endl
       << "  -o,--out               file path to recalibrated SFF/BAM (default same as input file with a suffix of '.rc.sff' or '.rc.bam')" << endl
       << "  -t,--flow-QV-Table     file produced by HPTableParser that produces flow QV out of per flow hp table" << endl
       << "  -j,--right-boundary   output right boundary json files" << endl;
  exit(1);
}

int main(int argc, const char *argv[]) {

  // Options handling
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  bool help;
  string flowQVFile;
  string outputFile;
  bool flowOrderCheck;
  bool rb = false;
  string inputType;
  vector<string> inputFiles;

  opts.GetOption(help,"false", 'h', "help");
  opts.GetOption(flowQVFile,"",'t',"flow-QV-Table");
  opts.GetOption(inputType,"",'i',"input-type");
  opts.GetOption(flowOrderCheck, "false", 'f', "flow-order-check");
  opts.GetOption(outputFile,"",'o',"out-sff");
    opts.GetOption(rb,"false",'j',"right-boundary");
  opts.GetLeftoverArguments(inputFiles); //process first element in the vector

  if (help || flowQVFile.empty() || inputFiles.size() != 1)
    usage();

  fprintf (stdout, "SeqBoost Version = %s-%s (%s) (%s)\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(),
      IonVersion::GetSvnRev().c_str(), IonVersion::GetBuildNum().c_str());

  string inputFile = inputFiles[0];
  parseFlowQVTable(flowQVFile, rb);
  int maxFlowSignal = hpsList[0].size();

  //assume input ends with .sff

  if(inputType.empty()){
      if(boost::algorithm::ends_with(inputFile, ".sff")){
          inputType = "sff";
      } else if (boost::algorithm::ends_with(inputFile, ".bam")){
          inputType = "bam";
      } else {
          std::cerr << "unknown input type [case insensitive: sff or bam]" << std::endl;
          exit(1);
      }
  } else {
      boost::algorithm::to_lower(inputType);
      if( !(inputType.compare( "sff") == 0 || inputType.compare("bam")) ) {
          std::cerr << "unknown input type [case insensitive: sff or bam]" << std::endl;
          exit(1);
      }
  }

  if (outputFile.empty()){
      if(inputType.compare("sff") == 0){
          outputFile = inputFile.substr(0, inputFile.size() - 4) + ".rc.sff";
      } else {
          outputFile = inputFile.substr(0, inputFile.size() - 4) + ".rc.bam";
      }
  }
  fprintf(stdout, "%s\n", outputFile.c_str());

  clock_t start_clock, end_clock;
  clock_t diff_clock;
  start_clock = clock();
  struct timeval tv1,tv2;
  struct timezone tz1;
  gettimeofday(&tv1,&tz1);

  //TODO: make correction and flow order check routine usable in both types of input
  if(inputType.compare("sff") == 0){

      // open the SFF for reading
      sff_file_t* sffIn = sff_fopen(inputFile.c_str(), "r", NULL, NULL);
      sff_t* sff = NULL;
      int totalReads = sffIn->header->n_reads;
    //  sffIn->header->n_reads = 9999;
      // open the SFF for writing
      sff_file_t* sffOut = sff_fopen(outputFile.c_str(), "w", sffIn->header, NULL);
      fprintf(stdout, "Reading SFF: %s; Total Reads: %d; Writing SFF: %s\n", inputFile.c_str(), totalReads, outputFile.c_str());

      // tallies
      uint32_t readCount = 0;
      uint32_t shorterReadCount = 0;
      uint32_t ignoreCorrection = 0;


      // iterate over each read in the SFF
      while(NULL != (sff = sff_read(sffIn))) {

//          if(strcmp(sff->rheader->name->s, "Y9VO3:144:385") != 0){
//            continue;
//          }
//          fprintf(stdout, "%s\n", sff->rheader->name->s);

        recalibrate(flowOrderCheck, maxFlowSignal, shorterReadCount, ignoreCorrection, sff, stratification);

        // write it to file
        sff_write(sffOut, sff);

        // cleanup
        sff_destroy(sff);

        readCount++;
      }

      fprintf(stdout, "Shorter reads after correction: %d\n", shorterReadCount);
      fprintf(stdout, "Reads ignoring correction: %d\n", ignoreCorrection);
      sff_fclose(sffIn);
      sff_fclose(sffOut);
  } else { //bam case
      BamReader bamReader;
      if (!bamReader.Open(inputFile))
      {
          std::cerr << "SeqBoost ERROR: fail to open bam" << inputFile << endl;
          exit(1);
      }

      SamHeader samHeader = bamReader.GetHeader();
      if(!samHeader.HasReadGroups())
      {
          bamReader.Close();
          std::cerr << "SeqBoost ERROR: There is no read group in " << inputFile << endl;
          exit(1);
      }

      string flow_order;
      string key;
	  string rgname;
      for (SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr )
      {
          if(itr->HasFlowOrder())
          {
              flow_order = itr->FlowOrder;
          }
          if(itr->HasKeySequence())
          {
              key = itr->KeySequence;
          }
		  rgname = itr->ID;
      }

      uint16_t nFlows = flow_order.length();
      uint16_t nKeys = key.length();
      if(nFlows < 1 || nKeys < 1)
      {
          bamReader.Close();
          std::cerr << "SeqBoost ERROR: there is no floworder or key in " << inputFile << endl;
          exit(1);
      }

      SamProgram sam_program("recal");
      sam_program.PreviousProgramID = samHeader.Programs.Begin()->ID;
      sam_program.Name = "SeqBoost";
      sam_program.Version = SEQBOOST_VERSION;
	  samHeader.Programs.Add(sam_program);

	  RefVector refvec = bamReader.GetReferenceData();
      BamWriter bamWriter;
	  bamWriter.SetCompressionMode(BamWriter::Compressed);
      bamWriter.SetNumThreads(8);
              
      if(!bamWriter.Open(outputFile, samHeader, refvec))
      {
          bamReader.Close();
          cerr << "SeqBoost ERROR: failed to open " << outputFile << endl;
          exit(1);
      }

	  int threadIndex = 0;

	  for(threadIndex = 0; threadIndex < NUMTHREADS; ++threadIndex)
	  {
          sem_init(&semLoadNull[threadIndex], 0, 0);
          sem_init(&semLoadFull[threadIndex], 0, 0);
          sem_init(&semSaveNull[threadIndex], 0, 0);
          sem_init(&semSaveFull[threadIndex], 0, 0);
		  sem_post(&semLoadNull[threadIndex]);
		  sem_post(&semSaveNull[threadIndex]);
		  quit[threadIndex] = false;
	  }

	  pthread_mutex_init(&mutexQuit, NULL); 
	  pthread_mutex_init(&mutexSave, NULL); 

	  loadArg argLoad;
      argLoad.bamReader = &bamReader;
	  pthread_t loadThread;
	  pthread_create(&loadThread, NULL, LoadFunc, &argLoad);

      uint32_t nReads[NUMTHREADS];
      uint32_t shorterReadCount[NUMTHREADS];
      uint32_t ignoreCorrection[NUMTHREADS];
	  recallArg argRecall[NUMTHREADS];
	  pthread_t recallThread[NUMTHREADS];
	  for(threadIndex = 0; threadIndex < NUMTHREADS; ++threadIndex)
	  {
		  nReads[threadIndex] = 0;
		  shorterReadCount[threadIndex] = 0;
		  ignoreCorrection[threadIndex] = 0;
		  argRecall[threadIndex].threadIndex = threadIndex;
		  argRecall[threadIndex].flowOrderCheck = flowOrderCheck;
          argRecall[threadIndex].rgname = rgname;
		  argRecall[threadIndex].maxFlowSignal = maxFlowSignal;
		  argRecall[threadIndex].flow_order = flow_order;
		  argRecall[threadIndex].key = key;
		  argRecall[threadIndex].nReads = &(nReads[threadIndex]);
		  argRecall[threadIndex].shorterReadCount = &(shorterReadCount[threadIndex]);
		  argRecall[threadIndex].ignoreCorrection = &(ignoreCorrection[threadIndex]);

          pthread_create(&recallThread[threadIndex], NULL, RecallFunc, &argRecall[threadIndex]);
	  }

	  int num = 0;
	  threadIndex = 0;
	  sem_wait(&semSaveFull[threadIndex]);
	  pthread_mutex_lock(&mutexSave);
	  num = num2save;
	  pthread_mutex_unlock(&mutexSave);
	  while(0 < num)
      {
          bamWriter.SaveAlignment(bamAlignmentOut[threadIndex]);

          sem_post(&semSaveNull[threadIndex]);
          threadIndex = (threadIndex + 1) % NUMTHREADS;
		  sem_wait(&semSaveFull[threadIndex]);
		  
		  pthread_mutex_lock(&mutexSave);
		  --num2save;
		  num = num2save;
		  pthread_mutex_unlock(&mutexSave);
	  }

      pthread_join(loadThread, NULL);
      for(int threadIndex = 0; threadIndex < NUMTHREADS; ++threadIndex)
      {
          pthread_join(recallThread[threadIndex], NULL);
      }

      bamReader.Close();
      bamWriter.Close();

	  for(threadIndex = 0; threadIndex < NUMTHREADS; ++threadIndex)
	  {
          sem_destroy(&semLoadNull[threadIndex]);
          sem_destroy(&semLoadFull[threadIndex]);
          sem_destroy(&semSaveNull[threadIndex]);
          sem_destroy(&semSaveFull[threadIndex]);
	  }

	  pthread_mutex_destroy(&mutexQuit);   
	  pthread_mutex_destroy(&mutexSave);

      for(threadIndex = 1; threadIndex < NUMTHREADS; ++threadIndex)
      {
          nReads[0] += nReads[threadIndex];
          shorterReadCount[0] += shorterReadCount[threadIndex];
          ignoreCorrection[0] += ignoreCorrection[threadIndex];
      }

      fprintf(stdout, "Total Reads: %d\n", nReads[0]);
      fprintf(stdout, "Shorter reads after correction: %d\n", shorterReadCount[0]);
      fprintf(stdout, "Reads ignoring correction: %d\n", ignoreCorrection[0]);
      bamReader.Close();
      bamWriter.Close();
  }

  end_clock = clock();
  diff_clock = end_clock - start_clock;
  gettimeofday(&tv2,&tz1);
  printf("\nSeqBoost CPU_TIME: %.3f\n", (double)diff_clock/CLOCKS_PER_SEC);
  printf("SeqBoost WALL_TIME: %0.3f\n",(double)(tv2.tv_sec - tv1.tv_sec) + ((tv2.tv_usec - tv1.tv_usec)/1000000.0));

  return 0;
}


