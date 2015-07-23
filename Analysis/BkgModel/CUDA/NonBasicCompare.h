/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * NonBasicCompare.h
 *
 *  Created on: Oct 29, 2014
 *      Author: Jakob Siegel
 */

#ifndef NONBASICCOMPARE_H_
#define NONBASICCOMPARE_H_


enum CompareOutput
{
  CompOutNone       = 0,
  CompOutAll        = ( 1<<0 ),
  CompOutByRegion   = ( 1<<1 )
};


template<typename T>
class CCompare{

protected:
  size_t cnt;
  size_t numFails;
public:
  CCompare():cnt(0),numFails(0){}
  virtual ~CCompare(){}

  size_t getNumValues() const {return cnt;}
  size_t getNumFails() const {return numFails;}
  float getFailPercent() const { return 100.0f*numFails/cnt; }

  virtual bool Compare(const T & gold, const T & comp) = 0; //compares comp vs gold needs to implement: update cnt: number of comparisons, and numFails: number of failed comparisons
  virtual void reset(){ cnt = 0; numFails = 0;}

  virtual string lastCompString() = 0; //returns string containing information about last comparisons
  virtual string compSummaryString() = 0; //returns string containing summary information about all performed comparisons

};


class CompareF : public CCompare<float>
{
  float ep;

  double sumAll;
  double sumError;
  float maxDiff;
  float maxDiffValue;
  float maxDiffGoldValue;
  float lastDiff;
  float lastGold;
  float lastComp;

public:

  CompareF(): ep(0.01f),sumAll(0),sumError(0),maxDiff(0),maxDiffValue(0),maxDiffGoldValue(0),lastDiff(0){}

  virtual bool Compare(const float & gold, const float & comp){
    lastComp = comp;
    lastGold = gold;
    const float d = gold-comp;
    sumAll += (d*d);
    cnt ++;
    lastDiff =  abs(d);

    if( lastDiff > ep ){
      if(lastDiff > maxDiff){
        maxDiff = lastDiff;
        maxDiffValue = comp;
        maxDiffGoldValue = gold;
      }
      sumError += (d*d);
      numFails++;
      return false;
    }
    return true;
  }
  virtual float Difference(){ return lastDiff; }

  void reset(){ CCompare<float>::reset(); sumAll=0.0; sumError=0.0; maxDiff=0.0f; maxDiffValue=0.0f; lastDiff=0.0f; }
  void setEpsilon(float epsilon){ ep = epsilon;}
  double getRSMeanError(){ return (numFails > 0)?( sqrt(sumError/numFails)):(0);}
  double getRSMeanErrorAll(){ return (cnt > 0)?( sqrt(sumAll/cnt)):(0);}
  float getLastDiff() {return lastDiff;}
  float getMaxDiff(){ return maxDiff;}
  float getMaxDiffValue(){ return maxDiffValue; }
  float getMaxDiffGoldValue(){ return maxDiffGoldValue; }

  virtual string lastCompString(){
    ostringstream message;
    message << "CompareF,";
    if(cnt>0){
      message << (cnt -1)  << "," << numFails << "," << lastGold << "," << lastComp << "," << lastDiff;
    }else{
      message << " no comparison performed yet";
    }
    return message.str();
  }

  virtual string compSummaryString(){
    ostringstream message;
    message << "CompareF Summary with e: " << ep  ;
    message << " #Comparisons: " <<  getNumValues()  << " #Fails: "<< getNumFails() << "(" << getFailPercent() << "%) max difference: abs(" <<  getMaxDiffGoldValue() << "-" << getMaxDiffValue() << ") = " << maxDiff << " RSME fails: "<< getRSMeanError() << " all: " << getRSMeanErrorAll() ;
    return message.str();
  }

};


#endif /* NONBASICCOMPARE_H_ */
