/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DEBUGME_H
#define DEBUGME_H

#include "stdlib.h"
#include "stdio.h"
#include <unistd.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <libgen.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "Region.h"
#include "IonVersion.h"
#include "file-io/ion_util.h"
#include "Utils.h"
#include "SpecialDataTypes.h"
#include "SeqList.h"
#include "OptBase.h"

class HashTable_xyflow {
public:
    HashTable_xyflow() {clear(); maxRow=maxCol=maxFlow=0; maxSet=false;}
    HashTable_xyflow(int rMax,int cMax,int fMax) {clear(); maxRow=rMax; maxCol=cMax; maxFlow=fMax; maxSet=true;}

    std::string makeTag_rc(int r,int c) {std::ostringstream ss; ss<<r<<":"<<c;return(ss.str());}
    std::string makeTag_rcflow(int r,int c,int f) {std::ostringstream ss; ss<<r<<":"<<c<<":"<<f;return(ss.str());}
    std::string makeTag_xyflow(int x,int y,int f) {std::ostringstream ss; ss<<y<<":"<<x<<":"<<f;return(ss.str());}
    bool has_key(std::string key) {try{return(exist[key]);} catch(...){return(false);};}
    bool has_key_xy(std::string key) {try{return(exist_xy[key]);} catch(...){return(false);};}
    bool has_rcflow(int r,int c,int f) {return (has_key(makeTag_rcflow(r,c,f)));}
    bool has_xyflow(int x,int y,int f) {return (has_rcflow(y,x,f));}
    void insert(std::string key) {if (!has_key(key)) {exist[key]=true; map_id[key]=count++;};}
    void insert_xy_key(std::string key) {if (!has_key(key)) {exist_xy[key]=true; map_xy[key]=count_xy++;};}
    void insert_rc(int r,int c) {if (ok_rc(r,c)) {insert_xy_key(makeTag_rc(r,c));}}
    void insert_xy(int x,int y) {insert_rc(y,x);}
    void insert_rcflow(int r,int c,int f) {if (ok_rcflow(r,c,f)) {insert(makeTag_rcflow(r,c,f));}}
    void insert_xyflow(int x,int y,int f) {insert_rcflow(y,x,f);}
    void insert(std::string key,bool mm,std::string hp) {if (!has_key(key)) {exist[key]=true; map_id[key]=count++; hp_ref[key]=hp; mismatch[key]=mm; if(mm)count_mm++;};}
    void insert_rcflow(int r,int c,int f,bool mm,std::string hp) {if (ok_rcflow(r,c,f)) {insert(makeTag_rcflow(r,c,f),mm,hp);}}
    void insert_xyflow(int x,int y,int f,bool mm,std::string hp) {insert_rcflow(y,x,f,mm,hp);}
    int id_key(std::string key) {int id=-1; if (has_key(key)) {try{id=map_id[key];} catch(...){};} return(id);}
    int id_key_xy(std::string key) {int id=-1; if (has_key_xy(key)) {try{id=map_xy[key];} catch(...){};} return(id);}
    int id_rcflow(int r,int c,int f) {return (id_key(makeTag_rcflow(r,c,f)));}
    int id_xyflow(int x,int y,int f) {return (id_rcflow(y,x,f));}
    int id_rc(int r,int c) {return (id_key_xy(makeTag_rc(r,c)));}
    int id_xy(int x,int y) {return (id_rc(y,x));}
    std::string hp_key(std::string key) {std::string hp="0N"; if (has_key(key)) {try{hp=hp_ref[key];} catch(...){}} return(hp);}
    std::string hp_rcflow(int r,int c,int f) {return (hp_key(makeTag_rcflow(r,c,f)));}
    std::string hp_xyflow(int x,int y,int f) {return (hp_rcflow(y,x,f));}
    int mm_key(std::string key) {int mm=-1; if (has_key(key)) {try{mm=mismatch[key];} catch(...){};} return(mm);}
    int mm_rcflow(int r,int c,int f) {return (mm_key(makeTag_rcflow(r,c,f)));}
    int mm_xyflow(int x,int y,int f) {return (mm_rcflow(y,x,f));}
    int size() {return (count);}
    int size_xy() {return (count_xy);}
    int size_mm() {return (count_mm);}
    //void remove(std::string key) {if (has_key(key)) {exist[key]=false; count--;};}
    //void remove_rcflow(int r,int c,int f) {remove(makeTag_rcflow(r,c,f));}
    //void remove_xyflow(int x,int y,int f) {remove_rcflow(y,x,f);}
    void clear(void) {count=count_mm=count_xy=0; maxSet=false; filename=""; map_id.clear(); exist.clear(); mismatch.clear(); hp_ref.clear(); map_xy.clear(); exist_xy.clear();}
    bool ok_rcflow(int r,int c,int f) {
        if (maxSet)
            return((r>=0 && c>=0 && f>=0 && r<maxRow && c<maxCol && f<maxFlow)?true:false);
        else
            return((r>=0 && c>=0 && f>=0)?true:false);
    }
    bool ok_rc(int r,int c) {
        if (maxSet)
            return((r>=0 && c>=0 && r<maxRow && c<maxCol)?true:false);
        else
            return((r>=0 && c>=0)?true:false);
    }
    void set_xyflow_limits(int numFlows, char *chipType = NULL) {
        maxFlow = numFlows;
        if (chipType==NULL) { maxCol=3392; maxRow=3792; }
        // add chipType check in the future for smaller maxCol/maxRow
        else { maxCol=3392; maxRow=3792; }
        maxSet = true;
    }
    void print() {
        std::cout << "HashTable_xyflow.print()... filename=" << filename << " count=" << count << std::endl << std::flush;
        for(std::map<std::string, int >::const_iterator it = map_id.begin(); it != map_id.end(); ++it)
        {
            std::string key = it->first;
            int id = map_id[key];
            std::cout << key << " " << id << std::endl << std::flush;
        }
    }
    void setFilename(std::string fname) {filename=fname;}
    std::string getFilename() { return filename;}
private:
    bool maxSet;
    int maxCol;
    int maxRow;
    int maxFlow;
    int count;
    int count_xy;
    int count_mm;
    std::map <std::string, int> map_id;
    std::map <std::string, int> map_xy;
    std::map <std::string, bool> exist;
    std::map <std::string, bool> exist_xy;
    std::map <std::string, bool> mismatch;
    std::map <std::string, std::string> hp_ref;
    std::string filename;
};


// the shop for your annoying debugging needs.
class DebugMe{
public:
  int bkg_debug_files;
   int bkgModelHdf5Debug;
   int bkgModelHdf5Debug_region_r;
   int bkgModelHdf5Debug_region_c;
   int bkgDebugParam;
   int bkgDebug_nSamples;

   // only the row and col fields are used to specify location of debug regions
    std::vector<Region> BkgTraceDebugRegions;

   // temporary: dump debugging information for all beads, not just one
   int debug_bead_only;
   int region_vfrc_debug; // dump a sample region as text file for debugging.
   // trace output: options --bkg-debug-trace-sse, --bkg-debug-trace-xyflow, --bkg-debug-trace-rcflow
   bool bkgModel_xyflow_output;
   int bkgModel_xyflow_fname_in_type;
   std::string bkgModel_xyflow_fname_in;
   bool read_file_sse(HashTable_xyflow &xyfs, int numFlows=400) const;
   bool read_file_xyflow(HashTable_xyflow &xyfs, int numFlows=400) const;
   bool read_file_rcflow(HashTable_xyflow &xyfs, int numFlows=400) const;
   DebugMe();
   void PrintHelp();
   void SetOpts(OptArgs &opts, Json::Value& json_params);
};

#endif // DEBUGME_H
