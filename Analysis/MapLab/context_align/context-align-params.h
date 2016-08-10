/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __context_align_params_h__
#define __context_align_params_h__

#include <string>
#include <set>
#include "process_params.h"

extern const char* CONTAL_HDR;
extern const char* CONTAL_PROCNAME;

class ContalignParams : public Process_params
{
private:
    int gip_;
    int gep_;
    int mat_;
    int mis_;
    unsigned bwid_;
    std::string floworder_;
    std::string inbam_;
    std::string outbam_;
    std::string bamidx_;
    std::string refname_;
    int refno_;
    unsigned begpos_;
    unsigned endpos_;
    unsigned skip_;
    unsigned limit_;
    std::string logfname_;
    std::string logopts_;

    std::set <std::string> logspec_;
    bool interprete_logopts ();
    bool interprete_algorithm ();


protected:
    bool prepareCmdlineFormat ();
    bool prepareParameters ();
    bool interpreteParameters ();

public:
    ContalignParams (const char* header = CONTAL_HDR, const char* procname = CONTAL_PROCNAME)
    :
    Process_params (header, procname)
    {}
    ~ContalignParams () {}

    // get functions
    int         gip     () const {return gip_;}
    int         gep     () const {return gep_;}
    int         mat     () const {return mat_;}
    int         mis     () const {return mis_;}
    unsigned    bwid    () const {return bwid_;}
    const char* ord     () const {return floworder_.c_str ();}
    const char* inbam   () const {return inbam_.c_str ();}
    const char* outbam  () const {return outbam_.c_str ();}
    const char* bamidx  () const {return bamidx_.c_str ();}
    const char* refname () const {return refname_.c_str ();}
    int         refno   () const {return refno_;}
    unsigned    begpos  () const {return begpos_;}
    unsigned    endpos  () const {return endpos_;}
    unsigned    skip    () const {return skip_;}
    unsigned    limit   () const {return limit_;}
    const char* logfname() const {return logfname_.c_str ();}
    const char* logopts () const {return logopts_.c_str ();}

    // set functions
    void        gip     (int op) {gip_ = op;}
    void        gep     (int op) {gep_ = op;}
    void        mat     (int op) {mat_ = op;}
    void        mis     (int op) {mis_ = op;}
    void        bwid    (unsigned op) {bwid_ = op;}
    void        ord     (const char* op) {floworder_ = op;}
    void        inbam   (const char* op) {inbam_ = op;}
    void        outbam  (const char* op) {outbam_ = op;}
    void        bamidx  (const char* op) {bamidx_ = op;}
    void        refname (const char *op) {refname_ = op;}
    void        refno   (int op) {refno_ = op;}
    void        begpos  (unsigned op) {begpos_ = op;}
    void        endpos  (unsigned op) {endpos_ = op;}
    void        skip    (unsigned op) {skip_ = op;}
    void        limit   (unsigned op) {limit_ = op;}
    void        logfname(const char* op) {logfname_ = op;}
    void        logopts (const char* op) {logopts_ = op;}

    // readonly access functions
    bool        logging (const char* log_option) const;

    // defaults
    const char* gip_default () const;
    const char* gep_default () const;
    const char* mat_default () const;
    const char* mis_default () const;
    const char* bwid_default () const;
    const char* ord_default () const;
    const char* bamidx_default () const;
    const char* refname_default () const;
    const char* refno_default () const;
    const char* begpos_default () const;
    const char* endpos_default () const;
    const char* skip_default () const;
    const char* limit_default () const;
    const char* logfname_default () const;
    const char* logopts_default () const;

    // help
    const char* gip_help () const;
    const char* gep_help () const;
    const char* mat_help () const;
    const char* mis_help () const;
    const char* bwid_help () const;
    const char* ord_help () const;
    const char* inbam_help () const;
    const char* outbam_help () const;
    const char* bamidx_help () const;
    const char* refname_help () const;
    const char* refno_help () const;
    const char* begpos_help () const;
    const char* endpos_help () const;
    const char* skip_help () const;
    const char* limit_help () const;
    const char* logfname_help () const;
    const char* logopts_help () const;

};


#endif // __context_align_params_h__
