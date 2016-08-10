/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_cmdline_parser_h__
#define __test_cmdline_parser_h__
#include <extesting.h>
#include <process_params.h>


extern const char* TESTPROCNAME;
extern const char* TESTHDR;

class TestCmdlineParser : public Test
{
protected:
    // structures for cmdline testing
    class TestParams : public Process_params
    {
        std::string kstr_;
        bool kbool_;
        int kint_;
        std::string farg_;
        std::string oarg_;
        std::vector<std::string> rargs_;
        std::string larg_;

    protected:
        bool prepareCmdlineFormat ();
        bool prepareParameters ();
        bool interpreteParameters ();

    public:
        TestParams (const char* header = TESTHDR, const char* procname = TESTPROCNAME)
        :
        Process_params (header, procname)
        {}

        // get functions

        const char* kstr   () const {return kstr_.c_str ();}
        bool        kbool  () const {return kbool_;}
        int         kint   () const {return kint_;}
        const char* farg   () const {return farg_.c_str ();}
        const char* oarg   () const {return oarg_.c_str ();}
        int         rargno () const {return rargs_.size ();}
        const char* rarg   (int rno) const {return rargs_ [rno].c_str ();}
        const char* larg   () const {return larg_.c_str ();}

        // set functions

        void        kstr        (const char* opt) {kstr_ = opt;}
        void        kbool       (bool opt)        {kbool_ = opt;}
        void        kint        (int  opt)        {kint_ = opt;}
        void        farg        (const char* opt) {farg_ = opt;}
        void        oarg        (const char* opt) {oarg_ = opt;}
        void        rargs       (const char* opt);
        void        larg        (const char* opt) {larg_ = opt;}
    };
public:
    TestCmdlineParser () : Test ("CmdLine: Command line parser") {}
    bool process ();
};
#endif // __test_cmdline_parser_h__
