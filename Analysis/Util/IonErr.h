/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef IONERR_H
#define IONERR_H
#include <string>
#include <stdlib.h>
#include "assert.h"
                                      
#define ION_ABORT(_Msg) { IonErr::Abort(__FILE__,__LINE__,_Msg); }

#define ION_ABORT_CODE(_Msg,_Code) { IonErr::Abort(__FILE__,__LINE__,_Msg,_Code); }

#define ION_WARN(_Msg) { IonErr::Warn(__FILE__,__LINE__,_Msg); }
                                      
#ifdef ION_DEBUG
#define ION_ASSERT(_Cond,_Msg) { if (!(_Cond)) { IonErr::Warn(__FILE__,__LINE__,_Msg); assert(_Cond);} }
#else
#define ION_ASSERT(_Cond,_Msg) { if (!(_Cond)) { IonErr::Abort(__FILE__,__LINE__,#_Cond,_Msg); } }
#endif

/** Class containing error handling configuration. */
class IonErrStatus {
public:
  IonErrStatus(); 
  bool mDoThrow;
};

/** Central location for some error handling. */
class IonErr {

 public:

  /** Get the global configuration. */
  static IonErrStatus &GetIonErrStatus();

  /** Change the throw exception/call exit status */
  static void SetThrowStatus(bool doThrow);

  /** Are we configured to throw and exception? */
  static bool GetThrowStatus();

  /** Issue a warning. */
  static void Warn(const std::string &file, 
		   int line,
		   const std::string &msg);
	
  /** Abort and throw exception or exit depending on configuration. */
  static void Abort(const std::string &file, 
		    int line,
		    const std::string &msg,
		    int code=EXIT_FAILURE);

  /** Wrapper for abort from assert with conditional. */
  static void Abort(const std::string &file, 
		    int line,
		    const std::string &cond,
		    const std::string &msg);
  
};

class ExitCode{
    static int exitCode;
public:
    static void UpdateExitCode( int newCode ){
        //Failure code overrides success, but not vice versa
        exitCode = (newCode==EXIT_FAILURE)?EXIT_FAILURE:exitCode;
    }
    static int GetExitCode(void) { return exitCode; }
};

#endif // IONERR_H
