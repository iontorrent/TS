/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ION_ERROR_H
#define ION_ERROR_H

#define BREAK_LINE "************************************************************\n"

#ifdef __cplusplus
extern "C" {
#endif

    /*! 
      @details  the type of action to be taken
      */
    enum {
        Exit,  /*!< exit the program */
        Warn,  /*!< print a warning  */
        LastActionType /*!< dummy action type */
    };

    /*! 
      @details  the type of error
      */
    enum {
        OutOfRange,  /*!< value was out of range */
        CommandLineArgument, /*!< improper command line argument */
        ReallocMemory, /*!< memory re-allocation failure */
        MallocMemory, /*!< memory allocation failure */
        OpenFileError, /*!< could not open a file */
        ReadFileError, /*!< could not read from a file */
        WriteFileError, /*!< could not write from a file */
        EndOfFile, /*!< reached the end-of-file prematurely */
        LastErrorType, /*!< dummy error type  */
    };

    /*! 
      process an error based on the given action
      @param  function_name  the function name reporting the error
      @param  variable_name  the variable name or value associated with the error
      @param  action_type    the action to be taken
      @param  error_type     the error type 
      */
    void 
      ion_error(const char *function_name, const char *variable_name, int action_type, int error_type);

#ifdef __cplusplus
}
#endif

#endif // ION_ERROR_H
