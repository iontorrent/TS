/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TMAP_ERROR_H
#define TMAP_ERROR_H

#define BREAK_LINE "************************************************************\n"

#include <stdint.h>

#if defined (__cplusplus)
extern "C" 
{
#endif


/*! 
  Error handling routines.
 */

/*! 
  the type of action to be taken
  @details  the type of action to take upon the detection of an error
  */
enum {
    Exit,  /*!< exit the program */
    Warn,  /*!< print a warning  */
    LastActionType /*!< dummy action type */
};

// error reporting policy
// for compliant functions, bitwise combination flags indicate what to do if error encountered:
#define T_REPORT_ERROR 0x1
#define T_EXIT_ON_ERROR 0x2
#define T_REACT_ON_ERROR (T_REPORT_ERROR|T_EXIT_ON_ERROR)

/*! 
  the type of error
  @details  the type of error detected
  */
enum {
    OutOfRange=0,  /*!< value was out of range */
    CommandLineArgument, /*!< improper command line argument */
    ReallocMemory, /*!< memory re-allocation failure */
    MallocMemory, /*!< memory allocation failure */
    OpenFileError, /*!< could not open a file */
    CloseFileError, /*!< could not close a file */
    ReadFileError, /*!< could not read from a file */
    WriteFileError, /*!< could not write from a file */
    EndOfFile, /*!< reached the end-of-file prematurely */
    ThreadError, /*!< error starting/joining threads or mantaining sync control structure */
    SigInt, /*!< SIGINT signal caught */
    SharedMemoryGet, /*!< could not get the shared memory */
    SharedMemoryAttach, /*!< could not attach the shared memory */
    SharedMemoryControl, /*!< could not control the shared memory */
    SharedMemoryDetach, /*!< could not detach the shared memory */
    SharedMemoryListing, /*!< could not find the listing in shared memory */
    BugEncountered, /*<! a unrecoverable bug was encountered */
    LastErrorType, /*!< dummy error type  */
};

/*! 
  checks if the value falls within the bounds
  @param  val     the value to be checked
  @param  lower   the lower integer value (inclusive)
  @param  upper   the upper integer value (inclusive)
  @param  option  the option being checked 
  @details        throws a command line argument error if the value is not within the bounds
  */
void
tmap_error_cmd_check_int (int32_t val, int32_t lower, int32_t upper, char *option);

void
tmap_error_cmd_check_int_x (int32_t val, int32_t lower, int32_t upper, int32_t special, char *option);

/*! 
  checks if the 64-bit integer value falls within the bounds
  @param  val     the value to be checked
  @param  lower   the lower integer value (inclusive)
  @param  upper   the upper integer value (inclusive)
  @param  option  the option being checked
  @details        throws a command line argument error if the value is not within the bounds
  */
void
tmap_error_cmd_check_int64(int64_t val, int64_t lower, int64_t upper, char *option);

void
tmap_error_cmd_check_int64_x (int64_t val, int64_t lower, int64_t upper, int64_t special, char *option);

void
tmap_error_cmd_check_double (double val, double lower, double upper, char *option);

void
tmap_error_cmd_check_double_x (double val, double lower, double upper, double special, char *option);

/*!
  process a bug
  */
#define tmap_bug() \
  (tmap_error_full(__FILE__, __LINE__, __func__, "bug encountered", Exit, BugEncountered))

/*! 
  process an error based on the given action
  @param  variable_name  the variable name or value associated with the error
  @param  action_type    the action to be taken
  @param  error_type     the error type 
  */
#define tmap_error(variable_name, action_type, error_type) \
  (tmap_error_full(__FILE__, __LINE__, __func__, variable_name, action_type, error_type))

/*! 
  process an error based on the given action
  @param  function_name  the function name reporting the error
  @param  variable_name  the variable name or value associated with the error
  @param  action_type    the action to be taken
  @param  error_type     the error type 
  */
#define tmap_error1(function_name, variable_name, action_type, error_type) \
  (tmap_error_full(__FILE__, __LINE__, function_name, variable_name, action_type, error_type))

/*! 
  process an error based on the given action
  @param  file            the calling file (use __FILE__)
  @param  line           the line number in the calling file (use __LINE__)
  @param  function_name  the function name reporting the error
  @param  variable_name  the variable name or value associated with the error
  @param  action_type    the action to be taken
  @param  error_type     the error type 
  */
void tmap_error_full (const char *file, const unsigned int line, const char *function_name, const char *variable_name, int action_type, int error_type);

void tmap_fail (int fail, const char* fname, const char* func_name, int lineno, const char *fmt, ...);

void tmap_warn (const char* fname, const char* func_name, int lineno, const char *fmt, ...);

void tmap_user_warning (const char *fmt, ...);

void tmap_user_fileproc_msg (const char* fname, int lineno, const char *fmt, ...);

#define tmap_flagerr(flag, fmt, ...) \
    if (flag & T_REACT_ON_ERROR) \
    {\
        tmap_fail ((flag & T_EXIT_ON_ERROR), __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__); \
    }


#define tmap_conderr(quit, fmt, ...) \
    tmap_fail (quit, __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__)

#define tmap_failure(fmt, ...) \
    tmap_fail (1, __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__)

#define tmap_warning(fmt, ...) \
    tmap_warn (__FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // TMAP_ERROR_H
