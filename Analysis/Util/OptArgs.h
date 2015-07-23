/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/**
 * Option argument parsing for the command line... 
 */
#ifndef OPTARGS_H
#define OPTARGS_H

#include <map>
#include <string>
#include <vector>
#include <iostream>

/** 
 * Class to represent a single option on the command line
 * and any associated options. 
 */ 
class OptArgument {

public:
    
  OptArgument();
    
  std::string mShortName; ///< Single character switch, '-' for no short switch
  std::string mLongName;  ///< Long style --my-option type name (without the -- prepended.
  std::vector<std::string> mValues; ///< String values of this option as passed in
  std::string mDefault;   ///< Default value of this option if set.
  std::string mSource;    ///< Where did this option come from? 
  std::string mHelp;      ///< Brief description of option and function
  bool mRequiresArg;      ///< Requires argument?
  int mQueried;           ///< Has this been asked for yet?
    
};

/**
 * Class for parsing arguments to programs. Currently supports command line
 * but extendible to specification files, environment variables etc.
 */
class OptArgs {
    
public:
      
  /** 
   * Read any arguments up to a bare '--' after which all arguments
   * will be ignored. Original data in argc and argv are left unchanged. 
   */
  void ParseCmdLine(int argc, const char *argv[]);

  /** 
   * Check if a given option was provided
   */
  bool HasOption(char shortOption, const std::string &longOption);

  /**
   * Get a string option value for the given short/long key
   */ 
  void GetOption(std::string &value, const std::string &defaultValue,
		 char shortOption, const std::string &longOption);

  /**
   * Get a boolean option value for the given short/long key
   */ 
  void GetOption(bool &value, const std::string &defaultValue,
		 char shortOption, const std::string &longOption);

  /**
   * Get a double option value for the given short/long key
   */ 
  void GetOption(double &value, const std::string &defaultValue,
		 char shortOption, const std::string &longOption);

  /**
   * Get a int option value for the given short/long key
   */ 
  void GetOption(int &value, const std::string &defaultValue,
		 char shortOption, const std::string &longOption);

  /**
   * Get an unsigned int option value for the given short/long key
   */ 
  void GetOption(unsigned int &value, const std::string &defaultValue,
		 char shortOption, const std::string &longOption);

 /**
  * Get a long option value for a given short/long key
  */
	void GetOption(long &value, const std::string &defaultValue,
				   char shortOption, const std::string &longOption);
  /**
   * Get the int option value for possibly multiple comma separated values
   */ 
  void GetOption(std::vector<int> &value, const std::string &defaultValue,
		 char shortOption, const std::string &longOption, char sep=',');

  /**
   * Get the unsigned int option value for possibly multiple comma separated values
   */ 
  void GetOption(std::vector<unsigned int> &value, const std::string &defaultValue,
		 char shortOption, const std::string &longOption, char sep=',');


  /**
   * Get the double option value for possibly multiple comma separated values
   */ 
  void GetOption(std::vector<double> &value, const std::string &defaultValue,
		 char shortOption, const std::string &longOption, char sep=',');

  /**
   * Get the double option value for possibly multiple comma separated values
   */ 
  void GetOption(std::vector<std::string> &value, const std::string &defaultValue,
		 char shortOption, const std::string &longOption, char sep=',');


  /**
   * Get a boolean option value for the given short/long key
   * If the option appears multiple times, use the earliest occurrence
   */
  bool GetFirstBoolean(char shortOption, const std::string &longOption, bool defaultValue);

  /**
   * Get a boolean option value for the given short/long key
   * If the option appears multiple times, use the earliest occurrence
   */
  bool GetFirstBoolean(char shortOption, const std::string &longOption, const char * defaultValue);

  /**
   * Get a string option value for the given short/long key
   * If the option appears multiple times, use the earliest occurrence
   */
  std::string GetFirstString(char shortOption, const std::string &longOption, const std::string &defaultValue);

  /**
   * Get a double option value for the givent short/long key
   * If the option appears multiple times, use the earliest occurrence
   */
  double GetFirstDouble(char shortOption, const std::string &longOption, double defaultDouble);

  /**
   * Get a int option value for the given short/long key
   * If the option appears multiple times, use the earliest occurrence
   */
  int GetFirstInt(char shortOption, const std::string &longOption, int defaultInt);

  /**
   * Get a vector of 'sep' separated string option values for the given short/long key
   * If the option appears multiple times, use the earliest occurrence
   */
  std::vector<std::string> GetFirstStringVector(char shortOption, const std::string &longOption, const std::string &defaultValue, char sep=',');

  /**
   * Get a vector of 'sep' separated double option values for the given short/long key
   * If the option appears multiple times, use the earliest occurrence
   */
  std::vector<double> GetFirstDoubleVector(char shortOption, const std::string &longOption, const std::string &defaultValue, char sep=',');

  /**
   * Get a vector of 'sep' separated int option values for the given short/long key
   * If the option appears multiple times, use the earliest occurrence
   */
  std::vector<int> GetFirstIntVector(char shortOption, const std::string &longOption, const std::string &defaultValue, char sep=',');



  void StringToIntVector(std::vector<int> & values, const std::string & value_string, const std::string &longOption, char sep=',');

  void StringToDoubleVector(std::vector<double> & values, const std::string & value_string, const std::string &longOption, char sep=',');


  /**
   * Fill in the options that the user supplied (possibly in error) that
   * were not ever queried by the program.
   */
  void GetUncheckedOptions(std::vector<std::string> &unchecked); 

  /**
   * Print the options specified on the command line. 
   */
  void PrintOptions(std::ostream &out); 

  /** 
   * Get non-argument/option command line tokens
   */
  void GetLeftoverArguments(std::vector<std::string> &leftover);

  /**
   * Check to make sure that there are no uncalled options (e.g. misspelled)
   */ 
  void CheckNoLeftovers();

protected:

  /**
   * Determine if string is a valid option or parameter to an option
   */
  bool IsOption(const std::string &name) const;

  /**
   * Parse out a long option from the command line arguments, incrementing the index into
   * argv as necessary
   */ 
  void HandleLongOption(std::string &option, int &index, int argc, const char *argv[]);

  /**
   * Parse out some short options from the command line
   * arguments, incrementing the index into argv as
   * necessary
   */ 
  void HandleShortOption(std::string &option, int &index, int argc, const char *argv[]);

  /**
   * Print a useful message and quit
   */
  void Abort(const std::string &msg);

  /**
   * Find the option object associated with these options,
   * if it doesn't exist then create it and return a pointer
   */
  OptArgument *GetOptArgument(char shortOption, const std::string &longOption,
			      const std::string &defaultValue);

private:
    
  std::map<std::string, OptArgument> mSeenOpts;
  std::vector<std::string> mNonOpts;
  std::vector<std::string> mOrigArs;
  std::string mExecutable;
};
  
#endif // OPTARGS_H 
