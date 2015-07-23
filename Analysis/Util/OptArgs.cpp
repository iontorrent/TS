/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdlib.h>
#include <errno.h>
#include "OptArgs.h"
#include "Utils.h"

using namespace std;

/** Constructor. */
OptArgument::OptArgument() {
  mShortName = "-";
  mSource = "unset";
  mQueried = 0;
  mRequiresArg = false;
}


/** 
 * Read any arguments up to a bare '--' after which all arguments
 * will be ignored. Original data in argc and argv are left unchanged. 
 */
void OptArgs::ParseCmdLine(int argc, const char *argv[]) {
  if (argc == 0) {
    Abort("Have to have at least one argument (the executable name)");
  }
  mExecutable = argv[0];
  for (int i = 1; i < argc; ++i) {
    std::string option = argv[i];
    size_t start = option.find_first_not_of('-');
    if (option == "--") {
      break; // ignore everything after a '--' string
    }
    if (start == 0) {
      mNonOpts.push_back(option);
    }
    else if (start == 1) {
      HandleShortOption(option, i, argc, argv);
    }
    else if (start == 2) {
      HandleLongOption(option, i, argc, argv);
    }
    else {
      Abort("Malformed option with more than '--' prefix: " + option);
    }
  }
}


/** 
 * Check if a given option was provided
 */ 
bool OptArgs::HasOption(char shortOption, const std::string &longOption) {

  std::string shortString = std::string(1, shortOption);
  if (shortOption != '-' && mSeenOpts.find(shortString) != mSeenOpts.end())
    return true;
  if (mSeenOpts.find(longOption) != mSeenOpts.end())
    return true;

  return false;
}


/**
 * Get a string option value for the given short/long key
 */
void OptArgs::GetOption(std::string &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  // 
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  value = opt->mValues[0];
}


/**
 * Get a boolean option value for the givent short/long key
 */ 
void OptArgs::GetOption(bool &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  // 
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }

  if (opt->mValues.size() == 0 || 
      opt->mValues[0] == "TRUE" || 
      opt->mValues[0] == "true" ||
      opt->mValues[0] == "on" ||
      opt->mValues[0] == "ON") {
    value = true;
  }
  else if (opt->mValues[0] == "FALSE" ||
	    opt->mValues[0] == "false" ||
      opt->mValues[0] == "off" ||
      opt->mValues[0] == "OFF") {
    value = false;
  }
  else {
    Abort("Don't recognize value: " + opt->mValues[0] + " for boolean");
  }
}


/**
 * Get a double option value for the givent short/long key
 */ 
void OptArgs::GetOption(double &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  char *end;
  int err = errno;
  errno = 0;
  value = strtod(opt->mValues[0].c_str(), &end);
  if (errno != 0 || *end != '\0') {
    Abort("Error converting: " + opt->mValues[0] + " to an double for option: " + longOption);
  }
  errno = err;
}


void OptArgs::GetOption(long &value, const std::string &defaultValue,
						char shortOption, const std::string &longOption) {
	OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
	if (opt->mValues.size() == 0) {
		Abort("No argument specified for option: " + longOption);
	}
	if (opt->mValues.size() > 1) {
		Abort("Multiple arguments specified for option: " + longOption + 
			  " use different api or specify single option");
	}
	char *end;
	int err = errno;
	errno = 0;
	value = strtol(opt->mValues[0].c_str(), &end,10);
	if (errno != 0 || *end != '\0') {
		Abort("Error converting: " + opt->mValues[0] + " to an long for option: " + longOption);
	}
	errno = err;
}


void OptArgs::GetOption(std::vector<double> &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption, char sep) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  if(opt->mValues[0] == "")
    return;

  StringToDoubleVector(value, opt->mValues[0], longOption, sep);
}


void OptArgs::GetOption(std::vector<int> &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption,char sep) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  if(opt->mValues[0] == "")
    return;

  StringToIntVector(value, opt->mValues.at(0), longOption, sep);
}


void OptArgs::GetOption(std::vector<unsigned int> &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption, char sep) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  vector<string> words;
  value.clear();
  if(opt->mValues[0] == "")
    return;
  split(opt->mValues[0],sep,words);
  for (size_t i = 0; i < words.size(); ++i) {
    char *end;
    int err = errno;
    errno = 0;
    value.push_back((unsigned int) strtol(words[i].c_str(), &end, 10));
    if (errno != 0 || *end != '\0') {
      Abort("Error converting: " + words[i] + " to an unsigned int for option: " + longOption);
    }
    errno = err;    
  }
}

void OptArgs::GetOption(std::vector<std::string> &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption, char sep) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  value.clear();
  if(opt->mValues[0] == "")
    return;
  split(opt->mValues[0],sep,value);
}


/**
 * Get a int option value for the given short/long key
 */ 
void OptArgs::GetOption(int &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  char *end = NULL;
  int err = errno;
  errno = 0;
  value = strtol(opt->mValues[0].c_str(), &end, 10); 
  if (errno != 0 || *end != '\0') {
    Abort("Error converting: " + opt->mValues[0] + " to an integer for option: " + longOption);
  }
  errno = err;
}

/**
 * Get an unsigned int option value for the given short/long key
 */ 
void OptArgs::GetOption(unsigned int &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  char *end = NULL;
  int err = errno;
  errno = 0;
  value = strtoul(opt->mValues[0].c_str(), &end, 10); 
  if (errno != 0 || *end != '\0') {
    Abort("Error converting: " + opt->mValues[0] + " to an unsigned integer for option: " + longOption);
  }
  errno = err;
}


/**
 * Get a boolean option value for the given short/long key
 * If the option appears multiple times, use the earliest occurrence
 */
bool OptArgs::GetFirstBoolean(char shortOption, const std::string &longOption, bool defaultValue)
{
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue ? "true" : "false");

  if (opt->mValues.size() == 0)
    return true;
  if (opt->mValues[0] == "TRUE" or opt->mValues[0] == "true" or opt->mValues[0] == "ON" or opt->mValues[0] == "on")
    return true;
  if (opt->mValues[0] == "FALSE" or opt->mValues[0] == "false" or opt->mValues[0] == "OFF" or opt->mValues[0] == "off")
    return false;

  Abort("Don't recognize value: " + opt->mValues[0] + " for boolean");
  return false; // Never actually reached
}


/**
 * Get a boolean option value for the given short/long key
 * If the option appears multiple times, use the earliest occurrence
 */
bool OptArgs::GetFirstBoolean(char shortOption, const std::string &longOption, const char * defaultValue)
{
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);

  if (opt->mValues.size() == 0)
    return true;
  if (opt->mValues[0] == "TRUE" or opt->mValues[0] == "true" or opt->mValues[0] == "ON" or opt->mValues[0] == "on")
    return true;
  if (opt->mValues[0] == "FALSE" or opt->mValues[0] == "false" or opt->mValues[0] == "OFF" or opt->mValues[0] == "off")
    return false;

  Abort("Don't recognize value: " + opt->mValues[0] + " for boolean");
  return false; // Never actually reached
}


/**
 * Get a string option value for the given short/long key
 * If the option appears multiple times, use the earliest occurrence
 */
std::string OptArgs::GetFirstString(char shortOption, const std::string &longOption, const std::string &defaultValue)
{
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0)
    Abort("No argument specified for option: " + longOption);

  return opt->mValues[0];
}


/**
 * Get a string vector option value for the given short/long key
 * If the option appears multiple times, use the earliest occurrence
 */
std::vector<std::string> OptArgs::GetFirstStringVector(char shortOption, const std::string &longOption, const std::string &defaultValue, char sep)
{
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  vector<string> value;
  split(opt->mValues[0],sep,value);
  return value;
}


/**
 * Get a double option value for the givent short/long key
 * If the option appears multiple times, use the earliest occurrence
 */
double OptArgs::GetFirstDouble(char shortOption, const std::string &longOption, double defaultDouble)
{
  char defaultValue[64];
  snprintf(defaultValue, 64, "%lf", defaultDouble);
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  char *end;
  int err = errno;
  errno = 0;
  double value = strtod(opt->mValues[0].c_str(), &end);
  if (errno != 0 || *end != '\0') {
    Abort("Error converting: " + opt->mValues[0] + " to an double for option: " + longOption);
  }
  errno = err;
  return value;
}


/**
 * Get a vector of 'sep' separated double option values for the given short/long key
 * If the option appears multiple times, use the earliest occurrence
 */
std::vector<double> OptArgs::GetFirstDoubleVector(char shortOption, const std::string &longOption, const std::string &defaultValue, char sep)
{
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
      Abort("No argument specified for option: " + longOption);
  }

  vector<double> values;
  if(opt->mValues[0] == "")
    return (values);

  StringToDoubleVector(values, opt->mValues[0], longOption, sep);
  return (values);
}


/**
 * Get a int option value for the given short/long key
 * If the option appears multiple times, use the earliest occurrence
 */
int OptArgs::GetFirstInt(char shortOption, const std::string &longOption, int defaultInt)
{
  char defaultValue[64];
  snprintf(defaultValue, 64, "%d", defaultInt);
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  char *end = NULL;
  int err = errno;
  errno = 0;
  int value = strtol(opt->mValues[0].c_str(), &end, 10);
  if (errno != 0 || *end != '\0') {
    Abort("Error converting: " + opt->mValues[0] + " to an integer for option: " + longOption);
  }
  errno = err;
  return value;
}


/**
 * Get a vector of 'sep' separated double option values for the given short/long key
 * If the option appears multiple times, use the earliest occurrence
 */
std::vector<int> OptArgs::GetFirstIntVector(char shortOption, const std::string &longOption, const std::string &defaultValue, char sep)
{
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }

  vector<int> values;
  if (opt->mValues[0] == "")
    return (values);

  StringToIntVector(values, opt->mValues.at(0), longOption, sep);
  return (values);
}


/*
 * Convert a string into a vector of integer values
 */
void OptArgs::StringToIntVector(std::vector<int> & values, const std::string & value_string, const std::string &longOption, char sep)
{
  vector<string> words;
  values.clear();
  split(value_string, sep, words);
  for (size_t i = 0; i < words.size(); ++i) {
    char *end;
    int err = errno;
    errno = 0;
    values.push_back(strtol(words[i].c_str(), &end, 10));
    if (errno != 0 || *end != '\0') {
      Abort("Error converting: " + words[i] + " to an double for option: " + longOption);
    }
    errno = err;
  }
}


/*
 * Convert a string into a vector of integer values
 */
void OptArgs::StringToDoubleVector(std::vector<double> & values, const std::string & value_string, const std::string &longOption, char sep)
{
  vector<string> words;
  values.clear();
  split(value_string,sep,words);
  for (size_t i = 0; i < words.size(); ++i) {
    char *end;
    int err = errno;
    errno = 0;
    values.push_back(strtod(words[i].c_str(), &end));
    if (errno != 0 || *end != '\0') {
      Abort("Error converting: " + words[i] + " to an double for option: " + longOption);
    }
    errno = err;
  }
}


/**
 * Fill in the options that the user supplied (possibly in error) that
 * were not ever queried by the program.
 */
void OptArgs::GetUncheckedOptions(std::vector<std::string> &unchecked) {
  unchecked.clear();
  std::map<std::string, OptArgument>::iterator it;
  for (it = mSeenOpts.begin(); it != mSeenOpts.end(); ++it) {
    if (it->second.mQueried == 0) {
      unchecked.push_back(it->second.mLongName);
    }
  }
}

/**
 * Print the options specified on the command line. 
 */
void OptArgs::PrintOptions(std::ostream &out) {
  std::map<std::string, OptArgument>::iterator it;
  for (it = mSeenOpts.begin(); it != mSeenOpts.end(); ++it) {
    out << it->first << "=";
    if (it->second.mValues.size() == 0) {
      out << std::endl;
    }
    for (size_t i = 0; i < it->second.mValues.size(); ++i) {
      out << it->second.mValues[i];
      if (i + 1 >= it->second.mValues.size()) {
	out << std::endl;
      }
      else {
	out << ",";
      }
    }
  }
}

/** 
 * Get non-argument/option command line tokens
 */
void OptArgs::GetLeftoverArguments(std::vector<std::string> &leftover) {
  leftover.resize(mNonOpts.size());
  copy(mNonOpts.begin(), mNonOpts.end(), leftover.begin());
}

/**
 * Determine if string is a valid option or parameter to an option
 */
bool OptArgs::IsOption(const std::string &name) const {
  if (name.length() > 0 && name[0] != '-') {
    return false;
  }
  // negative number
  else if (name.length() > 1 && name[0] == '-' && (isdigit(name[1]) || name[1] == '.')) {
    return false;
  }
  else if (name.length() > 1 && name[0] == '-') {
    return true;
  }
  fprintf(stderr, "Can't determine if name is an option: '%s'\n", name.c_str());
  exit(1);
  return false;
}

/**
 * Parse out a long option from the command line arguments, incrementing the index into
 * argv as necessary
 */ 
void OptArgs::HandleLongOption(std::string &option, int &index, int argc, const char *argv[]) {
      
  // Pull off any leading '-' or '--'
  size_t start = option.find_first_not_of('-');
  if (start > 2) {
    Abort("Can't have an option start with '---': '" + option + "'");
  }
  if (start == 1) {
    Abort("Long option have to start with at least one  with '--': '" + option + "'");
  }
  option.erase(0,start);

  // Support for --option=value format
  std::string next;
  size_t equalPos = option.find('=');
  if (equalPos == std::string::npos) { // No = sign found
    if (index+1 < argc) {
      if (!IsOption(argv[index+1])) { // Check if this is not just another option
        next = argv[index+1];
        index++;
      }
    }
  } else {  // = sign present
    if ((equalPos+1) < option.length()) {
      next = option.substr(equalPos+1);
    }
    option.erase(equalPos);
  }

  std::map<std::string, OptArgument>::iterator it = mSeenOpts.find(option);
  OptArgument opt;
  opt.mLongName = option;
  if (it != mSeenOpts.end()) {
    opt = it->second;
  }
  if (!next.empty()) {
    opt.mValues.push_back(next);
  }
  mSeenOpts[option] = opt;
}

/**
 * Parse out some short options from the command line
 * arguments, incrementing the index into argv as
 * necessary
 */ 
void OptArgs::HandleShortOption(std::string &option, int &index, int argc, const char *argv[]) {
  size_t start = option.find_first_not_of('-');
  if (start != 1) {
    Abort("Short options must start with a single '-': '" + option + "'");
  }
  option.erase(0,start);
  for (size_t i = 0; i < option.length(); ++i) {
    std::string shortOption(1, option[i]);
    std::map<std::string, OptArgument>::iterator it = mSeenOpts.find(shortOption);
    OptArgument opt;
    opt.mShortName = std::string(1, option[i]);
    if (it != mSeenOpts.end()) {
      opt = it->second;
    }
    if (opt.mRequiresArg || (index +1 < argc && !IsOption(argv[index+1]))) {
      if (index + 1 >= argc) {
	Abort("Option: -" + opt.mShortName + ",--" + opt.mLongName + " requires an argument");
      }
      std::string next  = argv[index + 1];
      index++;
      opt.mValues.push_back(next);
    }
    mSeenOpts[shortOption] = opt;
  }
}

/**
 * Print a useful message and quit
 */
void OptArgs::Abort(const std::string &msg) {
  std::cerr << "Error - " << msg << std::endl;
  exit(1);
}

/**
 * Find the option object associated with these options,
 * if it doesn't exist then create it and return a pointer
 */
OptArgument *OptArgs::GetOptArgument(char shortOption, const std::string &longOption,
				     const std::string &defaultValue) {
  OptArgument *opt = NULL;
  // Figure out source of argument: short option, long option or default;
  std::string shortString = std::string(1, shortOption);
  if (shortOption != '-' && mSeenOpts.find(shortString) != mSeenOpts.end()) {
    opt = &(mSeenOpts.find(shortString)->second);
    if (mSeenOpts.find(longOption) != mSeenOpts.end()) {
      Abort("Both short option " + std::string(1, shortOption) + " and long option for " +
	    longOption + " found.");
    }
  }
  else if (opt == NULL) {
    if (mSeenOpts.find(longOption) != mSeenOpts.end()) {
      opt = &(mSeenOpts.find(longOption)->second);

    }
    else {
      OptArgument optArg;
      optArg.mShortName = std::string(1,shortOption);
      optArg.mLongName = longOption;
      optArg.mDefault = defaultValue;
      optArg.mValues.push_back(defaultValue);
      optArg.mSource = "default-query";
      mSeenOpts[longOption] = optArg;
      opt = &(mSeenOpts.find(longOption)->second);
    }
  }
  opt->mQueried++;
  return opt;
}

void OptArgs::CheckNoLeftovers() {
  vector<string> unchecked;
  GetUncheckedOptions(unchecked);
  if (!unchecked.empty()) {
    string unknown;
    for (size_t i = 0; i < unchecked.size(); ++i) {
      unknown += unchecked[i] + " ";
    }
    Abort("Unknown options: " + unknown);
  }

  vector<string> leftovers;
  GetLeftoverArguments(leftovers);
  if (!leftovers.empty()) {
    string unknown;
    for (size_t i = 0; i < leftovers.size(); ++i) {
      unknown += leftovers[i] + " ";
    }
    Abort("Unknown arguments: " + unknown);
  }

}
