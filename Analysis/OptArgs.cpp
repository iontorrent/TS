/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdlib.h>
#include <errno.h>
#include "OptArgs.h"
#ifndef ALIGNSTATS_IGNORE
#include "FlowDiffStats.h"
#endif

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
  for (int i = 1; i < argc; i++) {
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
      opt->mValues[0] == "true") {
    value = true;
  }
  else if (opt->mValues[0] == "FALSE" ||
	   opt->mValues[0] == "false") {
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

#ifndef ALIGNSTATS_IGNORE
void OptArgs::GetOption(std::vector<double> &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  vector<string> words;
  FlowDiffStats::ChopLine(words, opt->mValues[0], ',');
  value.clear();
  for (size_t i = 0; i < words.size(); i++) {
    char *end;
    int err = errno;
    errno = 0;
    value.push_back(strtod(words[i].c_str(), &end));
    if (errno != 0 || *end != '\0') {
      Abort("Error converting: " + words[i] + " to an double for option: " + longOption);
    }
    errno = err;    
  }
}



void OptArgs::GetOption(std::vector<int> &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  vector<string> words;
  FlowDiffStats::ChopLine(words, opt->mValues[0], ',');
  value.clear();
  for (size_t i = 0; i < words.size(); i++) {
    char *end;
    int err = errno;
    errno = 0;
    value.push_back(strtol(words[i].c_str(), &end, 10));
    if (errno != 0 || *end != '\0') {
      Abort("Error converting: " + words[i] + " to an double for option: " + longOption);
    }
    errno = err;    
  }
}

void OptArgs::GetOption(std::vector<unsigned int> &value, const std::string &defaultValue,
			char shortOption, const std::string &longOption) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  vector<string> words;
  FlowDiffStats::ChopLine(words, opt->mValues[0], ',');
  value.clear();
  for (size_t i = 0; i < words.size(); i++) {
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
			char shortOption, const std::string &longOption) {
  OptArgument *opt = GetOptArgument(shortOption, longOption, defaultValue);
  if (opt->mValues.size() == 0) {
    Abort("No argument specified for option: " + longOption);
  }
  if (opt->mValues.size() > 1) {
    Abort("Multiple arguments specified for option: " + longOption + 
	  " use different api or specify single option");
  }
  FlowDiffStats::ChopLine(value, opt->mValues[0], ',');
}
#endif

/**
 * Get a int option value for the givent short/long key
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
 * Get an unsigned int option value for the givent short/long key
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
 * Fill in the options that the user supplied (possibly in error) that
 * were not ever queried by the program.
 */
void OptArgs::GetUncheckedOptions(std::vector<std::string> &unchecked) {
  unchecked.clear();
  std::map<std::string, OptArgument>::iterator it;
  for (it = mSeenOpts.begin(); it != mSeenOpts.end(); it++) {
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
  for (it = mSeenOpts.begin(); it != mSeenOpts.end(); it++) {
    out << it->first << "=";
    if (it->second.mValues.size() == 0) {
      out << std::endl;
    }
    for (size_t i = 0; i < it->second.mValues.size(); i++) {
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
 * Parse out a long option from the command line arguments, incrementing the index into
 * argv as necessary
 */ 
void OptArgs::HandleLongOption(std::string &option, int &index, int argc, const char *argv[]) {
  std::string next;
      
  // Pull off any leading '-' or '--'
  size_t start = option.find_first_not_of('-');
  if (start > 2) {
    Abort("Can't have an option start with '---': '" + option + "'");
  }
  if (start == 1) {
    Abort("Long option have to start with at least one  with '--': '" + option + "'");
  }
  option.erase(0,start);

  std::map<std::string, OptArgument>::iterator it = mSeenOpts.find(option);
  OptArgument opt;
  opt.mLongName = option;
  if (it != mSeenOpts.end()) {
    opt = it->second;
  }
  if (index+1 < argc) {
    next = argv[index+1];
    if (next.find('-') == 0) {
      // Another option, just leave it...
    }
    else {
      opt.mValues.push_back(next);
      index++;
    }
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
  for (size_t i = 0; i < option.length(); i++) {
    std::string shortOption(1, option[i]);
    std::map<std::string, OptArgument>::iterator it = mSeenOpts.find(shortOption);
    OptArgument opt;
    opt.mShortName = std::string(1, option[i]);
    if (it != mSeenOpts.end()) {
      opt = it->second;
    }
    if (opt.mRequiresArg || (index +1 < argc && argv[index+1][0] != '-')) {
      if (index + 1 >= argc) {
	    
	Abort("Option: -" + opt.mShortName + ",--" + opt.mLongName + " requires an argument");
      }
      std::string next  = argv[index + 1];
      if (next.find('-') == 0) {
	Abort("Arguments to options can't start with '-'");
      }
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
    for (size_t i = 0; i < unchecked.size(); i++) {
      unknown += unchecked[i] + " ";
    }
    Abort("Unknown options: " + unknown);
  }

  vector<string> leftovers;
  GetLeftoverArguments(leftovers);
  if (!leftovers.empty()) {
    string unknown;
    for (size_t i = 0; i < leftovers.size(); i++) {
      unknown += leftovers[i] + " ";
    }
    Abort("Unknown arguments: " + unknown);
  }

}
