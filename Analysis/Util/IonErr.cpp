/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <stdexcept>
#include <stdlib.h>

#include "IonErr.h"

int ExitCode::exitCode = EXIT_SUCCESS;

IonErrStatus::IonErrStatus() {
  mDoThrow = false;
}

IonErrStatus &IonErr::GetIonErrStatus() {
  static IonErrStatus ionStatus;
  return ionStatus;
}

void IonErr::SetThrowStatus(bool doThrow) {
  IonErrStatus &ionstats = GetIonErrStatus();
  ionstats.mDoThrow = doThrow;
}

bool IonErr::GetThrowStatus() {
  IonErrStatus &ionstats = GetIonErrStatus();
  return ionstats.mDoThrow;
}

void IonErr::Abort(const std::string &file, 
		   int line,
		   const std::string &msg,
		   int code) {
  std::cerr << "Error: '" << msg << "' in file: '" << file << "' at line: " << line << " code: " << code << std::endl;
  bool doThrow = GetThrowStatus();
  if (doThrow) {
    throw std::runtime_error(msg);
  }
  else {
    exit (code);
  }
}

void IonErr::Warn(const std::string &file, 
									int line,
									const std::string &msg) {
  std::cerr << "*Warning*: '" << msg << "' on in file: '" << file << "' at line: " << line << std::endl;
}

void IonErr::Abort(const std::string &file, 
		   int line,
		   const std::string &cond,
		   const std::string &msg) {
  std::string err = msg + " (" + cond + ")";
  Abort(file, line, err);
}
