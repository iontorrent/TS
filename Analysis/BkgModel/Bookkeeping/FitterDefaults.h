/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef FITTERDEFAULTS_H
#define FITTERDEFAULTS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "json/json.h"
#include "BkgMagicDefines.h"
#include "Serialization.h"

// this should have more entries, as it is the defaults for multi-flow fit
// currently just clonal call penalty is unique here, but it might need to have region-param fit entries as well

struct FitterDefaults{
  // weighting applied to clonal restriction on different hp lengths
  float clonal_call_scale[MAGIC_CLONAL_CALL_ARRAY_SIZE];
  float clonal_call_penalty;
  FitterDefaults();
  void FromJson(Json::Value &gopt_params);
  void FromCharacterLine(char *line);
  void DumpPoorlyStructuredText();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
        & clonal_call_scale
        & clonal_call_penalty;
  }
};

#endif //FITTERDEFAULTS_H
