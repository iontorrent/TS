/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "comma_locale.h"

dec_comma numpunct3;
hex_comma numpunct4;
std::locale system_locale;
std::locale deccomma_locale (system_locale, &numpunct3);
std::locale hexcomma_locale (system_locale, &numpunct4);
