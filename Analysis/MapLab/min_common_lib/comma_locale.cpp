/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#include "comma_locale.h"

DecimalWithCommas     num_punct_3rd_pos;
HexadecimalWithCommas num_punct_4th_pos;
std::locale system_locale;
std::locale deccomma_locale (system_locale, &num_punct_3rd_pos);
std::locale hexcomma_locale (system_locale, &num_punct_4th_pos);
