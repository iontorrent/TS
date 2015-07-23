/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __common_str_h__
#define __common_str_h__

#ifndef __common_str_cpp__

extern const char* TRUE_STR;
extern const char* FALSE_STR;
extern const char* YES_STR;
extern const char* NO_STR;
extern const char* Y_STR;
extern const char* N_STR;
extern const char* T_STR;
extern const char* F_STR;
extern const char* EMPTY_STR;
extern const char* SPACE_STR;
extern const char* NEWLINE_STR;
extern const char* SP2_STR;
extern const char* SP4_STR;
extern const char* SP6_STR;
extern const char* SP8_STR;
extern const char* SP10_STR;
extern const char* SP12_STR;
extern const char* SP14_STR;
extern const char* SP16_STR;
extern const char* ZERO_STR;
extern const char* ONE_STR;
extern const char* MINUS_ONE_STR;
extern const char* TWO_STR;
extern const char* MINUS_TWO_STR;
extern const char* THREE_STR;
extern const char* MINUS_THREE_STR;
extern const char* FOUR_STR;
extern const char* MINUS_FOUR_STR;
extern const char* FIVE_STR;
extern const char* MINUS_FIVE_STR;
extern const char* SIX_STR;
extern const char* MINUS_SIX_STR;
extern const char* SEVEN_STR;
extern const char* MINUS_SEVEN_STR;
extern const char* EIGHT_STR;
extern const char* MINUS_EIGHT_STR;
extern const char* NINE_STR;
extern const char* MINUS_NINE_STR;
extern const char* TEN_STR;
extern const char* MINUS_TEN_STR;
extern const char* NAME_STR;
extern const char* NUMBER_STR;
extern const char* FILE_STR;
extern const char* FILENAME_STR;
extern const char* FILENAMES_STR;
extern const char* FILEMASKS_STR;
extern const char* INTEGER_STR;
extern const char* BOOLEAN_STR;
extern const char* STRING_STR;
extern const char* TEXT_STR;
extern const char* FLOAT_STR;
extern const char* DOUBLE_STR;
extern const char* OBJNAME_STR;
extern const char* UNKNOWN_STR;
extern const char* BELOW_EQ_STR;
extern const char* ABOVE_EQ_STR;
extern const char* TWO_EXP_STR;
extern const char* TEN_EXP_STR;
extern const char* COMMA_STR;
extern const char* COMMASPACE_STR;
extern const char* TAB_STR;
extern const char* BYTES_STR;
extern const char* ITEMS_STR;
extern const char* RESULTS_STR;
extern const char* CREATED_STR;
extern const char* PRODUCED_STR;


#endif // __common_str_cpp__

#include <stddef.h>

const char* inverse_bs (const char* bs); // returns string expressing boolean value opposite to passed in
bool bool_eval (const char* bs, bool* value = NULL); // returns True if passed string contains valid stringified boolean value and sets value; otherwice returns false
const char* bool_str (bool val); // returns name for a boolean value

#endif
