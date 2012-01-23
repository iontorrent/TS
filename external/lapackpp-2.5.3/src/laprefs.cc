#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "lafnames.h"
#include LA_PREFS_H

LaPreferences::pFormat LaPreferences::print_format = LaPreferences::NORMAL;
bool LaPreferences::print_newlines = true;

void LaPreferences::setPrintFormat(pFormat p, bool newlines)
{
  print_format = p;
  print_newlines = newlines;
}

LaPreferences::pFormat LaPreferences::getPrintFormat()
{
  return print_format;
}

bool LaPreferences::getPrintNewLines()
{
  return print_newlines;
}

