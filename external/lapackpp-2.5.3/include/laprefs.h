// -*-C++-*- 

// lapackpp preferences by Jacob (Jack) Gryn 

/** @file
 * @brief Preferences class for Lapack++
 */

#ifndef _LA_PREFS_H_
#define _LA_PREFS_H_

/** @brief Global preferences class in Lapack++
 *
 * Class to store global (i.e. static i.e. library-wide) preferences.
 *
 * Currently this only concerns the output display format of
 * matrices. In the future, more preferences may or may not be
 * added. In addition to this class, there is one more library-wide
 * setting in LaException::enablePrint() but that concerns only the
 * behaviour of exceptions and is therefore stored in that class. 
 */
class LaPreferences
{
   public:
      /** Flags for choosing the output display format when printing
       * the matrix elements to an ostream */
      typedef enum pFormat 
      {
	 /** C++ display format (default) */
	 NORMAL, 
	 /** Matlab format (DocumentMe: what exactly does this mean?) */
	 MATLAB, 
	 /** Maple format (DocumentMe: what exactly does this mean?) */
	 MAPLE
      };

      /** Set the output display format. The default is @c
       * LaPreferences::NORMAL.
       *
       * The following is how one would modify the output display
       * format to be compatible with their favourite math program:
       *
       * Place
\verbatim
#include LA_PREFS_H 
\endverbatim
       *
       * in the include statements, somewhere after "lafnames.h". At
       * the beginning of your code, call e.g.
\verbatim
  LaPreferences::setPrintFormat(LaPreferences::MATLAB, true);
\endverbatim
       *
       * where the first argument is your preferred output format, and
       * the second argument toggles multiline matrix output (@c true
       * = place a newline after each matrix row, @c false = use only
       * the appropriate MATLAB/MAPLE delimiter). The second argument
       * is ignored if the output format is @c LaPreferences::NORMAL.
       *
       * @param p The preferred output format
       *
       * @param newlines Toggles multiline matrix output (@c true =
       * place a newline after each matrix row, @c false = use only
       * the appropriate MATLAB/MAPLE delimiter). This argument is
       * ignored if the output format is @c LaPreferences::NORMAL.
       */
      static void setPrintFormat(pFormat p, bool newlines=true);

      /** Get the current output display format as set by
       * setPrintFormat(). */
      static pFormat getPrintFormat();

      /** Get the current output display newline format as set by
       * setPrintFormat(). */
      static bool getPrintNewLines();

   private:
      static pFormat print_format;
      static bool print_newlines;
};

#endif  // _LA_PREFS_H_

