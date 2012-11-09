/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "HandleExpLog.h"
#include "Utils.h"
#include <assert.h>


#ifndef ALIGNSTATS_IGNORE

//@TODO:  Please replace this parser by json library and the file by a json file
// this parser has broken once already because coding up unique file formats is a bad, bad idea
char *GetExpLogParameter (const char *filename, const char *paramName)
{
  FILE *fp = NULL;
  fopen_s (&fp, filename, "rb");
  if (!fp)
  {
    strerror (errno);
    return (NULL);
  }
  char *parameter = NULL;
  size_t size = 256; //getline resets line size as needed
  char *line = (char *) malloc (size);
  size_t testlen = strlen (paramName);
  bool val_found = false;
  while ( ((getline (&line,&size,fp)) > 0) & !val_found)  // stop at first occurrence
  {
    //fprintf (stderr, "Size is %d\n",(int) size);
    //fprintf(stderr, "%s\n", line);
    //fprintf(stderr, "%s\n", paramName);
    //if (strstr (line, paramName))
    if (!strncmp (line,paramName,testlen)) // find >start of line< containing keyword
    {
      //fprintf(stderr,"Match: %s %s\n", paramName, line);
      if (strlen (line) > (testlen)) // make sure line contains useful information past the hypothetical colon separator
      {
        //fprintf(stderr,"line long enough %d \n", (int)testlen);
        if (line[testlen]==':') // make sure the keyword is separated from the data by a ':' and we are not in a substring hit by accident
        {
          //fprintf(stderr,"Found %s at %s\n", paramName, line);
          char *sPtr = strchr (line, ':');  // guaranteed by above
          sPtr++; //skip colon
          //skip leading white space
          while (isspace (*sPtr)) sPtr++;
	  if (parameter) free(parameter);
          parameter = (char *) malloc (sizeof (char) * (size + 2));
          strcpy (parameter, sPtr);
          val_found = true;
	  //if (sPtr) free(sPtr);
        }
      }
    }
  }
  if (line)
    free (line);
  fclose (fp);
  //DEBUG
  //fprintf (stdout, "getExpLogParameter: %s %s\n", paramName, parameter);
  return (parameter);
}

//@TODO:  what does this do when compared to the above?
// why do we not simply call the above and push_back the resulting value?
void GetExpLogParameters (const char *filename, const char *paramName, std::vector<std::string> &values)
{
  FILE *fp = NULL;
  values.resize (0);
  fopen_s (&fp, filename, "rb");
  if (!fp)
  {
    strerror (errno);
    return;
  }
  size_t size = 256; //getline resets line size as needed
  char *line = (char *) malloc (size);
  size_t testlen = strlen (paramName);
  bool val_found = false;
  while ( ((getline (&line,&size,fp)) > 0) & !val_found)
  {
    //fprintf (stderr, "Size is %d\n", size);
//    if (strstr (line, paramName))
    if (!strncmp (line,paramName,testlen)) // find >start of line< containing keyword
    {
      if (strlen (line) > testlen) // make sure line contains useful information past the hypothetical colon separator
      {
        if (line[testlen]==':') // make sure the keyword is separated from the data by a ':' and we are not in a substring hit by accident
        {
          char *sPtr = strchr (line, ':');
          sPtr++; //skip colon
          //skip leading white space
          while (isspace (*sPtr)) sPtr++;
          values.push_back (sPtr);
          val_found = true;
        }
      }
    }
  }
  if (line)
  {
    free (line);
  }
  fclose (fp);
}
#endif
#ifndef ALIGNSTATS_IGNORE
//
// Parses explog.txt and returns a flag indicating whether wash flows are
// present in the run data.
// Returns 0 - no wash flow
//   1 - wash flow present
//     -1 - could not determine
//
int HasWashFlow (char *filepath)
{
  int washImages = -1; // Set default to indicate error determine value.  
  char *argument = NULL;
  
  argument = GetExpLogParameter (filepath,"Image Map");
  if (argument)
  {
    // Typical arg is
    // "5 0 r4 1 r3 2 r2 3 r1 4 w2" - would be wash flow
    //  "4 0 r4 1 r3 2 r2 3 r1" - would be no wash
    int flowNum;

    int nret = sscanf (argument,"%d", &flowNum);
    if (nret!=1) flowNum = 0;  //different format, fall through

    if (flowNum == 5)
    {
      char *sPtr = strrchr (argument, 'w');
      if (sPtr)
        washImages = 1;
      else
        washImages = 0;
    }
    else if (flowNum == 4)
    {
      washImages = 0;
    }
    else
    {
      // Its not either the expected 5 or 4.
      //Check the last part of the string for a 'w' character
      char *sPtr = strrchr (argument, 'w');
      if (sPtr)
        washImages = 1;
      else
        washImages = 0;
    }
    free (argument);
  }

  return (washImages);
}

char *GetPGMFlowOrder (char *filepath)
{
  char *flowOrder = NULL;
  char *argument = NULL;
  
  argument = GetExpLogParameter (filepath,"Image Map");
  const char mapping[6] = {"0GCAT"};
  if (argument)
  {

    // Typical arg is
    // "5 0 r4 1 r3 2 r2 3 r1 4 w2" - would be wash flow
    //  "4 0 r4 1 r3 2 r2 3 r1" - would be no wash
    //  -OR-
    // "4 0 T 1 A 2 C 3 G"
    // -OR-
    // "tcagtcagtcag"
    //
    //fprintf (stderr, "Raw string = '%s'\n", argument);

    // First entry is number of flows in cycle, unless its not!
    int numFlows = 0;
    sscanf (argument,"%d", &numFlows);
    if (numFlows == 0)
    {
      numFlows = strlen (argument);
    }
    assert (numFlows > 0);

    // allocate memory for the floworder string
    flowOrder = (char *) malloc (numFlows+1);


    // Upper case everything
    ToUpper (argument);
    //fprintf (stdout, "argument = '%s'\n", argument);

    //  Read string char at a time.
    //  If its 'R' then
    //   read next char as an integer and convert integer to Nuke
    //  else if its T A C or G
    //   set Nuke
    //  else skip
    int num = 0; //increments index into argument string
    int cnt = 0; //increments index into flowOrder string
    for (num = 0; argument[num] != '\n'; num++)
    {
      //fprintf (stdout, "We Got '%c'\n", argument[num]);
      if (argument[num] == 'R')
      {
        //this will work as long as there are less than 10 reagent bottles on the PGM
        int index = argument[num+1] - 48; //does this make anyone nervous?
        //fprintf (stdout, "Index = %d\n", index);
        assert (index > 0 && index < 9);
        //fprintf (stdout, "mapping[%d] = %c\n", index, mapping[index]);
        flowOrder[cnt++] = mapping[index];
        flowOrder[cnt] = '\0';
      }
      else if (argument[num] == 'T' ||
               argument[num] == 'A' ||
               argument[num] == 'C' ||
               argument[num] == 'G')
      {
        flowOrder[cnt++] = argument[num];
        flowOrder[cnt] = '\0';
      }
      else
      {
        //skip this character
      }
    }

    free (argument);
  }
  return (flowOrder);
}
#endif

#ifndef ALIGNSTATS_IGNORE
//
//find number of cycles in dataset
//
int GetCycles (const char *filepath)
{
  int cycles = 0;

  // Method using the explog.txt
  char *argument = NULL;
  long value;
  
  argument = GetExpLogParameter (filepath,"Cycles");
  if (argument)
  {
    if (validIn (argument, &value))
    {
      fprintf (stderr, "Error getting num cycles from explog.txt\n");
      exit (1);
    }
    else
    {
      cycles = (int) value;
    }
    free (argument);
  }
  else
  {
    //DEBUG
    fprintf (stderr, "No Cycles keyword found\n");
  }
  return (cycles);
}

//
//Determine number of Flows in run from explog.txt
//
int GetTotalFlows (const char *filepath)
{
  int numFlows = 0;

  // Method using the explog.txt
  char *argument = NULL;
  long value;
  
  argument = GetExpLogParameter (filepath,"Flows");
  if (argument)
  {
    if (validIn (argument, &value))
    {
      fprintf (stderr, "Error getting num flows from explog.txt\n");
      fprintf (stderr, "Supplied: %s\n", argument);
      exit (1);
    }
    else
    {
      //DEBUG
      //fprintf (stderr, "GetTotalFlows: '%s' '%d'\n", argument, (int) value);
      numFlows = (int) value;
    }
    free (argument);
  }
  else
  {
    // No Flows keyword found - legacy file format pre Jan 2011
    //DEBUG
    fprintf (stderr, "No Flows keyword found\n");
    int cycles = GetCycles (filepath);
    numFlows = 4 * cycles;
  }
  //fprintf (stderr, "Returning numFlows = %d\n",numFlows);
  return (numFlows);
}

//
//return chip id string
//
char * GetChipId (const char *filepath)
{
  // Method using the explog.txt
  char *argument = NULL;
  
  argument = GetExpLogParameter (filepath,"ChipType");
  if (argument)
  {
    char *chip = (char *) malloc (10);
    int len = strlen (argument);
    int y = 0;
    for (int i = 0; i<len;i++)
    {
      if (isdigit (argument[i]))
        chip[y++] = argument[i];
    }
    chip[y] = '\0';
    free (argument);
    return (chip);
  } else {
    fprintf(stderr,"Failed to find ChipId in explog.txt\n");
  }
  return (NULL);
}

//@TODO: replace this by config file in json format because parsing is useless
void GetChipDim (const char *type, int dims[2], const char *filepath)
{
  if (type != NULL)
  {
    if (strncmp ("314",type,3) == 0)
    {
      dims[0] = 1280;
      dims[1] = 1152;
    }
    else if (strncmp ("324",type,3) == 0)
    {
      dims[0] = 1280;
      dims[1] = 1152;
    }
    else if (strncmp ("316",type,3) == 0)
    {
      dims[0] = 2736;
      dims[1] = 2640;
    }
    else if (strncmp ("318",type,3) == 0)
    {
      dims[0] = 3392;
      dims[1] = 3792;
    }
    else if (strncmp ("900",type,4) == 0)
    {

      // Method using the explog.txt
      char *argument = NULL;
      long value;

      argument = GetExpLogParameter (filepath,"Rows");
      if (validIn (argument, &value))
      {
        fprintf (stderr, "Error getting rows from explog.txt\n");
        dims[1] = 0;
      }
      else
      {
        //fprintf (stderr, "Rows: '%s' '%d'\n", argument, (int) value);
        dims[1] = (int) value;
      }
      if (argument) free(argument);
      argument = GetExpLogParameter (filepath,"Columns");
      if (validIn (argument, &value))
      {
        fprintf (stderr, "Error getting columns from explog.txt\n");
        dims[0] = 0;
      }
      else
      {
        //fprintf (stderr, "Columns: '%s' '%d'\n", argument, (int) value);
        dims[0] = (int) value;
      }
      if (argument) free(argument);
    }
    else
    {
      dims[0] = 0;
      dims[1] = 0;
    }
  }
  else
  {
    dims[0] = 0;
    dims[1] = 0;
  }
}
#endif
