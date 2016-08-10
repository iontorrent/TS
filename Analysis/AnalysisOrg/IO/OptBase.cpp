/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iomanip>
#include <errno.h>
#include "OptBase.h"
#include "Utils.h"

string GetRidOfDomainAndHyphens(const string& name)
{
	string s(name);
	int index = name.rfind("::");
	if(index > -1)
	{
		s = name.substr(index + 2, name.length() - index - 2);
	}

	for (unsigned int i = 0; i < s.size(); ++i)
	{
		if (s[i] == '-')
		s[i] = '_';
	}

	return s;
}

string GetParamsString(Json::Value& json, const string& key, const string& default_value) {
  string value = default_value;
  if (json.isMember(key))
  {
	  value = json[key].asCString();
  }

  return value;
}

int GetParamsInt(Json::Value& json, const string& key, const int default_value) {
  if (not json.isMember(key))
    return default_value;
  if (json[key].isString())
    return atoi(json[key].asCString());
  return json[key].asInt();
}

float GetParamsFloat(Json::Value& json, const string& key, const float default_value) {
  if (not json.isMember(key))
    return default_value;
  if (json[key].isString())
    return atof(json[key].asCString());
  return json[key].asFloat();
}

double GetParamsDouble(Json::Value& json, const string& key, const double default_value) {
  if (not json.isMember(key))
    return default_value;
  if (json[key].isString())
    return atof(json[key].asCString());
  return json[key].asDouble();
}

bool RetrieveParameterBool(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const bool default_value)
{
  string long_name_underscores = GetRidOfDomainAndHyphens(long_name_hyphens);
  bool value = default_value;
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
    if (json[long_name_underscores].isString())
      value = atoi(json[long_name_underscores].asCString());
    else
      value = json[long_name_underscores].asInt();
    source = "parameters json file";
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
    value = opts.GetFirstBoolean(short_name, long_name_hyphens, value);
    source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << (value ? "true" : "false") << " (boolean, " << source << ")" << endl;
  return value;
}

string RetrieveParameterString(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value)
{
  string long_name_underscores = GetRidOfDomainAndHyphens(long_name_hyphens);
  string value = default_value;
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
    value = json[long_name_underscores].asCString();
    source = "parameters json file";
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
    value = opts.GetFirstString(short_name, long_name_hyphens, value);
    source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (string, " << source << ")" << endl;
  return value;
}

int RetrieveParameterInt(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const int default_value)
{
  string long_name_underscores = GetRidOfDomainAndHyphens(long_name_hyphens);
  int value = default_value;
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
    if (json[long_name_underscores].isString())
      value = atoi(json[long_name_underscores].asCString());
    else
      value = json[long_name_underscores].asInt();
    source = "parameters json file";
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
    value = opts.GetFirstInt(short_name, long_name_hyphens, value);
    source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (integer, " << source << ")" << endl;
  return value;
}

float RetrieveParameterFloat(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const float default_value)
{
  string long_name_underscores = GetRidOfDomainAndHyphens(long_name_hyphens);
  float value = default_value;
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
    if (json[long_name_underscores].isString())
      value = atof(json[long_name_underscores].asCString());
    else
      value = json[long_name_underscores].asFloat();
    source = "parameters json file";
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
    value = (float)opts.GetFirstDouble(short_name, long_name_hyphens, value);
    source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (float,  " << source << ")" << endl;
  return value;
}

double RetrieveParameterDouble(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const double default_value)
{
  string long_name_underscores = GetRidOfDomainAndHyphens(long_name_hyphens);
  double value = default_value;
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
    if (json[long_name_underscores].isString())
      value = atof(json[long_name_underscores].asCString());
    else
      value = json[long_name_underscores].asDouble();
    source = "parameters json file";
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
    value = opts.GetFirstDouble(short_name, long_name_hyphens, value);
    source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (double,  " << source << ")" << endl;
  return value;
}

int RetrieveParameterVectorInt(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value, vector<int>& ret_vector)
{
  string long_name_underscores = GetRidOfDomainAndHyphens(long_name_hyphens);
  string value = default_value;

  if(value.length() > 0)
  {
	  vector<string> words;
	  split(value,',',words);
	  ret_vector.clear();
	  for (size_t i = 0; i < words.size(); i++) {
		char *end;
		int err = errno;
		errno = 0;
		ret_vector.push_back(strtol(words[i].c_str(), &end, 10));
		if (errno != 0 || *end != '\0') {
		  cout << "Error converting: " + words[i] + " to an int for option: " + long_name_hyphens << endl;
		  return errno;
		}
		errno = err;    
	  }
  }
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
	  ret_vector.clear();
	  size_t sz = json[long_name_underscores].size();
	  char buf[1000];
      if(sz > 0)
	  {
          if(sz == 1)
          {
              if(json[long_name_underscores][0].isString())
              {
                  ret_vector.push_back(atoi(json[long_name_underscores][0].asCString()));
                  value = json[long_name_underscores][0].asCString();
              }
              else
              {
                  ret_vector.push_back(json[long_name_underscores][0].asInt());
                  sprintf(buf, "%d", ret_vector[0]);
                  value = buf;
              }
          }
          else
          {
              value = "";
              for(int i = 0; i < (int)sz - 1; i++)
              {
                  if(json[long_name_underscores][i].isString())
                  {
                      ret_vector.push_back(atoi(json[long_name_underscores][i].asCString()));
                      value += json[long_name_underscores][i].asCString();
                      value += ",";
                  }
                  else
                  {
                      ret_vector.push_back(json[long_name_underscores][i].asInt());
                      sprintf(buf, "%d,", ret_vector[i]);
                      string ss = buf;
                      value += ss;
                  }
              }

              if(json[long_name_underscores][(int)sz - 1].isString())
              {
                  ret_vector.push_back(atoi(json[long_name_underscores][(int)sz - 1].asCString()));
                  value += json[long_name_underscores][(int)sz - 1].asCString();
              }
              else
              {
                  ret_vector.push_back(json[long_name_underscores][(int)sz - 1].asInt());
                  sprintf(buf, "%d", ret_vector[(int)sz - 1]);
                  string ss = buf;
                  value += ss;
              }
          }
          source = "parameters json file";
      }
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
	  ret_vector.clear();
	  opts.GetOption(ret_vector, default_value, short_name, long_name_hyphens);

	  char buf[1000];
	  if(ret_vector.empty())
	  {
	      cout << "Error setting: there is no value set for option: " + long_name_hyphens << endl;
		  return 1;
	  }
	  else if(ret_vector.size() == 1)
	  {
		  sprintf(buf, "%d", ret_vector[0]);
		  value = buf;
	  }
	  else
	  {
		  value = "";
		  for(size_t i = 0; i < ret_vector.size() - 1; i++) {
			  sprintf(buf, "%d,", ret_vector[i]);
			  string ss = buf;
			  value += ss;
		  }
		  sprintf(buf, "%d", ret_vector[ret_vector.size() - 1]);
		  string ss = buf;
		  value += ss;
	  }
	  source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (int,  " << source << ")" << endl;
  return 0;
}

int RetrieveParameterVectorFloat(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value, vector<float>& ret_vector)
{
  string long_name_underscores = GetRidOfDomainAndHyphens(long_name_hyphens);
  string value = default_value;

  if(value.length() > 0)
  {
	  vector<string> words;
	  split(value,',',words);
	  ret_vector.clear();
	  for (size_t i = 0; i < words.size(); i++) {
		char *end;
		int err = errno;
		errno = 0;
		ret_vector.push_back(strtod(words[i].c_str(), &end));
		if (errno != 0 || *end != '\0') {
		  cout << "Error converting: " + words[i] + " to an float for option: " + long_name_hyphens << endl;
		  return errno;
		}
		errno = err;    
	  }
  }
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
	  ret_vector.clear();
	  size_t sz = json[long_name_underscores].size();
	  char buf[1000];
      if(sz > 0)
	  {
          if(sz == 1)
          {
              if(json[long_name_underscores][0].isString())
              {
                  ret_vector.push_back(atof(json[long_name_underscores][0].asCString()));
                  value = json[long_name_underscores][0].asCString();
              }
              else
              {
                  ret_vector.push_back(json[long_name_underscores][0].asFloat());
                  sprintf(buf, "%f", ret_vector[0]);
                  value = buf;
              }
          }
          else
          {
              value = "";
              for(int i = 0; i < (int)sz - 1; i++)
              {
                  if(json[long_name_underscores][i].isString())
                  {
                      ret_vector.push_back(atof(json[long_name_underscores][i].asCString()));
                      value += json[long_name_underscores][i].asCString();
                      value += ",";
                  }
                  else
                  {
                      ret_vector.push_back(json[long_name_underscores][i].asFloat());
                      sprintf(buf, "%f,", ret_vector[i]);
                      string ss = buf;
                      value += ss;
                  }
              }

              if(json[long_name_underscores][(int)sz - 1].isString())
              {
                  ret_vector.push_back(atof(json[long_name_underscores][(int)sz - 1].asCString()));
                  value += json[long_name_underscores][(int)sz - 1].asCString();
              }
              else
              {
                  ret_vector.push_back(json[long_name_underscores][(int)sz - 1].asFloat());
                  sprintf(buf, "%f", ret_vector[(int)sz - 1]);
                  string ss = buf;
                  value += ss;
              }
          }
          source = "parameters json file";
      }
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
	  ret_vector.clear();
	  vector<double> ret_vector2;
	  opts.GetOption(ret_vector2, default_value, short_name, long_name_hyphens);
	  for(size_t i = 0; i < ret_vector2.size(); i++)
	  {
		  ret_vector.push_back((float)ret_vector2[i]);
	  }

	  char buf[1000];
	  if(ret_vector.empty())
	  {
	      cout << "Error setting: there is no value set for option: " + long_name_hyphens << endl;
		  return 1;
	  }
	  else if(ret_vector.size() == 1)
	  {
		  sprintf(buf, "%f", ret_vector[0]);
		  value = buf;
	  }
	  else
	  {
		  value = "";
		  for(size_t i = 0; i < ret_vector.size() - 1; i++) {
			  sprintf(buf, "%f,", ret_vector[i]);
			  string ss = buf;
			  value += ss;
		  }
		  sprintf(buf, "%f", ret_vector[ret_vector.size() - 1]);
		  string ss = buf;
		  value += ss;
	  }
	  source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (float,  " << source << ")" << endl;
  return 0;
}

int RetrieveParameterVectorDouble(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value, vector<double>& ret_vector)
{
  string long_name_underscores = GetRidOfDomainAndHyphens(long_name_hyphens);
  string value = default_value;

  if(value.length() > 0)
  {
	  vector<string> words;
	  split(value,',',words);
	  ret_vector.clear();
	  for (size_t i = 0; i < words.size(); i++) {
		char *end;
		int err = errno;
		errno = 0;
		ret_vector.push_back(strtod(words[i].c_str(), &end));
		if (errno != 0 || *end != '\0') {
		  cout << "Error converting: " + words[i] + " to an double for option: " + long_name_hyphens << endl;
		  return errno;
		}
		errno = err;    
	  }
  }
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
	  ret_vector.clear();
	  size_t sz = json[long_name_underscores].size();
	  char buf[1000];
      if(sz > 0)
	  {
          if(sz == 1)
          {
              if(json[long_name_underscores][0].isString())
              {
                  ret_vector.push_back(atof(json[long_name_underscores][0].asCString()));
                  value = json[long_name_underscores][0].asCString();
              }
              else
              {
                  ret_vector.push_back(json[long_name_underscores][0].asDouble());
                  sprintf(buf, "%f", ret_vector[0]);
                  value = buf;
              }
          }
          else
          {
              value = "";
              for(int i = 0; i < (int)sz - 1; i++)
              {
                  if(json[long_name_underscores][i].isString())
                  {
                      ret_vector.push_back(atof(json[long_name_underscores][i].asCString()));
                      value += json[long_name_underscores][i].asCString();
                      value += ",";
                  }
                  else
                  {
                      ret_vector.push_back(json[long_name_underscores][i].asDouble());
                      sprintf(buf, "%f,", ret_vector[i]);
                      string ss = buf;
                      value += ss;
                  }
              }

              if(json[long_name_underscores][(int)sz - 1].isString())
              {
                  ret_vector.push_back(atof(json[long_name_underscores][(int)sz - 1].asCString()));
                  value += json[long_name_underscores][(int)sz - 1].asCString();
              }
              else
              {
                  ret_vector.push_back(json[long_name_underscores][(int)sz - 1].asDouble());
                  sprintf(buf, "%f", ret_vector[(int)sz - 1]);
                  string ss = buf;
                  value += ss;
              }
          }
          source = "parameters json file";
      }
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
	  ret_vector.clear();
	  opts.GetOption(ret_vector, default_value, short_name, long_name_hyphens);

	  char buf[1000];
	  if(ret_vector.empty())
	  {
	      cout << "Error setting: there is no value set for option: " + long_name_hyphens << endl;
		  return 1;
	  }
	  else if(ret_vector.size() == 1)
	  {
		  sprintf(buf, "%f", ret_vector[0]);
		  value = buf;
	  }
	  else
	  {
		  value = "";
		  for(size_t i = 0; i < ret_vector.size() - 1; i++) {
			  sprintf(buf, "%f,", ret_vector[i]);
			  string ss = buf;
			  value += ss;
		  }
		  sprintf(buf, "%f", ret_vector[ret_vector.size() - 1]);
		  string ss = buf;
		  value += ss;
	  }
	  source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (double,  " << source << ")" << endl;
  return 0;
}

