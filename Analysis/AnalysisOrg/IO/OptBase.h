/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef OPTBASE_H
#define OPTBASE_H

#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "OptArgs.h"
#include "json/json.h"

using namespace std;

string GetRidOfDomainAndHyphens(const string& name);
string GetParamsString(Json::Value& json, const string& key, const string& default_value);
int GetParamsInt(Json::Value& json, const string& key, const int default_value);
float GetParamsFloat(Json::Value& json, const string& key, const float default_value);
double GetParamsDouble(Json::Value& json, const string& key, const double default_value);
bool RetrieveParameterBool(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const bool default_value);
string RetrieveParameterString(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value);
int RetrieveParameterInt(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const int default_value);
float RetrieveParameterFloat(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const float default_value);
double RetrieveParameterDouble(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const double default_value);
int RetrieveParameterVectorInt(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value, vector<int>& ret_vector);
int RetrieveParameterVectorFloat(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value, vector<float>& ret_vector);
int RetrieveParameterVectorDouble(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value, vector<double>& ret_vector);

template <class T>
bool CheckParameterLowerUpperBound(const string& identifier, T& parameter, T lower_limit, T upper_limit)
{
	bool is_ok = false;
	//cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (integer, " << source << ")" << endl;
	cout << "Limit check parameter " << identifier << ": lim. "
	     << lower_limit << " <= " << parameter << " <= lim. " << upper_limit << "? ";
	if(parameter < lower_limit)
	{
		cout << "Using " << identifier << "=" << lower_limit << " instead!";
		parameter = lower_limit;
	}
	else if(parameter > upper_limit)
	{
		cout << "Using " << identifier << "=" << upper_limit << " instead!";
		parameter = upper_limit;
	}
	else 
	{
		cout << "OK!";
		is_ok = true;
	}
	cout << endl;
	return (is_ok);
}

template <class T>
bool CheckParameterLowerBound(const string& identifier ,T &parameter, T lower_limit) 
{
	bool is_ok = false;
	cout << "Limit check parameter " << identifier << ": lim. "
	     << lower_limit << " <= " << parameter << "? ";
	if(parameter < lower_limit)
	{
		cout << "Using " << identifier << "=" << lower_limit << " instead!";
		parameter = lower_limit;
	}
	else
	{
		cout << "OK!";
		is_ok = true;
	}
	cout << endl;
	return (is_ok);
}

template <class T>
bool CheckParameterUpperBound(const string& identifier ,T &parameter, T upper_limit)
{
	bool is_ok = false;
	cout << "Limit check parameter " << identifier << ": "
	     << parameter << " <= lim. " << upper_limit << "? ";
	if(parameter > upper_limit)
	{
		cout << "Using " << identifier << "=" << upper_limit << " instead!";
		parameter = upper_limit;
	}
	else 
	{
		cout << "OK!";
		is_ok = true;
	}
	cout << endl;
	return (is_ok);
}
#endif // OPTBASE_H
