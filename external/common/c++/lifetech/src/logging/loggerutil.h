/*
 * Copyright (c) 2010 Life Technologies Corporation. All rights reserved.
 */
#ifndef _LOGGERUTIL_H
#define _LOGGERUTIL_H

#include <log4cxx/logger.h>
#include "log4cxx/propertyconfigurator.h"
#include "log4cxx/helpers/exception.h"
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>
using namespace std;
using namespace log4cxx;

#define ERROR 		1
#define WARN	 	2
#define INFO		3
#define DEBUG		4

extern LoggerPtr pLogger;

// Fwd declaration so that compiler can resolve the call in 1st overloaded function
void dumpmessage(int severity, const char *sourcefile, const char *string1, const char *string2);

void dumpmessage(int severity, const char *srcfile, const std::string& msg) 
{
	dumpmessage(severity, srcfile, msg.c_str(), "");
}


void dumpmessage(int severity, const char *sourcefile, const char *string1, const char *string2)
{
        /*      This function has 4 inputs:
         1. severity: 1=ERR 2=warning 3=status (all others will display as 'unknown')
         2. sourcefile: should have the name of the LogFile calling this function
         3. string1: text message to dump
         4. string2: additional text message to dump. If none, DO NOT make this
         pointer null (crash) ! Just send a zero length string, i.e. string2[0]=0.
         */
        std::string msg(sourcefile);
        msg.append(string1).append(string2);

        switch (severity)
        {
                case ERROR:
                        LOG4CXX_ERROR(pLogger, msg.c_str());
                        break;
                case WARN:
                        LOG4CXX_WARN(pLogger, msg.c_str());
                        break;
                case INFO:
                        LOG4CXX_INFO(pLogger, msg.c_str());
                        break;
                default: // By default, log the message at DEBUG level.
                        LOG4CXX_DEBUG(pLogger, msg.c_str());
                        break;
        }
}

void writeLog4cxxConfFile (string log4cxx_ConfFileName, string LogFileName)
{
	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	FILE	*log4cxx_ConfFile = fopen(log4cxx_ConfFileName.c_str(), "wt");
	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

	fprintf(log4cxx_ConfFile, "log4j.debug=true\n");
	fprintf(log4cxx_ConfFile, "\n");
	fprintf(log4cxx_ConfFile, "# Set root logger level to DEBUG and its appender to A1, stdout\n");
	fprintf(log4cxx_ConfFile, "log4j.rootLogger=INFO, A1, stdout\n");
	fprintf(log4cxx_ConfFile, "\n");
	fprintf(log4cxx_ConfFile, "# A1 is set to be a file\n");
	fprintf(log4cxx_ConfFile, "log4j.appender.A1=FileAppender\n");
	fprintf(log4cxx_ConfFile, "\n");
	fprintf(log4cxx_ConfFile, "# ACHTUNG! Don't use quotes for the following parameter\n");
	fprintf(log4cxx_ConfFile, "log4j.appender.A1.File=%s\n", LogFileName.c_str());
	fprintf(log4cxx_ConfFile, "\n");
	fprintf(log4cxx_ConfFile, "# A1 uses PatternLayout.\n");
	fprintf(log4cxx_ConfFile, "log4j.appender.A1.layout=org.apache.log4j.PatternLayout\n");

	fprintf(
		log4cxx_ConfFile,
		"log4j.appender.A1.layout.ConversionPattern=%s\n",
		string("%d [%t] %-5p %c - %m%n").c_str());
	fprintf(log4cxx_ConfFile, "\n");
	fprintf(log4cxx_ConfFile, "# stdout is set to be a ConsoleAppender\n");
	fprintf(log4cxx_ConfFile, "log4j.appender.stdout=org.apache.log4j.ConsoleAppender \n");
	fprintf(log4cxx_ConfFile, "log4j.appender.stdout.Target=System.out\n");
	fprintf(log4cxx_ConfFile, "\n");
	fprintf(log4cxx_ConfFile, "# stdout uses PatternLayout.\n");
	fprintf(log4cxx_ConfFile, "log4j.appender.stdout.layout=org.apache.log4j.PatternLayout\n");
	fprintf(
		log4cxx_ConfFile,
		"log4j.appender.stdout.layout.ConversionPattern=%s\n",
		string("%d [%t] %-5p %c - %m%n").c_str());
	fprintf(log4cxx_ConfFile, "\n");
	fclose(log4cxx_ConfFile);
}

void ConfigureLog4cxx (string LogFileName)
{
	if (!getenv("LOG4CXX_CONFIGURATION"))
	{
		string	log4cxx_ConfFileName = LogFileName;
		log4cxx_ConfFileName.append(".conf");

		cout <<
			"ERROR: Environment variable LOG4CXX_CONFIGURATION not set. Using default conf file at: " <<
			log4cxx_ConfFileName <<
			'\n';

		writeLog4cxxConfFile(log4cxx_ConfFileName, LogFileName);
		PropertyConfigurator::configure(log4cxx_ConfFileName.c_str());
	}
	else
	{
		string	l4cxx_conf (getenv("LOG4CXX_CONFIGURATION"));
		dumpmessage
			(
				3, "Environment variable LOG4CXX_CONFIGURATION is set. Using log4cxx conf file from: ", 
					l4cxx_conf.c_str(), ""
			);
	}
}






#endif
