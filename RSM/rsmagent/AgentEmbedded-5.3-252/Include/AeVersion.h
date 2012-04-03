/* $Id: AeVersion.h,v 1.12.2.12 2010/11/05 19:30:15 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeVersion.h
 *
 *  Subsystem  :  Axeda Agent Embedded
 *
 *  Description:  Version macros
 *
 **************************************************************************/
#ifndef _AE_VERSION_H_
#define _AE_VERSION_H_

#define AE_VERSION_MAJOR 5
#define AE_VERSION_MINOR 3
#define AE_VERSION_BUILD 252

#define AE_VERSION_STRING(major , minor, build) AE_VERSION_STRING_FUNC(major , minor, build)
#define AE_VERSION_STRING_FUNC(major , minor, build) #major "." #minor "-" #build

#endif
