/* $Id: AeDRMSOAP.h,v 1.4 2008/05/21 18:24:36 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeDRMSOAP.h
 *  
 *  Subsystem  :  Axeda Agent Embedded
 *  
 *  Description:  DRM SOAP declarations
 *
 **************************************************************************/
#ifndef _AE_DRM_SOAP_H_
#define _AE_DRM_SOAP_H_

/* SOAP message object. */
typedef struct _AeDRMSOAP AeDRMSOAP;

#ifdef __cplusplus
extern "C" {
#endif

/* Creates SOAP message object, which may be used for parsing. Returns new
   object pointer. */
AeDRMSOAP   *AeDRMSOAPNew(void);

/* Destroys SOAP message object. */
void        AeDRMSOAPDestroy(AeDRMSOAP *pSOAP);

/* Parses SOAP message of iLength size pointed by pSource. As result of parsing
   SOAP object (pSOAP) is modified. Returns true if parsing is successful, or
   false otherwise. */
AeBool      AeDRMSOAPParse(AeDRMSOAP *pSOAP, AeChar *pSource, AeInt iLength);

/* Returns handle for the first method in the SOAP message (pSOAP), or NULL if
   the message does not contain methods. */
AeHandle    AeDRMSOAPGetFirstMethod(AeDRMSOAP *pSOAP);

/* Returns handle for method following method pMethod, or NULL, if there are no
   more methods. */
AeHandle    AeDRMSOAPGetNextMethod(AeHandle pMethod);

/* Returns handle for method named pName in the SOAP message (pSOAP), or NULL
   if not found. */
AeHandle    AeDRMSOAPGetMethodByName(AeDRMSOAP *pSOAP, AeChar *pName);

/* Returns name for method pMethod. */
AeChar      *AeDRMSOAPGetMethodName(AeHandle pMethod);

/* Returns handle for the first parameter in method pMethod, or NULL if the
   method does not contain parameters. */
AeHandle    AeDRMSOAPGetFirstParameter(AeHandle pMethod);

/* Returns handle for parameter following parameter pParameter, or NULL if
   there are no more parameters. */
AeHandle    AeDRMSOAPGetNextParameter(AeHandle pParameter);

/* Returns handle for parameter named pName in the method (pMethod), or NULL
   if not found. */
AeHandle    AeDRMSOAPGetParameterByName(AeHandle pMethod, AeChar *pName);

/* Returns handle for the first child of parameter pParameter, or NULL if there
   are no children. */
AeHandle    AeDRMSOAPGetParameterFirstChild(AeHandle pParameter);

/* Returns name for parameter pParameter. */
AeChar      *AeDRMSOAPGetParameterName(AeHandle pParameter);

/* Returns value for parameter pParameter. Parameter value is a string
   consisting of the first character data portion of the parameter content,
   not including leading whitespace. */
AeChar      *AeDRMSOAPGetParameterValue(AeHandle pParameter);

/* Returns value for parameter named pName in method pMethod, or NULL if not
   found. */
AeChar      *AeDRMSOAPGetParameterValueByName(AeHandle pMethod, AeChar *pName);

/* Returns handle for the first attribute of node pnode, or NULL if there are
   no attributes. A node is either a method or a parameter. */
AeHandle    AeDRMSOAPGetFirstAttribute(AeHandle pNode);

/* Returns handle for attribute following attribute pAttribute, or NULL if
   there are no more attributes. */
AeHandle    AeDRMSOAPGetNextAttribute(AeHandle pAttribute);

/* Returns handle for attribute named pName in the node (pNode), or NULL if not
   found. */
AeHandle    AeDRMSOAPGetAttributeByName(AeHandle pNode, AeChar *pName);

/* Returns name for attribute pAttribute. */
AeChar      *AeDRMSOAPGetAttributeName(AeHandle pAttribute);

/* Returns value for attribute pAttribute. */
AeChar      *AeDRMSOAPGetAttributeValue(AeHandle pAttribute);

/* Returns value for attribute named pName in node pNode, or NULL if not
   found. */
AeChar      *AeDRMSOAPGetAttributeValueByName(AeHandle pNode, AeChar *pName);

#ifdef __cplusplus
}
#endif

#endif
