/* $Id: AeXML.h,v 1.1.2.2 2009/05/29 17:23:59 hfeng Exp $ */
/**************************************************************************
 *
 *  Copyright (c) 1999-2009 Axeda Corporation. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeXML.h
 *
 *  Subsystem  :  Axeda Agent Embedded
 *
 *  Description:  XML parsers
 *
 **************************************************************************/

#ifndef _AE_XML_H_
#define _AE_XML_H_

/* Escaped characters in xml format */
#define AE_XML_ENTITY_AMP   "&amp;"
#define AE_XML_ENTITY_LT    "&lt;"
#define AE_XML_ENTITY_GT    "&gt;"
#define AE_XML_ENTITY_APOS  "&apos;"
#define AE_XML_ENTITY_QUOT  "&quot;"

#define AE_XML_PROLOG       "<?xml ?>"

#define AE_XML_MAX_ATTR_SIZE 250

/* Forwared structure declarations */
typedef struct _AeXMLContent AeXMLContent;
typedef struct _AeXMLContentVTable AeXMLContentVTable;
typedef struct _AeXMLCharData AeXMLCharData;
typedef struct _AeXMLAttribute AeXMLAttribute;
typedef struct _AeXMLElement AeXMLElement;
typedef struct _AeXMLDocument AeXMLDocument;

/* XML content structure */
struct _AeXMLContent
{
    AeXMLContentVTable  *pVTable;
    AeBool              bElement;
    AeXMLElement        *pParent;
    AeXMLContent        *pNext;
};

/* XML content table structure */
struct _AeXMLContentVTable
{
    void    (*pDestroy)(AeXMLContent *pContent);
    void    (*pWrite)(AeXMLContent *pContent, AeArray *pArray);
    AeInt32 (*pGetFormattedSize)(AeXMLContent *pContent);
};

/* XML data structure */
struct _AeXMLCharData
{
    AeXMLContent    content;
    AeChar          *pData;
};

/* XML attribute structure */
struct _AeXMLAttribute
{
    AeChar          *pName;
    AeChar          *pValue;
    AeXMLAttribute  *pNext;
};

/* XML element structure */
struct _AeXMLElement
{
    AeXMLContent    content;
    AeChar          *pName;
    AeXMLAttribute  *pFirstAttribute;
    AeXMLAttribute  *pLastAttribute;
    AeXMLContent    *pFirstChild;
    AeXMLContent    *pLastChild;
    AeBool          bEmpty;
};

/* XML document structure */
struct _AeXMLDocument
{
    AeXMLElement    *pRoot;
};

/* wrapper call to AeXMLCharDataDestroy */
#define AeXMLContentDestroy(pContent) \
    (*(pContent)->pVTable->pDestroy)(pContent)

/* wrapper call to AeXMLCharDataWrite */
/* void AeXMLCharDataWrite(AeXMLContent *pContent, AeArray *pArray) writes pContent into pArray*/
#define AeXMLContentWrite(pContent, pArray) \
    (*(pContent)->pVTable->pWrite)(pContent, pArray)

/* wrapper call to AeXMLCharDataGetFormattedSize */
/* AeInt32 AeXMLCharDataGetFormattedSize(AeXMLContent *pContent) returns the size of pContents */
#define AeXMLContentGetFormattedSize(pContent) \
    (*(pContent)->pVTable->pGetFormattedSize)(pContent)

/* Creates a new AeXMLCharData structure */
AeXMLCharData   *AeXMLCharDataNew(void);

/* Destroys the AeXMLCharData structure */
void            AeXMLCharDataDestroy(AeXMLContent *pContent);

/* Set data to the AeXMLCharData structure */
/* p is AeXMLCharData pointer, x is char*, l is length. See AeSetString() for details */
#define         AeXMLCharDataSetData(p, x, l) AeSetString(&(p)->pData, x, l)

/* Creates a new AeXMLAttribute structure */
AeXMLAttribute  *AeXMLAttributeNew(void);

/* Destroys the AeXMLAttribute structure */
void            AeXMLAttributeDestroy(AeXMLAttribute *pAttribute);

/* Sets the AeXMLAttribute name */
#define         AeXMLAttributeSetName(p, x) AeSetString(&(p)->pName, x, -1)

/* Sets the AeXMLAttribute value */
void            AeXMLAttributeSetValue(AeXMLAttribute *pAttribute, AeChar *pValue);

/* Creates a new AeXMLElement structure */
AeXMLElement    *AeXMLElementNew(void);

/* Destroys the AeXMLElement structure and nested objects. pContent is AeXMLElement pointer type */
void            AeXMLElementDestroy(AeXMLContent *pContent);

/* Sets the name of the XML Element. p is AeXMLElement pointer, x is some string */
#define         AeXMLElementSetName(p, x) AeSetString(&(p)->pName, x, -1)

/* Mark the XML Element to be empty or not. p is AeXMLElement pointer, x is AeTrue or AeFalse */
#define         AeXMLElementSetEmpty(p, x) ((p)->bEmpty = (x))

/* Adds the specified attribute and value pair to the AeXMLElement */
AeXMLAttribute  *AeXMLElementAddAttribute(AeXMLElement *pElement, AeChar *pName, AeChar *pValue);

/* All the following three functions add child element to the current (parent) XML Element */
/* pName is child Element name, bEmpty is AeTrue or AeFalse */
AeXMLElement    *AeXMLElementAddElement(AeXMLElement *pElement, AeChar *pName, AeBool bEmpty);
/* pData is string. iLength is string length */
AeXMLCharData   *AeXMLElementAddCharData(AeXMLElement *pElement, AeChar *pData, AeInt32 iLength);
/* pContent is child Element */
void            AeXMLElementAddContent(AeXMLElement *pElement, AeXMLContent *pContent);

/* Creates a new AeXMLDocument structure */
AeXMLDocument   *AeXMLDocumentNew(void);

/* Destroys the AeXMLDocument structure and its nested objects */
void            AeXMLDocumentDestroy(AeXMLDocument *pDocument);

/* Parses the input xml formatted stream into AeXMLDocument object. iLength is pSource size */
AeBool          AeXMLDocumentParse(AeXMLDocument *pDocument, AeChar *pSource, AeInt iLength);

/* Writes the AeXMLDocument object to xml formatted stream. Applications should have created the AeArray */
void            AeXMLDocumentWrite(AeXMLDocument *pDocument, AeArray *pArray);

/* Get the size of the formatted xml stream of the current AeXMLDocument */
AeInt32         AeXMLDocumentGetFormattedSize(AeXMLDocument *pDocument);

/* Convert escaped characters into valid strings per xml protocol */
void            AeXMLEntitize(AeChar *pInput, AeArray *pOutput);

#endif
