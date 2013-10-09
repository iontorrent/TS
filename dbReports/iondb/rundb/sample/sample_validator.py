# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb import models
import datetime
import logging

import re


from iondb.rundb.models import SampleGroupType_CV,  \
    SampleAnnotation_CV


logger = logging.getLogger(__name__)

MAX_LENGTH_SAMPLE_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_EXTERNAL_ID = 127
MAX_LENGTH_SAMPLE_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_SET_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE = 1024

ERROR_MSG_INVALID_CHARS = " should contain only numbers, letters, spaces, and the following: . - _"
ERROR_MSG_INVALID_LENGTH = " length should be %s characters maximum. "
ERROR_MSG_INVALID_LEADING_CHARS = " should only start with numbers or letters. "
ERROR_MSG_INVALID_DATATYPE = " should be a whole number. "



def _is_valid_chars(value, validChars=r'^[a-zA-Z0-9-_\.\s]+$'):
    ''' Determines if value is valid: letters, numbers, spaces, dashes, underscores, dots only '''
    return bool(re.compile(validChars).match(value))


def _is_invalid_leading_chars(value, invalidChars=r'[\.\-\_]'):
    ''' Determines if leading characters contain dashes, underscores or dots '''
    if value:
        return bool(re.compile(invalidChars).match(value.strip(), 0))
    else:
        True
    

def _is_valid_length(value, maxLength = 0):
    ''' Determines if value length is within the maximum allowed '''
    if value:
        return len(value.strip()) <= maxLength
    return True


def _has_value(value):
    ''' Determines if a value is present '''
    if value:
        return len(value.strip()) > 0
    return False


def validate_sampleSet(queryDict):
    """ 
    validate the sampleSet input. 
    returns a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length willl not be validated since maxLength has been specified in the form.
    """  
    
    logger.debug("sample_validator.validate_sampleset() queryDict=%s" %(queryDict))
                 
    isValid = False
    if not queryDict:
        return isValid, "Error, No sample set data to validate."
    
    sampleSetName = queryDict.get("sampleSetName", "").strip()
                
    sampleSetDesc = queryDict.get("sampleSetDescription", "").strip()

    return validate_sampleSet_values(sampleSetName, sampleSetDesc)    

    
def validate_sampleSet_values(sampleSetName, sampleSetDesc):
    """ 
    validate the sampleSet input. 
    returns a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length willl not be validated since maxLength has been specified in the form.
    """
    
    isValid = False
    if not _has_value(sampleSetName):
        return isValid, "Error, Sample set name is required."
    else:        
        if not _is_valid_chars(sampleSetName):
            return isValid, "Error, Sample set name "+ ERROR_MSG_INVALID_CHARS
        
        if not _is_valid_length(sampleSetName, MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME):
            errorMessage = "Error, Sample set name should be %s characters maximum. It is currently %s characters long." % (str(MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME), str(len(sampleSetName.strip())))            
            return isValid, errorMessage
    
    if _has_value(sampleSetDesc):
        if not _is_valid_chars(sampleSetDesc):
            return isValid, "Error, Sample set description "+ ERROR_MSG_INVALID_CHARS
        
        if not _is_valid_length(sampleSetDesc, MAX_LENGTH_SAMPLE_SET_DESCRIPTION):
            errorMessage = "Error, Sample set description should be %s characters maximum. It is currently %s characters long." % (str( MAX_LENGTH_SAMPLE_SET_DESCRIPTION), str(len(sampleSetDesc.strip())))            
            return isValid, errorMessage

    isValid = True
    return isValid, None


def validate_sample_for_sampleSet(queryDict):
    """
    validate the sample for sample set item creation/update
    return a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length will not be validated since maxLength has been specified in the form.
    """
    
    isValid = False
    if not queryDict:
        return isValid, "Error, No sample data to validate."
    
    sampleDisplayedName = queryDict.get("sampleName", "").strip()
    
    isValid, errorMessage = validate_sampleDisplayedName(sampleDisplayedName)
    if not isValid:
        return isValid, errorMessage

    sampleExternalId = queryDict.get("sampleExternalId", "").strip()
    
    isValid, errorMessage = validate_sampleExternalId(sampleExternalId)
    if not isValid:
        return isValid, errorMessage
                    
    sampleDesc = queryDict.get("sampleDescription", "").strip()
    
    isValid, errorMessage = validate_sampleDescription(sampleDesc)
    if not isValid:
        return isValid, errorMessage

    isValid = True
    return isValid, None   


def validate_sampleDisplayedName(sampleDisplayedName):
    displayedTerm = "Sample name "
    isValid, errorMessage = _validate_textValue_mandatory(sampleDisplayedName, displayedTerm)

    if not isValid:
        return isValid, errorMessage
    
    isValid, errorMessage =  _validate_textValue(sampleDisplayedName, displayedTerm)

    if not isValid:
        return isValid, errorMessage
    
    isValid, errorMessage = _validate_textValue_leadingChars(sampleDisplayedName, displayedTerm)
    
    if not isValid:
        return isValid, errorMessage

    if not _is_valid_length(sampleDisplayedName.strip(), MAX_LENGTH_SAMPLE_DISPLAYED_NAME):
        errorMessage = "Error, Sample name should be %s characters maximum. It is currently %s characters long." % (str(MAX_LENGTH_SAMPLE_DISPLAYED_NAME), str(len(sampleDisplayedName.strip())))            
        return isValid, errorMessage
     
    return True, None
 
   
def validate_sampleExternalId(sampleExternalId):
    isValid = False
    isValid, errorMessage = _validate_textValue(sampleExternalId, "Sample ID ")
    
    if not isValid:
        return isValid, errorMessage

    if not _is_valid_length(sampleExternalId.strip(), MAX_LENGTH_SAMPLE_EXTERNAL_ID):
        errorMessage = "Error, Sample id should be %s characters maximum. It is currently %s characters long." % (str(MAX_LENGTH_SAMPLE_EXTERNAL_ID), str(len(sampleExternalId.strip())))            
        return isValid, errorMessage

    return True, None

   
def validate_sampleDescription(sampleDescription):
    isValid = False
        
    if _has_value(sampleDescription):
        isValid, errorMessage = _validate_textValue(sampleDescription, "Sample description ")
        if not isValid:
            return isValid, errorMessage
    
        if not _is_valid_length(sampleDescription.strip(), MAX_LENGTH_SAMPLE_DESCRIPTION):
            errorMessage = "Error, Sample description should be %s characters maximum. It is currently %s characters long." % (str(MAX_LENGTH_SAMPLE_DESCRIPTION), str(len(sampleDescription.strip())))            
            return isValid, errorMessage
    
    return True, None

       
def validate_sampleGender(sampleGender):
    if not sampleGender:
        return True, None, sampleGender
    
    genders = SampleAnnotation_CV.objects.filter(annotationType = "gender", isActive = True, value__iexact = sampleGender)
    
    isValid = False
    if genders.count() == 0:
        return isValid, "Error, Gender value is not valid. ", sampleGender
    
    return True, None, genders[0]

       
def validate_sampleGroupType(sampleGroupType):
    if not sampleGroupType:
        return True, None, sampleGroupType
    
    roles = SampleAnnotation_CV.objects.filter(annotationType = "relationshipRole", isActive = True, value__iexact = sampleGroupType)
    
    isValid = False
    if roles.count() == 0:
        return isValid, "Error, Group type value is not valid. ", sampleGroupType
    
    return True, None, roles[0]    
        
        
def validate_sampleGroup(sampleGroup):
    if sampleGroup.isdigit():
        return True, None
    
    return False, "Error, Sample group" + ERROR_MSG_INVALID_DATATYPE


def _validate_textValue_mandatory(value, displayedTerm):
    isValid = False
    if not _has_value(value):
        return isValid, "Error, " + displayedTerm + "is required."
            
    return True, None


def _validate_intValue(value, displayedTerm):
    if value.isdigit():
        return True, None
    
    return False, "Error, " + displayedTerm + ERROR_MSG_INVALID_DATATYPE


def _validate_textValue(value, displayedTerm):
    isValid = False
    if value and not _is_valid_chars(value):
        return isValid, "Error, " + displayedTerm + ERROR_MSG_INVALID_CHARS
        
    return True, None


def _validate_textValue_leadingChars(value, displayedTerm):
    isValid = False
    if value and _is_invalid_leading_chars(value):
        return isValid, "Error, " + displayedTerm + ERROR_MSG_INVALID_LEADING_CHARS
        
    return True, None


def validate_sampleAttribute(attribute, value):
    """
    validate the sample attribute value for the attribute of interest
    return a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length will not be validated since maxLength has been specified in the form.
    """    
        
    isValid = False
    if not attribute:
        return isValid, "Error, No sample attribute to validate."
    
    if not _has_value(value):
        if attribute.isMandatory:
            return isValid, "Error, "+ attribute.displayedName + " value is required."
    else:
        aValue = value.strip()
        if attribute.dataType.dataType == "Text" and not _is_valid_chars(aValue):
            return isValid, "Error, "+ attribute.displayedName + ERROR_MSG_INVALID_CHARS
        if attribute.dataType.dataType == "Integer" and not aValue.isdigit():
            return isValid, "Error, "+ attribute.displayedName + ERROR_MSG_INVALID_DATATYPE
        if not _is_valid_chars(aValue):
            return isValid, "Error, "+ attribute.displayedName + ERROR_MSG_INVALID_CHARS

        if not _is_valid_length(aValue, MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE):
            errorMessage = "Error, User-defined sample attribute value should be %s characters maximum. It is currently %s characters long." % (str(MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE), str(len(aValue.strip())))            
            return isValid, errorMessage
        
    isValid = True
    return isValid, None   


def validate_sampleAttribute_mandatory_for_no_value(attribute):
        
    isValid = False
    if not attribute:
        return isValid, "Error, No sample attribute to validate."
        
    if attribute.isMandatory:
        return isValid, "Error, "+ attribute.displayedName + " value is required."

    isValid = True
    return isValid, None   

                
def validate_sampleAttribute_definition(attributeName, attributeDescription):
    """
    validate the sample attribute definition
    return a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length will not be validated since maxLength has been specified in the form.
    """    
        
    isValid = False
    
    if not _has_value(attributeName):
        return isValid, "Error, Attribute name is required."
    if not _is_valid_chars(attributeName.strip()):
        return isValid, "Error, Attribute name " + ERROR_MSG_INVALID_CHARS
            
    if not _is_valid_length(attributeName.strip(), MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME):
        errorMessage = "Error, User-defined sample attribute should be %s characters maximum. It is currently %s characters long." % (str(MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME), str(len((attributeName.strip()))))            
        return isValid, errorMessage
    
    if not _is_valid_chars(attributeDescription):
        return isValid, "Error, Attribute description " + ERROR_MSG_INVALID_CHARS

    if not _is_valid_length(attributeDescription.strip(), MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION):
        errorMessage = "Error, User-defined sample attribute description should be %s characters maximum. It is currently %s characters long." % (str(MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION), str(len(attributeDescription.strip())))            
        return isValid, errorMessage
    
    isValid = True
    return isValid, None   
    