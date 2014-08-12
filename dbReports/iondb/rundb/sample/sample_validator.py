# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb import models
import types
import datetime
import logging

import re


from iondb.rundb.models import SampleGroupType_CV, SampleAnnotation_CV, SampleSet
from iondb.utils import validation 

logger = logging.getLogger(__name__)

MAX_LENGTH_SAMPLE_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_EXTERNAL_ID = 127
MAX_LENGTH_SAMPLE_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_SET_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE = 1024

ERROR_MSG_INVALID_DATATYPE = " should be a whole number. "
ERROR_MSG_INVALID_PERCENTAGE = " should be a whole number between 0 to 100"

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


def validate_sampleSet_values(sampleSetName, sampleSetDesc, isNew = False):
    """ 
    validate the sampleSet input. 
    returns a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length willl not be validated since maxLength has been specified in the form.
    """
    
    isValid = False
    if not validation.has_value(sampleSetName):
        return isValid, validation.required_error("Error, Sample set name")
    else:        
        if not validation.is_valid_chars(sampleSetName):
            return isValid, validation.invalid_chars_error("Error, Sample set name")
        
        if not validation.is_valid_length(sampleSetName, MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME):
            errorMessage = validation.invalid_length_error("Error, Sample set name", MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME) + ". It is currently %s characters long." % str(len(sampleSetName.strip()))
            return isValid, errorMessage

        if isNew:
            #error if new sample set already exists
            existingSampleSets = SampleSet.objects.filter(displayedName = sampleSetName)
            if existingSampleSets:
                errorMessage = "Error, Sample set %s already exists." % (sampleSetName)           
                return isValid, errorMessage
    
    if validation.has_value(sampleSetDesc):
        if not validation.is_valid_chars(sampleSetDesc):
            return isValid, validation.invalid_chars_error("Error, Sample set description")
        
        if not validation.is_valid_length(sampleSetDesc, MAX_LENGTH_SAMPLE_SET_DESCRIPTION):
            errorMessage = validation.invalid_length_error("Error, Sample set description", MAX_LENGTH_SAMPLE_SET_DESCRIPTION) + ". It is currently %s characters long." % str(len(sampleSetDesc.strip()))
            return isValid, errorMessage

    isValid = True
    return isValid, None

def validate_barcoding_samplesetitems(samplesetitems, barcodeKit, barcode, samplesetitem_id, pending_id=None):
    isValid = True
    errorMessage = None

    id_to_validate = pending_id if pending_id else samplesetitem_id
    
    for item in samplesetitems:        
        
        ##logger.debug("validate_barcoding_samplesetitems() item=%s; barcode=%s; id_to_validate=%s" %(item, barcode, id_to_validate))
        item_id = None
        
        if type(item) == types.DictType:
            #check if you are editing against your self
            if len(samplesetitems) == 1 and str(pending_id) == str(item.get('pending_id')): return True, None
            barcodeKit1 = item.get('barcodeKit', barcodeKit)
            barcode1 = item.get('barcode', None)
            
            item_id = item.get('pending_id', None)

            if item_id and id_to_validate and int(item_id) == int(id_to_validate):
                continue
        else:
            dnabarcode = models.dnaBarcode.objects.filter(name = barcodeKit, id_str = barcode)
            if int(item.pk) == int(samplesetitem_id):
                barcode1 = None
            else:
                barcode1 = item.dnabarcode.name if item.dnabarcode else None      
            if len(dnabarcode) > 0:
                barcodeKit1 = dnabarcode[0].name
            else:
                barcodeKit1 = None
                
            item_id = item.pk

        #ensure only 1 barcode kit for the whole sample set
        if barcodeKit and barcodeKit1 and barcodeKit != barcodeKit1:
            isValid = False
            errorMessage = "Error, Only one barcode kit can be used for a sample set"
            return isValid, errorMessage

        #ensure only 1 barcode id_str per sample
        if barcode and barcode1 and barcode == barcode1:
            isValid = False
            errorMessage = "Error, A barcode can be assigned to only one sample in the sample set and %s has been assigned to another sample" %(barcode)
            return isValid, errorMessage
        
    return isValid, errorMessage


def validate_barcoding_for_existing_sampleset(queryDict):
    samplesetitem_id = queryDict.get('id', None)
    item = models.SampleSetItem.objects.get(pk=samplesetitem_id)
    samplesetitems = item.sampleSet.samples.all()

    return validate_barcoding_samplesetitems(samplesetitems, queryDict.get('barcodeKit', None), queryDict.get('barcode', None), samplesetitem_id)


def validate_barcoding_for_new_sampleset(request, queryDict):

    if 'input_samples' in request.session:
        if 'pending_sampleSetItem_list' in request.session["input_samples"] and len(request.session["input_samples"]['pending_sampleSetItem_list']) > 0:
            samplesetitems = request.session["input_samples"]['pending_sampleSetItem_list']
            return validate_barcoding_samplesetitems(samplesetitems,queryDict.get('barcodeKit', None), queryDict.get('barcode', None), None, pending_id=queryDict.get('pending_id', ""))

    return True, None


def validate_barcoding(request, queryDict):
    """
        first we look at the session and retrieve the list of pending sampleset items 
        and validate barcodes according to jira ticket TS-7930
    """
    ##logger.debug("ENTER validate_barcoding() queryDict=%s" %(queryDict))
    
    samplesetitem_id = queryDict.get('id', None)

    if samplesetitem_id:
        return validate_barcoding_for_existing_sampleset(queryDict)
    else:
        return validate_barcoding_for_new_sampleset(request, queryDict)


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

    cellularityPct = queryDict.get("cellularityPct", None).strip()

    isValid, errorMessage, value = validate_cellularityPct(cellularityPct)

    if errorMessage and not isValid:
        return isValid, errorMessage
    elif value and not isValid:
        queryDict["cellularityPct"] = str(value)
    
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

    if not validation.is_valid_length(sampleDisplayedName.strip(), MAX_LENGTH_SAMPLE_DISPLAYED_NAME):
        errorMessage = validation.invalid_length_error("Error, Sample name", MAX_LENGTH_SAMPLE_DISPLAYED_NAME) + ". It is currently %s characters long." % str(len(sampleDisplayedName.strip()))
        return isValid, errorMessage
     
    return True, None
 
   
def validate_sampleExternalId(sampleExternalId):
    isValid = False
    isValid, errorMessage = _validate_textValue(sampleExternalId, "Sample ID ")
    
    if not isValid:
        return isValid, errorMessage

    if not validation.is_valid_length(sampleExternalId.strip(), MAX_LENGTH_SAMPLE_EXTERNAL_ID):
        errorMessage = validation.invalid_length_error("Error, Sample id", MAX_LENGTH_SAMPLE_EXTERNAL_ID) + ". It is currently %s characters long." % str(len(sampleExternalId.strip()))
        return isValid, errorMessage

    return True, None

   
def validate_sampleDescription(sampleDescription):
    isValid = False
        
    if validation.has_value(sampleDescription):
        isValid, errorMessage = _validate_textValue(sampleDescription, "Sample description ")
        if not isValid:
            return isValid, errorMessage
    
        if not validation.is_valid_length(sampleDescription.strip(), MAX_LENGTH_SAMPLE_DESCRIPTION):
            errorMessage = validation.invalid_length_error("Error, Sample description", MAX_LENGTH_SAMPLE_DESCRIPTION) + ". It is currently %s characters long." % str(len(sampleDescription.strip()))
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

   
def validate_cancerType(cancerType):
    if not cancerType:
        return True, None, cancerType
    
    cancerTypes = SampleAnnotation_CV.objects.filter(annotationType = "cancerType", isActive = True, value__iexact = cancerType)
    
    isValid = False
    if cancerTypes.count() == 0:
        return isValid, "Error, Cancer type value is not valid. ", cancerType
    
    return True, None, cancerTypes[0]
   

def validate_cellularityPct(cellularityPct):
    """ 
    check if input is a positive integer between 0 and 100 inclusively.
    If missing return default value to use.
    """
    
    if cellularityPct.isdigit():
        value = int(cellularityPct)
        if value < 0 or value > 100:
            return False, "Error, Cellularity %" + ERROR_MSG_INVALID_PERCENTAGE, value
        else:     
            return True, None, value
    else:
        if cellularityPct:
            return False, "Error, Cellularity %" + ERROR_MSG_INVALID_DATATYPE, cellularityPct
        else:
            return False, None, 0
        

 
def _validate_textValue_mandatory(value, displayedTerm):
    isValid = False
    if not validation.has_value(value):
        return isValid, "Error, " + validation.required_error(displayedTerm)
            
    return True, None


def _validate_intValue(value, displayedTerm):
    if value.isdigit():
        return True, None
    
    return False, "Error, " + displayedTerm + ERROR_MSG_INVALID_DATATYPE


def _validate_textValue(value, displayedTerm):
    isValid = False
    if value and not validation.is_valid_chars(value):
        return isValid, "Error, " + validation.invalid_chars_error(displayedTerm)
        
    return True, None


def _validate_textValue_leadingChars(value, displayedTerm):
    isValid = False
    if value and not validation.is_valid_leading_chars(value):
        return isValid, "Error, " + validation.invalid_leading_chars(displayedTerm)
        
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
    
    if not validation.has_value(value):
        if attribute.isMandatory:
            return isValid, "Error, "+ validation.required_error(attribute.displayedName)
    else:
        aValue = value.strip()
        if attribute.dataType.dataType == "Text" and not validation.is_valid_chars(aValue):
            return isValid, "Error, "+ validation.invalid_chars_error(attribute.displayedName)
        if attribute.dataType.dataType == "Integer" and not aValue.isdigit():
            return isValid, "Error, "+ attribute.displayedName + ERROR_MSG_INVALID_DATATYPE
        if not validation.is_valid_chars(aValue):
            return isValid, "Error, "+ validation.invalid_chars_error(attribute.displayedName)

        if not validation.is_valid_length(aValue, MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE):
            errorMessage = validation.invalid_length_error("Error, User-defined sample attribute value", MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE) + ". It is currently %s characters long." % str(len(aValue.strip()))
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
    
    if not validation.has_value(attributeName):
        return isValid, validation.required_error("Error, Attribute name")
    if not validation.is_valid_chars(attributeName.strip()):
        return isValid, validation.invalid_chars_error("Error, Attribute name")
            
    if not validation.is_valid_length(attributeName.strip(), MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME):
        errorMessage = validation.invalid_length_error("Error, User-defined sample attribute", MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME) + ". It is currently %s characters long." % str(len((attributeName.strip())))
        return isValid, errorMessage
    
    if not validation.is_valid_chars(attributeDescription):
        return isValid, validation.invalid_chars_error("Error, Attribute description")

    if not validation.is_valid_length(attributeDescription.strip(), MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION):
        errorMessage = validation.invalid_length_error("Error, User-defined sample attribute description", MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION) + ". It is currently %s characters long." % str(len(attributeDescription.strip()))
        return isValid, errorMessage
    
    isValid = True
    return isValid, None 

def validate_barcodekit_and_id_str(barcodeKit, barcode_id_str):
    isValid = True
    item = ''
    errorMessage = ''

    if not barcodeKit and not barcode_id_str:
        return isValid, errorMessage, item
    
    #First validate that if the barcodeKit is entered then the id_str must also be entered
    if barcodeKit and not barcode_id_str:
        return False, "Error, Please enter a barcode item", 'barcode_id_str'
    #Next validate that if the id_str is entered the barcodeKit must also be entered
    elif barcode_id_str and not barcodeKit:
        return False, "Error, Please enter a Barcoding Kit", 'barcodeKit'
    #Next validate that the barcodeKit is spelled correctly
    dnabarcode = models.dnaBarcode.objects.filter(name__iexact=barcodeKit)
    if dnabarcode.count() == 0:
        return False, "Error, Invalid Barcodekit", 'barcodeKit'
    #Next validate the that id_str is spelled correctly
    dnabarcode = models.dnaBarcode.objects.filter(id_str__iexact=barcode_id_str)
    if dnabarcode.count() == 0:
        return False, "Error, Invalid barcode", 'barcode_id_str'
    #Next validate that the Barcodekit and barcode belong together
    dnabarcode = models.dnaBarcode.objects.filter(name__iexact=barcodeKit, id_str__iexact=barcode_id_str)
    if dnabarcode.count() != 1:
        return False, "Error, Invalid Barcodekit and Barcode combination", 'barcodeKit'

    return isValid, errorMessage, item
  
    
