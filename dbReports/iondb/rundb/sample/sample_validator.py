# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from django.shortcuts import get_object_or_404

from iondb.rundb import models
import types
import datetime
import logging

import re

from iondb.rundb.models import SampleGroupType_CV, SampleAnnotation_CV, SampleSet, SampleSetItem, Sample
from iondb.utils import validation 

import views_helper

logger = logging.getLogger(__name__)

MAX_LENGTH_SAMPLE_NAME = 127
MAX_LENGTH_SAMPLE_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_EXTERNAL_ID = 127
MAX_LENGTH_SAMPLE_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_SET_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE = 1024
MAX_LENGTH_SAMPLE_NUCLEOTIDE_TYPE = 64
VALID_NUCLEOTIDE_TYPES = ["dna", "rna"]

MAX_LENGTH_PCR_PLATE_SERIAL_NUM = 64
MAX_LENGTH_SAMPLE_COUPLE_ID = 127
MAX_LENGTH_SAMPLE_EMBRYO_ID = 127

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
                
    pcrPlateSerialNum = queryDict.get("pcrPlateSerialNum", "").strip()

    return validate_sampleSet_values(sampleSetName, sampleSetDesc, pcrPlateSerialNum)


def validate_sampleSet_values(sampleSetName, sampleSetDesc, pcrPlateSerialNum, isNew = False):
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

    if validation.has_value(pcrPlateSerialNum):
        if not validation.is_valid_chars(pcrPlateSerialNum):
            return isValid, validation.invalid_chars_error("Error, Sample set PCR plate serial number")
        
        if not validation.is_valid_length(pcrPlateSerialNum, MAX_LENGTH_PCR_PLATE_SERIAL_NUM):
            errorMessage = validation.invalid_length_error("Error, Sample PCR plate serial number", MAX_LENGTH_PCR_PLATE_SERIAL_NUM) + ". It is currently %s characters long." % str(len(pcrPlateSerialNum.strip()))
            return isValid, errorMessage


    isValid = True
    return isValid, None

def validate_barcoding_samplesetitems(samplesetitems, barcodeKit, barcode, samplesetitem_id, pending_id=None, allPcrPlates=None, pcrPlateRow=None):
    isValid = True
    errorMessage = None
    id_to_validate = pending_id if pending_id else samplesetitem_id

    for item in samplesetitems:
        ##logger.debug("validate_barcoding_samplesetitems() item=%s; barcodeKit=%s; barcode=%s; id_to_validate=%s" %(item, barcodeKit, barcode, id_to_validate))
        item_id = None

        if type(item) == types.DictType:

            #check if you are editing against your self
            if len(samplesetitems) == 1 and str(pending_id) == str(item.get('pending_id')): return True, None
            barcodeKit1 = item.get('barcodeKit', barcodeKit)
            barcode1 = item.get('barcode', None)
            
            item_id = item.get('pending_id', None)
            item_pcrPlate = item.get('pcrPlateRow')
            if item_id and id_to_validate and int(item_id) == int(id_to_validate):
                continue
        else:
            dnabarcode = models.dnaBarcode.objects.filter(name = barcodeKit, id_str = barcode)
            #don't bother to compare to its old self
            if int(item.pk) == int(samplesetitem_id):
                barcode1 = None
                barcodeKit1 = None
            else:
                barcode1 = item.dnabarcode.id_str if item.dnabarcode else None
                barcodeKit1 = item.dnabarcode.name if item.dnabarcode else None

            item_id = item.pk
            item_pcrPlate = item.pcrPlateRow

        #ensure only 1 barcode kit for the whole sample set
        if barcodeKit and barcodeKit1 and barcodeKit != barcodeKit1:
            if not pcrPlateRow:
                isValid = False
                errorMessage = "Error, Only one barcode kit can be used for a sample set"
                return isValid, errorMessage

        #ensure only 1 barcode id_str per sample
        if barcode and barcode1 and barcode == barcode1:
            isValid = False
            errorMessage = "Error, A barcode can be assigned to only one sample in the sample set. %s has been assigned to another sample at PCR plate position (%s)" % (barcode, item_pcrPlate)
            if pcrPlateRow:
                if item.pcrPlateRow not in allPcrPlates:
                    return isValid, errorMessage
                else:
                    isValid = True
                    errorMessage = ""
                    return isValid, errorMessage
            isValid = False

            return isValid, errorMessage
        
    return isValid, errorMessage


def validate_barcoding_for_existing_sampleset(queryDict):
    samplesetitem_id = queryDict.get('id', None)
    item = models.SampleSetItem.objects.get(pk=samplesetitem_id)
    samplesetitems = item.sampleSet.samples.all()

    return validate_barcoding_samplesetitems(samplesetitems, queryDict.get('barcodeKit', None), queryDict.get('barcode', None), samplesetitem_id)


def validate_pcrPlate_position_samplesetitems(samplesetitems, pcrPlateRow, samplesetitem_id, pending_id=None, sampleset = None):
    isValid = True
    errorMessage = None
 
    id_to_validate = pending_id if pending_id else samplesetitem_id
     
    for item in samplesetitems:        
        ##logger.debug("validate_pcrPlate_position_samplesetitems() item=%s; pcrPlateRow=%s; id_to_validate=%s; sampleset=%s" %(item, pcrPlateRow, id_to_validate, sampleset))
        item_id = None

        if type(item) == types.DictType:
            pcrPlateRow1 = item.get("pcrPlateRow", "")
            item_id = item.get('pending_id', None)
        else:
            pcrPlateRow1 = item.pcrPlateRow
            item_id = item.pk
        
        #ensure only 1 pcr plate position per sample       
        #also check if you are editing against your self
        if pcrPlateRow and pcrPlateRow1 and pcrPlateRow.lower() == pcrPlateRow1.lower() and str(item_id) != str(id_to_validate):
            isValid = False
            errorMessage = "Error, A PCR plate position can only have one sample in it. Position %s has already been occupied by another sample" %(pcrPlateRow)

            return isValid, errorMessage

        if (sampleset and not pcrPlateRow and "amps_on_chef" in sampleset.libraryPrepType.lower()):
            isValid = False
            return isValid, "Error, A PCR plate position must be specified for AmpliSeq on Chef sample"
                   
    return isValid, errorMessage
 

def validate_pcrPlate_position_for_existing_sampleset(queryDict):
    samplesetitem_id = queryDict.get('id', None)
    item = models.SampleSetItem.objects.get(pk=samplesetitem_id)
    samplesetitems = item.sampleSet.samples.all()
 
    return validate_pcrPlate_position_samplesetitems(samplesetitems, queryDict.get('pcrPlateRow', ""), samplesetitem_id, None, item.sampleSet)


def validate_samplesetitem_update_for_existing_sampleset(sampleSetItem, sample, selectedDnaBarcode, selectedNucleotideType, selectedPcrPlateRow):
    """
    validate if the changed sampleSetItem will become identical to an existing one. Error off if identical 
    """
    isValid = True
    errorMessage = None
    
    sampleSet = get_object_or_404(SampleSet, pk = sampleSetItem.sampleSet.id)       
    
    if selectedPcrPlateRow:
        sampleSetItems = SampleSetItem.objects.filter(sampleSet = sampleSet, pcrPlateRow = selectedPcrPlateRow)
        
        if sampleSetItem.id:
            sampleSetItems = sampleSetItems.exclude(id = sampleSetItem.id)
            
        logger.debug("views_helper - pcrPlateRow _create_or_update_sampleSetItem sampleSetItem.id=%d sampleSetItems.count=%d" %(sampleSetItem.id, sampleSetItems.count()))
        
        if sampleSetItems.count() > 0: 
            logger.debug("views_helper - _create_or_update_sampleSetItem DUPLICATE - SKIP UPDATE for sampleSetItem.id=%d" %(sampleSetItem.id))
            isValid = False
            errorMessage = "Error, A PCR plate position can only have one sample in it. Position %s has already been occupied by another sample" %(selectedPcrPlateRow)
            return isValid, errorMessage
     
    sampleSetItems = SampleSetItem.objects.filter(sampleSet = sampleSet, sample = sample)

    if selectedNucleotideType:
        sampleSetItems = sampleSetItems.filter(nucleotideType = selectedNucleotideType)
        
        if sampleSetItem.id:
            sampleSetItems = sampleSetItems.exclude(id = sampleSetItem.id)
       
        logger.debug("views_helper - DNA/RNA _create_or_update_sampleSetItem sampleSetItem.id=%d sampleSetItems.count=%d" %(sampleSetItem.id, sampleSetItems.count()))
        
        if sampleSetItems.count() > 0: 
            logger.debug("views_helper - _create_or_update_sampleSetItem DUPLICATE - SKIP UPDATE for sampleSetItem.id=%d" %(sampleSetItem.id))
            isValid = False
            errorMessage = "Error, Another sample with the same name and DNA/RNA type already exists in this sample set"

    if selectedDnaBarcode: 
        sampleSetItems = SampleSetItem.objects.filter(sampleSet = sampleSet, dnabarcode = selectedDnaBarcode)     
        
        if sampleSetItem.id:
            sampleSetItems = sampleSetItems.exclude(id = sampleSetItem.id)
        
        logger.debug("views_helper - BARCODE _create_or_update_sampleSetItem sampleSetItem.id=%d sampleSetItems.count=%d" %(sampleSetItem.id, sampleSetItems.count()))
        
        #!!!could be same or different sample  
        if sampleSetItems.count() > 0: 
            logger.debug("views_helper - _create_or_update_sampleSetItem DUPLICATE - SKIP UPDATE for sampleSetItem.id=%d" %(sampleSetItem.id))
            isValid = False
            errorMessage = "Error, Another sample with the same barcode already exists in this sample set" 
 
    return isValid, errorMessage


def validate_barcoding_for_new_sampleset(request, queryDict):

    if 'input_samples' in request.session:
        if 'pending_sampleSetItem_list' in request.session["input_samples"] and len(request.session["input_samples"]['pending_sampleSetItem_list']) > 0:
            samplesetitems = request.session["input_samples"]['pending_sampleSetItem_list']
            return validate_barcoding_samplesetitems(samplesetitems,queryDict.get('barcodeKit', None), queryDict.get('barcode', None), None, pending_id=queryDict.get('pending_id', ""))

    return True, None

def validate_pcrPlate_position_for_new_sampleset(request, queryDict):

    if 'input_samples' in request.session:
        if 'pending_sampleSetItem_list' in request.session["input_samples"] and len(request.session["input_samples"]['pending_sampleSetItem_list']) > 0:
            samplesetitems = request.session["input_samples"]['pending_sampleSetItem_list']
            return validate_pcrPlate_position_samplesetitems(samplesetitems,queryDict.get('pcrPlateRow', ""), None, pending_id=queryDict.get('pending_id', ""))

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

def validate_pcrPlate_position(request, queryDict):
    """
        first we look at the session and retrieve the list of pending sampleset items 
        and validate if PCR plate is unique
    """
    samplesetitem_id = queryDict.get('id', None)

    if samplesetitem_id:
        return validate_pcrPlate_position_for_existing_sampleset(queryDict)
    else:
        return validate_pcrPlate_position_for_new_sampleset(request, queryDict)


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
        

def validate_sample_pgx_attributes_for_sampleSet(queryDict):
    """
    validate the sample PGx attribuets for sample set item creation/update
    return a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length will not be validated since maxLength has been specified in the form.
    """
    
    isValid = False
    if not queryDict:
        return isValid, "Error, No sample data to validate."

    biopsyDays = queryDict.get("biopsyDays", "0").strip()
    if not biopsyDays:
        biopsyDays = "0"
    isValid, errorMessage = validate_sampleBiopsyDays(biopsyDays)
    if not isValid:
        return isValid, errorMessage
    
    isValid, errorMessage = validate_sampleCoupleId(queryDict.get("coupleId", "").strip())
    if not isValid:
        return isValid, errorMessage
    
    isValid, errorMessage = validate_sampleEmbryoId(queryDict.get("embryoId", "").strip())
    return isValid, errorMessage

def validate_sampleBiopsyDays(sampleBiopsyDays):
    isValid, errorMessage = _validate_intValue(sampleBiopsyDays, "Biopsy Days")
    return isValid, errorMessage

def validate_sampleCoupleId(sampleCoupleId):
    return _validate_optional_text(sampleCoupleId,  MAX_LENGTH_SAMPLE_COUPLE_ID, "Couple ID")


def validate_sampleEmbryoId(sampleEmbryoId):
    return _validate_optional_text(sampleEmbryoId,  MAX_LENGTH_SAMPLE_EMBRYO_ID, "Embryo ID")

 
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


def _validate_optional_text(value, maxLength, displayedTerm):
    isValid, errorMessage = _validate_textValue(value.strip(), displayedTerm)
    if not isValid:
        return isValid, errorMessage

    if not validation.is_valid_length(value.strip(), maxLength):
        errorMessage = validation.invalid_length_error("Error, " + displayedTerm, maxLength) + ". It is currently %s characters long." % str(len(value.strip()))
        return isValid, errorMessage

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
  

def validate_nucleotideType(nucleotideType, displayedName='Sample Nucleotide Type'):
    """
    validate nucleotide type case-insensitively with leading/trailing blanks in the input ignored
    """
    isValid = True      
    errors = []
    input = ""
                    
    if nucleotideType:
        input = nucleotideType.strip().lower()
                           
        if not validation.is_valid_keyword(input, VALID_NUCLEOTIDE_TYPES):
            errors.append(validation.invalid_keyword_error(displayedName, VALID_NUCLEOTIDE_TYPES))
            isValid = False
             
    return isValid, errors, input

def validate_sample_data(queryDict):
    """
    validate the sample attributes for REST creation/update
    return a boolean isValid and a text string for error message, None if input passes validation
    """
    isValid = False
    if not queryDict:
        return isValid, "Error, No sample data to validate."

    sampleName = queryDict.get("sampleName", "")
    sampleDisplayedName = queryDict.get("sampleDisplayedName", "")

    if sampleName is None and sampleDisplayedName is None:
        return isValid, "Sample Name is missing. Require at least sample name or sample displayed name"

    sampleName = sampleName.strip()
    isValid, errorMessage = validate_sampleName(sampleName)
    if not isValid:
        return isValid, errorMessage

    sampleDisplayedName = sampleDisplayedName.strip()
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

    sampleStatus = queryDict.get("sampleStatus", "").strip()
    isValid, errorMessage = validate_sampleStatus(sampleStatus)
    if not isValid:
        return isValid, errorMessage

    isValid, errorMessage = validate_for_same_sampleName_displayedName(sampleDisplayedName,sampleName)
    if not isValid:
        return isValid, errorMessage


    isValid = True
    return isValid, None

def validate_for_same_sampleName_displayedName(sampleDisplayedName,sampleName):
    #Verify the sample name and displayed name are similar (allow spaces)
    sampleDisplayedName = '_'.join(sampleDisplayedName.split())
    isValid = False
    if sampleName != sampleDisplayedName:
        return isValid, "Sample name should match sample displayed name, with spaces allowed in the displayedName"

    isValid = True
    return isValid, None

def validate_sampleStatus(sampleStatus):
    displayedTerm = "Sample Status "
    isValid = False

    sample_default_status = Sample.ALLOWED_STATUS
    sample_default_status = [sample[1] for sample in sample_default_status]
    sample_default_status = [s.lower() for s in sample_default_status]

    for default_sample in sample_default_status:
        if sampleStatus.lower()==default_sample or sampleStatus == "":
            isValid = True
            return isValid, None

    if not isValid:
        sample_default_status_trim = ', '.join([sample for sample in sample_default_status])
        return isValid, "The sample status(%s) is not valid. Default Values are: %s"  % (sampleStatus, sample_default_status_trim)

def validate_sampleName(sampleName):
    displayedTerm = "Sample name"

    isValid, errorMessage = _validate_textValue_mandatory(sampleName, displayedTerm)

    if not isValid:
        return isValid, errorMessage

    isValid, errorMessage =  _validate_textValue(sampleName, displayedTerm)
    if not isValid:
        return isValid, errorMessage

    isValid, errorMessage = _validate_textValue_leadingChars(sampleName, displayedTerm)
    if not isValid:
        return isValid, errorMessage

    if not validation.is_valid_length(sampleName.strip(), MAX_LENGTH_SAMPLE_NAME):
        errorMessage = validation.invalid_length_error("Error, Sample name", MAX_LENGTH_SAMPLE_NAME) + ". It is currently %s characters long." % str(len(sampleName.strip()))
        return isValid, errorMessage

    return True, None

def validate_pcrPlateRow(pcrPlateRow, displayedName='PCR Plate Position'):
    """
    validate PCR plate row case-insensitively with leading/trailing blanks in the input ignored
    """
    isValid = True      
    errors = []
    input = ""
                    
    if pcrPlateRow:
        input = pcrPlateRow.strip().upper()

        validValues = views_helper._get_pcrPlateRow_valid_values(None)
        if not validation.is_valid_keyword(input, validValues):
            errors.append(validation.invalid_keyword_error(displayedName, validValues))
            isValid = False
             
    return isValid, errors, input

def validate_pcrPlateCol(pcrPlateCol, displayedName='PCR Plate Position'):
    """
    validate PCR plate row case-insensitively with leading/trailing blanks in the input ignored
    """
    isValid = True
    errors = []
    input = ""

    if pcrPlateCol:
        valid_tuples = SampleSetItem.ALLOWED_AMPLISEQ_PCR_PLATE_COLUMNS_V1
        input = pcrPlateCol.strip().upper()

        validValues = views_helper._get_pcrPlateCol_valid_values(None)
        if not validation.is_valid_keyword(input, validValues):
            errors.append(validation.invalid_keyword_error(displayedName, validValues))
            isValid = False

    return isValid, errors, input

def validate_samplesetStatus(samplesetStatus, displayedTerm="Status"):
    """
    validate samplesetStatus with leading/trailing blanks in the input ignored
    """
    isValid = True
    errors = []
    inputData = ""

    if samplesetStatus:
        inputData = samplesetStatus.strip().lower()
        validValues = views_helper._get_sampleset_choices(None)
        if not validation.is_valid_keyword(inputData, validValues):
            errors.append(validation.invalid_keyword_error(displayedTerm, validValues))
            errors = ''.join(errors).replace("are ,", ":")
            isValid = False
            return isValid, errors, samplesetStatus

    isValid = True
    return isValid, None, None

def validate_libPrepType(libPrepType, displayedTerm="Library Prep Type"):
    """
    validate libPrepType with leading/trailing blanks in the input ignored
    """
    isValid = True
    errors = []
    inputData = ""
    if libPrepType:
        inputData = libPrepType.strip()

        validValues = views_helper._get_libraryPrepType_choices(None)
        if not validation.is_valid_keyword(inputData, validValues):
            errors.append(validation.invalid_keyword_error(displayedTerm, validValues))
            errors = ''.join(errors).replace("are ,", ":")
            isValid = False
            return isValid, errors, libPrepType

    isValid = True
    return isValid, None, None

def validate_sampleBarcodeMapping(queryDict):
    """
    validate sampleBarcodeMapping input sent via API
        - BarcodeKit, Barcode
        - SampleRow, SampleColumn
    """
    isValid = False
    errordict = {}
    sampleset_id = queryDict.get('samplesetID', None)
    sampleSet = models.SampleSet.objects.get(pk=sampleset_id)
    pcrplateBarcodeQueryDict = queryDict.get('sampleBarcodeMapping',None)

    #Input JSON object from Chef to Update
    allBarcodeKits = [pcr_plate_barcode["sampleToBarcode"]["barcodeKit"] for pcr_plate_barcode in pcrplateBarcodeQueryDict ]
    allBarcodes = [pcr_plate_barcode["sampleToBarcode"]["barcode"] for pcr_plate_barcode in pcrplateBarcodeQueryDict ]
    allPcrPlates = [pcr_plate_barcode["sampleToBarcode"]["sampleRow"] for pcr_plate_barcode in pcrplateBarcodeQueryDict ]
    singleBarcode = allBarcodeKits and all(allBarcodeKits[0] == elem for elem in allBarcodeKits)

    #validate if same barcode is being used for multiple samples
    if len(allBarcodes) != len(set(allBarcodes)):
        dupBarcode = [x for x in allBarcodes if allBarcodes.count(x) >= 2]
        errordict = {'result': '1',
                     'message': 'Fail',
                     'detailMessage' : "Error, A barcode can be assigned to only one sample in the sample set.",
                     'inputData' : dupBarcode
                     }
        return isValid, errordict

    if not singleBarcode:
        errordict = {'result': '1',
                     'message': 'Fail',
                     'detailMessage' : "Error, Only one barcode Kit can be used for a sample set",
                     'inputData' : allBarcodeKits
                     }
        return isValid, errordict

    if pcrplateBarcodeQueryDict:
        sampleSetItems = sampleSet.samples.all()
        userPcrPlates = [item.pcrPlateRow for item in sampleSetItems]
        isValid, errorMessage = validate_user_chef_barcodeKit(sampleSetItems, allBarcodeKits, allPcrPlates)
        if not isValid:
            return isValid, errorMessage
        for pcr_plate_barcode in pcrplateBarcodeQueryDict:
            barcodeKit = pcr_plate_barcode["sampleToBarcode"]["barcodeKit"]
            barcode = pcr_plate_barcode["sampleToBarcode"]["barcode"]
            row = pcr_plate_barcode["sampleToBarcode"]["sampleRow"]

            #validate pcrPlate Row
            isValid, errormsg, inputData = validate_pcrPlateRow(row)
            if not isValid:
                errordict = {'result': '1',
                             'message': 'Fail',
                             'detailMessage' : ''.join(errormsg),
                             'inputData' : inputData
                            }
                return isValid, errordict
            else:
                isValid = False

            #validate pcrPlate Column
            col = pcr_plate_barcode["sampleToBarcode"]["sampleColumn"]
            isValid, errormsg, inputData = validate_pcrPlateCol(col)
            if not isValid:
                errordict = {'result': '1',
                             'message': 'Fail',
                             'detailMessage' : ''.join(errormsg),
                             'inputData' : inputData
                            }
                return isValid, errordict
            else:
                isValid = False

            #validate the specified barcode belongs to appropriate barcodeKit
            isValid, errormsg, items = validate_barcodekit_and_id_str(barcodeKit, barcode)
            if not isValid:
                errordict = {'inputData' : [barcodeKit,barcode],
                             'result': '1',
                             'message': 'Fail',
                             'detailMessage' : errormsg
                             }
                return isValid, errordict

            #Override the barcode and barcodeKit if User specified PCR Plate row and chef specified PCR Plate rows are similar
            #Validate if there is any pcrPlate Row mismatch between User and Chef Inputs for Data integrity
            mistmatch_PcrPlates = set(userPcrPlates) - set(allPcrPlates)
            if len(mistmatch_PcrPlates):
                isValid, errors = validate_barcoding_samplesetitems(sampleSetItems,barcodeKit,barcode,sampleset_id,allPcrPlates=allPcrPlates,pcrPlateRow=row)
            if not isValid:
                return isValid, errors

    isValid = True
    return isValid, None

def validate_user_chef_barcodeKit(samplesetitems,chef_barcodeKit, allPcrPlates):
    all_TS_barcodeKit = []
    userPcrPlates = []
    isValid = True
    errorMessage = ""

    for item in samplesetitems:
        barcodeKit_TS = item.dnabarcode.name if item.dnabarcode else None
        userPcrPlate = item.pcrPlateRow
        all_TS_barcodeKit.append(barcodeKit_TS)
        userPcrPlates.append(userPcrPlate)
    barcodeKit_mistmatch = set(chef_barcodeKit) - set(all_TS_barcodeKit)
    pcrPlate_mistmatch = set(userPcrPlates) - set(allPcrPlates)

    if len(pcrPlate_mistmatch) and len(barcodeKit_mistmatch):
        print pcrPlate_mistmatch
        print barcodeKit_mistmatch
        isValid = False
        errorMessage = "Error, Only one barcode kit can be used for a sample set"
        return isValid, errorMessage
    return isValid, errorMessage


def validate_sampleSets_for_planning(sampleSets):
    ''' Validate multiple sampleSets are compatible to create a Plan from '''
    errors = []
    items = SampleSetItem.objects.filter(sampleSet__in=sampleSets)
    if not items:
        errors.append('Sample Set must have at least one sample')
        return errors
    
    samples_w_barcodes = items.exclude(dnabarcode__isnull=True)
    barcodeKitNames = samples_w_barcodes.values_list('dnabarcode__name', flat=True).distinct()
    if len(barcodeKitNames) > 1:
        errors.append('Selected Sample Sets have different Barcode Kits: %s.' % ', '.join(barcodeKitNames))
    elif len(barcodeKitNames) == 1:
        barcodes = {}
        for barcode, sample, setname in samples_w_barcodes.values_list('dnabarcode__id_str', 'sample__name', 'sampleSet__displayedName'):
            if barcode in barcodes:
                msg = 'Multiple samples are assigned to barcode %s: %s (%s), %s (%s)' % (barcode, sample, setname, barcodes[barcode][0], barcodes[barcode][1])
                errors.append(msg)
            else:
                barcodes[barcode] = (sample, setname)

    return errors
