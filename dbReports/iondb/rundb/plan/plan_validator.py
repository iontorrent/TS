# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb.models import Chip, RunType, KitInfo, ApplicationGroup, SampleGroupType_CV, dnaBarcode, ReferenceGenome
from iondb.utils import validation

import logging
logger = logging.getLogger(__name__)


MAX_LENGTH_PLAN_NAME = 512
MAX_LENGTH_SAMPLE_NAME = 127
MAX_LENGTH_SAMPLE_ID = 127
MAX_LENGTH_PROJECT_NAME = 64
MAX_LENGTH_NOTES = 1024
MAX_LENGTH_SAMPLE_TUBE_LABEL = 512
MAX_FLOWS = 2000
MAX_QC_INT = 100
PROJECT_NAME_LENGTH = 64

VALID_NUCLEOTIDE_TYPES = ["DNA", "RNA"]

def validate_plan_name(value, displayedName='Plan Name'):
    errors = []
    if not validation.has_value(value):
        errors.append(validation.required_error(displayedName))
        
    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))
    
    if not validation.is_valid_length(value, MAX_LENGTH_PLAN_NAME):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_PLAN_NAME))
    
    return errors

def validate_notes(value, displayedName='Notes'):
    errors = []
    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))
    
    if not validation.is_valid_length(value, MAX_LENGTH_NOTES):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_NOTES))
        
    return errors

def validate_sample_name(value, displayedName='Sample Name'):
    errors = []    
    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))
        
    if not validation.is_valid_leading_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))
    
    if not validation.is_valid_length(value, MAX_LENGTH_SAMPLE_NAME):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_SAMPLE_NAME))

    return errors


def validate_barcoded_sample_info(sampleName, sampleId, nucleotideType, runTypeName, sampleReference, displayedName='Barcoded Sample'):
    errors = []
    if not validation.is_valid_chars(sampleName):
        errors.append(validation.invalid_chars_error(displayedName))
        
    if not validation.is_valid_leading_chars(sampleName):
        errors.append(validation.invalid_chars_error(displayedName))
    
    if not validation.is_valid_length(sampleName, MAX_LENGTH_SAMPLE_NAME):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_SAMPLE_NAME))

    sample_id_errors = validate_sample_id(sampleId)
    if sample_id_errors:
        errors.extend(sample_id_errors)

    sample_nucleotideType = ""
    
    nucleotideType_errors, sample_nucleotideType = validate_sample_nucleotideType(nucleotideType, runTypeName)
    if (nucleotideType_errors):
        errors.extend(nucleotideType_errors)

    ref_errors, ref_short_name = validate_reference(sampleReference, displayedName = "Sample Reference")
    ##logger.debug("plan_validator.validate_barcoded_sample_info() sampleReference=%s; ref_short_name=%s" %(sampleReference, ref_short_name))

    if (ref_errors):
        errors.extend(ref_errors)

    ##logger.debug("plan_validator.validate_barcoded_sample_info() errors=%s" %(errors))
    
    return errors, ref_short_name, sample_nucleotideType


def validate_sample_nucleotideType(nucleotideType, runType, displayedName='Sample Nucleotide Type'):
    """
    validate nucleotide type case-insensitively with leading/trailing blanks in the input ignored
    """    
    
    errors = []
    input = ""
    valid_values = VALID_NUCLEOTIDE_TYPES
    
    if nucleotideType:
        input = nucleotideType.strip().upper()
        
        if runType:
            runTypeObjs = RunType.objects.filter(runType = runType)
            if runTypeObjs:
                runTypeObj = runTypeObjs[0]

                if (runTypeObj.nucleotideType and runTypeObj.nucleotideType.upper() != "DNA_RNA"):
                    valid_values = [str(runTypeObj.nucleotideType.upper())]
                   
        if not validation.is_valid_keyword(input, valid_values):
            errors.append(validation.invalid_keyword_error(displayedName, valid_values))
                
    return errors, input


def validate_reference(referenceName, displayedName='Reference'):

    errors = []
    ref_short_name = ""
    
    if referenceName:
        input = referenceName.strip()

        selectedRefs = ReferenceGenome.objects.filter(name = input)
            
        if selectedRefs:
            ref_short_name = selectedRefs[0].short_name 
        else:
            selectedRefs = ReferenceGenome.objects.filter(name__iexact = input)
            if selectedRefs:
                ref_short_name = selectedRefs[0].short_name 
            else:
                selectedRefs = ReferenceGenome.objects.filter(short_name = input)
            
                if selectedRefs:
                    ref_short_name = selectedRefs[0].short_name 
                else:
                    selectedRefs = ReferenceGenome.objects.filter(short_name__iexact = input)
                    if selectedRefs:
                        ref_short_name = selectedRefs[0].short_name 
                    else:
                         errors.append(validation.invalid_not_found_error(displayedName, referenceName))
      
    return errors, ref_short_name


def validate_sample_id(value, displayedName='Sample Id'):
    errors = []    
    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))
    
    if not validation.is_valid_length(value, MAX_LENGTH_SAMPLE_ID):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_SAMPLE_ID))

    return errors


def validate_sample_tube_label(value, displayedName='Sample Tube Label'):    
    errors = []
    
    if value:
        if not validation.is_valid_chars(value):
            errors.append(validation.invalid_chars_error(displayedName))
            
        if not validation.is_valid_length(value, MAX_LENGTH_SAMPLE_TUBE_LABEL):
            errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_SAMPLE_TUBE_LABEL))

    return errors

def validate_chip_type(value, displayedName='Chip Type'):
    errors = []
    if not value:
        errors.append(validation.required_error(displayedName))
    else:
        value = value.strip()
        chip = Chip.objects.filter(name = value)
        if not chip:
            chip = Chip.objects.filter(description = value)
        
        if not chip:
            errors.append('Chip %s not found' % value)
        
    return errors

def validate_flows(value, displayedName='Flows'):
    errors = []
    if not validation.is_valid_uint(value):
        errors.append(validation.invalid_uint(displayedName))
    elif int(value) > MAX_FLOWS:
        errors.append(displayedName + ' must be a positive integer within range [1, 2000)' )
    
    return errors

    
def validate_QC(value, displayedName):
    errors = []
    if not validation.is_valid_uint(value):
        errors.append(validation.invalid_uint(displayedName))
    elif int(value) > MAX_QC_INT:
        errors.append(displayedName + ' must be a positive whole number within range [1, 100)')
    
    return errors
    
def validate_projects(value, displayedName='Project Name', delim=','):
    """
    validate projects case-insensitively with leading/trailing blanks in the input ignored
    """            
    
    errors = []
    trimmed_projects = ""
    if value:
        for project in value.split(delim):
            trimmed_project = project.strip()
            if trimmed_project:
                trimmed_projects = trimmed_projects + trimmed_project + delim
                
                if not validation.is_valid_chars(trimmed_project):
                    errors.append(validation.invalid_chars_error(displayedName))
                if not validation.is_valid_length(trimmed_project, PROJECT_NAME_LENGTH):
                    errors.append(validation.invalid_length_error(displayedName, PROJECT_NAME_LENGTH))
                if errors:
                    break

    return errors, trimmed_projects


def validate_barcode_kit_name(value, displayedName="Barcode Kit"):
    errors = []
    
    if validation.has_value(value):
        value = value.strip()
        kits = dnaBarcode.objects.filter(name = value)
        if not kits:
            errors.append("%s %s not found" %(displayedName, value))
            
    return errors


def validate_sequencing_kit_name(value, displayedName="Sequencing Kit"):
    errors = []
    
    if validation.has_value(value):
        value = value.strip()
        kit = KitInfo.objects.filter(kitType__in = ["SequencingKit"], name = value)
        if not kit:
            kit = KitInfo.objects.filter(kitType__in = ["SequencingKit"], description = value)
        
        if not kit:
            errors.append("%s %s not found" %(displayedName, value))
            
    return errors


def validate_plan_templating_kit_name(value, displayedName="Template Kit"):
    errors = []
    
    if not validation.has_value(value):
        errors.append(validation.required_error(displayedName))
    else:
        value = value.strip()
        kit = KitInfo.objects.filter(kitType__in = ["TemplatingKit", "IonChefPrepKit"], name = value)
        if not kit:
            kit = KitInfo.objects.filter(kitType__in = ["TemplatingKit", "IonChefPrepKit"], description = value)
        
        if not kit:
            errors.append("%s %s not found" %(displayedName, value))
            
    return errors


def validate_application_group_for_runType(value, runType, displayedName="Application Group"):
    errors = []
    
    if value:
        value = value.strip()
        applicationGroups = ApplicationGroup.objects.filter(name__iexact = value)
        if applicationGroups:
            applicationGroup = applicationGroups[0]
            if runType:
                runTypeObjs = RunType.objects.filter(runType__iexact = runType)
                if runTypeObjs:
                    associations = runTypeObjs[0].applicationGroups.filter(name__iexact = value)
                    if not associations:
                        errors.append("%s %s not valid for Run Type %s" %(displayedName, value, runType))
        else:
            errors.append("%s %s not found" %(displayedName, value))
        
    return errors
        

def validate_sample_grouping(value, displayedName="Sample Grouping"):
    errors = []
    
    if value:
        value = value.strip()
    
        groupings = SampleGroupType_CV.objects.filter(displayedName__iexact = value)
        if not groupings:
           errors.append("%s %s not found" %(displayedName, value))            
            
    return errors


def validate_barcode_sample_association(selectedBarcodes, selectedBarcodeKit):
    errors = {"MISSING_BARCODE" : "", "DUPLICATE_BARCODE" : ""}
    
    if not selectedBarcodeKit:
        return errors
    
    prior_barcodes = []
    
    if not selectedBarcodes:
        errors["MISSING_BARCODE"] = "Please select a barcode for each sample"
    else:
        for selectedBarcode in selectedBarcodes:
            if selectedBarcode in prior_barcodes:
                #only include unique barcode selection error messages
                message = "Barcode %s selections have to be unique\n" % selectedBarcode
                
                value = errors["DUPLICATE_BARCODE"]
                if message not in value:
                   errors["DUPLICATE_BARCODE"] = errors["DUPLICATE_BARCODE"] + message
            else:
                prior_barcodes.append(selectedBarcode)

    #logger.debug("errors=%s" %(errors))
    
    return errors

        
        
    
