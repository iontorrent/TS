# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb import models

import datetime
from django.utils import timezone
import logging

from django.db import transaction

from iondb.rundb import models

from iondb.rundb.models import Sample, SampleSet, SampleSetItem, SampleAttribute, SampleGroupType_CV,  \
    SampleAttributeDataType, SampleAttributeValue

from django.contrib.auth.models import User

import sample_validator

logger = logging.getLogger(__name__)

COLUMN_SAMPLE_EXT_ID = "Sample ID"
COLUMN_SAMPLE_NAME = "Sample Name (required)"
COLUMN_GENDER = "Gender"
COLUMN_GROUP_TYPE = "Type"
COLUMN_GROUP = "Group"
COLUMN_SAMPLE_DESCRIPTION = "Description"
COLUMN_BARCODE = "Barcode"


def process_csv_sampleSet(csvSampleDict, request, user, sampleSet_ids):
    """ read csv contents and convert data to raw data to prepare for sample persistence
    returns: a collection of error messages if errors found, a dictionary of raw data values
    """
    
    logger.debug("ENTER import_sample_processor.process_csv_sampleSet() csvSampleDict=%s; " %(csvSampleDict))
    failed = []
    isToSkipRow = False
    
    #check if mandatory fields are present
    requiredList = [COLUMN_SAMPLE_NAME]
    
    for required in requiredList:
        if required in csvSampleDict:
            if not csvSampleDict[required]:
                failed.append((required, "Required column is empty"))
        else:
            failed.append((required, "Required column is missing"))

    sample, sampleSetItem, ssi_sid = _create_sampleSetItem(csvSampleDict, request, user, sampleSet_ids)
    siv_sid = _create_sampleAttributeValue(csvSampleDict, request, user, sample)
    
    return failed, sample, sampleSetItem, isToSkipRow, ssi_sid, siv_sid


# sampleSets should have been validated - there must be at least 1 sample set
def _create_sampleSetItem(csvSampleDict, request, user, sampleSet_ids):
    sampleDisplayedName = csvSampleDict.get(COLUMN_SAMPLE_NAME, '').strip()
    sampleExtId = csvSampleDict.get(COLUMN_SAMPLE_EXT_ID, '').strip()
    sampleGender = csvSampleDict.get(COLUMN_GENDER, '').strip()
    sampleGroupType = csvSampleDict.get(COLUMN_GROUP_TYPE, None)
    sampleGroup = csvSampleDict.get(COLUMN_GROUP, '0').strip()
    sampleDescription = csvSampleDict.get(COLUMN_SAMPLE_DESCRIPTION, '').strip()
    barcodeAssignment = csvSampleDict.get(COLUMN_BARCODE, '').strip()
    
    if not sampleGroup:
        sampleGroup = '0'

    #validation has been done already, this is just to get the official value
    isValid, errorMessage, gender_CV_value = sample_validator.validate_sampleGender(sampleGender)      
    isValid, errorMessage, role_CV_value = sample_validator.validate_sampleGroupType(sampleGroupType)

    
    sampleName = sampleDisplayedName.replace(' ', '_')
    sample_kwargs = {
                     'displayedName' : sampleDisplayedName,
                     'status' : 'created',
                     'description' : sampleDescription,
                     'date' : timezone.now()  ##datetime.datetime.now() 
                     }
    
    sample, isCreated = Sample.objects.get_or_create(name = sampleName, externalId = sampleExtId, defaults = sample_kwargs)

    if isCreated:
        logger.debug("import_sample_processor._create_sampleSetItem() new sample created for sample=%s; id=%d" %(sampleDisplayedName, sample.id))
    else:
        if (sample.description != sampleDescription):
            sample.description = sampleDescription
            sample.save()
            
            logger.debug("import_sample_processor._create_sampleSetItem() just updated sample description for sample=%s; id=%d" %(sampleDisplayedName, sample.id))
        
    logger.debug("import_sample_processor._create_sampleSetItem() after get_or_create isCreated=%s; sample=%s; sample.id=%d" %(str(isCreated), sampleDisplayedName, sample.id))


    for sampleSetId in sampleSet_ids:
        
        logger.debug("import_sample_processor._create_sampleSetItem() going to create sampleSetItem for sample=%s; sampleSetId=%s in sampleSet_ids=%s" %(sampleDisplayedName, str(sampleSetId), sampleSet_ids))
        
        currentDateTime = timezone.now()  ##datetime.datetime.now() 
        
        sampleSetItem_kwargs = {
                                 'gender' : gender_CV_value, 
                                 'relationshipRole' : role_CV_value, 
                                 'relationshipGroup' : sampleGroup, 
                                 'creator' : user,
                                 'creationDate' : currentDateTime,
                                 'lastModifiedUser' : user,
                                 'lastModifiedDate' : currentDateTime,
                                 'barcode' : barcodeAssignment                                
                             }
    
        sampleSetItem, isCreated = SampleSetItem.objects.get_or_create(sample = sample, 
                                                                       sampleSet_id = sampleSetId, 
                                                                       defaults = sampleSetItem_kwargs)

        logger.debug("import_sample_processor._create_sampleSetItem() after get_or_create isCreated=%s; sampleSetItem=%s; samplesetItem.id=%d" %(str(isCreated), sampleDisplayedName, sampleSetItem.id))            
            
    ssi_sid = transaction.savepoint()
    
    return sample, sampleSetItem, ssi_sid


def _create_sampleAttributeValue(csvSampleDict, request, user, sample):
    """
    save sample customer attribute value to db.

    """
    customAttributes = SampleAttribute.objects.filter(isActive = True)
    currentDateTime = timezone.now()  ##datetime.datetime.now() 
            
    for attribute in customAttributes:
        newValue = None
        
        if attribute.displayedName not in csvSampleDict.keys():
    
            #add mandatory custom attributes for an imported sample if user has not added it            
            if attribute.isMandatory:                    
                if attribute.dataType and attribute.dataType.dataType == "Integer":
                    newValue = "0"
                else:
                    newValue = ""
        else:
            newValue = csvSampleDict.get(attribute.displayedName, "")
                   
        if newValue is None:       
            logger.debug("import_sample_processor._create_sampleAttributeValue SKIPPING due to NO VALUE for attribute=%s;" %(attribute.displayedName))            
        else:
            logger.debug("import_sample_processor._create_sampleAttributeValue going to get_or_create sample=%s; attribute=%s; value=%s" %(sample.displayedName, attribute.displayedName, newValue))
        
            sampleAttributeValues = SampleAttributeValue.objects.filter(sample = sample, sampleAttribute = attribute)
        
            if sampleAttributeValues: 
                sampleAttributeValue = sampleAttributeValues[0]   
                               
                #logger.debug("import_sample_processor._create_sampleAttributeValue ORIGINAL VALUE pk=%s; sample=%s; attribute=%s; orig value=%s" %(sampleAttributeValue.id, sample.displayedName, attribute.displayedName, sampleAttributeValue.value))

                #there should only be 1 attribute value for each sample/attribute pair if the old entry has value but the new import doesn't, do not override it.
                if newValue:
                    sampleAttributeValue_kwargs = {
                                                   'value' : newValue,
                                                   'lastModifiedUser' : user,                     
                                                   'lastModifiedDate' : currentDateTime                  
                                                   }
        
                    for field, value in sampleAttributeValue_kwargs.iteritems():
                        setattr(sampleAttributeValue, field, value)
                    
                    sampleAttributeValue.save()
                                        
                    #logger.debug("import_sample_processor._create_sampleAttributeValue UPDATED pk=%s; sample=%s; attribute=%s; newValue=%s" %(sampleAttributeValue.id, sample.displayedName, attribute.displayedName, newValue))

                else:
                    #logger.debug("import_sample_processor._create_sampleAttributeValue going to DELETE pk=%s; sample=%s; attribute=%s; newValue=%s" %(sampleAttributeValue.id, sample.displayedName, attribute.displayedName, newValue))

                    sampleAttributeValue.delete()
            else:
                #create a record only there is a value
                if newValue:
                    sampleAttributeValue_kwargs = {
                                                   'sample' : sample,
                                                   'sampleAttribute' : attribute,
                                                   'value' : newValue,
                                                   'creator' : user,                
                                                   'creationDate' : currentDateTime,
                                                   'lastModifiedUser' : user,                     
                                                   'lastModifiedDate' : currentDateTime                  
                                                   }

                    sampleAttributeValue = SampleAttributeValue(**sampleAttributeValue_kwargs)
                    sampleAttributeValue.save()
             
                    logger.debug("import_sample_processor._create_sampleAttributeValue CREATED sampleAttributeValue.pk=%d; sample=%s; attribute=%s; newValue=%s" %(sampleAttributeValue.pk, sample.displayedName, attribute.displayedName, newValue))               
    siv_sid = transaction.savepoint()
    return siv_sid


def validate_csv_sample(csvSampleDict, request):
    """ 
    validate csv contents and convert user input to raw data to prepare for sample persistence
    returns: a collection of error messages if errors found and whether to skip the row
    """
    failed = []
    isToSkipRow = False

    logger.debug("ENTER import_sample_processor.validate_csv_sample() csvSampleDict=%s; " %(csvSampleDict))
        
    sampleDisplayedName = csvSampleDict.get(COLUMN_SAMPLE_NAME, '').strip()
    sampleExtId = csvSampleDict.get(COLUMN_SAMPLE_EXT_ID, '').strip()
    sampleGender = csvSampleDict.get(COLUMN_GENDER, '').strip()
    sampleGroupType = csvSampleDict.get(COLUMN_GROUP_TYPE, '').strip()
    sampleGroup = csvSampleDict.get(COLUMN_GROUP, '').strip()
    sampleDescription = csvSampleDict.get(COLUMN_SAMPLE_DESCRIPTION, '').strip()
    barcodeAssignment = csvSampleDict.get(COLUMN_BARCODE, '').strip()
        
    #skip blank line
    hasAtLeastOneValue = bool([v for v in csvSampleDict.values() if v != ''])
    if not hasAtLeastOneValue:
        isToSkipRow = True
        return failed, isToSkipRow
    
    isValid, errorMessage = sample_validator.validate_sampleDisplayedName(sampleDisplayedName)
    if not isValid:
        failed.append((COLUMN_SAMPLE_NAME, errorMessage))    
        
    isValid, errorMessage = sample_validator.validate_sampleExternalId(sampleExtId)
    if not isValid:
        failed.append((COLUMN_SAMPLE_EXT_ID, errorMessage))
        
    isValid, errorMessage = sample_validator.validate_sampleDescription(sampleDescription)
    if not isValid:
        failed.append((COLUMN_SAMPLE_DESCRIPTION, errorMessage))
        
    isValid, errorMessage, gender_CV_value = sample_validator.validate_sampleGender(sampleGender)
    if not isValid:
        failed.append((COLUMN_GENDER, errorMessage))
       
    isValid, errorMessage, role_CV_value = sample_validator.validate_sampleGroupType(sampleGroupType)
    if not isValid:
        failed.append((COLUMN_GROUP_TYPE, errorMessage))
        
    if sampleGroup: 
        isValid, errorMessage = sample_validator.validate_sampleGroup(sampleGroup)
        if not isValid:
            failed.append((COLUMN_GROUP, errorMessage))
            
    if not isValid:
        failed.append((COLUMN_BARCODE, errorMessage))

    #validate user-defined custom attributes
    failed_userDefined = _validate_csv_user_defined_attributes(csvSampleDict, request)
    failed.extend(failed_userDefined)
               
    logger.debug("import_sample_processor.validate_csv_sample() failed=%s" %(failed))
        
    return failed, isToSkipRow



def _validate_csv_user_defined_attributes(csvSampleDict, request):
    failed = []
        
    customAttributes = SampleAttribute.objects.filter(isActive = True)
            
    for attribute in customAttributes:
        newValue = None
                     
        if attribute.displayedName not in csvSampleDict.keys():
    
            #add mandatory custom attributes for an imported sample if user has not added it            
            if attribute.isMandatory:  
                failed.append((attribute.displayedName, "Error, " + attribute.displayedName + " is required."))                  
        else:
            newValue = csvSampleDict.get(attribute.displayedName, "").strip()
                   
        if newValue:
            if attribute.dataType and attribute.dataType.dataType == "Integer":
                isValid, errorMessage = sample_validator._validate_intValue(newValue, attribute.displayedName)
                if not isValid:
                    failed.append((attribute.displayedName, errorMessage))
               
    logger.debug("import_sample_processor._validate_csv_user_defined_attributes() failed=%s" %(failed))
        
    return failed


def validate_barcodes(row_list):
    barcodekits = []
    barcodes = []
    id_strings = []
    msgs = dict()
    count = 0
    for row in row_list:
        count += 1
        barcode = row.get(COLUMN_BARCODE, "")
        if barcode.strip():
            dnabarcode = models.dnaBarcode.objects.filter(id_str__iexact=barcode.strip())
            if len(dnabarcode) > 0:
                id_strings.append(barcode)
            else:
                msgs[count] = ((COLUMN_BARCODE, "Error, %s is an invalid barcode" % (barcode)))

            if barcode in barcode in barcodes:
                msgs[count] = ((COLUMN_BARCODE, "Error, A barcode can be assigned to only one sample in the sample set and %s has been assigned to another sample" %(barcode)))
            else:
                barcodes.append(barcode)

    count = 0
    for id_str in id_strings:
        if len(barcodekits) > 0:
            d = models.dnaBarcode.objects.filter(id_str__iexact=id_str, name__in=[x.name for x in barcodekits])
        else:
            d = models.dnaBarcode.objects.filter(id_str__iexact=id_str)
        if len(d) > 0:
            barcodekits.extend(list(d))
        else:
            msgs[count] = ((COLUMN_BARCODE, "Error, Only one barcode kit can be used for a sample set"))
    return msgs

def validate_barcodes_for_existing_samples(row_list, sampleset_ids):
    msgs = dict()
    count = 0
    for row in row_list:
        count += 1
        barcode = row.get(COLUMN_BARCODE,"")
        barcode = barcode.strip()
        
        if barcode: 
            dnabarcode = models.dnaBarcode.objects.filter(id_str__iexact=barcode)
            barcodeKit = dnabarcode[0].name
            row[COLUMN_BARCODE] = dnabarcode[0].id_str
        else:
            barcode = None
            barcodeKit = None

        for sampleset_id in sampleset_ids:
        
            sampleset = models.SampleSet.objects.get(pk=sampleset_id)
            samplesetitems = sampleset.samples.all()
        
            for item in samplesetitems:
                #first validate that all barcode kits are the same for all samples
                if item.barcode:
        
                    dnabarcodes = models.dnaBarcode.objects.filter(id_str=item.barcode)
                    barcodeKit1 = dnabarcodes[0].name
                    barcode1 = item.barcode
                else:
                    barcodeKit1 = None
                    barcode1 = None

                if barcodeKit and barcodeKit1 and barcodeKit != barcodeKit1:
                    msgs[count] = ((COLUMN_BARCODE, "Error, Only one barcode kit can be used for a sample set"))
                    
                #next validate that all barcodes are unique per each sample
                if barcode and barcode1 and barcode == barcode1:
                    msgs[count] = ((COLUMN_BARCODE, "Error, A barcode can be assigned to only one sample in the sample set and %s has been assigned to another sample" %(barcode)))

    return msgs