# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from django.template import RequestContext
from django.shortcuts import get_object_or_404
    
import datetime
from django.utils import timezone
import logging

from traceback import format_exc
import json
import simplejson


from iondb.rundb import models

from iondb.rundb.models import Sample, SampleSet, SampleSetItem, SampleAttribute, SampleGroupType_CV,  \
    SampleAttributeDataType, SampleAttributeValue

from django.contrib.auth.models import User

import sample_validator

logger = logging.getLogger(__name__)



def validate_for_existing_samples(request, sampleSet_ids):
    if 'input_samples' in request.session:
        if 'pending_sampleSetItem_list' in request.session['input_samples']:
            pending_sampleSetItem_list = request.session['input_samples']['pending_sampleSetItem_list']
        else:
            pending_sampleSetItem_list = []
    else:
        pending_sampleSetItem_list = []

    for samplesetitem_id in sampleSet_ids:
        sampleset = models.SampleSet.objects.get(pk=samplesetitem_id)
        samplesetitems = sampleset.samples.all()
        for item in samplesetitems:
            #first validate that all barcode kits are the same for all samples
            if item.barcode:
                dnabarcode = models.dnaBarcode.objects.filter(id_str=item.barcode)
                barcodeKit1 = dnabarcode[0].name
                barcode1 = item.barcode
            else:
                barcodeKit1 = None
                barcode1 = None

            for item1 in pending_sampleSetItem_list:
                barcodeKit = item1.get('barcodeKit')
                barcode = item1.get('barcode')
                if barcodeKit and barcodeKit1 and barcodeKit != barcodeKit1:
                    return False, "Error, Only one barcode kit can be used for a sample set"
                    
                #next validate that all barcodes are unique per each sample
                if barcode and barcode1 and barcode == barcode1:
                    return False, "Error, A barcode can be assigned to only one sample in the sample set and %s has been assigned to another sample" %(barcode)
    return True, None

def _get_or_create_sampleSets(request, user):                
    queryDict = request.POST
    logger.info("views._get_or_create_sampleSets POST queryDict=%s" %(queryDict))

    sampleSet_ids = []

    new_sampleSetName = queryDict.get("new_sampleSetName", "").strip()
    new_sampleSetDesc = queryDict.get("new_sampleSetDescription", "").strip()
    new_sampleSet_groupType_id = queryDict.get("new_sampleSet_groupType", None)
    selected_sampleSet_ids = queryDict.getlist("sampleset", [])

    
    
    #logger.debug("views_helper._get_or_create_sampleSets selected_sampleSet_ids=%s" %(selected_sampleSet_ids))
    
    if selected_sampleSet_ids:
        selected_sampleSet_ids = [ssi.encode('utf8') for ssi in selected_sampleSet_ids]

    if (selected_sampleSet_ids):
        sampleSet_ids.extend(selected_sampleSet_ids)

    #logger.debug("views_helper._get_or_create_sampleSets sampleSet_ids=%s" %(sampleSet_ids))

    #nullify group type if user does not specify a group type
    if new_sampleSet_groupType_id == '0' or new_sampleSet_groupType_id == 0:
        new_sampleSet_groupType_id = None
        
    #if new_sampleSetName is missing, the rest of the input will be ignored
    if new_sampleSetName:
        isValid, errorMessage = sample_validator.validate_sampleSet_values(new_sampleSetName, new_sampleSetDesc, True)
        
        if errorMessage:
            return isValid, errorMessage,sampleSet_ids
        
        currentDateTime = timezone.now()  ##datetime.datetime.now() 
        
        sampleSet_kwargs = {
                            'description' : new_sampleSetDesc,  
                            'status' : "created",                 
                            'creationDate' : currentDateTime,
                            'lastModifiedUser' : user,                     
                            'lastModifiedDate' : currentDateTime                  
                            }

        sampleSet, isCreated = SampleSet.objects.get_or_create(
                                displayedName = new_sampleSetName.strip(), 
                                SampleGroupType_CV_id = new_sampleSet_groupType_id, 
                                creator = user, 
                                defaults = sampleSet_kwargs)
        
        #logger.debug("views_helper._get_or_create_sampleSets sampleSetName=%s isCreated=%s" %(new_sampleSetName, str(isCreated)))
       
        sampleSet_ids.append(sampleSet.id)

    logger.debug("views_helper._get_or_create_sampleSets EXIT sampleSet_ids=%s" %(sampleSet_ids))
        
    return True, None, sampleSet_ids
        


def _create_or_update_sample_for_sampleSetItem(request, user):                                        
    queryDict = request.POST
                
    sampleSetItem_id = queryDict.get("id", None)
    sampleDisplayedName = queryDict.get("sampleName", "").strip()
    sampleExternalId = queryDict.get("sampleExternalId", "").strip()
    sampleDesc = queryDict.get("sampleDescription", "").strip()
    #20130930-TODO
    #barcode info
    barcodeKit = queryDict.get("barcodeKit", "").strip()
    barcode = queryDict.get("barcode", "").strip()
          
    return _create_or_update_sample_for_sampleSetItem_with_id_values(request, user, sampleSetItem_id, sampleDisplayedName, sampleExternalId, sampleDesc, barcodeKit, barcode)



def _create_or_update_sample_for_sampleSetItem_with_id_values(request, user, sampleSetItem_id, sampleDisplayedName, sampleExternalId, sampleDesc, barcodeKit, barcode):
    currentDateTime = timezone.now()  ##datetime.datetime.now()        
           
    orig_sampleSetItem = get_object_or_404(SampleSetItem, pk = sampleSetItem_id)
    orig_sample = orig_sampleSetItem.sample

                                                
    if (orig_sample.displayedName == sampleDisplayedName and 
        orig_sample.externalId == sampleExternalId.strip()):
        #logger.debug("views_helper - _create_or_update_sample_for_sampleSetItem_with_id_values - #1 can REUSE SAMPLE for sure!! sampleSetItem.id=%d; sample.id=%d" %(orig_sampleSetItem.id, orig_sample.id))
       
        new_sample = orig_sample
    
        if (new_sample.description != sampleDesc):
            new_sample.description = sampleDesc
            new_sample.date = currentDateTime
            
            #logger.debug("views_helper - _create_or_update_sample_for_sampleSetItem_with_id_values - #2 REUSE SAMPLE + UPDATE DESC sampleSetItem.id=%d; sample.id=%d" %(orig_sampleSetItem.id, new_sample.id))            
            new_sample.save()        
    else:
        # link the renamed sample to an existing sample if one is found. Otherwise, rename the sample only if the sample has not yet been planned.
        existingSamples = Sample.objects.filter(displayedName = sampleDisplayedName, externalId = sampleExternalId)

        canSampleBecomeOrphan = (orig_sample.sampleSets.count() < 2) and orig_sample.experiments.count() == 0
        logger.info("views_helper - _create_or_update_sample_for_sampleSetItem_with_id_values - #3 can sample becomes ORPHAN? orig_sample.id=%d; orig_sample.name=%s; canSampleBecomeOrphan=%s" %(orig_sample.id, orig_sample.displayedName, str(canSampleBecomeOrphan)))
        
        if existingSamples.count() > 0:
            orig_sample_id = orig_sample.id
    
            #by sample uniqueness rule, there should only be 1 existing sample max
            existingSample = existingSamples[0]
            existingSample.description = sampleDesc
            orig_sampleSetItem.sample = existingSample
            orig_sampleSetItem.lastModifiedUser = user
            orig_sampleSetItem.lastModifiedDate = currentDateTime

            if barcode:
                orig_sampleSetItem.barcode = barcode
                
            orig_sampleSetItem.save()
                            
            new_sample = existingSample
                        
            logger.debug("views_helper - _create_or_update_sample_for_sampleSetItem_with_id_values - #4 SWITCH TO EXISTING SAMPLE sampleSetItem.id=%d; existingSample.id=%d" %(orig_sampleSetItem.id, existingSample.id))

            #cleanup if the replaced sample is not being used anywhere
            if canSampleBecomeOrphan:
                logger.debug("views_helper - _create_or_update_sample_for_sampleSetItem_with_id_values - #5 AFTER SWITCH orig_sample becomes ORPHAN! orig_sample.id=%d; orig_sample.name=%s" %(orig_sample.id, orig_sample.displayedName))
                orig_sample.delete()
            else:                        
                logger.debug("views_helper - _create_or_update_sample_for_sampleSetItem_with_id_values - #6 AFTER SWITCH orig_sample is still NOT ORPHAN YET! orig_sample.id=%d; orig_sample.name=%s sample.sampleSets.count=%d; sample.experiments.count=%d; " %(orig_sample.id, orig_sample.displayedName, orig_sample.sampleSets.count(), orig_sample.experiments.count()))                                                             
        else:
            name = sampleDisplayedName.replace(' ', '_')

            if canSampleBecomeOrphan:
                #update existing sample record
                sample_kwargs = {
                            'name' : name,
                            'displayedName' : sampleDisplayedName,
                            'description': sampleDesc,                                 
                            'externalId': sampleExternalId,
                            'description': sampleDesc,
                            'date' : currentDateTime,
                            }                
                for field, value in sample_kwargs.iteritems():
                    setattr(orig_sample, field, value)

                orig_sample.save()
                            
                logger.debug("views_helper - _create_or_update_sample_for_sampleSetItem_with_id_values - #7 RENAME SAMPLE sampleSetItem.id=%d; sample.id=%d" %(orig_sampleSetItem.id, orig_sample.id))
                new_sample = orig_sample
            else:
                #create a new sample record
                sample_kwargs = {
                                 'name' : name,
                                 'displayedName' : sampleDisplayedName,
                                 'externalId': sampleExternalId,
                                 'description': sampleDesc,
                                 'status' : "created",
                                 'date' : currentDateTime,
                                 }

                sample = Sample.objects.get_or_create(displayedName = sampleDisplayedName, externalId=sampleExternalId, defaults=sample_kwargs)[0]

                orig_sampleSetItem.sample = sample

                if barcode:
                    orig_sampleSetItem.barcode = barcode
                
                orig_sampleSetItem.save()
                            
                logger.debug("views_helper - _create_or_update_sample_for_sampleSetItem_with_id_values - #8 CREATE NEW SAMPLE sampleSetItem.id=%d; sample.id=%d" %(orig_sampleSetItem.id, sample.id))                            
                new_sample = sample                    

    return new_sample                    



def _create_or_update_sample_for_sampleSetItem_with_values(request, user, sampleDisplayedName, sampleExternalId, sampleDesc, barcodeKit, barcode):
    currentDateTime = timezone.now()  ##datetime.datetime.now()        
        
    sample = None
    existingSamples = Sample.objects.filter(displayedName = sampleDisplayedName, externalId = sampleExternalId)
        
    if existingSamples.count() > 0:
        existingSample = existingSamples[0]
        existingSample.description = sampleDesc
        existingSample.date = currentDateTime
                        
        logger.debug("views_helper._create_or_update_sample_for_sampleSetItem_with_values() #9 updating sample.id=%d; name=%s" %(existingSample.id, existingSample.displayedName))

        existingSample.save()
        sample = existingSample
    else:
        #create a new sample record
        name = sampleDisplayedName.replace(' ', '_')

        sample_kwargs = {
                         'name' : name,
                         'displayedName' : sampleDisplayedName,
                         'externalId': sampleExternalId,
                         'description': sampleDesc,
                         'status' : "created",
                         'date' : currentDateTime,
                         }

        sample = Sample.objects.get_or_create(displayedName = sampleDisplayedName, externalId=sampleExternalId, defaults=sample_kwargs)[0]
                        
        logger.debug("views_helper._create_or_update_sample_for_sampleSetItem_with_values() #10 create new sample.id=%d; name=%s" %(sample.id, sample.displayedName))

        
    return sample


def _update_input_samples_session_context(request, pending_sampleSetItem, isNew = True):
    logger.debug("views_helper._update_input_samples_session_context pending_sampleSetItem=%s" %(pending_sampleSetItem))

    _create_pending_session_if_needed(request)
    
    if isNew:
        request.session["input_samples"]["pending_sampleSetItem_list"].insert(0, pending_sampleSetItem)
    else:
        pendingList = request.session["input_samples"]["pending_sampleSetItem_list"]
        hasUpdated = False
        
        for index, item in enumerate(pendingList):
            #logger.debug("views_helper._update_input_samples_session_context - item[pending_id]=%s; updated item.pending_id=" %(str(item['pending_id']), str(pending_sampleSetItem['pending_id'])))
            
            if item['pending_id'] == pending_sampleSetItem['pending_id'] and not hasUpdated:
                pendingList[index] = pending_sampleSetItem
                hasUpdated = True
                
        if not hasUpdated:
            request.session["input_samples"]["pending_sampleSetItem_list"].insert(0, pending_sampleSetItem)
        
    request.session.modified = True

    logger.debug("views_helper._update_input_samples_session_context AFTER UPDATE!! session_contents=%s" %(request.session["input_samples"]))
    


def _create_pending_session_if_needed(request):
    """
    return or create a session context for entering samples manually to create a sample set
    """
    if "input_samples" not in request.session:    

        logger.debug("views_helper._create_pending_session_if_needed() going to CREATE new request.session")
            
        sampleSet_list = SampleSet.objects.all().order_by("-lastModifiedDate", "displayedName")
        sampleGroupType_list = list(SampleGroupType_CV.objects.filter(isActive=True).order_by("displayedName")) 
#        custom_sample_column_list = list(SampleAttribute.objects.filter(isActive = True).values_list('displayedName', flat=True).order_by('id'))
    
        pending_sampleSetItem_list = []
        
        request.session["input_samples"] = {}
        request.session["input_samples"]['pending_sampleSetItem_list'] = pending_sampleSetItem_list
        
    else:
        logger.debug("views_helper._create_pending_session_if_needed() ALREADY EXIST request.session[input_samples]=%s" %(request.session["input_samples"]))



def _handle_enter_samples_manually_request(request):
    _create_pending_session_if_needed(request)

    ctxd = _create_context_from_session(request)
    
    return ctxd



def _create_context_from_session(request):    
#    ctxd = request.session['input_samples'],

    custom_sample_column_list = list(SampleAttribute.objects.filter(isActive = True).values_list('displayedName', flat=True).order_by('id'))
        
    ctx = {
            'input_samples' : request.session.get('input_samples', {}),
           'custom_sample_column_list' : simplejson.dumps(custom_sample_column_list),
           }

    context = RequestContext(request, ctx)
    
    return context
        
       

def _create_pending_sampleSetItem_dict(request, userName, creationTimeStamp):
    currentDateTime = timezone.now()  ##datetime.datetime.now()        
                        
    queryDict = request.POST
                
    sampleSetItem_id = queryDict.get("id", None)
    sampleDisplayedName = queryDict.get("sampleName", "").strip()
    sampleExternalId = queryDict.get("sampleExternalId", "").strip()
    sampleDesc = queryDict.get("sampleDescription", "").strip()

    barcodeKit = queryDict.get("barcodeKit", None)        
    barcode = queryDict.get("barcode", None)    
    gender = queryDict.get("gender", "")
    relationshipRole = queryDict.get("relationshipRole", "")
    relationshipGroup = queryDict.get("relationshipGroup", None)
    
    isValid, errorMessage, sampleAttributes_dict = _create_pending_sampleAttributes_for_sampleSetItem(request)
    
    if errorMessage:
        return isValid, errorMessage, sampleAttributes_dict
    
    #create a sample object without saving it to db
    name = sampleDisplayedName.replace(' ', '_')
    
    sampleSetItem_dict = {}
    sampleSetItem_dict['pending_id'] = _get_pending_sampleSetItem_id(request)
    sampleSetItem_dict['name'] = name
    sampleSetItem_dict['displayedName'] = sampleDisplayedName
    sampleSetItem_dict['externalId'] = sampleExternalId
    sampleSetItem_dict['description'] = sampleDesc

    sampleSetItem_dict['barcodeKit'] = barcodeKit
    sampleSetItem_dict['barcode'] = barcode
    sampleSetItem_dict['status'] = "created"

    sampleSetItem_dict['gender'] = gender
    sampleSetItem_dict['relationshipRole'] = relationshipRole
    sampleSetItem_dict['relationshipGroup'] = relationshipGroup
    sampleSetItem_dict['attribute_dict'] = sampleAttributes_dict
    
    #logger.debug("views_helper._create_pending_sampleSetItem_dict=%s" %(sampleSetItem_dict))
    
    return isValid, errorMessage, sampleSetItem_dict

 
def _update_pending_sampleSetItem_dict(request, userName, creationTimeStamp):
    currentDateTime = timezone.now()  ##datetime.datetime.now()        
                        
    queryDict = request.POST
                
    sampleSetItem_id = queryDict.get("id", None)
    sampleSetItem_pendingId = queryDict.get("pending_id", None)
    
    #logger.debug("views_helper._update_pending_sampleSetItem_dict id=%s; pendingId=%s" %(sampleSetItem_id, sampleSetItem_pendingId))
    if sampleSetItem_pendingId is None:
        sampleSetItem_pendingId = _get_pending_sampleSetItem_id(request)
    else:
        sampleSetItem_pendingId = int(sampleSetItem_pendingId)
        
    sampleDisplayedName = queryDict.get("sampleName", "").strip()
    sampleExternalId = queryDict.get("sampleExternalId", "").strip()
    sampleDesc = queryDict.get("sampleDescription", "").strip()
    
    gender = queryDict.get("gender", "")
    relationshipRole = queryDict.get("relationshipRole", "")
    relationshipGroup = queryDict.get("relationshipGroup", None)
    

    barcodeKit = queryDict.get("barcodeKit", "")
    barcode = queryDict.get("barcode", "")
    
    isValid, errorMessage, sampleAttributes_dict = _create_pending_sampleAttributes_for_sampleSetItem(request)
    
    if errorMessage:
        return isValid, errorMessage, sampleAttributes_dict

    #create a sample object without saving it to db
    name = sampleDisplayedName.replace(' ', '_')
    
    sampleSetItem_dict = {}
    sampleSetItem_dict['pending_id'] = sampleSetItem_pendingId
    sampleSetItem_dict['name'] = name
    sampleSetItem_dict['displayedName'] = sampleDisplayedName
    sampleSetItem_dict['externalId'] = sampleExternalId
    sampleSetItem_dict['description'] = sampleDesc
    sampleSetItem_dict['status'] = "created"

    sampleSetItem_dict['gender'] = gender
    sampleSetItem_dict['relationshipRole'] = relationshipRole
    sampleSetItem_dict['relationshipGroup'] = relationshipGroup

    sampleSetItem_dict['barcodeKit'] = barcodeKit
    sampleSetItem_dict['barcode'] = barcode    
    sampleSetItem_dict['attribute_dict'] = sampleAttributes_dict
    
    #logger.debug("views_helper._create_pending_sampleSetItem_dict=%s" %(sampleSetItem_dict))
    
    return isValid, errorMessage, sampleSetItem_dict

def _get_pending_sampleSetItem_id(request):
    return  _get_pending_sampleSetItem_count(request) + 1 



def _get_pending_sampleSetItem_count(request):
    _create_pending_session_if_needed(request)
    
    return len(request.session["input_samples"]["pending_sampleSetItem_list"])


def _get_pending_sampleSetItem_by_id(request, _id):
       
    if _id and "input_samples" in request.session: 
        items = request.session["input_samples"]["pending_sampleSetItem_list"]
                
        for index, item in enumerate(request.session["input_samples"]["pending_sampleSetItem_list"]):
            #logger.debug("views_helper._get_pending_sampleSetItem_by_id - item[pending_id]=%s; _id=%s" %(str(item['pending_id']), str(_id)))
            
            if str(item['pending_id']) == str(_id):
                return item
            
        return None                
    else:
        return None


def _create_pending_sampleAttributes_for_sampleSetItem(request):
    
    sampleAttribute_list = SampleAttribute.objects.filter(isActive = True).order_by('id')  
            
    pending_attributeValue_dict = {}
    
    new_attributeValue_dict = {}
    for attribute in sampleAttribute_list:
        value = request.POST.get("sampleAttribute|" + str(attribute.id), None)

        if value:
            isValid, errorMessage = sample_validator.validate_sampleAttribute(attribute, value.encode('utf8'))
            if not isValid:
                return isValid, errorMessage, None
        else:
            isValid, errorMessage = sample_validator.validate_sampleAttribute_mandatory_for_no_value(attribute)
            if not isValid:
                return isValid, errorMessage, None
                         
        new_attributeValue_dict[attribute.id] = value.encode('utf8') if value else None

    #logger.debug("views_helper._create_pending_sampleAttributes_for_sampleSetItem#1 new_attributeValue_dict=%s" %(str(new_attributeValue_dict)))  
                     
    if new_attributeValue_dict:
        for key, newValue in new_attributeValue_dict.items():
            sampleAttribute_objs = SampleAttribute.objects.filter(id = key)
            
            if sampleAttribute_objs.count() > 0:    
                if newValue:
                    pending_attributeValue_dict[sampleAttribute_objs[0].displayedName] = newValue
                    
    return True, None, pending_attributeValue_dict


def _create_or_update_sampleAttributes_for_sampleSetItem(request, user, sample):                            
    queryDict = request.POST

    sampleAttribute_list = SampleAttribute.objects.filter(isActive = True).order_by('id')  
            
    new_attributeValue_dict = {}
    for attribute in sampleAttribute_list:
        value = request.POST.get("sampleAttribute|" + str(attribute.id), None)

        if value:
            logger.debug("views_helper._create_or_update_sampleAttributes_for_sampleSetItem() attribute=%s; value=%s" %(attribute.displayedName, value))
            
            isValid, errorMessage = sample_validator.validate_sampleAttribute(attribute, value.encode('utf8'))
            if not isValid:
                return isValid, errorMessage
        else:
                        
            logger.debug("views_helper._create_or_update_sampleAttributes_for_sampleSetItem() NO VALUE attribute=%s;" %(attribute.displayedName))

            isValid, errorMessage = sample_validator.validate_sampleAttribute_mandatory_for_no_value(attribute)
            if not isValid:
                return isValid, errorMessage
             
        new_attributeValue_dict[attribute.id] = value.encode('utf8') if value else None

    logger.debug("views_helper._create_or_update_sampleAttributes_for_sampleSetItem #1 new_attributeValue_dict=%s" %(str(new_attributeValue_dict)))  

    _create_or_update_sampleAttributes_for_sampleSetItem_with_values(request, user, sample, new_attributeValue_dict)

    return True, None


def _create_or_update_sampleAttributes_for_sampleSetItem_with_dict(request, user, sample, sampleAttribute_dict):
    """
    sampleAttribute_dict has the attribute name be the key
    """
    
    logger.debug("ENTER views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_dict - sampleAttribute_dict=%s" %(sampleAttribute_dict))
    
    new_attributeValue_dict = {}
    
    if sampleAttribute_dict:
        attribute_objs = SampleAttribute.objects.all()
        for attribute_obj in attribute_objs:
            value = sampleAttribute_dict.get(attribute_obj.displayedName, "")
            if (value):
                isValid, errorMessage = sample_validator.validate_sampleAttribute(attribute_obj, value.encode('utf8'))
                if not isValid:
                    return isValid, errorMessage
    
                new_attributeValue_dict[attribute_obj.id] = value.encode('utf8')          
    
        logger.debug("views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_dict - new_attributeValue_dict=%s" %(new_attributeValue_dict))
        
    _create_or_update_sampleAttributes_for_sampleSetItem_with_values(request, user, sample, new_attributeValue_dict)
    
    isValid = True
    return isValid, None


def _create_or_update_sampleAttributes_for_sampleSetItem_with_values(request, user, sample, new_attributeValue_dict):               
    if new_attributeValue_dict:
        currentDateTime = timezone.now()  ##datetime.datetime.now()

        #logger.debug("views_helper - ENTER new_attributeValue_dict=%s" %(new_attributeValue_dict))
            
        for key, newValue in new_attributeValue_dict.items():
            sampleAttribute_objs = SampleAttribute.objects.filter(id = key)
 
            logger.debug("views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_values() #3 sampleAttribute_objs.count=%d" %(sampleAttribute_objs.count()))
           
            if sampleAttribute_objs.count() > 0:    
                if newValue:
                       
                    attributeValue_kwargs = {
                                            'value' : newValue,
                                            'creator' : user,
                                            'creationDate' : currentDateTime,                            
                                            'lastModifiedUser' : user,                     
                                            'lastModifiedDate' : currentDateTime   
                                            }
                    attributeValue, isCreated = SampleAttributeValue.objects.get_or_create(sample = sample, sampleAttribute = sampleAttribute_objs[0], defaults=attributeValue_kwargs)
                
                    if not isCreated:
                        if attributeValue.value != newValue:
                            attributeValue.value = newValue
                            attributeValue.lastModifiedUser = user
                            attributeValue.lastModifiedDate = currentDateTime
                        
                            attributeValue.save()
                            logger.debug("views_helper - _create_or_update_sampleAttributes_for_sampleSetItem_with_values - #4 UPDATED!! isCreated=%s attributeValue.id=%d; value=%s" %(str(isCreated), attributeValue.id, newValue))                    
                    else:
                        logger.debug("views_helper - _create_or_update_sampleAttributes_for_sampleSetItem_with_values - #5 existing attributeValue!! attributeValue.id=%d; value=%s" %(attributeValue.id, newValue))
                else:
                    existingAttributeValues =  SampleAttributeValue.objects.filter(sample = sample, sampleAttribute = sampleAttribute_objs[0])
                     
                    logger.debug("views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_values() #6 existingAttributeValues.count=%d" %(existingAttributeValues.count()))

                    if (existingAttributeValues.count() > 0):
                        existingAttributeValue = existingAttributeValues[0]
                        existingAttributeValue.value = newValue
                        existingAttributeValue.lastModifiedUser = user
                        existingAttributeValue.lastModifiedDate = currentDateTime
                        
                        existingAttributeValue.save()
                        logger.debug("views_helper - _create_or_update_sampleAttributes_for_sampleSetItem_with_values - #7 UPDATED with None!! attributeValue.id=%d;" %(attributeValue.id))                                                                        
              

def _create_or_update_pending_sampleSetItem(request, user, sampleSet_ids, sample, sampleGender, sampleRelationshipRole, sampleRelationshipGroup, selectedBarcodeKit, selectedBarcode):
    currentDateTime = timezone.now()  ##datetime.datetime.now()      
    
    for sampleSet_id in sampleSet_ids:
        sampleSet = get_object_or_404(SampleSet, pk = sampleSet_id)        
    
        sampleSetItems = SampleSetItem.objects.filter(sampleSet = sampleSet, sample = sample)

        relationshipGroup = int(sampleRelationshipGroup) if sampleRelationshipGroup else 0
        
        if sampleSetItems.count() > 0:
            sampleSetItem = sampleSetItems[0]
            if sampleSetItem.gender == sampleGender and sampleSetItem.relationshipRole == sampleRelationshipRole and sampleSetItem.relationshipGroup == relationshipGroup:
                logger.debug("views_helper - _create_or_update_pending_sampleSetItem NO change for sampleSetItem.id=%d" %(sampleSetItem.id))
            else:
                    
                sampleSetItem_kwargs = {
                                        'gender' : sampleGender,
                                        'relationshipRole' : sampleRelationshipRole,
                                        'relationshipGroup' : relationshipGroup,
                                        'barcode' : selectedBarcode,                        
                                        'lastModifiedUser' : user,                     
                                        'lastModifiedDate' : currentDateTime   
                                        }
                for field, value in sampleSetItem_kwargs.iteritems():
                    setattr(sampleSetItem, field, value)
                                    
                sampleSetItem.save()                   
                logger.debug("views_helper - _create_or_update_pending_sampleSetItem UPDATED for sampleSetItem.id=%d" %(sampleSetItem.id))
        else: 
            sampleSetItem_kwargs = {
                                     'gender' : sampleGender, 
                                     'relationshipRole' : sampleRelationshipRole, 
                                     'relationshipGroup' : relationshipGroup,
                                     'barcode' : selectedBarcode,  
                                     'creator' : user,
                                     'creationDate' : currentDateTime,
                                     'lastModifiedUser' : user,
                                     'lastModifiedDate' : currentDateTime                                     
                                 }

            logger.debug("_create_or_update_pending_sampleSetItem - sampleSetItem_kwargs=%s" %(sampleSetItem_kwargs))
            
            sampleSetItem, isCreated = SampleSetItem.objects.get_or_create(sample = sample, 
                                                                           sampleSet_id = sampleSet_id, 
                                                                           defaults = sampleSetItem_kwargs)
    
            logger.debug("views_helper._create_or_update_pending_sampleSetItem() after get_or_create isCreated=%s; sampleSetItem=%s; samplesetItem.id=%d" %(str(isCreated), sample.displayedName, sampleSetItem.id))
    
    

def _create_or_update_sampleSetItem(request, user, sample):
    currentDateTime = timezone.now()  ##datetime.datetime.now()        
                        
    queryDict = request.POST
    sampleSetItem_id = queryDict.get("id", None)
                
    gender = queryDict.get("gender", "")
    relationshipRole = queryDict.get("relationshipRole", "")
    relationshipGroup = queryDict.get("relationshipGroup", None)

    sampleSetItem = get_object_or_404(SampleSetItem, pk = sampleSetItem_id)

    barcode = queryDict.get("barcode", "").strip()
                 
    if sampleSetItem.gender == gender and sampleSetItem.relationshipRole == relationshipRole and str(sampleSetItem.relationshipGroup) == str(relationshipGroup) and sampleSetItem.barcode == barcode:
        logger.debug("views_helper - _create_or_update_sampleSetItem NO change for sampleSetItem.id=%d" %(sampleSetItem.id))
    else:
            
        sampleSetItem_kwargs = {
                                'gender' : gender,
                                'relationshipRole' : relationshipRole,
                                'relationshipGroup' : relationshipGroup, 
                                'barcode' : barcode,                         
                                'lastModifiedUser' : user,                     
                                'lastModifiedDate' : currentDateTime   
                                }
        for field, value in sampleSetItem_kwargs.iteritems():
            setattr(sampleSetItem, field, value)
                            
        sampleSetItem.save()                   
        logger.debug("views_helper - _create_or_update_sampleSetItem UPDATED for sampleSetItem.id=%d" %(sampleSetItem.id))

