#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
from django.db import transaction

import sys
import os
from iondb.rundb import models

import datetime
import uuid
import random
import string

from traceback import format_exc

#20131024 change log:
# add or update all system plan template(s)
       
DEFAULT_LIB_KIT_NAME = "Ion Xpress Plus Fragment Library Kit"
        
DEFAULT_SEQ_KIT_NAME = "IonPGM200Kit-v2"
DEFAULT_TEMPLATE_KIT_NAME = "Ion PGM Template OT2 200 Kit"

DEFAULT_PROTON_SEQ_KIT_NAME = "ProtonI200Kit-v3"
DEFAULT_PROTON_TEMPLATE_KIT_NAME = "Ion PI Template OT2 200 Kit v3"

DEFAULT_LIBRARY_KEY_SEQUENCE = "TCAG"
DEFAULT_3_PRIME_ADAPTER_SEQUENCE = "ATCACCGACTGCCCATAGAGAGGCTGAGAC"

DEFAULT_MUSEEK_3_PRIME_ADAPTER_SEQUENCE = "TGAACTGACGCACGAAATCACCGACTGCCCATAGAGAGGCTGAGAC"



def finish_creating_sys_template(currentTime, sysTemplate, chipType = "", flowCount = 500, seqKitName = "", libKitName = "", barcodeKitName = "", isPGM = True, isSystemDefault = False, isMuSeek = False, targetRegionBedFile = "", hotSpotRegionBedFile = "", reference = "", plugins = {}):        
    planGUID = str(uuid.uuid4())  
    sysTemplate.planGUID = planGUID

    sysTemplate.date = currentTime


    planShortID = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(5))

    while models.PlannedExperiment.objects.filter(planShortID=planShortID, planExecuted=False):
        planShortID = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(5))      
 
    print "...Finished creating System template.id=%d; name=%s; shortID=%s" %(sysTemplate.id, sysTemplate.planDisplayedName, str(planShortID))
    
    sysTemplate.planShortID = planShortID

    sysTemplate.save()

    
    for qcType in models.QCType.objects.all():
        sysDefaultQC, isQcCreated = models.PlannedExperimentQC.objects.get_or_create(plannedExperiment = sysTemplate, 
            qcType = qcType, threshold = 30)
             
        sysTemplate.plannedexperimentqc_set.add(sysDefaultQC)
        sysTemplate.save()             


def create_sys_template_experiment(currentTime, sysTemplate, chipType, flowCount, seqKitName, libKitName, barcodeKitName, isPGM, isSystemDefault, isMuSeek, targetRegionBedFile, hotSpotRegionBedFile, reference, plugins):
    exp_kwargs = {
        'autoAnalyze' : True,
        'chipType' : chipType,
        'date' : currentTime,
        'flows' : flowCount,
        'plan' : sysTemplate,
        'sequencekitname' : seqKitName,
        'status' : sysTemplate.planStatus,
        #temp experiment name value below will be replaced in crawler
        'expName' : sysTemplate.planGUID,
        'displayName' : sysTemplate.planShortID,
        'pgmName' : '',
        'log' : '',
        #db constraint requires a unique value for experiment. temp unique value below will be replaced in crawler
        'unique' : sysTemplate.planGUID,
        'chipBarcode' : '',
        'seqKitBarcode' : '',
        'sequencekitbarcode' : '',
        'reagentBarcode' : '',
         'cycles' : 0,
         'diskusage' : 0,
         'expCompInfo' : '',
         'baselineRun' : '',
         'flowsInOrder' : '',
         'ftpStatus' : '',
         'runMode' : sysTemplate.runMode,
         'storageHost' : '',
         'notes' : ''
        }
    
    experiment = models.Experiment(**exp_kwargs) 
    experiment.save()

    print "*** AFTER saving experiment.id=%d for system template.id=%d; name=%s" % (experiment.id, sysTemplate.id, sysTemplate.planName) 
    return experiment


def create_sys_template_eas(currentTime, experiment, sysTemplate, chipType, flowCount, seqKitName, libKitName, barcodeKitName, isPGM, isSystemDefault, isMuSeek, targetRegionBedFile, hotSpotRegionBedFile, reference, plugins):
    threePrimeAdapter = DEFAULT_3_PRIME_ADAPTER_SEQUENCE
    if (isMuSeek):
        threePrimeAdapter = DEFAULT_MUSEEK_3_PRIME_ADAPTER_SEQUENCE

    eas_kwargs = {
        'barcodedSamples' : "",
        'barcodeKitName' : barcodeKitName,
        'date' : currentTime,
        'experiment' : experiment,
        'hotSpotRegionBedFile' : hotSpotRegionBedFile,
        'isEditable' : True,
        'isOneTimeOverride' : False,
        'libraryKey' : DEFAULT_LIBRARY_KEY_SEQUENCE,
        'libraryKitName' : libKitName,
        'reference' : reference,
        'selectedPlugins' : plugins,
        'status' : sysTemplate.planStatus,
        'targetRegionBedFile' : targetRegionBedFile,
        'threePrimeAdapter' : threePrimeAdapter
        }

    eas = models.ExperimentAnalysisSettings(**eas_kwargs)
    eas.save() 

    print "*** AFTER saving EAS.id=%d for system template.id=%d; name=%s" % (eas.id, sysTemplate.id, sysTemplate.planName) 
    return sysTemplate


def finish_sys_template(sysTemplate, isCreated, chipType = "", flowCount = 500, seqKitName = "", libKitName = "", barcodeKitName = "", isPGM = True, isSystemDefault = False, isMuSeek = False, targetRegionBedFile = "", hotSpotRegionBedFile = "", reference = "", plugins = {}):        
    currentTime = datetime.datetime.now()
        
    if isCreated:
        finish_creating_sys_template(currentTime, sysTemplate, chipType, flowCount, seqKitName, libKitName, barcodeKitName, isPGM, isSystemDefault, isMuSeek, targetRegionBedFile, hotSpotRegionBedFile, reference, plugins)        
        experiment = create_sys_template_experiment(currentTime, sysTemplate, chipType, flowCount, seqKitName, libKitName, barcodeKitName, isPGM, isSystemDefault, isMuSeek, targetRegionBedFile, hotSpotRegionBedFile, reference, plugins)
        create_sys_template_eas(currentTime, experiment, sysTemplate, chipType, flowCount, seqKitName, libKitName, barcodeKitName, isPGM, isSystemDefault, isMuSeek, targetRegionBedFile, hotSpotRegionBedFile, reference, plugins)

    exps = models.Experiment.objects.filter(plan = sysTemplate)

    if not exps:
        experiment = create_sys_template_experiment(currentTime, sysTemplate, chipType, flowCount, seqKitName, libKitName, barcodeKitName, isPGM, isSystemDefault, isMuSeek, targetRegionBedFile, hotSpotRegionBedFile, reference, plugins)
        return create_sys_template_eas(currentTime, experiment, sysTemplate, chipType, flowCount, seqKitName, libKitName, barcodeKitName, isPGM, isSystemDefault, isMuSeek, targetRegionBedFile, hotSpotRegionBedFile, reference, plugins)
            
    exp = exps[0]
        
    hasChanges = False
    if (exp.status != sysTemplate.planStatus):
        print ">>> DIFF: orig exp.status=%s for system template.id=%d; name=%s" % (exp.status, sysTemplate.id, sysTemplate.planName) 

        exp.status = sysTemplate.planStatus
        hasChanges = True
        
    if (exp.chipType != chipType):
        print ">>> DIFF: orig exp.chipType=%s for system template.id=%d; name=%s" % (exp.chipType, sysTemplate.id, sysTemplate.planName) 
        
        exp.chipType = chipType
        hasChanges = True
        
    if (exp.flows != flowCount):
        print ">>> DIFF: orig exp.flows=%s for system template.id=%d; name=%s" % (exp.flows, sysTemplate.id, sysTemplate.planName) 
        
        exp.flows = flowCount
        hasChanges = True
        
    if (exp.sequencekitname != seqKitName):
        print ">>> DIFF: orig exp.sequencekitname=%s for system template.id=%d; name=%s" % (exp.sequencekitname, sysTemplate.id, sysTemplate.planName) 
        
        exp.sequencekitname = seqKitName
        hasChanges = True

    if hasChanges:
        exp.date = currentTime
        exp.save()
        
        print "*** AFTER updating experiment.id=%d for system template.id=%d; name=%s" % (exp.id, sysTemplate.id, sysTemplate.planName) 
       
    eas_set = models.ExperimentAnalysisSettings.objects.filter(experiment = exp, isEditable = True, isOneTimeOverride = False)
    
    if not eas_set:
        return create_sys_template_eas(currentTime, exp, sysTemplate, chipType, flowCount, seqKitName, libKitName, barcodeKitName, isPGM, isSystemDefault, isMuSeek, targetRegionBedFile, hotSpotRegionBedFile, reference, plugins)

    eas = eas_set[0]
    
    hasChanges = False
    if (eas.barcodeKitName != barcodeKitName):
        print ">>> DIFF: orig eas.barcodeKitName=%s for system template.id=%d; name=%s" % (eas.barcodeKitName, sysTemplate.id, sysTemplate.planName) 
        eas.barcodeKitName = barcodeKitName
        hasChanges = True
        
    if (eas.hotSpotRegionBedFile != hotSpotRegionBedFile):
        print ">>> DIFF: orig eas.hotSpotRegionBedFile=%s for system template.id=%d; name=%s" % (eas.hotSpotRegionBedFile, sysTemplate.id, sysTemplate.planName) 
        
        eas.hotSpotRegionBedFile = hotSpotRegionBedFile
        hasChanges = True
    
    if (eas.libraryKey != DEFAULT_LIBRARY_KEY_SEQUENCE):
        print ">>> DIFF: orig eas.libraryKeye=%s for system template.id=%d; name=%s" % (eas.libraryKey, sysTemplate.id, sysTemplate.planName) 
        
        eas.libraryKey = DEFAULT_LIBRARY_KEY_SEQUENCE
        hasChanges = True
        
    if (eas.libraryKitName != libKitName):
        print ">>> DIFF: orig eas.libraryKitName=%s for system template.id=%d; name=%s" % (eas.libraryKitName, sysTemplate.id, sysTemplate.planName) 
        
        eas.libraryKitName = libKitName
        hasChanges = True
        
    if (eas.reference != reference):
        print ">>> DIFF: orig eas.reference=%s for system template.id=%d; name=%s" % (eas.reference, sysTemplate.id, sysTemplate.planName) 
        
        eas.reference = reference
        hasChanges = True
        
    if (eas.selectedPlugins != plugins):
        print ">>> DIFF: orig eas.selectedPlugins=%s for system template.id=%d; name=%s" % (eas.selectedPlugins, sysTemplate.id, sysTemplate.planName) 
        
        eas.selectedPlugins = plugins
        hasChanges = True
        
    if (eas.status != sysTemplate.planStatus):
        print ">>> DIFF: orig eas.status=%s for system template.id=%d; name=%s" % (eas.status, sysTemplate.id, sysTemplate.planName) 
        
        eas.status = sysTemplate.planStatus
        hasChanges = True   
        
    if (eas.targetRegionBedFile != targetRegionBedFile):
        print ">>> DIFF: orig eas.targetRegionBedFile=%s for system template.id=%d; name=%s" % (eas.targetRegionBedFile, sysTemplate.id, sysTemplate.planName) 
        
        eas.targetRegionBedFile = targetRegionBedFile
        hasChanges = True  
     
    threePrimeAdapter = DEFAULT_3_PRIME_ADAPTER_SEQUENCE
    if (isMuSeek):
        threePrimeAdapter = DEFAULT_MUSEEK_3_PRIME_ADAPTER_SEQUENCE

    if (eas.threePrimeAdapter != threePrimeAdapter): 
        print ">>> DIFF: orig eas.threePrimeAdapter=%s for system template.id=%d; name=%s" % (eas.threePrimeAdapter, sysTemplate.id, sysTemplate.planName) 
                  
        eas.threePrimeAdapter = threePrimeAdapter
        hasChanges = True    
                           
    if (hasChanges):
        eas.date = currentTime
        eas.save()
        
        print "*** AFTER saving EAS.id=%d for system default template.id=%d; name=%s" % (eas.id, sysTemplate.id, sysTemplate.planName) 
        
    return sysTemplate



def add_or_update_sys_template(planDisplayedName, application, applicationGroup = "DNA", isPGM = True, isSystemDefault = False, isMuSeek = False, templateKitName = "", controlSeqKitName = None, samplePrepKitName = None, sampleGrouping = None):        
    sysTemplate = None
    isCreated = False
    isUpdated = False
    
    if not planDisplayedName:
        return sysTemplate, isCreated, isUpdated
    
    planName = planDisplayedName.replace(' ', '_')
    currentTime = datetime.datetime.now()
    
    applicationGroup_objs = models.ApplicationGroup.objects.filter(name__iexact = applicationGroup)
    
    applicationGroup_obj = None
    if applicationGroup_objs:
        applicationGroup_obj = applicationGroup_objs[0]
     
    sampleGrouping_obj = None
    if (sampleGrouping):   
        sampleGrouping_objs = models.SampleGroupType_CV.objects.filter(displayedName__iexact = sampleGrouping)
        
        if sampleGrouping_objs:
            sampleGrouping_obj = sampleGrouping_objs[0]
        
    if isPGM:
        sysTemplate, isCreated = models.PlannedExperiment.objects.get_or_create(isSystemDefault = isSystemDefault, isSystem = True, 
            isReusable = True, isPlanGroup = False,
            planDisplayedName = planDisplayedName, planName = planName, 
            defaults={"planStatus" : "planned", "runMode" : "single", "isReverseRun" : False, "planExecuted" : False,
            "runType" : application, "usePreBeadfind" : True, "usePostBeadfind" : True, "preAnalysis" : True, 
            "chipBarcode" : "", "planPGM" : "",
            "templatingKitName" : templateKitName, "controlSequencekitname" : controlSeqKitName, "samplePrepKitName" : samplePrepKitName,
            "metaData" : "", "date" : currentTime, "applicationGroup" : applicationGroup_obj, "sampleGrouping" : sampleGrouping_obj})
    else:
        sysTemplate, isCreated = models.PlannedExperiment.objects.get_or_create(isSystemDefault = isSystemDefault, isSystem = True, 
            isReusable = True, isPlanGroup = False,
            planDisplayedName = planDisplayedName, planName = planName, 
            defaults={"planStatus" : "planned","runMode" : "single", "isReverseRun" : False, "planExecuted" : False,
            "runType" : application, "usePreBeadfind" : True, "usePostBeadfind" : False, "preAnalysis" : True,
            "chipBarcode" : "", "planPGM" : "",
            "templatingKitName" : templateKitName, "controlSequencekitname" : controlSeqKitName, "samplePrepKitName" : samplePrepKitName, 
            "metaData" : "", "date" : currentTime, "applicationGroup" : applicationGroup_obj, "sampleGrouping" : sampleGrouping_obj})
        
    if isCreated:
            print "...Created System template.id=%d; name=%s; isSystemDefault=%s" %(sysTemplate.id, sysTemplate.planDisplayedName, str(isSystemDefault))
    else:
        hasChanges = False
        if (sysTemplate.planStatus != "planned"):
            print ">>> DIFF: orig sysTemplate.planStatus=%s for system template.id=%d; name=%s" % (sysTemplate.planStatus, sysTemplate.id, sysTemplate.planName) 
            
            sysTemplate.planStatus = "planned"
            hasChanges = True
            
        if (sysTemplate.planExecuted):
            print ">>> DIFF: orig sysTemplate.planExecuted=%s for system template.id=%d; name=%s" % (sysTemplate.planExecuted, sysTemplate.id, sysTemplate.planName) 
                        
            sysTemplate.planExecuted = False
            hasChanges = True
            
        if (sysTemplate.runType != application):
            print ">>> DIFF: orig sysTemplate.runType=%s for system template.id=%d; name=%s" % (sysTemplate.runType, sysTemplate.id, sysTemplate.planName) 
                        
            sysTemplate.runType = application
            hasChanges = True
            
        if (sysTemplate.templatingKitName != templateKitName):
            print ">>> DIFF: orig sysTemplate.templatingKitName=%s for system template.id=%d; name=%s" % (sysTemplate.templatingKitName, sysTemplate.id, sysTemplate.planName) 
                        
            sysTemplate.templatingKitName = templateKitName
            hasChanges = True
            
        if (isPGM and sysTemplate.usePostBeadfind != True):
            print ">>> DIFF: orig PGM sysTemplate.usePostBeadfind=%s for system template.id=%d; name=%s" % (sysTemplate.usePostBeadfind, sysTemplate.id, sysTemplate.planName) 
                        
            sysTemplate.usePostBeadfind = True
            hasChanges = True
        
        if (not isPGM and sysTemplate.usePostBeadfind != False):
            print ">>> DIFF: orig PROTON sysTemplate.usePostBeadfind=%s for system template.id=%d; name=%s" % (sysTemplate.usePostBeadfind, sysTemplate.id, sysTemplate.planName) 
                        
            sysTemplate.usePostBeadfind = False
            hasChanges = True
            
        if (sysTemplate.applicationGroup != applicationGroup_obj):
            print ">>> DIFF: orig sysTemplate.applicationGroup=%s for system template.id=%d; name=%s" % (sysTemplate.applicationGroup, sysTemplate.id, sysTemplate.planName) 
                        
            sysTemplate.applicationGroup = applicationGroup_obj
            hasChanges = True
            
        if (sysTemplate.sampleGrouping != sampleGrouping_obj):
            print ">>> DIFF: orig sysTemplate.sampleGrouping=%s for system template.id=%d; name=%s" % (sysTemplate.sampleGrouping, sysTemplate.id, sysTemplate.planName) 
                        
            sysTemplate.sampleGrouping = sampleGrouping_obj
            hasChanges = True
                        
        if hasChanges:
            sysTemplate.date = currentTime
            sysTemplate.save()
            isUpdated = True            
        
    if isUpdated:
        print "...Updated System template.id=%d; name=%s" %(sysTemplate.id, sysTemplate.planDisplayedName)
    
    if (not isCreated and not isUpdated):
        print "...No changes in plannedExperiment for System template.id=%d; name=%s" %(sysTemplate.id, sysTemplate.planDisplayedName)
                                                                              
    return sysTemplate, isCreated, isUpdated


@transaction.commit_manually()  
def add_or_update_all_system_templates():

    try:
        #system default templates
        #1
        templateName = "Proton System Default Template"        
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "GENS", "DNA", False, True, False, DEFAULT_PROTON_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "900", 260, DEFAULT_PROTON_SEQ_KIT_NAME, DEFAULT_LIB_KIT_NAME, "", False, True)
       
        #2
        templateName = "System Default Template"        
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "GENS", "DNA", True, True, False, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "", 500, DEFAULT_SEQ_KIT_NAME, DEFAULT_LIB_KIT_NAME, "", True, True)

        #ampliseq
        #3
        templateName = "Ion AmpliSeq Cancer Hotspot Panel v2"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "AMPS", "DNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "", 500, DEFAULT_SEQ_KIT_NAME, "Ion AmpliSeq 2.0 Library Kit", "", True, False, False, "", "", "hg19")
                
        #4
        templateName = "Ion AmpliSeq Cancer Panel"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "AMPS", "DNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "", 500, DEFAULT_SEQ_KIT_NAME, "Ion AmpliSeq 2.0 Library Kit", "", True, False, False, "/hg19/unmerged/detail/HSMv12.1_reqions_NO_JAK2_NODUP.bed", "/hg19/unmerged/detail/HSMv12.1_hotspots_NO_JAK2.bed", "hg19")
        
        #5
        templateName = "Ion AmpliSeq Cancer Panel 1_0 Lib Chem"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "AMPS", "DNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "", 500, DEFAULT_SEQ_KIT_NAME, "Ion AmpliSeq Kit", "", True, False, False, "", "", "hg19")
        
        #6
        templateName = "Ion AmpliSeq Comprehensive Cancer Panel"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "AMPS", "DNA", False, False, False, DEFAULT_PROTON_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "P1.1.17", 360, DEFAULT_PROTON_SEQ_KIT_NAME, "Ion AmpliSeq 2.0 Library Kit", "", False, False, False, "/hg19/unmerged/detail/4477685_CCP_bedfile_20120517.bed", "", "hg19")
        
        #7
        templateName = "Ion AmpliSeq Custom"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "AMPS", "DNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "", 500, DEFAULT_SEQ_KIT_NAME, DEFAULT_LIB_KIT_NAME, "", True, False, False, "", "", "hg19")

        #8
        templateName = "Ion AmpliSeq Custom ID"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "AMPS", "DNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME, "Ion AmpliSeq Sample ID Panel")
        finish_sys_template(sysTemplate, isCreated, "", 500, DEFAULT_SEQ_KIT_NAME, DEFAULT_LIB_KIT_NAME, "", True, False, False, "", "", "hg19")
        
        #9
        templateName = "Ion AmpliSeq Inherited Disease Panel"        
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "AMPS", "DNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "316", 500, DEFAULT_SEQ_KIT_NAME, "Ion AmpliSeq 2.0 Library Kit", "", True, False, False, "/hg19/unmerged/detail/4477686_IDP_bedfile_20120613.bed", "", "hg19")

        #ampliseq RNA        
        #10
        templateName = "Ion AmpliSeq RNA Panel"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "AMPS_RNA", "RNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME)
        
 
        #pre-select plugins coverageAnalysis and ERCC_Analysis 
        plugins = {}
        pluginUserInput = {}
                                    
        selectedPlugins = models.Plugin.objects.filter(name__in = ["coverageAnalysis", "ERCC_Analysis"], selected = True, active = True)

        for selectedPlugin in selectedPlugins:
            pluginDict = {
                          "id" : selectedPlugin.id,
                          "name" : selectedPlugin.name,
                          "version" : selectedPlugin.version,
                          "userInput" : pluginUserInput,
                          "features": []
                          }
            
            plugins[selectedPlugin.name] = pluginDict
        
        finish_sys_template(sysTemplate, isCreated, "", 500, DEFAULT_SEQ_KIT_NAME, "Ion AmpliSeq RNA Library Kit", "RNA_Barcode_None", True, False, False, "", "", "", plugins)

        #generic sequencing           
        #11
        templateName = "Ion PGM E_coli DH10B Control 200"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "GENS", "DNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "", 500, DEFAULT_SEQ_KIT_NAME, DEFAULT_LIB_KIT_NAME, "", True, False, False, "", "", "e_coli_dh10b")        
        
        #12
        templateName = "Ion PGM E_coli DH10B Control 400"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "GENS", "DNA", True, False, False, "IonPGM400Kit")
        finish_sys_template(sysTemplate, isCreated, "314", 850, "IonPGM400Kit", DEFAULT_LIB_KIT_NAME, "", True, False, False, "", "", "e_coli_dh10b")
          
        #13
        templateName = "Ion Proton Human CEPH Control 170"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "GENS", "DNA", False, False, False, DEFAULT_PROTON_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "P1.1.17", 440, DEFAULT_PROTON_SEQ_KIT_NAME, DEFAULT_LIB_KIT_NAME, "", False, False, False, "", "", "hg19")
        
        #14
        templateName = "MuSeek Barcoded Library"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "GENS", "DNA", True, False, True, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "318", 500, DEFAULT_SEQ_KIT_NAME, "MuSeek(tm) Library Preparation Kit", "MuSeek Barcode set 1", False, False, True, "", "", "hg19")
        
        #15
        templateName = "MuSeek Library"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "GENS", "DNA", True, False, True, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "318", 500, DEFAULT_SEQ_KIT_NAME, "MuSeek(tm) Library Preparation Kit", "MuSeek_5prime_tag", False, False, True, "", "", "hg19")
             
        #16
        templateName = "System Generic Seq Template"        
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "GENS", "DNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "", 500, DEFAULT_SEQ_KIT_NAME, DEFAULT_LIB_KIT_NAME, "", True, False, False, "", "", "hg19")                

        #rna sequencing
        #17
        templateName = "Ion RNA - small"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "RNA", "RNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME)
        finish_sys_template(sysTemplate, isCreated, "", 160, DEFAULT_SEQ_KIT_NAME, "Ion Total RNA Seq Kit v2", "RNA_Barcode_None", True, False, False, "", "", "")        
            
        #18
        templateName = "Ion RNA - Whole Transcriptome"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "RNA", "RNA", False, False, False, DEFAULT_PROTON_TEMPLATE_KIT_NAME )
        finish_sys_template(sysTemplate, isCreated, "P1.1.17", 440, DEFAULT_PROTON_SEQ_KIT_NAME, "Ion Total RNA Seq Kit v2", "RNA_Barcode_None", True, False, False, "", "", "")        

        #targetSeq     
        #19
        templateName = "Ion TargetSeq Custom"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "TARS", "DNA", True, False, False, DEFAULT_TEMPLATE_KIT_NAME, None, "Ion TargetSeq(tm) Custom Enrichment Kit (100kb-500kb)")
        finish_sys_template(sysTemplate, isCreated, "318", 500, DEFAULT_SEQ_KIT_NAME, DEFAULT_LIB_KIT_NAME, "", True, False, False, "", "", "")        
        
        #20
        templateName = "Ion TargetSeq Proton Exome"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "TARS", "DNA", False, False, False, DEFAULT_PROTON_TEMPLATE_KIT_NAME, None, "Ion TargetSeq(tm) Exome Kit (4 rxn)" )
        finish_sys_template(sysTemplate, isCreated, "P1.1.17", 440, DEFAULT_PROTON_SEQ_KIT_NAME, DEFAULT_LIB_KIT_NAME, "", False, False, False, "", "", "hg19")        


        #16S      
        #21
        templateName = "Ion 16S Metagenomics Template"
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateName, "TARS_16S", "Metagenomics", True, False, False, "Ion PGM Template OT2 400 Kit", None, None, "Self")
        finish_sys_template(sysTemplate, isCreated, "316v2", 850, "IonPGM400Kit", "IonPlusFragmentLibKit", "", True, False, False)        
 
                   
    except:
        print format_exc()
        transaction.rollback()
        print "*** Exceptions found. System Template(s) rolled back."          
    else:
        transaction.commit()            
        print "*** System Template(s) committed."      
   
      
# main
add_or_update_all_system_templates()
#we can also delete system templates here...



        