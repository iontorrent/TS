#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
from django.db import transaction

import sys
import os
import json
from iondb.rundb import models

import datetime
import uuid
import random
import string

from traceback import format_exc

import logging
logger = logging.getLogger(__name__)

PGM = 'PGM'
PROTON = 'PROTON'
S5 = 'S5'
DEFAULT_3_PRIME_ADAPTER_SEQUENCE = "ATCACCGACTGCCCATAGAGAGGCTGAGAC" #Ion P1B
DEFAULT_MUSEEK_3_PRIME_ADAPTER_SEQUENCE = "TGAACTGACGCACGAAATCACCGACTGCCCATAGAGAGGCTGAGAC"

class TemplateParams:

    def __init__(self, templateName, instrument, runType="GENS"):
        # PlannedExperiment fields
        self.templateName = templateName
        self.runType = runType
        self.applicationGroup = "DNA"
        self.sampleGrouping = None
        self.categories = ""
        self.usePreBeadfind = True
        self.usePostBeadfind = True
        self.templatingKitName = ""
        self.controlSequencekitname = None
        self.samplePrepKitName = None
        self.templatingSize = None
        self.libraryReadLength = 0
        self.samplePrepProtocol = ""
        self.planStatus = "planned"
        # Experiment
        self.chipType = ""
        self.flows = 0
        self.sequencekitname = ""
        self.flowOrder = ""
        # EAS
        self.barcodeKitName = ""
        self.libraryKey = "TCAG"
        self.tfKey = "ATCG"
        self.threePrimeAdapter = ""
        self.reference = ""
        self.targetRegionBedFile = ""
        self.hotSpotRegionBedFile = ""
        self.libraryKitName = ""
        self.selectedPlugins = ""

        if instrument == PGM:
            self.instrumentType = "PGM"
            self.libraryKitName = "Ion Xpress Plus Fragment Library Kit"
            self.templatingKitName = "Ion PGM Template OT2 200 Kit"
            self.sequencekitname = "IonPGM200Kit-v2"
            self.threePrimeAdapter = DEFAULT_3_PRIME_ADAPTER_SEQUENCE
            self.flows = 500

        elif instrument == PROTON:
            self.instrumentType = "PROTON"
            self.libraryKitName = "Ion Xpress Plus Fragment Library Kit"
            self.templatingKitName = "Ion PI Template OT2 200 Kit v3"
            self.sequencekitname = "ProtonI200Kit-v3"
            self.threePrimeAdapter = DEFAULT_3_PRIME_ADAPTER_SEQUENCE
            self.flows = 260
            self.usePreBeadfind = True
            self.usePostBeadfind = False
        
        elif instrument == S5:
            self.instrumentType = "S5"
            self.libraryKitName = "Ion Xpress Plus Fragment Library Kit"
            self.templatingKitName = "Ion 540 Control Ion Spheres"
            self.sequencekitname = "Ion S5 Sequencing Kit"
            self.threePrimeAdapter = DEFAULT_3_PRIME_ADAPTER_SEQUENCE
            self.flows = 500
            self.usePreBeadfind = True
            self.usePostBeadfind = False

        else:
            raise Exception('Unknown instrument key: %s' % instrument)
    
    def update(self, d):
        fields = self.__dict__.keys()
        for key,value in d.items():
            if key in fields:
                setattr(self, key, value)
            else:
                raise Exception('Incorrect field key: %s' % key)


def finish_creating_sys_template(currentTime, sysTemplate, templateParams):
    planGUID = str(uuid.uuid4())
    sysTemplate.planGUID = planGUID

    sysTemplate.date = currentTime

    planShortID = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(5))

    while models.PlannedExperiment.objects.filter(planShortID=planShortID, planExecuted=False):
        planShortID = ''.join(random.choice(string.ascii_uppercase + string.digits)
                              for x in range(5))

    print "...Finished creating System template.id=%d; name=%s; shortID=%s" % \
        (sysTemplate.id, sysTemplate.planDisplayedName, str(planShortID))

    sysTemplate.planShortID = planShortID

    sysTemplate.save()

    for qcType in models.QCType.objects.all():
        sysDefaultQC, isQcCreated = models.PlannedExperimentQC.objects.get_or_create(
            plannedExperiment=sysTemplate,
            qcType=qcType, threshold=30)

        sysTemplate.plannedexperimentqc_set.add(sysDefaultQC)
        sysTemplate.save()


def create_sys_template_experiment(currentTime, sysTemplate, templateParams):
    
    exp_kwargs = {
        'autoAnalyze': True,
        'chipType': templateParams.chipType,
        'date': currentTime,
        'flows': templateParams.flows,
        'plan': sysTemplate,
        'sequencekitname': templateParams.sequencekitname,
        'status': sysTemplate.planStatus,
        # temp experiment name value below will be replaced in crawler
        'expName': sysTemplate.planGUID,
        'displayName': sysTemplate.planShortID,
        'pgmName': '',
        'log': '',
        # db constraint requires a unique value for experiment. temp unique value
        # below will be replaced in crawler
        'unique': sysTemplate.planGUID,
        'chipBarcode': '',
        'seqKitBarcode': '',
        'sequencekitbarcode': '',
        'reagentBarcode': '',
        'cycles': 0,
        'diskusage': 0,
        'expCompInfo': '',
        'baselineRun': '',
        'flowsInOrder': templateParams.flowOrder,
        'ftpStatus': '',
        'runMode': sysTemplate.runMode,
        'storageHost': '',
        'notes': '',
        'status' : templateParams.planStatus
        }

    experiment = models.Experiment(**exp_kwargs)
    experiment.save()

    print "*** AFTER saving experiment.id=%d for system template.id=%d; name=%s" % \
        (experiment.id, sysTemplate.id, sysTemplate.planName)
    return experiment


def create_sys_template_eas(currentTime, experiment, sysTemplate, templateParams, plugins):

    eas_kwargs = {
        'barcodedSamples': "",
        'barcodeKitName': templateParams.barcodeKitName ,
        'date': currentTime,
        'experiment': experiment,
        'hotSpotRegionBedFile': templateParams.hotSpotRegionBedFile,
        'isEditable': True,
        'isOneTimeOverride': False,
        'libraryKey': templateParams.libraryKey,
        'libraryKitName': templateParams.libraryKitName,
        'reference': templateParams.reference,
        'selectedPlugins': plugins,
        'status': sysTemplate.planStatus,
        'targetRegionBedFile': templateParams.targetRegionBedFile,
        'threePrimeAdapter': templateParams.threePrimeAdapter,
        'tfKey': templateParams.tfKey,
        }

    eas = models.ExperimentAnalysisSettings(**eas_kwargs)
    eas.save()

    sysTemplate.latestEAS = eas
    sysTemplate.save()

    print "*** AFTER saving EAS.id=%d for system template.id=%d; name=%s" % \
        (eas.id, sysTemplate.id, sysTemplate.planName)
    return sysTemplate


def finish_sys_template(sysTemplate, isCreated, templateParams, plugins={}):

    currentTime = datetime.datetime.now()

    if isCreated:
        finish_creating_sys_template(currentTime, sysTemplate, templateParams)
        experiment = create_sys_template_experiment(currentTime, sysTemplate, templateParams)
        create_sys_template_eas(currentTime, experiment, sysTemplate, templateParams, plugins)

    exps = models.Experiment.objects.filter(plan=sysTemplate)

    if not exps:
        experiment = create_sys_template_experiment(currentTime, sysTemplate, templateParams)
        return create_sys_template_eas(currentTime, experiment, sysTemplate, templateParams, plugins)

    exp = exps[0]

    hasChanges = False
    if (exp.status != sysTemplate.planStatus):
        print ">>> DIFF: orig exp.status=%s for system template.id=%d; name=%s" % \
            (exp.status, sysTemplate.id, sysTemplate.planName)

        exp.status = sysTemplate.planStatus
        hasChanges = True

    if (exp.chipType != templateParams.chipType):
        print ">>> DIFF: orig exp.chipType=%s for system template.id=%d; name=%s" % \
            (exp.chipType, sysTemplate.id, sysTemplate.planName)

        exp.chipType = templateParams.chipType
        hasChanges = True

    if (exp.flows != templateParams.flows):
        print ">>> DIFF: orig exp.flows=%s for system template.id=%d; name=%s" % \
            (exp.flows, sysTemplate.id, sysTemplate.planName)

        exp.flows = templateParams.flows
        hasChanges = True

    if (exp.sequencekitname != templateParams.sequencekitname):
        print ">>> DIFF: orig exp.sequencekitname=%s for system template.id=%d; name=%s" % \
            (exp.sequencekitname, sysTemplate.id, sysTemplate.planName)

        exp.sequencekitname = templateParams.sequencekitname
        hasChanges = True

    if (exp.platform != templateParams.instrumentType):
        print ">>> DIFF: orig exp.platform=%s new instrumentType=%s for system template.id=%d; name=%s" % \
            (exp.platform, templateParams.instrumentType,
             sysTemplate.id, sysTemplate.planName)

        exp.platform = templateParams.instrumentType
        hasChanges = True

    if (exp.flowsInOrder != templateParams.flowOrder):
        print ">>> DIFF: orig exp.flowInOrder=%s for system template.id=%d; name=%s" % \
            (exp.flowsInOrder, sysTemplate.id, sysTemplate.planName)

        exp.flowsInOrder = templateParams.flowOrder
        hasChanges = True

    if hasChanges:
        exp.date = currentTime
        exp.save()

        print "*** AFTER updating experiment.id=%d for system template.id=%d; name=%s" % \
            (exp.id, sysTemplate.id, sysTemplate.planName)


    eas_set = models.ExperimentAnalysisSettings.objects.filter(
        experiment=exp, isEditable=True, isOneTimeOverride=False)

    if not eas_set:
        return create_sys_template_eas(currentTime, exp, sysTemplate, templateParams, plugins)

    eas = eas_set[0]

    hasChanges = False
    if (eas.barcodeKitName != templateParams.barcodeKitName):
        print ">>> DIFF: orig eas.barcodeKitName=%s for system template.id=%d; name=%s" % \
            (eas.barcodeKitName, sysTemplate.id, sysTemplate.planName)
        eas.barcodeKitName = templateParams.barcodeKitName
        hasChanges = True

    if (eas.hotSpotRegionBedFile != templateParams.hotSpotRegionBedFile):
        print ">>> DIFF: orig eas.hotSpotRegionBedFile=%s for system template.id=%d; name=%s" % \
            (eas.hotSpotRegionBedFile,
             sysTemplate.id, sysTemplate.planName)

        eas.hotSpotRegionBedFile = templateParams.hotSpotRegionBedFile
        hasChanges = True

    if (eas.libraryKey != templateParams.libraryKey):
        print ">>> DIFF: orig eas.libraryKeye=%s for system template.id=%d; name=%s" % \
            (eas.libraryKey, sysTemplate.id, sysTemplate.planName)

        eas.libraryKey = templateParams.libraryKey
        hasChanges = True

    if (eas.libraryKitName != templateParams.libraryKitName):
        print ">>> DIFF: orig eas.libraryKitName=%s for system template.id=%d; name=%s" % \
            (eas.libraryKitName, sysTemplate.id, sysTemplate.planName)

        eas.libraryKitName = templateParams.libraryKitName
        hasChanges = True

    if (eas.reference != templateParams.reference):
        print ">>> DIFF: orig eas.reference=%s for system template.id=%d; name=%s" % \
            (eas.reference, sysTemplate.id, sysTemplate.planName)

        eas.reference = templateParams.reference
        hasChanges = True

    if not simple_compare_dict(eas.selectedPlugins, plugins):
        print ">>> DIFF: orig eas.selectedPlugins=%s for system template.id=%d; name=%s" % \
            (eas.selectedPlugins, sysTemplate.id, sysTemplate.planName)
        print ">>> DIFF: NEW selectedPlugins=%s for system template.id=%d; name=%s" % \
            (plugins, sysTemplate.id, sysTemplate.planName)

        eas.selectedPlugins = plugins
        hasChanges = True

    if (eas.status != sysTemplate.planStatus):
        print ">>> DIFF: orig eas.status=%s for system template.id=%d; name=%s" % \
            (eas.status, sysTemplate.id, sysTemplate.planName)

        eas.status = sysTemplate.planStatus
        hasChanges = True

    if (eas.targetRegionBedFile != templateParams.targetRegionBedFile):
        print ">>> DIFF: orig eas.targetRegionBedFile=%s for system template.id=%d; name=%s" % \
            (eas.targetRegionBedFile,
             sysTemplate.id, sysTemplate.planName)

        eas.targetRegionBedFile = templateParams.targetRegionBedFile
        hasChanges = True

    if (eas.threePrimeAdapter != templateParams.threePrimeAdapter):
        print ">>> DIFF: orig eas.threePrimeAdapter=%s for system template.id=%d; name=%s" % \
            (eas.threePrimeAdapter, sysTemplate.id, sysTemplate.planName)

        eas.threePrimeAdapter = templateParams.threePrimeAdapter
        hasChanges = True

    if (eas.tfKey != templateParams.tfKey):
        print ">>> DIFF: orig eas.tfKey=%s for system template.id=%d; name=%s" % \
            (eas.tfKey, sysTemplate.id, sysTemplate.planName)

        eas.tfKey = templateParams.tfKey
        hasChanges = True
        
    if (sysTemplate.latestEAS != eas):
        print ">>> DIFF: orig eas.latestEAS=%s for system template.id=%d; name=%s" % \
            (sysTemplate.latestEAS, sysTemplate.id, sysTemplate.planName)

        sysTemplate.latestEAS = eas
        sysTemplate.save()

    if (hasChanges):
        eas.date = currentTime
        eas.save()

        print "*** AFTER saving EAS.id=%d for system default template.id=%d; name=%s" % \
            (eas.id, sysTemplate.id, sysTemplate.planName)

    return sysTemplate


def _get_plugin_dict(pluginName, userInput={}):
    try:
        selectedPlugin = models.Plugin.objects.get(name=pluginName, selected=True, active=True)
        pluginDict = {
            "id": selectedPlugin.id,
            "name": selectedPlugin.name,
            "version": selectedPlugin.version,
            "userInput": userInput,
            "features": []
        }
    except models.Plugin.DoesNotExist:
        pluginDict = {
            "id": 9999,
            "name": pluginName,
            "version": "1.0",
            "userInput": userInput,
            "features": []
        }

    return pluginDict


def simple_compare_dict(dict1, dict2):
    ''' accepts multi-level dictionaries
        compares values as strings, will not report type mismatch
    '''
    if sorted(dict1.keys()) != sorted(dict2.keys()):
        return False

    for key, value in dict1.iteritems():
        if isinstance(value, dict):
            if not simple_compare_dict(value, dict2[key]):
                return False
        elif isinstance(value, list):
            if sorted(value) != sorted(dict2[key]):
                return False
        elif str(value) != str(dict2[key]):
                return False
    return True


def add_or_update_sys_template(templateParams, isSystemDefault=False):
    sysTemplate = None
    isCreated = False
    isUpdated = False
    planDisplayedName = templateParams.templateName
    if not planDisplayedName:
        return sysTemplate, isCreated, isUpdated

    planName = planDisplayedName.replace(' ', '_')
    currentTime = datetime.datetime.now()
    
    applicationGroup_objs = models.ApplicationGroup.objects.filter(name__iexact=templateParams.applicationGroup)

    applicationGroup_obj = None
    if applicationGroup_objs:
        applicationGroup_obj = applicationGroup_objs[0]

    sampleGrouping_obj = None
    if (templateParams.sampleGrouping):
        sampleGrouping_objs = models.SampleGroupType_CV.objects.filter(displayedName__iexact=templateParams.sampleGrouping)

        if sampleGrouping_objs:
            sampleGrouping_obj = sampleGrouping_objs[0]

    sysTemplate, isCreated = models.PlannedExperiment.objects.get_or_create(
        isSystemDefault=isSystemDefault, isSystem=True,
        isReusable=True, isPlanGroup=False,
        planDisplayedName=planDisplayedName, planName=planName,
        defaults={
            "planStatus": templateParams.planStatus,
            "runMode": "single",
            "isReverseRun": False,
            "planExecuted": False,
            "runType": templateParams.runType,
            "usePreBeadfind": templateParams.usePreBeadfind,
            "usePostBeadfind": templateParams.usePostBeadfind,
            "preAnalysis": True,
            "planPGM": "",
            "templatingKitName": templateParams.templatingKitName,
            "controlSequencekitname": templateParams.controlSequencekitname,
            "samplePrepKitName": templateParams.samplePrepKitName,
            "metaData": "",
            "date": currentTime,
            "applicationGroup": applicationGroup_obj,
            "sampleGrouping": sampleGrouping_obj,
            "categories": templateParams.categories,
            "templatingSize": templateParams.templatingSize,
            "libraryReadLength": templateParams.libraryReadLength,
            "samplePrepProtocol" : templateParams.samplePrepProtocol
        }
    )

    if isCreated:
        print "...Created System template.id=%d; name=%s; isSystemDefault=%s" % \
            (sysTemplate.id, sysTemplate.planDisplayedName, str(isSystemDefault))
    else:
        hasChanges = False
        if (sysTemplate.templatingSize != templateParams.templatingSize):
            print ">>> DIFF: orig sysTemplate.templatingSize=%s, new templatingSize=%s for system template.id=%d; name=%s" % \
                (sysTemplate.templatingSize, templateParams.templatingSize, sysTemplate.id, sysTemplate.planName)

            sysTemplate.templatingSize = templateParams.templatingSize
            hasChanges = True

        if (sysTemplate.libraryReadLength != templateParams.libraryReadLength):
            print ">>> DIFF: orig sysTemplate.libraryReadLength=%s for system template.id=%d; name=%s" % \
                (sysTemplate.libraryReadLength,
                 sysTemplate.id, sysTemplate.planName)

            sysTemplate.libraryReadLength = templateParams.libraryReadLength
            hasChanges = True

        if (sysTemplate.planStatus not in ["planned", "inactive"]):
            print ">>> DIFF: orig sysTemplate.planStatus=%s not supported for system template.id=%d; name=%s" % \
                (sysTemplate.planStatus,
                 sysTemplate.id, sysTemplate.planName)

            sysTemplate.planStatus = "planned"
            hasChanges = True
        else:
            if (sysTemplate.planStatus != templateParams.planStatus):
                print ">>> DIFF: orig sysTemplate.planStatus=%s for system template.id=%d; name=%s" % \
                    (sysTemplate.planStatus,
                     sysTemplate.id, sysTemplate.planName)

                sysTemplate.planStatus = templateParams.planStatus
                hasChanges = True

        if (sysTemplate.planExecuted):
            print ">>> DIFF: orig sysTemplate.planExecuted=%s for system template.id=%d; name=%s" % \
                (sysTemplate.planExecuted,
                 sysTemplate.id, sysTemplate.planName)

            sysTemplate.planExecuted = False
            hasChanges = True

        if (sysTemplate.runType != templateParams.runType):
            print ">>> DIFF: orig sysTemplate.runType=%s for system template.id=%d; name=%s" % \
                (sysTemplate.runType, sysTemplate.id,
                 sysTemplate.planName)

            sysTemplate.runType = templateParams.runType
            hasChanges = True

        if (sysTemplate.templatingKitName != templateParams.templatingKitName):
            print ">>> DIFF: orig sysTemplate.templatingKitName=%s for system template.id=%d; name=%s" % \
                (sysTemplate.templatingKitName, sysTemplate.id, sysTemplate.planName)

            sysTemplate.templatingKitName = templateParams.templatingKitName
            hasChanges = True

        if (sysTemplate.controlSequencekitname != templateParams.controlSequencekitname):
            print ">>> DIFF: orig sysTemplate.controlSequencekitname=%s for system template.id=%d; name=%s" % \
                (sysTemplate.controlSequencekitname, sysTemplate.id, sysTemplate.planName)

            sysTemplate.controlSequencekitname = templateParams.controlSequencekitname
            hasChanges = True

        if (sysTemplate.samplePrepKitName != templateParams.samplePrepKitName):
            print ">>> DIFF: orig sysTemplate.samplePrepKitName=%s for system template.id=%d; name=%s" % \
                (sysTemplate.samplePrepKitName, sysTemplate.id, sysTemplate.planName)

            sysTemplate.samplePrepKitName = templateParams.samplePrepKitName
            hasChanges = True

        if (sysTemplate.applicationGroup != applicationGroup_obj):
            print ">>> DIFF: orig sysTemplate.applicationGroup=%s for system template.id=%d; name=%s" % \
                (sysTemplate.applicationGroup, sysTemplate.id, sysTemplate.planName)

            sysTemplate.applicationGroup = applicationGroup_obj
            hasChanges = True

        if (sysTemplate.sampleGrouping != sampleGrouping_obj):
            print ">>> DIFF: orig sysTemplate.sampleGrouping=%s for system template.id=%d; name=%s" % \
                (sysTemplate.sampleGrouping, sysTemplate.id, sysTemplate.planName)

            sysTemplate.sampleGrouping = sampleGrouping_obj
            hasChanges = True

        if (sysTemplate.categories != templateParams.categories):
            print ">>> DIFF: orig sysTemplate.categories=%s new categories=%s for system template.id=%d; name=%s" % \
                (sysTemplate.categories, templateParams.categories, sysTemplate.id, sysTemplate.planName)

            sysTemplate.categories = templateParams.categories
            hasChanges = True

        if hasChanges:
            sysTemplate.date = currentTime
            sysTemplate.save()
            isUpdated = True

        if (sysTemplate.samplePrepProtocol != templateParams.samplePrepProtocol):
            print ">>>DIFF: orig sysTemplate.samplePrepProtocol=%s new samplePrepProtocol=%s for system template.id=%d; name=%s" % \
                (sysTemplate.samplePrepProtocol, templateParams.samplePrepProtocol, sysTemplate.id, sysTemplate.planName)

            sysTemplate.samplePrepProtocol = templateParams.samplePrepProtocol
            hasChanges = True

        if hasChanges:
            sysTemplate.date = currentTime
            sysTemplate.save()
            isUpdated = True
            
    if isUpdated:
        print "...Updated System template.id=%d; name=%s" % (sysTemplate.id, sysTemplate.planDisplayedName)

    if (not isCreated and not isUpdated):
        print "...No changes in plannedExperiment for System template.id=%d; name=%s" % (sysTemplate.id, sysTemplate.planDisplayedName)

    return sysTemplate, isCreated, isUpdated


def add_or_update_default_system_templates():

    # system default templates
    # 1
    templateParams = TemplateParams("Proton System Default Template", PROTON)
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams, isSystemDefault=True)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 2
    templateParams = TemplateParams("System Default Template", PGM)
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams, isSystemDefault=True)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 33
    templateParams = TemplateParams("S5 System Default Template", S5)
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams, isSystemDefault=True)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_ampliseq_system_templates():
    # ampliseq
    CATEGORIES = "onco_solidTumor"
    # 3
    templateParams = TemplateParams("Ion AmpliSeq Cancer Hotspot Panel v2", PGM, "AMPS")
    templateParams.update({
        "flows": 500,
        "libraryKitName": "Ion AmpliSeq 2.0 Library Kit",
        "reference": "hg19",
        "categories": CATEGORIES
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 4 - retired
    # templateParams = TemplateParams("Ion AmpliSeq Cancer Panel", PGM, "AMPS")

    # 5
    templateParams = TemplateParams("Ion AmpliSeq Cancer Panel 1_0 Lib Chem", PGM, "AMPS")
    templateParams.update({
        "flows": 500,
        "libraryKitName": "Ion AmpliSeq Kit",
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
    
    # 6
    templateParams = TemplateParams("Ion AmpliSeq Comprehensive Cancer Panel", PROTON, "AMPS")
    templateParams.update({
        "chipType": "P1.1.17",
        "flows": 360,
        "libraryKitName": "Ion AmpliSeq 2.0 Library Kit",
        "reference": "hg19",
        "targetRegionBedFile": "/hg19/unmerged/detail/4477685_CCP_bedfile_20120517.bed",
        "categories": CATEGORIES
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
    
    # 7
    templateParams = TemplateParams("Ion AmpliSeq Custom", PGM, "AMPS")
    templateParams.update({
        "flows": 500,
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
    
    # 8
    templateParams = TemplateParams("Ion AmpliSeq Custom ID", PGM, "AMPS")
    templateParams.update({
        "flows": 500,
        "reference": "hg19",
        "controlSequencekitname": "Ion AmpliSeq Sample ID Panel"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 9
    templateParams = TemplateParams("Ion AmpliSeq Inherited Disease Panel", PGM, "AMPS")
    templateParams.update({
        "chipType": "318",
        "flows": 500,
        "libraryKitName": "Ion AmpliSeq 2.0 Library Kit",
        "reference": "hg19",
        "categories": "inheritedDisease",
        "targetRegionBedFile": "/hg19/unmerged/detail/4477686_IDP_bedfile_20120613.bed",
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_ampliseq_rna_system_templates():
    # ampliseq RNA
    # 10
    templateParams = TemplateParams("Ion AmpliSeq RNA Panel", PGM, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "RNA",
        "chipType": "318",
        "flows": 500,
        "libraryKitName": "Ion AmpliSeq RNA Library Kit",
        "barcodeKitName": "IonXpress",
        "reference": "hg19_rna"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    
    # pre-select plugins
    plugins = {}
    plugins["coverageAnalysis"] = _get_plugin_dict("coverageAnalysis")

    thirdPartyPluginName = "PartekFlowUploader"
    plugins[thirdPartyPluginName] = _get_plugin_dict(thirdPartyPluginName)

    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_genericseq_system_templates():
    # generic sequencing
    # 11
    templateParams = TemplateParams("Ion PGM E_coli DH10B Control 200", PGM, "GENS")
    templateParams.update({
        "reference": "e_coli_dh10b"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 12
    templateParams = TemplateParams("Ion PGM E_coli DH10B Control 400", PGM, "GENS")
    templateParams.update({
        "chipType": "314",
        "flows": 850,
        "templatingKitName": "Ion PGM Template OT2 400 Kit",
        "sequencekitname": "IonPGM400Kit",
        "reference": "e_coli_dh10b"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 13
    templateParams = TemplateParams("Ion Proton Human CEPH Control 170", PROTON, "GENS")
    templateParams.update({
        "chipType": "P1.1.17",
        "flows": 440,
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 14
    templateParams = TemplateParams("System Generic Seq Template", PGM, "GENS")
    templateParams.update({
        "flows": 500,
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

 
def add_or_update_museek_system_templates():
    # MuSeek
    # 15
    templateParams = TemplateParams("MuSeek Barcoded Library", PGM, "GENS")
    templateParams.update({
        "chipType": "318",
        "flows": 500,
        "threePrimeAdapter": DEFAULT_MUSEEK_3_PRIME_ADAPTER_SEQUENCE,
        "libraryKitName": "MuSeek(tm) Library Preparation Kit",
        "barcodeKitName": "MuSeek Barcode set 1",
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 16
    templateParams = TemplateParams("MuSeek Library", PGM, "GENS")
    templateParams.update({
        "chipType": "318",
        "flows": 500,
        "threePrimeAdapter": DEFAULT_MUSEEK_3_PRIME_ADAPTER_SEQUENCE,
        "libraryKitName": "MuSeek(tm) Library Preparation Kit",
        "barcodeKitName": "MuSeek_5prime_tag",
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
    
    # 52
    templateParams = TemplateParams("Ion Xpress MuSeek Library", PGM, "GENS")
    templateParams.update({
        "threePrimeAdapter": "TGCACTGAAGCACACAATCACCGACTGCCC",
        "libraryKitName": "Ion Xpress MuSeek Library Preparation Kit",
        "barcodeKitName": "Ion Xpress MuSeek Barcode set 1",
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_rna_system_templates():

    # rna sequencing
    # 17
    templateParams = TemplateParams("Ion RNA - small", PGM, "RNA")
    templateParams.update({
        "applicationGroup": "RNA",
        "chipType": "318",
        "flows": 160,
        "libraryKitName": "Ion Total RNA Seq Kit v2",
        "barcodeKitName": "IonXpressRNA"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)

    # pre-select plugins
    plugins = {}

    thirdPartyPluginName = "PartekFlowUploader"
    plugins[thirdPartyPluginName] = _get_plugin_dict(thirdPartyPluginName)

    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)

    # 18
    templateParams = TemplateParams("Ion RNA - Whole Transcriptome", PROTON, "RNA")
    templateParams.update({
        "applicationGroup": "RNA",
        "chipType": "P1.1.17",
        "flows": 500,
        "libraryKitName": "Ion Total RNA Seq Kit v2",
        "barcodeKitName": "IonXpressRNA"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)

    # pre-select plugins
    plugins = {}
    plugins["RNASeqAnalysis"] = _get_plugin_dict("RNASeqAnalysis")

    thirdPartyPluginName = "PartekFlowUploader"
    plugins[thirdPartyPluginName] = _get_plugin_dict(thirdPartyPluginName)

    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_targetseq_system_templates():
    # targetSeq
    # 19
    templateParams = TemplateParams("Ion TargetSeq Custom", PGM, "TARS")
    templateParams.update({
        "chipType": "318",
        "flows": 500,
        "samplePrepKitName": "Ion TargetSeq(tm) Custom Enrichment Kit (100kb-500kb)",
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 20
    templateParams = TemplateParams("Ion TargetSeq Proton Exome", PROTON, "TARS")
    templateParams.update({
        "chipType": "P1.1.17",
        "flows": 440,
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_metagenomics_system_templates():
    # 16S
    # 21
    templateParams = TemplateParams("Ion 16S Metagenomics Template", PGM, "TARS_16S")
    templateParams.update({
        "applicationGroup": "Metagenomics",
        "chipType": "316v2",
        "flows": 850,
        "templatingKitName": "Ion PGM Template OT2 400 Kit",
        "sampleGrouping": "Self",
        "sequencekitname": "IonPGM400Kit",
        "libraryKitName": "IonPlusFragmentLibKit",
        "categories": "16s"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_oncomine_system_templates():
    # Oncomine
    CATEGORIES = "Oncomine;onco_solidTumor"
    # 22
    templateParams = TemplateParams("Oncomine Comprehensive DNA", PGM, "AMPS")
    templateParams.update({
        "chipType": "318",
        "flows": 400,
        "controlSequencekitname": "Ion AmpliSeq Sample ID Panel",
        "sampleGrouping": "Self",
        "categories": CATEGORIES,
        "libraryKitName": "Ion AmpliSeq 2.0 Library Kit",
        "barcodeKitName": "IonXpress",
        "reference": "hg19",
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 23
    templateParams = TemplateParams("Oncomine Comprehensive Fusions", PGM, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "chipType": "318",
        "flows": 400,
        "controlSequencekitname": "Ion AmpliSeq Sample ID Panel",
        "sampleGrouping": "Self",
        "categories": CATEGORIES,
        "libraryKitName": "Ion AmpliSeq RNA Library Kit",
        "barcodeKitName": "IonXpress",
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 24
    templateParams = TemplateParams("Oncomine Comprehensive DNA and Fusions", PGM, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "chipType": "318",
        "flows": 400,
        "controlSequencekitname": "Ion AmpliSeq Sample ID Panel",
        "sampleGrouping": "DNA and Fusions",
        "categories": CATEGORIES,
        "libraryKitName": "Ion AmpliSeq 2.0 Library Kit",
        "barcodeKitName": "IonXpress",
        "reference": "hg19",
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)

    # pre-select plugins
    plugins = {}
    plugins["coverageAnalysis"] = _get_plugin_dict("coverageAnalysis")

    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)

def add_or_update_onconet_system_templates():
    # OncoNetwork
    CATEGORIES = "Onconet;onco_solidTumor"
    # 25
    templateParams = TemplateParams("Ion AmpliSeq Colon and Lung Cancer Panel v2", PGM, "AMPS")
    templateParams.update({
        "chipType": "318",
        "flows": 400,
        "controlSequencekitname": "Ion AmpliSeq Sample ID Panel",
        "sampleGrouping": "Self",
        "categories": CATEGORIES,
        "libraryKitName": "Ion AmpliSeq 2.0 Library Kit",
        "barcodeKitName": "IonXpress",
        "reference": "hg19",
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 26
    templateParams = TemplateParams("Ion AmpliSeq RNA Lung Fusion Panel", PGM, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "chipType": "318",
        "flows": 400,
        "controlSequencekitname": "Ion AmpliSeq Sample ID Panel",
        "sampleGrouping": "Self",
        "categories": CATEGORIES,
        "libraryKitName": "Ion AmpliSeq RNA Library Kit",
        "barcodeKitName": "IonXpress",
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 27
    templateParams = TemplateParams("Ion AmpliSeq Colon Lung v2 with RNA Lung Fusion Panel", PGM, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "chipType": "318",
        "flows": 400,
        "controlSequencekitname": "Ion AmpliSeq Sample ID Panel",
        "sampleGrouping": "DNA and Fusions",
        "categories": CATEGORIES,
        "libraryKitName": "Ion AmpliSeq 2.0 Library Kit",
        "barcodeKitName": "IonXpress",
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 28
    templateParams = TemplateParams("Ion AmpliSeq Transcriptome Human Gene Expression Panel", PROTON, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "RNA",
        "chipType": "P1.1.17",
        "flows": 500,
        "templatingKitName": "Ion PI Template OT2 200 Kit v3",
        "libraryKitName": "Ion AmpliSeq Library Kit Plus",
        "barcodeKitName": "IonXpress",
        "reference": "hg19_AmpliSeq_Transcriptome_ERCC_v1",
        "targetRegionBedFile": "/hg19_AmpliSeq_Transcriptome_ERCC_v1/unmerged/detail/hg19_AmpliSeq_Transcriptome_ERCC_v1.bed",
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    
    # pre-select plugins
    plugins = {}
    plugins["ampliSeqRNA"] = _get_plugin_dict("ampliSeqRNA")

    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_ocp_focus_system_templates():
    CATEGORIES = "Oncomine;onco_solidTumor"
    # OCP Focus
    OCP_FOCUS_SEQ_KIT_NAME = "IonPGMSelectKit"
    OCP_FOCUS_LIB_KIT_NAME = "Ion PGM Select Library Kit"
    OCP_FOCUS_TEMPLATE_KIT_NAME = "Ion PGM OneTouch Select Template Kit"
    OCP_FOCUS_BARCODE_KIT_NAME = "Ion Select BC Set-1"
    # 29
    templateParams = TemplateParams("Oncomine Focus DNA", PGM, "AMPS")
    templateParams.update({
        "chipType": "318D",
        "flows": 400,
        "templatingKitName": OCP_FOCUS_TEMPLATE_KIT_NAME,
        "sampleGrouping": "Self",
        "categories": CATEGORIES,
        "sequencekitname": OCP_FOCUS_SEQ_KIT_NAME,
        "libraryKitName": OCP_FOCUS_LIB_KIT_NAME,
        "barcodeKitName": OCP_FOCUS_BARCODE_KIT_NAME,
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 30
    templateParams = TemplateParams("Oncomine Focus Fusions", PGM, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "chipType": "318D",
        "flows": 400,
        "templatingKitName": OCP_FOCUS_TEMPLATE_KIT_NAME,
        "sampleGrouping": "Self",
        "categories": CATEGORIES,
        "sequencekitname": OCP_FOCUS_SEQ_KIT_NAME,
        "libraryKitName": OCP_FOCUS_LIB_KIT_NAME,
        "barcodeKitName": OCP_FOCUS_BARCODE_KIT_NAME
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 31
    templateParams = TemplateParams("Oncomine Focus DNA and Fusions", PGM, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "chipType": "318D",
        "flows": 400,
        "templatingKitName": OCP_FOCUS_TEMPLATE_KIT_NAME,
        "sampleGrouping": "DNA and Fusions",
        "categories": CATEGORIES,
        "sequencekitname": OCP_FOCUS_SEQ_KIT_NAME,
        "libraryKitName": OCP_FOCUS_LIB_KIT_NAME,
        "barcodeKitName": OCP_FOCUS_BARCODE_KIT_NAME,
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_reproseq_system_templates():
    # ReproSeq
    DEFAULT_P1_3_PRIME_ADAPTER_SEQUENCE = "ATCACCGACTGCCCATAGAGAGGAAAGCGG"
    BARCODE_KIT_PGM = "Ion SingleSeq Barcode set 1-24"
    BARCODE_KIT_S5 = "Ion SingleSeq Barcode set 1-96"    
    CATEGORIES = "repro"
    LIBRARY_KIT = "IonPicoPlex"
    LIBRARY_READ_LENGTH = 0
    REFERENCE = "hg19"
    RUN_TYPE = "WGNM"
    
    # pre-select plugins
    plugins = {}
    plugins["FilterDuplicates"] = _get_plugin_dict("FilterDuplicates")

    # 32
    templateParams = TemplateParams("Ion ReproSeq Aneuploidy - Ion PGM System", PGM, RUN_TYPE)
    templateParams.update({
        "chipType": "318",
        "flows": 250,
        "templatingKitName": "Ion PGM Template IA Tech Access Kit",
        "sampleGrouping": "Self",
        "sequencekitname": "IonPGMHiQ",
        "libraryKitName": LIBRARY_KIT,
        "barcodeKitName": BARCODE_KIT_PGM,
        "reference": REFERENCE,        
        "threePrimeAdapter": DEFAULT_P1_3_PRIME_ADAPTER_SEQUENCE,
        "categories": CATEGORIES
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)

    #TODO- need finalized barcodeKit 
    # 33 is S5 System Default Template
    flowOrderList = models.FlowOrder.objects.filter(name="Ion samba.contradanza")
    PGS_S5_FLOWORDER = flowOrderList[0].flowOrder if flowOrderList else ""

    threePrimeAdapterList = models.ThreePrimeadapter.objects.filter(name = "Ion P1B")
    PGS_S5_3PrimeAdapter = threePrimeAdapterList[0].sequence if threePrimeAdapterList else ""

    templateParams = TemplateParams("Ion ReproSeq Aneuploidy - Ion S5 System", S5, RUN_TYPE)
    templateParams.update({
        "chipType": "530",
        "flows": 250,
        "flowOrder" : PGS_S5_FLOWORDER,        
        "templatingKitName": "Ion Chef PGS V1",
        "sampleGrouping": "Self",
        "sequencekitname": "Ion S5 ExT Sequencing Kit",
        "libraryKitName": LIBRARY_KIT,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "barcodeKitName": BARCODE_KIT_S5,
        "reference": REFERENCE,
        "threePrimeAdapter": PGS_S5_3PrimeAdapter,
        "categories": CATEGORIES
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)



def add_or_update_tagseq_system_templates():
    LIQUID_BIOPSY_PLUGIN_CONFIG_FILENAME = "rundb/fixtures/systemtemplateparams/tagseq_liquidbiopsy_parameters.json"
    TUMOR_PLUGIN_CONFIG_FILENAME = "rundb/fixtures/systemtemplateparams/tagseq_tumor_parameters.json"
   
    # 34 Lung Liquid Biopsy DNA
    sysTemplate, isCreated, isUpdated, templateParams = add_or_update_tagseq_system_template("Oncomine Lung Liquid Biopsy DNA")
    liquid_biopsy_plugins = create_tagseq_plugins(LIQUID_BIOPSY_PLUGIN_CONFIG_FILENAME)
    
    templateParams.selectedPlugins = liquid_biopsy_plugins
    finish_sys_template(sysTemplate, isCreated, templateParams, liquid_biopsy_plugins)

    # 35 Lung Tumor DNA
    sysTemplate, isCreated, isUpdated, templateParams = add_or_update_tagseq_system_template("Oncomine Lung Tumor DNA")
    tumor_plugins = create_tagseq_plugins(TUMOR_PLUGIN_CONFIG_FILENAME)
    
    templateParams.selectedPlugins = tumor_plugins
    finish_sys_template(sysTemplate, isCreated, templateParams, tumor_plugins)

    # 36 Breast Liquid Biopsy DNA
    sysTemplate, isCreated, isUpdated, templateParams = add_or_update_tagseq_system_template("Oncomine Breast Liquid Biopsy DNA")
    templateParams.selectedPlugins = liquid_biopsy_plugins    
    finish_sys_template(sysTemplate, isCreated, templateParams, liquid_biopsy_plugins)

    # 37 Breast Tumor DNA
    sysTemplate, isCreated, isUpdated, templateParams = add_or_update_tagseq_system_template("Oncomine Breast Tumor DNA")
    templateParams.selectedPlugins = tumor_plugins
    finish_sys_template(sysTemplate, isCreated, templateParams, tumor_plugins)

    # 38 Colon Liquid Biopsy DNA
    sysTemplate, isCreated, isUpdated, templateParams = add_or_update_tagseq_system_template("Oncomine Colon Liquid Biopsy DNA")
    templateParams.selectedPlugins = liquid_biopsy_plugins
    finish_sys_template(sysTemplate, isCreated, templateParams, liquid_biopsy_plugins)

    # 39 Colon Tumor DNA
    sysTemplate, isCreated, isUpdated, templateParams = add_or_update_tagseq_system_template("Oncomine Colon Tumor DNA")
    templateParams.selectedPlugins = tumor_plugins
    finish_sys_template(sysTemplate, isCreated, templateParams, tumor_plugins)


def add_or_update_tagseq_system_template(templateName):
    # Tag Sequencing
    TAG_SEQ_APPLICATION_GROUP = "onco_liquidBiopsy"
    TAG_SEQ_BARCODE_KIT_NAME = "TagSequencing"
    TAG_SEQ_CATEGORIES = "barcodes_8"
    TAG_SEQ_CHIP_NAME = "530"
    TAG_SEQ_FLOWS = 500
    TAG_SEQ_LIB_KIT_NAME = "Oncomine cfDNA Assay"    
    TAG_SEQ_LIBRARY_READ_LENGTH = 200
    TAG_SEQ_REFERENCE = "hg19"
    TAG_SEQ_RUN_TYPE = "TAG_SEQUENCING"
    TAG_SEQ_SAMPLE_GROUPING = "Self"    
    TAG_SEQ_SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    TAG_SEQ_TEMPLATE_KIT_NAME = "Ion Chef S530 V1"
    
    templateParams = TemplateParams(templateName, S5, TAG_SEQ_RUN_TYPE)
    templateParams.update({
        "applicationGroup" : TAG_SEQ_APPLICATION_GROUP,
        "barcodeKitName": TAG_SEQ_BARCODE_KIT_NAME,
        "categories": TAG_SEQ_CATEGORIES,
        "chipType": TAG_SEQ_CHIP_NAME,
        "flows": TAG_SEQ_FLOWS,
        "libraryKitName": TAG_SEQ_LIB_KIT_NAME,        
        "libraryReadLength" : TAG_SEQ_LIBRARY_READ_LENGTH,
        "reference": TAG_SEQ_REFERENCE,
        "sampleGrouping": TAG_SEQ_SAMPLE_GROUPING,
        "sequencekitname": TAG_SEQ_SEQ_KIT_NAME,
        "templatingKitName": TAG_SEQ_TEMPLATE_KIT_NAME
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    return sysTemplate, isCreated, isUpdated, templateParams


def create_tagseq_plugins(configFileName = None):  
    # pre-select plugins
    plugins = {}
    pluginUserInput = {}
    
    if configFileName:
        with open(configFileName) as f:
            data = json.load(f)
            if data:
                pluginUserInput = data

    plugins["variantCaller"] = _get_plugin_dict("variantCaller", pluginUserInput)

    return plugins


def add_or_update_oncomine_ocav1_system_templates():
    # OCAv1
    OCAV1_BARCODE_KIT_NAME = "IonXpress"    
    OCAV1_CATEGORIES = "Oncomine;onco_solidTumor"
    OCAV1_CHIP_NAME = "540"
    OCAV1_FLOWS = 400
    OCAV1_LIB_KIT_NAME = "Ion AmpliSeq 2.0 Library Kit"
    OCAV1_LIBRARY_READ_LENGTH = 200
    OCAV1_REFERENCE = "hg19"
    OCAV1_SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    OCAV1_TEMPLATE_KIT_NAME = "Ion Chef S540 V1"
    
    # 40 Oncomine Comprehensive v1 DNA for S5
    templateParams = TemplateParams("Oncomine Comprehensive v1 DNA for S5", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "DNA",
        "barcodeKitName": OCAV1_BARCODE_KIT_NAME,
        "categories": OCAV1_CATEGORIES,           
        "chipType": OCAV1_CHIP_NAME,
        "flows": OCAV1_FLOWS,
        "libraryKitName": OCAV1_LIB_KIT_NAME,  
        "libraryReadLength" : OCAV1_LIBRARY_READ_LENGTH,
        "reference": OCAV1_REFERENCE,
        "sampleGrouping": "Self",
        "sequencekitname": OCAV1_SEQ_KIT_NAME,
        "templatingKitName": OCAV1_TEMPLATE_KIT_NAME
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
        
    # 41 Oncomine Comprehensive v1 DNA and Fusions for S5
    templateParams = TemplateParams("Oncomine Comprehensive v1 DNA and Fusions for S5", S5, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": OCAV1_BARCODE_KIT_NAME,
        "categories": OCAV1_CATEGORIES, 
        "chipType": OCAV1_CHIP_NAME,
        "flows": OCAV1_FLOWS,
        "libraryKitName": OCAV1_LIB_KIT_NAME,  
        "libraryReadLength" : OCAV1_LIBRARY_READ_LENGTH,
        "reference": OCAV1_REFERENCE,
        "sampleGrouping": "DNA and Fusions",
        "sequencekitname": OCAV1_SEQ_KIT_NAME,
        "templatingKitName": OCAV1_TEMPLATE_KIT_NAME
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_oncomine_ocav2_system_templates():
    # OCAv2
    OCAV2_BARCODE_KIT_NAME = "IonXpress"    
    OCAV2_CATEGORIES = "Oncomine;ocav2;onco_solidTumor"
    OCAV2_CHIP_NAME = "540"
    OCAV2_FLOWS = 400
    OCAV2_LIB_KIT_NAME = "Ion AmpliSeq 2.0 Library Kit"
    OCAV2_LIBRARY_READ_LENGTH = 200
    OCAV2_REFERENCE = "hg19"
    OCAV2_SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    OCAV2_TEMPLATE_KIT_NAME = "Ion Chef S540 V1"
        
    # pre-select plugins
    plugins = {}
    plugins["coverageAnalysis"] = _get_plugin_dict("coverageAnalysis")
    
    # 42  Oncomine Comprehensive v2 DNA and Fusions for S5
    templateParams = TemplateParams("Oncomine Comprehensive v2 DNA and Fusions for S5", S5, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": OCAV2_BARCODE_KIT_NAME,
        "categories": OCAV2_CATEGORIES, 
        "chipType": OCAV2_CHIP_NAME,
        "flows": OCAV2_FLOWS,
        "libraryKitName": OCAV2_LIB_KIT_NAME,  
        "libraryReadLength" : OCAV2_LIBRARY_READ_LENGTH,
        "reference": OCAV2_REFERENCE,
        "sampleGrouping": "DNA and Fusions",
        "sequencekitname": OCAV2_SEQ_KIT_NAME,
        "templatingKitName": OCAV2_TEMPLATE_KIT_NAME
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_oncomine_ocav3_system_templates():
    # OCAv3
    OCAV3_BARCODE_KIT_NAME = "IonXpress"    
    OCAV3_CATEGORIES = "Oncomine;onco_solidTumor"
    OCAV3_CATEGORIES_2 = "Oncomine;barcodes_16;onco_solidTumor"    
    OCAV3_CHIP_NAME = "540"
    OCAV3_FLOWS = 400
    OCAV3_FUSION_LIB_KIT_NAME = "Ion AmpliSeq Library Kit Plus"
    OCAV3_LIB_KIT_NAME = "Ion AmpliSeq Library Kit Plus"
    OCAV3_LIBRARY_READ_LENGTH = 200
    OCAV3_REFERENCE = "hg19"
    OCAV3_SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    OCAV3_TEMPLATE_KIT_NAME = "Ion Chef S540 V1"
    OCAV3_STATUS = "planned"
        
    # pre-select plugins
    plugins = {}
    plugins["coverageAnalysis"] = _get_plugin_dict("coverageAnalysis")

    # 43 Oncomine Comprehensive v3 DNA
    templateParams = TemplateParams("Oncomine Comprehensive v3 DNA", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "DNA",
        "barcodeKitName": OCAV3_BARCODE_KIT_NAME,
        "categories": OCAV3_CATEGORIES,           
        "chipType": OCAV3_CHIP_NAME,
        "flows": OCAV3_FLOWS,
        "libraryKitName": OCAV3_LIB_KIT_NAME,  
        "libraryReadLength" : OCAV3_LIBRARY_READ_LENGTH,
        "reference": OCAV3_REFERENCE,
        "sampleGrouping": "Self",
        "sequencekitname": OCAV3_SEQ_KIT_NAME,
        "templatingKitName": OCAV3_TEMPLATE_KIT_NAME,
        "planStatus" : OCAV3_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)
        
    # 44 Oncomine Comprehensive v3 DNA and Fusions
    templateParams = TemplateParams("Oncomine Comprehensive v3 DNA and Fusions", S5, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": OCAV3_BARCODE_KIT_NAME,
        "categories": OCAV3_CATEGORIES_2, 
        "chipType": OCAV3_CHIP_NAME,
        "flows": OCAV3_FLOWS,
        "libraryKitName": OCAV3_LIB_KIT_NAME,  
        "libraryReadLength" : OCAV3_LIBRARY_READ_LENGTH,
        "reference": OCAV3_REFERENCE,
        "sampleGrouping": "DNA and Fusions",
        "sequencekitname": OCAV3_SEQ_KIT_NAME,
        "templatingKitName": OCAV3_TEMPLATE_KIT_NAME,
        "planStatus" : OCAV3_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)
                
    # 45 Oncomine Comprehensive v3 Fusions
    templateParams = TemplateParams("Oncomine Comprehensive v3 Fusions", S5, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": OCAV3_BARCODE_KIT_NAME,
        "categories": OCAV3_CATEGORIES, 
        "chipType": OCAV3_CHIP_NAME,
        "flows": OCAV3_FLOWS,
        "libraryKitName": OCAV3_FUSION_LIB_KIT_NAME,  
        "libraryReadLength" : OCAV3_LIBRARY_READ_LENGTH,
        "reference": "",
        "sampleGrouping": "Self",
        "sequencekitname": OCAV3_SEQ_KIT_NAME,
        "templatingKitName": OCAV3_TEMPLATE_KIT_NAME,
        "planStatus" : OCAV3_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_oncomine_pediatric_system_templates():
    # pediatric
    BARCODE_KIT_NAME = "IonXpress"    
    CATEGORIES = "onco_solidTumor;onco_heme"
    CHIP_NAME = "540"
    FLOWS = 400
    FUSION_LIB_KIT_NAME = "Ion AmpliSeq RNA Library Kit"
    LIB_KIT_NAME = "Ion AmpliSeq 2.0 Library Kit"
    LIBRARY_READ_LENGTH = 200
    REFERENCE = "hg19"
    SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    TEMPLATE_KIT_NAME = "Ion Chef S540 V1"
    PLAN_STATUS = "inactive"
    
    # 46 AmpliSeq Pediatric Cancer Research Panel DNA
    templateParams = TemplateParams("AmpliSeq Pediatric Cancer Research Panel DNA", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "DNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,  
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus" : PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
        
    # 47 AmpliSeq Pediatric Cancer Research Panel DNA and Fusion
    templateParams = TemplateParams("AmpliSeq Pediatric Cancer Research Panel DNA and Fusion", S5, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES, 
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,  
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "DNA and Fusions",
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus" : PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
                
    # 48 AmpliSeq Pediatric Cancer Research Panel Fusion
    templateParams = TemplateParams("AmpliSeq Pediatric Cancer Research Panel Fusion", S5, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES, 
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "libraryKitName": FUSION_LIB_KIT_NAME,  
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus" : PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_oncomine_BRCA_system_templates():
    # BRCA
    BARCODE_KIT_NAME = "IonXpress"
    CATEGORIES = "Oncomine;onco_solidTumor;inheritedDisease"
    CHIP_NAME_PGM = "318"
    CHIP_NAME_S5 = "530"
    FLOWS = 500
    LIB_KIT_NAME = "Ion AmpliSeq Library Kit Plus"
    LIBRARY_READ_LENGTH = 200
    REFERENCE = "hg19"
    SEQ_KIT_NAME_PGM = "IonPGMHiQ"
    SEQ_KIT_NAME_S5 = "Ion S5 Sequencing Kit"
    TEMPLATE_KIT_NAME_PGM = "Ion PGM Hi-Q View Chef Kit"
    TEMPLATE_KIT_NAME_S5 = "Ion Chef S530 V1"

    # 49 Oncomine BRCA for PGM
    templateParams = TemplateParams("Oncomine BRCA Research for PGM", PGM, "AMPS")
    templateParams.update({
        "applicationGroup" : "DNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_PGM,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,  
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "sequencekitname": SEQ_KIT_NAME_PGM,
        "templatingKitName": TEMPLATE_KIT_NAME_PGM
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
    
    # 50 Oncomine BRCA for S5
    templateParams = TemplateParams("Oncomine BRCA Research for S5", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "DNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_S5,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,  
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "sequencekitname": SEQ_KIT_NAME_S5,
        "templatingKitName": TEMPLATE_KIT_NAME_S5
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_immune_response_system_templates():
    PLAN_STATUS = "planned"
    CATEGORIES = "onco_immune"
    PLUGIN = "immuneResponseRNA"
    REFERENCE = "ImmuneResponse_v3.1"
    BEDFILE = "/%s/unmerged/detail/ImmuneResponse_v3.1_target_designed_20160908.bed" % REFERENCE

    # 51 Immune Response Panel S5
    templateParams = TemplateParams("Oncomine Immune Response Research Assay for S5 with Chef", S5, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "RNA",
        "barcodeKitName": "IonCode Barcodes 1-32",
        "categories": CATEGORIES,
        "chipType": "530",
        "flows": 500,
        "libraryKitName": "Ampliseq DNA V1",
        "libraryReadLength" : 200,
        "reference": REFERENCE,
        "targetRegionBedFile": BEDFILE,
        "sampleGrouping": "Self",
        "sequencekitname": "Ion S5 Sequencing Kit",
        "templatingKitName": "Ion Chef S530 V1",
        "planStatus" : PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)

    # pre-select plugins
    plugins = {}
    pluginUserInput = {}
    plugins[PLUGIN] = _get_plugin_dict(PLUGIN, pluginUserInput)

    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)

    #67 Immune Response Panel PGM
    templateParams = TemplateParams("Oncomine Immune Response Research Assay for PGM with OT2", PGM, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "RNA",
        "barcodeKitName": "IonXpress",
        "categories": CATEGORIES,        
        "chipType": "318",
        "flows": 500,
        "libraryKitName": "Ion AmpliSeq 2.0 Library Kit",
        "reference": REFERENCE,
        "targetRegionBedFile": BEDFILE,
        "sequencekitname": "IonPGMHiQView",
        "templatingKitName": "Ion PGM Hi-Q View OT2 Kit - 200",
        "planStatus" : PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_S5_ocp_focus_system_templates():
    # OCP Focus
    OCP_S5_FOCUS_CATEGORIES = "Oncomine;onco_solidTumor"
    OCP_S5_FOCUS_CHIP = "520"
    OCP_S5_FOCUS_FLOWS = 400
    OCP_S5_FOCUS_LIBRARY_KIT_NAME = "Ion AmpliSeq 2.0 Library Kit"
    OCP_S5_FOCUS_TEMPLATE_KIT_NAME = "Ion Chef S530 V1"
    OCP_S5_FOCUS_BARCODE_KIT_NAME = "IonXpress"

    # 53
    templateParams = TemplateParams("Oncomine Focus DNA for S5", S5, "AMPS")
    templateParams.update({
        "chipType": OCP_S5_FOCUS_CHIP,
        "categories": OCP_S5_FOCUS_CATEGORIES,
        "flows": OCP_S5_FOCUS_FLOWS,
        "libraryKitName": OCP_S5_FOCUS_LIBRARY_KIT_NAME,
        "templatingKitName": OCP_S5_FOCUS_TEMPLATE_KIT_NAME,
        "sampleGrouping": "Self",
        "barcodeKitName": OCP_S5_FOCUS_BARCODE_KIT_NAME,
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 54
    templateParams = TemplateParams("Oncomine Focus Fusions for S5", S5, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "categories": OCP_S5_FOCUS_CATEGORIES,        
        "chipType": OCP_S5_FOCUS_CHIP,
        "flows": OCP_S5_FOCUS_FLOWS,
        "libraryKitName": OCP_S5_FOCUS_LIBRARY_KIT_NAME,
        "templatingKitName": OCP_S5_FOCUS_TEMPLATE_KIT_NAME,
        "sampleGrouping": "Self",
        "categories": OCP_S5_FOCUS_CATEGORIES,
        "barcodeKitName": OCP_S5_FOCUS_BARCODE_KIT_NAME
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    # 55
    templateParams = TemplateParams("Oncomine Focus DNA and Fusions for S5", S5, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "categories": OCP_S5_FOCUS_CATEGORIES,        
        "chipType": OCP_S5_FOCUS_CHIP,
        "flows": OCP_S5_FOCUS_FLOWS,
        "libraryKitName": OCP_S5_FOCUS_LIBRARY_KIT_NAME,
        "templatingKitName": OCP_S5_FOCUS_TEMPLATE_KIT_NAME,
        "sampleGrouping": "DNA and Fusions",
        "barcodeKitName": OCP_S5_FOCUS_BARCODE_KIT_NAME,
        "reference": "hg19"
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_hid_system_templates():
    # HID
    default_flowOrderList = models.FlowOrder.objects.filter(name="Ion samba")
    flowOrderList = models.FlowOrder.objects.filter(name="Ion samba.gafieira")
    samplePrepProtocolList = models.common_CV.objects.filter(cv_type = "samplePrepProtocol", uid="CV0001")
        
    BARCODE_KIT_NAME = ""
    CATEGORIES = ""
    CHIP_NAME_PGM = "318"
    CHIP_NAME_S5 = "530"
    FLOWS_STR = 850
    FLOWS = 500
    FLOWORDER_STR = flowOrderList[0].flowOrder if flowOrderList else ""
    FLOWORDER_DEFAULT = default_flowOrderList[0].flowOrder if default_flowOrderList else ""
    LIB_KIT_NAME = "Precision ID Library Kit"
    LIBRARY_READ_LENGTH = 200
    REFERENCE = "hg19"
    REFERENCE_mtDNA = "PrecisionID_mtDNA_rCRS"
    SAMPLE_PREP_PROTOCOL_STR = samplePrepProtocolList[0].value if samplePrepProtocolList else ""
    SAMPLE_PREP_PROTOCOL = ""
    SEQ_KIT_NAME_PGM = "IonPGMHiQ"
    SEQ_KIT_NAME_S5 = "Ion S5 Sequencing Kit"
    TEMPLATE_KIT_NAME_PGM = "Ion PGM Hi-Q Chef Kit"
    TEMPLATE_KIT_NAME_S5 = "Ion Chef S530 V1"
    TEMPLATING_SIZE_PGM = "200"
    TEMPLATING_SIZE_S5 = None
    
    # 56 HID STR
    templateParams = TemplateParams("Applied Biosystems Precision ID GlobalFiler NGS STR Panel - PGM", PGM, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_PGM,
        "flows": FLOWS_STR,
        "flowOrder" : FLOWORDER_STR,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL_STR,
        "sequencekitname": SEQ_KIT_NAME_PGM,
        "templatingKitName": TEMPLATE_KIT_NAME_PGM,
        "templatingSize": TEMPLATING_SIZE_PGM
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
    
    # 57 HID identity PGM
    templateParams = TemplateParams("Applied Biosystems Precision ID Identity Panel - PGM", PGM, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_PGM,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME_PGM,
        "templatingKitName": TEMPLATE_KIT_NAME_PGM,
        "templatingSize": TEMPLATING_SIZE_PGM
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
        
    # 58 HID identity S5    
    templateParams = TemplateParams("Applied Biosystems Precision ID Identity Panel - S5", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_S5,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME_S5,
        "templatingKitName": TEMPLATE_KIT_NAME_S5,
        "templatingSize": TEMPLATING_SIZE_S5
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)

    
    # 59 HID ancestry PGM    
    templateParams = TemplateParams("Applied Biosystems Precision ID Ancestry Panel - PGM", PGM, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_PGM,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME_PGM,
        "templatingKitName": TEMPLATE_KIT_NAME_PGM,
        "templatingSize": TEMPLATING_SIZE_PGM
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
        
    # 60 HID ancestry S5    
    templateParams = TemplateParams("Applied Biosystems Precision ID Ancestry Panel - S5", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_S5,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME_S5,
        "templatingKitName": TEMPLATE_KIT_NAME_S5,
        "templatingSize": TEMPLATING_SIZE_S5
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


    # 61 HID mito whole genome PGM    
    templateParams = TemplateParams("Applied Biosystems Precision ID mtDNA Whole Genome Panel - PGM", PGM, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_PGM,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE_mtDNA,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME_PGM,
        "templatingKitName": TEMPLATE_KIT_NAME_PGM,
        "templatingSize": TEMPLATING_SIZE_PGM
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
        
    # 62 HID mito whole genome S5    
    templateParams = TemplateParams("Applied Biosystems Precision ID mtDNA Whole Genome Panel - S5", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_S5,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE_mtDNA,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME_S5,
        "templatingKitName": TEMPLATE_KIT_NAME_S5,
        "templatingSize": TEMPLATING_SIZE_S5
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


    # 63 HID mito control region PGM    
    templateParams = TemplateParams("Applied Biosystems Precision ID mtDNA Control Region Panel - PGM", PGM, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_PGM,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE_mtDNA,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME_PGM,
        "templatingKitName": TEMPLATE_KIT_NAME_PGM,
        "templatingSize": TEMPLATING_SIZE_PGM
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
        
    # 64 HID mito control region S5    
    templateParams = TemplateParams("Applied Biosystems Precision ID mtDNA Control Region Panel - S5", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME_S5,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE_mtDNA,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME_S5,
        "templatingKitName": TEMPLATE_KIT_NAME_S5,
        "templatingSize": TEMPLATING_SIZE_S5
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_hid_dexter_system_templates():
    # HID
    default_flowOrderList = models.FlowOrder.objects.filter(name="Ion samba HID2")
        
    BARCODE_KIT_NAME = "IonCode"
    CATEGORIES = ""
    CHIP_NAME = "530"
    FLOWS = 650
    FLOWORDER_DEFAULT = default_flowOrderList[0].flowOrder if default_flowOrderList else ""
    LIB_KIT_NAME = "Ion Chef HID Library V2"
    LIBRARY_READ_LENGTH = 200
    REFERENCE = "hg19"
    SAMPLE_PREP_PROTOCOL = ""
    SEQ_KIT_NAME = "precisionIDS5Kit"
    TEMPLATE_KIT_NAME = "Ion Chef HID S530 V2"
    TEMPLATING_SIZE = None
    
    # 65 HID STR
    templateParams = TemplateParams("Applied Biosystems Precision ID GlobalFiler Mixture ID Panel - S5", PGM, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "templatingSize": TEMPLATING_SIZE
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)
    
    # 66 HID identity PGM
    templateParams = TemplateParams("Applied Biosystems Precision ID GlobalFiler STR Panel - S5", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "HID",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,           
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "flowOrder" : FLOWORDER_DEFAULT,
        "libraryKitName": LIB_KIT_NAME,
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "Self",
        "samplePrepProtocol" : SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "templatingSize": TEMPLATING_SIZE
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_ocp_myeloid_pgm_system_templates():    
    BARCODE_KIT_NAME = "IonXpress"
    CATEGORIES_DNA = "barcodes_6;onco_heme"
    CATEGORIES_RNA_FUSIONS = "barcodes_24;onco_heme"
    CATEGORIES_DNA_n_FUSIONS = "barcodes_12;onco_heme"         
    CHIP_NAME = "318"
    FLOWS = 850
    LIB_KIT_NAME = "Ion AmpliSeq Library Kit Plus"
    LIBRARY_READ_LENGTH = 400
    REFERENCE = "hg19"
    SAMPLE_GROUPING = "Self"
    SAMPLE_PREP_PROTOCOL = ""
    SEQ_KIT_NAME = "IonPGMHiQView"
    TEMPLATE_KIT_NAME = "Ion PGM Hi-Q View Chef Kit"
    ##TEMPLATING_SIZE = "400"
    PLAN_STATUS = "inactive"

    # pre-select plugins
    plugins = {}
    plugins["coverageAnalysis"] = _get_plugin_dict("coverageAnalysis")

    # 67
    templateParams = TemplateParams("AmpliSeq Myeloid Research DNA for PGM", PGM, "AMPS")
    templateParams.update({
        "applicationGroup" : "DNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES_DNA,
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,        
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)    

    # 68
    templateParams = TemplateParams("AmpliSeq Myeloid Research Fusions for PGM", PGM, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES_RNA_FUSIONS,
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,        
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)

    # 69
    templateParams = TemplateParams("AmpliSeq Myeloid Research DNA and Fusions for PGM", PGM, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES_DNA_n_FUSIONS,
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,        
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "DNA and Fusions",
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_ocp_myeloid_s5_system_templates():    
    BARCODE_KIT_NAME = "IonXpress"
    CATEGORIES_DNA = "barcodes_12;onco_heme;chef_myeloid_protocol"
    CATEGORIES_RNA_FUSIONS = "barcodes_48;onco_heme;chef_myeloid_protocol"
    CATEGORIES_DNA_n_FUSIONS = "barcodes_24;onco_heme;chef_myeloid_protocol"
    CHIP_NAME = "530"
    FLOWS = 850
    LIB_KIT_NAME = "Ion AmpliSeq Library Kit Plus"
    LIBRARY_READ_LENGTH = 400
    REFERENCE = "hg19"
    SAMPLE_GROUPING = "Self"
    SAMPLE_PREP_PROTOCOL = "denature30_cycles45_20"
    SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    TEMPLATE_KIT_NAME = "Ion Chef S530 V2"
    ##TEMPLATING_SIZE = "400"
    PLAN_STATUS = "inactive"

    # pre-select plugins
    plugins = {}
    plugins["coverageAnalysis"] = _get_plugin_dict("coverageAnalysis")

    # 70
    templateParams = TemplateParams("AmpliSeq Myeloid Research DNA for S5", S5, "AMPS")
    templateParams.update({
        "applicationGroup" : "DNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES_DNA,
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,        
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "samplePrepProtocol": SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)

    # 71
    templateParams = TemplateParams("AmpliSeq Myeloid Research Fusions for S5", S5, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES_RNA_FUSIONS,
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,        
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "samplePrepProtocol": SAMPLE_PREP_PROTOCOL,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)

    # 72
    templateParams = TemplateParams("AmpliSeq Myeloid Research DNA and Fusions for S5", S5, "AMPS_DNA_RNA")
    templateParams.update({
        "applicationGroup": "DNA + RNA",
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES_DNA_n_FUSIONS,
        "chipType": CHIP_NAME,
        "flows": FLOWS,
        "libraryKitName": LIB_KIT_NAME,        
        "libraryReadLength" : LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": "DNA and Fusions",
        "samplePrepProtocol": SAMPLE_PREP_PROTOCOL,         
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_proton_PQ_system_template():
    CATEGORIES = ""
    CHIP = "P2.2.2"
    FLOWS = 150
    LIBRARY_KIT_NAME = "Ion Xpress Plus Fragment Library Kit"
    TEMPLATE_KIT_NAME = "Ion PQ Template OT2 Kit"
    SEQ_KIT_NAME = "IonProtonPQKit"
    BARCODE_KIT_NAME = "IonXpress"
    PLAN_STATUS = "inactive"
    
    # 73
    templateParams = TemplateParams("Ion NIPT template - PQ", PROTON, "WGNM")
    templateParams.update({
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,
        "chipType": CHIP,
        "flows": FLOWS,
        "libraryKitName": LIBRARY_KIT_NAME,
        "reference": "hg19",
        "sampleGrouping": "Self",
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_mouse_transcriptome_s5_system_templates():
    APPLICATION_GROUP = "RNA"
    BARCODE_KIT_NAME = "IonCode Barcodes 1-32"
    BARCODE_KIT_NAME_MANUAL = "IonXpress"    
    CATEGORIES = ""
    CHIP = "540"
    FLOWS = 500
    LIBRARY_KIT_NAME = "Ampliseq DNA V1"
    LIBRARY_KIT_NAME_MANUAL = "Ion AmpliSeq Library Kit Plus"
    LIBRARY_READ_LENGTH = 200
    REFERENCE = "mm10"
    SAMPLE_GROUPING = "Self"
    SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    TEMPLATE_KIT_NAME = "Ion Chef S540 V1"
    PLAN_STATUS = "inactive"

    # pre-select plugins
    plugins = {}
    plugins["ampliSeqRNA"] = _get_plugin_dict("ampliSeqRNA")

    # 74
    templateParams = TemplateParams("Ion AmpliSeq Transcriptome Mouse Gene Expression Chef-S5", S5, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": APPLICATION_GROUP,
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,
        "chipType": CHIP,
        "flows": FLOWS,
        "libraryKitName": LIBRARY_KIT_NAME,
        "libraryReadLength": LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)

    # 75
    templateParams = TemplateParams("Ion AmpliSeq Transcriptome Mouse Gene Expression Manual Chef-S5", S5, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": APPLICATION_GROUP,                           
        "barcodeKitName": BARCODE_KIT_NAME_MANUAL,
        "categories": CATEGORIES,
        "chipType": CHIP,
        "flows": FLOWS,
        "libraryKitName": LIBRARY_KIT_NAME_MANUAL,
        "libraryReadLength": LIBRARY_READ_LENGTH,      
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_mouse_transcriptome_proton_system_templates():
    APPLICATION_GROUP = "RNA"
    BARCODE_KIT_NAME = "IonXpress"    
    CATEGORIES = ""
    CHIP = "P1.1.17"
    FLOWS = 500
    LIBRARY_KIT_NAME = "Ion AmpliSeq Library Kit Plus"
    REFERENCE = "mm10"
    SAMPLE_GROUPING = "Self"
    SEQ_KIT_NAME = "    ProtonI200Kit-v3"
    TEMPLATE_KIT_NAME = "Ion PI Template OT2 200 Kit v3"
    PLAN_STATUS = "inactive"
                
    # pre-select plugins
    plugins = {}
    plugins["ampliSeqRNA"] = _get_plugin_dict("ampliSeqRNA")

    # 76
    templateParams = TemplateParams("Ion AmpliSeq Transcriptome Mouse Gene Expression Panel OT2-Proton", PROTON, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": APPLICATION_GROUP,                          
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,
        "chipType": CHIP,
        "flows": FLOWS,
        "libraryKitName": LIBRARY_KIT_NAME,
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_mutation_load_s5_system_templates():
    APPLICATION_GROUP = "DNA"
    BARCODE_KIT_NAME = "IonCode Barcodes 1-32" 
    CATEGORIES = "onco_immune"
    CHIP = "540"
    FLOWS = 400
    LIBRARY_KIT_NAME = "Ion AmpliSeq Library Kit Plus"
    LIBRARY_READ_LENGTH = 200
    REFERENCE = "hg19"
    SAMPLE_GROUPING = "Self"
    SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    TEMPLATE_KIT_NAME = "Ion Chef S540 V1"
    PLAN_STATUS = "inactive"

    # 77
    templateParams = TemplateParams("Oncomine Mutation Load", S5, "AMPS")
    templateParams.update({
        "applicationGroup": APPLICATION_GROUP,
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,
        "chipType": CHIP,
        "flows": FLOWS,
        "libraryKitName": LIBRARY_KIT_NAME,
        "libraryReadLength": LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams)


def add_or_update_tagseq_cfdna_system_templates():
    LIQUID_BIOPSY_PLUGIN_CONFIG_FILENAME = "rundb/fixtures/systemtemplateparams/tagseq_liquidbiopsy_parameters.json"
    TUMOR_PLUGIN_CONFIG_FILENAME = "rundb/fixtures/systemtemplateparams/tagseq_tumor_parameters.json"
   
    # 78
    sysTemplate, isCreated, isUpdated, templateParams = add_or_update_tagseq_cfdna_s5_chef_system_template("Oncomine TagSeq Tumor") 
    tumor_plugins = create_tagseq_plugins(TUMOR_PLUGIN_CONFIG_FILENAME)
    templateParams.selectedPlugins = tumor_plugins
    finish_sys_template(sysTemplate, isCreated, templateParams, tumor_plugins)
   
    # 79
    sysTemplate, isCreated, isUpdated, templateParams = add_or_update_tagseq_cfdna_s5_chef_system_template("Oncomine TagSeq Liquid Biopsy")
    liquid_biopsy_plugins = create_tagseq_plugins(LIQUID_BIOPSY_PLUGIN_CONFIG_FILENAME)
    templateParams.selectedPlugins = liquid_biopsy_plugins
    finish_sys_template(sysTemplate, isCreated, templateParams, liquid_biopsy_plugins)



def add_or_update_tagseq_cfdna_s5_chef_system_template(templateName):
    # Tag Sequencing
    TAG_SEQ_APPLICATION_GROUP = "onco_liquidBiopsy"
    TAG_SEQ_BARCODE_KIT_NAME = "TagSequencing"
    TAG_SEQ_CATEGORIES = "barcodes_8"
    TAG_SEQ_CHIP_NAME = "530"
    TAG_SEQ_FLOWS = 500
    TAG_SEQ_LIB_KIT_NAME = "Oncomine cfDNA Assay"    
    TAG_SEQ_LIBRARY_READ_LENGTH = 200
    TAG_SEQ_REFERENCE = "hg19"
    TAG_SEQ_RUN_TYPE = "TAG_SEQUENCING"
    TAG_SEQ_SAMPLE_GROUPING = "Self"    
    TAG_SEQ_SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    TAG_SEQ_TEMPLATE_KIT_NAME = "Ion Chef S530 V1"
    PLAN_STATUS = "planned"
    
    templateParams = TemplateParams(templateName, S5, TAG_SEQ_RUN_TYPE)
    templateParams.update({
        "applicationGroup" : TAG_SEQ_APPLICATION_GROUP,
        "barcodeKitName": TAG_SEQ_BARCODE_KIT_NAME,
        "categories": TAG_SEQ_CATEGORIES,
        "chipType": TAG_SEQ_CHIP_NAME,
        "flows": TAG_SEQ_FLOWS,
        "libraryKitName": TAG_SEQ_LIB_KIT_NAME,        
        "libraryReadLength" : TAG_SEQ_LIBRARY_READ_LENGTH,
        "reference": TAG_SEQ_REFERENCE,
        "sampleGrouping": TAG_SEQ_SAMPLE_GROUPING,
        "sequencekitname": TAG_SEQ_SEQ_KIT_NAME,
        "templatingKitName": TAG_SEQ_TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    return sysTemplate, isCreated, isUpdated, templateParams


def add_or_update_immune_repertoire_s5_system_templates():
    APPLICATION_GROUP = "immune_repertoire"
    BARCODE_KIT_NAME = "IonXpress"    
    CATEGORIES = "onco_immune"
    CHIP = "530"
    FLOWS = 800
    LIBRARY_KIT_NAME = "Ion AmpliSeq Library Kit Plus"
    LIBRARY_READ_LENGTH = 400
    REFERENCE = ""
    SAMPLE_GROUPING = "Self"
    SEQ_KIT_NAME = "Ion S5 Sequencing Kit"
    TEMPLATE_KIT_NAME = "Ion Chef S530 V2"
    PLAN_STATUS = "inactive"
            
    # pre-select plugins
    plugins = {}
    plugins["TCR_Repertoire"] = _get_plugin_dict("TCR_Repertoire")

    # 80
    templateParams = TemplateParams("Oncomine Immune Repertoire TCRB Assay - S5", S5, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": APPLICATION_GROUP,
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,
        "chipType": CHIP,
        "flows": FLOWS,
        "libraryKitName": LIBRARY_KIT_NAME,
        "libraryReadLength": LIBRARY_READ_LENGTH,        
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


def add_or_update_immune_repertoire_pgm_system_templates():
    APPLICATION_GROUP = "immune_repertoire"
    BARCODE_KIT_NAME = "IonXpress"    
    CATEGORIES = "onco_immune"
    CHIP = "318"
    FLOWS = 800
    LIBRARY_KIT_NAME = "Ion AmpliSeq Library Kit Plus"
    LIBRARY_READ_LENGTH = 400
    REFERENCE = ""
    SAMPLE_GROUPING = "Self"
    SEQ_KIT_NAME = "IonPGMHiQView"
    TEMPLATE_KIT_NAME = "Ion PGM Hi-Q View Chef Kit"
    PLAN_STATUS = "inactive"
                
    # pre-select plugins
    plugins = {}
    plugins["TCR_Repertoire"] = _get_plugin_dict("TCR_Repertoire")

    # 81
    templateParams = TemplateParams("Oncomine Immune Repertoire TCRB Assay - PGM", PGM, "AMPS_RNA")
    templateParams.update({
        "applicationGroup": APPLICATION_GROUP,
        "barcodeKitName": BARCODE_KIT_NAME,
        "categories": CATEGORIES,
        "chipType": CHIP,
        "flows": FLOWS,
        "libraryKitName": LIBRARY_KIT_NAME,
        "libraryReadLength": LIBRARY_READ_LENGTH,
        "reference": REFERENCE,
        "sampleGrouping": SAMPLE_GROUPING,
        "sequencekitname": SEQ_KIT_NAME,
        "templatingKitName": TEMPLATE_KIT_NAME,
        "planStatus": PLAN_STATUS
    })
    
    sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams)
    finish_sys_template(sysTemplate, isCreated, templateParams, plugins)


@transaction.commit_manually()
def add_or_update_all_system_templates():

    try:
        add_or_update_default_system_templates()
        add_or_update_ampliseq_system_templates()
        add_or_update_ampliseq_rna_system_templates()
        add_or_update_genericseq_system_templates()
        add_or_update_museek_system_templates()
        add_or_update_rna_system_templates()
        add_or_update_targetseq_system_templates()
        add_or_update_metagenomics_system_templates()
        add_or_update_oncomine_system_templates()
        add_or_update_onconet_system_templates()
        add_or_update_ocp_focus_system_templates()
        add_or_update_reproseq_system_templates()
        add_or_update_tagseq_system_templates()
        add_or_update_oncomine_ocav1_system_templates()
        add_or_update_oncomine_ocav2_system_templates()        
        add_or_update_oncomine_ocav3_system_templates()
        add_or_update_oncomine_pediatric_system_templates()
        add_or_update_oncomine_BRCA_system_templates()
        add_or_update_immune_response_system_templates()
        add_or_update_S5_ocp_focus_system_templates()
        add_or_update_hid_system_templates()
        add_or_update_hid_dexter_system_templates()
        add_or_update_ocp_myeloid_pgm_system_templates()
        add_or_update_ocp_myeloid_s5_system_templates()
        add_or_update_proton_PQ_system_template()
        add_or_update_mouse_transcriptome_s5_system_templates()
        add_or_update_mouse_transcriptome_proton_system_templates()
        add_or_update_mutation_load_s5_system_templates()
        add_or_update_tagseq_cfdna_system_templates()
        add_or_update_immune_repertoire_s5_system_templates()
        add_or_update_immune_repertoire_pgm_system_templates()
    except:
        print format_exc()
        transaction.rollback()
        print "*** Exceptions found. System Template(s) rolled back."
    else:
        transaction.commit()
        print "*** System Template(s) committed."


@transaction.commit_manually()
def clean_up_obsolete_templates():
    try:
        # Oncomine
        templateNames = [
            "OCP DNA",
            "OCP Fusions",
            "OCP DNA and Fusions",
            "Oncomine DNA",
            "Oncomine RNA",
            "Oncomine DNA-RNA",
            "Onconet DNA",
            "Onconet RNA",
            "Onconet DNA-RNA",
            "Ion AmpliSeq Colon and Lung Plus Cancer Panel",
            "Ion AmpliSeq Lung Fusion Cancer Panel",
            "Ion AmpliSeq Colon Lung Plus with Lung Fusion Cancer Panel",
            "Pharmacogenomics Research Panel",
            "Immuno Oncology Panel",
            "Oncomine Comprehensive v1 for S5 DNA",
            "Oncomine Comprehensive v1 for S5 DNA and Fusions",
            " Oncomine Comprehensive v2 for S5 DNA and Fusions",
            "Oncomine BRCA for PGM",
            "Oncomine BRCA for S5", 
            "Applied Biosystems Precision ID GlobalFiler NGS STR Panel",
            "Oncomine Comprehensive v3 DNA and Fusion ST",
            "Oncomine Fusion ST",
            "Ion ReproSeq Aneuploidy",
            "Ion AmpliSeq Cancer Panel",
            "AmpliSeq Myeloid DNA for PGM",
            "AmpliSeq Myeloid Fusions for PGM",
            "AmpliSeq Myeloid DNA and Fusions for PGM", 
            "AmpliSeq Myeloid DNA for S5",
            "AmpliSeq Myeloid Fusions for S5",
            "AmpliSeq Myeloid DNA and Fusions for S5",
            "Ion ReproSeq - PGM",
            "Ion ReproSeq - S5",
            " Oncomine Lung Liquid Biopsy Total Nucleic Acid",
            "Oncomine Mutational Load",
            "Oncomine Lung Liquid Biopsy Total Nucleic Acid",
            "Oncomine Lung Tumor Total Nucleic Acid",
            "Oncomine Breast Liquid Biopsy DNA v2",
            "Oncomine Breast Tumor DNA v2"
            ]

        templates = models.PlannedExperiment.objects.filter(
            planDisplayedName__in=templateNames, isReusable=True, isSystem=True)

        for template in templates:
            print "...Deleting system template.id=%d; name=%s" % (template.id, template.planDisplayedName)
            template.delete()
    except:
        print format_exc()
        transaction.rollback()
        print "*** Exceptions found. System Template(s) deletion rolled back."
    else:
        transaction.commit()
        print "*** System Template(s) deletion committed."


'''
    The below method is used to install System Templates via Off-Cycle Release path
    NOTES:
    1) off-cycle json file must include ALL of the same fields as in the above on-cycle functions
    2) update and create methods need identical off-cycle files, i.e. cannot just include single field to update
'''
def add_or_updateSystemTemplate_OffCycleRelease(**sysTemp):

    logger.debug("Start Installing System Template via Off cycle release")
    templateName = sysTemp.pop('templateName')
    application = sysTemp.pop('application', 'GENS')
    instrumentType = sysTemp.pop('instrumentType','').upper() or PGM

    isSystemDefault = sysTemp.pop('isSystemDefault', False)
    pluginsList = [plugin for plugin in sysTemp.pop("plugins_preselect", [])]

    try:
        templateParams = TemplateParams(templateName, instrumentType, application)
        templateParams.update(sysTemp)
        sysTemplate, isCreated, isUpdated = add_or_update_sys_template(templateParams, isSystemDefault)

        plugins = {}
        if (len(pluginsList) > 0):
            logger.debug("List of plugins(%s) to be pre-selected for the template(%s)" %
                         (pluginsList, templateName))
            plugins = pre_select_plugins(application, pluginsList)
        else:
            logger.debug("warning: No Plugins selected for the template: %s" % templateName)

        finish_sys_template(sysTemplate, isCreated, templateParams, plugins)
        status = {'isValid': True, 'msg': None}

    except Exception as err:
        status = {'isValid': False, 'msg': err.message}
        print format_exc()

    return status


def pre_select_plugins(application, pluginsList):
    plugins = {}

    # pre-select plugins
    for plugin in pluginsList:
        if isinstance(plugin, dict):
            pluginName = plugin.get('name')
            userInput = plugin.get('userInput')
        else:
            pluginName = plugin
            userInput = {}

        plugins[pluginName] = _get_plugin_dict(pluginName, userInput)

    if (application == "AMPS_RNA" or application == "RNA"):
        thirdPartyPluginName = "PartekFlowUploader"
        plugins[thirdPartyPluginName] = _get_plugin_dict(thirdPartyPluginName)

    return plugins

if __name__ == '__main__':
    # main
    add_or_update_all_system_templates()

    # delete system templates here...
    clean_up_obsolete_templates()
