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

# 20130228 change log:
# add system plan template(s) if missing


defaultLibKitName = "Ion Xpress Plus Fragment Library Kit"

defaultSeqKitName = "IonPGM200Kit"
defaultTemplateKitName = "Ion OneTouch 200 Template Kit v2 DL"

defaultProtonSeqKitName = "ProtonIKit"
defaultProtonTemplateKitName = "Ion Proton I Template OT2 Kit"

defaultPELibAdapterKitName = "Paired-end library adapter kit"
defaultPESeqKitName = "Ion PGM 2x200 paired end sequencing kit"

defaultLibraryKeySequence = "TCAG"
default3PrimeAdapterSequence = "ATCACCGACTGCCCATAGAGAGGCTGAGAC"

defaultReverseLibraryKeySequence = "TCAGC"
defaultReverse3PrimeAdapterSequence = "CTGAGTCGGAGACACGCAGGGATGAGATGG"


def check_system_template():
    sysDefaultTemplate, isCreated = models.PlannedExperiment.objects.get_or_create(
        isSystemDefault=True,
        isSystem=True,
        isReusable=True,
        isPlanGroup=False,
        planDisplayedName="System Default Template",
        planName="System_Default_Template",
        defaults={
            "runMode": "single",
            "isReverseRun": False,
            "runType": "GENS",
            "usePreBeadfind": True,
            "usePostBeadfind": True,
            "preAnalysis": True,
            "templatingKitName": defaultTemplateKitName,
            "metaData": "",
        },
    )

    print(
        "*** AFTER get_or_create system default template isCreated=%s; id=%d;"
        % (str(isCreated), sysDefaultTemplate.id)
    )

    return sysDefaultTemplate, isCreated


def check_proton_system_template():
    sysDefaultTemplate, isCreated = models.PlannedExperiment.objects.get_or_create(
        isSystemDefault=True,
        isSystem=True,
        isReusable=True,
        isPlanGroup=False,
        planDisplayedName="Proton System Default Template",
        planName="Proton_System_Default_Template",
        defaults={
            "runMode": "single",
            "isReverseRun": False,
            "runType": "GENS",
            "usePreBeadfind": True,
            "usePostBeadfind": False,
            "preAnalysis": True,
            "templatingKitName": defaultProtonTemplateKitName,
            "metaData": "",
        },
    )

    print(
        "*** AFTER get_or_create Proton system default template isCreated=%s "
        % (str(isCreated))
    )

    return sysDefaultTemplate, isCreated


def finish_create_system_template(
    sysDefaultTemplate, chipType="", flows=500, seqKit=defaultSeqKitName
):
    planGUID = str(uuid.uuid4())
    sysDefaultTemplate.planGUID = planGUID

    date = datetime.datetime.now()
    sysDefaultTemplate.date = date

    planShortID = "".join(
        random.choice(string.ascii_uppercase + string.digits) for x in range(5)
    )

    while models.PlannedExperiment.objects.filter(
        planShortID=planShortID, planExecuted=False
    ):
        planShortID = "".join(
            random.choice(string.ascii_uppercase + string.digits) for x in range(5)
        )

    print("*** System Default Template shortID=%s" % str(planShortID))

    sysDefaultTemplate.planShortID = planShortID

    sysDefaultTemplate.save()

    print("*** AFTER System Default Template is saved ")

    for qcType in models.QCType.objects.all():
        sysDefaultQC, isCreated = models.PlannedExperimentQC.objects.get_or_create(
            plannedExperiment=sysDefaultTemplate, qcType=qcType, threshold=30
        )

        print(
            "*** AFTER get_or_create system default qc for %s isCreated=%s "
            % (qcType.qcName, str(isCreated))
        )

        sysDefaultTemplate.plannedexperimentqc_set.add(sysDefaultQC)
        sysDefaultTemplate.save()

    exp_kwargs = {
        "autoAnalyze": True,
        "chipType": chipType,
        "date": date,
        "flows": flows,
        "plan": sysDefaultTemplate,
        "sequencekitname": seqKit,
        "status": sysDefaultTemplate.planStatus,
        # temp experiment name value below will be replaced in crawler
        "expName": sysDefaultTemplate.planGUID,
        "displayName": sysDefaultTemplate.planShortID,
        "pgmName": "",
        "log": "",
        # db constraint requires a unique value for experiment. temp unique value
        # below will be replaced in crawler
        "unique": sysDefaultTemplate.planGUID,
        "chipBarcode": "",
        "seqKitBarcode": "",
        "sequencekitbarcode": "",
        "reagentBarcode": "",
        "cycles": 0,
        "diskusage": 0,
        "expCompInfo": "",
        "baselineRun": "",
        "flowsInOrder": "",
        "ftpStatus": "",
        "runMode": sysDefaultTemplate.runMode,
        "storageHost": "",
    }

    experiment = models.Experiment(**exp_kwargs)
    experiment.save()

    print(
        "*** AFTER saving experiment.id=%d for system default template.id=%d; name=%s"
        % (experiment.id, sysDefaultTemplate.id, sysDefaultTemplate.planName)
    )

    eas_kwargs = {
        "barcodedSamples": "",
        "barcodeKitName": "",
        "date": date,
        "experiment": experiment,
        "hotSpotRegionBedFile": "",
        "isEditable": True,
        "isOneTimeOverride": False,
        "libraryKey": defaultLibraryKeySequence,
        "libraryKitName": defaultLibKitName,
        "reference": "",
        "selectedPlugins": "",
        "status": sysDefaultTemplate.planStatus,
        "targetRegionBedFile": "",
        "threePrimeAdapter": default3PrimeAdapterSequence,
    }

    eas = models.ExperimentAnalysisSettings(**eas_kwargs)
    eas.save()

    print(
        "*** AFTER saving EAS.id=%d for system default template.id=%d; name=%s"
        % (eas.id, sysDefaultTemplate.id, sysDefaultTemplate.planName)
    )


@transaction.commit_manually()
def doFix():
    hasNewCreation = False

    try:
        sysTemplate, isCreated = check_system_template()
        if isCreated:
            hasNewCreation = True

            print(
                "*** WARNING: System Default Template is missing. Creating new entry now..."
            )
            finish_create_system_template(sysTemplate)
        else:
            print("*** System Default Template is found. So far so good...")

        sysTemplate, isCreated = check_proton_system_template()
        if isCreated:
            hasNewCreation = True
            print(
                "*** WARNING: Proton System Default Template is missing. Creating new entry now..."
            )
            finish_create_system_template(
                sysTemplate, "900", 260, defaultProtonSeqKitName
            )
        else:
            print("*** Proton System Default Template is found.")
    except:
        print(format_exc())
        transaction.rollback()
        print("*** Exceptions found. System Default Template(s) rolled back.")
    else:
        if hasNewCreation:
            transaction.commit()
            print("*** System Default Template(s) committed.")


# main
sysTemplates = models.PlannedExperiment.objects.filter(
    isSystem=True, isReusable=True, isSystemDefault=True
)
if sysTemplates and sysTemplates.count() == 2:
    print("*** Good! All system default templates are found. Nothing to fix.")
else:
    doFix()
