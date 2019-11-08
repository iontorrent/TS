#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
from django.db import transaction

import sys
import os
from iondb.rundb import models

import copy
import datetime
import uuid
import random
import string

from traceback import format_exc

# 20130906 change log:
# add system plan template for Avalanche if missing


def create_system_template():
    sysDefaultTemplates = models.PlannedExperiment.objects.filter(
        isSystem=True, isSystemDefault=True, planDisplayedName="System Default Template"
    )

    newSysTemplate = None
    isCreated = False

    if sysDefaultTemplates:
        sysDefaultTemplate = sysDefaultTemplates[0]

        currentTime = datetime.datetime.now()

        # clone the system default template
        newSysTemplate = copy.copy(sysDefaultTemplate)
        newSysTemplate.pk = None

        newSysTemplate.planDisplayedName = "PGM Avalanche Template"
        newSysTemplate.planName = "PGM_Avalanche_Template"
        newSysTemplate.templatingKitName = "Ion PGM Template AV 500 Kit"
        newSysTemplate.runType = "WGNM"
        newSysTemplate.chipType = ""
        newSysTemplate.isSystemDefault = False

        newSysTemplate.planGUID = None
        newSysTemplate.planShortID = None

        planGUID = str(uuid.uuid4())
        newSysTemplate.planGUID = planGUID

        planShortID = "".join(
            random.choice(string.ascii_uppercase + string.digits) for x in range(5)
        )

        while models.PlannedExperiment.objects.filter(
            planShortID=planShortID, planExecuted=False
        ):
            planShortID = "".join(
                random.choice(string.ascii_uppercase + string.digits) for x in range(5)
            )

        newSysTemplate.planShortID = planShortID

        newSysTemplate.date = currentTime
        newSysTemplate.save()

        print(
            "*** fix.. after saving new Avalanche sysTemplate.id=%d; name=%s "
            % (newSysTemplate.id, newSysTemplate.planDisplayedName)
        )

        # copy Experiment
        expObj = copy.copy(sysDefaultTemplate.experiment)
        expObj.pk = None
        expObj.expName = newSysTemplate.planGUID
        expObj.unique = newSysTemplate.planGUID
        expObj.chipType = ""
        expObj.sequencekitname = "IonPGM400Kit"
        expObj.flows = 1100
        expObj.plan = newSysTemplate
        expObj.date = currentTime
        expObj.save()

        print(
            "*** fix.. after saving new Avalanche sysTemplate.experiment.id=%d; name=%s "
            % (expObj.id, expObj.expName)
        )

        # copy EAS
        for easObj in sysDefaultTemplate.experiment.eas_set.all():
            easObj.pk = None
            easObj.experiment = expObj
            easObj.threePrimeAdapter = "ATCACCGACTGCCCATAGAGAGGAAAGCGG"
            easObj.date = currentTime
            easObj.save()

            print(
                "*** fix.. after saving new Avalanche sysTemplate.experiment.eas.id=%d "
                % (easObj.id)
            )

        # clone the qc thresholds as well
        qcValues = sysDefaultTemplate.plannedexperimentqc_set.all()

        for qcValue in qcValues:
            qcObj = copy.copy(qcValue)

            qcObj.pk = None
            qcObj.plannedExperiment = newSysTemplate
            qcObj.save()

            print(
                "*** fix.. after saving new Avalanche sysTemplate.qc.id=%d "
                % (qcObj.id)
            )

        isCreated = True
    else:
        print(
            "*** fix.. WARNING - NO System Default Template is found to create an Avalanche template!!"
        )

    return newSysTemplate, isCreated


@transaction.commit_manually()
def doFix():
    hasNewCreation = False

    try:
        sysTemplate, isCreated = create_system_template()
        if isCreated:
            hasNewCreation = True

            print("*** fix.. Congratulations, you are all set now...")
        else:
            print(
                "*** There is a bigger problem; even the System Default Template is missing..."
            )
    except:
        print(format_exc())
        transaction.rollback()
        print("*** Exceptions found. Avalanche System Template rolled back.")
    else:
        if hasNewCreation:
            transaction.commit()
            print("*** Avalanche System Template committed.")


# main
sysTemplates = models.PlannedExperiment.objects.filter(
    isSystem=True, isReusable=True, planName="PGM_Avalanche_Template2"
)
if sysTemplates and sysTemplates.count() > 0:
    print("*** Good! Avalanche system template is found. Nothing to fix.")
else:
    doFix()
