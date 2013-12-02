# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Models
======

Django database models for the Torrent Analysis Suite.

``models`` defines the way the frontend database stores information about
Personal Genome Machine experiments. This information includes...

* The location of experiment data (the ``Experiment`` class)
* Analysis statistics generated from experiment data (the ``Results`` class)
* Test fragment templates  (the ``Template`` class).

``models`` also contains information about infrastructure that supports
the data gathering and analysis.
"""
import datetime
import re
import os
import fnmatch
import traceback
import pytz
from django.core.exceptions import ValidationError

import iondb.settings


from django.db import models
from iondb.utils import devices
import json
import simplejson

from iondb.rundb import json_field

import shutil
import uuid
import hmac
import random
import string
import logging
from iondb.rundb import tasks
from iondb.rundb.data import dmactions_types
from iondb.rundb.separatedValuesField import SeparatedValuesField
try:
    from hashlib import sha1
except ImportError:
    import sha
    sha1 = sha.sha

logger = logging.getLogger(__name__)

from django.contrib.auth.models import User, Group
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes import generic
from django.utils.encoding import force_unicode
from django.utils import timezone
from django.core import urlresolvers
from django.db.models.signals import post_save, pre_delete, post_delete
from django.dispatch import receiver
from distutils.version import LooseVersion
from celery.task.control import revoke

import copy

# Auto generate tastypie API key for users
from tastypie.models import create_api_key
models.signals.post_save.connect(create_api_key, sender=User)

# Create your models here.

class Project(models.Model):
    #TODO The name should not be uniq because there can be a public one
    #and a private one with the same name and that would not work
    name = models.CharField(max_length=64, unique = True)
    creator = models.ForeignKey(User)
    public = models.BooleanField(default=True)
    # This will be set to the time a new record is created
    created = models.DateTimeField(auto_now_add=True)
    # This will be set to the current time every time the model is updated
    modified = models.DateTimeField(auto_now=True)

    def __unicode__(self):
        return self.name

#    def save(self, force_insert=False, force_update=False, using=None):
#        logger.error(self.pk)
#        if self.pk:
#            current = Project.objects.get(self.pk)
#            logger.error( current.creator.username, self.creator.username)
#            if current and current.creator.pk != self.creator.pk:
#                raise ValidationError('creator is readonly')
#        models.Model.save(self, force_insert=force_insert, force_update=force_update, using=using)

    @classmethod
    def bulk_get_or_create(cls, projects, user=None):
        """projects is either a string specifying a single project name or
        a list of strings for such.  Each project name is queried and a project
        with that name is created if necessary and all named projects are
        returned in a list.
         """
        if not user:
            user = User.objects.order_by('pk')[0]
        if isinstance(projects, basestring):
            projects = [projects]
        projects = filter(None, projects)
        projects = [name.strip() for name in projects]
        projects = filter(None, projects)
        projects = [name.replace(' ', '_') for name in projects]
        return [Project.objects.get_or_create(name__iexact=name, defaults={'name':name,'creator':user})[0] for name in projects]


class KitInfoManager(models.Manager):
    def get_by_natural_key(self, uid):
        return self.get(uid = uid)

class KitInfo(models.Model):

    ALLOWED_KIT_TYPES = (
        ('SequencingKit', 'SequencingKit'),
        ('LibraryKit', 'LibraryKit'),
        ('TemplatingKit', 'TemplatingKit'),
        ('AdapterKit', "AdapterKit"),
        ('ControlSequenceKit', "ControlSequenceKit"),
        ('SamplePrepKit', "SamplePrepKit"),
        ('IonChefPrepKit', 'IonChefPrepKit'),
        ('AvalancheTemplateKit', 'AvalancheTemplateKit')        
    )

    kitType = models.CharField(max_length=20, choices=ALLOWED_KIT_TYPES)
    name = models.CharField(max_length=512, blank=False, unique=True)
    description = models.CharField(max_length=3024, blank=True)
    flowCount = models.PositiveIntegerField()

    ALLOWED_NUCLEOTIDE_TYPES = (
        ('', 'Any'),
        ('dna', 'DNA'),
        ('rna', 'RNA')
    )

    nucleotideType = models.CharField(max_length=64, choices=ALLOWED_NUCLEOTIDE_TYPES, default='', blank=True)

    ALLOWED_RUN_MODES = (
        ('', 'Undefined'),
        ('single', 'SingleRead'),
        ('pe', 'PairedEnd')
    )

    #run mode
    runMode = models.CharField(max_length=64, choices=ALLOWED_RUN_MODES, default='', blank=True)

    isActive = models.BooleanField(default = True)

    ALLOWED_INSTRUMENT_TYPES = (
        ('', 'Any'),
        ('pgm', 'PGM'),
        ('proton', 'Proton')
    )
    #compatible instrument type
    instrumentType = models.CharField(max_length=64, choices=ALLOWED_INSTRUMENT_TYPES, default='', blank=True)

    uid = models.CharField(max_length=10, unique=True, blank=False)

    objects = KitInfoManager()

    def __unicode__(self):
        return u'%s' % self.name

    def natural_key(self):
        return (self.uid,)  # must return a tuple

    class Meta:
        unique_together = (('kitType', 'name'),)
        #unique_together = (('uid'),)


class KitPartManager(models.Manager):
    def get_by_natural_key(self, kitBarcode):
        return self.get(barcode = kitBarcode)


class KitPart(models.Model):
    kit = models.ForeignKey(KitInfo, null=False)
    barcode = models.CharField(max_length=64, unique=True, blank=False)

    objects = KitPartManager()

    def __unicode__(self):
        return u'%s' % self.barcode

    def natural_key(self):
        return (self.barcode, )  # must return a tuple

    class Meta:
        unique_together = (('barcode'),)


class RunTypeManager(models.Manager):
    def get_by_natural_key(self, runType):
        return self.get(runType = runType)

class RunType(models.Model):
    runType = models.CharField(max_length=512, unique = True)
    barcode = models.CharField(max_length=512, blank=True)
    description = models.TextField(blank=True)
    meta = json_field.JSONField(blank=True, null=True, default = "")

    ALLOWED_NUCLEOTIDE_TYPES = (
        ('', 'Any'),
        ('dna', 'DNA'),
        ('rna', 'RNA')
    )

    nucleotideType = models.CharField(max_length=64, choices=ALLOWED_NUCLEOTIDE_TYPES, default='dna', blank=True)

    objects = RunTypeManager()

    applicationGroups = models.ManyToManyField("ApplicationGroup", related_name='applications', null = True)

    alternate_name = models.CharField(max_length=512, blank = True, null = True, default = "")

    def __unicode__(self):
        return self.runType

    def natural_key(self):
        return (self.runType, )  # must return a tuple


class ApplicationGroupManager(models.Manager):
    def get_by_natural_key(self, uid):
        return self.get(uid = uid)

class ApplicationGroup(models.Model):
    name = models.CharField(max_length = 127, blank = False, null = False)
    description = models.CharField(max_length = 1024, blank = True, null = True)
    isActive = models.BooleanField(default = True)

    uid = models.CharField(max_length = 32, unique = True, blank = False)

    objects = ApplicationGroupManager()

    def __unicode__(self):
        return self.name

    def natural_key(self):
        return (self.uid,)  # must return a tuple


class ApplProductManager(models.Manager):
    def get_by_natural_key(self, productCode):
        return self.get(productCode = productCode)

class ApplProduct(models.Model):
    #application with no product will have a default product pre-loaded to db to hang all the
    #application-specific settings
    productCode = models.CharField(max_length = 64, unique=True, default='any', blank=False)
    productName = models.CharField(max_length = 128, blank=False)
    description = models.CharField(max_length = 1024, blank=True)
    applType = models.ForeignKey(RunType)
    isActive = models.BooleanField(default = True)
    #if isVisible is false, it will not be shown as a choice in UI
    isVisible = models.BooleanField(default = False)
    defaultSequencingKit = models.ForeignKey(KitInfo, related_name='seqKit_applProduct_set', null=True)
    defaultLibraryKit = models.ForeignKey(KitInfo, related_name='libKit_applProduct_set', null=True)
    defaultPairedEndSequencingKit = models.ForeignKey(KitInfo, related_name='peSeqKit_applProduct_set', null=True)
    defaultPairedEndLibraryKit = models.ForeignKey(KitInfo, related_name='peLibKit_applProduct_set', null=True)
    defaultGenomeRefName = models.CharField(max_length = 1024, blank=True, null=True)
    #this is analogous to bedFile in PlannedExperiment
    defaultTargetRegionBedFileName = models.CharField(max_length = 1024, blank=True, null=True)
    #this is analogous to regionFile in PlannedExperiment
    defaultHotSpotRegionBedFileName = models.CharField(max_length = 1024, blank=True, null=True)

    defaultChipType = models.CharField(max_length=128, blank=True, null=True)
    isDefault = models.BooleanField(default = False)
    isPairedEndSupported = models.BooleanField(default = True)
    isDefaultPairedEnd = models.BooleanField(default = False)
    defaultVariantFrequency = models.CharField(max_length = 512, blank=True, null=True)

    defaultFlowCount = models.PositiveIntegerField(default = 0)
    defaultPairedEndAdapterKit = models.ForeignKey(KitInfo, related_name='peAdapterKit_applProduct_set', null=True)
    defaultTemplateKit = models.ForeignKey(KitInfo, related_name='templateKit_applProduct_set', null=True)
    defaultControlSeqKit = models.ForeignKey(KitInfo, related_name='controlSeqKit_applProduct_set', null=True)

    #sample preparation kit
    defaultSamplePrepKit = models.ForeignKey(KitInfo, related_name='samplePrepKit_applProduct_set', null=True)

    isHotspotRegionBEDFileSuppported = models.BooleanField(default = True)

    isDefaultBarcoded = models.BooleanField(default = False)
    defaultBarcodeKitName = models.CharField(max_length=128, blank=True, null=True)
    defaultIonChefPrepKit = models.ForeignKey(KitInfo, related_name='ionChefPrepKit_applProduct_set', null=True)
    defaultAvalancheTemplateKit = models.ForeignKey(KitInfo, related_name='avalancheTemplateKit_applProduct_set', null=True)
    defaultAvalancheSequencingKit = models.ForeignKey(KitInfo, related_name='avalancheSeqKit_applProduct_set', null=True)

    ALLOWED_INSTRUMENT_TYPES = (
        ('', 'Any'),
        ('pgm', 'PGM'),
        ('proton', 'Proton')
    )
    #compatible instrument type
    instrumentType = models.CharField(max_length=64, choices=ALLOWED_INSTRUMENT_TYPES, default='', blank=True)

    def __unicode__(self):
        return u'%s' % self.productName

    def natural_key(self):
        return (self.productCode, )    # must return a tuple


class QCType(models.Model):
    qcName = models.CharField(max_length=512, blank=False, unique=True)
    description = models.CharField(max_length=1024, blank=True)
    minThreshold = models.PositiveIntegerField(default=0)
    maxThreshold = models.PositiveIntegerField(default=100)
    defaultThreshold = models.PositiveIntegerField(default=0)

    def __unicode__(self):
        return u'%s' % self.qcName


class PlannedExperimentManager(models.Manager):
    def _extract_extra_kwargs(self, **kwargs):
        #PDD-TODO: remove the x_ prefix. the x_ prefix is just a reminder what the obsolete attributes to remove during the next phase

        extra_kwargs = {}
        extra_kwargs['x_autoAnalyze'] = kwargs.pop('x_autoAnalyze', True)
        extra_kwargs['x_barcodedSamples'] = kwargs.pop('x_barcodedSamples', {})
        extra_kwargs['x_barcodeId'] = kwargs.pop('x_barcodeId', "")
        extra_kwargs['x_bedfile'] = kwargs.pop('x_bedfile', "")
        extra_kwargs['x_chipType'] = kwargs.pop('x_chipType', "")
        extra_kwargs['x_flows'] = kwargs.pop('x_flows', None)
        extra_kwargs['x_forward3primeadapter'] = kwargs.pop('x_forward3primeadapter', "")
        ###'_isReverseRun':  = self.isReverseRun
        extra_kwargs['x_library'] = kwargs.pop('x_library', "")
        extra_kwargs['x_libraryKey'] = kwargs.pop('x_libraryKey', "")
        extra_kwargs['x_librarykitname'] = kwargs.pop('x_librarykitname', "")
        extra_kwargs['x_notes'] = kwargs.pop('x_notes', "")
        extra_kwargs['x_regionfile'] = kwargs.pop('x_regionfile', "")
        extra_kwargs['x_sample'] = kwargs.pop('x_sample', "")
        extra_kwargs['x_sample_external_id'] = kwargs.pop('x_sample_external_id', "")
        extra_kwargs['x_sample_description'] = kwargs.pop('x_sample_description', "")
        extra_kwargs['x_sampleDisplayedName'] = kwargs.pop('x_sampleDisplayedName', "")
        extra_kwargs['x_selectedPlugins'] = kwargs.pop('x_selectedPlugins', {})
        extra_kwargs['x_sequencekitname'] = kwargs.pop('x_sequencekitname', "")
        extra_kwargs['x_variantfrequency'] = kwargs.pop('x_variantfrequency', "")
        extra_kwargs['x_isDuplicateReads'] = kwargs.pop('x_isDuplicateReads', False)
        extra_kwargs['x_isSaveBySample'] = kwargs.pop('x_isSaveBySample', False)
        extra_kwargs['x_numberOfChips'] = kwargs.pop('x_numberOfChips', 1)

        logger.info("EXIT PlannedExpeirmentManager.extract_extra_kwargs... extra_kwargs=%s" %(extra_kwargs))

        return kwargs, extra_kwargs


    def save_plan(self, planOid, **kwargs):
        popped_kwargs, extra_kwargs = self._extract_extra_kwargs(**kwargs)

        logger.info("PlannedExpeirmentManager.save_plan() planOid=%s; after extract_extra_kwargs... popped_kwargs==%s" %(str(planOid), popped_kwargs))
        logger.info("PlannedExpeirmentManager.save_plan() after extract_extra_kwargs... extra_kwargs=%s" %(extra_kwargs))

        if planOid < 0:
            plan = self.create(**popped_kwargs)
            # if extra_kwargs.pop('x_isSaveBySample', False):
            #     plan.sampleSet_uid = str(uuid.uuid4())
            #     #setattr(plan, 'sampleSet', extra_kwargs.get('x_sampleSet'))
            #     plan.sampleSet_planTotal = 1
            #     plan.save()
        else:
            plan = self.get(pk = planOid)

            for key, value in popped_kwargs.items():
                setattr(plan, key, value)

        plan.save()

        if plan:
            isPlanCreated = (planOid < 0)
            plan.save_plannedExperiment_association(isPlanCreated, **extra_kwargs)
            plan.update_plan_qcValues(**popped_kwargs)


        return plan, extra_kwargs


class PlannedExperiment(models.Model):
    """
    Create a planned run to ease the pain on manually entry on the PGM
    """

    #plan name
    planName = models.CharField(max_length=512,blank=True,null=True)

    #Global uniq id for planned run
    planGUID = models.CharField(max_length=512,blank=True,null=True)

    #make a id for easy entry
    planShortID = models.CharField(max_length=5,blank=True,null=True,db_index=True)

    #was the plan already executed?
    planExecuted = models.BooleanField(default=False)

    #Typical status transition of a plan/experiment:
    # pending (optional) -> planned -> run (e.g., from LIMS or HUB)
    # reserved (optional)-> planned -> run (e.g., from Sample Prep instrument like IonChef)
    #
    #planStatus summary:
    #"pending" status : means the plan creation is not fully completed  - user can further edit a pending plan
    #from the plan wizard and have its status promoted to "planned" upon successful db plan update

    #"planned" status : means the entire plan object has been created and saved and ready to be selected for
    #sequencing run
    #"run" status     : means the plan has been claimed by a sequencing run on the instrument

    #"reserved" status: the original intent of this status is to indicate that part of a group plan
    #(e.g., forward of a paired-end plan) has been claimed for a run - user can't further edit a reserved plan
    #from the plan wizard

    #"voided" status  : the original intent of this status is to indicate this group plan has been abandoned
    #even though it has not been fully used for its sequencing runs - user can't further edit a voided plan
    #from the plan wizard

    #"blank" status   : for backward compatibility, blank status + planExecuted = True is same as "executed"
    #while "blank" status + planExecuted = False is same as "planned" status. [TODO: replace blank status]

    ALLOWED_PLAN_STATUS = (
        ('', 'Undefined'),
        ('pending', 'Pending'),
        ('voided', 'Voided'),
        ('reserved', 'Reserved'),
        ('planned', 'Planned'),
        ('run', 'Run')
    )

    #planStatus
    planStatus = models.CharField(max_length=512, blank=True, choices=ALLOWED_PLAN_STATUS, default='')

    #who ran this
    username = models.CharField(max_length=128, blank=True, null=True)

    #what PGM started this
    planPGM = models.CharField(max_length=128, blank=True, null=True)

    #when was this added to the plans
    date = models.DateTimeField(blank=True,null=True)

    #When was the plan executed?
    planExecutedDate = models.DateTimeField(blank=True,null=True)

    #add metadata grab bag
    metaData = json_field.JSONField(blank=True)

    chipBarcode = models.CharField(max_length=64, blank=True,null=True)

    seqKitBarcode = models.CharField(max_length=64, blank=True,null=True)

    #name of the experiment
    expName = models.CharField(max_length=128,blank=True)

    #Pre-Run/Beadfind
    usePreBeadfind = models.BooleanField()

    #Post-Run/Beadfind
    usePostBeadfind = models.BooleanField()

    #cycles
    cycles = models.IntegerField(blank=True,null=True)

    #autoName string
    autoName = models.CharField(max_length=512, blank=True, null=True)

    preAnalysis = models.BooleanField()

    #RunType -- this is from a list of possible types (aka application)
    runType = models.CharField(max_length=512, blank=False, null=False, default="GENS")

    #adapter (20120313: this was probably for forward 3' adapter but was never used.  Consider this column obsolete)
    adapter = models.CharField(max_length=256, blank=True, null=True)

    #Project
    projects = models.ManyToManyField(Project, related_name='plans', blank=True)

    #runname - name of the raw data directory
    runname = models.CharField(max_length=255, blank=True, null=True)

    #flowsInOrder = models.CharField(max_length=512, blank=True, null=True)
    storageHost = models.CharField(max_length=128, blank=True, null=True)
    reverse_primer = models.CharField(max_length=128, blank=True, null=True)

    #add field for ion reporter upload plugin workflow
    irworkflow = models.CharField(max_length=1024,blank=True)

    #we now persist the sequencing kit name instead of its part number. to be phased out
    libkit = models.CharField(max_length=512, blank=True, null=True)

    STORAGE_CHOICES = (
        ('KI', 'Keep'),
        ('A', 'Archive Raw'),

        ('D', 'Delete Raw'),
        )

    storage_options = models.CharField(max_length=200, choices=STORAGE_CHOICES,
                                       default='A')

    isReverseRun = models.BooleanField(default=False)

    #plan displayed name allows embedded blanks.
    #planName is the display name with embedded blanks converted to underscores

    planDisplayedName = models.CharField(max_length=512,blank=True,null=True)

    ALLOWED_RUN_MODES = (
        ('', 'Undefined'),
        ('single', 'SingleRead'),
        ('pe', 'PairedEnd')
    )

    #run mode
    runMode = models.CharField(max_length=64, choices=ALLOWED_RUN_MODES, default='', blank=True)

    #whether this is a plan template
    isReusable = models.BooleanField(default=False)
    isFavorite = models.BooleanField(default=False)

    #whether this is a pre-defined plan template
    isSystem = models.BooleanField(default=False)

    #if instrument user does not select a plan for the run,
    #crawler will use the properties from the system default plan or template
    #for the run
    isSystemDefault = models.BooleanField(default=False)

    #used for paired-end plan & plan template
    #for PE plan template, there will only be 1 template in db
    #for PE plan, there will be 1 parent plan and 2 children (1 forward and 1 reverse)

    isPlanGroup = models.BooleanField(default=False)
    parentPlan = models.ForeignKey('self', related_name='childPlan_set', null=True, blank=True)

    qcValues = models.ManyToManyField(QCType, through="PlannedExperimentQC", null=True)

    templatingKitBarcode = models.CharField(max_length=64, blank=True, null=True)
    templatingKitName = models.CharField(max_length=512, blank=True, null=True)

    controlSequencekitname = models.CharField(max_length=512, blank=True, null=True)

    #pairedEnd library adapter name
    pairedEndLibraryAdapterName = models.CharField(max_length=512, blank=True, null=True)

    #sample preparation kit name
    samplePrepKitName = models.CharField(max_length=512, blank=True, null=True)

    #the barcode sample prep label on the sample tube
    sampleTubeLabel = models.CharField(max_length=512, blank=True, null=True)

    sampleSet = models.ForeignKey('SampleSet', related_name = "plans", null = True, blank = True)
    sampleSet_uid = models.CharField(max_length = 512, blank = True, null = True)
    sampleSet_planIndex = models.PositiveIntegerField(default=0)
    sampleSet_planTotal = models.PositiveIntegerField(default=0)

    sampleGrouping = models.ForeignKey("SampleGroupType_CV", blank=True, null=True, default=None)
    applicationGroup = models.ForeignKey(ApplicationGroup, null=True)

    objects = PlannedExperimentManager()


    def __unicode__(self):
        if self.planName:
            return self.planName
        return "Plan_%s" % self.planShortID

    def findShortID(self):
        """Search for a plan short ID that has not been used"""

        planShortID = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(5))

        if PlannedExperiment.objects.filter(planShortID=planShortID, planExecuted=False):
            self.findShortID()

        return planShortID


    @classmethod
    def get_latest_plan_or_template_by_chipType(cls, chipType = None, isReusable = True, isSystem = True, isSystemDefault = True):
        """
        return the latest plan or template with matching chipType. Input chipType can be None
        """
        plans = PlannedExperiment.objects.filter(isReusable = isReusable, isSystem = isSystem, isSystemDefault = isSystemDefault).order_by("-date")

        if plans:
            for plan in plans:
                planChipType = plan.get_chipType()

                if chipType:
                    if (planChipType == chipType) or (planChipType == chipType[:3]):
                        logger.info("PlanExperiment.get_latest_plan_or_template_by_chipType() match - chipType=%s; planChipType=%s; found plan.pk=%s" %(chipType, planChipType, str(plan.pk)))
                        return plan
                elif (not planChipType and not chipType):
                    logger.info("PlanExperiment.get_latest_plan_or_template_by_chipType() found plan.pk=%s" %(str(plan.pk)))
                    return plan

        if (isReusable and isSystem and isSystemDefault):
            chips = dict((chip.name, chip.instrumentType) for chip in Chip.objects.all())

            chipInstrumentType = "pgm"
            if chipType:
                chipInstrumentType = chips.get(chipType, "")
                if (not chipInstrumentType):
                    chipInstrumentType = chips.get(chipType[:3], "pgm")

                    logger.debug("PlanExperiment.get_latest_plan_or_template_by_chipType() chipType=%s; instrumentType=%s; " %(chipType, chipInstrumentType))
            else:
                logger.debug("PlanExperiment.get_latest_plan_or_template_by_chipType() NO chipType - use chipInstrumentType=%s; " %(chipInstrumentType))

            for template in plans:
                plan_chipInstrumentType = "pgm"

                planChipType = template.get_chipType()
                if planChipType:
                    plan_chipInstrumentType = chips.get(planChipType, "")
                    if not plan_chipInstrumentType:
                        plan_chipInstrumentType = chips.get(chipType[:3], "pgm")

                if plan_chipInstrumentType == chipInstrumentType:
                    logger.debug("EXIT PlanExperiment.get_latest_plan_or_template_by_chipType() return template.id=%d; for chipInstrumentType=%s; " %(template.id, chipInstrumentType))
                    return template

        return None


    def get_autoAnalyze(self):
        experiment = self.experiment
        if experiment:
            return experiment.autoAnalyze
        else:
            return False

    def get_barcodedSamples(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.barcodedSamples
            else:
                return ""
        else:
            return ""

    def get_barcodeId(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.barcodeKitName
            else:
                return ""
        else:
            return ""

    def get_bedfile(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.targetRegionBedFile
            else:
                return ""
        else:
            return ""

    def get_chipType(self):
        experiment = self.experiment
        if experiment:
            chipType = experiment.chipType
            # older chip names compatibility, e.g 314R, 318B, "318C"
            if chipType not in Chip.objects.values_list('name',flat=True):
                chipType = chipType.replace('"','')
                if chipType and chipType[0].isdigit():
                    chipType = chipType[:3]
            return chipType
        else:
            return ""

    def get_flows(self):
        experiment = self.experiment
        if experiment:
            return experiment.flows
        else:
            return 0

    def get_forward3primeadapter(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.threePrimeAdapter
            else:
                return ""
        else:
            return ""

    def get_library(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.reference
            else:
                return ""
        else:
            return ""

    def get_libraryKey(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.libraryKey
            else:
                return ""
        else:
            return ""

    def get_librarykitname(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.libraryKitName
            else:
                return ""
        else:
            return ""

    def get_notes(self):
        experiment = self.experiment
        if experiment:
            return experiment.notes
        else:
            return ""

    def get_regionfile(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.hotSpotRegionBedFile
            else:
                return ""
        else:
            return ""

    def get_sample(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas and latest_eas.barcodeKitName:
                return ""
            else:
                if experiment.samples.count() > 0:
                    sample = experiment.samples.values()[0]
                    name = sample['name']
                    return name
                else:
                    return ""
        else:
            return ""

    def get_sample_external_id(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas and latest_eas.barcodeKitName:
                return ""
            else:
                if experiment.samples.count() > 0:
                    sample = experiment.samples.values()[0]
                    externalId = sample['externalId']
                    return externalId
                else:
                    return ""
        else:
            return ""

    def get_sample_description(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas and latest_eas.barcodeKitName:
                return ""
            else:
                if experiment.samples.count() > 0:
                    sample = experiment.samples.values()[0]
                    description = sample['description']
                    return description
                else:
                    return ""
        else:
            return ""

    def get_sampleDisplayedName(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas and latest_eas.barcodeKitName:
                return ""
            else:
                if experiment.samples.count() > 0:
                    sample = experiment.samples.values()[0]
                    displayedName = sample['displayedName']
                    return displayedName
                else:
                    return ""
        else:
            return ""

    def get_selectedPlugins(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.selectedPlugins
            else:
                return ""
        else:
            return ""

    def get_sequencekitname(self):
        experiment = self.experiment
        if experiment:
            return experiment.sequencekitname
        else:
            return ""

    def get_variantfrequency(self):
        return ""

    def is_ionChef(self):
        kitName = self.templatingKitName
        if kitName:
            for icKit in KitInfo.objects.filter(kitType = "IonChefPrepKit"):
                if (kitName == icKit.name):
                    return True

        return False

    def is_duplicateReads(self):
        experiment = self.experiment
        if experiment:
            latest_eas = experiment.get_EAS()
            if latest_eas:
                return latest_eas.isDuplicateReads
            else:
                return False
        else:
            return False

    def save(self, *args, **kwargs):

        logger.debug("PDD ENTER models.PlannedExperiment.save(self, args, kwargs)")
        logger.debug("PDD kwargs=%s " %(kwargs))

        if not self.planStatus:
            self.planStatus = "planned"

        #if user uses the old ui to save a plan directly, planDisplayedName will have no user input
        if not self.planDisplayedName:
            self.planDisplayedName = self.planName;

        self.date = timezone.now()
        if not self.planShortID:
            self.planShortID = self.findShortID()
        if not self.planGUID:
            self.planGUID = str(uuid.uuid4())

        if not self.runType:
            self.runType = "GENS"

        isPlanCreated = True

        #for backward and non-gui compatibility if user is not using the v3.0 plan/template wizard to create a plan
        if (self.isReverseRun == True):
            self.runMode = "pe"

        if (self.id):
            isPlanCreated = False

            dbPlan = PlannedExperiment.objects.get(pk=self.id)

            #last modified date
            self.date = timezone.now()

            #historical paired-end plan (can't raise error because this could be for db schema change
            if (self.runMode == "pe" or self.isReverseRun or dbPlan.runMode == "pe" or dbPlan.isReverseRun):
                logger.warning("Paired-end plan is no longer supported even though it is being updated. Plan name=%s; id=%d" % (self.planDisplayedName, self.id))

            super(PlannedExperiment, self).save()
        else:
            #if user creates a new forward plan
            if (self.runMode == "pe" or self.isReverseRun):
                raise ValidationError("Error: paired-end plan is no longer supported. Plan %s will not be saved." % self.planDisplayedName)
            else:
                if PlannedExperiment.objects.filter(planShortID=self.planShortID, planExecuted=False):
                    self.planShortID = self.findShortID()

                if (self.sampleSet):
                        if (not self.sampleSet_uid):
                            self.sampleSet_uid = str(uuid.uuid4())
                        if (self.sampleSet_planTotal == 0):
                            self.sampleSet_planTotal = 1
                            self.sampleSet_planIndex = 0

                #logger.info('Going to CREATE the 1 UNTOUCHED plan with name=%s' % self.planName)
                super(PlannedExperiment, self).save()

                if (self.sampleSet):
                        if (self.sampleSet.status == "" or self.sampleSet.status == "created"):
                            self.sampleSet.status = "planned"
                            self.sampleSet.save()


    def save_plannedExperiment_association(self, isPlanCreated, **kwargs):
        """
        create/update the associated records for the plan

        1. for template: create or update exp, eas
        2. for plan: create or update exp, eas, samples
        3. for executed plan: update existing exp, eas, samples
        """
        logger.debug("PDD ENTER models.save_plannedExperiment_association() isPlanCreated=%s; self_id=%d; planExecuted=%s; kwargs=%s" %(str(isPlanCreated), self.id, str(self.planExecuted), kwargs))

        # ===================== Experiment ==================================
        exp_kwargs = {
            'autoAnalyze' : kwargs.get('x_autoAnalyze', True),
            'chipType' : kwargs.get('x_chipType', ""),
            'flows' : kwargs.get('x_flows', "0"),
            'isReverseRun' : kwargs.get('x_isReverseRun', False),
            'notes' : kwargs.get('x_notes', ""),
            'runMode' : self.runMode,
            'sequencekitname' : kwargs.get('x_sequencekitname', ""),
        }

        #there must be one experiment for each plan
        try:
            experiment = self.experiment
            for key,value in exp_kwargs.items():
                setattr(experiment, key, value)
            logger.debug("PDD models.save_plannedExperiment_association() going to UPDATE experiment id=%d" % experiment.id)

        except Experiment.DoesNotExist:
            # creating new experiment
            exp_kwargs.update({
                'date' : timezone.now(),
                'status' : self.planStatus,
                'expDir' : '',
                #temp expName value below will be replaced in crawler
                'expName' : self.planGUID,
                'displayName' : self.planShortID,
                'pgmName' : '',
                'log' : '',
                #db constraint requires a unique value for experiment. temp unique value below will be replaced in crawler
                'unique' : self.planGUID,
                'chipBarcode' : '',
                'seqKitBarcode' : '',
                'sequencekitbarcode' : '',
                'reagentBarcode' : '',
                'cycles' : 0,
                'expCompInfo' : '',
                'baselineRun' : '',
                'flowsInOrder' : '',
                'ftpStatus' : 'Complete',
                'displayName' : '',
                'storageHost' : ''
            })
            experiment = Experiment(**exp_kwargs)
            logger.debug("PDD models.save_plannedExperiment_association() #2 going to CREATE experiment")

        logger.debug("PDD models.save_plannedExperiment_association() self_id=%d exp_kwargs=..." %(self.id))
        logger.debug(exp_kwargs)

        #need this!
        experiment.plan = self
        experiment.save()
        logger.debug("PDD models.save_plannedExperiment_association() AFTER saving experiment_id=%d" %(experiment.id))

        # ===================== ExperimentAnalysisSettings =====================
        eas_kwargs = {
            'barcodedSamples' : kwargs.get('x_barcodedSamples', {}),
            'barcodeKitName' : kwargs.get('x_barcodeId', ""),
            'hotSpotRegionBedFile' : kwargs.get('x_regionfile', ""),
            'libraryKey' : kwargs.get('x_libraryKey', ""),
            'libraryKitName' : kwargs.get('x_librarykitname', ""),
            'reference' : kwargs.get('x_library', ""),
            'selectedPlugins' : kwargs.get('x_selectedPlugins', {}),
            'status' : self.planStatus,
            'targetRegionBedFile' : kwargs.get('x_bedfile', ""),
            'threePrimeAdapter' : kwargs.get('x_forward3primeadapter' ""),
            'isDuplicateReads' : kwargs.get('x_isDuplicateReads', False)
        }
        # add default cmdline args
        args = self.get_default_cmdline_args(libraryKitName=eas_kwargs['libraryKitName'])
        eas_kwargs.update(args)

        eas, eas_created = experiment.get_or_create_EAS(editable=True)
        for key,value in eas_kwargs.items():
            setattr(eas, key, value)
        eas.save()
        logger.debug("PDD models.save_plannedExperiment_association() AFTER saving EAS_id=%d" %(eas.id))

        # ===================== Samples ==========================================
        if not self.isReusable:
            #if this is not a template need to create/update single sample or multiple barcoded samples
            samples_kwargs = []
            barcodedSamples = kwargs.get('x_barcodedSamples', {})

            if barcodedSamples:
                barcodedSampleDict = simplejson.loads(barcodedSamples)
                for displayedName, sampleDict in barcodedSampleDict.items():
                    name = displayedName.replace(' ', '_')
                    if barcodedSampleDict[displayedName].get('barcodeSampleInfo'):
                        externalId = barcodedSampleDict[displayedName]['barcodeSampleInfo'].values()[0].get('externalId','')
                        description = barcodedSampleDict[displayedName]['barcodeSampleInfo'].values()[0].get('description','')
                    else:
                        externalId = ""
                        description = ""

                    samples_kwargs.append({
                        'name' : name,
                        'displayedName' : displayedName,
                        'date' : self.date,
                        'status' : self.planStatus,
                        'externalId': externalId,
                        'description': description
                    })
                logger.debug("PDD models.save_plannedExperiment_association() barcoded samples kwargs=")
                logger.debug(samples_kwargs)
            else:
                displayedName = kwargs.get('x_sampleDisplayedName', "")
                name = kwargs.get('x_sample', "") or displayedName
                if name:
                    samples_kwargs.append({
                        'name' : name.replace(' ', '_'),
                        'displayedName' : displayedName or name,
                        'date' : self.date,
                        'status' : self.planStatus,
                        'externalId': kwargs.get('x_sample_external_id', ""),
                        'description': kwargs.get('x_sample_description', "")
                    })
                    logger.debug("PDD models.save_plannedExperiment_association() samples kwargs=")
                    logger.debug(samples_kwargs)

            # add sample(s)
            for sample_kwargs in samples_kwargs:
                try:
                    sample = Sample.objects.get(name=sample_kwargs['name'], externalId=sample_kwargs['externalId'])
                    for key,value in sample_kwargs.items():
                        setattr(sample, key, value)
                except Sample.DoesNotExist:
                    sample = Sample(**sample_kwargs)

                sample.save()
                sample.experiments.add(experiment)

                logger.debug("PDD models.save_plannedExperiment_association() AFTER saving sample_id=%d" %(sample.id))

            # clean up old samples that may remain on experiment
            sample_names = [ d['name'] for d in samples_kwargs ]
            for sample in experiment.samples.all():
                if sample.name not in sample_names:
                    experiment.samples.remove(sample)

                #delete sample if it is not associated with any experiments
                if sample.experiments.all().count() == 0 and not (sample.status == "created"):
                    logger.debug("PDD models.save_plannedExperiment_association() going to DELETE sample=%s" %(sample.name))
                    sample.delete()


    def update_plan_qcValues(self, **kwargs):
        qcTypes = QCType.objects.all()
        for qcType in qcTypes:
            qc_threshold = kwargs.get(qcType.qcName, '')
            if qc_threshold:
                try:
                    plannedExpQc = PlannedExperimentQC.objects.get(plannedExperiment=self, qcType=qcType)
                    plannedExpQc.threshold = qc_threshold
                    plannedExpQc.save()
                except PlannedExperimentQC.DoesNotExist:
                    plannedExpQc_kwargs = {
                        'plannedExperiment': self,
                        'qcType': qcType,
                        'threshold': qc_threshold
                    }
                    plannedExpQc = PlannedExperimentQC(**plannedExpQc_kwargs)
                    plannedExpQc.save()

    def get_default_cmdline_args(self, **kwargs):
        # retrieve args from AnalysisArgs table
        chipType = kwargs.get('chipType') or self.get_chipType()
        sequenceKitName = kwargs.get('sequenceKitName') or self.get_sequencekitname()
        templateKitName = kwargs.get('templateKitName') or self.templatingKitName
        libraryKitName = kwargs.get('libraryKitName') or self.get_librarykitname()
        samplePrepKitName = kwargs.get('samplePrepKitName') or self.samplePrepKitName
        
        args = AnalysisArgs.best_match(chipType, sequenceKitName, templateKitName, libraryKitName, samplePrepKitName)
        if args:
            args_dict = args.get_args()
        else:
            args_dict = {
                'beadfindargs':   'justBeadFind',
                'analysisargs':   'Analysis',
                'basecallerargs': 'BaseCaller',
                'prebasecallerargs': 'BaseCaller',
                'alignmentargs': '',
                'thumbnailbeadfindargs':    'justBeadFind',
                'thumbnailanalysisargs':    'Analysis',
                'thumbnailbasecallerargs':  'BaseCaller',
                'prethumbnailbasecallerargs':  'BaseCaller',
                'thumbnailalignmentargs': ''
            }
            logger.error('No default command line args found for chip type = %s' % chipType)
            
        return args_dict

    class Meta:
        ordering = [ '-id' ]


class PlannedExperimentQC(models.Model):
    plannedExperiment = models.ForeignKey(PlannedExperiment)
    qcType = models.ForeignKey(QCType)
    threshold = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ( ('plannedExperiment', 'qcType'), )

class Experiment(models.Model):
    _CSV_METRICS = (('Sample', 'sample'),
                    #('Project', 'project'),
                    #('Library', 'library'),
                    ('Notes', 'notes'),
                    ('Run Name', 'expName'),
                    ('PGM Name', 'pgmName'),
                    ('Run Date', 'date'),
                    ('Run Directory', 'expDir'),
                    )
    STORAGE_CHOICES = (
        ('KI', 'Keep'),
        ('A', 'Archive Raw'),
        ('D', 'Delete Raw'),
    )
    # Archive action states
    ACK_CHOICES = (
        ('U', 'Unset'),         # (1) No action is pending
        ('S', 'Selected'),      # (2) Selected for action
        ('N', 'Notified'),      # (3) user has been notified of action pending
        ('A', 'Acknowledged'),  # (4) user acknowledges to proceed with action
        ('P', 'Partial'),       # (5) Some deletion has occurred
        ('I', 'In-process'),    # (6) Being processed by system
        ('D', 'Disposed'),      # (7) Action has been completed
        ('E', 'Error'),         # (8) Action resulted in error
    )
    PRETTY_PRINT_RE = re.compile(r'R_(\d{4})_(\d{2})_(\d{2})_(\d{2})'
                                 '_(\d{2})_(\d{2})_')
    # raw data lives here absolute path prefix
    expDir = models.CharField(max_length=512)
    expName = models.CharField(max_length=128,db_index=True)
    displayName = models.CharField(max_length=128,default="")
    pgmName = models.CharField(max_length=64)
    log = json_field.JSONField(blank=True)
    unique = models.CharField(max_length=512, unique=True)
    date = models.DateTimeField(db_index=True)
    resultDate = models.DateTimeField(auto_now_add=True, db_index=True, blank=True, null=True)
    repResult = models.OneToOneField('Results', null=True, blank=True, related_name="+", on_delete=models.SET_NULL)
    pinnedRepResult = models.BooleanField("pinned representative result", default=False)

    #deprecated as of TS 3.6: see DMFileStat and DMFileSet class
    storage_options = models.CharField(max_length=200, choices=STORAGE_CHOICES,
                                       default='A')

    user_ack = models.CharField(max_length=24, choices=ACK_CHOICES,default='U')
    notes = models.CharField(max_length=1024, blank=True, null=True)
    chipBarcode = models.CharField(max_length=64, blank=True)
    seqKitBarcode = models.CharField(max_length=64, blank=True)
    reagentBarcode = models.CharField(max_length=64, blank=True)
    autoAnalyze = models.BooleanField()
    usePreBeadfind = models.BooleanField()
    chipType = models.CharField(max_length=32)
    cycles = models.IntegerField()
    flows = models.IntegerField()
    expCompInfo = models.TextField(blank=True)
    baselineRun = models.BooleanField()
    flowsInOrder = models.TextField(blank=True)
    star = models.BooleanField()
    ftpStatus = models.CharField(max_length=512, blank=True)
    storageHost = models.CharField(max_length=128, blank=True, null=True)
    reverse_primer = models.CharField(max_length=128, blank=True, null=True)
    rawdatastyle = models.CharField(max_length=24, blank=True, null=True, default='single')

    sequencekitname = models.CharField(max_length=512, blank=True, null=True)
    sequencekitbarcode = models.CharField(max_length=512, blank=True, null=True)

    isReverseRun = models.BooleanField(default=False)
    diskusage = models.IntegerField(blank=True, null=True)

    #progress_flows = models.IntegerField(default=0)
    #progress_status =

    #add metadata
    metaData = json_field.JSONField(blank=True)

    ALLOWED_RUN_MODES = (
        ('', 'Undefined'),
        ('single', 'SingleRead'),
        ('pe', 'PairedEnd')
    )

    #run mode
    runMode = models.CharField(max_length=64, choices=ALLOWED_RUN_MODES, default='', blank=True)

    # plan may be null because Experiments do not Have to have a plan
    # and it may be blank for the same reason.
    plan = models.OneToOneField(PlannedExperiment, blank=True, null=True, related_name='experiment')

    #Typical status transition of a plan/experiment:
    # pending (optional) -> planned -> run (e.g., from LIMS or HUB)
    # reserved (optional)-> planned -> run (e.g., from Sample Prep instrument like IonChef)
    #
    #A plan's status can be cascaded down to experiment.
    #For more details, refer to the plannedExperiment's planStatus comments
    ALLOWED_STATUS = (
        ('', 'Undefined'),
        ('pending', 'Pending'),
        ('voided', 'Voided'),
        ('reserved', 'Reserved'),
        ('planned', 'Planned'),
        ('run', 'Run')
    )

    status = models.CharField(max_length=512, blank=True, choices=ALLOWED_STATUS, default='')

    def __unicode__(self): return self.expName

    def runtype(self):
        runType = self.log.get("runtype","")
        return runType if runType else "GENS"

    def pretty_print(self):
        nodate = self.PRETTY_PRINT_RE.sub("", self.expName)
        ret = " ".join(nodate.split('_')).strip()
        if not ret:
            return nodate
        return ret

    def pretty_print_no_space(self):
        return self.pretty_print().replace(" ","_")

    def sorted_results(self):
        try:
            ret = self.results_set.all().order_by('-timeStamp')
        except IndexError:
            ret = None
        return ret
    def sorted_results_with_reports(self):
        """returns only results that have valid reports, in inverse time order"""
        try:
            ret = [r for r in self.results_set.all().order_by('-timeStamp') if r.report_exist()]
        except IndexError:
            ret = None
        return ret
    def get_storage_choices(self):
        return self.STORAGE_CHOICES
    def best_result(self, metric):
        try:
            rset = self.results_set.all()
            rset = rset.exclude(libmetrics__i50Q17_reads=0)
            rset = rset.exclude(libmetrics=None)
            rset = rset.order_by('-libmetrics__%s' % metric)[0]
        except IndexError:
            rset = None
        return rset
    def best_aq17(self):
        """best 100bp aq17 score"""
        rset = self.results_set.all()
        rset = rset.exclude(libmetrics=None)

        best_report = rset.order_by('-libmetrics__i100Q17_reads')
        sampled_best_report = rset.order_by('-libmetrics__extrapolated_100q17_reads')

        if not best_report and not sampled_best_report:
            return False

        best_report = best_report[0].libmetrics_set.all()[0].i100Q17_reads
        sampled_best_report = sampled_best_report[0].libmetrics_set.all()[0].extrapolated_100q17_reads

        if best_report > sampled_best_report:
            return rset.order_by('-libmetrics__i100Q17_reads')
        else:
            return rset.order_by('-libmetrics__extrapolated_100q17_reads')

    #TODO: deprecated in TS3.6
    def available(self):
        try:
            backup = Backup.objects.get(backupName=self.expName)
        except:
            return False
        if backup.backupPath == 'DELETED':
            return 'Deleted'
        if backup.backupPath == 'PARTIAL-DELETE':
            return 'Partial-delete'
        if backup.backupPath == 'PURGED':
            return 'Purged'
        if backup.isBackedUp:
            return 'Archived'


    def isBarcoded(self):
        try:
            eas = self.eas_set.all().order_by("-date")[0]
            if eas.barcodeKitName:
                return True
            else:
                return False
        except:
            return False

    def get_or_create_EAS(self, editable=False, reusable=True):
        """
        Retrieve latest experimentAnalysisSettings that is editable and/or reusable for this experiment
        Special case: If caller requests for EAS that is both editable and reusable, attempt to retrieve
        the latest EAS that meets the criteria. If not found, copy the latest reusable EAS.
        If no appropriate EAS is found, create one based on latest reusable EAS.
        """

        eas_set = self.eas_set.all().order_by("-date", "-id")

        if editable:
            queryset = eas_set.filter(isEditable=True)
            if eas_set.filter(isOneTimeOverride = False):
                eas_set = eas_set.filter(isOneTimeOverride = False)
        elif reusable:
            queryset = eas_set.filter(isOneTimeOverride=False)
        else:
            queryset = eas_set

        created = False
        try:
            eas = queryset[0]
        except:
            created = True
            try:
                eas = eas_set[0]

                eas.pk = None # this will create a new instance
                eas.date = timezone.now()
                eas.isEditable = True
                eas.isOneTimeOverride = False
                eas.save()
            except:
                logger.debug('No ExperimentAnalysisSettings found for experiment id=%d' % self.pk)

                # generate blank EAS
                default_eas_kwargs = {
                              'date' : timezone.now(),
                              'experiment' : self,
                              'libraryKey' : GlobalConfig.objects.all()[0].default_library_key,
                              'isEditable' : True,
                              'isOneTimeOverride' : False,
                              'status' : 'run',
                              }
                eas = ExperimentAnalysisSettings(**default_eas_kwargs)
                eas.save()

        return (eas, created)


    def get_EAS(self, editable=True, reusable=True):
        """
        Retrieve latest experimentAnalysisSettings that is editable and/or reusable for this experiment.
        Special case: If caller requests for EAS that is both editable and reusable, attempt to retrieve
        the latest EAS that meets the criteria. If not found, attempt to retrieve the latest reusable EAS.
        Return None if no EAS matching the criteria is found.
        """
        eas_set = self.eas_set.all().order_by("-date", "-id")

        queryset = None
        if editable and reusable:
            queryset = eas_set.filter(isEditable = True, isOneTimeOverride = False)
            if not queryset:
                queryset = eas_set.filter(isEditable = False, isOneTimeOverride = False)

        if not editable and not reusable:
            queryset = eas_set.filter(isEditable = False, isOneTimeOverride = True)

        if not queryset:
            if editable:
                queryset = eas_set.filter(isEditable=True)

            elif reusable:
                queryset = eas_set.filter(isOneTimeOverride=False)
            else:
                queryset = eas_set

        if queryset.exists():
            return queryset[0]
        else:
            return None

    def get_barcodeId(self):
        latest_eas = self.get_EAS()
        if latest_eas:
            return latest_eas.barcodeKitName
        else:
            return ""

    def get_library(self):
        latest_eas = self.get_EAS()
        if latest_eas:
            return latest_eas.reference
        else:
            return ""

    def _get_firstSampleValue(self, attribute):
        if not attribute:
            return ""

        samples = self.samples.values_list(attribute, flat=True)

        try:
            firstSample = str(samples[0])
        except IndexError:
            firstSample = ""

        return firstSample

    def get_sample(self):
        """Return the first sample name"""
        return self._get_firstSampleValue('name')

    def get_sampleDisplayedName(self):
        """Return the first sample displayed name"""
        return self._get_firstSampleValue('displayedName')

    def isProton(self):
        if self.chipType:
            chips = Chip.objects.filter(name = self.chipType)
            if chips:
                chip = chips[0]
                return (chip.instrumentType == "proton")
            else:
                chipPrefix = self.chipType[:3]
                chips = Chip.objects.filter(name = chipPrefix)
                if chips:
                    chip = chips[0]
                    return (chip.instrumentType == "proton")

            #if somehow the chip is not in the chip table but it starts with p...
            if (self.chipType[:1].lower() == 'p'):
                return True

        return False


    def save(self):
        """on save we need to sync up the log JSON and the other values that might have been set
        this was put in place primarily for the runtype field"""

        #Make sure that that pretty print name is saved as the display name for the experiment
        self.displayName = self.pretty_print()

        #we now save the sequencing kit name instead of the kit's part number to Experiment
        #when updating a pre-existing run from TS, we want to populate the new field with the info
        if ((self.sequencekitbarcode or self.seqKitBarcode) and not self.sequencekitname):
            try:
                kitBarcode = self.sequencekitbarcode
                if (not kitBarcode):
                    kitBarcode = self.seqKitBarcode

                selectedSeqKitPart = KitPart.objects.get(barcode=kitBarcode)
                if (selectedSeqKitPart):
                    selectedSeqKit = selectedSeqKitPart.kit
                    if (selectedSeqKit):
                        self.sequencekitname = selectedSeqKit.name
            except KitPart.DoesNotExist:
                #if we can't determine the seq kit name, leave it as is
                #do not fail the save()
                logger.info("NO kit part found at Experiment for sequencingKitBarcode=%s" % kitBarcode)


        super(Experiment, self).save()

    def getdeviceid(self):
        '''Returns deviceid of the storage of the raw data directory'''
        return os.stat(self.expDir)[2]



@receiver(pre_delete, sender=Experiment, dispatch_uid="pre_delete_experiment")
def on_experiment_preDelete(sender, instance, **kwargs):
    """Going to delete the experiment.
    """
    plan = instance.plan
    if plan:
        instance.plan = None
        instance.save()
        plan.delete()

    for sample in instance.samples.all():
        instance.samples.remove(sample)
        #if sample is created via loading instead of plan/run creation, keep the sample around!!
        if (sample.experiments.count() == 0) and not (sample.status == "created"):
            sample.delete()

    logger.info("Deleting companion records before deleting experiment=%s; pk=%d" % (instance.expName, instance.id))


@receiver(post_delete, sender=Experiment, dispatch_uid="delete_experiment")
def on_experiment_delete(sender, instance, **kwargs):
    """Log the deletion of the Experiment.
    """
    logger.info("Deleting Experiment %d" % (instance.id))


class ExperimentAnalysisSettings(models.Model):
    """
    Supports versioning.  Once used for analysis, no editing is allowed and a new instance will be created for the edits
    """

    #planned status: means the entire plan object has been saved but not yet used
    #run status    : means the plan has been claimed for a run and the corresponding sequencing run has started
    ALLOWED_STATUS = (
        ('', 'Undefined'),
        ('planned', 'Planned'),
        ('run', 'Run')
    )

    status = models.CharField(max_length=512, blank=True, choices=ALLOWED_STATUS, default='')

    barcodeKitName = models.CharField(max_length=128, blank=True, null=True)

    libraryKey = models.CharField(max_length=64, blank=True)

    libraryKitName = models.CharField(max_length=512, blank=True, null=True)
    libraryKitBarcode = models.CharField(max_length=512, blank=True, null=True)

    threePrimeAdapter = models.CharField("3' adapter", max_length=512, blank=True, null=True)

    #bed file
    #Target Regions BED File: old name is bedfile
    #Hotspot Regions BED File: old name is regionfile
    targetRegionBedFile = models.CharField(max_length=1024,blank=True,null=True)
    hotSpotRegionBedFile = models.CharField(max_length=1024,blank=True,null=True)

    reference = models.CharField(max_length=512, blank=True, null=True)

    barcodedSamples = json_field.JSONField(blank=True, null=True)
    selectedPlugins = json_field.JSONField(blank=True, null=True)

    date = models.DateTimeField(blank=True, null=True)
    isEditable = models.BooleanField(default=False)

    #for reanalysis, user can enter parameters that are intended for just this re-analysis attempt
    isOneTimeOverride = models.BooleanField(default=False)

    # foreign key to the experiment
    experiment = models.ForeignKey(Experiment, related_name='eas_set', blank=True, null=True)

    isDuplicateReads = models.BooleanField(default=False)

    # Beadfind args
    beadfindargs = models.CharField(max_length=5000, blank=True, verbose_name="Beadfind args")
    thumbnailbeadfindargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Beadfind args")
    # Analysis args
    analysisargs = models.CharField(max_length=5000, blank=True, verbose_name="Analysis args")
    thumbnailanalysisargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Analysis args")
    # PreBasecaller args
    prebasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Pre Basecaller args, used for recalibration")
    prethumbnailbasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Pre Basecaller args, used for recalibration")
    # Basecaller args
    basecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Basecaller args")
    thumbnailbasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Basecaller args")
    # Alignment args
    alignmentargs = models.CharField(max_length=5000, blank=True, verbose_name="Alignment args")
    thumbnailalignmentargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Alignment args")

    def __unicode__(self):
        return "%s/%d" % (self.experiment, self.pk)

    class Meta:
        verbose_name_plural = "Experiment Analysis Settings"



class SampleGroupType_CVManager(models.Manager):
    def get_by_natural_key(self, uid):
        return self.get(uid = uid)

class SampleGroupType_CV(models.Model):
    '''
    To support sample and sample set creation regardless a TS instance is IR-enabled or not,
    we need to define the necessary terms in TS and flag a term that has an IR-eqivalence.
    '''
    displayedName = models.CharField(max_length = 127, blank = False, null = False, unique = True)
    description = models.CharField(max_length = 1024, blank = True, null = True)

    isIRCompatible = models.BooleanField(default = False)
    iRAnnotationType = models.CharField(max_length = 127, blank = True, null = True)
    iRValue = models.CharField(max_length = 127, blank = True, null = True)
    isActive = models.BooleanField(default = True)

    #keep uid, displayedName value might change per requirement
    uid = models.CharField(max_length = 32, unique = True, blank = False)

    objects = SampleGroupType_CVManager()

    def __unicode__(self):
        return u'%s' % (self.displayedName)

    def natural_key(self):
        return (self.uid,)  # must return a tuple


class SampleAnnotation_CVManager(models.Manager):
    def get_by_natural_key(self, uid):
        return self.get(uid = uid)

class SampleAnnotation_CV(models.Model):
    value = models.CharField(max_length = 127, blank = True, null = False)

    ALLOWED_TYPES = (
        ('gender', 'Gender'),
        ('relationship', 'Relationship'),
        ('relationshipRole', 'RelationshipRole'),
        ('relationshipGroup', 'RelationshipGroup')
    )

    annotationType = models.CharField(max_length = 127, blank = False, null = False, choices = ALLOWED_TYPES)

    isIRCompatible = models.BooleanField(default = False)
    iRAnnotationType = models.CharField(max_length = 127, blank = True, null = True)
    iRValue = models.CharField(max_length = 127, blank = True, null = True)
    isActive = models.BooleanField(default = True)

    #optional n-to-n between sampleMetaData and SampleGroupType_CV (some meta data will not belong to a SampleGroupType_CV (e.g., gender)
    sampleGroupType_CV = models.ForeignKey(SampleGroupType_CV, related_name = "sampleAnnotation_set", blank = True, null = True)

    uid = models.CharField(max_length = 32, unique = True, blank = False)

    objects = SampleAnnotation_CVManager()

    def __unicode__(self):
        return u'%s' % self.value

    def natural_key(self):
        return (self.uid,)  # must return a tuple


class SampleSet(models.Model):
    displayedName = models.CharField(max_length = 127, blank = False, null = False, unique = True)
    description = models.CharField(max_length=1024, blank=True, null = True)

    creator = models.ForeignKey(User, related_name = "created_sampleSet")
    # This will be set to the time a new record is created
    creationDate = models.DateTimeField(auto_now_add = True)

    lastModifiedUser = models.ForeignKey(User, related_name = "lastModified_sampleSet")

    # This will be set to the current time every time the model is updated
    lastModifiedDate = models.DateTimeField(auto_now = True)

    #many sampleSets can have the same SampleGroupType_CV but a sampleSet can only have 1 SampleGroupType_CV
    SampleGroupType_CV = models.ForeignKey(SampleGroupType_CV, related_name = "sampleSets", null=True)

    #created status       : means the sample set is created for sample-driven planning. it may or may not have plans associated with it.
    #planned status      : means the sample set has at least one plan created for it but is not yet used
    #run status          : means the sample set has at least one plan that with "planExecuted" set to True
    ALLOWED_SAMPLESET_STATUS = (
        ('', 'Undefined'),
        ('created', 'Created'),
        ('planned', 'Planned'),
        ('run', 'Run')
    )

    status = models.CharField(max_length=512, blank=True, choices=ALLOWED_SAMPLESET_STATUS, default='')


    def __unicode__(self):
        return u'%s' % (self.displayedName)

class SampleSetItem(models.Model):
    #reference Sample but Sample class is defined after SampleSetItem!!!

    #a sample can be in many sampleSets but a sample can only associate with one sampleSetItem
    sample = models.ForeignKey("Sample", related_name = "sampleSets", blank = False, null = False)

    #a sampleSet can have many sampleSetItem but a sampleSetItem can only associate with one sampleSet
    sampleSet = models.ForeignKey(SampleSet, related_name = "samples", blank = False, null = False)

    gender = models.CharField(max_length = 127, blank = True, null = True)
    relationshipRole = models.CharField(max_length = 127, blank = True, null = True)
    relationshipGroup = models.IntegerField()

    creator = models.ForeignKey(User, related_name = "created_sampleSetItem")

    # This will be set to the time a new record is created
    creationDate = models.DateTimeField(auto_now_add = True)

    lastModifiedUser = models.ForeignKey(User, related_name = "lastModified_sampleSetItem")

    # This will be set to the current time every time the model is updated
    lastModifiedDate = models.DateTimeField(auto_now = True)

    #optional sample-dnabarcode.id_str assignment
    barcode = models.CharField(max_length = 128, blank = True, null = True)
    
    def __unicode__(self):
        return u'%s/%s/%d' % (self.sampleSet, self.sample, self.relationshipGroup)


class SampleAttributeDataType(models.Model):
    ALLOWED_TYPES = (
        ('Text', 'Text'),
        ('Number', 'Number')
    )

    dataType = models.CharField(max_length = 64, blank = False, null = False, unique = True, choices = ALLOWED_TYPES)
    description = models.CharField(max_length=1024, blank=True, null = True)
    isActive = models.BooleanField(default = True)

    def __unicode__(self):
        return self.dataType


class SampleAttribute(models.Model):
    displayedName = models.CharField(max_length = 127, blank = False, null = False, unique = True)
    description = models.CharField(max_length=1024, blank=True, null = True)

    isMandatory = models.BooleanField(default = False)
    isActive = models.BooleanField(default = True)

    creator = models.ForeignKey(User, related_name = "created_sampleAttribute")

    # This will be set to the time a new record is created
    creationDate = models.DateTimeField(auto_now_add = True)

    lastModifiedUser = models.ForeignKey(User, related_name = "lastModified_sampleAttribute")

    # This will be set to the current time every time the model is updated
    lastModifiedDate = models.DateTimeField(auto_now = True)

    dataType = models.ForeignKey(SampleAttributeDataType, related_name = "sampleAttributes")

    def __unicode__(self):
        return self.displayedName


class SampleAttributeValue(models.Model):
    value = models.CharField(max_length = 1024, blank = True, null = True)

    creator = models.ForeignKey(User, related_name = "created_sampleAttributeValue")

    # This will be set to the time a new record is created
    creationDate = models.DateTimeField(auto_now_add = True)

    lastModifiedUser = models.ForeignKey(User, related_name = "lastModified_sampleAttributeValue")

    # This will be set to the current time every time the model is updated
    lastModifiedDate = models.DateTimeField(auto_now = True)

    sample = models.ForeignKey("Sample", related_name = "sampleAttributeValues")
    sampleAttribute = models.ForeignKey(SampleAttribute, related_name = "samples")

    def __unicode__(self):
        return u'%s' % (self.value)

class Sample(models.Model):
    #created status       : means the sample is created for sample-driven planning. it may or may not have plans associated with it.
    #planned status      : means the sample is created as a by-product of a plan entire plan object has been saved but not yet used
    #run status          : means the plan has been claimed for a run and the corresponding sequencing run has started
    ALLOWED_STATUS = (
        ('', 'Undefined'),
        ('created', 'Created'),
        ('planned', 'Planned'),
        ('run', 'Run')
    )

    status = models.CharField(max_length=512, blank=True, choices=ALLOWED_STATUS, default='')

    name = models.CharField(max_length=127, blank=True, null=True)
    displayedName = models.CharField(max_length=127, blank=True, null=True)
    externalId = models.CharField(max_length=127, blank=True, null=True, default = '')
    description = models.CharField(max_length=1024, blank=True, null=True)
    date = models.DateTimeField(auto_now_add=True, blank=True,null=True)

    experiments = models.ManyToManyField(Experiment, related_name='samples', null=True)

    def __unicode__(self):
        return self.name

    class Meta:
        unique_together=( ('name', 'externalId'), )


class Lookup(object):
    _ALIASES = {}
    def lookup(self, path):
        def alias(e):
            key = e.lower().replace(' ', '')
            if key in self._ALIASES:
                return self._ALIASES[key]
            return e
        def down(obj, name):
            if not hasattr(obj, name):
                return None
            else:
                return getattr(obj, name)
        elements = os.path.split('.')
        curr = self
        for e in elements:
            if curr is None:
                break
            key = alias(e)
            curr = down(curr, key)
        return curr
    def tabulate(self, fields=None):
        if fields is None:
            fields = self.TABLE_FIELDS
        return [self.lookup(f) for f in fields]
    @classmethod
    def to_table(cls, qset):
        rows = [cls.TABLE_FIELDS]
        empty = ''
        for res in qset:
            rows.append([ele or empty for ele in res.tabulate()])
        return rows

class Results(models.Model, Lookup):
    _CSV_METRICS = (("Report", "resultsName"),
                    ("Status", 'status'),
                    ("Flows", 'processedflows'),
                    ("Library", 'reference')
                    #("Plugin Data", 'pluginStore')
                    )
    _ALIASES = {
        "report":"resultsName",
        #"experiment":"experiment",
        "date":"timeStamp",
        "status":"status",
        "Flows":"processedflows",
        "q17mean":"best_metrics.Q17Mean",
        "q17mode":"best_metrics.Q17Mode",
        "systemsnr":"best_metrics.SysSNR",
        "q17reads":"best_metrics.Q17ReadCount",
        "keypass":"best_metrics.keypass",
        "cf":"best_metrics.CF"
        }
    TABLE_FIELDS = ("Report", "Status", "Flows",
                    "Lib Key Signal",
                     "Q20 Bases", "100 bp AQ20 Reads", "AQ20 Bases")
    PRETTY_FIELDS = TABLE_FIELDS
    experiment = models.ForeignKey(Experiment, related_name='results_set')
    representative = models.BooleanField(default=False)
    resultsName = models.CharField(max_length=512)
    timeStamp = models.DateTimeField(auto_now_add=True, db_index=True)
    sffLink = models.CharField(max_length=512)    #absolute paths
    fastqLink = models.CharField(max_length=512)  #absolute paths
    reportLink = models.CharField(max_length=512)  #absolute paths
    status = models.CharField(max_length=64)
    tfSffLink = models.CharField(max_length=512)    #absolute paths
    tfFastq = models.CharField(max_length=512)    #absolute paths
    log = models.TextField(blank=True)
    analysisVersion = models.CharField(max_length=256)
    processedCycles = models.IntegerField()
    processedflows = models.IntegerField()
    framesProcessed = models.IntegerField()
    timeToComplete = models.CharField(max_length=64)
    reportstorage = models.ForeignKey("ReportStorage", related_name="storage", blank=True, null=True)
    runid = models.CharField(max_length=10,blank=True)

    #deprecated as of TS 3.6: see DMFileStatand DMFileSet class
    reportStatus = models.CharField(max_length=64, null=True, default="Nothing")

    autoExempt = models.BooleanField(default=False)

    projects = models.ManyToManyField(Project, related_name='results')
    # reference genome for alignment
    reference = models.CharField(max_length=64, blank=True, null=True)
    # a way to make Multi-Reports:
    resultsType = models.CharField(max_length=512, blank=True)
    #ids of parent reports separated by colon (e.g :1:32:3:)
    parentIDs = models.CharField(max_length=512, blank=True)
    diskusage = models.IntegerField(blank=True, null=True)

    #metaData
    metaData = json_field.JSONField(blank=True)

    # foreign key to the experimentAnalysisSettings
    eas = models.ForeignKey(ExperimentAnalysisSettings, related_name='results_set', blank=True, null=True)

    analysismetrics = models.OneToOneField('AnalysisMetrics', null=True, blank=True)
    libmetrics = models.OneToOneField('LibMetrics', null=True, blank=True)
    qualitymetrics = models.OneToOneField('QualityMetrics', null=True, blank=True)

    def save(self, *args, **kwargs):

        super(Results, self).save(*args, **kwargs)

        # Test for related DMFileStat objects
        if DMFileStat.objects.filter(result=self).count() == 0:
            # Create data management file tracking objects
            try:
                dmfilesets = DMFileSet.objects.filter(version=iondb.settings.RELVERSION)
                #dmfilesets = DMFileSet.objects.all()
            except:
                logger.exception(traceback.format_exc())
            else:
                previous_result = None
                if self.experiment.results_set.exclude(pk=self.pk).count() > 0:
                    previous_result = self.experiment.results_set.exclude(pk=self.pk)[0]

                for dmfileset in dmfilesets:
                    try:
                        if dmfileset.type == dmactions_types.SIG and previous_result is not None:
                            # copy previous Sigproc dmfilestat
                            dmfilestat = previous_result.get_filestat(dmactions_types.SIG)
                            dmfilestat.result = self
                            dmfilestat.pk = None
                        else:
                            kwargs = {
                                'result':self,
                                'dmfileset':dmfileset,
                            }
                            dmfilestat = DMFileStat(**kwargs)
                        dmfilestat.save()
                    except:
                        logger.exception(traceback.format_exc())
                    else:
                        #EventLog.objects.add_entry(self,"Created DMFileStat (%s)" % dmfileset.type)
                        pass

    def get_filestat(self, typeStr):
        return self.dmfilestat_set.filter(dmfileset__type=typeStr)[0]

    def isProton(self):
        if self.experiment:
            return self.experiment.isProton()

        return False

    #a place for plugins to store information
    # NB: These two functions facilitate compatibility with earlier model,
    # which had pluginStore and pluginState as members
    #pluginStore = json_field.JSONField(blank=True)
    def getPluginStore(self):
        pluginDict = {}
        for p in self.pluginresult_set.all().order_by('id').select_related('plugin__name'):
            pluginDict[p.plugin.name] = p.store
        return pluginDict
    #pluginState = json_field.JSONField(blank=True)
    def getPluginState(self):
        pluginDict = {}
        for p in self.pluginresult_set.all().order_by('id').select_related('plugin__name'):
            pluginDict[p.plugin.name] = p.state
        return pluginDict

    def planShortID(self):
        expLog = self.experiment.log
        try:
            plan = expLog["planned_run_short_id"]
        except KeyError:
            plan = expLog.get("pending_run_short_id","")
        return plan

    def projectNames(self):
      names = [p.name for p in self.projects.all().order_by('-modified')]
      return ','.join(names)

    def sffLinkPatch(self):
        link = self.fastqLink.replace("fastq","sff")
        return link if os.path.exists(link) else None

    def sffTFLinkPatch(self):
        link = self.fastqLink.replace("fastq","tf.sff")
        return link if os.path.exists(link) else None

    def bamLink(self):
        """a method to used by the API to provide a link to the bam file"""
        location = self.server_and_location()

        if location is not False:
            bamFile = self.experiment.expName + "_" + self.resultsName + ".bam"
            webPath = self.web_path(location)
            if not webPath:
                logger.warning("Bam link, webpath missing for " + bamFile)
                return False
            return os.path.join(webPath , bamFile)
        else:
            logger.warning("Bam link, Report Storage: %s, Location: %s", self.reportstorage, location)
            return False

    def reportWebLink(self):
        """a method to used get the web url with no fuss"""
        location = self.server_and_location()

        if location is not False:
            webPath = self.web_path(location)
            if not webPath:
                logger.warning("Web link, webpath missing for %s" % self)
                return False
            return webPath
        else:
            logger.warning("Web link, Report %s has no storage location")
            return False


    def verboseStatus(self):
        if self.status.lower() == "completed":
            return "The run analysis has completed"
        elif self.status.lower() == "error":
            return "The run analysis failed, Please check run log for specific error"
        elif self.status.lower() == "terminated":
            return "User terminated analysis job"
        elif self.status.lower() == "started":
            return "The analysis is currently processing"
        elif self.status.lower() == "checksum":
            return "One of the raw signal files (DAT) is corrupt. Try re-transferring the data from the PGM"
        elif self.status.lower() == "pgm operation error":
            return "Unexpected raw data values. Please check PGM for clogs or problems with chip"
        else:
            return self.status

    def _basename(self):
        return "%s_%03d" % (self.resultsName, self.pk)

    def server_and_location(self):
        if getattr(self, '_location', None):
            return self._location
        try:
            loc = Rig.objects.get(name=self.experiment.pgmName).location
        except Rig.DoesNotExist:
            #logger.error("Rerport %s references Rig %s which does not exist." % (self, self.experiment.pgmName))
            loc = Location.objects.filter(defaultlocation=True)
            if not loc:
                #if there is not a default, just take the first one
                loc = Location.objects.all().order_by('pk')
            if loc:
                loc = loc[0]
            else:
                logger.critical("Report %s is requesting the location of PGM %s and no locations exist!" % (self, self.experiment.pgmName))
                return False
        setattr(self, '_location', loc)
        return loc

    def _findReportStorage(self):
        """
        Tries to determine the correct ReportStorage object by testing for
        a valid filesystem path using the reportLink path with the ReportStorage
        dirPath value.

        Algorithm for determining path from report link:
        strip off the first directory from report link and prepend the dirPath
        """
        logger.warning("Report %s is looking for it's storage." % self)
        storages = ReportStorage.objects.all()
        for storage in storages:
            link = os.path.dirname(self.reportLink)
            tmpPath = link.split('/')
            index = len(tmpPath) - 3
            linkstub = link.split('/' + tmpPath[index])
            new_path = storage.dirPath + linkstub[1]
            if os.path.exists(new_path):
                return storage
        return None

    def web_root_path(self, location):
        """Returns filesystem path to Results directory"""
        basename = self._basename()
        if self.reportstorage == None:
            storage = self._findReportStorage()
            if storage is not None:
                self.reportstorage = storage
                self.save()
        if self.reportstorage is not None:
            return os.path.join(self.reportstorage.dirPath, location.name, basename)
        else:
            return None

    def report_exist(self):
        """check to see if a report exists inside of the report path"""
        fs_path = self.get_report_path()
        #TODO: is this operation slowing down page loading?  on thousands of reports?
        return fs_path and os.path.exists(fs_path)

    def get_report_storage(self):
        """Returns reportstorage object"""
        if self.reportstorage == None:
            storage = self._findReportStorage()
            if storage is not None:
                self.reportstorage = storage
                self.save()
        return self.reportstorage

    def get_report_path(self):
        """Returns filesystem path to report file"""
        if self.reportstorage == None:
            storage = self._findReportStorage()
            if storage is not None:
                self.reportstorage = storage
                self.save()
        tmpPath = self.reportLink.split('/')
        index = len(tmpPath) - 4
        linkstub = self.reportLink.split('/' + tmpPath[index])
        if self.reportstorage is not None:
            return self.reportstorage.dirPath + linkstub[1]
        else:
            logger.error("Cannot find the path to %s" % self)
            return None

    def get_report_dir(self):
        """Returns filesystem path to results directory"""
        fs_path = self.get_report_path()
        if fs_path:
            fs_path = os.path.split(fs_path)[0]
        return fs_path

    def is_archived(self):
        '''Returns True in either case: Report Files deleted or archived'''
        # Pre-3.6 did not allow Report delete, only archive so we return either case here.
        return self.get_filestat(dmactions_types.OUT).isdisposed()

    def web_path(self, location):
        basename = self._basename()
        if self.reportstorage == None:
            storage = self._findReportStorage()
            if storage is not None:
                self.reportstorage = storage
                self.save()
        webServerPath = os.path.join(self.reportstorage.webServerPath, location.name, basename)

        #TODO: the webpath is not the same as the path of the filesystem. Check the webpath somehow.
        return webServerPath

    def __unicode__(self):
        return self.resultsName

    def updateMetaData(self, status, info, size, comment, logger = None):
        retLog = logger
        self.reportStatus = status
        self.save()

        if retLog:
            retLog.info(self.resultsName+": Status: "+status+" | Info: "+info + " | Comments: %s"%comment)
            # This string is parsed when calculating disk space saved:
            if size > 0: retLog.info(self.resultsName+": Size saved:* %7.2f KB"%(float(size/1024)))

        self.metaData["Status"] = status
        self.metaData["Date"] = "%s" % timezone.now()
        self.metaData["Info"] = info
        self.metaData["Comment"] = comment

        # Try to read the Log entry, if it does not exist, create it
        if len(self.metaData.get("Log",[])) == 0:
            self.metaData["Log"] = []
        self.metaData["Log"].append({"Status":self.metaData.get("Status"), "Date":self.metaData.get("Date"), "Info":self.metaData.get("Info"), "Comment":comment})
        self.save()

    # TODO: Cycles -> flows hack, very temporary.
    @property
    def processedFlowsorCycles(self):
        """This is an extremely intermediate hack, holding down the fort until
        cycles are removed from the model.
        """
        if self.processedflows:
            return self.processedflows
        else:
            return self.processedCycles * 4

    @property
    def best_metrics(self):
        try:
            ret = self.tfmetrics_set.all().order_by('-Q17Mean')[0]
        except IndexError:
            ret = None
        return ret
    @property
    def best_lib_metrics(self):
        try:
            ret = self.libmetrics_set.all().order_by('-i50Q17_reads')[0]
        except IndexError:
            ret = None
        return ret
    def best_lib_by_value(self, metric):
        try:
            ret = self.libmetrics_set.all().order_by('-%s' % metric)[0]
        except IndexError:
            ret = None
        return ret
    def pretty_tabulate(self):
        try:
            frags = self.tfmetrics_set.all().exclude(CF=0, IE=0, DR=0).order_by('name')
            return frags
        except:
            traceback.print_exc()
    def best_metrics_tabulate(self):
        return [self.lookup(f) for f in self.best_metrics_headings()]
    @classmethod
    def best_metrics_headings(cls):
        return cls.TABLE_FIELDS

    @classmethod
    def get_lib_metrics(cls, obj, metrics):
        q = obj.libmetrics_set.all()
        ret = []
        if len(q) > 0:
            for i in q:
                for metric in metrics:
                    ret.append(getattr(i, metric))
        else:
            for metrics in metrics:
                ret.append(' ')
        return tuple(ret)

    @classmethod
    def get_exp_metrics(cls, obj, metrics):
        q = obj.experiment
        ret = []
        for metric in metrics:
            if metric == 'sample':
                ret.append(q.get_sample())
            else:
                ret.append(getattr(q, metric))
        return ret

    @classmethod
    def get_analysis_metrics(cls, obj, metrics):
        q = obj.analysismetrics_set.all()
        ret = []
        if len(q) > 0:
            for i in q:
                for metric in metrics:
                    ret.append(getattr(i, metric))
        else:
            for metrics in metrics:
                ret.append(' ')
        return tuple(ret)
    @classmethod
    def get_tf_metrics(cls, obj, metrics):
        q = obj.tfmetrics_set.all()
        ret = []
        if len(q) > 0:
            for i in q:
                s = cls.get_result_metrics(obj, cls.get_values(cls._CSV_METRICS))
                for metric in metrics:
                    s.append(getattr(i, metric))
                ret.append(s)
        else:
            s = cls.get_result_metrics(obj, cls.get_values(cls._CSV_METRICS))
            for metric in metrics:
                s.append(' ')
            ret.append(s)
        return ret
    @classmethod
    def get_result_metrics(cls, obj, metrics):
        ret = []
        for metric in metrics:
            try:
                ret.append(getattr(obj, metric))
            except:
                ret.append(' ')
        return ret
    @classmethod
    def get_plugin_metrics(cls, obj, metrics):
        q = obj.pluginresult_set.all()
        ret = []
        if len(q) > 0:
            for i in q:
                try:
                    ret.append(json.dumps({i.plugin.name: i.store }))
                except TypeError:
                    #there are some hosts which had bad results.json in the past. Skip pumping that data out
                    pass
        return tuple(ret)

    @classmethod
    def getNameFromTable(cls, metrics):
        ret = []
        for k, v in metrics.iteritems():
            ret.append(k)
        return tuple(ret)
    @classmethod
    def get_values(cls, listotuple):
        values = []
        for ele in listotuple:
            values.append(ele[1])
        return tuple(values)
    @classmethod
    def get_keys(csl, listotuple):
        keys = []
        for ele in listotuple:
            keys.append(ele[0])
        return tuple(keys)
    @classmethod
    def to_pretty_table(cls, qset):
        ret = [cls.get_keys(cls._CSV_METRICS)
               + cls.get_keys(TFMetrics._CSV_METRICS)
               + cls.get_keys(LibMetrics._CSV_METRICS)
               + cls.get_keys(Experiment._CSV_METRICS)
               + cls.get_keys(AnalysisMetrics._CSV_METRICS)
               + cls.get_keys(PluginResult._CSV_METRICS)
               ]
        for obj in qset:
            new = cls.get_tf_metrics(obj, cls.get_values(TFMetrics._CSV_METRICS))
            if len(new) > 0:
                new[0].extend(cls.get_lib_metrics(obj, cls.get_values(LibMetrics._CSV_METRICS)))
                new[0].extend(cls.get_exp_metrics(obj, cls.get_values(Experiment._CSV_METRICS)))
                new[0].extend(cls.get_analysis_metrics(obj, cls.get_values(AnalysisMetrics._CSV_METRICS)))
                new[0].extend(cls.get_plugin_metrics(obj, cls.get_values(PluginResult._CSV_METRICS)))
            ret.extend(new)
        return ret
    class Meta:
        verbose_name_plural = "Results"

    def getdeviceid(self):
        '''Returns deviceid of the storage of the results directory'''
        return os.stat(self.get_report_dir())[2]


@receiver(post_save, sender=Results, dispatch_uid="create_result")
def on_result_created(sender, instance, created, **kwargs):
    if created:
        experiment = instance.experiment
        experiment.resultDate = instance.timeStamp
        if not (experiment.repResult and experiment.pinnedRepResult):
            experiment.repResult = instance
            experiment.pinnedRepResult = False
        experiment.save()


@receiver(post_delete, sender=Results, dispatch_uid="delete_result")
def on_result_delete(sender, instance, **kwargs):
    """Delete all of the files represented by a Experiment object which was
    deleted and all of files derived from that Experiment which are in it's
    folder.
    """
    # Note, this completely sucks: is there a better way of determining this?
    root = instance.reportstorage.dirPath
    prefix = len(instance.reportstorage.webServerPath)
    postfix = os.path.dirname(instance.reportLink[prefix+1:])
    directory = os.path.join(root, postfix)
    logger.info("Deleting Result %d in %s" % (instance.id, directory))
    tasks.delete_that_folder.delay(directory,
                       "Triggered by Results %d deletion" % instance.id)
    
    if Experiment.objects.filter(id = instance.experiment_id).exists():
        experiment = instance.experiment
        if experiment.repResult is None:
            experiment.pinnedRepResult = False
            results = experiment.results_set.order_by('-timeStamp')[:1]
            if results:
                experiment.repResult = results[0]
            experiment.save()


class TFMetrics(models.Model, Lookup):
    _CSV_METRICS = (
        ("TF Name", "name"),
        ("Q10 Mean", "Q10Mean"),
        ("Q17 Mean", "Q17Mean"),
        ("System SNR", "SysSNR"),
        ("50Q10 Reads", "Q10ReadCount"),
        ("50Q17 Reads", "Q17ReadCount"),
        ("Keypass Reads", "keypass"),
        ("TF Key Peak Counts", 'aveKeyCount'),
        )
    _ALIASES = {
        "tfname":"name",
        "q17mean":"Q17Mean",
        "systemsnr":"SysSNR",
        "50q17reads":"Q17ReadCount",
        "keypassreads": "keypass",
        "tfkeypeakcounts":'aveKeyCount'
        }
    TABLE_FIELDS = ("TF Name", "Q17 Mean",
                    "System SNR", "50Q17 Reads", "Keypass Reads",
                    "TF Key Peak Counts")
    report = models.ForeignKey(Results, db_index=True, related_name='tfmetrics_set')
    name = models.CharField(max_length=128, db_index=True)
    Q10Histo = models.TextField(blank=True)
    Q10Mean = models.FloatField()
    Q17Histo = models.TextField(blank=True)
    Q17Mean = models.FloatField()
    SysSNR = models.FloatField()
    corrHPSNR = models.TextField(blank=True)
    HPAccuracy = models.TextField(blank=True)
    Q10ReadCount = models.FloatField()
    Q17ReadCount = models.FloatField()
    sequence = models.CharField(max_length=512)#ambitious
    keypass = models.FloatField()
    ####CAFIE#####
    number = models.FloatField()
    aveKeyCount = models.FloatField()
    def __unicode__(self):
        return "%s/%s" % (self.report, self.name)

    def get_csv_metrics(self):
        ret = []
        for metric in self._CSV_METRICS:
            ret.append((metric[0], getattr(self, metric[1], ' ')))

    class Meta:
        verbose_name_plural = "TF metrics"

class Location(models.Model):
    name = models.CharField(max_length=200)
    comments = models.TextField(blank=True)
    defaultlocation = models.BooleanField("Set as the Default Location",default=False,help_text="Only one location can be the default")

    def __unicode__(self):
        return u'%s' % self.name

    def save(self, *args, **kwargs):
        """make sure only one location is checked."""
        super(Location, self).save(*args, **kwargs)
        if self.defaultlocation:
            # If self is marked as default, mark all others as not default
            others = Location.objects.all().exclude(id=self.id)
            for other in others:
                other.defaultlocation = False
                super(Location, other).save(*args, **kwargs)


class Rig(models.Model):
    name = models.CharField(max_length=200, primary_key=True)
    location = models.ForeignKey(Location)
    comments = models.TextField(blank=True)
    ftpserver	 = models.CharField(max_length=128, default="192.168.201.1")
    ftpusername	 = models.CharField(max_length=64, default="ionguest")
    ftppassword	 = models.CharField(max_length=64, default="ionguest")
    ftprootdir	 = models.CharField(max_length=64, default="results")
    updatehome	 = models.CharField(max_length=256, default="192.168.201.1")
    updateflag = models.BooleanField(default=False)
    serial = models.CharField(max_length=24, blank=True, null=True)

    state = models.CharField(max_length=512, blank=True)
    version = json_field.JSONField(blank=True)
    alarms = json_field.JSONField(blank=True)
    last_init_date = models.CharField(max_length=512, blank=True)
    last_clean_date = models.CharField(max_length=512, blank=True)
    last_experiment = models.CharField(max_length=512, blank=True)

    host_address = models.CharField(blank=True,max_length=1024)
    type = models.CharField(blank=True,max_length=1024)


    def __unicode__(self): return self.name

class FileServer(models.Model):
    name = models.CharField(max_length=200)
    comments = models.TextField(blank=True)
    #TODO require this field's contents to be terminated with trailing delimiter
    filesPrefix = models.CharField(max_length=200)
    location = models.ForeignKey(Location)
    percentfull = models.FloatField(default=0.0,blank=True,null=True)
    def __unicode__(self): return self.name

class ReportStorage(models.Model):
    name = models.CharField(max_length=200)
    #path to webserver as http://localhost/results/
    webServerPath = models.CharField(max_length=200)
    dirPath = models.CharField(max_length=200)
    default=models.BooleanField(default=False)
    def __unicode__(self):
        return "%s (%s)" % (self.name, self.dirPath)

    def save(self, *args, **kwargs):
        """make sure only one object is checked."""
        super(ReportStorage, self).save(*args, **kwargs)
        if self.default:
            # If self is marked as default, mark all others as not default
            others = ReportStorage.objects.all().exclude(pk=self.id)
            for other in others:
                other.default = False
                super(ReportStorage, other).save(*args, **kwargs)

class RunScript(models.Model):
    name = models.CharField(max_length=200)
    script = models.TextField(blank=True)
    def __unicode__(self):
        return self.name

class Cruncher(models.Model):
    name = models.CharField(max_length=200)
    prefix = models.CharField(max_length=512)
    location = models.ForeignKey(Location)
    comments = models.TextField(blank=True)
    def __unicode__(self): return self.name

class AnalysisMetrics(models.Model):
    _CSV_METRICS = (('Num_Washouts', 'washout'),
                    ('Num_Dud_Washouts', 'washout_dud'),
                    ('Num_Washout_Ambiguous', 'washout_ambiguous'),
                    ('Num_Washout_Live', 'washout_live'),
                    ('Num_Washout_Test_Fragment', 'washout_test_fragment'),
                    ('Num_Washout_Library', 'washout_library'),
                    ('Library_Pass_Basecalling', 'lib_pass_basecaller'),
                    ('Library_pass_Cafie', 'lib_pass_cafie'),
                    ('Number_Ambiguous', 'amb'),
                    ('Nubmer_Live', 'live'),
                    ('Number_Dud', 'dud'),
                    ('Number_TF', 'tf'),
                    ('Number_Lib', 'lib'),
                    ('Number_Bead', 'bead'),
                    ('Library_Live', 'libLive'),
                    ('Library_Keypass', 'libKp'),
                    ('TF_Live', 'live'),
                    ('TF_Keypass', 'tfKp'),
                    ('Keypass_All_Beads', 'keypass_all_beads'),
                    )
    report = models.ForeignKey(Results, related_name='analysismetrics_set')
    libLive = models.IntegerField()
    libKp = models.IntegerField()
    libMix = models.IntegerField()
    libFinal = models.IntegerField()
    tfLive = models.IntegerField()
    tfKp = models.IntegerField()
    tfMix = models.IntegerField()
    tfFinal = models.IntegerField()
    empty = models.IntegerField()
    bead = models.IntegerField()
    live = models.IntegerField()
    dud = models.IntegerField()
    amb = models.IntegerField()
    tf = models.IntegerField()
    lib = models.IntegerField()
    pinned = models.IntegerField()
    ignored = models.IntegerField()
    excluded = models.IntegerField()
    washout = models.IntegerField()
    washout_dud = models.IntegerField()
    washout_ambiguous = models.IntegerField()
    washout_live = models.IntegerField()
    washout_test_fragment = models.IntegerField()
    washout_library = models.IntegerField()
    lib_pass_basecaller = models.IntegerField()
    lib_pass_cafie = models.IntegerField()
    keypass_all_beads = models.IntegerField()
    sysCF = models.FloatField()
    sysIE = models.FloatField()
    sysDR = models.FloatField()
    def __unicode__(self):
        return "%s/%d" % (self.report, self.pk)
    class Meta:
        verbose_name_plural = "Analysis metrics"

class LibMetrics(models.Model):
    _CSV_METRICS = (('Total_Num_Reads', 'totalNumReads'),
                    ('Library_50Q10_Reads', 'i50Q10_reads'),
                    ('Library_100Q10_Reads', 'i100Q10_reads'),
                    ('Library_200Q10_Reads', 'i200Q10_reads'),
                    ('Library_Mean_Q10_Length', 'q10_mean_alignment_length'),
                    ('Library_Q10_Coverage', 'q10_coverage_percentage'),
                    ('Library_Q10_Longest_Alignment', 'q10_longest_alignment'),
                    ('Library_Q10_Mapped Bases', 'q10_mapped_bases'),
                    ('Library_Q10_Alignments', 'q10_alignments'),
                    ('Library_50Q17_Reads', 'i50Q17_reads'),
                    ('Library_100Q17_Reads', 'i100Q17_reads'),
                    ('Library_200Q17_Reads', 'i200Q17_reads'),
                    ('Library_Mean_Q17_Length', 'q17_mean_alignment_length'),
                    ('Library_Q17_Coverage', 'q17_coverage_percentage'),
                    ('Library_Q17_Longest_Alignment', 'q17_longest_alignment'),
                    ('Library_Q17_Mapped Bases', 'q17_mapped_bases'),
                    ('Library_Q17_Alignments', 'q17_alignments'),
                    ('Library_50Q20_Reads', 'i50Q20_reads'),
                    ('Library_100Q20_Reads', 'i100Q20_reads'),
                    ('Library_200Q20_Reads', 'i200Q20_reads'),
                    ('Library_Mean_Q20_Length', 'q20_mean_alignment_length'),
                    ('Library_Q20_Coverage', 'q20_coverage_percentage'),
                    ('Library_Q20_Longest_Alignment', 'q20_longest_alignment'),
                    ('Library_Q20_Mapped Bases', 'q20_mapped_bases'),
                    ('Library_Q20_Alignments', 'q20_alignments'),
                    ('Library_Key_Peak_Counts', 'aveKeyCounts'),
                    ('Library_50Q47_Reads', 'i50Q47_reads'),
                    ('Library_100Q47_Reads', 'i100Q47_reads'),
                    ('Library_200Q47_Reads', 'i200Q47_reads'),
                    ('Library_Mean_Q47_Length', 'q47_mean_alignment_length'),
                    ('Library_Q47_Coverage', 'q47_coverage_percentage'),
                    ('Library_Q47_Longest_Alignment', 'q47_longest_alignment'),
                    ('Library_Q47_Mapped Bases', 'q47_mapped_bases'),
                    ('Library_Q47_Alignments', 'q47_alignments'),
                    ('Library_CF', 'cf'),
                    ('Library_IE', 'ie'),
                    ('Library_DR', 'dr'),
                    ('Library_SNR', 'sysSNR'),
                    ('Raw Accuracy', 'raw_accuracy'),
                    )
    report = models.ForeignKey(Results, db_index=True, related_name='libmetrics_set')
    raw_accuracy = models.FloatField()
    sysSNR = models.FloatField()
    aveKeyCounts = models.FloatField()
    totalNumReads = models.IntegerField()
    total_mapped_reads = models.BigIntegerField()
    total_mapped_target_bases = models.BigIntegerField()
    genomelength = models.IntegerField()
    rNumAlignments = models.IntegerField()
    rMeanAlignLen = models.IntegerField()
    rLongestAlign = models.IntegerField()
    rCoverage = models.FloatField()
    r50Q10 = models.IntegerField()
    r100Q10 = models.IntegerField()
    r200Q10 = models.IntegerField()
    r50Q17 = models.IntegerField()
    r100Q17 = models.IntegerField()
    r200Q17 = models.IntegerField()
    r50Q20 = models.IntegerField()
    r100Q20 = models.IntegerField()
    r200Q20 = models.IntegerField()
    sNumAlignments = models.IntegerField()
    sMeanAlignLen = models.IntegerField()
    sLongestAlign = models.IntegerField()
    sCoverage = models.FloatField()
    s50Q10 = models.IntegerField()
    s100Q10 = models.IntegerField()
    s200Q10 = models.IntegerField()
    s50Q17 = models.IntegerField()
    s100Q17 = models.IntegerField()
    s200Q17 = models.IntegerField()
    s50Q20 = models.IntegerField()
    s100Q20 = models.IntegerField()
    s200Q20 = models.IntegerField()

    q7_coverage_percentage = models.FloatField()
    q7_alignments = models.IntegerField()
    q7_mapped_bases = models.BigIntegerField()
    q7_qscore_bases = models.BigIntegerField()
    q7_mean_alignment_length = models.IntegerField()
    q7_longest_alignment = models.IntegerField()
    i50Q7_reads = models.IntegerField()
    i100Q7_reads = models.IntegerField()
    i150Q7_reads = models.IntegerField()
    i200Q7_reads = models.IntegerField()
    i250Q7_reads = models.IntegerField()
    i300Q7_reads = models.IntegerField()
    i350Q7_reads = models.IntegerField()
    i400Q7_reads = models.IntegerField()
    i450Q7_reads = models.IntegerField()
    i500Q7_reads = models.IntegerField()
    i550Q7_reads = models.IntegerField()
    i600Q7_reads = models.IntegerField()

    q10_coverage_percentage = models.FloatField()
    q10_alignments = models.IntegerField()
    q10_mapped_bases = models.BigIntegerField()
    q10_qscore_bases = models.BigIntegerField()
    q10_mean_alignment_length = models.IntegerField()
    q10_longest_alignment = models.IntegerField()
    i50Q10_reads = models.IntegerField()
    i100Q10_reads = models.IntegerField()
    i150Q10_reads = models.IntegerField()
    i200Q10_reads = models.IntegerField()
    i250Q10_reads = models.IntegerField()
    i300Q10_reads = models.IntegerField()
    i350Q10_reads = models.IntegerField()
    i400Q10_reads = models.IntegerField()
    i450Q10_reads = models.IntegerField()
    i500Q10_reads = models.IntegerField()
    i550Q10_reads = models.IntegerField()
    i600Q10_reads = models.IntegerField()

    q17_coverage_percentage = models.FloatField()
    q17_alignments = models.IntegerField()
    q17_mapped_bases = models.BigIntegerField()
    q17_qscore_bases = models.BigIntegerField()
    q17_mean_alignment_length = models.IntegerField()
    q17_longest_alignment = models.IntegerField()
    i50Q17_reads = models.IntegerField()
    i100Q17_reads = models.IntegerField()
    i150Q17_reads = models.IntegerField()
    i200Q17_reads = models.IntegerField()
    i250Q17_reads = models.IntegerField()
    i300Q17_reads = models.IntegerField()
    i350Q17_reads = models.IntegerField()
    i400Q17_reads = models.IntegerField()
    i450Q17_reads = models.IntegerField()
    i500Q17_reads = models.IntegerField()
    i550Q17_reads = models.IntegerField()
    i600Q17_reads = models.IntegerField()

    q20_coverage_percentage = models.FloatField()
    q20_alignments = models.IntegerField()
    q20_mapped_bases = models.BigIntegerField()
    q20_qscore_bases = models.BigIntegerField()
    q20_mean_alignment_length = models.IntegerField()
    q20_longest_alignment = models.IntegerField()
    i50Q20_reads = models.IntegerField()
    i100Q20_reads = models.IntegerField()
    i150Q20_reads = models.IntegerField()
    i200Q20_reads = models.IntegerField()
    i250Q20_reads = models.IntegerField()
    i300Q20_reads = models.IntegerField()
    i350Q20_reads = models.IntegerField()
    i400Q20_reads = models.IntegerField()
    i450Q20_reads = models.IntegerField()
    i500Q20_reads = models.IntegerField()
    i550Q20_reads = models.IntegerField()
    i600Q20_reads = models.IntegerField()

    q47_coverage_percentage = models.FloatField()
    q47_mapped_bases = models.BigIntegerField()
    q47_qscore_bases = models.BigIntegerField()
    q47_alignments = models.IntegerField()
    q47_mean_alignment_length = models.IntegerField()
    q47_longest_alignment = models.IntegerField()
    i50Q47_reads = models.IntegerField()
    i100Q47_reads = models.IntegerField()
    i150Q47_reads = models.IntegerField()
    i200Q47_reads = models.IntegerField()
    i250Q47_reads = models.IntegerField()
    i300Q47_reads = models.IntegerField()
    i350Q47_reads = models.IntegerField()
    i400Q47_reads = models.IntegerField()
    i450Q47_reads = models.IntegerField()
    i500Q47_reads = models.IntegerField()
    i550Q47_reads = models.IntegerField()
    i600Q47_reads = models.IntegerField()

    cf = models.FloatField()
    ie = models.FloatField()
    dr = models.FloatField()
    Genome_Version = models.CharField(max_length=512)
    Index_Version = models.CharField(max_length=512)
    #lots of additional fields in the case that only a sampled+extrapolated alignment is done
    #first add a int to let me know if it is full of sampled align
    align_sample = models.IntegerField()
    genome = models.CharField(max_length=512)
    genomesize = models.BigIntegerField()
    total_number_of_sampled_reads = models.IntegerField()
    sampled_q7_coverage_percentage = models.FloatField()
    sampled_q7_mean_coverage_depth = models.FloatField()
    sampled_q7_alignments = models.IntegerField()
    sampled_q7_mean_alignment_length = models.IntegerField()
    sampled_mapped_bases_in_q7_alignments = models.BigIntegerField()
    sampled_q7_longest_alignment = models.IntegerField()
    sampled_50q7_reads = models.IntegerField()
    sampled_100q7_reads = models.IntegerField()
    sampled_200q7_reads = models.IntegerField()
    sampled_300q7_reads = models.IntegerField()
    sampled_400q7_reads = models.IntegerField()
    sampled_q10_coverage_percentage = models.FloatField()
    sampled_q10_mean_coverage_depth = models.FloatField()
    sampled_q10_alignments = models.IntegerField()
    sampled_q10_mean_alignment_length = models.IntegerField()
    sampled_mapped_bases_in_q10_alignments = models.BigIntegerField()
    sampled_q10_longest_alignment = models.IntegerField()
    sampled_50q10_reads = models.IntegerField()
    sampled_100q10_reads = models.IntegerField()
    sampled_200q10_reads = models.IntegerField()
    sampled_300q10_reads = models.IntegerField()
    sampled_400q10_reads = models.IntegerField()
    sampled_q17_coverage_percentage = models.FloatField()
    sampled_q17_mean_coverage_depth = models.FloatField()
    sampled_q17_alignments = models.IntegerField()
    sampled_q17_mean_alignment_length = models.IntegerField()
    sampled_mapped_bases_in_q17_alignments = models.BigIntegerField()
    sampled_q17_longest_alignment = models.IntegerField()
    sampled_50q17_reads = models.IntegerField()
    sampled_100q17_reads = models.IntegerField()
    sampled_200q17_reads = models.IntegerField()
    sampled_300q17_reads = models.IntegerField()
    sampled_400q17_reads = models.IntegerField()
    sampled_q20_coverage_percentage = models.FloatField()
    sampled_q20_mean_coverage_depth = models.FloatField()
    sampled_q20_alignments = models.IntegerField()
    sampled_q20_mean_alignment_length = models.IntegerField()
    sampled_mapped_bases_in_q20_alignments = models.BigIntegerField()
    sampled_q20_longest_alignment = models.IntegerField()
    sampled_50q20_reads = models.IntegerField()
    sampled_100q20_reads = models.IntegerField()
    sampled_200q20_reads = models.IntegerField()
    sampled_300q20_reads = models.IntegerField()
    sampled_400q20_reads = models.IntegerField()
    sampled_q47_coverage_percentage = models.FloatField()
    sampled_q47_mean_coverage_depth = models.FloatField()
    sampled_q47_alignments = models.IntegerField()
    sampled_q47_mean_alignment_length = models.IntegerField()
    sampled_mapped_bases_in_q47_alignments = models.BigIntegerField()
    sampled_q47_longest_alignment = models.IntegerField()
    sampled_50q47_reads = models.IntegerField()
    sampled_100q47_reads = models.IntegerField()
    sampled_200q47_reads = models.IntegerField()
    sampled_300q47_reads = models.IntegerField()
    sampled_400q47_reads = models.IntegerField()
    extrapolated_from_number_of_sampled_reads = models.IntegerField()
    extrapolated_q7_coverage_percentage = models.FloatField()
    extrapolated_q7_mean_coverage_depth = models.FloatField()
    extrapolated_q7_alignments = models.IntegerField()
    extrapolated_q7_mean_alignment_length = models.IntegerField()
    extrapolated_mapped_bases_in_q7_alignments = models.BigIntegerField()
    extrapolated_q7_longest_alignment = models.IntegerField()
    extrapolated_50q7_reads = models.IntegerField()
    extrapolated_100q7_reads = models.IntegerField()
    extrapolated_200q7_reads = models.IntegerField()
    extrapolated_300q7_reads = models.IntegerField()
    extrapolated_400q7_reads = models.IntegerField()
    extrapolated_q10_coverage_percentage = models.FloatField()
    extrapolated_q10_mean_coverage_depth = models.FloatField()
    extrapolated_q10_alignments = models.IntegerField()
    extrapolated_q10_mean_alignment_length = models.IntegerField()
    extrapolated_mapped_bases_in_q10_alignments = models.BigIntegerField()
    extrapolated_q10_longest_alignment = models.IntegerField()
    extrapolated_50q10_reads = models.IntegerField()
    extrapolated_100q10_reads = models.IntegerField()
    extrapolated_200q10_reads = models.IntegerField()
    extrapolated_300q10_reads = models.IntegerField()
    extrapolated_400q10_reads = models.IntegerField()
    extrapolated_q17_coverage_percentage = models.FloatField()
    extrapolated_q17_mean_coverage_depth = models.FloatField()
    extrapolated_q17_alignments = models.IntegerField()
    extrapolated_q17_mean_alignment_length = models.IntegerField()
    extrapolated_mapped_bases_in_q17_alignments = models.BigIntegerField()
    extrapolated_q17_longest_alignment = models.IntegerField()
    extrapolated_50q17_reads = models.IntegerField()
    extrapolated_100q17_reads = models.IntegerField()
    extrapolated_200q17_reads = models.IntegerField()
    extrapolated_300q17_reads = models.IntegerField()
    extrapolated_400q17_reads = models.IntegerField()
    extrapolated_q20_coverage_percentage = models.FloatField()
    extrapolated_q20_mean_coverage_depth = models.FloatField()
    extrapolated_q20_alignments = models.IntegerField()
    extrapolated_q20_mean_alignment_length = models.IntegerField()
    extrapolated_mapped_bases_in_q20_alignments = models.BigIntegerField()
    extrapolated_q20_longest_alignment = models.IntegerField()
    extrapolated_50q20_reads = models.IntegerField()
    extrapolated_100q20_reads = models.IntegerField()
    extrapolated_200q20_reads = models.IntegerField()
    extrapolated_300q20_reads = models.IntegerField()
    extrapolated_400q20_reads = models.IntegerField()
    extrapolated_q47_coverage_percentage = models.FloatField()
    extrapolated_q47_mean_coverage_depth = models.FloatField()
    extrapolated_q47_alignments = models.IntegerField()
    extrapolated_q47_mean_alignment_length = models.IntegerField()
    extrapolated_mapped_bases_in_q47_alignments = models.BigIntegerField()
    extrapolated_q47_longest_alignment = models.IntegerField()
    extrapolated_50q47_reads = models.IntegerField()
    extrapolated_100q47_reads = models.IntegerField()
    extrapolated_200q47_reads = models.IntegerField()
    extrapolated_300q47_reads = models.IntegerField()
    extrapolated_400q47_reads = models.IntegerField()
    duplicate_reads = models.IntegerField(null=True,blank=True)

    def __unicode__(self):
        return "%s/%d" % (self.report, self.pk)
    class Meta:
        verbose_name_plural = "Lib Metrics"

class QualityMetrics(models.Model):
    """a place in the database to store the quality metrics from SFFSumary"""
    #make csv metrics lookup here
    report = models.ForeignKey(Results, db_index=True, related_name='qualitymetrics_set')
    q0_bases = models.BigIntegerField()
    q0_reads = models.IntegerField()
    q0_max_read_length = models.IntegerField()
    q0_mean_read_length = models.FloatField()
    q0_50bp_reads = models.IntegerField()
    q0_100bp_reads = models.IntegerField()
    q0_150bp_reads = models.IntegerField(default=0)
    q17_bases = models.BigIntegerField()
    q17_reads = models.IntegerField()
    q17_max_read_length = models.IntegerField()
    q17_mean_read_length = models.FloatField()
    q17_50bp_reads = models.IntegerField()
    q17_100bp_reads = models.IntegerField()
    q17_150bp_reads = models.IntegerField()
    q20_bases = models.BigIntegerField()
    q20_reads = models.IntegerField()
    q20_max_read_length = models.FloatField()
    q20_mean_read_length = models.IntegerField()
    q20_50bp_reads = models.IntegerField()
    q20_100bp_reads = models.IntegerField()
    q20_150bp_reads = models.IntegerField()

    def __unicode__(self):
        return "%s/%d" % (self.report, self.pk)
    class Meta:
        verbose_name_plural = "Quality Metrics"

class Template(models.Model):
    name = models.CharField(max_length=64)
    sequence = models.TextField(blank=True)
    key = models.CharField(max_length=64)
    comments = models.TextField(blank=True)
    isofficial = models.BooleanField(default=True)
    def __unicode__(self):
        return self.name

class Backup(models.Model):
    experiment = models.ForeignKey(Experiment)
    backupName = models.CharField(max_length=256, unique=True)
    # boolean indicator whether raw data has been archived
    isBackedUp = models.BooleanField()
    backupDate = models.DateTimeField()
    # backupPath will be a filesystem path if the data directory was archived.
    # else it will be 'DELETED' string.
    # else it will be 'PARTIAL-DELETE' string if some, not all deletes have occurred
    backupPath = models.CharField(max_length=512)
    def __unicode__(self):
        return u'%s' % self.experiment

class BackupConfig(models.Model):
    name = models.CharField(max_length=64)
    location = models.ForeignKey(Location)
    backup_directory = models.CharField(max_length=256, blank=True, default=None)
    backup_threshold = models.IntegerField(blank=True)
    number_to_backup = models.IntegerField(blank=True)
    grace_period = models.IntegerField(default=72)
    timeout = models.IntegerField(blank=True)
    bandwidth_limit = models.IntegerField(blank=True)
    status = models.CharField(max_length=512, blank=True)
    online = models.BooleanField()
    comments = models.TextField(blank=True)
    email = models.EmailField(blank=True)
    keepTN = models.BooleanField(default=True)
    def __unicode__(self):
        return self.name

    def get_free_space(self):
        dev = devices.disk_report()
        for d in dev:
            if self.backup_directory == d.get_path():
                return d.get_free_space()

    def check_if_online(self):
        if os.path.exists(self.backup_directory):
            return True
        else:
            return False

    @classmethod
    def get(cls):
        """This represents pretty much the only query on this entire
        table, find the 'canonical' BackupConfig record.  The primary
        key order is used in all cases as the tie breaker.
        Since there is *always* supposed to be one of these in the DB,
        this call to get will properly raises a DoesNotExist error.
        """
        return cls.objects.order_by('pk')[:1].get()

class dm_reports(models.Model):
    '''This object holds the options for report actions.
    Which level(s) to prune at, what those levels are, how far back to look when recording space saved, etc.'''
    location = models.CharField(max_length=512)
    pruneLevel = models.CharField(max_length=128, default = 'No-op')    #FRAGILE: requires dm_prune_group.name == "No-op"
    autoPrune = models.BooleanField(default = False)
    autoType = models.CharField(max_length=32, default = 'P')
    autoAge = models.IntegerField(default=90)
    class Meta:
        verbose_name_plural = "DM - Configuration"

    def __unicode__(self):
        return self.location

    @classmethod
    def get(cls):
        """This represents pretty much the only query on this entire
        table, find the 'canonical' dm_reports record.  The primary
        key order is used in all cases as the tie breaker.
        Since there is *always* supposed to be one of these in the DB,
        this call to get will properly raises a DoesNotExist error.
        """
        return cls.objects.order_by('pk')[:1].get()


class dm_prune_group(models.Model):
    name = models.CharField(max_length=128, default="")
    editable = models.BooleanField(default=True)    # This actually signifies "deletable by customer"
    ruleNums = models.CommaSeparatedIntegerField(max_length=128, default='', blank = True)
    class Meta:
        verbose_name_plural = "DM - Prune Groups"

    def __unicode__(self):
        return self.name

class dm_prune_field(models.Model):
    rule = models.CharField(max_length=64, default = "")
    class Meta:
        verbose_name_plural = "DM - Prune Rules"

class Chip(models.Model):
    name = models.CharField(max_length=128)
    slots = models.IntegerField()
    description = models.CharField(max_length=128, default="")

    isActive = models.BooleanField(default=True)

    ALLOWED_INSTRUMENT_TYPES = (
        ('', "Undefined"),
        ('pgm', 'PGM'),
        ('proton', 'Proton')
    )
    #compatible instrument type
    instrumentType = models.CharField(max_length=64, choices=ALLOWED_INSTRUMENT_TYPES, default='', blank=True)


    def getChipDisplayedName(self):
        if self.description:
            return self.description;
        else:
            return self.name;


class GlobalConfig(models.Model):
    name = models.CharField(max_length=512)
    selected = models.BooleanField()
    plugin_folder = models.CharField(max_length=512, blank=True)

    fasta_path = models.CharField(max_length=512, blank=True)
    reference_path = models.CharField(max_length=1000, blank=True)
    records_to_display = models.IntegerField(default=20, blank=True)
    default_test_fragment_key = models.CharField(max_length=50, blank=True)
    default_library_key = models.CharField(max_length=50, blank=True)
    default_flow_order = models.CharField(max_length=100, blank=True)
    plugin_output_folder = models.CharField(max_length=500, blank=True)
    default_plugin_script = models.CharField(max_length=500, blank=True)
    web_root = models.CharField(max_length=500, blank=True)
    site_name = models.CharField(max_length=500, blank=True)
    default_storage_options = models.CharField(max_length=500,
                                       choices=Experiment.STORAGE_CHOICES,
                                       default='D', blank=True)
    auto_archive_ack = models.BooleanField("Auto-Acknowledge Delete?", default=False)
    auto_archive_enable = models.BooleanField("Enable Auto Actions?", default=False)

    barcode_args = json_field.JSONField(blank=True)
    enable_auto_pkg_dl = models.BooleanField("Enable Package Auto Download", default=True)
    #enable_alt_apt = models.BooleanField("Enable USB apt repository", default=False)
    enable_version_lock = models.BooleanField("Enable TS Version Lock", default=False)
    ts_update_status = models.CharField(max_length=256,blank=True)
    # Controls analysis pipeline alternate processing path
    base_recalibrate = models.BooleanField(default=True)
    mark_duplicates = models.BooleanField(default=False)
    realign = models.BooleanField(default=False)
    check_news_posts = models.BooleanField("check for news posts", default=True)


    def set_TS_update_status(self,inputstr):
        self.ts_update_status = inputstr

    def set_enableAutoPkgDL(self,flag):
        if type(flag) is bool:
            self.enableAutoPkgDL = flag

    def get_enableAutoPkgDL(self):
        return self.enableAutoPkgDL

    @classmethod
    def get(cls):
        """This represents pretty much the only query on this entire
        table, find the 'canonical' GlobalConfig record.  The primary
        key order is used in all cases as the tie breaker.
        Since there is *always* supposed to be one of these in the DB,
        this call to get will properly raises a DoesNotExist error.
        """
        return cls.objects.order_by('pk')[:1].get()


@receiver(post_save, sender=GlobalConfig, dispatch_uid="save_globalconfig")
def on_save_config_sitename(sender, instance, created, **kwargs):
    """Very sneaky, we open the Default Report base template which the PHP
    file for the report renders itself inside of and find the name, replace it,
    and rewrite the thing.
    """
    try:
        with open("/opt/ion/iondb/templates/rundb/php_base.html", 'r+') as name:
            text = name.read().decode('utf8')
            name.seek(0)
            # .*? is a non-greedy qualifier.
            # It will match the minimally satisfying text
            target = '<h1 id="sitename">.*?</h1>'
            replacement = '<h1 id="sitename">%s</h1>' % instance.site_name
            text = re.sub(target, replacement, text)
            target = '<title>.*?</title>'
            replacement = '<title>%s - Torrent Browser</title>' % instance.site_name
            name.write(re.sub(target, replacement, text).encode('utf8'))
    except IOError as err:
        logger.warning("Problem with /opt/ion/iondb/templates/rundb/php_base.html: %s" % err)


class EmailAddress(models.Model):
    email = models.EmailField(blank=True)
    selected = models.BooleanField()
    class Meta:
        verbose_name_plural = "Email addresses"


class Plugin(models.Model):
    name = models.CharField(max_length=512, db_index=True)
    version = models.CharField(max_length=256)
    description = models.TextField(blank=True, default="")

    date = models.DateTimeField(auto_now_add=True)
    selected = models.BooleanField(default=False)
    path = models.CharField(max_length=512, blank=True, default="")
    autorun = models.BooleanField(default=False)
    majorBlock = models.BooleanField(default=False)
    config = json_field.JSONField(blank=True, null=True, default="")
    #status -- install messages, install status
    status = json_field.JSONField(blank=True, null=True, default="")

    # Store and mask inactive (uninstalled) plugins
    active = models.BooleanField(default=True)

    # Plugin Feed URL - 0install xml
    url = models.URLField(max_length=256, blank=True, default="")

    # pluginsettings.json or python class features, runtype, runlevel attributes
    pluginsettings = json_field.JSONField(blank=True, null=True, default="")

    # Cached getUserConfig  / getUserSettings function
    # plan time alternative to instance.html
    userinputfields = json_field.JSONField(blank=True, null=True)

    # These were functions, but make more sense to cache in db
    autorunMutable = models.BooleanField(default=True)

    ## file containing plugin definition. launch.sh or PluginName.py
    script = models.CharField(max_length=256, blank=True, default="")

    def isConfig(self):
        try:
            if os.path.exists(os.path.join(self.path, "config.html")):
                #provide a link to load the plugins html
                return urlresolvers.reverse('configure_plugins_plugin_configure', kwargs = {'pk':self.pk, 'action':'config'})
        except OSError:
            pass
        return False

    def isPlanConfig(self):
        try:
            if os.path.exists(os.path.join(self.path, "plan.html")):
                #provide a link to load the plugins html
                return urlresolvers.reverse('configure_plugins_plugin_configure', kwargs = {'pk':self.pk, 'action':'plan'})
        except OSError:
            pass
        return False

    def hasAbout(self):
        try:
            if os.path.exists(os.path.join(self.path, "about.html")):
                #provide a link to load the plugins html
                return urlresolvers.reverse('configure_plugins_plugin_configure', kwargs = {'pk':self.pk, 'action':'about'})
        except OSError:
            pass
        return False

    def isInstance(self):
        try:
            if os.path.exists(os.path.join(self.path, "instance.html")):
                return urlresolvers.reverse('configure_plugins_plugin_configure', kwargs = {'pk':self.pk, 'action':'report'})
        except OSError:
            pass
        return False

    def __unicode__(self):
        return self.name

    def versionedName(self):
        return "%s--v%s" % (self.name, self.version)

    # Help for comparing plugins by version number
    def versionGreater(self, other):
        return(LooseVersion(self.version) > LooseVersion(other.version))

    def installStatus(self):
        """this method helps us know if a plugin was installed sucessfully"""
        if self.status.get("result"):
            if self.status["result"] == "queued":
                return "queued"
        return self.status.get("installStatus", "installed" )

    def pluginscript(self):
        # Now cached, join path and script
        if not self.script:
            from iondb.plugins.manager import pluginmanager
            self.script, _ = pluginmanager.find_pluginscript(self.path, self.name)
        if not self.script:
            self.script = '' # Avoid Null value in db column. find_pluginscript can return None.
            return None
        return os.path.join(self.path, self.script)

    def info(self, use_cache=True):
        """ This requires external process call when use_cache=False.
            Can be expensive. Avoid in API calls, only fetch when necessary. """
        if use_cache:
            return self.info_from_model()

        context = { 'plugin': self }
        from iondb.plugins.manager import pluginmanager
        info = pluginmanager.get_plugininfo(self.name, self.pluginscript(), context, use_cache)
        # Cache is updated in background task.
        if info is not None:
            self.updateFromInfo(info)  # update persistent db cache
        else:
            logger.error("Failed to get info from plugin: %s", self.name)

        return self.info_from_model()

    def updateFromInfo(self, info):
        version = info.get('version', None)
        if version and version != self.version:
            logger.warn("Queried plugin but got version mismatch. Plugin: %s has been updated from %s to %s", self.name, self.version, version)
            raise ValueError

        changed = False
        # User Input Fields from info config
        userinputfields = info.get('config', None)
        if userinputfields is not None and self.userinputfields != userinputfields:
            self.userinputfields = userinputfields
            changed = True

        # Set features, runtype, runlevel from info
        pluginsettings = {
            'features': info.get('features'),
            'runtype': info.get('runtypes'),
            'runlevel': info.get('runlevels'),
            'depends': info.get('depends'),
        }
        if self.pluginsettings != pluginsettings:
            self.pluginsettings = pluginsettings
            changed = True

        majorBlock = info.get('major_block', False)
        if self.majorBlock != majorBlock:
            self.majorBlock = majorBlock
            changed = True

        allow_autorun = info.get('allow_autorun', True)
        if self.autorunMutable != allow_autorun:
            self.autorunMutable = allow_autorun
            if not self.autorunMutable:
                self.autorun = False
            changed = True

        docs = info.get('docs', None)
        if docs and self.description != docs:
            self.description = docs
            changed = True

        if changed:
            self.save()

        return changed

    def info_from_model(self):
        info = {
            'name': self.name,
            'version': self.version,
            'runtypes': self.pluginsettings.get('runtype', []),
            'runlevels': self.pluginsettings.get('runlevel', []),
            'features': self.pluginsettings.get('features', []),
            'depends': self.pluginsettings.get('depends', []),
            'config': self.userinputfields,
            'allow_autorun': self.autorunMutable,
            'docs': self.description,
            'major_block' : self.majorBlock,
            'pluginorm': self.pk,
            'active': self.active,
        }
        from ion.plugin.info import PluginInfo
        return PluginInfo(info).todict()

    def natural_key(self):
        return (self.name, self.version)

    class Meta:
        unique_together=( ('name','version'), )

@receiver(pre_delete, sender=Plugin, dispatch_uid="delete_plugin")
def on_plugin_delete(sender, instance, **kwargs):
    """ Uninstall plugin on db record removal.
    Note: DB Delete is a purge and destroys all plugin content.
    Best to uninstall and leave records / results """
    if instance.path or instance.active:
        from iondb.plugins.manager import pluginmanager
        pluginmanager.uninstall(instance)

class PluginResult(models.Model):
    _CSV_METRICS = (
                    ("Plugin Data", 'store')
    )
    """ Many to Many mapping at the intersection of Results and Plugins """
    plugin = models.ForeignKey(Plugin)
    result = models.ForeignKey(Results, related_name='pluginresult_set')

    ALLOWED_STATES = (
        ('Completed', 'Completed'),
        ('Error', 'Error'),
        ('Started', 'Started'),
        ('Declined', 'Declined'),
        ('Unknown', 'Unknown'),
        ('Queued', 'Queued'), # In SGE queue
        ('Pending', 'Pending'), # Prior to submitting to SGE
        ('Resource', 'Exceeded Resource Limits'), ## SGE Errors
    )
    state = models.CharField(max_length=20, choices=ALLOWED_STATES)
    store = json_field.JSONField(blank=True)

    # Plugin instance config
    config = json_field.JSONField(blank=True, default='')

    jobid = models.IntegerField(null=True, blank=True)
    apikey = models.CharField(max_length=256, blank=True, null=True)

    # Track duration, disk use
    starttime = models.DateTimeField(null=True, blank=True)
    endtime = models.DateTimeField(null=True, blank=True)

    size = models.BigIntegerField(default=-1)
    inodes = models.BigIntegerField(default=-1)

    owner = models.ForeignKey(User)

    def path(self):
        return os.path.join(self.result.get_report_dir(),
                            'plugin_out',
                            self.plugin.name + '_out'
                           )
    # compat - return just size, not (size,inodes) tuple
    def calc_size(self):
        return self._calc_size()[0]

    def _calc_size(self):
        d = self.path()
        if not d or not os.path.exists(d):
            return (0,0)

        total_size = 0
        inodes = 0

        file_walker = (
            os.path.join(root, f)
            for root, _, files in os.walk( d )
            for f in files
        )
        for f in file_walker:
            if not os.path.isfile(f):
                continue
            inodes += 1
            try:
                total_size += os.lstat(f).st_size
            except OSError:
                logger.exception("Failed accessing %s during calc_size", f)

        logger.info("PluginResult %d for %s has %d byte(s) in %d file(s)",
                    self.id, self.plugin.name, total_size, inodes)
        return (total_size, inodes)

    # Taken from tastypie apikey generation
    def _generate_key(self):
        # Get a random UUID.
        new_uuid = uuid.uuid4()
        # Hmac that beast.
        return hmac.new(str(new_uuid), digestmod=sha1).hexdigest()

    #  Helpers for maintaining extra fields during status transitions
    def prepare(self, config='', jobid=None):
        self.state = 'Pending'
        # Always overwrite key - possibly invalidating existing key from running instance
        self.apikey = self._generate_key()
        if config:
            self.config = config
        else:
            # Not cleared if empty - reuses config from last invocation.
            pass
        self.starttime = None
        self.endtime = None
        self.jobid = jobid

    def start(self, jobid=None):
        self.state = 'Started'
        self.starttime = timezone.now()
        if self.jobid:
            if self.jobid != jobid:
                logger.warn("(Re-)started as different queue jobid: '%d' was '%d'", jobid, self.jobid)
            self.jobid = jobid

    def complete(self, state='Completed'):
        """ Call with state = Completed, Error, or other valid state """
        self.endtime = timezone.now()
        self.state = state
        self.apikey = None
        try:
            self.size, self.inodes = self._calc_size()
        except OSError:
            logger.exception("Failed to compute plugin size: '%s'", self.path())
            self.size = self.inodes = -1

    def duration(self):
        if not self.starttime:
            return 0
        if not self.endtime:
            return (timezone.now() - self.starttime)
        if (self.endtime > self.starttime):
            return (self.endtime - self.starttime)
        return 0

    def __unicode__(self):
        return "%s/%s" % (self.result, self.plugin)

    class Meta:
        unique_together=( ('plugin', 'result'), )
        ordering = [ '-id' ]

# NB: Fails if not pre-delete, as path() queries linked plugin and result.
@receiver(pre_delete, sender=PluginResult, dispatch_uid="delete_pluginresult")
def on_pluginresult_delete(sender, instance, **kwargs):
    """Delete all of the files for a pluginresult record """
    try:
        directory = instance.path()
    except:
        #if we can't find the result to delete
        return

    if directory and os.path.exists(directory):
        logger.info("Deleting Plugin Result %d in %s" % (instance.id, directory))
        tasks.delete_that_folder.delay(directory,
                       "Triggered by Plugin Results %d deletion" % instance.id)

class dnaBarcode(models.Model):
    """Store a dna barcode"""
    name = models.CharField(max_length=128)     # name of barcode SET
    id_str = models.CharField(max_length=128)   # id of this barcode sequence
    type = models.CharField(max_length=64, blank=True)
    sequence = models.CharField(max_length=128)
    length = models.IntegerField(default=0, blank=True)
    floworder = models.CharField(max_length=128, blank=True, default="")
    index = models.IntegerField()
    annotation = models.CharField(max_length=512,blank=True,default="")
    adapter = models.CharField(max_length=128,blank=True,default="")
    score_mode = models.IntegerField(default=0, blank=True)
    score_cutoff = models.FloatField(default=0)

    def __unicode__(self):
        return self.name

    class Meta:
        verbose_name_plural = "DNA Barcodes"

class ReferenceGenome(models.Model):
    """store info about the reference genome
    This should really hold the real path, it should also have methods for deleting the dirs for the files"""
    #long name
    name = models.CharField(max_length=512)
    #short name , we can change these
    short_name = models.CharField(max_length=512, unique=False)

    enabled = models.BooleanField(default=True)
    reference_path = models.CharField(max_length=1024, blank=True)
    date = models.DateTimeField(default=timezone.now)
    version = models.CharField(max_length=100, blank=True)
    species = models.CharField(max_length=512, blank=True)
    source = models.CharField(max_length=512, blank=True)
    notes = models.TextField(blank=True)
    #needs a status for index generation process
    status = models.CharField(max_length=512, blank=True)
    index_version = models.CharField(max_length=512, blank=True)
    verbose_error = models.CharField(max_length=3000, blank=True)
    identity_hash = models.CharField(max_length=40, null=True, blank=True, default=None)
    file_monitor = models.OneToOneField('FileMonitor', null=True, blank=True, on_delete=models.SET_NULL)
    celery_task_id = models.CharField(max_length=60, default="", blank=True)

    class Meta:
        ordering = ['short_name']

    def delete(self):
        #delete the genome from the filesystem as well as the database
        from celery.result import AsyncResult
        if self.celery_task_id:
            revoke(self.celery_task_id, terminate=True)
            children = AsyncResult(self.celery_task_id).children
            if children:
                for result in children:
                    result.revoke(terminate=True)
        logger.warning("Revoking celery task: {0}".format(self.celery_task_id))
        if os.path.exists(self.reference_path):
            try:
                shutil.rmtree(self.reference_path)
            except OSError:
                logger.error("Failed to delete reference %d at %s" % (self.pk, self.reference_path))
                return False
        else:
            logger.error("Files do not exists for reference %d at %s"  % (self.pk, self.reference_path))

        super(ReferenceGenome, self).delete()
        return True

    def enable_genome(self):
        """this should be around to move the genome in a disabled dir or not"""
        #get the new path to move the reference to
        enabled_path = os.path.join(iondb.settings.TMAP_DIR, self.short_name)
        try:
            shutil.move(self.reference_path, enabled_path)
            self.reference_path = enabled_path
            self.save()
        except:
            logger.exception("Failed to enable gnome %s" % self.short_name)
            return False
        return True

    def disable_genome(self):
        """this should be around to move the genome in a disabled dir or not"""
        #get the new path to move the reference to
        disabled_path = os.path.join(iondb.settings.TMAP_DIR, "disabled", self.index_version, self.short_name)
        try:
            shutil.move(self.reference_path, disabled_path)
            self.reference_path = disabled_path
            self.save()
        except:
            logger.exception("Failed to disable gnome %s" % reference.short_name)
            return False
        return True

    def info_text(self):
        return os.path.join(self.reference_path , self.short_name + ".info.txt")

    def genome_length(self):
        genome_ini = open(self.info_text()).readlines()
        try:
            length = float([g.split("\t")[1].strip() for g in genome_ini if g.startswith("genome_length")][0])
            return length
        except:
            return -1

    def fastaOrig(self):
        """
        if there was a file named .orig then the fasta was autofixed.
        """
        orig = os.path.join(self.reference_path , self.short_name + ".orig")
        return os.path.exists(orig)

    def __unicode__(self):
        return u'%s' % self.name


class ThreePrimeAdapterManager(models.Manager):
    def get_by_natural_key(self, uid):
        return self.get(uid = uid)

class ThreePrimeadapter(models.Model):
    ALLOWED_DIRECTIONS = (
        ('Forward', 'Forward'),
        ('Reverse', 'Reverse')
    )

    direction = models.CharField(max_length=20, choices=ALLOWED_DIRECTIONS, default='Forward')

    ALLOWED_RUN_MODES = (
        ('', 'Undefined'),
        ('single', 'SingleRead'),
        ('pe', 'PairedEnd')
    )

    #run mode
    runMode = models.CharField(max_length=64, choices=ALLOWED_RUN_MODES, default='single', blank=True)

    name = models.CharField(max_length=256, blank=False, unique=True)
    sequence = models.CharField(max_length=512, blank=False)
    description = models.CharField(max_length=1024, blank=True)

    isDefault = models.BooleanField("use this by default", default=False)

    uid = models.CharField(max_length = 32, unique = True, blank = False)

    ALLOWED_CHEMISTRY = (
        ('', 'Undefined'),
        ('avalanche', 'Avalanche')
    )

    chemistryType = models.CharField(max_length=64, choices= ALLOWED_CHEMISTRY, default='', blank=True)

    objects = ThreePrimeAdapterManager()

    class Meta:
        verbose_name_plural = "3' Adapters"

    def __unicode__(self):
        return u'%s' % self.name

    def natural_key(self):
        return (self.uid,)  # must return a tuple


    def save(self):
        if (self.isDefault == False and ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType = self.chemistryType).count() == 1):
            currentDefaults = ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType = self.chemistryType)
            #there should only be 1 default for a given direction and chemistry type at any time
            for currentDefault in currentDefaults:
                if (self.id == currentDefault.id):
                    raise ValidationError("Error: Please set another adapter for %s direction and %s chemistry to be the default before changing this adapter not to be the default." % (self.direction, self.chemistryType))

        if (self.isDefault == True and ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType = self.chemistryType).count() > 0):
            currentDefaults = ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType = self.chemistryType)
            #there should only be 1 default for a given direction and chemistry type at any time
            for currentDefault in currentDefaults:
                if (self.id <> currentDefault.id):
                    currentDefault.isDefault = False
                    super(ThreePrimeadapter, currentDefault).save()

        super(ThreePrimeadapter, self).save()



    def delete(self):
        if (self.isDefault == False and ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType = self.chemistryType).count() == 1):
            currentDefaults = ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType = self.chemistryType)
            #there should only be 1 default for a given direction and chemistry type at any time
            for currentDefault in currentDefaults:
                if (self.id == currentDefault.id):
                    raise ValidationError("Error: Deleting the default adapter is not allowed. Please set another adapter for %s direction and %s chemistry to be the default before deleting this adapter."  % (self.direction, self.chemistryType))

        if (self.isDefault == True and ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType = self.chemistryType).count() > 0):
            currentDefaults = ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType = self.chemistryType)
            #there should only be 1 default for a given direction and chemistry type at any time
            for currentDefault in currentDefaults:
                if (self.id == currentDefault.id):
                    raise ValidationError("Error: Deleting the default adapter is not allowed. Please set another adapter for %s direction and %s chemistry to be the default before deleting this adapter."  % (self.direction, self.chemistryType))

        super(ThreePrimeadapter, self).delete()


class Publisher(models.Model):
    name = models.CharField(max_length=200, unique=True)
    version = models.CharField(max_length=256)
    date = models.DateTimeField()
    path = models.CharField(max_length=512)
    global_meta = json_field.JSONField(blank=True)

    def __unicode__(self): return self.name

    def get_editing_scripts(self):
        pub_files = os.listdir(self.path)
        stages = ( ("pre_process", "Pre-processing"),
                   ("validate", "Validating"),
                   ("post_process", "Post-processing"),
                   ("register", "Registering"),
        )
        pub_scripts = []
        for stage, name in stages:
            for pub_file in pub_files:
                if pub_file.startswith(stage):
                    script_path = os.path.join(self.path, pub_file)
                    pub_scripts.append((script_path, name))
                    break
        return pub_scripts


class ContentUpload(models.Model):
    file_path = models.CharField(max_length=255)
    status = models.CharField(max_length=255, blank=True)
    meta = json_field.JSONField(blank=True)
    publisher = models.ForeignKey(Publisher, null=True)

    def __unicode__(self): return u'ContentUpload %d' % self.id


@receiver(post_delete, sender=ContentUpload, dispatch_uid="delete_upload")
def on_contentupload_delete(sender, instance, **kwargs):
    """Delete all of the files represented by a ContentUpload object which was
    deleted and all of files derived from that ContentUpload which are in it's
    folder.
    Note, it is traditional for publishers to store their Content in the folder
    of the ContentUpload from which the Content is derived.
    """
    def delete_error(func, path, info):
        logger.error("Deleting ContentUpload %d: failed to delete %s" %
                     (instance.id, path))
    directory = os.path.dirname(instance.file_path)
    logger.info("Deleting ContentUpload %d in %s" % (instance.id, directory))
    shutil.rmtree(directory, onerror=delete_error)


class Content(models.Model):
    publisher = models.ForeignKey(Publisher, related_name="contents")
    contentupload = models.ForeignKey(ContentUpload, related_name="contents")
    file = models.CharField(max_length=255)
    path = models.CharField(max_length=255)
    meta = json_field.JSONField(blank=True)

    def __unicode__(self): return self.path


@receiver(pre_delete, sender=Content, dispatch_uid="delete_content")
def on_content_delete(sender, instance, **kwargs):
    """Delete the file represented by a Content object which was deleted."""
    # I had cosidered attempting to intelligently remove empty parent
    # directories created by the Publisher; however, that's ever so slightly
    # risky in exchange for nearly 0 gain.  Uncomment everything to use.
    #directory = os.path.dirname(instance.file)
    #base = os.path.join("/results/uploads", instance.publisher.name)
    logger.info("Deleting Content from %s, %s" % (instance.publisher.name, instance.file))
    try:
        os.remove(instance.file)
        ## If the content is stored somewhere other than where we expect
        ## do nothing beyond removing it
        #if directory.startswith(base):
        #    # This is an emulation of rmdir --parents
        #    # It removes each empty directory starting at directory and then
        #    # removing each, then empty, parent until something isn't empty,
        #    # raising an OSError, or until we're at base and we stop.
        #    while not os.path.samefile(directory != base):
        #        try:
        #            os.rmdir(directory)
        #        except OSError:
        #            break
    except OSError:
        logger.error("Deleting Content from %s, %s failed." %
                     (instance.publisher.name, instance.file))


class UserEventLog(models.Model):
    text = models.TextField(blank=True)
    timeStamp = models.DateTimeField(auto_now_add=True)
    # note, this isn't exactly how I think it should go.
    # Really, we want to see log association with a diversity
    # of different user facing entities and in each of their pages, you could
    # just read the logs variable and get a list of log objects associated with
    # it.
    upload = models.ForeignKey(ContentUpload, related_name="logs")

    def __unicode__(self):
        if len(self.text) > 23:
            return u'%s...' % self.text[:20]
        else:
            return u'%s' % self.text


class UserProfile(models.Model):
    # This field is required.
    user = models.OneToOneField(User, unique=True)

    # Optional fields here
    name = models.CharField(max_length=93)
    phone_number = models.CharField(max_length=256, default="", blank=True)
    # title will not necessarily even be exposed to the end user; however,
    # we can use it to re-use this system to store important service contacts
    # such as a "Lab Manager" and an "IT Manager" which have a special
    # representation in the UI.
    title = models.CharField(max_length=256, default="user")
    # This is a simple, plain-text dumping ground for whatever the user might
    # want to document about themselves that isn't captured in the above.
    note = models.TextField(default="", blank=True)
    last_read_news_post = models.DateTimeField(default=datetime.datetime(1984, 11, 6, tzinfo=timezone.utc))

    def save(self, *args, **kwargs):
        if self.user and not self.name:
            self.name = self.user.get_full_name()
        super(UserProfile, self).save(*args, **kwargs)

    def __unicode__(self):
        return u'profile: %s' % self.user.username


@receiver(post_save, sender=User, dispatch_uid="create_profile")
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

#201203 - SequencingKit is now obsolete
class SequencingKit(models.Model):
    name = models.CharField(max_length=512, blank=True)
    description = models.CharField(max_length=3024, blank=True)
    sap = models.CharField(max_length=7, blank=True)

    def __unicode__(self):
        return u'%s' % self.name

#201203 - LibraryKit is now obsolete
class LibraryKit(models.Model):
    name = models.CharField(max_length=512, blank=True)
    description = models.CharField(max_length=3024, blank=True)
    sap = models.CharField(max_length=7, blank=True)

    def __unicode__(self):
        return u'%s' % self.name

class VariantFrequencies(models.Model):
    name = models.CharField(max_length=512, blank=True)
    description = models.CharField(max_length=3024, blank=True)

    def __unicode__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Variant Frequencies"


class LibraryKey(models.Model):
    ALLOWED_DIRECTIONS = (
        ('Forward', 'Forward'),
        ('Reverse', 'Reverse')
    )

    direction = models.CharField(max_length=20, choices=ALLOWED_DIRECTIONS, default='Forward')

    ALLOWED_RUN_MODES = (
        ('', 'Undefined'),
        ('single', 'SingleRead'),
        ('pe', 'PairedEnd')
    )

    #run mode
    runMode = models.CharField(max_length=64, choices=ALLOWED_RUN_MODES, default='single', blank=True)

    name = models.CharField(max_length=256, blank=False, unique=True)
    sequence = models.CharField(max_length=64, blank=False)
    description = models.CharField(max_length=1024, blank=True)
    isDefault = models.BooleanField("use this by default", default=False)

    class Meta:
        verbose_name_plural = "Library keys"

    def __unicode__(self):
        return u'%s' % self.name


    def save(self):
        if (self.isDefault == False and LibraryKey.objects.filter(direction=self.direction, isDefault=True).count() == 1):
            currentDefaults = LibraryKey.objects.filter(direction=self.direction, isDefault=True)
            #there should only be 1 default for a given direction at any time
            for currentDefault in currentDefaults:
                if (self.id == currentDefault.id):
                    raise ValidationError("Error: Please set another library key for %s direction to be the default before changing this key not to be the default." % self.direction)

        if (self.isDefault == True and LibraryKey.objects.filter(direction=self.direction, isDefault=True).count() > 0):
            currentDefaults = LibraryKey.objects.filter(direction=self.direction, isDefault=True)
            #there should only be 1 default for a given direction at any time
            for currentDefault in currentDefaults:
                if (self.id <> currentDefault.id):
                    currentDefault.isDefault = False
                    super(LibraryKey, currentDefault).save()

        ###print 'Going to call super.save() for LibraryKey'
        super(LibraryKey, self).save()



    def delete(self):
        if (self.isDefault == False and LibraryKey.objects.filter(direction=self.direction, isDefault=True).count() == 1):
            currentDefaults = LibraryKey.objects.filter(direction=self.direction, isDefault=True)
            #there should only be 1 default for a given direction at any time
            for currentDefault in currentDefaults:
                if (self.id == currentDefault.id):
                    raise ValidationError("Error: Deleting the default library key is not allowed. Please set another library key for %s direction to be the default before deleting this key." % self.direction)

        if (self.isDefault == True and LibraryKey.objects.filter(direction=self.direction, isDefault=True).count() > 0):
            currentDefaults = LibraryKey.objects.filter(direction=self.direction, isDefault=True)
            #there should only be 1 default for a given direction at any time
            for currentDefault in currentDefaults:
                if (self.id == currentDefault.id):
                    raise ValidationError("Error: Deleting the default library key is not allowed. Please set another library key for %s direction to be the default before deleting this key." % self.direction)

        super(LibraryKey, self).delete()


class MessageManager(models.Manager):

    def bound(self, *bindings):
        query = models.Q()
        for route_binding in bindings:
            query |= models.Q(route__startswith=route_binding)
        return self.get_query_set().filter(query)


class Message(models.Model):
    """This is a semi persistent, user oriented message intended to be
    displayed in the UI.
    """

    objects = MessageManager()

    body = models.TextField(blank=True, default="")
    level = models.IntegerField(default=20)
    route = models.TextField(blank=True, default="")
    expires = models.TextField(blank=True, default="read")
    tags = models.TextField(blank=True, default="")
    status = models.TextField(blank=True, default="unread")
    time = models.DateTimeField(auto_now_add=True)

    # Message alert levels
    DEBUG    = 10
    INFO     = 20
    SUCCESS  = 25
    WARNING  = 30
    ERROR    = 40
    CRITICAL = 50

    def __unicode__(self):
        if len(self.body) > 80:
            return u'%s...' % self.body[:77]
        else:
            return u'%s' % self.body[:80]

    @classmethod
    def log_new_message(cls, level, body, route, **kwargs):
        """A Message factory method which logs and creates a message with the
         given log level
         """
        logger.log(level, 'User Message route:\t"%s" body:\t"%s"' %
                          (route, body))
        msg = cls(body=body, route=route, level=level, **kwargs)
        msg.save()
        return msg

    @classmethod
    def debug(cls, body, route="", **kwargs):
        return cls.log_new_message(cls.DEBUG, body, route, **kwargs)

    @classmethod
    def info(cls, body, route="", **kwargs):
        return cls.log_new_message(cls.INFO, body, route, **kwargs)

    @classmethod
    def success(cls, body, route="", **kwargs):
        return cls.log_new_message(cls.SUCCESS, body, route, **kwargs)

    @classmethod
    def warn(cls, body, route="", **kwargs):
        return cls.log_new_message(cls.WARNING, body, route, **kwargs)

    @classmethod
    def error(cls, body, route="", **kwargs):
        return cls.log_new_message(cls.ERROR, body, route, **kwargs)

    @classmethod
    def critical(cls, body, route="", **kwargs):
        return cls.log_new_message(cls.CRITICAL, body, route, **kwargs)

# JUST DON'T THINK THIS IS BAKED (THOUGHT OUT) ENOUGH
#class ProjectUserVisibility(models.Model):
#    """By default, a given project for a given user has no ProjectUserVisibility
#    and the project is 'show' visibility for that user.
#    """
#    VISIBILITY_CHOICES = (
#        ('star', 'Star'), #Is this useful or a even a good idea?
#        ('show', 'Normal'),
#        ('hide', 'Hide')
#        )
#    user = models.ForeignKey(User)
#    project = models.ForeignKey(Project)
#    visibility = models.CharField(max_length=10, choices=VISIBILITY_CHOICES)


class EventLogManager(models.Manager):
    def for_model(self, model):
        """
        QuerySet for all comments for a particular model (either an instance or
        a class).
        """
        ct = ContentType.objects.get_for_model(model)
        qs = self.get_query_set().filter(content_type=ct)
        if isinstance(model, models.Model):
            qs = qs.filter(object_pk=force_unicode(model._get_pk_val()))
        return qs
    def add_entry(self, instance, message, username="ION"):
        ct = ContentType.objects.get_for_model(instance)
        ev = EventLog(object_pk=instance.pk, content_type=ct, username=username, text=message)
        ev.save()
        return ev


class EventLog (models.Model):
    # Content-object field
    content_type   = models.ForeignKey(ContentType,
            verbose_name='content type',
            related_name="content_type_set_for_%(class)s")
    object_pk      = models.PositiveIntegerField()
    content_object = generic.GenericForeignKey(ct_field="content_type", fk_field="object_pk")

    text = models.TextField('comment', max_length=3000)
    username = models.CharField(max_length=32, default="ION", blank=True)

    def get_content_object_url(self):
        """
        Get a URL suitable for redirecting to the content object.
        """
        return urlresolvers.reverse(
            "eventlog-url-redirect",
            args=(self.content_type_id, self.object_pk)
        )
    objects = EventLogManager()
    created = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return self.text


class DMFileSet (models.Model):
    # Global settings record.  For each instance, user can enable/disable
    # auto action (archive/delete), set the age threshold and diskusage threshold.
    AUTO_ACTION = (
        ('OFF', 'Disabled'),
        ('ARC', 'Archive'),
        ('DEL', 'Delete'),
    )
    type = models.CharField(max_length=48, null=False)
    description = models.CharField(max_length=256, null=True, blank=True)
    backup_directory = models.CharField(max_length=256, blank=True, null=True)
    bandwidth_limit = models.IntegerField(blank=True, null=True)
    enabled = models.BooleanField(default=False)
    include = SeparatedValuesField(null=True, blank=True)
    exclude = SeparatedValuesField(null=True, blank=True)
    keepwith = json_field.JSONField(blank=True, null=True)
    version = models.CharField(max_length=8)
    auto_trigger_age = models.IntegerField(null=True, blank=True)
    auto_trigger_usage = models.IntegerField(null=True, blank=True)
    auto_action = models.CharField(max_length=8, choices=AUTO_ACTION, default='OFF')
    del_empty_dir = models.BooleanField(default=True)

    def __unicode__(self):
        return u'%s (%s)' % (self.type, self.version)


class DMFileStat (models.Model):
    # These are status fields for a Report instance to track state and action status of
    # the selected files.
    ACT_STATES = (
        ('L', 'Local'),             # Files are local
        ('S', 'Selected'),          # Selected for autoaction
        ('N', 'Notified'),          # user has been notified of action pending
        ('A', 'Acknowledged'),      # user acknowledges to proceed with action
        ('SA', 'Archive Pending'),  # Selected for archive manual action
        ('SE', 'Export Pending'),   # Selected for export manual action
        ('SD', 'Delete Pending'),   # Selected for delete (after suspended)
        ('AG', 'Archiving'),
        ('DG', 'Deleting'),
        ('EG', 'Exporting'),
        ('AD', 'Archived'),
        ('DD', 'Deleted'),
        ('E', 'Error'),             # Action resulted in error
        )

    # state of file action
    action_state = models.CharField(max_length=8, choices=ACT_STATES, default='L')

    # path to archived filed
    archivepath = models.CharField(max_length=512, blank=True, null=True)

    # kilobytes used by files
    diskspace = models.FloatField(blank=True, null=True)

    # status of files currently being used by analysis pipeline
    files_in_use = models.CharField(max_length=512, blank=True, default='')

    # link to result object (the Report)
    result = models.ForeignKey(Results, null=True, blank=True)

    # link to DMFileSet object (the file selection filter)
    dmfileset = models.ForeignKey(DMFileSet, null=True, blank=True)

    # Preserve fileset files (except for sigproc files which is stored in experiment.storage_options)
    preserve_data = models.BooleanField(default=False)

    # datetime stamp when object created; do not auto fill
    created = models.DateTimeField(blank=True, null=True)

    # field to store manual action username and comment
    user_comment = json_field.JSONField(blank=True)

    def save(self, *args, **kwargs):
        if not self.id:
            self.created = timezone.now()
        super(DMFileStat, self).save(*args, **kwargs)

    def getpreserved(self):
        if self.dmfileset.type == dmactions_types.SIG:
            return self.result.experiment.storage_options == 'KI'
        else:
            return self.preserve_data

    def setpreserved(self, keep_flag):
        if self.dmfileset.type == dmactions_types.SIG:
            self.result.experiment.storage_options = 'KI' if keep_flag else 'D'
            self.result.experiment.save()
        else:
            self.preserve_data = keep_flag
            self.save()

    def setactionstate(self, state):
        if state in [c for (c,d) in DMFileStat.ACT_STATES]:
            self.action_state = state
            self.save()
            # update related dmfilestats
            if self.dmfileset.type == dmactions_types.SIG:
                exp_id = self.result.experiment_id
                DMFileStat.objects.filter(dmfileset__type=dmactions_types.SIG, result__experiment__id=exp_id).update(action_state = state)
        else:
            raise Exception("Failed to set action_state. Invalid state: '%s'" % state)

    def isdisposed(self):
        return bool(self.action_state in ['AG','DG','AD','DD'])

    def isarchived(self):
        return bool(self.action_state in ['AG','AD'])

    def isdeleted(self):
        return bool(self.action_state in ['DG','DD'])

    def in_process(self):
        return bool(self.action_state in ['AG','DG','EG','SA','SE','SD'])


class FileMonitor(models.Model):
    """Record the details of a file download from a remote server to the TS
    by a backround or automated process.
    """

    url = models.CharField(max_length=2000)
    local_dir = models.CharField(max_length=512, default="")
    name = models.CharField(max_length=255, default="")
    size = models.BigIntegerField(default=None, null=True)
    progress = models.BigIntegerField(default=0)
    status = models.CharField(max_length=60, default="")
    celery_task_id = models.CharField(max_length=60, default="", blank=True)
    tags = models.CharField(max_length=1024, default="")

    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def percent_progress(self):
        if self.size:
            return "{0:.1f}".format(100.0 * self.progress / self.size)
        else:
            return "..."

    def full_path(self):
        if self.local_dir and self.name:
            return os.path.join(self.local_dir, self.name)

    def delete(self):
        # delete the files from the filesystem as well as the database
        revoke(self.celery_task_id, terminate=True)
        try:
            if os.path.exists(self.local_dir):
                shutil.rmtree(self.local_dir)
        except OSError:
            return False

        super(FileMonitor, self).delete()
        return True

class MonitorData(models.Model):
    treeDat = json_field.JSONField(blank=True)
    name = models.CharField(max_length=128, default="")


class NewsPost(models.Model):
    guid = models.CharField(max_length=64, null=True, blank=True)
    title = models.CharField(max_length=140, blank=True, default="")
    summary = models.CharField(max_length=300, blank=True, default="")
    link = models.CharField(max_length=2000, blank=True, default="")
    updated = models.DateTimeField(default=timezone.now)

    def __unicode__(self):
        return self.title if len(self.title) <= 60 else self.title[:57] + u'...'

class AnalysisArgs(models.Model):
    # Holds default command line args for selected chipType and kits
    name = models.CharField(max_length=256, blank=False, unique=True)
    chipType = models.CharField(max_length=128, default='')
    active = models.BooleanField(default=True)
    # specify these args are to be used as default for this chip type, only one chip_default should exist per chip
    chip_default = models.BooleanField(default=False)

    sequenceKitName = models.CharField(max_length=512, blank=True, default='')
    templateKitName = models.CharField(max_length=512, blank=True, default='')
    libraryKitName = models.CharField(max_length=512, blank=True, default='')
    samplePrepKitName = models.CharField(max_length=512, blank=True, default='')

    # Beadfind args
    beadfindargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Beadfind args")
    thumbnailbeadfindargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Thumbnail Beadfind args")
    # Analysis args
    analysisargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Analysis args")
    thumbnailanalysisargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Thumbnail Analysis args")
    # PreBasecaller args
    prebasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Pre Basecaller args, used for recalibration")
    prethumbnailbasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Thumbnail Pre Basecaller args, used for recalibration")
    # Basecaller args
    basecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Basecaller args")
    thumbnailbasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Thumbnail Basecaller args")
    # Alignment args
    alignmentargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Alignment args")
    thumbnailalignmentargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Thumbnail Alignment args")

    def get_args(self):
        args = {
            'beadfindargs': self.beadfindargs,
            'analysisargs': self.analysisargs,
            'basecallerargs': self.basecallerargs,
            'prebasecallerargs': self.prebasecallerargs,
            'alignmentargs': self.alignmentargs,
            'thumbnailbeadfindargs':    self.thumbnailbeadfindargs,
            'thumbnailanalysisargs':    self.thumbnailanalysisargs,
            'thumbnailbasecallerargs':  self.thumbnailbasecallerargs,
            'prethumbnailbasecallerargs':  self.prethumbnailbasecallerargs,
            'thumbnailalignmentargs': self.thumbnailalignmentargs
        }
        return args

    @classmethod
    def best_match(cls, chipType, sequenceKitName='', templateKitName='', libraryKitName='', samplePrepKitName=''):
        ''' Find args that best match given chip type and kits.
            If chipType not found returns None.
            If none of the kits matched returns chip default.
        '''
        # chip name backwards compatibility
        chipType = chipType.replace('"','')
        if Chip.objects.filter(name=chipType).count() == 0:
            chipType = chipType[:3]

        args = AnalysisArgs.objects.filter(chipType=chipType).order_by('-pk')
        args_count = args.count()
        if args_count == 0:
            best_match = None
        elif args_count == 1:
            best_match = args[0]
        else:
            kits = {
                'sequenceKitName': sequenceKitName,
                'templateKitName': templateKitName,
                'libraryKitName': libraryKitName,
                'samplePrepKitName': samplePrepKitName
            }
            match = [0]*args_count
            match_no_blanks = [0]*args_count
            for i, arg in enumerate(args):
                for key, value in kits.items():
                    if getattr(arg, key) == value:
                        match[i] += 1
                        if value:
                            match_no_blanks[i] += 1
                
            if max(match_no_blanks) > 0:
                best_match = args[match.index(max(match))]
            else:
                try: 
                    best_match = args.get(chip_default=True)
                except:
                    best_match = None

        return best_match

    class Meta:
        verbose_name_plural = "Analysis Args"
