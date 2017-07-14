# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

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

from __future__ import absolute_import

import datetime
import re
import os
import traceback
from django.core.exceptions import ValidationError
from ion.plugin.info import PluginInfo
import iondb.settings

from django.db import models
from django.core.exceptions import ObjectDoesNotExist
from iondb.utils.utils import bytesToHumanReadableSize, directorySize, getPackageName
from distutils.version import LooseVersion

from django.db.models.query_utils import Q

from iondb.rundb import json_field

import ion
import tempfile
import shutil
import uuid
import hmac
import random
import string
import base64
import logging
from iondb.rundb import tasks

from iondb.rundb.data import dmactions_types
from iondb.rundb.separatedValuesField import SeparatedValuesField
from iondb.plugins.manager import pluginmanager
try:
    from hashlib import sha1
except ImportError:
    import sha
    sha1 = sha.sha

logger = logging.getLogger(__name__)

from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes import generic
from django.contrib.sessions.serializers import PickleSerializer
from django.utils.encoding import force_unicode
from django.utils.functional import cached_property
from django.utils import timezone
from django.core import urlresolvers
from django.db import transaction
from django.db.models import *
from django.db.models.signals import post_save, pre_delete, post_delete, m2m_changed
from django.dispatch import receiver
from distutils.version import LooseVersion
from iondb.celery import app as celery
from django.conf import settings
import json
import xmlrpclib
from iondb.utils.utils import convert
import subprocess
import StringIO
import csv

def model_to_csv(models, fields=None):
    """generates a csv from a list of model objects and returns a string object"""
    if len(models) == 0:
        return ''

    # get a memory file instance and create a csv writer
    memory_file = StringIO.StringIO()
    writer = csv.writer(memory_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    # get a list of field name from the first model object
    model_field_white_list = [BooleanField, CharField, DateField, DateTimeField, DecimalField,
                              EmailField, FilePathField, FloatField, IntegerField,
                              GenericIPAddressField, NullBooleanField, PositiveSmallIntegerField,
                              SlugField, SmallIntegerField, TextField, TimeField, URLField]

    header_titles = list()
    if fields:
        header_titles += fields
    else:
        for f in models[0]._meta.fields:
            if type(f) in model_field_white_list:
                header_titles.append(f.name)

    # write to the csv memory file
    writer.writerow(header_titles)
    for instance in models:
        writer.writerow([unicode(getattr(instance, f)).encode('utf-8') for f in header_titles])

    # return the string object
    return memory_file.getvalue()


# Taken from tastypie apikey generation
def _generate_key():
    # Hmac that beast, generate with a new random uuid
    return hmac.new(str(uuid.uuid4()), digestmod=sha1).hexdigest()


class Project(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ManyToMany 'plans' from PlannedExperiment
    # ForeignKey 'parentPlan' from PlannedExperiment
    # ManyToManyField 'results' from Results

    #TODO The name should not be uniq because there can be a public one
    #and a private one with the same name and that would not work
    name = models.CharField(max_length=64, unique=True)
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
        projects = [item for item in projects if item]
        projects = [name.strip() for name in projects]
        projects = ['_'.join(name.split()) for name in projects]

        #for 2 projects with the same name (but just different case), __iexact will fail with MultipleObjectsReturned error
        return [Project.objects.get_or_create(name__exact=name, defaults={'name': name, 'creator': user})[0] for name in projects]


class KitInfoManager(models.Manager):

    def get_by_natural_key(self, uid):
        return self.get(uid=uid)


class KitInfo(models.Model):

    # FOREIGN KEY DEFINITIONS
    # Foreign key 'seqKit_applProduct_set' from ApplProduct
    # Foreign key 'libKit_applProduct_set' from ApplProduct
    # Foreign key 'templateKit_applProduct_set' from ApplProduct
    # Foreign key 'controlSeqKit_applProduct_set' from ApplProduct
    # Foreign key 'samplePrepKit_applProduct_set' from ApplProduct
    # Foreign key 'ionChefPrepKit_applProduct_set' from ApplProduct
    # Foreign key 'ionChefSeqKit_applProduct_set' from ApplProduct

    ALLOWED_KIT_TYPES = (
        ('SequencingKit', 'SequencingKit'),
        ('LibraryKit', 'LibraryKit'),
        ('LibraryPrepKit', "LibraryPrepKit"),
        ('TemplatingKit', 'TemplatingKit'),
        ('AdapterKit', "AdapterKit"),
        ('ControlSequenceKit', "ControlSequenceKit"),
        ('SamplePrepKit', "SamplePrepKit"),
        ('IonChefPrepKit', 'IonChefPrepKit'),
        ('AvalancheTemplateKit', 'AvalancheTemplateKit'),
        ('ControlSequenceKitType', 'ControlSequenceKitType'),
    )

    kitType = models.CharField(max_length=64, choices=ALLOWED_KIT_TYPES)
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

    isActive = models.BooleanField(default=True)

    ALLOWED_INSTRUMENT_TYPES = (
        ('', 'Any'),
        ('pgm', 'PGM'),
        ('proton', 'Proton'),
        ('S5', 'S5'),
        ('proton;S5', "Proton or S5")
    )
    #compatible instrument type
    instrumentType = models.CharField(max_length=64, choices=ALLOWED_INSTRUMENT_TYPES, default='', blank=True)

    ALLOWED_APPLICATION_TYPES = (
        ('', 'Any'),
        ('AMPS_ANY', 'Any AmpliSeq application'),
        ('RNA', 'RNA'),
        ('AMPS', "AmpliSeq DNA"),
        ('AMPS_RNA', 'AmpliSeq RNA'),
        ('RNA;AMPS_RNA', 'Any RNA application'),
        ('AMPS_EXOME', 'AmpliSeq Exome'),
        ('TARS', "Target Sequencing"),
        ('TAG_SEQUENCING', "Tag Sequencing"),
        ('GENS', "Generic Sequencing"),
        ('WGNM', "Whole Genome")
    )
    #compatible application types
    applicationType = models.CharField(max_length=64, choices=ALLOWED_APPLICATION_TYPES, default='', blank=True, null=True)


    # Rules to get default flows for Kit categories, used by Plan GUI
    _category_flowCount_rules = [
        {"category": "multipleTemplatingSize", "flowCount": 500, "templatingSize": "200"},
        {"category": "multipleTemplatingSize", "flowCount": 850, "templatingSize": "400"},

        {"category": "s5v1Kit", "flowCount": 500, "readLength": "200"},
        {"category": "s5v1Kit", "flowCount": 850, "readLength": "400"},

        {"category": "s5ExTKit", "flowCount": 1350, "chipType": "521"},
        {"category": "s5ExTKit", "flowCount": 1350, "chipType": "530"},
    ]

    ALLOWED_CATEGORIES = (
        ('', 'Any'),
        ('flowOverridable', 'Flows can be overridden'),
        ('flowOverridable;', 'Flow count can be overridden'),        
        ('bcShowSubset;bcRequired;', 'Mandatory barcode kit selection'),
        #("flowOverridable;readLengthDerivableFromFlows;flowsDerivableFromReadLength;", "Hi-Q sequencing kit categories"),
        ("flowOverridable;readLengthDerivableFromFlows;", "Hi-Q sequencing kit categories"),
        ("multipleTemplatingSize;supportLibraryReadLength", "Hi-Q templating kit categories"),
        #("readLengthDerivableFromFlows;flowsDerivableFromReadLength;", "Non-Hi-Q sequencing kit categories"),
        ("readLengthDerivableFromFlows;", "Non-Hi-Q sequencing kit categories"),
        ("s5v1Kit", "S5 v1 kit categories"),
        ("s5v1Kit;flowOverridable;", "S5 v1 kit categories with special flow count handling"),
        ("s5v1Kit;flowOverridable;multipleReadLength;", "S5 v1 templating kit categories"),
        ("s5v1Install", "S5 v1 install categories"),
        ("s5ExTKit", "S5 ExT kit categories"),
        ("s5541", "S5 541 kit categories"),
        ("s5v1Kit;flowOverridable;multipleReadLength;s5541", "S5 541 v1 templating kit categories"),
        ("filter_s5HidKit", "S5 HID kit filter"),
        ("filter_s5HidEA", "S5 HID early access filter"),
        ("filter_muSeek", "MuSeek filter"),
        ('multipleTemplatingSize;supportLibraryReadLength;samplePrepProtocol;protonRnaWTSamplePrep', 'Proton templating size, read length, RNA WT Chef Protocol'),
        ('multipleTemplatingSize;supportLibraryReadLength;samplePrepProtocol;hidSamplePrep', 'templating size, read length, HID Chef Protocol'),
        ('s5v1Kit;flowOverridable;multipleReadLength;samplePrepProtocol;s5RnaWTSamplePrep', "S5 flow count, read length, RNA WT Protocol"),
        ('samplePrepProtocol;hidSamplePrep', 'Library kit-based HID Sample Prep Protocol'),
        ("s5v1Kit;flowOverridable;multipleReadLength;samplePrepProtocol;s5MyeloidSamplePrep", "S5 flow count, read length, Myeloid Protocol"),
        ("s5v1Kit;multipleTemplatingSize;flowOverridable;multipleReadLength;samplePrepProtocol;s5MyeloidSamplePrep", "Templating size, S5 flow count, read length, Myeloid Protocol"),
        ("s5v1Kit;multipleTemplatingSize;flowOverridable;samplePrepProtocol;s5MyeloidSamplePrep", "Templating size, S5 flow count, Myeloid Protocol"),
    )
    categories = models.CharField(max_length=256, choices=ALLOWED_CATEGORIES, default='', blank=True, null=True)
    libraryReadLength = models.PositiveIntegerField(default=0)

    #the first value, if present, is used as the default value
    ALLOWED_TEMPLATING_SIZE = (
        ("", 'Unspecified'),
        ("200", "200"),
        ("400", "400"),
        ("200;400", "200 or 400")
    )
    templatingSize = models.CharField(max_length=64, choices=ALLOWED_TEMPLATING_SIZE, default='', blank=True, null=True)

    ALLOWED_SAMPLE_PREP_INSTRUMENT_TYPES = (
        ('', 'Unspecified'),
        ('OT', 'OneTouch'),
        ('IC', 'IonChef'),
        ("OT_IC", "Both OneTouch and IonChef")
    )
    #compatible sample prep instrument type
    samplePrep_instrumentType = models.CharField(max_length=64, choices=ALLOWED_SAMPLE_PREP_INSTRUMENT_TYPES, default='', blank=True)

    ALLOWED_CHIP_TYPES = (
        ('', 'Unspecified'),
        ('510;520;530', 'S5 510, 520 or 530'),        
        ('520;530', 'S5 520 or 530'),
        ('540', 'S5 540'),
        ('540;550', 'S5 540 or 550'),
        ('541', 'S5 541'),
        ('521;530', 'S5 521 or 530'),
        ('510;520;521;530', 'S5 510, 520, 521 or 530'),     
        ('520;521;530', 'S5 520, 521 or 530'),
        ('P2.2.1;P2.2.2', "Proton PQ"),
        ('900;P1.0.19;P1.0.20;P1.1.17;P1.1.541;P1.2.18;P2.0.1;P2.1.1;P2.3.1', "Proton PI")
    )

    #compatible chip types
    chipTypes = models.CharField(max_length=127, choices=ALLOWED_CHIP_TYPES, default='', blank=True)
    
    defaultFlowOrder = models.ForeignKey("FlowOrder", blank=True, null=True, default=None)
    defaultThreePrimeAdapter = models.ForeignKey("ThreePrimeadapter", blank=True, null=True, default=None)
    
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
        return self.get(barcode=kitBarcode)


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
        return self.get(runType=runType)


class RunType(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'applType_analysisArgs' from AnalysisArgs

    runType = models.CharField(max_length=512, unique=True)
    barcode = models.CharField(max_length=512, blank=True)
    description = models.TextField(blank=True)
    meta = json_field.JSONField(blank=True, null=True, default="")
    isActive = models.BooleanField(default=True)

    ALLOWED_NUCLEOTIDE_TYPES = (
        ('', 'Any'),
        ('dna', 'DNA'),
        ('rna', 'RNA'),
        ('dna_rna', 'DNA+RNA')
    )

    nucleotideType = models.CharField(max_length=64, choices=ALLOWED_NUCLEOTIDE_TYPES, default='dna', blank=True)

    objects = RunTypeManager()

    applicationGroups = models.ManyToManyField("ApplicationGroup", related_name='applications', null=True)

    alternate_name = models.CharField(max_length=512,
                                      blank=True,
                                      null=True,
                                      default="")

    def __unicode__(self):
        return self.runType

    def natural_key(self):
        return (self.runType, )  # must return a tuple


class ApplicationGroupManager(models.Manager):

    def get_by_natural_key(self, uid):
        return self.get(uid=uid)

class common_CVManager(models.Manager):

    def get_by_natural_key(self, uid):
        return self.get(uid=uid)


class common_CV(models.Model):
    # Common Control vacabulary model - TS-12954
    # The current use case of this model is to allow the same Chef script to be executed with slight variation in steps and/or parameters

    # FOREIGN KEY DEFINITIONS
    # samplePrepProtocol_applProduct_set from ApplProduct

    ALLOWED_CONTROL_VACABULORIES = (
        ('samplePrepProtocol', 'Sample Prep Protocol'),
        ('applicationCategory', 'Application Category')
    )

    cv_type = models.CharField(max_length=127, choices=ALLOWED_CONTROL_VACABULORIES, blank=False, null=False)
    value = models.CharField(max_length=127, blank=False, null=False)
    displayedValue = models.CharField(max_length=127, blank=False, null=False, unique=True)
    description = models.CharField(max_length=1024, blank=True, null=True)
    isDefault = models.BooleanField(default=True)
    isActive = models.BooleanField(default=True)
    isVisible = models.BooleanField(default=True)

    ALLOWED_CATEGORIES = (
        ('', 'Unspecified'),
        ('hidSamplePrep', 'HID Sample Prep Protocol'),
        ('protonRnaWTSamplePrep', 'Proton RNA WT Sample Prep Protocol'),
        ('s5RnaWTSamplePrep', 'S5 RNA WT Sample Prep Protocol'),
        ('protonRnaWTSamplePrep;s5RnaWTSamplePrep', 'RNA WT Sample Prep Protocol'),
        ("s5MyeloidSamplePrep", 'S5 Myeloid Sample Prep Protocol'),
        ("AMPS", 'AmpliSeq DNA products'),
        ("AMPS;AMPS_RNA;AMPS_DNA_RNA", "Oncology products"),
        ("AMPS_RNA", "AmpliSeq RNA products"),
        ("WGNM", "Whole genome products"),
        ("TARS_16S", "16S products")
    )
    categories = models.CharField(max_length=256, choices=ALLOWED_CATEGORIES, default='', blank=True, null=True)

    # compatible sample prep instrument type
    ALLOWED_SAMPLE_PREP_INSTRUMENT_TYPES = (
        ('', 'Unspecified'),
        ('OT', 'OneTouch'),
        ('IC', 'IonChef'),
        ("OT_IC", "Both OneTouch and IonChef")
    )
    samplePrep_instrumentType = models.CharField(max_length=64, choices=ALLOWED_SAMPLE_PREP_INSTRUMENT_TYPES,
                                                 default='', blank=True)

    # compatible sequencing instrument type
    ALLOWED_SEQUENCING_INSTRUMENT_TYPES = (
        ('', 'Any'),
        ('pgm', 'PGM'),
        ('proton', 'Proton'),
        ('s5', 'S5')
    )
    sequencing_instrumentType = models.CharField(max_length=64, choices=ALLOWED_SEQUENCING_INSTRUMENT_TYPES,
                                                 default='', blank=True)

    uid = models.CharField(max_length=10, unique=True, blank=False)

    objects = common_CVManager()

    def __unicode__(self):
        return u'%s' % self.displayedValue

    def natural_key(self):
        return (self.uid,)  # must return a tuple

    class Meta:
        unique_together = (('cv_type', 'value'),)

class ApplicationGroup(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ManyToMany 'applications' from RunType
    # ForeignKey 'applGroup_analysisArgs' from AnalysisArgs

    name = models.CharField(max_length=127, blank=False, null=False)
    description = models.CharField(max_length=1024, blank=True, null=True)
    isActive = models.BooleanField(default=True)

    uid = models.CharField(max_length=32, unique=True, blank=False)

    objects = ApplicationGroupManager()

    def __unicode__(self):
        return self.name

    def natural_key(self):
        return (self.uid,)  # must return a tuple


class ApplProductManager(models.Manager):

    def get_by_natural_key(self, productCode):
        return self.get(productCode=productCode)


class ApplProduct(models.Model):
    #application with no product will have a default product pre-loaded to db to hang all the
    #application-specific settings
    productCode = models.CharField(max_length=64, unique=True, default='any', blank=False)
    productName = models.CharField(max_length=128, blank=False)
    description = models.CharField(max_length=1024, blank=True)
    applType = models.ForeignKey(RunType)
    applicationGroup = models.ForeignKey(ApplicationGroup, blank=True, null=True)
    isActive = models.BooleanField(default=True)
    #if isVisible is false, it will not be shown as a choice in UI
    isVisible = models.BooleanField(default=False)
    defaultSequencingKit = models.ForeignKey(KitInfo, related_name='seqKit_applProduct_set', blank=True, null=True)
    defaultLibraryKit = models.ForeignKey(KitInfo, related_name='libKit_applProduct_set', blank=True, null=True)
    defaultGenomeRefName = models.CharField(max_length=1024, blank=True, null=True)
    #this is analogous to bedFile in PlannedExperiment
    defaultTargetRegionBedFileName = models.CharField(max_length=1024, blank=True, null=True)
    #this is analogous to regionFile in PlannedExperiment
    defaultHotSpotRegionBedFileName = models.CharField(max_length=1024, blank=True, null=True)

    defaultChipType = models.CharField(max_length=128, blank=True, null=True)
    isDefault = models.BooleanField(default=False)

    defaultFlowCount = models.PositiveIntegerField(default=0)
    defaultTemplateKit = models.ForeignKey(KitInfo, related_name='templateKit_applProduct_set', blank=True, null=True)
    defaultControlSeqKit = models.ForeignKey(KitInfo, related_name='controlSeqKit_applProduct_set', blank=True, null=True)

    #sample preparation kit
    defaultSamplePrepKit = models.ForeignKey(KitInfo, related_name='samplePrepKit_applProduct_set', blank=True, null=True)

    isHotspotRegionBEDFileSuppported = models.BooleanField(default=True)

    isDefaultBarcoded = models.BooleanField(default=False)
    defaultBarcodeKitName = models.CharField(max_length=128, blank=True, null=True)
    defaultIonChefPrepKit = models.ForeignKey(KitInfo, related_name='ionChefPrepKit_applProduct_set', blank=True, null=True)
    defaultIonChefSequencingKit = models.ForeignKey(KitInfo, related_name='ionChefSeqKit_applProduct_set', blank=True, null=True)

    isReferenceSelectionSupported = models.BooleanField(default=True)
    isTargetTechniqueSelectionSupported = models.BooleanField(default=True)
    isTargetRegionBEDFileSupported = models.BooleanField(default=True)
    isTargetRegionBEDFileSelectionRequiredForRefSelection = models.BooleanField(default=False)

    isControlSeqTypeBySampleSupported = models.BooleanField(default=False)
    isReferenceBySampleSupported = models.BooleanField(default=False)
    isTargetRegionBEDFileBySampleSupported = models.BooleanField(default=False)
    isHotSpotBEDFileBySampleSupported = models.BooleanField(default=False)
    isDualNucleotideTypeBySampleSupported = models.BooleanField(default=False)

    isBarcodeKitSelectionRequired = models.BooleanField(default=False)

    ALLOWED_BARCODE_KIT_SELECTABLE_TYPES = (
        ('all', 'All'),
        ('', 'Unspecified'),
        ('dna', 'DNA'),
        ('rna', 'RNA'),
        ('dna+', "DNA and Unspecified"),
        ('rna+', "RNA and Unspecified")
    )
    barcodeKitSelectableType = models.CharField(max_length=64, choices=ALLOWED_BARCODE_KIT_SELECTABLE_TYPES, default='', blank=True)
    isSamplePrepKitSupported = models.BooleanField(default=True)


    ALLOWED_INSTRUMENT_TYPES = (
        ('', 'Any'),
        ('pgm', 'PGM'),
        ('proton', 'Proton'),
        ('s5', 'S5')
    )
    #compatible instrument type
    instrumentType = models.CharField(max_length=64, choices=ALLOWED_INSTRUMENT_TYPES, default='', blank=True)

    defaultFlowOrder = models.ForeignKey("FlowOrder", related_name='flowOrder_applProduct_set', blank=True, null=True)

    isDefaultForInstrumentType = models.BooleanField(default=False)

    ALLOWED_CATEGORIES = (
        ('', 'Unspecified'),
        ('hidSamplePrep', 'HID Sample Prep Protocol'),
        ('protonRnaWTSamplePrep', 'Proton RNA WT Sample Prep Protocol'),
        ('s5RnaWTSamplePrep', 'S5 RNA WT Sample Prep Protocol'),
        ('s5MyeloidSamplePrep', 'S5 Myeloid Sample Prep Protocol')   
    )
    categories = models.CharField(max_length=256, choices=ALLOWED_CATEGORIES, default='', blank=True, null=True)
    defaultSamplePrepProtocol = models.ForeignKey(common_CV, related_name='common_CV_applProduct_set', blank=True, null=True)

    def __unicode__(self):
        return u'%s' % self.productName

    def natural_key(self):
        return (self.productCode, )    # must return a tuple

    @property
    def barcodeKitSelectableTypes_list(self):
        if self.barcodeKitSelectableType == "":
            types = ["", "none"]
        elif self.barcodeKitSelectableType == "dna":
            types = ["dna"]
        elif self.barcodeKitSelectableType == "rna":
            types = ["rna"]
        elif self.barcodeKitSelectableType == "dna+":
            types = ["dna", "", "none"]
        elif self.barcodeKitSelectableType == "rna+":
            types = ["rna", "", "none"]
        else:
            types = [v[0] for v in dnaBarcode.ALLOWED_BARCODE_TYPES]

        return types


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
        # Experiment
        extra_kwargs['autoAnalyze'] = kwargs.pop('x_autoAnalyze', True)
        extra_kwargs['usePreBeadfind'] = kwargs.pop('x_usePreBeadfind', True)
        extra_kwargs['star'] = kwargs.pop('x_star', False)
        extra_kwargs['chipType'] = kwargs.pop('x_chipType', "")
        extra_kwargs['chipBarcode'] = kwargs.pop('chipBarcode', "") or ""
        extra_kwargs['flows'] = kwargs.pop('x_flows', "0")
        extra_kwargs['notes'] = kwargs.pop('x_notes', "")
        extra_kwargs['sequencekitname'] = kwargs.pop('x_sequencekitname', "")
        if 'x_platform' in kwargs:
            extra_kwargs['platform'] = kwargs.pop('x_platform', "")
        extra_kwargs['x_isSaveBySample'] = kwargs.pop('x_isSaveBySample', False)
        extra_kwargs['flowsInOrder'] = kwargs.pop('x_flowsInOrder', "")

        # EAS
        extra_kwargs['barcodedSamples'] = kwargs.pop('x_barcodedSamples', {})
        extra_kwargs['barcodeKitName'] = kwargs.pop('x_barcodeId', "")
        extra_kwargs['targetRegionBedFile'] = kwargs.pop('x_bedfile', "")
        extra_kwargs['hotSpotRegionBedFile'] = kwargs.pop('x_regionfile', "")
        extra_kwargs['sseBedFile'] = kwargs.pop('x_sseBedFile', "")
        extra_kwargs['libraryKey'] = kwargs.pop('x_libraryKey', "")
        extra_kwargs['tfKey'] = kwargs.pop('tfKey', "")
        extra_kwargs['threePrimeAdapter'] = kwargs.pop('x_forward3primeadapter', "")
        extra_kwargs['reference'] = kwargs.pop('x_library', "")
        extra_kwargs['selectedPlugins'] = kwargs.pop('x_selectedPlugins', {})
        extra_kwargs['libraryKitName'] = kwargs.pop('x_librarykitname', "")
        extra_kwargs['variantfrequency'] = kwargs.pop('x_variantfrequency', "")
        extra_kwargs['isDuplicateReads'] = kwargs.pop('x_isDuplicateReads', False)
        extra_kwargs['base_recalibration_mode'] = kwargs.pop('x_base_recalibration_mode', "no_recal")
        extra_kwargs['realign'] = kwargs.pop('x_realign', False)
        extra_kwargs['mixedTypeRNA_reference'] = kwargs.pop("x_mixedTypeRNA_library", "")
        extra_kwargs['mixedTypeRNA_targetRegionBedFile'] = kwargs.pop("x_mixedTypeRNA_bedfile", "")
        extra_kwargs['mixedTypeRNA_hotSpotRegionBedFile'] = kwargs.pop("x_mixedTypeRNA_regionfile", "")

        #EAS - analysis args
        extra_kwargs["beadfindargs"] = kwargs.pop("x_beadfindargs", "")
        extra_kwargs["analysisargs"] = kwargs.pop("x_analysisargs", "")
        extra_kwargs["prebasecallerargs"] = kwargs.pop("x_prebasecallerargs", "")
        extra_kwargs["calibrateargs"] = kwargs.pop("x_calibrateargs", "")
        extra_kwargs["basecallerargs"] = kwargs.pop("x_basecallerargs", "")
        extra_kwargs["alignmentargs"] = kwargs.pop("x_alignmentargs", "")
        extra_kwargs["ionstatsargs"] = kwargs.pop("x_ionstatsargs", "")

        extra_kwargs["thumbnailbeadfindargs"] = kwargs.pop("x_thumbnailbeadfindargs", "")
        extra_kwargs["thumbnailanalysisargs"] = kwargs.pop("x_thumbnailanalysisargs", "")
        extra_kwargs["prethumbnailbasecallerargs"] = kwargs.pop("x_prethumbnailbasecallerargs", "")
        extra_kwargs["thumbnailcalibrateargs"] = kwargs.pop("x_thumbnailcalibrateargs", "")
        extra_kwargs["thumbnailbasecallerargs"] = kwargs.pop("x_thumbnailbasecallerargs", "")
        extra_kwargs["thumbnailalignmentargs"] = kwargs.pop("x_thumbnailalignmentargs", "")
        extra_kwargs["thumbnailionstatsargs"] = kwargs.pop("x_thumbnailionstatsargs", "")
        extra_kwargs["custom_args"] = kwargs.pop("x_custom_args", "")

        # Sample
        extra_kwargs['sample'] = kwargs.pop('x_sample', "")
        extra_kwargs['sample_external_id'] = kwargs.pop('x_sample_external_id', "")
        extra_kwargs['sample_description'] = kwargs.pop('x_sample_description', "")
        extra_kwargs['sampleDisplayedName'] = kwargs.pop('x_sampleDisplayedName', "")

        # Sample Set
        extra_kwargs['sampleSet'] = kwargs.pop('sampleSet', "")

        # QC
        for qcType in QCType.objects.values_list('qcName', flat=True):
            extra_kwargs[qcType] = kwargs.pop(qcType, "")

        logger.info("EXIT PlannedExpeirmentManager.extract_extra_kwargs... extra_kwargs=%s" % (extra_kwargs))

        return kwargs, extra_kwargs


    def save_plan(self, planOid, **kwargs):
        popped_kwargs, extra_kwargs = self._extract_extra_kwargs(**kwargs)

        logger.info("PlannedExpeirmentManager.save_plan() planOid=%s; after extract_extra_kwargs... popped_kwargs==%s" % (str(planOid), popped_kwargs))
        logger.info("PlannedExpeirmentManager.save_plan() after extract_extra_kwargs... extra_kwargs=%s" % (extra_kwargs))

        if planOid < 0:
            plan = self.create(**popped_kwargs)
        else:
            plan = self.get(pk=planOid)
            # start recording changed fields for history log
            # this must be done for PlannedExperiment and each plan-related object before it's updated
            plan.update_changed_fields_for_plan_history(popped_kwargs, plan)

            for key, value in popped_kwargs.items():
                setattr(plan, key, value)

        plan.save()

        if plan:
            isPlanCreated = (planOid < 0)
            plan.save_plannedExperiment_association(isPlanCreated, **extra_kwargs)
            plan.update_plan_qcValues(**extra_kwargs)

            # sampleSets
            sampleSets = extra_kwargs.get('sampleSet')
            if sampleSets:
                for sampleSet in sampleSets.all():
                    plan.sampleSets.add(sampleSet)

            plan.save_plan_history_log()

        return plan, extra_kwargs


class PlannedExperiment(models.Model):

    """
    Create a planned run to ease the pain on manually entry on the PGM
    """

    # FOREIGN KEY DEFINITIONS
    # OneToOneField 'experiment' from Experiment

    #plan name
    planName = models.CharField(max_length=512, blank=True, null=True)

    #Global uniq id for planned run
    planGUID = models.CharField(max_length=512, blank=True, null=True)

    #make a id for easy entry
    planShortID = models.CharField(max_length=5, blank=True, null=True, db_index=True)

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

    # "transferred" : for plan share feature, plan has been copied to another server and cannot be changed on this TS
    # "inactive" : status to exclude plan template from displaying on the Templates web page

    ALLOWED_PLAN_STATUS = (
        ('', 'Undefined'),
        ('pending', 'Pending'),
        ('voided', 'Voided'),
        ('reserved', 'Reserved'),
        ('planned', 'Planned'),
        ('transferred', 'Transferred'),
        ('inactive', 'Inactive'),
        ('run', 'Run')
    )

    #planStatus
    planStatus = models.CharField(max_length=512, blank=True, choices=ALLOWED_PLAN_STATUS, default='')

    #who ran this
    username = models.CharField(max_length=128, blank=True, null=True)

    #what PGM started this
    planPGM = models.CharField(max_length=128, blank=True, null=True)

    #when was this added to the plans
    date = models.DateTimeField(blank=True, null=True)

    #When was the plan executed?
    planExecutedDate = models.DateTimeField(blank=True, null=True)

    #add metadata grab bag
    metaData = json_field.JSONField(blank=True)

    seqKitBarcode = models.CharField(max_length=64, blank=True, null=True)

    #name of the experiment
    expName = models.CharField(max_length=128, blank=True)

    #Pre-Run/Beadfind
    usePreBeadfind = models.BooleanField(default=True)

    #Post-Run/Beadfind
    usePostBeadfind = models.BooleanField(default=True)

    #cycles
    cycles = models.IntegerField(blank=True, null=True)

    #autoName string
    autoName = models.CharField(max_length=512, blank=True, null=True)

    preAnalysis = models.BooleanField(default=True)

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
    irworkflow = models.CharField(max_length=1024, blank=True)

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

    planDisplayedName = models.CharField(max_length=512, blank=True, null=True)

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

    sampleSets = models.ManyToManyField('SampleSet', related_name="plans", null=True, blank=True)

    sampleGrouping = models.ForeignKey("SampleGroupType_CV", blank=True, null=True, default=None)
    applicationGroup = models.ForeignKey(ApplicationGroup, null=True)

    latestEAS = models.OneToOneField('ExperimentAnalysisSettings', null=True, blank=True, related_name='+', on_delete=models.SET_NULL)

    ALLOWED_CATEGORIES = (
        ('', 'Unspecified'),
        ('Onconet', 'OncoNet'),
        ('Oncomine', 'Oncomine'),
        ('Oncomine;ocav2', 'Oncomine and OCAv2'),
        ('barcodes_8', 'default to 8 barcodes'),
        ('onco_solidTumor', 'Solid tumor'),
        ('onco_solidTumor;inheritedDisease', 'Solid tumor and inherited disease'),
        ('onco_solidTumor;onco_heme', 'Solid tumor and hematology'),           
        ('Oncomine;onco_solidTumor', 'Oncomine solid tumor'),
        ('Onconet;onco_solidTumor', 'Onconet solid tumor'),
        ('Oncomine;onco_solidTumor;inheritedDisease', 'Oncomine solid tumor and inherited disease'),       
        ('Oncomine;ocav2;onco_solidTumor', ''),
        ('Oncomine;barcodes_8;onco_solidTumor', 'OCAv3 DNA only or Fusions only'),
        ('Oncomine;barcodes_16;onco_solidTumor', 'OCAv3 DNA and Fusions'),
        ('onco_immune', 'Immunology'),
        ('barcodes_6;onco_heme', 'PGM Oncomine Myeloid DNA only'),
        ('barcodes_24;onco_heme', 'PGM Oncomine Myeloid RNA Fusions'),         
        ('barcodes_12;onco_heme', 'PGM Oncomine Myeloid DNA and Fusions'),
        ('barcodes_12;onco_heme;chef_myeloid_protocol', 'S5 Oncomine Myeloid DNA only'),
        ('barcodes_48;onco_heme;chef_myeloid_protocol', 'S5 Oncomine Myeloid RNA Fusions'),
        ('barcodes_24;onco_heme;chef_myeloid_protocol', 'S5 Oncomine Myeloid DNA and Fusions'),
        ('repro', 'Reproductive'),
        ('inheritedDisease', 'Inherited Disease'),
        ('onco_heme', "Hematology")
    )

    categories = models.CharField(max_length=64, choices=ALLOWED_CATEGORIES, default='', blank=True, null=True)
    libraryReadLength = models.PositiveIntegerField(default=0)

    ALLOWED_TEMPLATING_SIZE = (
        ("", 'Unspecified'),
        ("200", "200"),
        ("400", "400"),
    )
    templatingSize = models.CharField(max_length=64, choices=ALLOWED_TEMPLATING_SIZE, default='', blank=True, null=True)

    # this will be set based on plan creation method, e.g. gui, csv, api, etc.
    origin = models.CharField(max_length=64, blank=True, null=True, default="")

    # this is to use alternate chef script protocal
    samplePrepProtocol = models.CharField(max_length=64, blank=True, null=True, default="")

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
    def get_latest_plan_or_template_by_chipType(cls, chipType=None, isReusable=True, isSystem=True, isSystemDefault=True):
        """
        return the latest plan or template with matching chipType. Input chipType can be None
        """
        plans = PlannedExperiment.objects.filter(isReusable=isReusable, isSystem=isSystem, isSystemDefault=isSystemDefault).order_by("-date")

        if plans:
            for plan in plans:
                planChipType = plan.get_chipType()

                if chipType:
                    if (planChipType == chipType) or (planChipType == chipType[:3]):
                        logger.info("PlanExperiment.get_latest_plan_or_template_by_chipType() match - chipType=%s; planChipType=%s; found plan.pk=%s" % (chipType, planChipType, str(plan.pk)))
                        return plan
                elif (not planChipType and not chipType):
                    logger.info("PlanExperiment.get_latest_plan_or_template_by_chipType() found plan.pk=%s" % (str(plan.pk)))
                    return plan

        if (isReusable and isSystem and isSystemDefault):
            chips = dict((chip.name, chip.instrumentType) for chip in Chip.objects.all())

            chipInstrumentType = "pgm"
            if chipType:
                chipInstrumentType = chips.get(chipType, "")
                if (not chipInstrumentType):
                    chipInstrumentType = chips.get(chipType[:3], "pgm")

                    logger.debug("PlanExperiment.get_latest_plan_or_template_by_chipType() chipType=%s; instrumentType=%s; " % (chipType, chipInstrumentType))
            else:
                logger.debug("PlanExperiment.get_latest_plan_or_template_by_chipType() NO chipType - use chipInstrumentType=%s; " % (chipInstrumentType))

            for template in plans:
                plan_chipInstrumentType = "pgm"

                planChipType = template.get_chipType()
                if planChipType:
                    plan_chipInstrumentType = chips.get(planChipType, "")
                    if not plan_chipInstrumentType:
                        plan_chipInstrumentType = chips.get(chipType[:3], "pgm")

                if plan_chipInstrumentType == chipInstrumentType:
                    logger.debug("EXIT PlanExperiment.get_latest_plan_or_template_by_chipType() return template.id=%d; for chipInstrumentType=%s; " % (template.id, chipInstrumentType))
                    return template

        return None

    @cached_property
    # NOTE: This does not update latestEAS if not set
    def latest_eas(self):
        return self.latestEAS or self.experiment.get_EAS()

    def get_autoAnalyze(self):
        experiment = self.experiment
        if experiment:
            return experiment.autoAnalyze
        else:
            return False

    def get_barcodedSamples(self):
        if self.latest_eas:
            return self.latest_eas.barcodedSamples
        else:
            return ""

    def get_barcodeId(self):
        if self.latest_eas:
            return self.latest_eas.barcodeKitName
        else:
            return ""

    def get_bedfile(self):
        if self.latest_eas:
            return self.latest_eas.targetRegionBedFile
        else:
            return ""

    def get_chipType(self):
        experiment = self.experiment
        if experiment:
            chipType = experiment.chipType
            # older chip names compatibility, e.g 314R, 318B, "318C"
            if chipType not in Chip.objects.values_list('name', flat=True):
                chipType = chipType.replace('"', '')
                if chipType and chipType[0].isdigit():
                    chipType = chipType[:3]
            return chipType
        else:
            return ""

    def get_chipBarcode(self):
        return self.experiment.chipBarcode if self.experiment else ''

    def get_flows(self):
        experiment = self.experiment
        if experiment:
            return experiment.flows
        else:
            return 0

    def get_forward3primeadapter(self):
        if self.latest_eas:
            return self.latest_eas.threePrimeAdapter
        else:
            return ""

    def get_library(self):
        if self.latest_eas:
            return self.latest_eas.reference
        else:
            return ""

    def get_libraryKey(self):
        if self.latest_eas:
            return self.latest_eas.libraryKey
        else:
            return ""

    def get_tfKey(self):
        if self.latest_eas:
            return self.latest_eas.tfKey
        else:
            return ""

    def get_librarykitname(self):
        if self.latest_eas:
            return self.latest_eas.libraryKitName
        else:
            return ""

    def get_notes(self):
        experiment = self.experiment
        if experiment:
            return experiment.notes
        else:
            return ""

    def get_LIMS_meta(self):
        if self.metaData:
            data = self.metaData.get("LIMS", "")
            #convert unicode to str
            return convert(data)
        else:
            return ""

    def get_mixedType_rna_bedfile(self):
        if self.latest_eas:
            return self.latest_eas.mixedTypeRNA_targetRegionBedFile
        else:
            return ""

    def get_mixedType_rna_library(self):
        if self.latest_eas:
            return self.latest_eas.mixedTypeRNA_reference
        else:
            return ""


    def get_mixedType_rna_regionfile(self):
        if self.latest_eas:
            return self.latest_eas.mixedTypeRNA_hotSpotRegionBedFile
        else:
            return ""

    def get_regionfile(self):
        if self.latest_eas:
            return self.latest_eas.hotSpotRegionBedFile
        else:
            return ""

    def get_sseBedFile(self):
        if self.latest_eas:
            return self.latest_eas.sseBedFile
        else:
            return ""

    def get_sample_count(self):
        experiment = self.experiment
        if experiment:
            return experiment.samples.count()
        else:
            return 0


    def get_sample(self):
        experiment = self.experiment
        if self.latest_eas and self.latest_eas.barcodeKitName:
            return ""
        elif experiment.samples.count() > 0:
            sample = experiment.samples.values()[0]
            return sample['name']
        else:
            return ""

    def get_sample_external_id(self):
        experiment = self.experiment
        if self.latest_eas and self.latest_eas.barcodeKitName:
            return ""
        elif experiment.samples.count() > 0:
            sample = experiment.samples.values()[0]
            return sample['externalId']
        else:
            return ""

    def get_sample_description(self):
        experiment = self.experiment
        if self.latest_eas and self.latest_eas.barcodeKitName:
            return ""
        elif experiment.samples.count() > 0:
            sample = experiment.samples.values()[0]
            return sample['description']
        else:
            return ""

    def get_sampleDisplayedName(self):
        experiment = self.experiment
        if self.latest_eas and self.latest_eas.barcodeKitName:
            return ""
        elif experiment.samples.count() > 0:
            sample = experiment.samples.values()[0]
            return sample['displayedName']
        else:
            return ""

    def get_selectedPlugins(self):
        if self.latest_eas:
            return self.latest_eas.selectedPlugins
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
            for icKit in KitInfo.objects.filter(kitType="IonChefPrepKit"):
                if (kitName == icKit.name):
                    return True

        return False

    def is_duplicateReads(self):
        if self.latest_eas:
            return self.latest_eas.isDuplicateReads
        else:
            return False

    def get_base_recalibration_mode(self):
        if self.latest_eas:
            return self.latest_eas.base_recalibration_mode
        else:
            return "no_recal"

    def do_realign(self):
        if self.latest_eas:
            return self.latest_eas.realign
        else:
            return False

    def get_default_nucleotideType(self):
        # return default nucleotideType based on runType and applicationGroup
        nucleotideType = ""
        if self.runType:
            runTypeObjs = RunType.objects.filter(runType=self.runType)
            if runTypeObjs:
                nucleotideType = runTypeObjs[0].nucleotideType.upper()

        if nucleotideType not in ["DNA", "RNA"]:
            if self.applicationGroup:
                nucleotideType = self.applicationGroup.name

        return nucleotideType if nucleotideType in ["DNA", "RNA"] else ""

    def is_same_refInfo_as_defaults_per_sample(self):
        """
        Returns True if all the barcoded samples have the same reference and BED files as the plan's defaults
        """
        if not self.get_barcodeId():
            return True

        barcodedSamples = self.get_barcodedSamples()
        ##logger.debug("plannedExperiment.is_same_refInfo_as_defaults_per_sample() barcodedSamples=%s" %(barcodedSamples))

        planReference = self.get_library()
        planHotSpotRegionBedFile = self.get_regionfile()
        planTargetRegionBedFile = self.get_bedfile()

        mixedRNA_planReference = self.get_mixedType_rna_library()
        mixedRNA_planHotSpotRegionBedFile = self.get_mixedType_rna_regionfile()
        mixedRNA_planTargetRegionBedFile = self.get_mixedType_rna_bedfile()

        for sample, value in barcodedSamples.items():
            if 'barcodeSampleInfo' in value:

                for barcode, sampleInfo in value['barcodeSampleInfo'].items():
                    sampleReference = sampleInfo.get("reference", "")
                    sampleHotSpotRegionBedFile = sampleInfo.get("hotSpotRegionBedFile", "")
                    sampleTargetRegionBedFile = sampleInfo.get("targetRegionBedFile", "")

                    if self.runType == "AMPS_DNA_RNA":
                        sampleNucleotideType = sampleInfo.get("nucleotideType", "")

                        if sampleNucleotideType == "DNA":
                            if planReference != sampleReference or planHotSpotRegionBedFile != sampleHotSpotRegionBedFile or planTargetRegionBedFile != sampleTargetRegionBedFile:
                                #logger.debug("plannedExperiment.is_same_refInfo_as_defaults_per_sample() - AMPS_DNA_RNA - DNA SAMPLE - DIFFERENT!!! return FALSE planReference=%s; sampleReference=%s" %(planReference, sampleReference))
                                return False
                        elif sampleNucleotideType == "RNA":
                            if mixedRNA_planReference != sampleReference or mixedRNA_planHotSpotRegionBedFile != sampleHotSpotRegionBedFile or mixedRNA_planTargetRegionBedFile != sampleTargetRegionBedFile:
                                #logger.debug("plannedExperiment.is_same_refInfo_as_defaults_per_sample() - AMPS_DNA_RNA - RNA SAMPLE - DIFFERENT!!! return FALSE mixedRNA_planReference=%s; sampleReference=%s" %(mixedRNA_planReference, sampleReference))
                                return False
                    else:
                        if planReference != sampleReference or planHotSpotRegionBedFile != sampleHotSpotRegionBedFile or planTargetRegionBedFile != sampleTargetRegionBedFile:
                            #logger.debug("plannedExperiment.is_same_refInfo_as_defaults_per_sample() DIFFERENT!!! return FALSE planReference=%s; sampleReference=%s" %(planReference, sampleReference))
                            return False

        return True


    @staticmethod
    def get_applicationCategoryDisplayedName(categories):
        """ 
        return the category displayed name with no validation
        """
        categoryDisplayedName = ""
        if categories:
            tokens = categories.split(';')
            for token in tokens:
                categories_cv = common_CV.objects.filter(cv_type = "applicationCategory", isVisible = True, value__iexact = token)
                if (categories_cv.count() > 0):
                    category_cv = categories_cv[0]
                    if categoryDisplayedName:
                        categoryDisplayedName += " | "
                    categoryDisplayedName += category_cv.displayedValue
        return categoryDisplayedName


    @staticmethod
    def get_validatedApplicationCategoryDisplayedName(categories, runType):
        """ 
        return the category displayed name if the runType is compatible with the matching category controlled volcabulary
        """
        categoryDisplayedName = ""
        if categories:
            tokens = categories.split(';')
            for token in tokens:
                categories_cv = common_CV.objects.filter(cv_type = "applicationCategory", isVisible = True, value__iexact = token)
                if (categories_cv.count() > 0):
                    category_cv = categories_cv[0]
                    if category_cv.categories:
                        if categoryDisplayedName:
                            categoryDisplayedName += " | "
                        categoryDisplayedName += category_cv.displayedValue if runType in category_cv.categories else ""
        return categoryDisplayedName
    
    def save(self, *args, **kwargs):

        logger.debug("PDD ENTER models.PlannedExperiment.save(self, args, kwargs)")
        logger.debug("PDD kwargs=%s " % (kwargs))

        if not self.planStatus:
            if self.is_ionChef() and not self.isReusable:
                self.planStatus = "pending"
            else:
                self.planStatus = "planned"

        planName = self.planName
        if not self.planDisplayedName:
            self.planDisplayedName = planName

        self.planName = '_'.join(planName.split())

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

                #logger.info('Going to CREATE the 1 UNTOUCHED plan with name=%s' % self.planName)
                super(PlannedExperiment, self).save()

                # update status for any sampleSets that have status = "" or "created"
                self.sampleSets.filter(status__in=["", "created"]).update(status='planned')

            if self.runType == "TAG_SEQUENCING":
                self.categories = "barcodes_8"

            self.origin += "|" + ion.version

    def save_plannedExperiment_association(self, isPlanCreated, **kwargs):
        """
        create/update the associated records for the plan

        1. for template: create or update exp, eas
        2. for plan: create or update exp, eas, samples
        3. for executed plan: update existing exp, eas, samples
        """
        logger.debug("PDD ENTER models.save_plannedExperiment_association() isPlanCreated=%s; self_id=%d; planExecuted=%s; kwargs=%s" % (str(isPlanCreated), self.id, str(self.planExecuted), kwargs))

        # ===================== Experiment ==================================
        exp_kwargs = {
            'runMode': self.runMode,
            'status': self.planStatus
        }
        exp_keys = ['autoAnalyze', 'usePreBeadfind', 'star', 'chipType', 'flows', 'notes', 'sequencekitname', 'platform', 'chipBarcode', 'flowsInOrder']
        for key in exp_keys:
            if key in kwargs:
                exp_kwargs[key] = kwargs[key]

        #there must be one experiment for each plan
        try:
            experiment = self.experiment
            if not isPlanCreated:
                self.update_changed_fields_for_plan_history(exp_kwargs, obj=experiment)

            for key, value in exp_kwargs.items():
                setattr(experiment, key, value)
            logger.debug("PDD models.save_plannedExperiment_association() going to UPDATE experiment id=%d" % experiment.id)

        except Experiment.DoesNotExist:
            # creating new experiment
            exp_kwargs.update({
                'date': timezone.now(),
                'expDir': '',
                #temp expName value below will be replaced in crawler
                'expName': self.planGUID,
                'displayName': self.planShortID,
                'pgmName': '',
                'log': '',
                #db constraint requires a unique value for experiment. temp unique value below will be replaced in crawler
                'unique': self.planGUID,
                'seqKitBarcode': '',
                'sequencekitbarcode': '',
                'reagentBarcode': '',
                'flows': exp_kwargs.get('flows', '0'),
                'cycles': 0,
                'expCompInfo': '',
                'baselineRun': '',
                'flowsInOrder': exp_kwargs.get('flowsInOrder') or '',
                'ftpStatus': '',
                'displayName': '',
                'storageHost': '',
                'chipBarcode': kwargs.get('chipBarcode', '') or ''
            })
            experiment = Experiment(**exp_kwargs)
            logger.debug("PDD models.save_plannedExperiment_association() #2 going to CREATE experiment")

        logger.debug("PDD models.save_plannedExperiment_association() self_id=%d exp_kwargs=..." % (self.id))
        logger.debug(exp_kwargs)

        # update Chef fields, if specified
        chefInfo = kwargs.get('chefInfo')
        if chefInfo:
            if not isPlanCreated:
                self.update_changed_fields_for_plan_history(chefInfo, obj=experiment)
            for key, value in chefInfo.items():
                setattr(experiment, key, value)

        #need this!
        experiment.plan = self
        experiment.save()
        logger.debug("PDD models.save_plannedExperiment_association() AFTER saving experiment_id=%d" % (experiment.id))


        # ===================== ExperimentAnalysisSettings =====================
        eas_kwargs = {'status': self.planStatus}
        eas_keys = [
            'barcodedSamples', 'barcodeKitName', 'targetRegionBedFile', 'hotSpotRegionBedFile', 'libraryKey', 'tfKey', 'threePrimeAdapter',
            'reference', 'selectedPlugins', 'isDuplicateReads', 'base_recalibration_mode', 'realign', 'libraryKitName',
            "mixedTypeRNA_reference", "mixedTypeRNA_targetRegionBedFile", "mixedTypeRNA_hotSpotRegionBedFile",
            "beadfindargs", "analysisargs", "prebasecallerargs", "calibrateargs", "basecallerargs", "alignmentargs", "ionstatsargs",
            "thumbnailbeadfindargs", "thumbnailanalysisargs", "prethumbnailbasecallerargs", "thumbnailcalibrateargs",
            "thumbnailbasecallerargs", "thumbnailalignmentargs", "thumbnailionstatsargs", "custom_args", "sseBedFile"
        ]
        for key in eas_keys:
            if key in kwargs:
                eas_kwargs[key] = kwargs[key]

        eas, eas_created = experiment.get_or_create_EAS(editable=True)
        if not eas_created:
            self.update_changed_fields_for_plan_history(eas_kwargs, obj=eas)

        for key, value in eas_kwargs.items():
            setattr(eas, key, value)
        eas.save()
        logger.debug("PDD models.save_plannedExperiment_association() AFTER saving EAS_id=%d" % (eas.id))

        # update default analysis args
        if not eas.custom_args:
            eas.reset_args_to_default()

        self.latestEAS = eas
        self.save()

        # ===================== Samples ==========================================
        if not self.isReusable:
            #if this is not a template need to create/update single sample or multiple barcoded samples
            samples_kwargs = []
            barcodedSamples = kwargs.get('barcodedSamples', {})

            if barcodedSamples:
                if isinstance(barcodedSamples, str):
                    barcodedSampleDict = json_field.loads(barcodedSamples)
                else:
                    barcodedSampleDict = barcodedSamples
                if not isinstance(barcodedSampleDict, dict):
                    logger.debug("PDD models.save_plannedExperiment_association() barcoded samples is not Dict. Verify the Input")#TS-11396

                for displayedName, sampleDict in barcodedSampleDict.items():
                    if barcodedSampleDict[displayedName].get('barcodeSampleInfo'):
                        externalId = barcodedSampleDict[displayedName]['barcodeSampleInfo'].values()[0].get('externalId', '')
                        description = barcodedSampleDict[displayedName]['barcodeSampleInfo'].values()[0].get('description', '')
                    else:
                        externalId = ""
                        description = ""

                    samples_kwargs.append({
                        'name': '_'.join(displayedName.split()),
                        'displayedName': displayedName,
                        'date': self.date,
                        'status': self.planStatus,
                        'externalId': externalId,
                        'description': description
                    })
                logger.debug("PDD models.save_plannedExperiment_association() barcoded samples kwargs=")
                logger.debug(samples_kwargs)
            else:
                displayedName = kwargs.get('sampleDisplayedName', "")
                name = kwargs.get('sample', "") or displayedName

                if name:
                    samples_kwargs.append({
                        'name': '_'.join(name.split()),
                        'displayedName': displayedName or name,
                        'date': self.date,
                        'status': self.planStatus,
                        'externalId': kwargs.get('sample_external_id', ""),
                        'description': kwargs.get('sample_description', "")
                    })
                    logger.debug("PDD models.save_plannedExperiment_association() samples kwargs=")
                    logger.debug(samples_kwargs)

                    if not isPlanCreated:
                        old_sample = experiment.samples.first() or Sample(name="")
                        self.update_changed_fields_for_plan_history(samples_kwargs[0], obj=old_sample, key_prefix='sample_')

            # add sample(s)
            #The sample's status state transition should be "" or "created" -> "planned" -> "run"
            #If the sample has been sequenced ie if 'status' is 'run' then the status should not be updated.
            allowedSampleStatus = Sample.ALLOWED_STATUS
            for sample_kwargs in samples_kwargs:
                try:
                    sample = Sample.objects.get(name=sample_kwargs['name'], externalId=sample_kwargs['externalId'])
                    for key, value in sample_kwargs.items():
                        if key == 'status':
                            if (sample.status.lower() != "run" and
                                  any(value in sample_status for sample_status in allowedSampleStatus)):
                                setattr(sample, key, value)
                        else:
                            setattr(sample, key, value)
                except Sample.DoesNotExist:
                    status_to_update = sample_kwargs['status']
                    if any(status_to_update in sample_status for sample_status in allowedSampleStatus):
                        sample_kwargs['status'] = status_to_update
                    else:
                        sample_kwargs['status'] = "planned"
                    sample = Sample(**sample_kwargs)

                sample.save()
                sample.experiments.add(experiment)

                logger.debug("PDD models.save_plannedExperiment_association() AFTER saving sample_id=%d" % (sample.id))

            # clean up old samples that may remain on experiment
            # only do this if samples were created/modified above
            if samples_kwargs:
                for sample in experiment.samples.all():
                    found = [s for s in samples_kwargs if sample.name == s['name'] and sample.externalId == s['externalId']]
                    if len(found) == 0:
                        experiment.samples.remove(sample)

                    #delete sample if it is not associated with any experiments
                    if sample.experiments.all().count() == 0 and not (sample.status == "created"):
                        logger.debug("PDD models.save_plannedExperiment_association() going to DELETE sample=%s" % (sample.name))
                        sample.delete()


    def update_plan_qcValues(self, **kwargs):
        qcTypes = QCType.objects.all()
        for qcType in qcTypes:
            qc_threshold = kwargs.get(qcType.qcName, '')
            if qc_threshold:
                try:
                    plannedExpQc = PlannedExperimentQC.objects.get(plannedExperiment=self, qcType=qcType)
                    if plannedExpQc.threshold != qc_threshold:
                        self._temp_changedFields[plannedExpQc.qcType.qcName] = [plannedExpQc.threshold, qc_threshold]

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

    def get_default_cmdline_args_obj(self, **kwargs):
        # retrieve args from AnalysisArgs table
        chipType = kwargs.get('chipType') or self.get_chipType()
        sequenceKitName = kwargs.get('sequenceKitName') or self.get_sequencekitname()
        templateKitName = kwargs.get('templateKitName') or self.templatingKitName
        libraryKitName = kwargs.get('libraryKitName') or self.get_librarykitname()
        samplePrepKitName = kwargs.get('samplePrepKitName') or self.samplePrepKitName

        applicationTypeName = kwargs.get('runType') or self.runType
        if kwargs.get('applicationGroupName'):
            applicationGroupName = kwargs.get('applicationGroupName')
        elif self.applicationGroup:
            applicationGroupName = self.applicationGroup.name
        else:
            applicationGroupName = ""

        args = AnalysisArgs.best_match(chipType, sequenceKitName, templateKitName, libraryKitName, samplePrepKitName, None, applicationTypeName, applicationGroupName)
        return args

    def get_default_cmdline_args(self, **kwargs):
        args = self.get_default_cmdline_args_obj(**kwargs)
        if args:
            args_dict = args.get_args()
        else:
            args_dict = {}

        return args_dict


    def update_changed_fields_for_plan_history(self, fields_dict, obj, key_prefix=''):
        # returns fields in obj that have different values than in compare dict
        # excludes relations and DateTimeField, ignores any dict keys that don't correspond to a field
        if not hasattr(self, "_temp_changedFields"):
            self._temp_changedFields = {}

        for key, value in fields_dict.items():
            if hasattr(obj, key) and key != 'pk':
                field, model, direct, m2m = obj._meta.get_field_by_name(key)
                if direct and field.get_internal_type() not in ['ForeignKey', 'OneToOneField', 'ManyToManyField', 'DateTimeField']:
                    orig = getattr(obj, key)
                    if orig != value:
                        self._temp_changedFields[key_prefix + key] = [orig, value]

    def save_plan_history_log(self):
        try:
            if hasattr(self, "_temp_changedFields") and self._temp_changedFields:
                # some fields need special handling
                if 'selectedPlugins' in self._temp_changedFields:
                    selectedPlugins = self._temp_changedFields.pop('selectedPlugins')
                    old_plugins = selectedPlugins[0].keys() if selectedPlugins[0] else ''
                    new_plugins = selectedPlugins[1].keys() if selectedPlugins[1] else ''
                    if set(old_plugins) != set(new_plugins):
                        self._temp_changedFields['plugins'] = [', '.join(old_plugins), ', '.join(new_plugins)]
    
                    for name in selectedPlugins[1]:
                        old_userInput = selectedPlugins[0][name].get('userInput','') if selectedPlugins[0].get(name,{}) else ''
                        new_userInput = selectedPlugins[1][name].get('userInput','') if selectedPlugins[1][name] else ''
                        if old_userInput != new_userInput:
                            self._temp_changedFields['plugin_'+name] = [old_userInput, new_userInput]

                # record changes to analysis args only if custom
                custom_args =  self.latestEAS.custom_args or self._temp_changedFields.get('custom_args', False)
                if not custom_args:
                    for args_key in ["beadfindargs","analysisargs","prebasecallerargs","calibrateargs","basecallerargs",
                        "alignmentargs","ionstatsargs","thumbnailbeadfindargs","thumbnailanalysisargs","prethumbnailbasecallerargs",
                        "thumbnailcalibrateargs","thumbnailbasecallerargs","thumbnailalignmentargs","thumbnailionstatsargs"]:
                        self._temp_changedFields.pop(args_key, None)

                # add log entry
                EventLog.objects.add_entry(self, json.dumps(self._temp_changedFields), self.username)
        except:
            logger.error('Unable to update Plan history log')
            logger.error(traceback.format_exc())


    class Meta:
        ordering = ['-id']


class PlannedExperimentQC(models.Model):
    plannedExperiment = models.ForeignKey(PlannedExperiment)
    qcType = models.ForeignKey(QCType)
    threshold = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = (('plannedExperiment', 'qcType'), )


class Experiment(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'eas_set' from ExperimentAnalysisSettings
    # ManyToManyField 'samples' from Sample
    # ManyToManyField 'results_set' from Results

    _CSV_METRICS = (('Run Name', 'expName'),
                    ('Sample', 'sample'),
                    ('Run Date', 'date'),
                    #('Project', 'project'),
                    #('Library', 'library'),
                    ('PGM Name', 'pgmName'),
                    ('Run Directory', 'expDir'),
                    ('Notes', 'notes'),
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
                                 r'_(\d{2})_(\d{2})_')
    # raw data lives here absolute path prefix
    expDir = models.CharField(max_length=512)
    expName = models.CharField(max_length=128, db_index=True)
    displayName = models.CharField(max_length=128, default="")
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

    user_ack = models.CharField(max_length=24, choices=ACK_CHOICES, default='U')
    notes = models.CharField(max_length=1024, blank=True, null=True)
    chipBarcode = models.CharField(max_length=64, blank=True)
    seqKitBarcode = models.CharField(max_length=64, blank=True)
    reagentBarcode = models.CharField(max_length=64, blank=True)
    autoAnalyze = models.BooleanField(default=True)
    usePreBeadfind = models.BooleanField(default=True)
    chipType = models.CharField(max_length=32)
    cycles = models.IntegerField()
    flows = models.IntegerField()
    expCompInfo = models.TextField(blank=True)
    baselineRun = models.BooleanField(default=False)
    flowsInOrder = models.CharField(max_length=1200, blank=True)
    star = models.BooleanField(default=False)
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
        ('transferred', 'Transferred'),
        ('inactive', 'Inactive'),
        ('run', 'Run')
    )

    status = models.CharField(max_length=512, blank=True, choices=ALLOWED_STATUS, default='')

    ALLOWED_PLATFORM_TYPES = (
        ('', 'Unspecified'),
        ('PGM', 'PGM'),
        ('PROTON', 'Proton'),
        ('S5', 'S5')
    )

    platform = models.CharField(max_length=128, choices=ALLOWED_PLATFORM_TYPES, blank=True, default='')

    # Ion Chef status tracking.
    # The Chef moves through several processes each of which has a status label
    # optionally a deterministic progress percentage
    # optionally a longer message explaining the status, such as an error message
    chefStatus = models.CharField(max_length=256, blank=True, default='')
    chefProgress = models.FloatField(default=0.0, blank=True)
    chefMessage = models.TextField(blank=True, default='')
    chefLastUpdate = models.DateTimeField(null=True, blank=True)
    chefLogPath = models.CharField(max_length=512, blank=True, null=True)

    chefReagentID = models.CharField(max_length=64, blank=True, default='')
    chefLotNumber = models.CharField(max_length=64, blank=True, default='')
    chefManufactureDate = models.CharField(max_length=64, blank=True, default='')
    chefTipRackBarcode = models.CharField(max_length=64, blank=True, default='')

    chefInstrumentName = models.CharField(max_length=200, blank=True, default='')

    chefKitType = models.CharField(max_length=64, blank=True, default='')
    chefChipType1 = models.CharField(max_length=64, blank=True, default='')
    chefChipType2 = models.CharField(max_length=64, blank=True, default='')
    chefChipExpiration1 = models.CharField(max_length=64, blank=True, default='')
    chefChipExpiration2 = models.CharField(max_length=64, blank=True, default='')

    chefReagentsPart = models.CharField(max_length=64, blank=True, default='')
    chefReagentsLot = models.CharField(max_length=64, blank=True, default='')
    chefReagentsExpiration = models.CharField(max_length=64, blank=True, default='')

    chefSolutionsPart = models.CharField(max_length=64, blank=True, default='')
    chefSolutionsLot = models.CharField(max_length=64, blank=True, default='')
    chefSolutionsExpiration = models.CharField(max_length=64, blank=True, default='')

    chefSamplePos = models.CharField(max_length=64, blank=True, default='')
    chefPackageVer = models.CharField(max_length=64, blank=True, default='')

    chefExtraInfo_1 = models.CharField(max_length=128, blank=True, default='')
    chefExtraInfo_2 = models.CharField(max_length=128, blank=True, default='')
    chefScriptVersion = models.CharField(max_length=64, blank=True, default='')

    def __unicode__(self): return self.expName

    @property
    def has_status(self):
        ftp = self.ftpStatus != "Complete" or self.ftpStatus == ''

    def runtype(self):
        runType = self.log.get("runtype", "")
        return runType if runType else "GENS"

    def pretty_print(self):
        nodate = self.PRETTY_PRINT_RE.sub("", self.expName)
        ret = " ".join(nodate.split('_')).strip()
        if not ret:
            return nodate
        return ret

    def pretty_print_no_space(self):
        return self.pretty_print().replace(" ", "_")

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
            if eas_set.filter(isOneTimeOverride=False):
                eas_set = eas_set.filter(isOneTimeOverride=False)
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
                gc = GlobalConfig.objects.get()
                default_eas_kwargs = {
                              'date': timezone.now(),
                              'experiment': self,
                              'libraryKey': gc.default_library_key,
                              'tfKey': gc.default_test_fragment_key,
                              'isEditable': True,
                              'isOneTimeOverride': False,
                              'status': 'run',
                              'isDuplicateReads': gc.mark_duplicates,
                              'base_recalibration_mode': gc.base_recalibration_mode,
                              'realign': gc.realign
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
            queryset = eas_set.filter(isEditable=True, isOneTimeOverride=False)
            if not queryset:
                queryset = eas_set.filter(isEditable=False, isOneTimeOverride=False)

        if not editable and not reusable:
            queryset = eas_set.filter(isEditable=False, isOneTimeOverride=True)

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

    @cached_property
    def latest_eas(self):
        return self.get_EAS()

    def get_barcodeId(self):
        if self.latest_eas:
            return self.latest_eas.barcodeKitName
        else:
            return ""

    def get_library(self):
        self.latest_eas = self.get_EAS_cached()
        if self.latest_eas:
            return self.latest_eas.reference
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

    @cached_property
    def isProton(self):
        if self.chipType:
            chips = Chip.objects.filter(name=self.chipType)
            if chips:
                chip = chips[0]
                return (chip.instrumentType == "proton")
            else:
                chipPrefix = self.chipType[:3]
                chips = Chip.objects.filter(name=chipPrefix)
                if chips:
                    chip = chips[0]
                    return (chip.instrumentType == "proton")

            #if somehow the chip is not in the chip table but it starts with p...
            if (self.chipType[:1].lower() == 'p'):
                return True
        return False

    @cached_property
    def getPlatform(self):
        if self.platform:
            return self.platform.__str__().lower()
        else:
            if self.chipType:
                chips = Chip.objects.filter(name=self.chipType)
                if chips:
                    chip = chips[0]
                    return chip.instrumentType.__str__().lower()
                else:
                    chipPrefix = self.chipType[:3]
                    chips = Chip.objects.filter(name=chipPrefix)
                    if chips:
                        chip = chips[0]
                        return chip.instrumentType.__str__().lower()
            return ""

    @cached_property
    def isPQ(self):
        return self.chipType.lower().startswith("p2.2.") if self.chipType else False


    def location(self):
        return self._location

    @cached_property
    def _location(self):
        try:
            loc = Rig.objects.get(name=self.pgmName).location
        except Rig.DoesNotExist:
            loc = Location.objects.filter(defaultlocation=True)
            if not loc:
                #if there is not a default, just take the first one
                loc = Location.objects.all().order_by('pk')
            if loc:
                loc = loc[0]
            else:
                logger.critical("No Location objects exist!")
                return False
        return loc

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

        # If this is the initial save, ie, the creation of the object, set the storage_options to globalconfig default
        if self.pk is None:
            self.storage_options = GlobalConfig.get().default_storage_options

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
        #if sample has no plans associated with it and is not a member of a sample set, we can delete the orphaned sample
        #if sample is created via loading instead of plan/run creation, keep the sample around even if it is not assigned to a sample set yet
        if (sample.experiments.count() == 0) and (sample.sampleSets.count() == 0) and not (sample.status == "created"):
            logger.debug("Going to delete orphaned sample.pk=%d; sample.name=%s when deleting plan.name=%s" % (sample.id, sample.displayedName, plan.planDisplayedName))
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

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'results_set' from Results

    #planned status: means the entire plan object has been saved but not yet used
    #run status    : means the plan has been claimed for a run and the corresponding sequencing run has started
    ALLOWED_STATUS = (
        ('', 'Undefined'),
        ('planned', 'Planned'),
        ('inactive', 'Inactive'),
        ('run', 'Run')
    )

    status = models.CharField(max_length=512, blank=True, choices=ALLOWED_STATUS, default='')

    barcodeKitName = models.CharField(max_length=128, blank=True, null=True)

    libraryKey = models.CharField(max_length=64, blank=True)
    tfKey = models.CharField(max_length=64, blank=True)

    libraryKitName = models.CharField(max_length=512, blank=True, null=True)
    libraryKitBarcode = models.CharField(max_length=512, blank=True, null=True)

    threePrimeAdapter = models.CharField("3' adapter", max_length=512, blank=True, null=True)

    #bed file
    #Target Regions BED File: old name is bedfile
    #Hotspot Regions BED File: old name is regionfile
    targetRegionBedFile = models.CharField(max_length=1024, blank=True, null=True)
    hotSpotRegionBedFile = models.CharField(max_length=1024, blank=True, null=True)
    sseBedFile = models.CharField(max_length=1024, blank=True, null=True, default='')

    reference = models.CharField(max_length=512, blank=True, null=True)

    #For plans (e.g., OCP) that have both DNA and RNA samples on the same chip,
    #the plan defaults for RNA sample's reference & BED files will be persisted in mixedTypeRNA_*
    #For RNA-only plans, the reference & BED file plan defaults will continue to be persisted in
    #reference, targetRegionBedFile and hotSpotRegionBedFile
    mixedTypeRNA_reference = models.CharField(max_length=512, blank=True, null=True)

    mixedTypeRNA_targetRegionBedFile = models.CharField(max_length=1024, blank=True, null=True)
    mixedTypeRNA_hotSpotRegionBedFile = models.CharField(max_length=1024, blank=True, null=True)

    barcodedSamples = json_field.JSONField(blank=True, null=True)
    selectedPlugins = json_field.JSONField(blank=True, null=True)

    date = models.DateTimeField(blank=True, null=True)
    isEditable = models.BooleanField(default=False)

    #for reanalysis, user can enter parameters that are intended for just this re-analysis attempt
    isOneTimeOverride = models.BooleanField(default=False)

    # foreign key to the experiment
    experiment = models.ForeignKey(Experiment, related_name='eas_set', blank=True, null=True)

    isDuplicateReads = models.BooleanField(default=False)

    ALLOWED_RECALIBRATION_MODES = (
         ('standard_recal', 'Default Calibration'),
         ('no_recal', 'No Calibration'),
         ('panel_recal', 'Enable Calibration Standard'),
         ('blind_recal', 'Blind Calibration')
     )

    base_recalibration_mode = models.CharField(max_length=64, blank=False, null=False, choices=ALLOWED_RECALIBRATION_MODES, default='standard_recal')

    realign = models.BooleanField(default=False)

    custom_args = models.BooleanField(default=False)

    # Beadfind args
    beadfindargs = models.CharField(max_length=5000, blank=True, verbose_name="Beadfind args")
    thumbnailbeadfindargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Beadfind args")
    # Analysis args
    analysisargs = models.CharField(max_length=5000, blank=True, verbose_name="Analysis args")
    thumbnailanalysisargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Analysis args")
    # PreBasecaller args
    prebasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Pre Basecaller args, used for recalibration")
    prethumbnailbasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Pre Basecaller args, used for recalibration")
    # Calibration args
    calibrateargs = models.CharField(max_length=5000, blank=True, verbose_name="Calibration args, used for recalibration")
    thumbnailcalibrateargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Calibration args, used for recalibration")
    # Basecaller args
    basecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Basecaller args")
    thumbnailbasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Basecaller args")
    # Alignment args
    alignmentargs = models.CharField(max_length=5000, blank=True, verbose_name="Alignment args")
    thumbnailalignmentargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Alignment args")
    # Ionstats args
    ionstatsargs = models.CharField(max_length=5000, blank=True, verbose_name="Ionstats args")
    thumbnailionstatsargs = models.CharField(max_length=5000, blank=True, verbose_name="Thumbnail Ionstats args")


    def get_base_recalibration_mode_choices(self):
        choices = (("standard_recal", "Default Calibration"),
                   ('panel_recal', 'Enable Calibration Standard'),
                   ('blind_recal', 'Blind Calibration'),
                   ("no_recal", "No Calibration"))
        return choices


    def get_base_recalibration_mode_for_display(self):
        if self.base_recalibration_mode == "no_recal":
            return "No Calibration"
        elif self.base_recalibration_mode == "standard_recal":
            return "Default Calibration"
        elif self.base_recalibration_mode == "panel_recal":
            return "Enable Calibration Standard"
        elif self.base_recalibration_mode == "blind_recal":
            return "Blind Calibration"
        else:
            return "Unsupported calibration mode"

    @cached_property
    def barcoded_samples_reference_names(self):
        """
        Returns a sorted, unique list of references the barcoded samples are using
        """
        ref_list = []
        if self.barcodedSamples:
            for sample, value in self.barcodedSamples.items():
                if 'barcodeSampleInfo' in value:
                    for barcode, sampleInfo in value['barcodeSampleInfo'].items():
                        sampleReference = sampleInfo.get("reference", "")
                        if sampleReference and sampleReference.lower() != "none":
                            ref_list.append(sampleReference)

            ref_list = sorted(list(set(ref_list)))
            if self.reference and self.reference not in ref_list:
                ref_list.insert(0, self.reference)

        return ref_list

    def have_args(self, thumbnail=False, include_calibration=False, include_alignment=False):
        # check if we have required cmdline args
        if thumbnail:
            check_args = ["thumbnailbeadfindargs", "thumbnailanalysisargs", "thumbnailbasecallerargs"]
            if include_calibration:
                check_args += ["prethumbnailbasecallerargs", "thumbnailcalibrateargs"]
            if include_alignment:
                check_args += ["thumbnailalignmentargs"]
        else:
            check_args = ["beadfindargs", "analysisargs", "basecallerargs"]
            if include_calibration:
                check_args += ["prebasecallerargs", "calibrateargs"]
            if include_alignment:
                check_args += ["alignmentargs"]

        have_args = all(bool(getattr(self, key)) for key in check_args)
        return have_args

    def reset_args_to_default(self):
        # reset all command line args to default values
        plan = self.experiment.plan
        if plan:
            args = plan.get_default_cmdline_args()
        else:
            args = AnalysisArgs.best_match(self.experiment.chipType).get_args()

        for key, value in args.items():
            setattr(self, key, value)
        self.save()

    def get_cmdline_args(self):
        args = {
            'beadfindargs': self.beadfindargs,
            'analysisargs': self.analysisargs,
            'basecallerargs': self.basecallerargs,
            'prebasecallerargs': self.prebasecallerargs,
            'calibrateargs': self.calibrateargs,
            'alignmentargs': self.alignmentargs,
            'ionstatsargs': self.ionstatsargs,
            'thumbnailbeadfindargs': self.thumbnailbeadfindargs,
            'thumbnailanalysisargs': self.thumbnailanalysisargs,
            'thumbnailbasecallerargs': self.thumbnailbasecallerargs,
            'prethumbnailbasecallerargs': self.prethumbnailbasecallerargs,
            'thumbnailcalibrateargs': self.thumbnailcalibrateargs,
            'thumbnailalignmentargs': self.thumbnailalignmentargs,
            'thumbnailionstatsargs': self.thumbnailionstatsargs
        }
        return args

    def __unicode__(self):
        return "%s/%d" % (self.experiment, self.pk)

    class Meta:
        verbose_name_plural = "Experiment Analysis Settings"



class SampleGroupType_CVManager(models.Manager):

    def get_by_natural_key(self, uid):
        return self.get(uid=uid)


class SampleGroupType_CV(models.Model):

    '''
    To support sample and sample set creation regardless a TS instance is IR-enabled or not,
    we need to define the necessary terms in TS and flag a term that has an IR-eqivalence.
    '''

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'sampleAnnotation_set' from SampleAnnotation_CV
    # ForeignKey 'sampleSets' from SampleSet

    displayedName = models.CharField(max_length=127, blank=False, null=False, unique=True)
    description = models.CharField(max_length=1024, blank=True, null=True)

    isIRCompatible = models.BooleanField(default=False)
    iRAnnotationType = models.CharField(max_length=127, blank=True, null=True)
    iRValue = models.CharField(max_length=127, blank=True, null=True)
    isActive = models.BooleanField(default=True)

    #keep uid, displayedName value might change per requirement
    uid = models.CharField(max_length=32, unique=True, blank=False)

    objects = SampleGroupType_CVManager()

    def __unicode__(self):
        return u'%s' % (self.displayedName)

    def natural_key(self):
        return (self.uid,)  # must return a tuple


class SampleAnnotation_CVManager(models.Manager):

    def get_by_natural_key(self, uid):
        return self.get(uid=uid)


class SampleAnnotation_CV(models.Model):
    value = models.CharField(max_length=127, blank=True, null=False)

    ALLOWED_TYPES = (
        ('gender', 'Gender'),
        ('relationship', 'Relationship'),
        ('relationshipRole', 'RelationshipRole'),
        ('relationshipGroup', 'RelationshipGroup'),
        ('cancerType', "CancerType"),
        ('controlType', 'ControlType')
    )

    annotationType = models.CharField(max_length=127, blank=False, null=False, choices=ALLOWED_TYPES)

    isIRCompatible = models.BooleanField(default=False)
    iRAnnotationType = models.CharField(max_length=127, blank=True, null=True)
    iRValue = models.CharField(max_length=127, blank=True, null=True)
    isActive = models.BooleanField(default=True)

    #optional n-to-n between sampleMetaData and SampleGroupType_CV (some meta data will not belong to a SampleGroupType_CV (e.g., gender)
    sampleGroupType_CV = models.ForeignKey(SampleGroupType_CV, related_name="sampleAnnotation_set", blank=True, null=True)

    uid = models.CharField(max_length=32, unique=True, blank=False)

    objects = SampleAnnotation_CVManager()

    def __unicode__(self):
        return u'%s' % self.value

    def natural_key(self):
        return (self.uid,)  # must return a tuple



class SampleSet(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'plans' from PlannedExperiment
    # ForeignKey 'samples' from SampleSetItem

    displayedName = models.CharField(max_length=127, blank=False, null=False, unique=True)
    description = models.CharField(max_length=1024, blank=True, null=True)

    creator = models.ForeignKey(User, related_name="created_sampleSet")
    # This will be set to the time a new record is created
    creationDate = models.DateTimeField(auto_now_add=True)

    lastModifiedUser = models.ForeignKey(User, related_name="lastModified_sampleSet")

    # This will be set to the current time every time the model is updated
    lastModifiedDate = models.DateTimeField(auto_now=True)

    #many sampleSets can have the same SampleGroupType_CV but a sampleSet can only have 1 SampleGroupType_CV
    SampleGroupType_CV = models.ForeignKey(SampleGroupType_CV, related_name="sampleSets", null=True)

    #created status       : means the sample set is created for sample-driven planning. it may or may not have plans associated with it.
    #planned status      : means the sample set has at least one plan created for it but is not yet used
    #run status          : means the sample set has at least one plan that with "planExecuted" set to True
    ALLOWED_SAMPLESET_STATUS = (
        ('', 'Unspecified'),
        ('created', 'Created'),
        ('planned', 'Planned'),
        ('libPrep_pending', "Pending for Library Prep"),
        ('libPrep_reserved', "Reserved for Library Prep"),
        ('libPrep_done', "Done for Library Prep"),
        ('voided', "Aborted during Library Prep"),
        ('run', 'Run')  #Status run is now for backward compatibility only. Since a sample set can be sequenced multiple time, status run is not totally correct
    )

    status = models.CharField(max_length=512, blank=True, choices=ALLOWED_SAMPLESET_STATUS, default='')

    ALLOWED_LIBRARY_PREP_INSTRUMENTS = (
        ('', 'Unspecified'),
        ('chef', 'Ion Chef'),
    )
    libraryPrepInstrument = models.CharField(max_length=64, blank=True, choices=ALLOWED_LIBRARY_PREP_INSTRUMENTS, default='')

    pcrPlateSerialNum = models.CharField(max_length=64, default='', blank=True, null=True)

    ALLOWED_LIBRARY_PREP_TYPES = (
        ('', 'Unspecified'),
        ('amps_on_chef_v1', 'AmpliSeq on Chef'),
    )
    libraryPrepType = models.CharField(max_length=64, blank=True, choices=ALLOWED_LIBRARY_PREP_TYPES, default='')
    libraryPrepPlateType = models.CharField(max_length=64, blank=True, default='')
    combinedLibraryTubeLabel = models.CharField(max_length=64, blank=True, default='')

    libraryPrepInstrumentData = models.ForeignKey("SamplePrepData", related_name="libraryPrepData_sampleSet", null=True)
    libraryPrepKitName = models.CharField(max_length=512, blank=True, null=True)

    def __unicode__(self):
        return u'%s' % (self.displayedName)


class SamplePrepData(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'libraryPrepData_sampleSet' from SampleSet

    instrumentName = models.CharField(max_length=200, blank=True, default='')
    instrumentStatus = models.CharField(max_length=256, blank=True, default='')

    lastUpdate = models.DateTimeField(null=True, blank=True)

    ALLOWED_SAMPLE_PREP_DATA_TYPES = (
        ('', 'Unspecified'),
        ('lib_prep', 'Library Prep'),
        ('template_prep', 'Template Prep'),
    )
    samplePrepDataType = models.CharField(max_length=64, blank=True, choices=ALLOWED_SAMPLE_PREP_DATA_TYPES, default='')

    progress = models.FloatField(default=0.0, blank=True)
    message = models.TextField(blank=True, default='')
    logPath = models.CharField(max_length=512, blank=True, null=True)

    tipRackBarcode = models.CharField(max_length=64, blank=True, default='')
    kitType = models.CharField(max_length=64, blank=True, default='')

    reagentsPart = models.CharField(max_length=64, blank=True, default='')
    reagentsLot = models.CharField(max_length=64, blank=True, default='')
    reagentsExpiration = models.CharField(max_length=64, blank=True, default='')

    solutionsPart = models.CharField(max_length=64, blank=True, default='')
    solutionsLot = models.CharField(max_length=64, blank=True, default='')
    solutionsExpiration = models.CharField(max_length=64, blank=True, default='')

    packageVer = models.CharField(max_length=64, blank=True, default='')
    scriptVersion = models.CharField(max_length=64, blank=True, default='')


    def __unicode__(self):
        return u'%s/%d' % (self.instrumentName, self.pk)

    def __iter__(self):
        for field_name in self._meta.get_all_field_names():
            try:
                value = getattr(self, field_name)
            except:
                value = None
            yield (field_name, value)


class SampleSetItem(models.Model):
    #reference Sample but Sample class is defined after SampleSetItem!!!

    #a sample can be in many sampleSets but a sample can only associate with one sampleSetItem
    sample = models.ForeignKey("Sample", related_name="sampleSets", blank=False, null=False)
    dnabarcode = models.ForeignKey("dnaBarcode", blank=True, null=True)

    #a sampleSet can have many sampleSetItem but a sampleSetItem can only associate with one sampleSet
    sampleSet = models.ForeignKey(SampleSet, related_name="samples", blank=False, null=False)

    gender = models.CharField(max_length=127, blank=True, null=True)
    relationshipRole = models.CharField(max_length=127, blank=True, null=True)
    relationshipGroup = models.IntegerField()
    description = models.CharField(max_length=1024, blank=True, null=True, default='')
    controlType = models.CharField(max_length=127, blank=True, null=True, default='')

    creator = models.ForeignKey(User, related_name="created_sampleSetItem")
    # This will be set to the time a new record is created
    creationDate = models.DateTimeField(auto_now_add=True)

    lastModifiedUser = models.ForeignKey(User, related_name="lastModified_sampleSetItem")
    # This will be set to the current time every time the model is updated
    lastModifiedDate = models.DateTimeField(auto_now=True)

    cancerType = models.CharField(max_length=127, blank=True, null=True)
    cellularityPct = models.IntegerField(blank=True, null=True)

    ALLOWED_NUCLEOTIDE_TYPES = (
        ('', 'Unspecified'),
        ('dna', 'DNA'),
        ('rna', 'RNA')
    )

    nucleotideType = models.CharField(max_length=64, choices=ALLOWED_NUCLEOTIDE_TYPES, default='', blank=True)

    pcrPlateColumn = models.CharField(max_length=10, default='', blank=True, null=True)
    pcrPlateRow = models.CharField(max_length=10, default='', blank=True, null=True)

    biopsyDays = models.IntegerField(blank=True, null=True, default=0)
    coupleId = models.CharField(max_length=127, blank=True, null=True, default="")
    embryoId = models.CharField(max_length=127, blank=True, null=True, default="")

    ALLOWED_AMPLISEQ_PCR_PLATE_ROWS_V1 = (
        ('A', 'A'),
        ('B', 'B'),
        ('C', 'C'),
        ('D', 'D'),
        ('E', 'E'),
        ('F', 'F'),
        ('G', 'G'),
        ('H', 'H')
    )

    ALLOWED_AMPLISEQ_PCR_PLATE_COLUMNS_V1 = (
        ('1', '1'),
    )

    @staticmethod
    def get_ampliseq_plate_v1_row_choices():
        return SampleSetItem.ALLOWED_AMPLISEQ_PCR_PLATE_ROWS_V1

    @staticmethod
    def get_ampliseq_plate_v1_column_choices():
        return SampleSetItem.ALLOWED_AMPLISEQ_PCR_PLATE_COLUMNS_V1

    @staticmethod
    def get_nucleotideType_choices():
        return SampleSetItem.ALLOWED_NUCLEOTIDE_TYPES


    def __unicode__(self):
        return u'%s/%s/%d' % (self.sampleSet, self.sample, self.relationshipGroup)


class SampleAttributeDataType(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'sampleAttributes' from SampleAttribute

    ALLOWED_TYPES = (
        ('Text', 'Text'),
        ('Number', 'Number')
    )

    dataType = models.CharField(max_length=64, blank=False, null=False, unique=True, choices=ALLOWED_TYPES)
    description = models.CharField(max_length=1024, blank=True, null=True)
    isActive = models.BooleanField(default=True)

    def __unicode__(self):
        return self.dataType


class SampleAttribute(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'samples' from SampleAttributeValue

    displayedName = models.CharField(max_length=127, blank=False, null=False, unique=True)
    description = models.CharField(max_length=1024, blank=True, null=True)

    isMandatory = models.BooleanField(default=False)
    isActive = models.BooleanField(default=True)

    creator = models.ForeignKey(User, related_name="created_sampleAttribute")

    # This will be set to the time a new record is created
    creationDate = models.DateTimeField(auto_now_add=True)

    lastModifiedUser = models.ForeignKey(User, related_name="lastModified_sampleAttribute")

    # This will be set to the current time every time the model is updated
    lastModifiedDate = models.DateTimeField(auto_now=True)

    dataType = models.ForeignKey(SampleAttributeDataType, related_name="sampleAttributes")

    def __unicode__(self):
        return self.displayedName


class SampleAttributeValue(models.Model):
    value = models.CharField(max_length=1024, blank=True, null=True)

    creator = models.ForeignKey(User, related_name="created_sampleAttributeValue")

    # This will be set to the time a new record is created
    creationDate = models.DateTimeField(auto_now_add=True)

    lastModifiedUser = models.ForeignKey(User, related_name="lastModified_sampleAttributeValue")

    # This will be set to the current time every time the model is updated
    lastModifiedDate = models.DateTimeField(auto_now=True)

    sample = models.ForeignKey("Sample", related_name="sampleAttributeValues")
    sampleAttribute = models.ForeignKey(SampleAttribute, related_name="samples")

    def __unicode__(self):
        return u'%s' % (self.value)


class Sample(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'sampleSets' from SampleSetItem
    # ForeignKey 'sampleAttributeValues' from SampleAttributeValue

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
    externalId = models.CharField(max_length=127, blank=True, null=True, default='')
    description = models.CharField(max_length=1024, blank=True, null=True)
    date = models.DateTimeField(auto_now_add=True, blank=True, null=True)

    experiments = models.ManyToManyField(Experiment, related_name='samples', null=True)

    def __unicode__(self):
        return self.name

    class Meta:
        unique_together = (('name', 'externalId'), )


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

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'tfmetrics_set' from TFMetrics
    # ForeignKey 'analysismetrics_set' from AnalysisMetrics
    # ForeignKey 'libmetrics_set' from LibMetrics
    # ForeignKey 'qualitymetrics_set' from QualityMetrics
    # ForeignKey 'pluginresult_set' from PluginResult

    _CSV_METRICS = (("Report", "resultsName"),
                    ("Status", 'status'),
                    ("Flows", 'processedflows'),
                    ("Library", 'reference')
                    #("Plugin Data", 'pluginStore')
                    )
    _ALIASES = {
        "report": "resultsName",
        #"experiment":"experiment",
        "date": "timeStamp",
        "status": "status",
        "Flows": "processedflows",
        "q17mean": "best_metrics.Q17Mean",
        "q17mode": "best_metrics.Q17Mode",
        "systemsnr": "best_metrics.SysSNR",
        "q17reads": "best_metrics.Q17ReadCount",
        "keypass": "best_metrics.keypass",
        "cf": "best_metrics.CF"
        }
    TABLE_FIELDS = ("Report", "Status", "Flows",
                    "Lib Key Signal",
                     "Q20 Bases", "100 bp AQ20 Reads", "AQ20 Bases")
    PRETTY_FIELDS = TABLE_FIELDS
    experiment = models.ForeignKey(Experiment, related_name='results_set')
    representative = models.BooleanField(default=False)
    resultsName = models.CharField(max_length=512)
    timeStamp = models.DateTimeField(auto_now_add=True, db_index=True)
    reportLink = models.CharField(max_length=512)  #absolute paths
    status = models.CharField(max_length=64)
    log = models.TextField(blank=True)
    analysisVersion = models.CharField(max_length=256)
    processedCycles = models.IntegerField()
    processedflows = models.IntegerField()
    framesProcessed = models.IntegerField()
    timeToComplete = models.CharField(max_length=64)
    reportstorage = models.ForeignKey("ReportStorage", related_name="storage", blank=True, null=True)
    runid = models.CharField(max_length=10, blank=True)

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

    # link for re-Analysis starting from a parent report
    parentResult = models.ForeignKey('self', null=True, blank=True, related_name='childResults_set', on_delete=models.SET_NULL)

    def save(self, *args, **kwargs):

        super(Results, self).save(*args, **kwargs)

        # Test for related DMFileStat objects
        if DMFileStat.objects.filter(result=self).count() == 0:
            # Create data management file tracking objects
            try:
                dmfilesets = DMFileSet.objects.filter(version=iondb.settings.RELVERSION)
                if not dmfilesets:
                    raise Exception("No DMFileSet objects for version: '%s'" % iondb.settings.RELVERSION)
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
                                'result': self,
                                'dmfileset': dmfileset,
                            }
                            dmfilestat = DMFileStat(**kwargs)
                        dmfilestat.save()
                    except:
                        logger.exception(traceback.format_exc())
                    else:
                        #EventLog.objects.add_entry(self,"Created DMFileStat (%s)" % dmfileset.type)
                        pass

    def get_filestat(self, typeStr):
        return self.dmfilestat_set.filter(dmfileset__type=typeStr).first()

    @cached_property
    def isProton(self):
        if self.experiment:
            return self.experiment.isProton

        return False

    @cached_property
    def getPlatform(self):
        if self.experiment:
            return self.experiment.getPlatform
        return None


    @cached_property
    def isThumbnail(self):
        return bool(self.metaData and self.metaData.get("thumb", False))

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
            pluginDict[p.plugin.name] = p.state()
        return pluginDict

    @cached_property
    def planShortID(self):
        expLog = self.experiment.log
        try:
            plan = expLog["planned_run_short_id"]
        except KeyError:
            plan = expLog.get("pending_run_short_id", "")
        return plan

    def projectNames(self):
        return self._projectNames

    @cached_property
    def _projectNames(self):
        names = [p.name for p in self.projects.all().order_by('-modified')]
        return ','.join(names)

    def bamLink(self):
        return self._bamLink

    @cached_property
    def _bamLink(self):
        """a method to used by the API to provide a link to the bam file"""
        if self._location:
            bamFile = self.experiment.expName + "_" + self.resultsName + ".bam"
            webPath = self.web_path(self._location)
            if not webPath:
                logger.warning("Bam link, webpath missing for " + bamFile)
                return False
            return os.path.join(webPath, 'download_links', bamFile)
        else:
            logger.warning("Bam link, Report Storage: %s, Location: %s", self.reportstorage, self._location)
            return False

    def reportWebLink(self):
        return self._reportWebLink

    @cached_property
    def _reportWebLink(self):
        """a method to used get the web url with no fuss"""
        if self._location:
            webPath = self.web_path(self._location)
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

    @cached_property
    def _location(self):
        return self.experiment._location

    def server_and_location(self):
        return self._location

    def _findReportStorage(self):
        return self._findReportStorageCached

    @cached_property
    def _findReportStorageCached(self):
        """
        Tries to determine the correct ReportStorage object by testing for
        a valid filesystem path using the reportLink path with the ReportStorage
        dirPath value.

        Algorithm for determining path from report link:
        strip off the first directory from report link and prepend the dirPath
        """
        logger.warning("Report %s is looking for it's storage." % self)

        # Pre 3.0 versions had php script in reportLink.  Post 3.0 have just directory.
        if "Default_Report.php" in self.reportLink:
            report_directory = os.path.basename(os.path.dirname(self.reportLink))
        else:
            report_directory = os.path.basename(os.path.abspath(self.reportLink))
        location_name = self.experiment._location.name
        storages = ReportStorage.objects.all()
        for storage in storages:
            if os.path.exists(os.path.join(storage.dirPath, location_name, report_directory)):
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
        return self._report_exist

    @cached_property
    def _report_exist(self):
        """check to see if a report exists inside of the report path"""
        fs_path = self.get_report_path()
        #TODO: is this operation slowing down page loading?  on thousands of reports?
        return fs_path and os.path.exists(fs_path)

    # Has side-effects - do not cache (self-caching anyway)
    def get_report_storage(self):
        """Returns reportstorage object"""
        if self.reportstorage == None:
            storage = self._findReportStorage()
            if storage is not None:
                self.reportstorage = storage
                self.save()
        return self.reportstorage

    def get_report_path(self):
        return self._report_path

    @cached_property
    def _report_path(self):
        """Returns filesystem path to report file"""
        self.get_report_storage()
        # Pre 3.0 versions had php script in reportLink.  Post 3.0 have just directory.
        if self.reportLink.endswith('.php') or self.reportLink.endswith('.html'):
            report_directory = os.path.basename(os.path.dirname(self.reportLink))
        else:
            report_directory = os.path.basename(os.path.abspath(self.reportLink))
        return os.path.join(self.reportstorage.dirPath, self.experiment._location.name, report_directory)

    def get_report_dir(self):
        return self._get_report_dir

    @cached_property
    def _get_report_dir(self):
        """Returns filesystem path to results directory"""
        output_dmfilestat = self.get_filestat(dmactions_types.OUT)
        # this allows importing reports to create PDF file from archived location
        if output_dmfilestat.action_state == 'IG':
            archivepath = output_dmfilestat.archivepath
            if archivepath and os.path.exists(archivepath):
                return archivepath

        fs_path = self.get_report_path()
        return fs_path

    def is_archived(self):
        return self._is_archived

    @cached_property
    def _is_archived(self):
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

    def updateMetaData(self, status, info, size, comment, logger=None):
        retLog = logger
        self.reportStatus = status
        self.save()

        if retLog:
            retLog.info(self.resultsName+": Status: "+status+" | Info: "+info + " | Comments: %s" % comment)
            # This string is parsed when calculating disk space saved:
            if size > 0: retLog.info(self.resultsName+": Size saved:* %7.2f KB" % (float(size/1024)))

        self.metaData["Status"] = status
        self.metaData["Date"] = "%s" % timezone.now()
        self.metaData["Info"] = info
        self.metaData["Comment"] = comment

        # Try to read the Log entry, if it does not exist, create it
        if len(self.metaData.get("Log", [])) == 0:
            self.metaData["Log"] = []
        self.metaData["Log"].append({"Status": self.metaData.get("Status"), "Date": self.metaData.get("Date"), "Info": self.metaData.get("Info"), "Comment": comment})
        self.save()

    @cached_property
    def best_metrics(self):
        try:
            ret = self.tfmetrics_set.all().order_by('-Q17Mean')[0]
        except IndexError:
            ret = None
        return ret

    @cached_property
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
    def get_result_and_tf_metrics(cls, obj, metrics):
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
                    ret.append(json.dumps({i.plugin.name: i.store}))
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
        table = [cls.get_keys(Experiment._CSV_METRICS)
               + cls.get_keys(cls._CSV_METRICS)
               + cls.get_keys(TFMetrics._CSV_METRICS)
               + cls.get_keys(LibMetrics._CSV_METRICS)
               + cls.get_keys(AnalysisMetrics._CSV_METRICS)
               + cls.get_keys(PluginResult._CSV_METRICS)
               ]
        for obj in qset:
            # We get one row per tf fragment per report per run
            run_rows = cls.get_result_and_tf_metrics(obj, cls.get_values(TFMetrics._CSV_METRICS))
            # If we have rows for this run, the first row gets all columns filled out
            if len(run_rows) > 0:
                run_rows[0] = cls.get_exp_metrics(obj, cls.get_values(Experiment._CSV_METRICS)) + run_rows[0]
                run_rows[0] += cls.get_lib_metrics(obj, cls.get_values(LibMetrics._CSV_METRICS))
                run_rows[0] += cls.get_analysis_metrics(obj, cls.get_values(AnalysisMetrics._CSV_METRICS))
                run_rows[0] += cls.get_plugin_metrics(obj, cls.get_values(PluginResult._CSV_METRICS))
            # If we have more than one row (multiple tfs) we need to pad the row on the left
            if len(run_rows) > 1:
                for i in range(1, len(run_rows)):
                    for _ in cls.get_keys(Experiment._CSV_METRICS):
                        run_rows[i].insert(0, "")
            table.extend(run_rows)
        return table

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

    if Experiment.objects.filter(id=instance.experiment_id).exists():
        experiment = instance.experiment
        results = experiment.results_set.exclude(id=instance.pk).order_by('-timeStamp')[:1]
        if results:
            experiment.resultDate = results[0].timeStamp
        if experiment.repResult is None or experiment.repResult.pk == instance.pk:
            experiment.pinnedRepResult = False
            experiment.repResult = results[0] if results else None
        experiment.save()


@receiver(m2m_changed, sender=Results.projects.through, dispatch_uid="projects_changed")
def on_projects_added_removed(sender, instance, action, reverse, pk_set, **kwargs):
    # update project's modified date when results added or removed
    if action in ['post_add', 'post_remove']:
        if reverse:
            instance.save() # instance is a Project
        else:
            for project in Project.objects.filter(pk__in=pk_set):
                project.save()


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
        "tfname": "name",
        "q17mean": "Q17Mean",
        "systemsnr": "SysSNR",
        "50q17reads": "Q17ReadCount",
        "keypassreads": "keypass",
        "tfkeypeakcounts": 'aveKeyCount'
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
    defaultlocation = models.BooleanField("Set as the Default Location", default=False, help_text="Only one location can be the default")

    def __unicode__(self):
        return u'%s' % self.name

    @classmethod
    def getdefault(cls):
        ''' Return default location. Raises exception if no default is specified and there is more than one Location object'''
        loc = cls.objects.filter(defaultlocation=True)
        if loc:
            loc = loc[0]
        else:
            try:
                loc = cls.objects.get()
            except Location.MultipleObjectsReturned:
                raise Location.MultipleObjectsReturned("Multiple Location objects found! Please specify default Location.")
            except Location.DoesNotExist:
                raise Location.DoesNotExist("No Location object exist! Please specify default Location.")

        return loc


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
    ftpserver = models.CharField(max_length=128, default="192.168.201.1")
    ftpusername = models.CharField(max_length=64, default="ionguest")
    ftppassword = models.CharField(max_length=64, default="ionguest")
    ftprootdir = models.CharField(max_length=64, default="results")
    updatehome = models.CharField(max_length=256, default="192.168.201.1")
    updateflag = models.BooleanField(default=False)
    serial = models.CharField(max_length=24, blank=True, null=True)

    state = models.CharField(max_length=512, blank=True)
    version = json_field.JSONField(blank=True)
    alarms = json_field.JSONField(blank=True)
    last_init_date = models.CharField(max_length=512, blank=True)
    last_clean_date = models.CharField(max_length=512, blank=True)
    last_experiment = models.CharField(max_length=512, blank=True)

    host_address = models.CharField(blank=True, max_length=1024)
    type = models.CharField(blank=True, max_length=1024)

    updateCommand = json_field.JSONField(blank=True)

    def __unicode__(self): return self.name


class FileServer(models.Model):
    name = models.CharField(max_length=200)
    comments = models.TextField(blank=True)
    #TODO require this field's contents to be terminated with trailing delimiter
    filesPrefix = models.CharField(max_length=200)
    location = models.ForeignKey(Location)
    percentfull = models.FloatField(default=0.0, blank=True, null=True)

    def __unicode__(self): return self.name


class ReportStorage(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'storage' from Results

    name = models.CharField(max_length=200)
    #path to webserver as http://localhost/results/
    webServerPath = models.CharField(max_length=200)
    dirPath = models.CharField(max_length=200)
    default = models.BooleanField(default=False)

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


class Cruncher(models.Model):
    STATES = (
        ('G', 'good'),
        ('E', 'error'),     # connection test failed
        ('W', 'warning'),   # configuration test failed
        ('U', 'unknown')
    )

    name = models.CharField(max_length=200)
    location = models.ForeignKey(Location)
    comments = models.TextField(blank=True)
    state = models.CharField(max_length=8, choices=STATES, default='U')
    date = models.DateTimeField(auto_now=True, null=True)
    info = json_field.JSONField(blank=True, null=True)

    def __unicode__(self): return self.name

    def last_log_date(self):
        try:
            log_date = EventLog.objects.for_model(Cruncher).filter(object_pk=self.pk).latest('created').created
        except:
            log_date = None
        return log_date


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
    total = models.IntegerField(default=0)
    adjusted_addressable = models.IntegerField(default=0)
    loading = models.FloatField(default=0.0)

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
                    ('Library_Q10_Longest_Alignment', 'q10_longest_alignment'),
                    ('Library_Q10_Mapped Bases', 'q10_mapped_bases'),
                    ('Library_Q10_Alignments', 'q10_alignments'),
                    ('Library_50Q17_Reads', 'i50Q17_reads'),
                    ('Library_100Q17_Reads', 'i100Q17_reads'),
                    ('Library_200Q17_Reads', 'i200Q17_reads'),
                    ('Library_Mean_Q17_Length', 'q17_mean_alignment_length'),
                    ('Library_Q17_Longest_Alignment', 'q17_longest_alignment'),
                    ('Library_Q17_Mapped Bases', 'q17_mapped_bases'),
                    ('Library_Q17_Alignments', 'q17_alignments'),
                    ('Library_50Q20_Reads', 'i50Q20_reads'),
                    ('Library_100Q20_Reads', 'i100Q20_reads'),
                    ('Library_200Q20_Reads', 'i200Q20_reads'),
                    ('Library_Mean_Q20_Length', 'q20_mean_alignment_length'),
                    ('Library_Q20_Longest_Alignment', 'q20_longest_alignment'),
                    ('Library_Q20_Mapped Bases', 'q20_mapped_bases'),
                    ('Library_Q20_Alignments', 'q20_alignments'),
                    ('Library_Key_Peak_Counts', 'aveKeyCounts'),
                    ('Library_50Q47_Reads', 'i50Q47_reads'),
                    ('Library_100Q47_Reads', 'i100Q47_reads'),
                    ('Library_200Q47_Reads', 'i200Q47_reads'),
                    ('Library_Mean_Q47_Length', 'q47_mean_alignment_length'),
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

    q7_alignments = models.IntegerField()
    q7_mapped_bases = models.BigIntegerField()
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

    q10_alignments = models.IntegerField()
    q10_mapped_bases = models.BigIntegerField()
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

    q17_alignments = models.IntegerField()
    q17_mapped_bases = models.BigIntegerField()
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

    q20_alignments = models.IntegerField()
    q20_mapped_bases = models.BigIntegerField()
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

    q47_mapped_bases = models.BigIntegerField()
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

    duplicate_reads = models.IntegerField(null=True, blank=True)

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
    q0_median_read_length = models.IntegerField(default=0)
    q0_mode_read_length = models.IntegerField(default=0)
    q0_50bp_reads = models.IntegerField()
    q0_100bp_reads = models.IntegerField()
    q0_150bp_reads = models.IntegerField(default=0)
    q17_bases = models.BigIntegerField()
    q17_reads = models.IntegerField()
    q17_max_read_length = models.IntegerField()
    q17_mean_read_length = models.FloatField()
    q17_median_read_length = models.IntegerField(default=0)
    q17_mode_read_length = models.IntegerField(default=0)
    q17_50bp_reads = models.IntegerField()
    q17_100bp_reads = models.IntegerField()
    q17_150bp_reads = models.IntegerField()
    q20_bases = models.BigIntegerField()
    q20_reads = models.IntegerField()
    q20_max_read_length = models.FloatField()
    q20_mean_read_length = models.IntegerField()
    q20_median_read_length = models.IntegerField(default=0)
    q20_mode_read_length = models.IntegerField(default=0)
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

    def save(self, *args, **kwargs):
        try:
            with transaction.commit_on_success():
                super(Template, self).save(*args, **kwargs)
                tasks.generate_TF_files(self.key)
        except:
            logger.exception('Failed saving Test Fragment %s' % self.name)
            raise

    def delete(self, *args, **kwargs):
        try:
            with transaction.commit_on_success():
                super(Template, self).delete(*args, **kwargs)
                tasks.generate_TF_files(self.key)
        except:
            logger.exception('Failed deleting Test Fragment %s' % self.name)
            raise

# Backup objects are obsolete, replaced by DMFileStat, DMFileSet; not removing the class for Data Management update from pre-TS3.6 servers


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


class Chip(models.Model):
    name = models.CharField(max_length=128)
    slots = models.IntegerField()
    description = models.CharField(max_length=128, default="")

    isActive = models.BooleanField(default=True)

    ALLOWED_INSTRUMENT_TYPES = (
        ('', "Undefined"),
        ('pgm', 'PGM'),
        ('proton', 'Proton'),
        ('S5', 'S5')
    )
    #compatible instrument type
    instrumentType = models.CharField(max_length=64, choices=ALLOWED_INSTRUMENT_TYPES, default='', blank=True)

    ALLOWED_EARLY_DAT_FILE_DELETION_MODES = (
        ('', "Not supported"),
        ('0', 'Do not delete dat file early'),
        ('1', 'Delete dat file for the block early once the block is transferred')
    )
    earlyDatFileDeletion = models.CharField(max_length=1, choices=ALLOWED_EARLY_DAT_FILE_DELETION_MODES, default="", blank=True, verbose_name="Delete .dat files early")

    def getChipDisplayedName(self):
        if self.description:
            return self.description
        else:
            return self.name


    def getChipDisplayedVersion(self):
        value = self.getChipDisplayedName()

        parts = value.split("v", 1)
        return "v" + parts[1]  if len(parts) > 1 else ""

    def getChipDisplayedNamePrimaryPrefix(self):
        """
        Returns all the primary chip displayed name for UI to display
        e.g., for chip name "318 Select", return 318
        """
        isVersionInfoFound, prefixes = Chip.getChipDisplayedNameParts(self.getChipDisplayedName())

        return prefixes[0]

    def getChipDisplayedNameSecondaryPrefix(self):
        """
        Returns all the second portion of the chip displayed name for UI to display
        e.g., for chip name "318 Select", return "Select"
        """
        isVersionInfoFound, prefixes = Chip.getChipDisplayedNameParts(self.getChipDisplayedName())

        return prefixes[-1] if len(prefixes) > 1 else ""

    @staticmethod
    def getChipDisplayedNameParts(value):
        """
        Returns all the relevant parts of the chip displayed name for UI to display
        If value is "318v2 Select", this API will skip version info and return "318" and "Select"
        """

        parts = value.split("v", 1)
        isVersionInfoFound = True if len(parts) > 1 else False
        index = len(parts) - 1
        if index >= 0:
            parts2 = parts[index].rsplit(" ", 1)

            parts3 = []

            if len(parts2) > 1:
                if isVersionInfoFound:
                    parts3.append(parts[0])

                for i, part in enumerate(parts2):
                    if isVersionInfoFound:
                        if i > 0:
                            parts3.append(part)
                    else:
                        parts3.append(part)

                return isVersionInfoFound, parts3
            else:
                parts3.append(parts[0])

        return isVersionInfoFound, parts3


class GlobalConfig(models.Model):
    name = models.CharField(max_length=512)
    selected = models.BooleanField()

    records_to_display = models.IntegerField(default=20, blank=True)
    default_test_fragment_key = models.CharField(max_length=50, blank=True)
    default_library_key = models.CharField(max_length=50, blank=True)
    default_flow_order = models.CharField(max_length=100, blank=True)
    plugin_output_folder = models.CharField(max_length=500, blank=True)
    web_root = models.CharField(max_length=500, blank=True)
    site_name = models.CharField(max_length=500, blank=True)
    default_storage_options = models.CharField(max_length=500,
                                       choices=Experiment.STORAGE_CHOICES,
                                       default='D', blank=True)
    auto_archive_ack = models.BooleanField("Auto-Acknowledge Delete?", default=False)
    auto_archive_enable = models.BooleanField("Enable Auto Actions?", default=False)

    enable_auto_pkg_dl = models.BooleanField("Enable Package Auto Download", default=True)
    #enable_alt_apt = models.BooleanField("Enable USB apt repository", default=False)
    enable_version_lock = models.BooleanField("Enable TS Version Lock", default=False)
    ts_update_status = models.CharField(max_length=256, blank=True)
    # Controls analysis pipeline alternate processing path

    ALLOWED_RECALIBRATION_MODES = (
         ('standard_recal', 'Default Calibration'),
         ('no_recal', 'No Calibration'),
         ('panel_recal', 'Special')
     )

    base_recalibration_mode = models.CharField(max_length=64, blank=False, null=False, choices=ALLOWED_RECALIBRATION_MODES, default='standard_recal')

    mark_duplicates = models.BooleanField(default=False)
    realign = models.BooleanField(default=False)
    check_news_posts = models.BooleanField("check for news posts", default=True)
    enable_auto_security = models.BooleanField("Enable Security Updates", default=True)
    sec_update_status = models.CharField(max_length=128, blank=True)
    enable_compendia_OCP = models.BooleanField("Enable Oncomine?", default=False)
    enable_support_upload = models.BooleanField("Enable Support Upload?", default=False)
    enable_nightly_email = models.BooleanField("Enable Nightly Email Notifications?", default=True)
    cluster_auto_disable = models.BooleanField("Automatically disable SGE queue on node errors?", default=True)

    def set_TS_update_status(self, inputstr):
        self.ts_update_status = inputstr

    def set_enableAutoPkgDL(self, flag):
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
        o = cls.objects.order_by('pk')[:1].get()
        return o


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

    """
    The model which will be run against an result to produce plugin results
    """

    # the name of the shell script to launch for plugins
    LAUNCH_SHELL = 'launch.sh'

    # maximum size of a plugin which will be archived
    MAX_ARCHIVE_SIZE = 50 * 1024 * 1024

    # this flag will indicate if the plugin will be included by default in new plans (not from template)
    defaultSelected = models.BooleanField(default=False)

    # the name of the plugin
    name = models.CharField(max_length=512, db_index=True)

    # version of the plugin for a given plugin via it's name
    version = models.CharField(max_length=256)

    # a brief description of the plugin
    description = models.TextField(blank=True, default="")

    # the date of the install (Not sure if that is the intended use)
    date = models.DateTimeField(auto_now_add=True)

    # this toggles visibility on the interface
    selected = models.BooleanField(default=False)

    # path the plugin directory
    path = models.CharField(max_length=512, blank=True, default="")

    # this will indicate if the plugin should display it's results on the "primary" section of the results page
    # indicating that it's results are part of the experiment results
    majorBlock = models.BooleanField(default=False)

    # the default configuration for the plugin
    config = json_field.JSONField(blank=True, null=True, default="")

    # the install status of the the plugin
    # TODO: This field is inconsistantly populated and should be broken out
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

    ## file containing plugin definition. launch.sh or PluginName.py
    script = models.CharField(max_length=256, blank=True, default="")

    # link to the debian package name
    packageName = models.CharField(max_length=256, blank=True, default="", db_column="packagename")

    @cached_property
    def isConfig(self):
        """
        Shows if this plugin has a configuration page to show
        :return: True if config.html is present, false otherwise
        """
        try:
            if os.path.exists(os.path.join(self.path, "config.html")):
                #provide a link to load the plugins html
                return urlresolvers.reverse('configure_plugins_plugin_configure', kwargs={'pk': self.pk, 'action': 'config'})
        except OSError:
            pass
        return False

    @cached_property
    def isPlanConfig(self):
        """
        Shows if this plugin can be configured for a plan.
        :return: True if plan.html is present, false otherwise
        """
        try:
            if os.path.exists(os.path.join(self.path, "plan.html")):
                #provide a link to load the plugins html
                return urlresolvers.reverse('configure_plugins_plugin_configure', kwargs={'pk': self.pk, 'action': 'plan'})
        except OSError:
            pass
        return False

    @cached_property
    def hasAbout(self):
        """
        Shows if this plugin has a about page to show
        :return: Returns True if about.html is present, false otherwise
        """
        try:
            if os.path.exists(os.path.join(self.path, "about.html")):
                #provide a link to load the plugins html
                return urlresolvers.reverse('configure_plugins_plugin_configure', kwargs={'pk': self.pk, 'action': 'about'})
        except OSError:
            pass
        return False

    @cached_property
    def isInstance(self):
        """

        :return:
        """
        try:
            if os.path.exists(os.path.join(self.path, "instance.html")):
                return urlresolvers.reverse('configure_plugins_plugin_configure', kwargs={'pk': self.pk, 'action': 'report'})
        except OSError:
            pass
        return False

    @cached_property
    def isSupported(self):
        """
        This will return if this plugin is available from aptitude
        :return: A boolean
        """
        return True if self.packageName and self.packageName != 'ion-rndplugins' else False

    @cached_property
    def availableVersions(self):
        """
        For supported plugins only, this will get all of the versions available for the plugin
        :return: A list of verison names
        """

        # Get the plugin package name and check if this is supported
        pluginPackageName = self.packageName
        if not pluginPackageName:
            return list()

        # return a list of the available version gotten from the plugin daemon
        client = xmlrpclib.ServerProxy("http://" + settings.IPLUGIN_HOST + ":" + str(settings.IPLUGIN_PORT))
        pluginDict = client.GetSupportedPluginInfo(pluginPackageName)
        return pluginDict['AvailableVersions']

    @cached_property
    def isUpgradable(self):
        """
        For supported plugins this will indicate if an upgrade is available
        :return: True if the supported plugin has an aptitude version available, false otherwise
        """

        # Get the plugin package name and check if this is supported
        pluginPackageName = self.packageName
        if not pluginPackageName:
            return False

        # return a list of the available version gotten from the plugin daemon
        client = xmlrpclib.ServerProxy("http://" + settings.IPLUGIN_HOST + ":" + str(settings.IPLUGIN_PORT))
        pluginDict = client.GetSupportedPluginInfo(pluginPackageName)
        return pluginDict['UpgradeAvailable']

    def __unicode__(self):
        """
        Standard method to "stringify" the class
        :return: Returns the name of the plugin
        """
        return self.name

    def versionedName(self):
        """
        Creates a string identifier for this plugin
        :return:
        """
        return "%s--v%s" % (self.name, self.version)

    def versionGreater(self, other):
        """
        Help for comparing plugins by version number
        :param other:
        :return: True if other is less than this version, false otherwise
        """
        return(LooseVersion(self.version) > LooseVersion(other.version))

    def installStatus(self):
        """
        This method helps us know if a plugin was installed sucessfully.
        :return: The string install status if it's contained in the JSON structure
        """
        if self.status.get("result"):
            if self.status["result"] == "queued":
                return "queued"
        return self.status.get("installStatus", "installed")

    @cached_property
    def pluginscript(self):
        """
        Gets the script which will be executed by the plugin
        :return: The full path to the executable script
        """
        # Now cached, join path and script
        if not self.script:

            self.script, _ = pluginmanager.find_pluginscript(self.path, self.name)
        if not self.script:
            self.script = '' # Avoid Null value in db column. find_pluginscript can return None.
            return None
        return os.path.join(self.path, self.script)

    def info(self, use_cache=True):
        """
        This requires external process call when use_cache=False.
        Can be expensive. Avoid in API calls, only fetch when necessary.
        """
        if use_cache:
            return self.info_from_model()

        context = {'plugin': self}
        info = PluginManager.get_plugininfo(self.name, self.pluginscript, context, use_cache)
        # Cache is updated in background task.
        if info is not None:
            self.updateFromInfo(info)  # update persistent db cache
        else:
            logger.error("Failed to get info from plugin: %s", self.name)

        return self.info_from_model()

    def updateFromInfo(self, info):
        """
        Updates the data structure from an "info" dictionary typically constructed from the plugin manager method
        :param info: A dictionary containing key information used regarding the plugin
        :return: True if anything has been changed, false otherwise
        """
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
            'runtypes': info.get('runtypes'),
            'runlevels': info.get('runlevels'),
            'depends': info.get('depends'),
        }
        if self.pluginsettings != pluginsettings:
            self.pluginsettings = pluginsettings
            changed = True

        majorBlock = info.get('major_block', False)
        if self.majorBlock != majorBlock:
            self.majorBlock = majorBlock
            changed = True

        docs = info.get('docs', None)
        if docs and self.description != docs:
            self.description = docs
            changed = True

        if changed:
            self.save()

        return changed

    def info_from_model(self):
        """
        This will construct a dictionary of the information for this model
        :return:
        """
        info = {
            'name': self.name,
            'version': self.version,
            'runtypes': self.pluginsettings.get('runtypes', []),
            'runlevels': self.pluginsettings.get('runlevels', []),
            'features': self.pluginsettings.get('features', []),
            'depends': self.pluginsettings.get('depends', []),
            'config': self.userinputfields,
            'docs': self.description,
            'major_block': self.majorBlock,
            'pluginorm': self.pk,
            'active': self.active,
        }

        return PluginInfo(info).todict()

    @staticmethod
    def InstallZip(pathToZip):
        """
        Installs a plugin from a zip file
        :param pathToZip: A local file path to the zip file in question
        :return: Primary key to newly created plugin
        """

        logger.info("Starting install process for plugin at " + pathToZip)

        # check that the file exists
        if not os.path.exists(pathToZip):
            raise Exception("Attempt to install plugin from zip failed because " + pathToZip + " does not exist.")

        zipSize = os.path.getsize(pathToZip)
        if zipSize == 0:
            raise Exception("The zip file " + pathToZip + " is of zero size and has no contents.")

        # create a temporary directory to extract the zip file to
        pathToExtracted = tempfile.mkdtemp()

        # this try statement only is here for the accompanying finally statement to clean up the extracted zip contents
        # any exception should be allowed to bubble up and be handled at a higher level
        try:
            # extract the zip file
            tasks.extract_zip(pathToZip, pathToExtracted, logger=logger)

            # get the plugin name from the top level directory created from the zip file
            # there is an assumption that only one directory will exists
            listOfDirectories = [name for name in os.listdir(pathToExtracted) if os.path.isdir(os.path.join(pathToExtracted, name))]
            if len(listOfDirectories) != 1:
                raise Exception("The zip file contained a number of directories where the specification only calls for one.  This has caused an ambiguous state where the plugin name cannot be divined.")

            pluginName = listOfDirectories[0].strip()

            # predict where each of the scripts should be
            pathToShellScript = os.path.join(pathToExtracted, pluginName, Plugin.LAUNCH_SHELL)
            pathToPythonScript = os.path.join(pathToExtracted, pluginName, pluginName + '.py')

            # detect if neither of these was presented to the installer
            if not os.path.exists(pathToShellScript) and not os.path.exists(pathToPythonScript):
                raise Exception("Neither a shell script or a python script was included in the plugin package in root of the plugin directory.")

            # detect if both of the scripts are present
            script = pathToPythonScript if os.path.exists(pathToPythonScript) else pathToShellScript

            # get information from the script
            try:
                from iondb.plugins.manager import PluginManager
                info = PluginManager.get_plugininfo(pluginName, script)
            except Exception as exc:
                raise Exception("The following error occurred while attempting to load the plugin: " + str(exc))

            # get the old version of the plugin
            oldPlugin = None
            try:
                oldPlugin = Plugin.objects.get(name=pluginName, active=True)
                # check to see if the script is currently under package management control
                if getPackageName(oldPlugin.script):
                    logger.warning("Replacing a supported plugin with and unsupported contents fo a zip file.")

            except ObjectDoesNotExist:
                # just continue
                pass

            oldVersion = oldPlugin.version if oldPlugin else '0.0.0-' + str(datetime.datetime.now())
            #make sure this version is an upgrade
            if LooseVersion(info['version']) <= LooseVersion(oldVersion):
                raise Exception("Cannot install this plugin because the it would be a downgrade from version " + oldVersion + " to " + info['version'] + ".")

            # update the plugin executable in the file system by first moving the old one if it exists
            basePluginsPath = os.path.join('/results', 'plugins')
            pluginBinaryPath = os.path.join(basePluginsPath, pluginName)
            if os.path.exists(pluginBinaryPath):

                size = directorySize(pluginBinaryPath)
                if size > Plugin.MAX_ARCHIVE_SIZE:
                    logger.info("The old plugin directory " + pluginBinaryPath + " is " + bytesToHumanReadableSize(size) + " which makes it to big to archive so instead it will be deleted.")
                    shutil.rmtree(pluginBinaryPath)
                else:
                    try:
                        # make sure the archive path is present
                        archivePath = os.path.join(basePluginsPath, 'archive')
                        if not os.path.exists(archivePath):
                            os.mkdir(archivePath)

                        # make the archive directory
                        pluginArchivePath = os.path.join(archivePath, pluginName)
                        if not os.path.exists(pluginArchivePath):
                            os.mkdir(pluginArchivePath)

                        # od the move from the current directory to the old
                        pluginArchiveVersionPath = os.path.join(pluginArchivePath, oldVersion)
                        if not os.path.exists(pluginArchiveVersionPath):
                            os.rename(pluginBinaryPath, pluginArchiveVersionPath)
                        else:
                            # if the archive already exists, we still need to delete this one
                            shutil.rmtree(pluginBinaryPath)
                    except Exception as exc:
                        # if something when wrong we should set a warning log and just remove the base directory
                        logger.warning("Issue when archiving old plugin: " + str(exc))
                        if os.path.exists(pluginBinaryPath):
                            shutil.rmtree(pluginBinaryPath)

            # move the extracted contents into the new position
            shutil.move(os.path.join(pathToExtracted, pluginName), basePluginsPath)

            # create the new database entry
            newPlugin, created = Plugin.objects.get_or_create(name=pluginName, version=info['version'])
            newPlugin.script = os.path.join(pluginBinaryPath, os.path.basename(script))
            newPlugin.date = datetime.datetime.now()
            newPlugin.selected = True
            newPlugin.path = pluginBinaryPath
            newPlugin.url = ''
            newPlugin.active = True
            newPlugin.status = dict() #TODO: not really sure what to do with this
            newPlugin.pluginsettings = dict()
            newPlugin.pluginsettings['features'] = info['features']
            newPlugin.pluginsettings['runtypes'] = info['runtypes']
            newPlugin.pluginsettings['runlevels'] = info['runlevels']
            newPlugin.pluginsettings['depends'] = info['depends']
            newPlugin.description = info['docs']
            newPlugin.majorBlock = info['major_block']
            newPlugin.config = oldPlugin.config if oldPlugin else info['config']

            # push the new plugin to the database
            newPlugin.save()

            # deactivate the old plugin
            if oldPlugin:
                oldPlugin.active = False
                oldPlugin.save()

            return newPlugin.pk
        finally:
            shutil.rmtree(pathToExtracted, True)

    @staticmethod
    def Uninstall(primaryKey):
        """
        This will remove supported plugins from the system
        :param primaryKey: The primary key of the package to uninstall
        """

        plugin = Plugin.objects.get(pk=primaryKey)
        plugin.active = False
        logger.info("Now uninstalling the plugin " + plugin.name + ".")

        if plugin.isSupported:
            # remove supported plugin
            pluginPackageName = plugin.packageName

            process = subprocess.Popen(['sudo', '/opt/ion/iondb/bin/ion_plugin_install.py', pluginPackageName + '=remove'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, err = process.communicate()
            logger.warning(output)

        else:
            # remove unsupported plugin
            plugin_path = os.path.normpath(plugin.path)
            plugin.path = '' ## None?

            # Warning - removing symlink and removing tree may break old reports
            # In current usage, only used by instance.html, not reports.
            # If we remove the path, we must remove the symlink too,
            # or we're left with a broken symlink.
            oldPluginMedia = os.path.join('/results/pluginMedia', plugin.name)
            if os.path.lexists(oldPluginMedia):
                try:
                    os.unlink(oldPluginMedia)
                except OSError:
                    pass

            # remove the plugin path
            try:
                logger.info("Now removing the path " + plugin_path + ".")
                shutil.rmtree(plugin_path)
            except (IOError, OSError):
                logger.error("Failed to delete plugin path on uninstall: " + plugin_path)

        plugin.save()
        # force a refresh with the TS database
        pluginmanager.rescan()

    @staticmethod
    def UpgradeSupported(pluginPackageName, version=''):
        """
        This will mark plugin packages for upgrading
        :param pluginPackageName: The string name of the package
        :param version: The version you are targeting to upgrade to.  Can also be a lower version number to downgrade
        """

        # make sure it is always used as a list
        if version:
            pluginPackageName += "=" + version

        logger.info("Upgrading supported plugin: " + pluginPackageName)
        # call out to sudo to run the install script at elevated executable levels
        process = subprocess.Popen(['sudo', '/opt/ion/iondb/bin/ion_plugin_install.py', pluginPackageName], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()

        if process.returncode:
            raise Exception("Error upgrading: " + err)

        # force a refresh with the TS database
        pluginmanager.rescan()

    @staticmethod
    def InstallDeb(pathToDeb):
        """
        Install a plugin from a deb package file
        :param pathToDeb: A path to the deb file which will be installed.
        """

        # do some sanity checks on the package
        if not os.path.exists(pathToDeb):
            raise Exception("No file at " + pathToDeb)

        import subprocess
        p = subprocess.Popen(["sudo", "/opt/ion/iondb/bin/ion_plugin_install_deb.py", pathToDeb], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # check to the error
        if p.returncode:
            raise Exception(err)

        # force a refresh with the TS database
        pluginmanager.rescan()

    def natural_key(self):
        """
        Gets the "natural key" for this data table
        :return: union of the name and the version
        """
        return (self.name, self.version)

    class Meta:
        unique_together = (('name', 'version'), )


@receiver(pre_delete, sender=Plugin, dispatch_uid="delete_plugin")
def on_plugin_delete(sender, instance, **kwargs):
    """ Uninstall plugin on db record removal.
    Note: DB Delete is a purge and destroys all plugin content.
    Best to uninstall and leave records / results """
    if instance.path or instance.active:
        Plugin.Uninstall(instance.id)


class PluginResult(models.Model):
    """The results for a plugin"""

    # ForeignKey: PluginResultJob plugin_result_jobs
    _CSV_METRICS = (("Plugin Data", 'store'))

    # Many to Many mapping at the intersection of Results and Plugins
    plugin = models.ForeignKey(Plugin)

    # create a reference to the result
    result = models.ForeignKey(Results, related_name='pluginresult_set')

    # not really sure what this is....
    store = json_field.JSONField(blank=True)

    # the key to be used to access the rest api
    apikey = models.CharField(max_length=256, blank=True, null=True)

    # the cached size of the results folder
    size = models.BigIntegerField(default=-1)

    # the totaly number of inodes in the results folder
    inodes = models.BigIntegerField(default=-1)

    # the owner of the results folder
    owner = models.ForeignKey(User)

    # cache global context
    _gc = None

    def starttime(self):
        """Gets the first start time"""
        prj = self.plugin_result_jobs.earliest('starttime')
        return prj.starttime if prj else None

    def endtime(self):
        """Gets the last end time"""
        prj = self.plugin_result_jobs.order_by('-endtime').first()
        return prj.endtime if prj else None

    def state(self):
        """This will report the latest job's status"""
        prj = self.plugin_result_jobs.order_by('-starttime').first()
        return prj.state if prj else 'Unknown'

    def can_terminate(self):
        """This will evaluate if the plugin result has any jobs which can be terminated"""
        for plugin_result_job in self.plugin_result_jobs.all():
            if plugin_result_job.can_terminate():
                return True
        return False

    def files(self):
        """returns a list of all html and php files"""
        if not os.path.exists(self.path()):
            return list()

        return [f for f in os.listdir(self.path()) if (f.endswith(".html") or f.endswith(".php"))]

    def url(self):
        gc = GlobalConfig.get()
        return os.path.join(self.result.reportLink, gc.plugin_output_folder, os.path.basename(self.path()), '')

    @cached_property
    def default_path(self):
        return self.path(create=False, fallback=True)

    def path(self, create=False, fallback=True):
        if not self._gc:
            self._gc = GlobalConfig.get()

        base_path = os.path.join(self.result.get_report_dir(),
                                 self._gc.plugin_output_folder, # Default:'plugin_out'
                                )
        path = os.path.join(base_path, "%s_out.%d" % (self.plugin.name, self.pk))
        if os.path.exists(path):
            return path

        if create:
            os.makedirs(path, 0775)
            return path

        if not fallback:
            return path

        legacy_path = os.path.join(self.result.get_report_dir(),
                            self._gc.plugin_output_folder, # Default:'plugin_out'
                            "%s_out" % (self.plugin.name,))
        if os.path.exists(legacy_path):
            return legacy_path

        return path

    def UpdateSizeAndINodeCount(self):

        # reset the size and the inodes
        self.size = 0
        self.inodes = 0
        total_size = 0
        inodes = 0

        d = self.default_path
        if d and os.path.exists(d):
            file_walker = (os.path.join(root, f) for root, _, files in os.walk(d) for f in files)
            for f in file_walker:
                if not os.path.isfile(f):
                    continue
                inodes += 1
                try:
                    total_size += os.lstat(f).st_size
                except OSError:
                    logger.exception("Failed accessing %s during calc_size", f)

        logger.info("PluginResult %d for %s has %d byte(s) in %d file(s)", self.id, self.plugin.name, total_size, inodes)

        # update based on finding
        self.size = total_size
        self.inodes = inodes
        self.save()

        return total_size, inodes

    def prepare(self):
        # Always overwrite key - possibly invalidating existing key from running instance
        self.apikey = _generate_key()

    def SetState(self, state, grid_engine_jobid, jobid=None):
        """
        :param state: The new state of the plugin results
        :param jobid: In the case of a starting state this is the job id which was started
        """

        prj = self.plugin_result_jobs.get(grid_engine_jobid=grid_engine_jobid)
        prj.state = state
        if state == 'Started':
            prj.starttime = timezone.now()
            if jobid is not None:
                if prj.grid_engine_jobid is not None and (prj.grid_engine_jobid != jobid):
                    logger.warn("(Re-)started as different queue jobid: '%d' was '%d'", jobid, self.jobid)
                prj.grid_engine_jobid = jobid
        elif state in COMPLETED_STATES:
            prj.endtime = timezone.now()
            prj.grid_engine_jobid = -1
            self.apikey = None
            self.UpdateSizeAndINodeCount()
        prj.save()
        self.save()

    def start(self, jobid=None):
        self.state = 'Started'
        self.starttime = timezone.now()
        if jobid is not None:
            if self.jobid is not None and (self.jobid != jobid):
                logger.warn("(Re-)started as different queue jobid: '%d' was '%d'", jobid, self.jobid)
            self.jobid = jobid

    def stop(self):
        """Send a stop command to all sub-jobs"""
        for plugin_result_job in self.plugin_result_jobs.all():
            plugin_result_job.stop()
        self.UpdateSizeAndINodeCount()

    def complete(self, runlevel, state='Completed'):
        """ Call with state = Completed, Error, or other valid state """
        self.apikey = None
        prj = self.plugin_result_jobs.get(run_level=runlevel)
        prj.endtime = timezone.now()
        prj.state = state
        prj.grid_engine_jobid = -1

        # save the completed state
        self.save()
        prj.save()
        self.UpdateSizeAndINodeCount()

    @staticmethod
    def create_from_ophan(result, path):
        """This method will create a plugin result object from an orphaned file system folder"""

        # check to make sure that all of the required file system items exist
        start_plugin_path = os.path.join(path, 'startplugin.json')
        if not os.path.exists(start_plugin_path):
            return

        # read the start plugin item
        with open(start_plugin_path) as start_plugin_fp:
            start_plugin = json.load(start_plugin_fp)

        # get the plugin name and version number to look up the database entry and return if no such plugin exists on this system
        plugin_name = start_plugin['runinfo']['plugin']['name']
        plugin_version = start_plugin['runinfo']['plugin']['version']
        plugin = Plugin.objects.get(name=plugin_name, version=plugin_version)
        if not plugin:
            return

        # look up user and make sure it's present on the system
        username = start_plugin['runinfo']['username']
        user = User.objects.get(username=username)
        if not user:
            return

        # we need to save the result first
        plugin_result = PluginResult(result=result, plugin=plugin, owner=user)
        plugin_result.save()

        plugin_result_job = PluginResultJob(grid_engine_jobid=-1, run_level=start_plugin['runplugin']['runlevel'], state='Unknown')
        plugin_result_job.plugin_result_id = plugin_result.id
        plugin_result_job.save()

        # now we have to rename the directory
        root, ext = os.path.splitext(path)
        os.rename(path, root + "." + str(plugin_result.id))

        #update the inodes and size
        plugin_result.UpdateSizeAndINodeCount()

    def __unicode__(self):
        return "%s/%s" % (self.result, self.plugin)

    class Meta:
        ordering = ['-id']


# NB: Fails if not pre-delete, as path() queries linked plugin and result.
@receiver(pre_delete, sender=PluginResult, dispatch_uid="delete_pluginresult")
def on_pluginresult_delete(sender, instance, **kwargs):
    """Delete all of the files for a pluginresult record """
    try:
        directory = instance.default_path
    except:
        #if we can't find the result to delete
        return

    if directory and os.path.exists(directory):
        logger.info("Deleting Plugin Result %d in %s" % (instance.id, directory))
        client = xmlrpclib.ServerProxy(settings.IPLUGIN_STR)
        client.delete_pr_directory(directory)

RUNNING_STATES = ['Pending', 'Queued', 'Started']
COMPLETED_STATES = ['Completed', 'Error', 'Declined', 'Unknown', 'Resource', 'Timed Out', 'Cancelled', 'Archived']

class PluginResultJob(models.Model):
    """A job which was run to produce a plugin result"""

    ALLOWED_STATES = (
        ('Completed', 'Completed'),
        ('Error', 'Error'),
        ('Started', 'Started'),
        ('Declined', 'Declined'),
        ('Unknown', 'Unknown'),
        ('Queued', 'Queued'),  # In SGE queue
        ('Pending', 'Pending'),  # Prior to submitting to SGE
        ('Resource', 'Exceeded Resource Limits'),  ## SGE Errors
        ('Timed Out', 'Exceeded allowed Time'),  # SGE timed out
        ('Cancelled', 'User Cancelled'),  # User killed SGE job
        ('Archived', 'Archived'),  # Soft Deletion
    )

    RUN_LEVEL_CHOICES = (
        ('pre', 'Pre'),
        ('default', 'default'),
        ('block', 'block'),
        ('post', 'post'),
        ('separator', 'separator'),
        ('last', 'last')
    )

    # the job id of the grid engine
    grid_engine_jobid = models.IntegerField(null=True, blank=True)

    # the run level of the job
    run_level = models.CharField(null=False, max_length=20,blank=False, choices=RUN_LEVEL_CHOICES)

    # the start time of the job
    starttime = models.DateTimeField(null=True, blank=True)

    # the end time of the job
    endtime = models.DateTimeField(null=True, blank=True)

    # the configuration used to run this job
    config = json_field.JSONField(blank=True, default='')

    # setup a relationship with a many to one to the plugin result
    plugin_result = models.ForeignKey(PluginResult, related_name="plugin_result_jobs")

    # A record of the state of the job
    state = models.CharField(max_length=20, choices=ALLOWED_STATES)

    def can_terminate(self):
        """Returns if this plugin job is terminatable"""
        return self.state in RUNNING_STATES

    def stop(self):
        """Send a stop command to the SGE"""
        if self.grid_engine_jobid > 0:
            pluginServer = xmlrpclib.ServerProxy(settings.IPLUGIN_STR)
            endtime = datetime.datetime.now()
            try:
                pluginServer.sgeStop(self.grid_engine_jobid)
            except xmlrpclib.Fault as exc:
                pass
            self.grid_engine_jobid = -1
            self.state = 'Cancelled'
            self.save()


class dnaBarcode(models.Model):

    """Store a dna barcode"""
    name = models.CharField(max_length=128)     # name of barcode SET
    id_str = models.CharField(max_length=128)   # id of this barcode sequence
    active = models.BooleanField(default=True)

    ALLOWED_BARCODE_TYPES = (
        ('', 'Unspecified'),
        ('none', 'None'),
        ('dna', 'DNA'),
        ('rna', 'RNA'),
    )

    type = models.CharField(max_length=64, choices=ALLOWED_BARCODE_TYPES, default='', blank=True)
    sequence = models.CharField(max_length=128)
    length = models.IntegerField(default=0, blank=True)
    floworder = models.CharField(max_length=128, blank=True, default="")
    index = models.IntegerField()
    annotation = models.CharField(max_length=512, blank=True, default="")
    adapter = models.CharField(max_length=128, blank=True, default="")
    score_mode = models.IntegerField(default=0, blank=True)
    score_cutoff = models.FloatField(default=0)

    def __unicode__(self):
        return self.id_str

    class Meta:
        verbose_name_plural = "DNA Barcodes"


class ReferenceGenome(models.Model):

    """store info about the reference genome
    This should really hold the real path, it should also have methods for deleting the dirs for the files"""
    # Description
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
            celery.control.revoke(self.celery_task_id, terminate=True)
            children = AsyncResult(self.celery_task_id).children
            if children:
                for result in children:
                    result.revoke(terminate=True)
        logger.warning("Revoking celery task: {0}".format(self.celery_task_id))
        if os.path.exists(self.reference_path):
            tasks.delete_that_folder.delay(self.reference_path, "Deleting reference %s (%s)" % (self.short_name, self.pk))
        else:
            logger.error("Files do not exists for reference %d at %s" % (self.pk, self.reference_path))

        if self.file_monitor:
            self.file_monitor.delete()

        super(ReferenceGenome, self).delete()
        return True

    def enable_genome(self):
        """this should be around to move the genome in a disabled dir or not"""
        #get the new path to move the reference to
        enabled_path = os.path.join(iondb.settings.TMAP_DIR, self.short_name)
        try:
            shutil.move(self.reference_path, enabled_path)
        except:
            logger.exception("Failed to enable gnome %s" % self.short_name)
            return False
        else:
            self.reference_path = enabled_path
            self.enabled = True
            self.save()
        return True

    def disable_genome(self):
        """this should be around to move the genome in a disabled dir or not"""
        #get the new path to move the reference to
        disabled_path = os.path.join(iondb.settings.TMAP_DIR, "disabled", self.index_version, self.short_name)
        try:
            shutil.move(self.reference_path, disabled_path)
        except:
            logger.exception("Failed to disable gnome %s" % self.short_name)
            return False
        else:
            self.reference_path = disabled_path
            self.enabled = False
            self.save()
        return True

    def info_text(self):
        return os.path.join(self.reference_path, self.short_name + ".info.txt")

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
        orig = os.path.join(self.reference_path, self.short_name + ".orig")
        return os.path.exists(orig)

    def __unicode__(self):
        return u'%s' % self.name


class ThreePrimeAdapterManager(models.Manager):

    def get_by_natural_key(self, uid):
        return self.get(uid=uid)


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
    isActive = models.BooleanField(default=True)

    uid = models.CharField(max_length=32, unique=True, blank=False)

    ALLOWED_CHEMISTRY = (
        ('', 'Undefined'),
        ('avalanche', 'Avalanche')
    )

    chemistryType = models.CharField(max_length=64, choices=ALLOWED_CHEMISTRY, default='', blank=True)

    objects = ThreePrimeAdapterManager()

    class Meta:
        verbose_name_plural = "3' Adapters"

    def __unicode__(self):
        return u'%s' % self.name

    def natural_key(self):
        return (self.uid,)  # must return a tuple


    def save(self, *args, **kwargs):
        if (self.isDefault == False and ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType=self.chemistryType).count() == 1):
            currentDefaults = ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType=self.chemistryType)
            #there should only be 1 default for a given direction and chemistry type at any time
            for currentDefault in currentDefaults:
                if (self.id == currentDefault.id):
                    raise ValidationError("Error: Please set another adapter for %s direction and %s chemistry to be the default before changing this adapter not to be the default." % (self.direction, self.chemistryType))

        if (self.isDefault == True and ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType=self.chemistryType).count() > 0):
            currentDefaults = ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType=self.chemistryType)
            #there should only be 1 default for a given direction and chemistry type at any time
            for currentDefault in currentDefaults:
                if (self.id != currentDefault.id):
                    currentDefault.isDefault = False
                    super(ThreePrimeadapter, currentDefault).save()

        super(ThreePrimeadapter, self).save()



    def delete(self):
        if (self.isDefault == False and ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType=self.chemistryType).count() == 1):
            currentDefaults = ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType=self.chemistryType)
            #there should only be 1 default for a given direction and chemistry type at any time
            for currentDefault in currentDefaults:
                if (self.id == currentDefault.id):
                    raise ValidationError("Error: Deleting the default adapter is not allowed. Please set another adapter for %s direction and %s chemistry to be the default before deleting this adapter." % (self.direction, self.chemistryType))

        if (self.isDefault == True and ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType=self.chemistryType).count() > 0):
            currentDefaults = ThreePrimeadapter.objects.filter(direction=self.direction, isDefault=True, chemistryType=self.chemistryType)
            #there should only be 1 default for a given direction and chemistry type at any time
            for currentDefault in currentDefaults:
                if (self.id == currentDefault.id):
                    raise ValidationError("Error: Deleting the default adapter is not allowed. Please set another adapter for %s direction and %s chemistry to be the default before deleting this adapter." % (self.direction, self.chemistryType))

        super(ThreePrimeadapter, self).delete()


class FlowOrder(models.Model):
    name = models.CharField(max_length=64, blank=False, unique=True)
    description = models.CharField(max_length=128, default="", blank=True)
    flowOrder = models.CharField(max_length=1200, blank=True)

    isActive = models.BooleanField(default=True)
    isSystem = models.BooleanField(default=False)
    isDefault = models.BooleanField("use this by default", default=False)

    def __unicode__(self):
        return u'%s' % self.name

    def natural_key(self):
        return (self.name,)  # must return a tuple

    class Meta:
        verbose_name_plural = "Flow Orders"


class Publisher(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'contents' from Content

    name = models.CharField(max_length=200, unique=True)
    version = models.CharField(max_length=256)
    date = models.DateTimeField()
    path = models.CharField(max_length=512)
    global_meta = json_field.JSONField(blank=True)

    def __unicode__(self): return self.name

    def get_editing_scripts(self):
        pub_files = os.listdir(self.path)
        # pylint: disable=bad-whitespace
        stages = (
            ("pre_process",     "Pre-processing"),
            ("validate",        "Validating"),
            ("post_process",    "Post-processing"),
            ("register",        "Registering"),
        )
        pub_scripts = []
        for stage, name in stages:
            for pub_file in pub_files:
                if pub_file.startswith(stage):
                    script_path = os.path.join(self.path, pub_file)
                    pub_scripts.append((script_path, name))
                    break
        return pub_scripts

@receiver(post_delete, sender=Publisher, dispatch_uid="delete_publisher")
def on_publisher_delete(sender, instance, **kwargs):
    """Delete publisher source files
    """
    logger.info("Deleting Publisher folder %s" % instance.path)
    shutil.rmtree(instance.path)

class ContentUpload(models.Model):

    # FOREIGN KEY DEFINITIONS
    # ForeignKey 'contents' from Content
    # ForeignKey 'logs' from UserEventLog

    file_path = models.CharField(max_length=255)
    status = models.CharField(max_length=255, blank=True)
    meta = json_field.JSONField(blank=True)
    publisher = models.ForeignKey(Publisher, null=True)

    def __unicode__(self): return u'ContentUpload %d' % self.id

    def upload_type(self):
        # TODO this can go into a separate field
        upload_type = 'Unknown'
        meta = self.meta
        if meta:
            upload_type = meta.get('upload_type', 'Custom (%s)' % self.publisher.name)
            if meta.get('is_ampliseq', False):
                upload_type = 'AmpliSeq ZIP'
            elif 'hotspot' in meta:
                upload_type = 'Hotspots' if meta['hotspot'] else 'Target Regions'

        return upload_type


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

    def __unicode__(self):
        return self.path

    def get_file_name(self):
        return os.path.basename(self.path)


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
def create_user_profile(sender, **kwargs):
    if kwargs.get('raw', False) or not kwargs.get('created', False):
        # raw - Do not run when loading fixtures
        # created - Only run on inital creation
        return
    instance = kwargs.get('instance')
    if instance is None:
        return
    UserProfile.objects.create(user=instance)


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


    def save(self, *args, **kwargs):
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
                if (self.id != currentDefault.id):
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

    # Special users, can be specified in route
    USER_STAFF = "_StaffOnly"

    # Message alert levels
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
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


class EventLog(models.Model):
    # Content-object field
    content_type = models.ForeignKey(ContentType,
            verbose_name='content type',
            related_name="content_type_set_for_%(class)s")
    object_pk = models.PositiveIntegerField()
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


class DMFileSet(models.Model):
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
    enabled = models.BooleanField(default=False)    # TODO: This is not used??
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


class DMFileStat(models.Model):
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
        ('IG', 'Importing'),
        ('E', 'Error'),             # Action resulted in error
        )

    # state of file action
    action_state = models.CharField(max_length=8, choices=ACT_STATES, default='L')

    # path to archived filed
    archivepath = models.CharField(max_length=512, blank=True, null=True)

    # megabytes used by files
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
        if state in [c for (c, d) in DMFileStat.ACT_STATES]:
            self.action_state = state
            self.save()
            # update related dmfilestats
            if self.dmfileset.type == dmactions_types.SIG:
                exp_id = self.result.experiment_id
                DMFileStat.objects.filter(dmfileset__type=dmactions_types.SIG, result__experiment__id=exp_id).update(action_state=state)
        else:
            raise Exception("Failed to set action_state. Invalid state: '%s'" % state)

    def isdisposed(self):
        return bool(self.action_state in ['AG', 'DG', 'AD', 'DD'])

    def isarchived(self):
        return bool(self.action_state in ['AG', 'AD'])

    def isdeleted(self):
        return bool(self.action_state in ['DG', 'DD'])

    def in_process(self):
        return bool(self.action_state in ['AG', 'DG', 'EG', 'SA', 'SE', 'SD', 'IG'])


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
    md5sum = models.CharField(max_length=32, default=None, null=True, blank=True)
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
        celery.control.revoke(self.celery_task_id, terminate=True)
        try:
            if os.path.exists(self.local_dir):
                shutil.rmtree(self.local_dir)
        except OSError:
            return False

        super(FileMonitor, self).delete()
        return True

    def in_progress(self):
        return self.status != "Complete" and self.progress < self.size

    def __unicode__(self):
        return u"FileMonitor/{0:d}".format(self.id)


class MonitorData(models.Model):
    treeDat = json_field.JSONField(blank=True)
    name = models.CharField(max_length=128, default="")


class NewsPost(models.Model):
    guid = models.CharField(max_length=2083, null=True, blank=True)  # There is no URL length limit but IE likes < 2038
    title = models.CharField(max_length=1024, blank=True, default="")
    summary = models.TextField(blank=True, default="")
    link = models.CharField(max_length=2083, blank=True, default="")  # There is no URL length limit but IE likes < 2038
    updated = models.DateTimeField(default=timezone.now)

    def __unicode__(self):
        return self.title if len(self.title) <= 60 else self.title[:57] + u'...'


class AnalysisArgs(models.Model):
    # Holds default command line args for selected chipType and kits
    name = models.CharField(max_length=256, blank=False, unique=True)
    description = models.CharField(max_length=256, blank=True, null=True, unique=True)
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
    # Calibration args
    calibrateargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Calibration args, used for recalibration")
    thumbnailcalibrateargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Thumbnail Calibration args, used for recalibration")
    # Basecaller args
    basecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Basecaller args")
    thumbnailbasecallerargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Thumbnail Basecaller args")
    # Alignment args
    alignmentargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Alignment args")
    thumbnailalignmentargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Thumbnail Alignment args")
    # Ionstats args
    ionstatsargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Ionstats args")
    thumbnailionstatsargs = models.CharField(max_length=5000, blank=True, verbose_name="Default Thumbnail Ionstats args")

    applType = models.ForeignKey(RunType, related_name="applType_analysisArgs", blank=True, null=True)
    applGroup = models.ForeignKey(ApplicationGroup, related_name="applGroup_analysisArgs", blank=True, null=True)

    isSystem = models.BooleanField(default=False)
    creator = models.ForeignKey(User, related_name="created_analysisArgs", blank=True, null=True)
    creationDate = models.DateTimeField(blank=True, null=True, auto_now_add=True, default=timezone.now)
    lastModifiedUser = models.ForeignKey(User, related_name="lastModified_analysisArgs", blank=True, null=True)
    lastModifiedDate = models.DateTimeField(blank=True, null=True, auto_now=True, default=timezone.now)


    def get_args(self):
        args = {
            'beadfindargs': self.beadfindargs,
            'analysisargs': self.analysisargs,
            'basecallerargs': self.basecallerargs,
            'prebasecallerargs': self.prebasecallerargs,
            'calibrateargs': self.calibrateargs,
            'alignmentargs': self.alignmentargs,
            'ionstatsargs': self.ionstatsargs,
            'thumbnailbeadfindargs':    self.thumbnailbeadfindargs,
            'thumbnailanalysisargs':    self.thumbnailanalysisargs,
            'thumbnailbasecallerargs':  self.thumbnailbasecallerargs,
            'prethumbnailbasecallerargs':  self.prethumbnailbasecallerargs,
            'thumbnailcalibrateargs': self.thumbnailcalibrateargs,
            'thumbnailalignmentargs': self.thumbnailalignmentargs,
            'thumbnailionstatsargs': self.thumbnailionstatsargs
        }
        return args


    @classmethod
    def best_match(cls, chipType, sequenceKitName='', templateKitName='', libraryKitName='', samplePrepKitName='', analysisArgs_objs=None, applicationTypeName="", applicationGroupName=""):
        ''' Find args that best match given chip type and kits.
            If chipType not found returns None.
            If none of the kits matched returns chip default.
        '''
        # chip name backwards compatibility
        chipType = chipType.replace('"', '')
        if Chip.objects.filter(name=chipType).count() == 0:
            chipType = chipType[:3]

        if analysisArgs_objs:
            args = analysisArgs_objs
        else:
            args = AnalysisArgs.objects.filter(chipType=chipType, isSystem=True).order_by('-pk')

        args_count = args.count()

        if args_count == 0:
            best_match = None
        elif args_count == 1:
            best_match = args[0]
        else:
            applicationType = None
            if applicationTypeName:
                applicationTypes = RunType.objects.filter(runType=applicationTypeName)
                if applicationTypes:
                    applicationType = applicationTypes[0]

            applicationGroup = None
            if applicationGroupName:
                applicationGroups = ApplicationGroup.objects.filter(name=applicationGroupName)
                if applicationGroups:
                    applicationGroup = applicationGroups[0]

            kits = {
                'sequenceKitName': sequenceKitName,
                'templateKitName': templateKitName,
                'libraryKitName': libraryKitName,
                'samplePrepKitName': samplePrepKitName,
                'applType': applicationType,
                'applGroup':  applicationGroup
            }
            match = [0]*args_count
            match_no_blanks = [0]*args_count
            unmatch_criteria = [0]*args_count

            for i, arg in enumerate(args):
                for key, value in kits.items():
                    if getattr(arg, key) == value:
                        match[i] += 1
                        if value:
                            match_no_blanks[i] += 1
                    else:
                        if getattr(arg, key):
                            #if analysisArgs has criteria specified
                            unmatch_criteria[i] += 1

                if unmatch_criteria[i] > 0:
                    match_no_blanks[i] = -1
                    match[i] = -1

            if max(match_no_blanks) > 0:
                best_match = args[match.index(max(match))]
            else:
                try:
                    best_match = args.get(chip_default=True)
                except:
                    best_match = None

        if best_match is not None:
            logger.debug("best_match() chipType=%s; applicationTypeName=%s; applicationGroupName=%s FOUND best_match.pk=%d" % (chipType, applicationTypeName, applicationGroupName, best_match.pk))

        return best_match


    @classmethod
    def possible_matches(cls, chipType, sequenceKitName='', templateKitName='', libraryKitName='', samplePrepKitName='', analysisArgs_objs=None, applicationTypeName="", applicationGroupName=""):
        ''' Find all the args that are potentially good for to choose from.
            If chipType not found returns None.
            If none of the kits matched returns chip default.
        '''
        # chip name backwards compatibility
        if chipType:
            chipType = chipType.replace('"', '')
            if Chip.objects.filter(name=chipType).count() == 0:
                chipType = chipType[:3]
        else:
            chipType = ""

        best_match = AnalysisArgs.best_match(chipType, sequenceKitName, templateKitName, libraryKitName, samplePrepKitName, analysisArgs_objs, applicationTypeName, applicationGroupName)

        active_analysisArgs_objs = AnalysisArgs.objects.filter(active=True).order_by("description")

        if best_match:
            if applicationTypeName and applicationGroupName:
                ##logger.debug("possible_matches() #1... chipType=%s; applicationTypeName=%s; applicationGroupName=%s" %(chipType, applicationTypeName, applicationGroupName))

                qset = (
                    Q(pk=best_match.pk) |
                    Q(chipType__in=[chipType]) |
                    Q(chipType__in=["", chipType]) & Q(applType__runType=applicationTypeName) |
                    Q(chipType__in=["", chipType]) & Q(applGroup__name=applicationGroupName)
                )
                possible_analysisArgs_objs = active_analysisArgs_objs.filter(qset).order_by("description")

                ##logger.debug("possible_matches() #1...EXIT...  possible_analysisArgs_objs=%s" %(possible_analysisArgs_objs))

                return possible_analysisArgs_objs
            elif best_match and applicationTypeName:
                ##logger.debug("possible_matches() #2... chipType=%s; runType=%s" %(chipType, applicationTypeName))

                qset = (
                    Q(pk=best_match.pk) |
                    Q(chipType__in=[chipType]) |
                    Q(chipType__in=["", chipType]) & Q(applType__runType=applicationTypeName)
                )
                possible_analysisArgs_objs = active_analysisArgs_objs.filter(qset).order_by("description")

                return possible_analysisArgs_objs
            elif best_match and applicationGroupName:
                ##logger.debug("possible_matches() #3...  chipType=%s; applicationGroupName=%s" %(chipType, applicationGroupName))

                qset = (
                    Q(pk=best_match.pk) |
                    Q(chipType__in=[chipType]) |
                    Q(chipType__in=["", chipType]) & Q(applGroup__name=applicationGroupName)
                )
                possible_analysisArgs_objs = active_analysisArgs_objs.filter(qset).order_by("description")

                return possible_analysisArgs_objs
            else:
                ##logger.debug("possible_matches() #4...  best_match chipType=%s; " %(chipType))

                qset = (
                    Q(pk=best_match.pk) |
                    Q(chipType__in=["", chipType])
                )
                possible_analysisArgs_objs = active_analysisArgs_objs.filter(qset).order_by("description")

                return possible_analysisArgs_objs
        else:

            if applicationTypeName and applicationGroupName:
                ##ogger.debug("possible_matches() #5... chipType=%s; applicationTypeName=%s; applicationGroupName=%s" %(chipType, applicationTypeName, applicationGroupName))

                qset = (
                    Q(chipType__in=["", chipType]) |
                    Q(chipType__in=["", chipType]) & Q(applType__runType=applicationTypeName) |
                    Q(chipType__in=["", chipType]) & Q(applGroup__name=applicationGroupName)
                )
                possible_analysisArgs_objs = active_analysisArgs_objs.filter(qset).order_by("description")

                return possible_analysisArgs_objs
            elif applicationTypeName:
                ##logger.debug("possible_matches() #6... chipType=%s; runType=%s" %(chipType, applicationTypeName))

                qset = (
                    Q(chipType__in=[chipType]) |
                    Q(chipType__in=["", chipType]) & Q(applType__runType=applicationTypeName)
                )
                possible_analysisArgs_objs = active_analysisArgs_objs.filter(qset).order_by("description")

                return possible_analysisArgs_objs
            elif applicationGroupName:
                ##logger.debug("possible_matches() #7...  chipType=%s; applicationGroupName=%s" %(chipType, applicationGroupName))

                qset = (
                    Q(chipType__in=[chipType]) |
                    Q(chipType__in=["", chipType]) & Q(applGroup__name=applicationGroupName)
                )
                possible_analysisArgs_objs = active_analysisArgs_objs.filter(qset).order_by("description")

                return possible_analysisArgs_objs
            else:
                ##logger.debug("possible_matches() #8...  best_match chipType=%s; " %(chipType))

                qset = (
                    Q(chipType__in=["", chipType])
                )
                possible_analysisArgs_objs = active_analysisArgs_objs.filter(qset).order_by("description")

                return possible_analysisArgs_objs

        return best_match

    def __unicode__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Analysis Args"


class RemoteAccount(models.Model):

    # A convenience label for this account to be shown in the TS UI
    account_label = models.CharField(max_length=64, default='Unnamed Account')

    # URL of the remote service
    remote_resource = models.CharField(max_length=2048)

    # User name for the account at the remote service
    user_name = models.CharField(max_length=255, blank=True, default='')

    # Access token for remote service (OAuth 2.0)
    access_token = models.CharField(max_length=2048, null=True, blank=True)

    # Refresh token for remote service (OAuth 2.0, optional)
    refresh_token = models.CharField(max_length=2048, null=True, blank=True)

    # Expire time for the access token
    token_expires = models.DateTimeField(null=True, blank=True)

    def has_access(self):
        return True


class SupportUpload(models.Model):

    account = models.ForeignKey('RemoteAccount')

    result = models.ForeignKey('Results', null=True, blank=True)

    # Basic date tracking
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    # This is the TS's processing state of the support upload
    local_status = models.CharField(max_length=255, default='', blank=True)
    local_message = models.CharField(max_length=2048, default='', blank=True)
    celery_task_id = models.CharField(max_length=60, default="", blank=True)

    # This references the file to be uploaded or which has been uploaded
    file = models.OneToOneField('FileMonitor', null=True, blank=True, on_delete=models.SET_NULL)

    # These models are to be collected from the user initiating the request
    contact_email = models.EmailField(default='')
    description = models.TextField(default='')
    user = models.ForeignKey(User)

    # These models are populated from the remote system
    ticket_id = models.CharField(max_length=255, default="", blank=True)
    ticket_status = models.CharField(max_length=255, default="", blank=True)
    ticket_message = models.CharField(max_length=2048, default="", blank=True)

    def delete(self):
        # delete the files from the filesystem as well as the database
        if self.file:
            self.file.delete()
        revoke(self.celery_task_id, terminate=True)
        super(SupportUpload, self).delete()
        return True


class IonMeshNode(models.Model):
    """This object will hold the requirements for remote ion mesh notes"""

    # the name of the node mesh computer
    hostname = models.CharField(max_length=128, unique=True, null=False)

    # this is the identifie
    system_id = models.CharField(max_length=37, unique=True)

    # this is the key to be used to make a request from them remotely
    apikey_remote = models.CharField(max_length=256, blank=True, null=True)

    # this is the key to be checked against for verifying incoming requests
    apikey_local = models.CharField(max_length=256, blank=True, null=True)

    share_plans = models.BooleanField(default=True)
    share_data = models.BooleanField(default=True)
    share_monitoring = models.BooleanField(default=True)

    @classmethod
    def create(cls, system_id):
        """Create a ion mesh node with keys"""

        node, existed = IonMeshNode.objects.get_or_create(system_id=system_id)
        if not node.apikey_local:
            node.apikey_local = _generate_key()
            node.save()

        return node


    @classmethod
    def canAccess(cls, system_id, api_key, data_type):
        """determines if a system with a given id can access information from this system"""

        try:
            # get the node from the database
            node = IonMeshNode.objects.get(system_id=system_id)

            # get the data access type
            data_access = False
            if data_type == 'admin':
                data_access = True
            elif data_type == 'data':
                data_access = node.share_data
            elif data_type == 'plans':
                data_access = node.share_plans
            elif data_type == 'monitoring':
                data_access = node.share_monitoring

            # return if access is allowed
            return data_access and node.apikey_local == api_key
        except IonMeshNode.DoesNotExist:
            return False


class SharedServer(models.Model):
    name = models.CharField(max_length=128, unique=True)
    address = models.CharField(max_length=128)
    username = models.CharField(max_length=64)
    password = models.CharField(max_length=64)
    active = models.BooleanField(default=True)
    comments = models.TextField(blank=True)

    def setup_session(self):
        import requests

        try:
            s = requests.Session()
            # convenient variables for communication
            s.api_url = 'http://%s/rundb/api/v1/' % self.address
            s.address = self.address
            s.server = self.name
            # call the account api, this also sets up a session cookie
            # TODO api_key authentication
            r = s.get(s.api_url + 'account/', auth=(self.username, self.password))
            # throw exception if unsuccessful
            r.raise_for_status()
        except (requests.exceptions.ConnectionError, requests.exceptions.TooManyRedirects):
            raise Exception('Connection Error: Torrent Server %s (%s) is unreachable' % (s.server, s.address))
        except requests.exceptions.HTTPError as e:
            raise Exception('HTTP Error: Unable to connect to Torrent Server %s (%s): %s' % (s.server, s.address, e))
        except Exception as e:
            raise Exception('Error: Unable to access Torrent Server %s (%s): %s' % (s.server, s.address, e))

        try:
            # get software version
            r = s.get(s.api_url + 'torrentsuite/version')
            version = r.json()['meta_version']
        except:
            msg = 'Error getting software version for Torrent Server %s (%s). ' % (s.server, s.address)
            raise Exception(msg + 'Only TSS 4.4 and above are supported for Plan Transfer')

        return s, version

    def clean(self):
        # if active server make sure we can set up communication
        if self.active:
            try:
                self.setup_session()
            except Exception as e:
                raise ValidationError(str(e))

    def __unicode__(self):
        return self.name

class PlanSession(models.Model):
    ''' Stores temporary step_helper session data during Run Planning '''
    session_key = models.CharField(max_length=64)
    plan_key = models.CharField(max_length=64)
    expire_date = models.DateTimeField(blank=True, null=True)
    _session_data = models.TextField(blank=True)

    PLAN_SESSION_EXPIRE = datetime.timedelta(hours=3)

    def get_data(self):
        data = None
        if self._session_data:
            decoded = base64.b64decode(self._session_data)
            data = PickleSerializer().loads(decoded)
        return data

    def set_data(self, data):
        serialized = PickleSerializer().dumps(data)
        self._session_data = base64.b64encode(serialized)
        self.save()

    def save(self, *args, **kwargs):
        self.expire_date = timezone.now() + self.PLAN_SESSION_EXPIRE
        super(PlanSession, self).save(*args, **kwargs)
