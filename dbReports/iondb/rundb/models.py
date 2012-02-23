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
from os import path
import datetime
import re
import os
import os.path

from django.core.exceptions import ValidationError

import iondb.settings

from django import forms

from django.conf import settings
from django.contrib import admin
from django.db import models
from django.db.models.query import QuerySet
from iondb.backup import devices
try:
    import json
except ImportError:
    import simplejson as json

from iondb.rundb import json_field

import shutil
import uuid
import random
import string
import logging

logger = logging.getLogger(__name__)

from django.core.urlresolvers import reverse
from django.contrib.auth.models import User
from django.db.models.signals import post_save, pre_delete, post_delete
from django.dispatch import receiver
from distutils.version import StrictVersion

# Create your models here.

class Experiment(models.Model):
    _CSV_METRICS = (('Sample', 'sample'),
                    ('Project', 'project'),
                    ('Library', 'library'),
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
        ('U', 'Unset'),          # (1) No action is pending
        ('S', 'Selected'),      # (2) Selected for action
        ('N', 'Notified'),      # (3) user has been notified of action pending
        ('A', 'Acknowledged'),  # (4) user acknowledges to proceed with action
        ('D', 'Disposed'),      # (5) Action has been completed
    )
    PRETTY_PRINT_RE = re.compile(r'R_(\d{4})_(\d{2})_(\d{2})_(\d{2})'
                                 '_(\d{2})_(\d{2})_')
    # raw data lives here absolute path prefix
    expDir = models.CharField(max_length=512)
    expName = models.CharField(max_length=128)
    pgmName = models.CharField(max_length=64)
    log = json_field.JSONField(blank=True)
    unique = models.CharField(max_length=512, unique=True)
    date = models.DateTimeField()
    storage_options = models.CharField(max_length=200, choices=STORAGE_CHOICES,
                                       default='A')
    user_ack = models.CharField(max_length=24, choices=ACK_CHOICES,default='U')
    project = models.CharField(max_length=64, blank=True, null=True)
    sample = models.CharField(max_length=64, blank=True, null=True)
    library = models.CharField(max_length=64, blank=True, null=True)
    notes = models.CharField(max_length=128, blank=True, null=True)
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
    flowsInOrder = models.CharField(max_length=512)
    star = models.BooleanField()
    ftpStatus = models.CharField(max_length=512, blank=True)
    libraryKey = models.CharField(max_length=64, blank=True)
    storageHost = models.CharField(max_length=128, blank=True, null=True)
    barcodeId = models.CharField(max_length=128, blank=True, null=True)
    reverse_primer = models.CharField(max_length=128, blank=True, null=True)
    rawdatastyle = models.CharField(max_length=24, blank=True, null=True, default='single')

    sequencekitname = models.CharField(max_length=512, blank=True, null=True)
    sequencekitbarcode = models.CharField(max_length=512, blank=True, null=True)
    librarykitname = models.CharField(max_length=512, blank=True, null=True)
    librarykitbarcode = models.CharField(max_length=512, blank=True, null=True)

    #add metadata
    metaData = json_field.JSONField(blank=True)

    def __unicode__(self): return self.expName

    def runtype(self):
        return self.log.get("runtype","")

    def pretty_print(self):
        nodate = self.PRETTY_PRINT_RE.sub("", self.expName)
        ret = " ".join(nodate.split('_')[1:]).strip()
        if not ret:
            return nodate
        return ret
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

    def available(self):
        try:
            backup = Backup.objects.get(backupName=self.expName)
        except:
            return False
        if backup.backupPath == 'DELETED':
            return 'Deleted'
        if backup.isBackedUp:
            return 'Archived'

    def save(self):
        """on save we need to sync up the log JSON and the other values that might have been set
        this was put in place primarily for the runtype field"""
        super(Experiment, self).save()

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
        elements = path.split('.')
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
                    ("Flows", 'processedCycles'),
                    #("Plugin Data", 'pluginStore')
                    )
    _ALIASES = {
        "report":"resultsName",
        #"experiment":"experiment",
        "date":"timeStamp",
        "status":"status",
        "Flows":"processedCycles",
        "q17mean":"best_metrics.Q17Mean",
        "q17mode":"best_metrics.Q17Mode",
        "systemsnr":"best_metrics.SysSNR",
        "q17reads":"best_metrics.Q17ReadCount",
        "keypass":"best_metrics.keypass",
        "cf":"best_metrics.CF"
        }
    TABLE_FIELDS = ("Report", "Status", "Flows",
                    "Lib Key Signal",
                     "Q17 Bases", "100 bp AQ17 Reads", "AQ17 Bases")
    PRETTY_FIELDS = TABLE_FIELDS
    experiment = models.ForeignKey(Experiment)
    resultsName = models.CharField(max_length=512)
    timeStamp = models.DateTimeField(auto_now_add=True, db_index=True)
    sffLink = models.CharField(max_length=512)    #absolute paths
    fastqLink = models.CharField(max_length=512)  #absolute paths
    reportLink = models.CharField(max_length=512)  #absolute paths
    status = models.CharField(max_length=64)
    tfSffLink = models.CharField(max_length=512)    #absolute paths
    tfFastq = models.CharField(max_length=512)    #absolute paths
    log = models.TextField(blank=True)
    analysisVersion = models.CharField(max_length=64)
    processedCycles = models.IntegerField()
    framesProcessed = models.IntegerField()
    timeToComplete = models.CharField(max_length=64)
    reportstorage = models.ForeignKey("ReportStorage", related_name="storage", blank=True, null=True)

    #metaData
    metaData = json_field.JSONField(blank=True)

    #a place for plugins to store information
    # NB: These two functions facilitate compatibility with earlier model, 
    # which had pluginStore and pluginState as members
    #pluginStore = json_field.JSONField(blank=True)
    def getPluginStore(self):
        pluginDict = {}
        for p in self.pluginresult_set.all():
            pluginDict[p.plugin.name] = p.store
        #if not pluginDict:
        #    return None
        return pluginDict
    #pluginState = json_field.JSONField(blank=True)
    def getPluginState(self):
        pluginDict = {}
        for p in self.pluginresult_set.all():
            pluginDict[p.plugin.name] = p.state
        #if not pluginDict:
        #    return None
        return pluginDict

    def planShortID(self):
        expLog = self.experiment.log
        plan = expLog.get("pending_run_short_id","")
        return plan

    def bamLink(self):
        """a method to used by the API to provide a link to the bam file"""
        reportStorage = self._findReportStorage()
        location = self.server_and_location()

        if reportStorage:
            return os.path.join(self.web_path(location) , self.experiment.expName + "_" + self.resultsName + ".bam")
        else:
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
        loc = Rig.objects.get(name=self.experiment.pgmName).location
        #server = FileServer.objects.get(location=loc)
        return loc

    def _findReportStorage(self):
        """
        Tries to determine the correct ReportStorage object by testing for
        a valid filesystem path using the reportLink path with the ReportStorage
        dirPath value.

        Algorithm for determining path from report link:
        strip off the first directory from report link and prepend the dirPath
        """
        storages = ReportStorage.objects.all()
        for storage in storages:
            tmpPath = self.reportLink.split('/')
            index = len(tmpPath) - 4
            linkstub = self.reportLink.split('/' + tmpPath[index])
            new_path = storage.dirPath + linkstub[1]
            if os.path.exists(new_path):
                return storage
        return None

    def web_root_path(self, location):
        """Returns filesystem path to Results directory"""
        basename = self._basename()
        if self.reportstorage == None:
            storage = self._findReportStorage ()
            if storage is not None:
                self.reportstorage = storage
                self.save()
        return path.join(self.reportstorage.dirPath, location.name, basename)

    def report_exist(self):
        """check to see if a report exists inside of the report path"""
        fs_path = self.get_report_path()
        #TODO: is this operation slowing down page loading?  on thousands of reports?
        return os.path.exists(fs_path)

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
        fs_path = self.reportstorage.dirPath + linkstub[1]
        return fs_path

    def get_report_dir(self):
        """Returns filesystem path to results directory"""
        fs_path = self.get_report_path()
        fs_path = path.split(fs_path)[0]
        return fs_path

    def web_path(self, location):
        basename = self._basename()
        if self.reportstorage == None:
            storage = self._findReportStorage()
            if storage is not None:
                self.reportstorage = storage
                self.save()
        webServerPath = self.reportstorage.webServerPath
        return path.join(webServerPath, location.name, basename)

    def __unicode__(self):
        return self.resultsName

    # TODO: Cycles -> flows hack, very temporary.
    @property
    def processedFlows(self):
        """This is an extremely intermediate hack, holding down the fort until
        cycles are removed from the model.
        """
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
            import traceback
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


class TFMetrics(models.Model, Lookup):
    _CSV_METRICS = (
        ("TF Name", "name"),
        ("Q10 Mean", "Q10Mean"),
        ("Q17 Mean", "Q17Mean"),
        ("Q10 Mode", "Q10Mode"),
        ("Q17 Mode", "Q17Mode"),
        ("System SNR", "SysSNR"),
        ("50Q10 Reads", "Q10ReadCount"),
        ("50Q17 Reads", "Q17ReadCount"),
        ("Keypass Reads", "keypass"),
        ("CF", "CF"),
        ("IE", "IE"),
        ("DR", "DR"),
        ("TF Key Peak Counts", 'aveKeyCount'),
        )
    _ALIASES = {
        "tfname":"name",
        "q17mean":"Q17Mean",
        "q17mode":"Q17Mode",
        "systemsnr":"SysSNR",
        "50q17reads":"Q17ReadCount",
        "keypassreads": "keypass",
        "CF":"cf",
        "IE":"ie",
        "DR":"dr",
        "tfkeypeakcounts":'aveKeyCount'
        }
    TABLE_FIELDS = ("TF Name", "Q17 Mean", "Q17 Mode",
                    "System SNR", "50Q17 Reads", "Keypass Reads",
                    "CF", "IE", "DR", " TF Key Peak Counts")
    report = models.ForeignKey(Results, db_index=True)
    name = models.CharField(max_length=128, db_index=True)
    matchMismatchHisto = models.TextField(blank=True)
    matchMismatchMean = models.FloatField()
    matchMismatchMode = models.FloatField()
    Q10Histo = models.TextField(blank=True)
    Q10Mean = models.FloatField()
    Q10Mode = models.FloatField()
    Q17Histo = models.TextField(blank=True)
    Q17Mean = models.FloatField()
    Q17Mode = models.FloatField()
    SysSNR = models.FloatField()
    HPSNR = models.TextField(blank=True)
    corrHPSNR = models.TextField(blank=True)
    HPAccuracy = models.TextField(blank=True)
    rawOverlap = models.TextField(blank=True)
    corOverlap = models.TextField(blank=True)
    hqReadCount = models.FloatField()
    aveHqReadCount = models.FloatField()
    Q10ReadCount = models.FloatField()
    aveQ10ReadCount = models.FloatField()
    Q17ReadCount = models.FloatField()
    aveQ17ReadCount = models.FloatField()
    sequence = models.CharField(max_length=512)#ambitious
    keypass = models.FloatField()
    preCorrSNR = models.FloatField()
    postCorrSNR = models.FloatField()
    ####CAFIE#####
    rawIonogram = models.TextField(blank=True)
    corrIonogram = models.TextField(blank=True)
    CF = models.FloatField()
    IE = models.FloatField()
    DR = models.FloatField()
    error = models.FloatField()
    number = models.FloatField()
    aveKeyCount = models.FloatField()
    def __unicode__(self):
        return "%s/%s" % (self.report, self.name)

    def get_csv_metrics(self):
        ret = []
        for metric in self._CSV_METRICS:
            ret.append((metric[0], getattr(self, metric[1], ' ')))
        print ret

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
        others = Location.objects.filter(defaultlocation=True)
        for other in others:
            other.defaultlocation = False
            super(Location, other).save(*args, **kwargs)
        super(Location, self).save(*args, **kwargs)


def on_result_created(sender, instance, created, **kwargs):
    experiment = result.experiment
    metrics = LibMetrics.objects.get(report=instance)
    #tasks.update_experiment_metrics.delay(experiment, instance, metrics)

#post_save.connect(on_result_created, sender=Results, dispatch_uid="create_result")


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


    def __unicode__(self): return self.name

class FileServer(models.Model):
    name = models.CharField(max_length=200)
    comments = models.TextField(blank=True)
    filesPrefix = models.CharField(max_length=200)
    location = models.ForeignKey(Location)
    def __unicode__(self): return self.name

class ReportStorage(models.Model):
    name = models.CharField(max_length=200)
    #path to webserver as http://localhost/results/
    webServerPath = models.CharField(max_length=200)
    dirPath = models.CharField(max_length=200)
    def __unicode__(self):
        return "%s (%s)" % (self.name, self.dirPath)

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
    report = models.ForeignKey(Results)
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
                    )
    report = models.ForeignKey(Results, db_index=True)
    sysSNR = models.FloatField()
    aveKeyCounts = models.FloatField()
    totalNumReads = models.IntegerField()
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

    def __unicode__(self):
        return "%s/%d" % (self.report, self.pk)
    class Meta:
        verbose_name_plural = "Lib Metrics"

class QualityMetrics(models.Model):
    """a place in the database to store the quality metrics from SFFSumary"""
    #make csv metrics lookup here
    report = models.ForeignKey(Results, db_index=True)
    q0_bases = models.BigIntegerField()
    q0_reads = models.IntegerField()
    q0_max_read_length = models.IntegerField()
    q0_mean_read_length = models.FloatField()
    q0_50bp_reads = models.IntegerField()
    q0_100bp_reads = models.IntegerField()
    q0_15bp_reads = models.IntegerField()
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
    isBackedUp = models.BooleanField()
    backupDate = models.DateTimeField()
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
    def __unicode__(self):
        return self.name

    def get_free_space(self):
        dev = devices.disk_report()
        for d in dev:
            if self.backup_directory == d.get_path():
                return d.get_free_space()

    def check_if_online(self):
        if path.exists(self.backup_directory):
            return True
        else:
            return False

class Chip(models.Model):
    name = models.CharField(max_length=128)
    slots = models.IntegerField()
    args = models.CharField(max_length=512, blank=True)

class GlobalConfig(models.Model):
    name = models.CharField(max_length=512)
    selected = models.BooleanField()
    plugin_folder = models.CharField(max_length=512, blank=True)
    default_command_line = models.CharField(max_length=512, blank=True)
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
    auto_archive_ack = models.BooleanField("Auto-Acknowledge Archive?", default=False)
    #sff triming options
    sfftrim = models.BooleanField("Disable SFF Trim?")
    sfftrim_args_help = "Note: The report input (--in-sff) and output (--out-sff) will be added automatically for all cases."
    sfftrim_args = models.CharField("Args to use for SFFTrim", max_length=500, blank=True, help_text=sfftrim_args_help)



    def get_default_command(self):
        return str(self.default_command_line)


@receiver(post_save, sender=GlobalConfig, dispatch_uid="save_globalconfig")
def on_save_config_sitename(sender, instance, created, **kwargs):
    """Very sneaky, we open the Default Report base template which the PHP
    file for the report renders itself inside of and find the name, replace it,
    and rewrite the thing.
    """
    with open("/opt/ion/iondb/templates/rundb/php_base.html", 'r+') as name:
        text = name.read()
        name.seek(0)
        # .*? is a non-greedy qualifier.
        # It will match the minimally satisfying text
        target = '<h1 id="sitename">.*?</h1>'
        replacement = '<h1 id="sitename">%s</h1>' % instance.site_name
        text = re.sub(target, replacement, text)
        target = '<title>.*?</title>'
        replacement = '<title>%s - Torrent Browser</title>' % instance.site_name
        name.write(re.sub(target, replacement, text))


class EmailAddress(models.Model):
    email = models.EmailField(blank=True)
    selected = models.BooleanField()
    class Meta:
        verbose_name_plural = "Email addresses"


class RunType(models.Model):
    runType = models.CharField(max_length=512)
    barcode = models.CharField(max_length=512, blank=True)
    description = models.TextField(blank=True)
    meta = json_field.JSONField(blank=True, null=True, default = "")

    def __unicode__(self):
        return self.runType

class Plugin(models.Model):
    name = models.CharField(max_length=512)
    version = models.CharField(max_length=256)
    date = models.DateTimeField(default=datetime.datetime.now())
    selected = models.BooleanField(default=False)
    path = models.CharField(max_length=512)
    project = models.CharField(max_length=512, blank=True, default="")
    sample = models.CharField(max_length=512, blank=True, default="")
    libraryName = models.CharField(max_length=512, blank=True, default="")
    chipType = models.CharField(max_length=512, blank=True, default="")
    autorun = models.BooleanField(default=False)
    config = json_field.JSONField(blank=True, null=True, default = "")
    #status
    status = json_field.JSONField(blank=True, null=True, default = "")

    # Store and mask inactive (uninstalled) plugins
    active = models.BooleanField(default=True)

    # Plugin Feed URL
    url = models.URLField(verify_exists=False, max_length=256, blank=True, default="")

    def isConfig(self):
        try:
            if os.path.exists(os.path.join(self.path, "config.html")):
                #provide a link to load the plugins html
                return "/rundb/plugininput/" + str(self.pk) + "/"
        except:
            return False

    def hasAbout(self):
        try:
            if os.path.exists(os.path.join(self.path, "about.html")):
                #provide a link to load the plugins html
                return "/rundb/plugininput/" + str(self.pk) + "/"
        except:
            return False

    def autorunMutable(self):
        """
        if the string AUTORUNDISABLE is in the lunch script
        don't allow the autorun settings to be changed on the config tab
        """

        try:
            for line in open(os.path.join(self.path, "launch.sh")):
                if line.startswith("#AUTORUNDISABLE"):
                    return False
        except:
            return True

        return True


    def __unicode__(self):
        return self.name

    # Help for comparing plugins by version number
    def versionGreater(self, other):
        return(StrictVersion(self.version) > StrictVersion(other.version))

    def installStatus(self):
        """this method helps us know if a plugin was installed sucessfully"""
        if self.status.get("result"):
            if self.status["result"] == "queued":
                return "queued"
        return self.status.get("installStatus", "installed" )


class PluginResult(models.Model):
    _CSV_METRICS = (
                    ("Plugin Data", 'store')
    )
    """ Many to Many mapping at the intersection of Results and Plugins """
    plugin = models.ForeignKey(Plugin)
    result = models.ForeignKey(Results)

    ALLOWED_STATES = (
        ('Completed', 'Completed'),
        ('Error', 'Error'),
        ('Started', 'Started'),
        ('Declined', 'Declined'),
        ('Unknown', 'Unknown'),
        ('Queued', 'Queued'),
    )
    state = models.CharField(max_length=20, choices=ALLOWED_STATES)
    store = json_field.JSONField(blank=True)

    def __unicode__(self):
        return "%s/%s" % (self.result, self.plugin)


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
    date = models.DateTimeField()
    version = models.CharField(max_length=100, blank=True)
    species = models.CharField(max_length=512, blank=True)
    source = models.CharField(max_length=512, blank=True)
    notes = models.TextField(blank=True)
    #needs a status for index generation process
    status = models.CharField(max_length=512, blank=True)
    index_version = models.CharField(max_length=512, blank=True)
    verbose_error = models.CharField(max_length=3000, blank=True)

    def delete(self):
        #delete the genome from the filesystem as well as the database

        if os.path.exists(self.reference_path):
            try:
                shutil.rmtree(self.reference_path)
            except OSError:
                return False

        super(ReferenceGenome, self).delete()
        return True

    def enable_genome(self):
        """this should be around to move the genome in a disabled dir or not"""

        #get the new path to move the reference to
        enabled_path = os.path.join(iondb.settings.TMAP_DIR, self.short_name)

        try:
            shutil.move(self.reference_path, enabled_path)
        except:
            return False

        return True

    def disable_genome(self):
        """this should be around to move the genome in a disabled dir or not"""
        #get the new path to move the reference to
        base_path = os.path.split(os.path.split(self.reference_path)[0])[0]
        disabled_path = os.path.join(base_path, "disabled", self.index_version, self.short_name)

        try:
            shutil.move(self.reference_path, disabled_path)
        except:
            return False

        return True

    def info_text(self):
        return os.path.join(self.reference_path , self.short_name + ".info.txt")

    def fastaOrig(self):
        """
        if there was a file named .orig then the fasta was autofixed.
        """
        orig = os.path.join(self.reference_path , self.short_name + ".orig")
        print orig
        return os.path.exists(orig)

    def __unicode__(self):
        return u'%s' % self.name

class ThreePrimeadapter(models.Model):
    name = models.CharField(max_length=256, blank=True)
    sequence = models.CharField(max_length=512, blank=True)
    description = models.CharField(max_length=1024, blank=True)
    qual_cutoff = models.IntegerField()
    qual_window = models.IntegerField()
    adapter_cutoff = models.IntegerField()

    class Meta:
        verbose_name_plural = "3' Adapters"

    def __unicode__(self):
        return u'%s' % self.name

class PlannedExperiment(models.Model):
    """
    Create a planned run to ease the pain on manually entry on the PGM
    """

    #plan name
    planName = models.CharField(max_length=512,blank=True,null=True)

    #Global uniq id for planned run
    planGUID = models.CharField(max_length=512,blank=True,null=True)

    #make a id for easy entry
    planShortID = models.CharField(max_length=5,blank=True,null=True)

    #was the plan already executed?
    planExecuted = models.BooleanField(default=False)

    #planStatus - Did the plan work?
    planStatus = models.CharField(max_length=512, blank=True)

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

    chipType = models.CharField(max_length=32,blank=True,null=True)
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

    #flow vs cycles ? do we need this?
    flows = models.IntegerField(blank=True,null=True)

    #AutoAnalysis - autoName string
    autoAnalyze = models.BooleanField()
    autoName = models.CharField(max_length=512, blank=True, null=True)

    preAnalysis = models.BooleanField()

    #RunType -- this is from a list of possible types
    runType = models.CharField(max_length=512, blank=True, null=True)

    #Library - should this be a text field?
    library = models.CharField(max_length=512, blank=True, null=True)

    #barcode
    barcodeId = models.CharField(max_length=256, blank=True, null=True)

    #adapter
    adapter = models.CharField(max_length=256, blank=True, null=True)

    #Project
    project = models.CharField(max_length=127, blank=True, null=True)

    #runname - name of the raw data directory
    runname = models.CharField(max_length=255, blank=True, null=True)


    #Sample
    sample = models.CharField(max_length=127, blank=True, null=True)

    #usernotes
    notes = models.CharField(max_length=255, blank=True, null=True)

    flowsInOrder = models.CharField(max_length=512, blank=True, null=True)
    libraryKey = models.CharField(max_length=64, blank=True,null=True)
    storageHost = models.CharField(max_length=128, blank=True, null=True)
    reverse_primer = models.CharField(max_length=128, blank=True, null=True)

    #bed file
    bedfile = models.CharField(max_length=1024,blank=True)
    regionfile = models.CharField(max_length=1024,blank=True)


    libkit = models.CharField(max_length=512, blank=True, null=True)

    variantfrequency = models.CharField(max_length=512, blank=True, null=True)

    STORAGE_CHOICES = (
        ('KI', 'Keep'),
        ('A', 'Archive Raw'),

        ('D', 'Delete Raw'),
        )

    storage_options = models.CharField(max_length=200, choices=STORAGE_CHOICES,
                                       default='A')

    #TODO: What is to be done about barcode experiments, where each barcode has different libs etc?

    def __unicode__(self):
        return self.planName

    def findShortID(self):
        """Search for a plan short ID that has not been used"""

        planShortID = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(5))

        if PlannedExperiment.objects.filter(planShortID=planShortID, planExecuted=False):
            self.findShortID()

        return planShortID


    def save(self):
        self.date = datetime.datetime.now()
        if not self.planShortID:
            self.planShortID = self.findShortID()
        if not self.planGUID:
            self.planGUID = str(uuid.uuid4())

        super(PlannedExperiment, self).save()


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
    user = models.ForeignKey(User, unique=True)

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

    def __unicode__(self):
        return u'profile: %s' % self.user.username


@receiver(post_save, sender=User, dispatch_uid="create_profile")
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


class SequencingKit(models.Model):
    name = models.CharField(max_length=512, blank=True)
    description = models.CharField(max_length=3024, blank=True)
    sap = models.CharField(max_length=7, blank=True)

    def __unicode__(self):
        return u'%s' % self.name

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

