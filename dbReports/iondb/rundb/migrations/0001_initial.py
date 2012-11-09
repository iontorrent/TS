# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# encoding: utf-8
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models

class Migration(SchemaMigration):

    def forwards(self, orm):
        
        # Adding model 'Experiment'
        db.create_table('rundb_experiment', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('expDir', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('expName', self.gf('django.db.models.fields.CharField')(max_length=128)),
            ('pgmName', self.gf('django.db.models.fields.CharField')(max_length=64)),
            ('log', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
            ('unique', self.gf('django.db.models.fields.CharField')(unique=True, max_length=512)),
            ('date', self.gf('django.db.models.fields.DateTimeField')()),
            ('storage_options', self.gf('django.db.models.fields.CharField')(default='A', max_length=200)),
            ('user_ack', self.gf('django.db.models.fields.CharField')(default='U', max_length=24)),
            ('project', self.gf('django.db.models.fields.CharField')(max_length=64, null=True, blank=True)),
            ('sample', self.gf('django.db.models.fields.CharField')(max_length=64, null=True, blank=True)),
            ('library', self.gf('django.db.models.fields.CharField')(max_length=64, null=True, blank=True)),
            ('notes', self.gf('django.db.models.fields.CharField')(max_length=128, null=True, blank=True)),
            ('chipBarcode', self.gf('django.db.models.fields.CharField')(max_length=64, blank=True)),
            ('seqKitBarcode', self.gf('django.db.models.fields.CharField')(max_length=64, blank=True)),
            ('reagentBarcode', self.gf('django.db.models.fields.CharField')(max_length=64, blank=True)),
            ('autoAnalyze', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('usePreBeadfind', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('chipType', self.gf('django.db.models.fields.CharField')(max_length=32)),
            ('cycles', self.gf('django.db.models.fields.IntegerField')()),
            ('flows', self.gf('django.db.models.fields.IntegerField')()),
            ('expCompInfo', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('baselineRun', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('flowsInOrder', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('star', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('ftpStatus', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('libraryKey', self.gf('django.db.models.fields.CharField')(max_length=64, blank=True)),
            ('storageHost', self.gf('django.db.models.fields.CharField')(max_length=128, null=True, blank=True)),
            ('barcodeId', self.gf('django.db.models.fields.CharField')(max_length=128, null=True, blank=True)),
            ('reverse_primer', self.gf('django.db.models.fields.CharField')(max_length=128, null=True, blank=True)),
            ('rawdatastyle', self.gf('django.db.models.fields.CharField')(default='single', max_length=24, null=True, blank=True)),
            ('sequencekitname', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('sequencekitbarcode', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('librarykitname', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('librarykitbarcode', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('reverselibrarykey', self.gf('django.db.models.fields.CharField')(max_length=64, null=True, blank=True)),
            ('reverse3primeadapter', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('forward3primeadapter', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('isReverseRun', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('metaData', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
        ))
        db.send_create_signal('rundb', ['Experiment'])

        # Adding model 'Results'
        db.create_table('rundb_results', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('experiment', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Experiment'])),
            ('resultsName', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('timeStamp', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, db_index=True, blank=True)),
            ('sffLink', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('fastqLink', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('reportLink', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('status', self.gf('django.db.models.fields.CharField')(max_length=64)),
            ('tfSffLink', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('tfFastq', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('log', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('analysisVersion', self.gf('django.db.models.fields.CharField')(max_length=64)),
            ('processedCycles', self.gf('django.db.models.fields.IntegerField')()),
            ('processedflows', self.gf('django.db.models.fields.IntegerField')()),
            ('framesProcessed', self.gf('django.db.models.fields.IntegerField')()),
            ('timeToComplete', self.gf('django.db.models.fields.CharField')(max_length=64)),
            ('reportstorage', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='storage', null=True, to=orm['rundb.ReportStorage'])),
            ('runid', self.gf('django.db.models.fields.CharField')(max_length=10, blank=True)),
            ('metaData', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
        ))
        db.send_create_signal('rundb', ['Results'])

        # Adding model 'TFMetrics'
        db.create_table('rundb_tfmetrics', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('report', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Results'])),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=128, db_index=True)),
            ('matchMismatchHisto', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('matchMismatchMean', self.gf('django.db.models.fields.FloatField')()),
            ('matchMismatchMode', self.gf('django.db.models.fields.FloatField')()),
            ('Q10Histo', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('Q10Mean', self.gf('django.db.models.fields.FloatField')()),
            ('Q10Mode', self.gf('django.db.models.fields.FloatField')()),
            ('Q17Histo', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('Q17Mean', self.gf('django.db.models.fields.FloatField')()),
            ('Q17Mode', self.gf('django.db.models.fields.FloatField')()),
            ('SysSNR', self.gf('django.db.models.fields.FloatField')()),
            ('HPSNR', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('corrHPSNR', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('HPAccuracy', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('rawOverlap', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('corOverlap', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('hqReadCount', self.gf('django.db.models.fields.FloatField')()),
            ('aveHqReadCount', self.gf('django.db.models.fields.FloatField')()),
            ('Q10ReadCount', self.gf('django.db.models.fields.FloatField')()),
            ('aveQ10ReadCount', self.gf('django.db.models.fields.FloatField')()),
            ('Q17ReadCount', self.gf('django.db.models.fields.FloatField')()),
            ('aveQ17ReadCount', self.gf('django.db.models.fields.FloatField')()),
            ('sequence', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('keypass', self.gf('django.db.models.fields.FloatField')()),
            ('preCorrSNR', self.gf('django.db.models.fields.FloatField')()),
            ('postCorrSNR', self.gf('django.db.models.fields.FloatField')()),
            ('rawIonogram', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('corrIonogram', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('CF', self.gf('django.db.models.fields.FloatField')()),
            ('IE', self.gf('django.db.models.fields.FloatField')()),
            ('DR', self.gf('django.db.models.fields.FloatField')()),
            ('error', self.gf('django.db.models.fields.FloatField')()),
            ('number', self.gf('django.db.models.fields.FloatField')()),
            ('aveKeyCount', self.gf('django.db.models.fields.FloatField')()),
        ))
        db.send_create_signal('rundb', ['TFMetrics'])

        # Adding model 'Location'
        db.create_table('rundb_location', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('comments', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('defaultlocation', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('rundb', ['Location'])

        # Adding model 'Rig'
        db.create_table('rundb_rig', (
            ('name', self.gf('django.db.models.fields.CharField')(max_length=200, primary_key=True)),
            ('location', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Location'])),
            ('comments', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('ftpserver', self.gf('django.db.models.fields.CharField')(default='192.168.201.1', max_length=128)),
            ('ftpusername', self.gf('django.db.models.fields.CharField')(default='ionguest', max_length=64)),
            ('ftppassword', self.gf('django.db.models.fields.CharField')(default='ionguest', max_length=64)),
            ('ftprootdir', self.gf('django.db.models.fields.CharField')(default='results', max_length=64)),
            ('updatehome', self.gf('django.db.models.fields.CharField')(default='192.168.201.1', max_length=256)),
            ('updateflag', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('serial', self.gf('django.db.models.fields.CharField')(max_length=24, null=True, blank=True)),
            ('state', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('version', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
            ('alarms', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
            ('last_init_date', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('last_clean_date', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('last_experiment', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
        ))
        db.send_create_signal('rundb', ['Rig'])

        # Adding model 'FileServer'
        db.create_table('rundb_fileserver', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('comments', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('filesPrefix', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('location', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Location'])),
        ))
        db.send_create_signal('rundb', ['FileServer'])

        # Adding model 'ReportStorage'
        db.create_table('rundb_reportstorage', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('webServerPath', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('dirPath', self.gf('django.db.models.fields.CharField')(max_length=200)),
        ))
        db.send_create_signal('rundb', ['ReportStorage'])

        # Adding model 'RunScript'
        db.create_table('rundb_runscript', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('script', self.gf('django.db.models.fields.TextField')(blank=True)),
        ))
        db.send_create_signal('rundb', ['RunScript'])

        # Adding model 'Cruncher'
        db.create_table('rundb_cruncher', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('prefix', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('location', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Location'])),
            ('comments', self.gf('django.db.models.fields.TextField')(blank=True)),
        ))
        db.send_create_signal('rundb', ['Cruncher'])

        # Adding model 'AnalysisMetrics'
        db.create_table('rundb_analysismetrics', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('report', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Results'])),
            ('libLive', self.gf('django.db.models.fields.IntegerField')()),
            ('libKp', self.gf('django.db.models.fields.IntegerField')()),
            ('libMix', self.gf('django.db.models.fields.IntegerField')()),
            ('libFinal', self.gf('django.db.models.fields.IntegerField')()),
            ('tfLive', self.gf('django.db.models.fields.IntegerField')()),
            ('tfKp', self.gf('django.db.models.fields.IntegerField')()),
            ('tfMix', self.gf('django.db.models.fields.IntegerField')()),
            ('tfFinal', self.gf('django.db.models.fields.IntegerField')()),
            ('empty', self.gf('django.db.models.fields.IntegerField')()),
            ('bead', self.gf('django.db.models.fields.IntegerField')()),
            ('live', self.gf('django.db.models.fields.IntegerField')()),
            ('dud', self.gf('django.db.models.fields.IntegerField')()),
            ('amb', self.gf('django.db.models.fields.IntegerField')()),
            ('tf', self.gf('django.db.models.fields.IntegerField')()),
            ('lib', self.gf('django.db.models.fields.IntegerField')()),
            ('pinned', self.gf('django.db.models.fields.IntegerField')()),
            ('ignored', self.gf('django.db.models.fields.IntegerField')()),
            ('excluded', self.gf('django.db.models.fields.IntegerField')()),
            ('washout', self.gf('django.db.models.fields.IntegerField')()),
            ('washout_dud', self.gf('django.db.models.fields.IntegerField')()),
            ('washout_ambiguous', self.gf('django.db.models.fields.IntegerField')()),
            ('washout_live', self.gf('django.db.models.fields.IntegerField')()),
            ('washout_test_fragment', self.gf('django.db.models.fields.IntegerField')()),
            ('washout_library', self.gf('django.db.models.fields.IntegerField')()),
            ('lib_pass_basecaller', self.gf('django.db.models.fields.IntegerField')()),
            ('lib_pass_cafie', self.gf('django.db.models.fields.IntegerField')()),
            ('keypass_all_beads', self.gf('django.db.models.fields.IntegerField')()),
            ('sysCF', self.gf('django.db.models.fields.FloatField')()),
            ('sysIE', self.gf('django.db.models.fields.FloatField')()),
            ('sysDR', self.gf('django.db.models.fields.FloatField')()),
        ))
        db.send_create_signal('rundb', ['AnalysisMetrics'])

        # Adding model 'LibMetrics'
        db.create_table('rundb_libmetrics', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('report', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Results'])),
            ('sysSNR', self.gf('django.db.models.fields.FloatField')()),
            ('aveKeyCounts', self.gf('django.db.models.fields.FloatField')()),
            ('totalNumReads', self.gf('django.db.models.fields.IntegerField')()),
            ('genomelength', self.gf('django.db.models.fields.IntegerField')()),
            ('rNumAlignments', self.gf('django.db.models.fields.IntegerField')()),
            ('rMeanAlignLen', self.gf('django.db.models.fields.IntegerField')()),
            ('rLongestAlign', self.gf('django.db.models.fields.IntegerField')()),
            ('rCoverage', self.gf('django.db.models.fields.FloatField')()),
            ('r50Q10', self.gf('django.db.models.fields.IntegerField')()),
            ('r100Q10', self.gf('django.db.models.fields.IntegerField')()),
            ('r200Q10', self.gf('django.db.models.fields.IntegerField')()),
            ('r50Q17', self.gf('django.db.models.fields.IntegerField')()),
            ('r100Q17', self.gf('django.db.models.fields.IntegerField')()),
            ('r200Q17', self.gf('django.db.models.fields.IntegerField')()),
            ('r50Q20', self.gf('django.db.models.fields.IntegerField')()),
            ('r100Q20', self.gf('django.db.models.fields.IntegerField')()),
            ('r200Q20', self.gf('django.db.models.fields.IntegerField')()),
            ('sNumAlignments', self.gf('django.db.models.fields.IntegerField')()),
            ('sMeanAlignLen', self.gf('django.db.models.fields.IntegerField')()),
            ('sLongestAlign', self.gf('django.db.models.fields.IntegerField')()),
            ('sCoverage', self.gf('django.db.models.fields.FloatField')()),
            ('s50Q10', self.gf('django.db.models.fields.IntegerField')()),
            ('s100Q10', self.gf('django.db.models.fields.IntegerField')()),
            ('s200Q10', self.gf('django.db.models.fields.IntegerField')()),
            ('s50Q17', self.gf('django.db.models.fields.IntegerField')()),
            ('s100Q17', self.gf('django.db.models.fields.IntegerField')()),
            ('s200Q17', self.gf('django.db.models.fields.IntegerField')()),
            ('s50Q20', self.gf('django.db.models.fields.IntegerField')()),
            ('s100Q20', self.gf('django.db.models.fields.IntegerField')()),
            ('s200Q20', self.gf('django.db.models.fields.IntegerField')()),
            ('q7_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('q7_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('q7_mapped_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q7_qscore_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q7_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('q7_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('i50Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i100Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i150Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i200Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i250Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i300Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i350Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i400Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i450Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i500Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i550Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i600Q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q10_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('q10_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('q10_mapped_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q10_qscore_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q10_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('q10_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('i50Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i100Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i150Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i200Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i250Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i300Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i350Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i400Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i450Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i500Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i550Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i600Q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q17_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('q17_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('q17_mapped_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q17_qscore_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q17_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('q17_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('i50Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i100Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i150Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i200Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i250Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i300Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i350Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i400Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i450Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i500Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i550Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i600Q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q20_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('q20_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('q20_mapped_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q20_qscore_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q20_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('q20_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('i50Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i100Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i150Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i200Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i250Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i300Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i350Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i400Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i450Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i500Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i550Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i600Q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q47_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('q47_mapped_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q47_qscore_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q47_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('q47_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('q47_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('i50Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i100Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i150Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i200Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i250Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i300Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i350Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i400Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i450Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i500Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i550Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('i600Q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('cf', self.gf('django.db.models.fields.FloatField')()),
            ('ie', self.gf('django.db.models.fields.FloatField')()),
            ('dr', self.gf('django.db.models.fields.FloatField')()),
            ('Genome_Version', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('Index_Version', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('align_sample', self.gf('django.db.models.fields.IntegerField')()),
            ('genome', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('genomesize', self.gf('django.db.models.fields.BigIntegerField')()),
            ('total_number_of_sampled_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q7_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q7_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q7_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q7_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_mapped_bases_in_q7_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('sampled_q7_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_50q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_100q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_200q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_300q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_400q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q10_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q10_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q10_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q10_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_mapped_bases_in_q10_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('sampled_q10_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_50q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_100q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_200q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_300q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_400q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q17_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q17_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q17_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q17_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_mapped_bases_in_q17_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('sampled_q17_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_50q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_100q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_200q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_300q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_400q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q20_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q20_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q20_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q20_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_mapped_bases_in_q20_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('sampled_q20_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_50q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_100q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_200q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_300q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_400q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q47_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q47_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('sampled_q47_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_q47_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_mapped_bases_in_q47_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('sampled_q47_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_50q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_100q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_200q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_300q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('sampled_400q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_from_number_of_sampled_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q7_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q7_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q7_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q7_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_mapped_bases_in_q7_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('extrapolated_q7_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_50q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_100q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_200q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_300q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_400q7_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q10_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q10_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q10_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q10_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_mapped_bases_in_q10_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('extrapolated_q10_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_50q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_100q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_200q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_300q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_400q10_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q17_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q17_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q17_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q17_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_mapped_bases_in_q17_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('extrapolated_q17_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_50q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_100q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_200q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_300q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_400q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q20_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q20_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q20_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q20_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_mapped_bases_in_q20_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('extrapolated_q20_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_50q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_100q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_200q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_300q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_400q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q47_coverage_percentage', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q47_mean_coverage_depth', self.gf('django.db.models.fields.FloatField')()),
            ('extrapolated_q47_alignments', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_q47_mean_alignment_length', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_mapped_bases_in_q47_alignments', self.gf('django.db.models.fields.BigIntegerField')()),
            ('extrapolated_q47_longest_alignment', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_50q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_100q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_200q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_300q47_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('extrapolated_400q47_reads', self.gf('django.db.models.fields.IntegerField')()),
        ))
        db.send_create_signal('rundb', ['LibMetrics'])

        # Adding model 'QualityMetrics'
        db.create_table('rundb_qualitymetrics', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('report', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Results'])),
            ('q0_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q0_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q0_max_read_length', self.gf('django.db.models.fields.IntegerField')()),
            ('q0_mean_read_length', self.gf('django.db.models.fields.FloatField')()),
            ('q0_50bp_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q0_100bp_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q0_15bp_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q17_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q17_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q17_max_read_length', self.gf('django.db.models.fields.IntegerField')()),
            ('q17_mean_read_length', self.gf('django.db.models.fields.FloatField')()),
            ('q17_50bp_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q17_100bp_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q17_150bp_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q20_bases', self.gf('django.db.models.fields.BigIntegerField')()),
            ('q20_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q20_max_read_length', self.gf('django.db.models.fields.FloatField')()),
            ('q20_mean_read_length', self.gf('django.db.models.fields.IntegerField')()),
            ('q20_50bp_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q20_100bp_reads', self.gf('django.db.models.fields.IntegerField')()),
            ('q20_150bp_reads', self.gf('django.db.models.fields.IntegerField')()),
        ))
        db.send_create_signal('rundb', ['QualityMetrics'])

        # Adding model 'Template'
        db.create_table('rundb_template', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=64)),
            ('sequence', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('key', self.gf('django.db.models.fields.CharField')(max_length=64)),
            ('comments', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('isofficial', self.gf('django.db.models.fields.BooleanField')(default=True)),
        ))
        db.send_create_signal('rundb', ['Template'])

        # Adding model 'Backup'
        db.create_table('rundb_backup', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('experiment', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Experiment'])),
            ('backupName', self.gf('django.db.models.fields.CharField')(unique=True, max_length=256)),
            ('isBackedUp', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('backupDate', self.gf('django.db.models.fields.DateTimeField')()),
            ('backupPath', self.gf('django.db.models.fields.CharField')(max_length=512)),
        ))
        db.send_create_signal('rundb', ['Backup'])

        # Adding model 'BackupConfig'
        db.create_table('rundb_backupconfig', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=64)),
            ('location', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Location'])),
            ('backup_directory', self.gf('django.db.models.fields.CharField')(default=None, max_length=256, blank=True)),
            ('backup_threshold', self.gf('django.db.models.fields.IntegerField')(blank=True)),
            ('number_to_backup', self.gf('django.db.models.fields.IntegerField')(blank=True)),
            ('grace_period', self.gf('django.db.models.fields.IntegerField')(default=72)),
            ('timeout', self.gf('django.db.models.fields.IntegerField')(blank=True)),
            ('bandwidth_limit', self.gf('django.db.models.fields.IntegerField')(blank=True)),
            ('status', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('online', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('comments', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('email', self.gf('django.db.models.fields.EmailField')(max_length=75, blank=True)),
        ))
        db.send_create_signal('rundb', ['BackupConfig'])

        # Adding model 'Chip'
        db.create_table('rundb_chip', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=128)),
            ('slots', self.gf('django.db.models.fields.IntegerField')()),
            ('args', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
        ))
        db.send_create_signal('rundb', ['Chip'])

        # Adding model 'GlobalConfig'
        db.create_table('rundb_globalconfig', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('selected', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('plugin_folder', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('default_command_line', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('basecallerargs', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('fasta_path', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('reference_path', self.gf('django.db.models.fields.CharField')(max_length=1000, blank=True)),
            ('records_to_display', self.gf('django.db.models.fields.IntegerField')(default=20, blank=True)),
            ('default_test_fragment_key', self.gf('django.db.models.fields.CharField')(max_length=50, blank=True)),
            ('default_library_key', self.gf('django.db.models.fields.CharField')(max_length=50, blank=True)),
            ('default_flow_order', self.gf('django.db.models.fields.CharField')(max_length=100, blank=True)),
            ('plugin_output_folder', self.gf('django.db.models.fields.CharField')(max_length=500, blank=True)),
            ('default_plugin_script', self.gf('django.db.models.fields.CharField')(max_length=500, blank=True)),
            ('web_root', self.gf('django.db.models.fields.CharField')(max_length=500, blank=True)),
            ('site_name', self.gf('django.db.models.fields.CharField')(max_length=500, blank=True)),
            ('default_storage_options', self.gf('django.db.models.fields.CharField')(default='D', max_length=500, blank=True)),
            ('auto_archive_ack', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('barcode_args', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
            ('enable_auto_pkg_dl', self.gf('django.db.models.fields.BooleanField')(default=True)),
            ('ts_update_status', self.gf('django.db.models.fields.CharField')(max_length=256, blank=True)),
        ))
        db.send_create_signal('rundb', ['GlobalConfig'])

        # Adding model 'EmailAddress'
        db.create_table('rundb_emailaddress', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('email', self.gf('django.db.models.fields.EmailField')(max_length=75, blank=True)),
            ('selected', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('rundb', ['EmailAddress'])

        # Adding model 'RunType'
        db.create_table('rundb_runtype', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('runType', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('barcode', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('description', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('meta', self.gf('django.db.models.fields.TextField')(default='', null=True, blank=True)),
        ))
        db.send_create_signal('rundb', ['RunType'])

        # Adding model 'Plugin'
        db.create_table('rundb_plugin', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('version', self.gf('django.db.models.fields.CharField')(max_length=256)),
            ('date', self.gf('django.db.models.fields.DateTimeField')(default=datetime.datetime(2012, 4, 2, 15, 40, 11, 877988))),
            ('selected', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('path', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('project', self.gf('django.db.models.fields.CharField')(default='', max_length=512, blank=True)),
            ('sample', self.gf('django.db.models.fields.CharField')(default='', max_length=512, blank=True)),
            ('libraryName', self.gf('django.db.models.fields.CharField')(default='', max_length=512, blank=True)),
            ('chipType', self.gf('django.db.models.fields.CharField')(default='', max_length=512, blank=True)),
            ('autorun', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('config', self.gf('django.db.models.fields.TextField')(default='', null=True, blank=True)),
            ('status', self.gf('django.db.models.fields.TextField')(default='', null=True, blank=True)),
            ('active', self.gf('django.db.models.fields.BooleanField')(default=True)),
            ('url', self.gf('django.db.models.fields.URLField')(default='', max_length=256, blank=True)),
        ))
        db.send_create_signal('rundb', ['Plugin'])

        # Adding model 'PluginResult'
        db.create_table('rundb_pluginresult', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('plugin', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Plugin'])),
            ('result', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Results'])),
            ('state', self.gf('django.db.models.fields.CharField')(max_length=20)),
            ('store', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
        ))
        db.send_create_signal('rundb', ['PluginResult'])

        # Adding model 'dnaBarcode'
        db.create_table('rundb_dnabarcode', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=128)),
            ('id_str', self.gf('django.db.models.fields.CharField')(max_length=128)),
            ('type', self.gf('django.db.models.fields.CharField')(max_length=64, blank=True)),
            ('sequence', self.gf('django.db.models.fields.CharField')(max_length=128)),
            ('length', self.gf('django.db.models.fields.IntegerField')(default=0, blank=True)),
            ('floworder', self.gf('django.db.models.fields.CharField')(default='', max_length=128, blank=True)),
            ('index', self.gf('django.db.models.fields.IntegerField')()),
            ('annotation', self.gf('django.db.models.fields.CharField')(default='', max_length=512, blank=True)),
            ('adapter', self.gf('django.db.models.fields.CharField')(default='', max_length=128, blank=True)),
            ('score_mode', self.gf('django.db.models.fields.IntegerField')(default=0, blank=True)),
            ('score_cutoff', self.gf('django.db.models.fields.FloatField')(default=0)),
        ))
        db.send_create_signal('rundb', ['dnaBarcode'])

        # Adding model 'ReferenceGenome'
        db.create_table('rundb_referencegenome', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('short_name', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('enabled', self.gf('django.db.models.fields.BooleanField')(default=True)),
            ('reference_path', self.gf('django.db.models.fields.CharField')(max_length=1024, blank=True)),
            ('date', self.gf('django.db.models.fields.DateTimeField')()),
            ('version', self.gf('django.db.models.fields.CharField')(max_length=100, blank=True)),
            ('species', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('source', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('notes', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('status', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('index_version', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('verbose_error', self.gf('django.db.models.fields.CharField')(max_length=3000, blank=True)),
        ))
        db.send_create_signal('rundb', ['ReferenceGenome'])

        # Adding model 'ThreePrimeadapter'
        db.create_table('rundb_threeprimeadapter', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('direction', self.gf('django.db.models.fields.CharField')(default='Forward', max_length=20)),
            ('name', self.gf('django.db.models.fields.CharField')(unique=True, max_length=256)),
            ('sequence', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=1024, blank=True)),
            ('qual_cutoff', self.gf('django.db.models.fields.IntegerField')()),
            ('qual_window', self.gf('django.db.models.fields.IntegerField')()),
            ('adapter_cutoff', self.gf('django.db.models.fields.IntegerField')()),
            ('isDefault', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('rundb', ['ThreePrimeadapter'])

        # Adding model 'PlannedExperiment'
        db.create_table('rundb_plannedexperiment', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('planName', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('planGUID', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('planShortID', self.gf('django.db.models.fields.CharField')(max_length=5, null=True, blank=True)),
            ('planExecuted', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('planStatus', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('username', self.gf('django.db.models.fields.CharField')(max_length=128, null=True, blank=True)),
            ('planPGM', self.gf('django.db.models.fields.CharField')(max_length=128, null=True, blank=True)),
            ('date', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('planExecutedDate', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('metaData', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
            ('chipType', self.gf('django.db.models.fields.CharField')(max_length=32, null=True, blank=True)),
            ('chipBarcode', self.gf('django.db.models.fields.CharField')(max_length=64, null=True, blank=True)),
            ('seqKitBarcode', self.gf('django.db.models.fields.CharField')(max_length=64, null=True, blank=True)),
            ('expName', self.gf('django.db.models.fields.CharField')(max_length=128, blank=True)),
            ('usePreBeadfind', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('usePostBeadfind', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('cycles', self.gf('django.db.models.fields.IntegerField')(null=True, blank=True)),
            ('flows', self.gf('django.db.models.fields.IntegerField')(null=True, blank=True)),
            ('autoAnalyze', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('autoName', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('preAnalysis', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('runType', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('library', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('barcodeId', self.gf('django.db.models.fields.CharField')(max_length=256, null=True, blank=True)),
            ('adapter', self.gf('django.db.models.fields.CharField')(max_length=256, null=True, blank=True)),
            ('project', self.gf('django.db.models.fields.CharField')(max_length=127, null=True, blank=True)),
            ('runname', self.gf('django.db.models.fields.CharField')(max_length=255, null=True, blank=True)),
            ('sample', self.gf('django.db.models.fields.CharField')(max_length=127, null=True, blank=True)),
            ('notes', self.gf('django.db.models.fields.CharField')(max_length=255, null=True, blank=True)),
            ('flowsInOrder', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('libraryKey', self.gf('django.db.models.fields.CharField')(max_length=64, null=True, blank=True)),
            ('storageHost', self.gf('django.db.models.fields.CharField')(max_length=128, null=True, blank=True)),
            ('reverse_primer', self.gf('django.db.models.fields.CharField')(max_length=128, null=True, blank=True)),
            ('bedfile', self.gf('django.db.models.fields.CharField')(max_length=1024, blank=True)),
            ('regionfile', self.gf('django.db.models.fields.CharField')(max_length=1024, blank=True)),
            ('irworkflow', self.gf('django.db.models.fields.CharField')(max_length=1024, blank=True)),
            ('libkit', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('variantfrequency', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('storage_options', self.gf('django.db.models.fields.CharField')(default='A', max_length=200)),
            ('reverselibrarykey', self.gf('django.db.models.fields.CharField')(max_length=64, null=True, blank=True)),
            ('reverse3primeadapter', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('forward3primeadapter', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('isReverseRun', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('librarykitname', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
            ('sequencekitname', self.gf('django.db.models.fields.CharField')(max_length=512, null=True, blank=True)),
        ))
        db.send_create_signal('rundb', ['PlannedExperiment'])

        # Adding model 'Publisher'
        db.create_table('rundb_publisher', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(unique=True, max_length=200)),
            ('version', self.gf('django.db.models.fields.CharField')(max_length=256)),
            ('date', self.gf('django.db.models.fields.DateTimeField')()),
            ('path', self.gf('django.db.models.fields.CharField')(max_length=512)),
            ('global_meta', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
        ))
        db.send_create_signal('rundb', ['Publisher'])

        # Adding model 'ContentUpload'
        db.create_table('rundb_contentupload', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('file_path', self.gf('django.db.models.fields.CharField')(max_length=255)),
            ('status', self.gf('django.db.models.fields.CharField')(max_length=255, blank=True)),
            ('meta', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
            ('publisher', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.Publisher'], null=True)),
        ))
        db.send_create_signal('rundb', ['ContentUpload'])

        # Adding model 'Content'
        db.create_table('rundb_content', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('publisher', self.gf('django.db.models.fields.related.ForeignKey')(related_name='contents', to=orm['rundb.Publisher'])),
            ('contentupload', self.gf('django.db.models.fields.related.ForeignKey')(related_name='contents', to=orm['rundb.ContentUpload'])),
            ('file', self.gf('django.db.models.fields.CharField')(max_length=255)),
            ('path', self.gf('django.db.models.fields.CharField')(max_length=255)),
            ('meta', self.gf('django.db.models.fields.TextField')(default='{}', blank=True)),
        ))
        db.send_create_signal('rundb', ['Content'])

        # Adding model 'UserEventLog'
        db.create_table('rundb_usereventlog', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('text', self.gf('django.db.models.fields.TextField')(blank=True)),
            ('timeStamp', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('upload', self.gf('django.db.models.fields.related.ForeignKey')(related_name='logs', to=orm['rundb.ContentUpload'])),
        ))
        db.send_create_signal('rundb', ['UserEventLog'])

        # Adding model 'UserProfile'
        db.create_table('rundb_userprofile', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['auth.User'], unique=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=93)),
            ('phone_number', self.gf('django.db.models.fields.CharField')(default='', max_length=256, blank=True)),
            ('title', self.gf('django.db.models.fields.CharField')(default='user', max_length=256)),
            ('note', self.gf('django.db.models.fields.TextField')(default='', blank=True)),
        ))
        db.send_create_signal('rundb', ['UserProfile'])

        # Adding model 'SequencingKit'
        db.create_table('rundb_sequencingkit', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=3024, blank=True)),
            ('sap', self.gf('django.db.models.fields.CharField')(max_length=7, blank=True)),
        ))
        db.send_create_signal('rundb', ['SequencingKit'])

        # Adding model 'LibraryKit'
        db.create_table('rundb_librarykit', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=3024, blank=True)),
            ('sap', self.gf('django.db.models.fields.CharField')(max_length=7, blank=True)),
        ))
        db.send_create_signal('rundb', ['LibraryKit'])

        # Adding model 'VariantFrequencies'
        db.create_table('rundb_variantfrequencies', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=512, blank=True)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=3024, blank=True)),
        ))
        db.send_create_signal('rundb', ['VariantFrequencies'])

        # Adding model 'KitInfo'
        db.create_table('rundb_kitinfo', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('kitType', self.gf('django.db.models.fields.CharField')(max_length=20)),
            ('name', self.gf('django.db.models.fields.CharField')(unique=True, max_length=512)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=3024, blank=True)),
            ('flowCount', self.gf('django.db.models.fields.PositiveIntegerField')()),
        ))
        db.send_create_signal('rundb', ['KitInfo'])

        # Adding model 'KitPart'
        db.create_table('rundb_kitpart', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('kit', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['rundb.KitInfo'])),
            ('barcode', self.gf('django.db.models.fields.CharField')(unique=True, max_length=7)),
        ))
        db.send_create_signal('rundb', ['KitPart'])

        # Adding model 'LibraryKey'
        db.create_table('rundb_librarykey', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('direction', self.gf('django.db.models.fields.CharField')(default='Forward', max_length=20)),
            ('name', self.gf('django.db.models.fields.CharField')(unique=True, max_length=256)),
            ('sequence', self.gf('django.db.models.fields.CharField')(max_length=64)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=1024, blank=True)),
            ('isDefault', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('rundb', ['LibraryKey'])

        # Adding model 'Message'
        db.create_table('rundb_message', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('body', self.gf('django.db.models.fields.TextField')(default='', blank=True)),
            ('level', self.gf('django.db.models.fields.IntegerField')(default=20)),
            ('route', self.gf('django.db.models.fields.TextField')(default='', blank=True)),
            ('expires', self.gf('django.db.models.fields.TextField')(default='read', blank=True)),
            ('tags', self.gf('django.db.models.fields.TextField')(default='', blank=True)),
            ('status', self.gf('django.db.models.fields.TextField')(default='unread', blank=True)),
            ('time', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
        ))
        db.send_create_signal('rundb', ['Message'])


    def backwards(self, orm):
        
        # Deleting model 'Experiment'
        db.delete_table('rundb_experiment')

        # Deleting model 'Results'
        db.delete_table('rundb_results')

        # Deleting model 'TFMetrics'
        db.delete_table('rundb_tfmetrics')

        # Deleting model 'Location'
        db.delete_table('rundb_location')

        # Deleting model 'Rig'
        db.delete_table('rundb_rig')

        # Deleting model 'FileServer'
        db.delete_table('rundb_fileserver')

        # Deleting model 'ReportStorage'
        db.delete_table('rundb_reportstorage')

        # Deleting model 'RunScript'
        db.delete_table('rundb_runscript')

        # Deleting model 'Cruncher'
        db.delete_table('rundb_cruncher')

        # Deleting model 'AnalysisMetrics'
        db.delete_table('rundb_analysismetrics')

        # Deleting model 'LibMetrics'
        db.delete_table('rundb_libmetrics')

        # Deleting model 'QualityMetrics'
        db.delete_table('rundb_qualitymetrics')

        # Deleting model 'Template'
        db.delete_table('rundb_template')

        # Deleting model 'Backup'
        db.delete_table('rundb_backup')

        # Deleting model 'BackupConfig'
        db.delete_table('rundb_backupconfig')

        # Deleting model 'Chip'
        db.delete_table('rundb_chip')

        # Deleting model 'GlobalConfig'
        db.delete_table('rundb_globalconfig')

        # Deleting model 'EmailAddress'
        db.delete_table('rundb_emailaddress')

        # Deleting model 'RunType'
        db.delete_table('rundb_runtype')

        # Deleting model 'Plugin'
        db.delete_table('rundb_plugin')

        # Deleting model 'PluginResult'
        db.delete_table('rundb_pluginresult')

        # Deleting model 'dnaBarcode'
        db.delete_table('rundb_dnabarcode')

        # Deleting model 'ReferenceGenome'
        db.delete_table('rundb_referencegenome')

        # Deleting model 'ThreePrimeadapter'
        db.delete_table('rundb_threeprimeadapter')

        # Deleting model 'PlannedExperiment'
        db.delete_table('rundb_plannedexperiment')

        # Deleting model 'Publisher'
        db.delete_table('rundb_publisher')

        # Deleting model 'ContentUpload'
        db.delete_table('rundb_contentupload')

        # Deleting model 'Content'
        db.delete_table('rundb_content')

        # Deleting model 'UserEventLog'
        db.delete_table('rundb_usereventlog')

        # Deleting model 'UserProfile'
        db.delete_table('rundb_userprofile')

        # Deleting model 'SequencingKit'
        db.delete_table('rundb_sequencingkit')

        # Deleting model 'LibraryKit'
        db.delete_table('rundb_librarykit')

        # Deleting model 'VariantFrequencies'
        db.delete_table('rundb_variantfrequencies')

        # Deleting model 'KitInfo'
        db.delete_table('rundb_kitinfo')

        # Deleting model 'KitPart'
        db.delete_table('rundb_kitpart')

        # Deleting model 'LibraryKey'
        db.delete_table('rundb_librarykey')

        # Deleting model 'Message'
        db.delete_table('rundb_message')


    models = {
        'auth.group': {
            'Meta': {'object_name': 'Group'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '80'}),
            'permissions': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['auth.Permission']", 'symmetrical': 'False', 'blank': 'True'})
        },
        'auth.permission': {
            'Meta': {'ordering': "('content_type__app_label', 'content_type__model', 'codename')", 'unique_together': "(('content_type', 'codename'),)", 'object_name': 'Permission'},
            'codename': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'content_type': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['contenttypes.ContentType']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '50'})
        },
        'auth.user': {
            'Meta': {'object_name': 'User'},
            'date_joined': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime.now'}),
            'email': ('django.db.models.fields.EmailField', [], {'max_length': '75', 'blank': 'True'}),
            'first_name': ('django.db.models.fields.CharField', [], {'max_length': '30', 'blank': 'True'}),
            'groups': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['auth.Group']", 'symmetrical': 'False', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'is_active': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'is_staff': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'is_superuser': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'last_login': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime.now'}),
            'last_name': ('django.db.models.fields.CharField', [], {'max_length': '30', 'blank': 'True'}),
            'password': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'user_permissions': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['auth.Permission']", 'symmetrical': 'False', 'blank': 'True'}),
            'username': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '30'})
        },
        'contenttypes.contenttype': {
            'Meta': {'ordering': "('name',)", 'unique_together': "(('app_label', 'model'),)", 'object_name': 'ContentType', 'db_table': "'django_content_type'"},
            'app_label': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'model': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '100'})
        },
        'rundb.analysismetrics': {
            'Meta': {'object_name': 'AnalysisMetrics'},
            'amb': ('django.db.models.fields.IntegerField', [], {}),
            'bead': ('django.db.models.fields.IntegerField', [], {}),
            'dud': ('django.db.models.fields.IntegerField', [], {}),
            'empty': ('django.db.models.fields.IntegerField', [], {}),
            'excluded': ('django.db.models.fields.IntegerField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'ignored': ('django.db.models.fields.IntegerField', [], {}),
            'keypass_all_beads': ('django.db.models.fields.IntegerField', [], {}),
            'lib': ('django.db.models.fields.IntegerField', [], {}),
            'libFinal': ('django.db.models.fields.IntegerField', [], {}),
            'libKp': ('django.db.models.fields.IntegerField', [], {}),
            'libLive': ('django.db.models.fields.IntegerField', [], {}),
            'libMix': ('django.db.models.fields.IntegerField', [], {}),
            'lib_pass_basecaller': ('django.db.models.fields.IntegerField', [], {}),
            'lib_pass_cafie': ('django.db.models.fields.IntegerField', [], {}),
            'live': ('django.db.models.fields.IntegerField', [], {}),
            'pinned': ('django.db.models.fields.IntegerField', [], {}),
            'report': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Results']"}),
            'sysCF': ('django.db.models.fields.FloatField', [], {}),
            'sysDR': ('django.db.models.fields.FloatField', [], {}),
            'sysIE': ('django.db.models.fields.FloatField', [], {}),
            'tf': ('django.db.models.fields.IntegerField', [], {}),
            'tfFinal': ('django.db.models.fields.IntegerField', [], {}),
            'tfKp': ('django.db.models.fields.IntegerField', [], {}),
            'tfLive': ('django.db.models.fields.IntegerField', [], {}),
            'tfMix': ('django.db.models.fields.IntegerField', [], {}),
            'washout': ('django.db.models.fields.IntegerField', [], {}),
            'washout_ambiguous': ('django.db.models.fields.IntegerField', [], {}),
            'washout_dud': ('django.db.models.fields.IntegerField', [], {}),
            'washout_library': ('django.db.models.fields.IntegerField', [], {}),
            'washout_live': ('django.db.models.fields.IntegerField', [], {}),
            'washout_test_fragment': ('django.db.models.fields.IntegerField', [], {})
        },
        'rundb.backup': {
            'Meta': {'object_name': 'Backup'},
            'backupDate': ('django.db.models.fields.DateTimeField', [], {}),
            'backupName': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '256'}),
            'backupPath': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'experiment': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Experiment']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isBackedUp': ('django.db.models.fields.BooleanField', [], {'default': 'False'})
        },
        'rundb.backupconfig': {
            'Meta': {'object_name': 'BackupConfig'},
            'backup_directory': ('django.db.models.fields.CharField', [], {'default': 'None', 'max_length': '256', 'blank': 'True'}),
            'backup_threshold': ('django.db.models.fields.IntegerField', [], {'blank': 'True'}),
            'bandwidth_limit': ('django.db.models.fields.IntegerField', [], {'blank': 'True'}),
            'comments': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'email': ('django.db.models.fields.EmailField', [], {'max_length': '75', 'blank': 'True'}),
            'grace_period': ('django.db.models.fields.IntegerField', [], {'default': '72'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'location': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Location']"}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '64'}),
            'number_to_backup': ('django.db.models.fields.IntegerField', [], {'blank': 'True'}),
            'online': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'status': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'timeout': ('django.db.models.fields.IntegerField', [], {'blank': 'True'})
        },
        'rundb.chip': {
            'Meta': {'object_name': 'Chip'},
            'args': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'slots': ('django.db.models.fields.IntegerField', [], {})
        },
        'rundb.content': {
            'Meta': {'object_name': 'Content'},
            'contentupload': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'contents'", 'to': "orm['rundb.ContentUpload']"}),
            'file': ('django.db.models.fields.CharField', [], {'max_length': '255'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'meta': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'path': ('django.db.models.fields.CharField', [], {'max_length': '255'}),
            'publisher': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'contents'", 'to': "orm['rundb.Publisher']"})
        },
        'rundb.contentupload': {
            'Meta': {'object_name': 'ContentUpload'},
            'file_path': ('django.db.models.fields.CharField', [], {'max_length': '255'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'meta': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'publisher': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Publisher']", 'null': 'True'}),
            'status': ('django.db.models.fields.CharField', [], {'max_length': '255', 'blank': 'True'})
        },
        'rundb.cruncher': {
            'Meta': {'object_name': 'Cruncher'},
            'comments': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'location': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Location']"}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'prefix': ('django.db.models.fields.CharField', [], {'max_length': '512'})
        },
        'rundb.dnabarcode': {
            'Meta': {'object_name': 'dnaBarcode'},
            'adapter': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '128', 'blank': 'True'}),
            'annotation': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'floworder': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '128', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'id_str': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'index': ('django.db.models.fields.IntegerField', [], {}),
            'length': ('django.db.models.fields.IntegerField', [], {'default': '0', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'score_cutoff': ('django.db.models.fields.FloatField', [], {'default': '0'}),
            'score_mode': ('django.db.models.fields.IntegerField', [], {'default': '0', 'blank': 'True'}),
            'sequence': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'type': ('django.db.models.fields.CharField', [], {'max_length': '64', 'blank': 'True'})
        },
        'rundb.emailaddress': {
            'Meta': {'object_name': 'EmailAddress'},
            'email': ('django.db.models.fields.EmailField', [], {'max_length': '75', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'selected': ('django.db.models.fields.BooleanField', [], {'default': 'False'})
        },
        'rundb.experiment': {
            'Meta': {'object_name': 'Experiment'},
            'autoAnalyze': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'barcodeId': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'baselineRun': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'chipBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'blank': 'True'}),
            'chipType': ('django.db.models.fields.CharField', [], {'max_length': '32'}),
            'cycles': ('django.db.models.fields.IntegerField', [], {}),
            'date': ('django.db.models.fields.DateTimeField', [], {}),
            'expCompInfo': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'expDir': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'expName': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'flows': ('django.db.models.fields.IntegerField', [], {}),
            'flowsInOrder': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'forward3primeadapter': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'ftpStatus': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isReverseRun': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'library': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'libraryKey': ('django.db.models.fields.CharField', [], {'max_length': '64', 'blank': 'True'}),
            'librarykitbarcode': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'librarykitname': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'log': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'metaData': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'notes': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'pgmName': ('django.db.models.fields.CharField', [], {'max_length': '64'}),
            'project': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'rawdatastyle': ('django.db.models.fields.CharField', [], {'default': "'single'", 'max_length': '24', 'null': 'True', 'blank': 'True'}),
            'reagentBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'blank': 'True'}),
            'reverse3primeadapter': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'reverse_primer': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'reverselibrarykey': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'sample': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'seqKitBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'blank': 'True'}),
            'sequencekitbarcode': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'sequencekitname': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'star': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'storageHost': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'storage_options': ('django.db.models.fields.CharField', [], {'default': "'A'", 'max_length': '200'}),
            'unique': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '512'}),
            'usePreBeadfind': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'user_ack': ('django.db.models.fields.CharField', [], {'default': "'U'", 'max_length': '24'})
        },
        'rundb.fileserver': {
            'Meta': {'object_name': 'FileServer'},
            'comments': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'filesPrefix': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'location': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Location']"}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'rundb.globalconfig': {
            'Meta': {'object_name': 'GlobalConfig'},
            'auto_archive_ack': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'barcode_args': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'basecallerargs': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'default_command_line': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'default_flow_order': ('django.db.models.fields.CharField', [], {'max_length': '100', 'blank': 'True'}),
            'default_library_key': ('django.db.models.fields.CharField', [], {'max_length': '50', 'blank': 'True'}),
            'default_plugin_script': ('django.db.models.fields.CharField', [], {'max_length': '500', 'blank': 'True'}),
            'default_storage_options': ('django.db.models.fields.CharField', [], {'default': "'D'", 'max_length': '500', 'blank': 'True'}),
            'default_test_fragment_key': ('django.db.models.fields.CharField', [], {'max_length': '50', 'blank': 'True'}),
            'enable_auto_pkg_dl': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'fasta_path': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'plugin_folder': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'plugin_output_folder': ('django.db.models.fields.CharField', [], {'max_length': '500', 'blank': 'True'}),
            'records_to_display': ('django.db.models.fields.IntegerField', [], {'default': '20', 'blank': 'True'}),
            'reference_path': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'blank': 'True'}),
            'selected': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'site_name': ('django.db.models.fields.CharField', [], {'max_length': '500', 'blank': 'True'}),
            'ts_update_status': ('django.db.models.fields.CharField', [], {'max_length': '256', 'blank': 'True'}),
            'web_root': ('django.db.models.fields.CharField', [], {'max_length': '500', 'blank': 'True'})
        },
        'rundb.kitinfo': {
            'Meta': {'object_name': 'KitInfo'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '3024', 'blank': 'True'}),
            'flowCount': ('django.db.models.fields.PositiveIntegerField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'kitType': ('django.db.models.fields.CharField', [], {'max_length': '20'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '512'})
        },
        'rundb.kitpart': {
            'Meta': {'object_name': 'KitPart'},
            'barcode': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '7'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'kit': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.KitInfo']"})
        },
        'rundb.libmetrics': {
            'Genome_Version': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'Index_Version': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'Meta': {'object_name': 'LibMetrics'},
            'align_sample': ('django.db.models.fields.IntegerField', [], {}),
            'aveKeyCounts': ('django.db.models.fields.FloatField', [], {}),
            'cf': ('django.db.models.fields.FloatField', [], {}),
            'dr': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_100q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_100q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_100q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_100q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_100q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_200q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_200q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_200q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_200q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_200q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_300q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_300q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_300q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_300q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_300q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_400q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_400q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_400q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_400q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_400q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_50q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_50q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_50q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_50q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_50q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_from_number_of_sampled_reads': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_mapped_bases_in_q10_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'extrapolated_mapped_bases_in_q17_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'extrapolated_mapped_bases_in_q20_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'extrapolated_mapped_bases_in_q47_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'extrapolated_mapped_bases_in_q7_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'extrapolated_q10_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q10_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_q10_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q10_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q10_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_q17_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q17_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_q17_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q17_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q17_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_q20_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q20_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_q20_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q20_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q20_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_q47_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q47_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_q47_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q47_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q47_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_q7_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q7_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'extrapolated_q7_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q7_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'extrapolated_q7_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'genome': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'genomelength': ('django.db.models.fields.IntegerField', [], {}),
            'genomesize': ('django.db.models.fields.BigIntegerField', [], {}),
            'i100Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i100Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i100Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i100Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i100Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i150Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i150Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i150Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i150Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i150Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i200Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i200Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i200Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i200Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i200Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i250Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i250Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i250Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i250Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i250Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i300Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i300Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i300Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i300Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i300Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i350Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i350Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i350Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i350Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i350Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i400Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i400Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i400Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i400Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i400Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i450Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i450Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i450Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i450Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i450Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i500Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i500Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i500Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i500Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i500Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i50Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i50Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i50Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i50Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i50Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i550Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i550Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i550Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i550Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i550Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i600Q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i600Q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i600Q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i600Q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'i600Q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'ie': ('django.db.models.fields.FloatField', [], {}),
            'q10_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'q10_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'q10_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'q10_mapped_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q10_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'q10_qscore_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q17_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'q17_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'q17_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'q17_mapped_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q17_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'q17_qscore_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q20_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'q20_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'q20_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'q20_mapped_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q20_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'q20_qscore_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q47_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'q47_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'q47_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'q47_mapped_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q47_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'q47_qscore_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q7_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'q7_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'q7_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'q7_mapped_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q7_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'q7_qscore_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'r100Q10': ('django.db.models.fields.IntegerField', [], {}),
            'r100Q17': ('django.db.models.fields.IntegerField', [], {}),
            'r100Q20': ('django.db.models.fields.IntegerField', [], {}),
            'r200Q10': ('django.db.models.fields.IntegerField', [], {}),
            'r200Q17': ('django.db.models.fields.IntegerField', [], {}),
            'r200Q20': ('django.db.models.fields.IntegerField', [], {}),
            'r50Q10': ('django.db.models.fields.IntegerField', [], {}),
            'r50Q17': ('django.db.models.fields.IntegerField', [], {}),
            'r50Q20': ('django.db.models.fields.IntegerField', [], {}),
            'rCoverage': ('django.db.models.fields.FloatField', [], {}),
            'rLongestAlign': ('django.db.models.fields.IntegerField', [], {}),
            'rMeanAlignLen': ('django.db.models.fields.IntegerField', [], {}),
            'rNumAlignments': ('django.db.models.fields.IntegerField', [], {}),
            'report': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Results']"}),
            's100Q10': ('django.db.models.fields.IntegerField', [], {}),
            's100Q17': ('django.db.models.fields.IntegerField', [], {}),
            's100Q20': ('django.db.models.fields.IntegerField', [], {}),
            's200Q10': ('django.db.models.fields.IntegerField', [], {}),
            's200Q17': ('django.db.models.fields.IntegerField', [], {}),
            's200Q20': ('django.db.models.fields.IntegerField', [], {}),
            's50Q10': ('django.db.models.fields.IntegerField', [], {}),
            's50Q17': ('django.db.models.fields.IntegerField', [], {}),
            's50Q20': ('django.db.models.fields.IntegerField', [], {}),
            'sCoverage': ('django.db.models.fields.FloatField', [], {}),
            'sLongestAlign': ('django.db.models.fields.IntegerField', [], {}),
            'sMeanAlignLen': ('django.db.models.fields.IntegerField', [], {}),
            'sNumAlignments': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_100q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_100q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_100q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_100q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_100q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_200q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_200q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_200q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_200q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_200q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_300q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_300q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_300q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_300q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_300q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_400q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_400q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_400q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_400q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_400q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_50q10_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_50q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_50q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_50q47_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_50q7_reads': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_mapped_bases_in_q10_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'sampled_mapped_bases_in_q17_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'sampled_mapped_bases_in_q20_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'sampled_mapped_bases_in_q47_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'sampled_mapped_bases_in_q7_alignments': ('django.db.models.fields.BigIntegerField', [], {}),
            'sampled_q10_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q10_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'sampled_q10_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q10_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q10_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'sampled_q17_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q17_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'sampled_q17_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q17_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q17_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'sampled_q20_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q20_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'sampled_q20_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q20_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q20_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'sampled_q47_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q47_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'sampled_q47_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q47_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q47_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'sampled_q7_alignments': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q7_coverage_percentage': ('django.db.models.fields.FloatField', [], {}),
            'sampled_q7_longest_alignment': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q7_mean_alignment_length': ('django.db.models.fields.IntegerField', [], {}),
            'sampled_q7_mean_coverage_depth': ('django.db.models.fields.FloatField', [], {}),
            'sysSNR': ('django.db.models.fields.FloatField', [], {}),
            'totalNumReads': ('django.db.models.fields.IntegerField', [], {}),
            'total_number_of_sampled_reads': ('django.db.models.fields.IntegerField', [], {})
        },
        'rundb.librarykey': {
            'Meta': {'object_name': 'LibraryKey'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'direction': ('django.db.models.fields.CharField', [], {'default': "'Forward'", 'max_length': '20'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isDefault': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '256'}),
            'sequence': ('django.db.models.fields.CharField', [], {'max_length': '64'})
        },
        'rundb.librarykit': {
            'Meta': {'object_name': 'LibraryKit'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '3024', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'sap': ('django.db.models.fields.CharField', [], {'max_length': '7', 'blank': 'True'})
        },
        'rundb.location': {
            'Meta': {'object_name': 'Location'},
            'comments': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'defaultlocation': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'rundb.message': {
            'Meta': {'object_name': 'Message'},
            'body': ('django.db.models.fields.TextField', [], {'default': "''", 'blank': 'True'}),
            'expires': ('django.db.models.fields.TextField', [], {'default': "'read'", 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'level': ('django.db.models.fields.IntegerField', [], {'default': '20'}),
            'route': ('django.db.models.fields.TextField', [], {'default': "''", 'blank': 'True'}),
            'status': ('django.db.models.fields.TextField', [], {'default': "'unread'", 'blank': 'True'}),
            'tags': ('django.db.models.fields.TextField', [], {'default': "''", 'blank': 'True'}),
            'time': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'})
        },
        'rundb.plannedexperiment': {
            'Meta': {'object_name': 'PlannedExperiment'},
            'adapter': ('django.db.models.fields.CharField', [], {'max_length': '256', 'null': 'True', 'blank': 'True'}),
            'autoAnalyze': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'autoName': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'barcodeId': ('django.db.models.fields.CharField', [], {'max_length': '256', 'null': 'True', 'blank': 'True'}),
            'bedfile': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'chipBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'chipType': ('django.db.models.fields.CharField', [], {'max_length': '32', 'null': 'True', 'blank': 'True'}),
            'cycles': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'date': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'expName': ('django.db.models.fields.CharField', [], {'max_length': '128', 'blank': 'True'}),
            'flows': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'flowsInOrder': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'forward3primeadapter': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'irworkflow': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'isReverseRun': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'libkit': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'library': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'libraryKey': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'librarykitname': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'metaData': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'notes': ('django.db.models.fields.CharField', [], {'max_length': '255', 'null': 'True', 'blank': 'True'}),
            'planExecuted': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'planExecutedDate': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'planGUID': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'planName': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'planPGM': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'planShortID': ('django.db.models.fields.CharField', [], {'max_length': '5', 'null': 'True', 'blank': 'True'}),
            'planStatus': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'preAnalysis': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'project': ('django.db.models.fields.CharField', [], {'max_length': '127', 'null': 'True', 'blank': 'True'}),
            'regionfile': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'reverse3primeadapter': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'reverse_primer': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'reverselibrarykey': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'runType': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'runname': ('django.db.models.fields.CharField', [], {'max_length': '255', 'null': 'True', 'blank': 'True'}),
            'sample': ('django.db.models.fields.CharField', [], {'max_length': '127', 'null': 'True', 'blank': 'True'}),
            'seqKitBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'sequencekitname': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'storageHost': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'storage_options': ('django.db.models.fields.CharField', [], {'default': "'A'", 'max_length': '200'}),
            'usePostBeadfind': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'usePreBeadfind': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'username': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'variantfrequency': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'})
        },
        'rundb.plugin': {
            'Meta': {'object_name': 'Plugin'},
            'active': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'autorun': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'chipType': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'config': ('django.db.models.fields.TextField', [], {'default': "''", 'null': 'True', 'blank': 'True'}),
            'date': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2012, 4, 2, 15, 40, 11, 877988)'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'libraryName': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'path': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'project': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'sample': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'selected': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'status': ('django.db.models.fields.TextField', [], {'default': "''", 'null': 'True', 'blank': 'True'}),
            'url': ('django.db.models.fields.URLField', [], {'default': "''", 'max_length': '256', 'blank': 'True'}),
            'version': ('django.db.models.fields.CharField', [], {'max_length': '256'})
        },
        'rundb.pluginresult': {
            'Meta': {'object_name': 'PluginResult'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'plugin': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Plugin']"}),
            'result': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Results']"}),
            'state': ('django.db.models.fields.CharField', [], {'max_length': '20'}),
            'store': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'})
        },
        'rundb.publisher': {
            'Meta': {'object_name': 'Publisher'},
            'date': ('django.db.models.fields.DateTimeField', [], {}),
            'global_meta': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '200'}),
            'path': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'version': ('django.db.models.fields.CharField', [], {'max_length': '256'})
        },
        'rundb.qualitymetrics': {
            'Meta': {'object_name': 'QualityMetrics'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'q0_100bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q0_15bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q0_50bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q0_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q0_max_read_length': ('django.db.models.fields.IntegerField', [], {}),
            'q0_mean_read_length': ('django.db.models.fields.FloatField', [], {}),
            'q0_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q17_100bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q17_150bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q17_50bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q17_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q17_max_read_length': ('django.db.models.fields.IntegerField', [], {}),
            'q17_mean_read_length': ('django.db.models.fields.FloatField', [], {}),
            'q17_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q20_100bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q20_150bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q20_50bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q20_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'q20_max_read_length': ('django.db.models.fields.FloatField', [], {}),
            'q20_mean_read_length': ('django.db.models.fields.IntegerField', [], {}),
            'q20_reads': ('django.db.models.fields.IntegerField', [], {}),
            'report': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Results']"})
        },
        'rundb.referencegenome': {
            'Meta': {'object_name': 'ReferenceGenome'},
            'date': ('django.db.models.fields.DateTimeField', [], {}),
            'enabled': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'index_version': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'notes': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'reference_path': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'short_name': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'source': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'species': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'status': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'verbose_error': ('django.db.models.fields.CharField', [], {'max_length': '3000', 'blank': 'True'}),
            'version': ('django.db.models.fields.CharField', [], {'max_length': '100', 'blank': 'True'})
        },
        'rundb.reportstorage': {
            'Meta': {'object_name': 'ReportStorage'},
            'dirPath': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'webServerPath': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'rundb.results': {
            'Meta': {'object_name': 'Results'},
            'analysisVersion': ('django.db.models.fields.CharField', [], {'max_length': '64'}),
            'experiment': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Experiment']"}),
            'fastqLink': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'framesProcessed': ('django.db.models.fields.IntegerField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'log': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'metaData': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'processedCycles': ('django.db.models.fields.IntegerField', [], {}),
            'processedflows': ('django.db.models.fields.IntegerField', [], {}),
            'reportLink': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'reportstorage': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'storage'", 'null': 'True', 'to': "orm['rundb.ReportStorage']"}),
            'resultsName': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'runid': ('django.db.models.fields.CharField', [], {'max_length': '10', 'blank': 'True'}),
            'sffLink': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'status': ('django.db.models.fields.CharField', [], {'max_length': '64'}),
            'tfFastq': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'tfSffLink': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'timeStamp': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'db_index': 'True', 'blank': 'True'}),
            'timeToComplete': ('django.db.models.fields.CharField', [], {'max_length': '64'})
        },
        'rundb.rig': {
            'Meta': {'object_name': 'Rig'},
            'alarms': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'comments': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'ftppassword': ('django.db.models.fields.CharField', [], {'default': "'ionguest'", 'max_length': '64'}),
            'ftprootdir': ('django.db.models.fields.CharField', [], {'default': "'results'", 'max_length': '64'}),
            'ftpserver': ('django.db.models.fields.CharField', [], {'default': "'192.168.201.1'", 'max_length': '128'}),
            'ftpusername': ('django.db.models.fields.CharField', [], {'default': "'ionguest'", 'max_length': '64'}),
            'last_clean_date': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'last_experiment': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'last_init_date': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'location': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Location']"}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200', 'primary_key': 'True'}),
            'serial': ('django.db.models.fields.CharField', [], {'max_length': '24', 'null': 'True', 'blank': 'True'}),
            'state': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'updateflag': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'updatehome': ('django.db.models.fields.CharField', [], {'default': "'192.168.201.1'", 'max_length': '256'}),
            'version': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'})
        },
        'rundb.runscript': {
            'Meta': {'object_name': 'RunScript'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'script': ('django.db.models.fields.TextField', [], {'blank': 'True'})
        },
        'rundb.runtype': {
            'Meta': {'object_name': 'RunType'},
            'barcode': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'description': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'meta': ('django.db.models.fields.TextField', [], {'default': "''", 'null': 'True', 'blank': 'True'}),
            'runType': ('django.db.models.fields.CharField', [], {'max_length': '512'})
        },
        'rundb.sequencingkit': {
            'Meta': {'object_name': 'SequencingKit'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '3024', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'sap': ('django.db.models.fields.CharField', [], {'max_length': '7', 'blank': 'True'})
        },
        'rundb.template': {
            'Meta': {'object_name': 'Template'},
            'comments': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isofficial': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'key': ('django.db.models.fields.CharField', [], {'max_length': '64'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '64'}),
            'sequence': ('django.db.models.fields.TextField', [], {'blank': 'True'})
        },
        'rundb.tfmetrics': {
            'CF': ('django.db.models.fields.FloatField', [], {}),
            'DR': ('django.db.models.fields.FloatField', [], {}),
            'HPAccuracy': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'HPSNR': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'IE': ('django.db.models.fields.FloatField', [], {}),
            'Meta': {'object_name': 'TFMetrics'},
            'Q10Histo': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'Q10Mean': ('django.db.models.fields.FloatField', [], {}),
            'Q10Mode': ('django.db.models.fields.FloatField', [], {}),
            'Q10ReadCount': ('django.db.models.fields.FloatField', [], {}),
            'Q17Histo': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'Q17Mean': ('django.db.models.fields.FloatField', [], {}),
            'Q17Mode': ('django.db.models.fields.FloatField', [], {}),
            'Q17ReadCount': ('django.db.models.fields.FloatField', [], {}),
            'SysSNR': ('django.db.models.fields.FloatField', [], {}),
            'aveHqReadCount': ('django.db.models.fields.FloatField', [], {}),
            'aveKeyCount': ('django.db.models.fields.FloatField', [], {}),
            'aveQ10ReadCount': ('django.db.models.fields.FloatField', [], {}),
            'aveQ17ReadCount': ('django.db.models.fields.FloatField', [], {}),
            'corOverlap': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'corrHPSNR': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'corrIonogram': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'error': ('django.db.models.fields.FloatField', [], {}),
            'hqReadCount': ('django.db.models.fields.FloatField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'keypass': ('django.db.models.fields.FloatField', [], {}),
            'matchMismatchHisto': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'matchMismatchMean': ('django.db.models.fields.FloatField', [], {}),
            'matchMismatchMode': ('django.db.models.fields.FloatField', [], {}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '128', 'db_index': 'True'}),
            'number': ('django.db.models.fields.FloatField', [], {}),
            'postCorrSNR': ('django.db.models.fields.FloatField', [], {}),
            'preCorrSNR': ('django.db.models.fields.FloatField', [], {}),
            'rawIonogram': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'rawOverlap': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'report': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Results']"}),
            'sequence': ('django.db.models.fields.CharField', [], {'max_length': '512'})
        },
        'rundb.threeprimeadapter': {
            'Meta': {'object_name': 'ThreePrimeadapter'},
            'adapter_cutoff': ('django.db.models.fields.IntegerField', [], {}),
            'description': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'direction': ('django.db.models.fields.CharField', [], {'default': "'Forward'", 'max_length': '20'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isDefault': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '256'}),
            'qual_cutoff': ('django.db.models.fields.IntegerField', [], {}),
            'qual_window': ('django.db.models.fields.IntegerField', [], {}),
            'sequence': ('django.db.models.fields.CharField', [], {'max_length': '512'})
        },
        'rundb.usereventlog': {
            'Meta': {'object_name': 'UserEventLog'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'text': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'timeStamp': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'upload': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'logs'", 'to': "orm['rundb.ContentUpload']"})
        },
        'rundb.userprofile': {
            'Meta': {'object_name': 'UserProfile'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '93'}),
            'note': ('django.db.models.fields.TextField', [], {'default': "''", 'blank': 'True'}),
            'phone_number': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '256', 'blank': 'True'}),
            'title': ('django.db.models.fields.CharField', [], {'default': "'user'", 'max_length': '256'}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['auth.User']", 'unique': 'True'})
        },
        'rundb.variantfrequencies': {
            'Meta': {'object_name': 'VariantFrequencies'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '3024', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'})
        }
    }

    complete_apps = ['rundb']
