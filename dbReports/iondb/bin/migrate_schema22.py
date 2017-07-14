# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# encoding: utf-8
import djangoinit
import datetime

from south.db import db
from south.v2 import SchemaMigration
from django.db import models, DatabaseError
from south.migration.migrators import Forwards
from south.orm import FakeORM

import logging

# (ab)use South internals to run non-migration from a fixed schema
logging.basicConfig(level=logging.INFO)
logging.disable(logging.DEBUG)  # hide south DEBUG messages
log = logging.getLogger('LegacyMigration')


class LegacyMigration(SchemaMigration):

    def forwards(self, orm):
        from django.db import models
        try:
            db.start_transaction()
            # Adding model 'Experiment'
            db.create_table('rundb_experiment', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('expDir', models.fields.CharField(max_length=512)),
                ('expName', models.fields.CharField(max_length=128)),
                ('pgmName', models.fields.CharField(max_length=64)),
                ('log', models.fields.TextField(default='{}', blank=True)),
                ('unique', models.fields.CharField(unique=True, max_length=512)),
                ('date', models.fields.DateTimeField()),
                ('storage_options', models.fields.CharField(default='A', max_length=200)),
                ('user_ack', models.fields.CharField(default='U', max_length=24)),
                ('project', models.fields.CharField(max_length=64, null=True, blank=True)),
                ('sample', models.fields.CharField(max_length=64, null=True, blank=True)),
                ('library', models.fields.CharField(max_length=64, null=True, blank=True)),
                ('notes', models.fields.CharField(max_length=128, null=True, blank=True)),
                ('chipBarcode', models.fields.CharField(max_length=64, blank=True)),
                ('seqKitBarcode', models.fields.CharField(max_length=64, blank=True)),
                ('reagentBarcode', models.fields.CharField(max_length=64, blank=True)),
                ('autoAnalyze', models.fields.BooleanField(default=False)),
                ('usePreBeadfind', models.fields.BooleanField(default=False)),
                ('chipType', models.fields.CharField(max_length=32)),
                ('cycles', models.fields.IntegerField()),
                ('flows', models.fields.IntegerField()),
                ('expCompInfo', models.fields.TextField(blank=True)),
                ('baselineRun', models.fields.BooleanField(default=False)),
                ('flowsInOrder', models.fields.CharField(max_length=512)),
                ('star', models.fields.BooleanField(default=False)),
                ('ftpStatus', models.fields.CharField(max_length=512, blank=True)),
                ('libraryKey', models.fields.CharField(max_length=64, blank=True)),
                ('storageHost', models.fields.CharField(max_length=128, null=True, blank=True)),
                ('barcodeId', models.fields.CharField(max_length=128, null=True, blank=True)),
                ('reverse_primer', models.fields.CharField(max_length=128, null=True, blank=True)),
                ('rawdatastyle', models.fields.CharField(
                    default='single', max_length=24, null=True, blank=True)),
                ('sequencekitname', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('sequencekitbarcode', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('librarykitname', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('librarykitbarcode', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('reverselibrarykey', models.fields.CharField(max_length=64, null=True, blank=True)),
                ('reverse3primeadapter', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('forward3primeadapter', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('isReverseRun', models.fields.BooleanField(default=False)),
                ('metaData', models.fields.TextField(default='{}', blank=True)),
            ))
            db.send_create_signal('rundb', ['Experiment'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Experiment table already exists")
            db.clear_deferred_sql()

        # Must be before Results due to ForeignKey
        try:
            db.start_transaction()
            # Adding model 'ReportStorage'
            db.create_table('rundb_reportstorage', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=200)),
                ('webServerPath', models.fields.CharField(max_length=200)),
                ('dirPath', models.fields.CharField(max_length=200)),
            ))
            db.send_create_signal('rundb', ['Results'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("ReportStorage table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'TFMetrics'
            db.create_table('rundb_tfmetrics', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('report', models.fields.related.ForeignKey(to=orm['rundb.Results'])),
                ('name', models.fields.CharField(max_length=128, db_index=True)),
                ('matchMismatchHisto', models.fields.TextField(blank=True)),
                ('matchMismatchMean', models.fields.FloatField()),
                ('matchMismatchMode', models.fields.FloatField()),
                ('Q10Histo', models.fields.TextField(blank=True)),
                ('Q10Mean', models.fields.FloatField()),
                ('Q10Mode', models.fields.FloatField()),
                ('Q17Histo', models.fields.TextField(blank=True)),
                ('Q17Mean', models.fields.FloatField()),
                ('Q17Mode', models.fields.FloatField()),
                ('SysSNR', models.fields.FloatField()),
                ('HPSNR', models.fields.TextField(blank=True)),
                ('corrHPSNR', models.fields.TextField(blank=True)),
                ('HPAccuracy', models.fields.TextField(blank=True)),
                ('rawOverlap', models.fields.TextField(blank=True)),
                ('corOverlap', models.fields.TextField(blank=True)),
                ('hqReadCount', models.fields.FloatField()),
                ('aveHqReadCount', models.fields.FloatField()),
                ('Q10ReadCount', models.fields.FloatField()),
                ('aveQ10ReadCount', models.fields.FloatField()),
                ('Q17ReadCount', models.fields.FloatField()),
                ('aveQ17ReadCount', models.fields.FloatField()),
                ('sequence', models.fields.CharField(max_length=512)),
                ('keypass', models.fields.FloatField()),
                ('preCorrSNR', models.fields.FloatField()),
                ('postCorrSNR', models.fields.FloatField()),
                ('rawIonogram', models.fields.TextField(blank=True)),
                ('corrIonogram', models.fields.TextField(blank=True)),
                ('CF', models.fields.FloatField()),
                ('IE', models.fields.FloatField()),
                ('DR', models.fields.FloatField()),
                ('error', models.fields.FloatField()),
                ('number', models.fields.FloatField()),
                ('aveKeyCount', models.fields.FloatField()),
            ))
            db.send_create_signal('rundb', ['TFMetrics'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("TFMetrics table already exists")
            db.clear_deferred_sql()

        # Must be before Rig, FileServer
        try:
            db.start_transaction()
            # Adding model 'Location'
            db.create_table('rundb_location', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=200)),
                ('comments', models.fields.TextField(blank=True)),
                ('defaultlocation', models.fields.BooleanField(default=False)),
            ))
            db.send_create_signal('rundb', ['Location'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Location table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'Rig'
            db.create_table('rundb_rig', (
                ('name', models.fields.CharField(max_length=200, primary_key=True)),
                ('location', models.fields.related.ForeignKey(to=orm['rundb.Location'])),
                ('comments', models.fields.TextField(blank=True)),
                ('ftpserver', models.fields.CharField(default='192.168.201.1', max_length=128)),
                ('ftpusername', models.fields.CharField(default='ionguest', max_length=64)),
                ('ftppassword', models.fields.CharField(default='ionguest', max_length=64)),
                ('ftprootdir', models.fields.CharField(default='results', max_length=64)),
                ('updatehome', models.fields.CharField(default='192.168.201.1', max_length=256)),
                ('updateflag', models.fields.BooleanField(default=False)),
                ('serial', models.fields.CharField(max_length=24, null=True, blank=True)),
                ('state', models.fields.CharField(max_length=512, blank=True)),
                ('version', models.fields.TextField(default='{}', blank=True)),
                ('alarms', models.fields.TextField(default='{}', blank=True)),
                ('last_init_date', models.fields.CharField(max_length=512, blank=True)),
                ('last_clean_date', models.fields.CharField(max_length=512, blank=True)),
                ('last_experiment', models.fields.CharField(max_length=512, blank=True)),
            ))
            db.send_create_signal('rundb', ['Rig'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Rig table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'FileServer'
            db.create_table('rundb_fileserver', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=200)),
                ('comments', models.fields.TextField(blank=True)),
                ('filesPrefix', models.fields.CharField(max_length=200)),
                ('location', models.fields.related.ForeignKey(to=orm['rundb.Location'])),
            ))
            db.send_create_signal('rundb', ['FileServer'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("FileServer table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'RunScript'
            db.create_table('rundb_runscript', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=200)),
                ('script', models.fields.TextField(blank=True)),
            ))
            db.send_create_signal('rundb', ['RunScript'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("RunScript table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'Cruncher'
            db.create_table('rundb_cruncher', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=200)),
                ('prefix', models.fields.CharField(max_length=512)),
                ('location', models.fields.related.ForeignKey(to=orm['rundb.Location'])),
                ('comments', models.fields.TextField(blank=True)),
            ))
            db.send_create_signal('rundb', ['Cruncher'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Cruncher table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'AnalysisMetrics'
            db.create_table('rundb_analysismetrics', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('report', models.fields.related.ForeignKey(to=orm['rundb.Results'])),
                ('libLive', models.fields.IntegerField()),
                ('libKp', models.fields.IntegerField()),
                ('libMix', models.fields.IntegerField()),
                ('libFinal', models.fields.IntegerField()),
                ('tfLive', models.fields.IntegerField()),
                ('tfKp', models.fields.IntegerField()),
                ('tfMix', models.fields.IntegerField()),
                ('tfFinal', models.fields.IntegerField()),
                ('empty', models.fields.IntegerField()),
                ('bead', models.fields.IntegerField()),
                ('live', models.fields.IntegerField()),
                ('dud', models.fields.IntegerField()),
                ('amb', models.fields.IntegerField()),
                ('tf', models.fields.IntegerField()),
                ('lib', models.fields.IntegerField()),
                ('pinned', models.fields.IntegerField()),
                ('ignored', models.fields.IntegerField()),
                ('excluded', models.fields.IntegerField()),
                ('washout', models.fields.IntegerField()),
                ('washout_dud', models.fields.IntegerField()),
                ('washout_ambiguous', models.fields.IntegerField()),
                ('washout_live', models.fields.IntegerField()),
                ('washout_test_fragment', models.fields.IntegerField()),
                ('washout_library', models.fields.IntegerField()),
                ('lib_pass_basecaller', models.fields.IntegerField()),
                ('lib_pass_cafie', models.fields.IntegerField()),
                ('keypass_all_beads', models.fields.IntegerField()),
                ('sysCF', models.fields.FloatField()),
                ('sysIE', models.fields.FloatField()),
                ('sysDR', models.fields.FloatField()),
            ))
            db.send_create_signal('rundb', ['AnalysisMetrics'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("AnalysisMetrics table already exists")

        try:
            db.start_transaction()
            # Adding model 'LibMetrics'
            db.create_table('rundb_libmetrics', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('report', models.fields.related.ForeignKey(to=orm['rundb.Results'])),
                ('sysSNR', models.fields.FloatField()),
                ('aveKeyCounts', models.fields.FloatField()),
                ('totalNumReads', models.fields.IntegerField()),
                ('genomelength', models.fields.IntegerField()),
                ('rNumAlignments', models.fields.IntegerField()),
                ('rMeanAlignLen', models.fields.IntegerField()),
                ('rLongestAlign', models.fields.IntegerField()),
                ('rCoverage', models.fields.FloatField()),
                ('r50Q10', models.fields.IntegerField()),
                ('r100Q10', models.fields.IntegerField()),
                ('r200Q10', models.fields.IntegerField()),
                ('r50Q17', models.fields.IntegerField()),
                ('r100Q17', models.fields.IntegerField()),
                ('r200Q17', models.fields.IntegerField()),
                ('r50Q20', models.fields.IntegerField()),
                ('r100Q20', models.fields.IntegerField()),
                ('r200Q20', models.fields.IntegerField()),
                ('sNumAlignments', models.fields.IntegerField()),
                ('sMeanAlignLen', models.fields.IntegerField()),
                ('sLongestAlign', models.fields.IntegerField()),
                ('sCoverage', models.fields.FloatField()),
                ('s50Q10', models.fields.IntegerField()),
                ('s100Q10', models.fields.IntegerField()),
                ('s200Q10', models.fields.IntegerField()),
                ('s50Q17', models.fields.IntegerField()),
                ('s100Q17', models.fields.IntegerField()),
                ('s200Q17', models.fields.IntegerField()),
                ('s50Q20', models.fields.IntegerField()),
                ('s100Q20', models.fields.IntegerField()),
                ('s200Q20', models.fields.IntegerField()),
                ('q7_coverage_percentage', models.fields.FloatField()),
                ('q7_alignments', models.fields.IntegerField()),
                ('q7_mapped_bases', models.fields.BigIntegerField()),
                ('q7_qscore_bases', models.fields.BigIntegerField()),
                ('q7_mean_alignment_length', models.fields.IntegerField()),
                ('q7_longest_alignment', models.fields.IntegerField()),
                ('i50Q7_reads', models.fields.IntegerField()),
                ('i100Q7_reads', models.fields.IntegerField()),
                ('i150Q7_reads', models.fields.IntegerField()),
                ('i200Q7_reads', models.fields.IntegerField()),
                ('i250Q7_reads', models.fields.IntegerField()),
                ('i300Q7_reads', models.fields.IntegerField()),
                ('i350Q7_reads', models.fields.IntegerField()),
                ('i400Q7_reads', models.fields.IntegerField()),
                ('i450Q7_reads', models.fields.IntegerField()),
                ('i500Q7_reads', models.fields.IntegerField()),
                ('i550Q7_reads', models.fields.IntegerField()),
                ('i600Q7_reads', models.fields.IntegerField()),
                ('q10_coverage_percentage', models.fields.FloatField()),
                ('q10_alignments', models.fields.IntegerField()),
                ('q10_mapped_bases', models.fields.BigIntegerField()),
                ('q10_qscore_bases', models.fields.BigIntegerField()),
                ('q10_mean_alignment_length', models.fields.IntegerField()),
                ('q10_longest_alignment', models.fields.IntegerField()),
                ('i50Q10_reads', models.fields.IntegerField()),
                ('i100Q10_reads', models.fields.IntegerField()),
                ('i150Q10_reads', models.fields.IntegerField()),
                ('i200Q10_reads', models.fields.IntegerField()),
                ('i250Q10_reads', models.fields.IntegerField()),
                ('i300Q10_reads', models.fields.IntegerField()),
                ('i350Q10_reads', models.fields.IntegerField()),
                ('i400Q10_reads', models.fields.IntegerField()),
                ('i450Q10_reads', models.fields.IntegerField()),
                ('i500Q10_reads', models.fields.IntegerField()),
                ('i550Q10_reads', models.fields.IntegerField()),
                ('i600Q10_reads', models.fields.IntegerField()),
                ('q17_coverage_percentage', models.fields.FloatField()),
                ('q17_alignments', models.fields.IntegerField()),
                ('q17_mapped_bases', models.fields.BigIntegerField()),
                ('q17_qscore_bases', models.fields.BigIntegerField()),
                ('q17_mean_alignment_length', models.fields.IntegerField()),
                ('q17_longest_alignment', models.fields.IntegerField()),
                ('i50Q17_reads', models.fields.IntegerField()),
                ('i100Q17_reads', models.fields.IntegerField()),
                ('i150Q17_reads', models.fields.IntegerField()),
                ('i200Q17_reads', models.fields.IntegerField()),
                ('i250Q17_reads', models.fields.IntegerField()),
                ('i300Q17_reads', models.fields.IntegerField()),
                ('i350Q17_reads', models.fields.IntegerField()),
                ('i400Q17_reads', models.fields.IntegerField()),
                ('i450Q17_reads', models.fields.IntegerField()),
                ('i500Q17_reads', models.fields.IntegerField()),
                ('i550Q17_reads', models.fields.IntegerField()),
                ('i600Q17_reads', models.fields.IntegerField()),
                ('q20_coverage_percentage', models.fields.FloatField()),
                ('q20_alignments', models.fields.IntegerField()),
                ('q20_mapped_bases', models.fields.BigIntegerField()),
                ('q20_qscore_bases', models.fields.BigIntegerField()),
                ('q20_mean_alignment_length', models.fields.IntegerField()),
                ('q20_longest_alignment', models.fields.IntegerField()),
                ('i50Q20_reads', models.fields.IntegerField()),
                ('i100Q20_reads', models.fields.IntegerField()),
                ('i150Q20_reads', models.fields.IntegerField()),
                ('i200Q20_reads', models.fields.IntegerField()),
                ('i250Q20_reads', models.fields.IntegerField()),
                ('i300Q20_reads', models.fields.IntegerField()),
                ('i350Q20_reads', models.fields.IntegerField()),
                ('i400Q20_reads', models.fields.IntegerField()),
                ('i450Q20_reads', models.fields.IntegerField()),
                ('i500Q20_reads', models.fields.IntegerField()),
                ('i550Q20_reads', models.fields.IntegerField()),
                ('i600Q20_reads', models.fields.IntegerField()),
                ('q47_coverage_percentage', models.fields.FloatField()),
                ('q47_mapped_bases', models.fields.BigIntegerField()),
                ('q47_qscore_bases', models.fields.BigIntegerField()),
                ('q47_alignments', models.fields.IntegerField()),
                ('q47_mean_alignment_length', models.fields.IntegerField()),
                ('q47_longest_alignment', models.fields.IntegerField()),
                ('i50Q47_reads', models.fields.IntegerField()),
                ('i100Q47_reads', models.fields.IntegerField()),
                ('i150Q47_reads', models.fields.IntegerField()),
                ('i200Q47_reads', models.fields.IntegerField()),
                ('i250Q47_reads', models.fields.IntegerField()),
                ('i300Q47_reads', models.fields.IntegerField()),
                ('i350Q47_reads', models.fields.IntegerField()),
                ('i400Q47_reads', models.fields.IntegerField()),
                ('i450Q47_reads', models.fields.IntegerField()),
                ('i500Q47_reads', models.fields.IntegerField()),
                ('i550Q47_reads', models.fields.IntegerField()),
                ('i600Q47_reads', models.fields.IntegerField()),
                ('cf', models.fields.FloatField()),
                ('ie', models.fields.FloatField()),
                ('dr', models.fields.FloatField()),
                ('Genome_Version', models.fields.CharField(max_length=512)),
                ('Index_Version', models.fields.CharField(max_length=512)),
                ('align_sample', models.fields.IntegerField()),
                ('genome', models.fields.CharField(max_length=512)),
                ('genomesize', models.fields.BigIntegerField()),
                ('total_number_of_sampled_reads', models.fields.IntegerField()),
                ('sampled_q7_coverage_percentage', models.fields.FloatField()),
                ('sampled_q7_mean_coverage_depth', models.fields.FloatField()),
                ('sampled_q7_alignments', models.fields.IntegerField()),
                ('sampled_q7_mean_alignment_length', models.fields.IntegerField()),
                ('sampled_mapped_bases_in_q7_alignments', models.fields.BigIntegerField()),
                ('sampled_q7_longest_alignment', models.fields.IntegerField()),
                ('sampled_50q7_reads', models.fields.IntegerField()),
                ('sampled_100q7_reads', models.fields.IntegerField()),
                ('sampled_200q7_reads', models.fields.IntegerField()),
                ('sampled_300q7_reads', models.fields.IntegerField()),
                ('sampled_400q7_reads', models.fields.IntegerField()),
                ('sampled_q10_coverage_percentage', models.fields.FloatField()),
                ('sampled_q10_mean_coverage_depth', models.fields.FloatField()),
                ('sampled_q10_alignments', models.fields.IntegerField()),
                ('sampled_q10_mean_alignment_length', models.fields.IntegerField()),
                ('sampled_mapped_bases_in_q10_alignments', models.fields.BigIntegerField()),
                ('sampled_q10_longest_alignment', models.fields.IntegerField()),
                ('sampled_50q10_reads', models.fields.IntegerField()),
                ('sampled_100q10_reads', models.fields.IntegerField()),
                ('sampled_200q10_reads', models.fields.IntegerField()),
                ('sampled_300q10_reads', models.fields.IntegerField()),
                ('sampled_400q10_reads', models.fields.IntegerField()),
                ('sampled_q17_coverage_percentage', models.fields.FloatField()),
                ('sampled_q17_mean_coverage_depth', models.fields.FloatField()),
                ('sampled_q17_alignments', models.fields.IntegerField()),
                ('sampled_q17_mean_alignment_length', models.fields.IntegerField()),
                ('sampled_mapped_bases_in_q17_alignments', models.fields.BigIntegerField()),
                ('sampled_q17_longest_alignment', models.fields.IntegerField()),
                ('sampled_50q17_reads', models.fields.IntegerField()),
                ('sampled_100q17_reads', models.fields.IntegerField()),
                ('sampled_200q17_reads', models.fields.IntegerField()),
                ('sampled_300q17_reads', models.fields.IntegerField()),
                ('sampled_400q17_reads', models.fields.IntegerField()),
                ('sampled_q20_coverage_percentage', models.fields.FloatField()),
                ('sampled_q20_mean_coverage_depth', models.fields.FloatField()),
                ('sampled_q20_alignments', models.fields.IntegerField()),
                ('sampled_q20_mean_alignment_length', models.fields.IntegerField()),
                ('sampled_mapped_bases_in_q20_alignments', models.fields.BigIntegerField()),
                ('sampled_q20_longest_alignment', models.fields.IntegerField()),
                ('sampled_50q20_reads', models.fields.IntegerField()),
                ('sampled_100q20_reads', models.fields.IntegerField()),
                ('sampled_200q20_reads', models.fields.IntegerField()),
                ('sampled_300q20_reads', models.fields.IntegerField()),
                ('sampled_400q20_reads', models.fields.IntegerField()),
                ('sampled_q47_coverage_percentage', models.fields.FloatField()),
                ('sampled_q47_mean_coverage_depth', models.fields.FloatField()),
                ('sampled_q47_alignments', models.fields.IntegerField()),
                ('sampled_q47_mean_alignment_length', models.fields.IntegerField()),
                ('sampled_mapped_bases_in_q47_alignments', models.fields.BigIntegerField()),
                ('sampled_q47_longest_alignment', models.fields.IntegerField()),
                ('sampled_50q47_reads', models.fields.IntegerField()),
                ('sampled_100q47_reads', models.fields.IntegerField()),
                ('sampled_200q47_reads', models.fields.IntegerField()),
                ('sampled_300q47_reads', models.fields.IntegerField()),
                ('sampled_400q47_reads', models.fields.IntegerField()),
                ('extrapolated_from_number_of_sampled_reads', models.fields.IntegerField()),
                ('extrapolated_q7_coverage_percentage', models.fields.FloatField()),
                ('extrapolated_q7_mean_coverage_depth', models.fields.FloatField()),
                ('extrapolated_q7_alignments', models.fields.IntegerField()),
                ('extrapolated_q7_mean_alignment_length', models.fields.IntegerField()),
                ('extrapolated_mapped_bases_in_q7_alignments', models.fields.BigIntegerField()),
                ('extrapolated_q7_longest_alignment', models.fields.IntegerField()),
                ('extrapolated_50q7_reads', models.fields.IntegerField()),
                ('extrapolated_100q7_reads', models.fields.IntegerField()),
                ('extrapolated_200q7_reads', models.fields.IntegerField()),
                ('extrapolated_300q7_reads', models.fields.IntegerField()),
                ('extrapolated_400q7_reads', models.fields.IntegerField()),
                ('extrapolated_q10_coverage_percentage', models.fields.FloatField()),
                ('extrapolated_q10_mean_coverage_depth', models.fields.FloatField()),
                ('extrapolated_q10_alignments', models.fields.IntegerField()),
                ('extrapolated_q10_mean_alignment_length', models.fields.IntegerField()),
                ('extrapolated_mapped_bases_in_q10_alignments', models.fields.BigIntegerField()),
                ('extrapolated_q10_longest_alignment', models.fields.IntegerField()),
                ('extrapolated_50q10_reads', models.fields.IntegerField()),
                ('extrapolated_100q10_reads', models.fields.IntegerField()),
                ('extrapolated_200q10_reads', models.fields.IntegerField()),
                ('extrapolated_300q10_reads', models.fields.IntegerField()),
                ('extrapolated_400q10_reads', models.fields.IntegerField()),
                ('extrapolated_q17_coverage_percentage', models.fields.FloatField()),
                ('extrapolated_q17_mean_coverage_depth', models.fields.FloatField()),
                ('extrapolated_q17_alignments', models.fields.IntegerField()),
                ('extrapolated_q17_mean_alignment_length', models.fields.IntegerField()),
                ('extrapolated_mapped_bases_in_q17_alignments', models.fields.BigIntegerField()),
                ('extrapolated_q17_longest_alignment', models.fields.IntegerField()),
                ('extrapolated_50q17_reads', models.fields.IntegerField()),
                ('extrapolated_100q17_reads', models.fields.IntegerField()),
                ('extrapolated_200q17_reads', models.fields.IntegerField()),
                ('extrapolated_300q17_reads', models.fields.IntegerField()),
                ('extrapolated_400q17_reads', models.fields.IntegerField()),
                ('extrapolated_q20_coverage_percentage', models.fields.FloatField()),
                ('extrapolated_q20_mean_coverage_depth', models.fields.FloatField()),
                ('extrapolated_q20_alignments', models.fields.IntegerField()),
                ('extrapolated_q20_mean_alignment_length', models.fields.IntegerField()),
                ('extrapolated_mapped_bases_in_q20_alignments', models.fields.BigIntegerField()),
                ('extrapolated_q20_longest_alignment', models.fields.IntegerField()),
                ('extrapolated_50q20_reads', models.fields.IntegerField()),
                ('extrapolated_100q20_reads', models.fields.IntegerField()),
                ('extrapolated_200q20_reads', models.fields.IntegerField()),
                ('extrapolated_300q20_reads', models.fields.IntegerField()),
                ('extrapolated_400q20_reads', models.fields.IntegerField()),
                ('extrapolated_q47_coverage_percentage', models.fields.FloatField()),
                ('extrapolated_q47_mean_coverage_depth', models.fields.FloatField()),
                ('extrapolated_q47_alignments', models.fields.IntegerField()),
                ('extrapolated_q47_mean_alignment_length', models.fields.IntegerField()),
                ('extrapolated_mapped_bases_in_q47_alignments', models.fields.BigIntegerField()),
                ('extrapolated_q47_longest_alignment', models.fields.IntegerField()),
                ('extrapolated_50q47_reads', models.fields.IntegerField()),
                ('extrapolated_100q47_reads', models.fields.IntegerField()),
                ('extrapolated_200q47_reads', models.fields.IntegerField()),
                ('extrapolated_300q47_reads', models.fields.IntegerField()),
                ('extrapolated_400q47_reads', models.fields.IntegerField()),
            ))
            db.send_create_signal('rundb', ['LibMetrics'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("LibMetrics table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'QualityMetrics'
            db.create_table('rundb_qualitymetrics', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('report', models.fields.related.ForeignKey(to=orm['rundb.Results'])),
                ('q0_bases', models.fields.BigIntegerField()),
                ('q0_reads', models.fields.IntegerField()),
                ('q0_max_read_length', models.fields.IntegerField()),
                ('q0_mean_read_length', models.fields.FloatField()),
                ('q0_50bp_reads', models.fields.IntegerField()),
                ('q0_100bp_reads', models.fields.IntegerField()),
                ('q0_15bp_reads', models.fields.IntegerField()),
                ('q17_bases', models.fields.BigIntegerField()),
                ('q17_reads', models.fields.IntegerField()),
                ('q17_max_read_length', models.fields.IntegerField()),
                ('q17_mean_read_length', models.fields.FloatField()),
                ('q17_50bp_reads', models.fields.IntegerField()),
                ('q17_100bp_reads', models.fields.IntegerField()),
                ('q17_150bp_reads', models.fields.IntegerField()),
                ('q20_bases', models.fields.BigIntegerField()),
                ('q20_reads', models.fields.IntegerField()),
                ('q20_max_read_length', models.fields.FloatField()),
                ('q20_mean_read_length', models.fields.IntegerField()),
                ('q20_50bp_reads', models.fields.IntegerField()),
                ('q20_100bp_reads', models.fields.IntegerField()),
                ('q20_150bp_reads', models.fields.IntegerField()),
            ))
            db.send_create_signal('rundb', ['QualityMetrics'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("QualityMetrics table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'Template'
            db.create_table('rundb_template', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=64)),
                ('sequence', models.fields.TextField(blank=True)),
                ('key', models.fields.CharField(max_length=64)),
                ('comments', models.fields.TextField(blank=True)),
                ('isofficial', models.fields.BooleanField(default=True)),
            ))
            db.send_create_signal('rundb', ['Template'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Template table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'Backup'
            db.create_table('rundb_backup', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('experiment', models.fields.related.ForeignKey(to=orm['rundb.Experiment'])),
                ('backupName', models.fields.CharField(unique=True, max_length=256)),
                ('isBackedUp', models.fields.BooleanField(default=False)),
                ('backupDate', models.fields.DateTimeField()),
                ('backupPath', models.fields.CharField(max_length=512)),
            ))
            db.send_create_signal('rundb', ['Backup'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Backup table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'BackupConfig'
            db.create_table('rundb_backupconfig', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=64)),
                ('location', models.fields.related.ForeignKey(to=orm['rundb.Location'])),
                ('backup_directory', models.fields.CharField(default=None, max_length=256, blank=True)),
                ('backup_threshold', models.fields.IntegerField(blank=True)),
                ('number_to_backup', models.fields.IntegerField(blank=True)),
                ('grace_period', models.fields.IntegerField(default=72)),
                ('timeout', models.fields.IntegerField(blank=True)),
                ('bandwidth_limit', models.fields.IntegerField(blank=True)),
                ('status', models.fields.CharField(max_length=512, blank=True)),
                ('online', models.fields.BooleanField(default=False)),
                ('comments', models.fields.TextField(blank=True)),
                ('email', models.fields.EmailField(max_length=75, blank=True)),
            ))
            db.send_create_signal('rundb', ['BackupConfig'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("BackupConfig table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'Chip'
            db.create_table('rundb_chip', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=128)),
                ('slots', models.fields.IntegerField()),
                ('args', models.fields.CharField(max_length=512, blank=True)),
            ))
            db.send_create_signal('rundb', ['Chip'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Chip table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'GlobalConfig'
            db.create_table('rundb_globalconfig', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=512)),
                ('selected', models.fields.BooleanField(default=False)),
                ('plugin_folder', models.fields.CharField(max_length=512, blank=True)),
                ('basecallerargs', models.fields.CharField(max_length=512, blank=True)),
                ('fasta_path', models.fields.CharField(max_length=512, blank=True)),
                ('reference_path', models.fields.CharField(max_length=1000, blank=True)),
                ('records_to_display', models.fields.IntegerField(default=20, blank=True)),
                ('default_test_fragment_key', models.fields.CharField(max_length=50, blank=True)),
                ('default_library_key', models.fields.CharField(max_length=50, blank=True)),
                ('default_flow_order', models.fields.CharField(max_length=100, blank=True)),
                ('plugin_output_folder', models.fields.CharField(max_length=500, blank=True)),
                ('default_plugin_script', models.fields.CharField(max_length=500, blank=True)),
                ('web_root', models.fields.CharField(max_length=500, blank=True)),
                ('site_name', models.fields.CharField(max_length=500, blank=True)),
                ('default_storage_options', models.fields.CharField(default='D', max_length=500, blank=True)),
                ('auto_archive_ack', models.fields.BooleanField(default=False)),
                ('barcode_args', models.fields.TextField(default='{}', blank=True)),
                ('enable_auto_pkg_dl', models.fields.BooleanField(default=True)),
                ('ts_update_status', models.fields.CharField(max_length=256, blank=True)),
            ))
            db.send_create_signal('rundb', ['GlobalConfig'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("GlobalConfig table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'EmailAddress'
            db.create_table('rundb_emailaddress', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('email', models.fields.EmailField(max_length=75, blank=True)),
                ('selected', models.fields.BooleanField(default=False)),
            ))
            db.send_create_signal('rundb', ['EmailAddress'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("EmailAddress table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'RunType'
            db.create_table('rundb_runtype', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('runType', models.fields.CharField(max_length=512)),
                ('barcode', models.fields.CharField(max_length=512, blank=True)),
                ('description', models.fields.TextField(blank=True)),
                ('meta', models.fields.TextField(default='', null=True, blank=True)),
            ))
            db.send_create_signal('rundb', ['RunType'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("RunType table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'Plugin'
            db.create_table('rundb_plugin', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=512)),
                ('version', models.fields.CharField(max_length=256)),
                ('date', models.fields.DateTimeField(
                    default=datetime.datetime(2012, 4, 2, 15, 40, 11, 877988))),
                ('selected', models.fields.BooleanField(default=False)),
                ('path', models.fields.CharField(max_length=512)),
                ('project', models.fields.CharField(default='', max_length=512, blank=True)),
                ('sample', models.fields.CharField(default='', max_length=512, blank=True)),
                ('libraryName', models.fields.CharField(default='', max_length=512, blank=True)),
                ('chipType', models.fields.CharField(default='', max_length=512, blank=True)),
                ('autorun', models.fields.BooleanField(default=False)),
                ('config', models.fields.TextField(default='', null=True, blank=True)),
                ('status', models.fields.TextField(default='', null=True, blank=True)),
                ('active', models.fields.BooleanField(default=True)),
                ('url', models.fields.URLField(default='', max_length=256, blank=True)),
            ))
            db.send_create_signal('rundb', ['Plugin'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Plugin table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'PluginResult'
            db.create_table('rundb_pluginresult', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('plugin', models.fields.related.ForeignKey(to=orm['rundb.Plugin'])),
                ('result', models.fields.related.ForeignKey(to=orm['rundb.Results'])),
                ('state', models.fields.CharField(max_length=20)),
                ('store', models.fields.TextField(default='{}', blank=True)),
            ))
            db.send_create_signal('rundb', ['PluginResult'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("PluginResult table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'dnaBarcode'
            db.create_table('rundb_dnabarcode', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=128)),
                ('id_str', models.fields.CharField(max_length=128)),
                ('type', models.fields.CharField(max_length=64, blank=True)),
                ('sequence', models.fields.CharField(max_length=128)),
                ('length', models.fields.IntegerField(default=0, blank=True)),
                ('floworder', models.fields.CharField(default='', max_length=128, blank=True)),
                ('index', models.fields.IntegerField()),
                ('annotation', models.fields.CharField(default='', max_length=512, blank=True)),
                ('adapter', models.fields.CharField(default='', max_length=128, blank=True)),
                ('score_mode', models.fields.IntegerField(default=0, blank=True)),
                ('score_cutoff', models.fields.FloatField(default=0)),
            ))
            db.send_create_signal('rundb', ['dnaBarcode'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("dnaBarcode table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'ReferenceGenome'
            db.create_table('rundb_referencegenome', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(max_length=512)),
                ('short_name', models.fields.CharField(max_length=512)),
                ('enabled', models.fields.BooleanField(default=True)),
                ('reference_path', models.fields.CharField(max_length=1024, blank=True)),
                ('date', models.fields.DateTimeField()),
                ('version', models.fields.CharField(max_length=100, blank=True)),
                ('species', models.fields.CharField(max_length=512, blank=True)),
                ('source', models.fields.CharField(max_length=512, blank=True)),
                ('notes', models.fields.TextField(blank=True)),
                ('status', models.fields.CharField(max_length=512, blank=True)),
                ('index_version', models.fields.CharField(max_length=512, blank=True)),
                ('verbose_error', models.fields.CharField(max_length=3000, blank=True)),
            ))
            db.send_create_signal('rundb', ['ReferenceGenome'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("ReferenceGenome table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'ThreePrimeadapter'
            db.create_table('rundb_threeprimeadapter', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('direction', models.fields.CharField(default='Forward', max_length=20)),
                ('name', models.fields.CharField(unique=True, max_length=256)),
                ('sequence', models.fields.CharField(max_length=512)),
                ('description', models.fields.CharField(max_length=1024, blank=True)),
                ('qual_cutoff', models.fields.IntegerField()),
                ('qual_window', models.fields.IntegerField()),
                ('adapter_cutoff', models.fields.IntegerField()),
                ('isDefault', models.fields.BooleanField(default=False)),
            ))
            db.send_create_signal('rundb', ['ThreePrimeadapter'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("ThreePrimeadapter table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'PlannedExperiment'
            db.create_table('rundb_plannedexperiment', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('planName', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('planGUID', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('planShortID', models.fields.CharField(max_length=5, null=True, blank=True)),
                ('planExecuted', models.fields.BooleanField(default=False)),
                ('planStatus', models.fields.CharField(max_length=512, blank=True)),
                ('username', models.fields.CharField(max_length=128, null=True, blank=True)),
                ('planPGM', models.fields.CharField(max_length=128, null=True, blank=True)),
                ('date', models.fields.DateTimeField(null=True, blank=True)),
                ('planExecutedDate', models.fields.DateTimeField(null=True, blank=True)),
                ('metaData', models.fields.TextField(default='{}', blank=True)),
                ('chipType', models.fields.CharField(max_length=32, null=True, blank=True)),
                ('chipBarcode', models.fields.CharField(max_length=64, null=True, blank=True)),
                ('seqKitBarcode', models.fields.CharField(max_length=64, null=True, blank=True)),
                ('expName', models.fields.CharField(max_length=128, blank=True)),
                ('usePreBeadfind', models.fields.BooleanField(default=False)),
                ('usePostBeadfind', models.fields.BooleanField(default=False)),
                ('cycles', models.fields.IntegerField(null=True, blank=True)),
                ('flows', models.fields.IntegerField(null=True, blank=True)),
                ('autoAnalyze', models.fields.BooleanField(default=False)),
                ('autoName', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('preAnalysis', models.fields.BooleanField(default=False)),
                ('runType', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('library', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('barcodeId', models.fields.CharField(max_length=256, null=True, blank=True)),
                ('adapter', models.fields.CharField(max_length=256, null=True, blank=True)),
                ('project', models.fields.CharField(max_length=127, null=True, blank=True)),
                ('runname', models.fields.CharField(max_length=255, null=True, blank=True)),
                ('sample', models.fields.CharField(max_length=127, null=True, blank=True)),
                ('notes', models.fields.CharField(max_length=255, null=True, blank=True)),
                ('flowsInOrder', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('libraryKey', models.fields.CharField(max_length=64, null=True, blank=True)),
                ('storageHost', models.fields.CharField(max_length=128, null=True, blank=True)),
                ('reverse_primer', models.fields.CharField(max_length=128, null=True, blank=True)),
                ('bedfile', models.fields.CharField(max_length=1024, blank=True)),
                ('regionfile', models.fields.CharField(max_length=1024, blank=True)),
                ('irworkflow', models.fields.CharField(max_length=1024, blank=True)),
                ('libkit', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('variantfrequency', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('storage_options', models.fields.CharField(default='A', max_length=200)),
                ('reverselibrarykey', models.fields.CharField(max_length=64, null=True, blank=True)),
                ('reverse3primeadapter', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('forward3primeadapter', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('isReverseRun', models.fields.BooleanField(default=False)),
                ('librarykitname', models.fields.CharField(max_length=512, null=True, blank=True)),
                ('sequencekitname', models.fields.CharField(max_length=512, null=True, blank=True)),
            ))
            db.send_create_signal('rundb', ['PlannedExperiment'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("PlannedExperiment table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'Publisher'
            db.create_table('rundb_publisher', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('name', models.fields.CharField(unique=True, max_length=200)),
                ('version', models.fields.CharField(max_length=256)),
                ('date', models.fields.DateTimeField()),
                ('path', models.fields.CharField(max_length=512)),
                ('global_meta', models.fields.TextField(default='{}', blank=True)),
            ))
            db.send_create_signal('rundb', ['Publisher'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Publisher table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'ContentUpload'
            db.create_table('rundb_contentupload', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('file_path', models.fields.CharField(max_length=255)),
                ('status', models.fields.CharField(max_length=255, blank=True)),
                ('meta', models.fields.TextField(default='{}', blank=True)),
                ('publisher', models.fields.related.ForeignKey(to=orm['rundb.Publisher'], null=True)),
            ))
            db.send_create_signal('rundb', ['ContentUpload'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("ContentUpload table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'Content'
            db.create_table('rundb_content', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('publisher', models.fields.related.ForeignKey(
                    related_name='contents', to=orm['rundb.Publisher'])),
                ('contentupload', models.fields.related.ForeignKey(
                    related_name='contents', to=orm['rundb.ContentUpload'])),
                ('file', models.fields.CharField(max_length=255)),
                ('path', models.fields.CharField(max_length=255)),
                ('meta', models.fields.TextField(default='{}', blank=True)),
            ))
            db.send_create_signal('rundb', ['Content'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Content table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'UserEventLog'
            db.create_table('rundb_usereventlog', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('text', models.fields.TextField(blank=True)),
                ('timeStamp', models.fields.DateTimeField(auto_now_add=True, blank=True)),
                ('upload', models.fields.related.ForeignKey(
                    related_name='logs', to=orm['rundb.ContentUpload'])),
            ))
            db.send_create_signal('rundb', ['UserEventLog'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("UserEventLog table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'UserProfile'
            db.create_table('rundb_userprofile', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('user', models.fields.related.ForeignKey(to=orm['auth.User'], unique=True)),
                ('name', models.fields.CharField(max_length=93)),
                ('phone_number', models.fields.CharField(default='', max_length=256, blank=True)),
                ('title', models.fields.CharField(default='user', max_length=256)),
                ('note', models.fields.TextField(default='', blank=True)),
            ))
            db.send_create_signal('rundb', ['UserProfile'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("UserProfile table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'KitInfo'
            db.create_table('rundb_kitinfo', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('kitType', models.fields.CharField(max_length=20)),
                ('name', models.fields.CharField(unique=True, max_length=512)),
                ('description', models.fields.CharField(max_length=3024, blank=True)),
                ('flowCount', models.fields.PositiveIntegerField()),
            ))
            db.send_create_signal('rundb', ['KitInfo'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("KitInfo table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'KitPart'
            db.create_table('rundb_kitpart', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('kit', models.fields.related.ForeignKey(to=orm['rundb.KitInfo'])),
                ('barcode', models.fields.CharField(unique=True, max_length=7)),
            ))
            db.send_create_signal('rundb', ['KitPart'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("KitPart table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'LibraryKey'
            db.create_table('rundb_librarykey', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('direction', models.fields.CharField(default='Forward', max_length=20)),
                ('name', models.fields.CharField(unique=True, max_length=256)),
                ('sequence', models.fields.CharField(max_length=64)),
                ('description', models.fields.CharField(max_length=1024, blank=True)),
                ('isDefault', models.fields.BooleanField(default=False)),
            ))
            db.send_create_signal('rundb', ['LibraryKey'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("LibraryKey table already exists")
            db.clear_deferred_sql()

        try:
            db.start_transaction()
            # Adding model 'Message'
            db.create_table('rundb_message', (
                ('id', models.fields.AutoField(primary_key=True)),
                ('body', models.fields.TextField(default='', blank=True)),
                ('level', models.fields.IntegerField(default=20)),
                ('route', models.fields.TextField(default='', blank=True)),
                ('expires', models.fields.TextField(default='read', blank=True)),
                ('tags', models.fields.TextField(default='', blank=True)),
                ('status', models.fields.TextField(default='unread', blank=True)),
                ('time', models.fields.DateTimeField(auto_now_add=True, blank=True)),
            ))
            db.send_create_signal('rundb', ['Message'])
            db.execute_deferred_sql()
            db.commit_transaction()
        except DatabaseError:
            db.rollback_transaction()
            log.info("Message table already exists")
            db.clear_deferred_sql()

        return

    def backwards(self):
        raise NotImplemented()

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
        }
    }

complete_apps = ['rundb']


def install_missing_tables():
    f = Forwards()
    m = LegacyMigration()
    orm = FakeORM(LegacyMigration, 'rundb')

    log.info("Initiate Legacy Schema 2.2 Migration")
    try:
        m.forwards(orm)
    except:
        log.exception("Errors during Migration")
        if not db.has_ddl_transactions:
            print self.run_migration_error(migration)
            raise
    else:
        log.info("Successful Legacy Schema 2.2 Migration")

if __name__ == '__main__':
    install_missing_tables()
