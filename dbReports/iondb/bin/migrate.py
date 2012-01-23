#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


"""
This script will do PostgreSQL database migrations, when they are needed.
"""

from djangoinit import *
from django import db
from django.db import transaction, IntegrityError, DatabaseError
import traceback
import re
import sys
import os
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
#from django.db import models
from iondb.rundb import models
from datetime import datetime

def q(s):
    cursor = db.connection.cursor()
    cursor.execute(s)
    transaction.commit_unless_managed()
    cursor.close()
        
def add_ftp_status(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_experiment limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == 'ftpStatus':
                return False
        return True

    if check():
        q("""ALTER TABLE rundb_experiment """
          """ADD COLUMN "ftpStatus" character varying(512);""")
        log.write("Added 'ftpStatus' to database\n")

def add_prebead_status(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_experiment limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if 'usePreBeadfind' == col:
                return False
        return True
 
    if check():
        q("""ALTER TABLE rundb_experiment """
          """ADD COLUMN "usePreBeadfind" boolean DEFAULT False;""")
        log.write("Added 'usePreBead' to database\n")

def add_int_field(log,tablename,columnname,default):
    def check(tablename,field): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM %s limit 1" % tablename)
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == field:
                return False
        return True
    
    for field in columnname:
        if check(tablename,field):
            q("""ALTER TABLE %s """
              """ADD COLUMN "%s" integer NOT NULL DEFAULT %d;""" % (tablename,field,default))
            log.write("Added %s to %s\n" % (field, tablename))

def add_char_field(log,tablename,columnname,col_size,default=False):
    def check(tablename,field): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM %s limit 1" % tablename)
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == field:
                return False
        return True

    if check(tablename,columnname):
        if default:
            q("""ALTER TABLE %s """
            """ADD COLUMN %s character varying(%d) DEFAULT '%s';""" % (tablename,columnname,col_size,default))
            log.write("Added %s to %s with default value %s\n" % (columnname, tablename,default))
        else:
            q("""ALTER TABLE %s """
              """ADD COLUMN %s character varying(%d);""" % (tablename,columnname,col_size))
            log.write("Added %s to %s\n" % (columnname, tablename))

def add_text_field(log,tablename,columnname):
    def check(tablename,field):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM %s limit 1" % tablename)
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == field:
                return False
        return True

    if check(tablename,columnname):
        q("""ALTER TABLE %s """
          """ADD COLUMN %s text;""" % (tablename,columnname))
        log.write("Added %s to %s\n" % (columnname, tablename))

def add_bool_field(log,tablename,columnname,default):
    def check(tablename,field): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM %s limit 1" % tablename)
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == field:
                return False
        return True

    for field in columnname:
        if check(tablename,field):
            #TODO: inline
            q("""ALTER TABLE %s """
              """ADD COLUMN "%s" boolean DEFAULT %s;""" % (tablename,field,"TRUE" if default else "FALSE"))
            log.write("Added %s to %s\n" % (field, tablename))

def add_float_field(log,tablename,columnnames,default):
    def check(field,tablename): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM %s limit 1" %tablename)
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == field:
                return False
        return True

    for field in columnnames:
        if check(field,tablename):
            q("""ALTER TABLE "%s" """
              """ADD COLUMN "%s" double precision NOT NULL DEFAULT %f;""" % (tablename,field,default))
            log.write("Added %s to database\n" % field)
            
def add_lib_metrics_int(log):
    def check(tablename):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_libmetrics limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == tablename:
                return False
        return True
    qmetrics = [ 'align_sample',
                 'extrapolated_100q10_reads',
                 'extrapolated_100q17_reads',
                 'extrapolated_100q20_reads',
                 'extrapolated_100q47_reads',
                 'extrapolated_100q7_reads',
                 'extrapolated_200q10_reads',
                 'extrapolated_200q17_reads',
                 'extrapolated_200q20_reads',
                 'extrapolated_200q47_reads',
                 'extrapolated_200q7_reads',
                 'extrapolated_300q10_reads',
                 'extrapolated_300q17_reads',
                 'extrapolated_300q20_reads',
                 'extrapolated_300q47_reads',
                 'extrapolated_300q7_reads',
                 'extrapolated_400q10_reads',
                 'extrapolated_400q17_reads',
                 'extrapolated_400q20_reads',
                 'extrapolated_400q47_reads',
                 'extrapolated_400q7_reads',
                 'extrapolated_50q10_reads',
                 'extrapolated_50q17_reads',
                 'extrapolated_50q20_reads',
                 'extrapolated_50q47_reads',
                 'extrapolated_50q7_reads',
                 'extrapolated_from_number_of_sampled_reads',
                 'extrapolated_mapped_bases_in_q10_alignments',
                 'extrapolated_mapped_bases_in_q17_alignments',
                 'extrapolated_mapped_bases_in_q20_alignments',
                 'extrapolated_mapped_bases_in_q47_alignments',
                 'extrapolated_mapped_bases_in_q7_alignments',
                 'extrapolated_q10_alignments',
                 'extrapolated_q10_longest_alignment',
                 'extrapolated_q10_mean_alignment_length',
                 'extrapolated_q17_alignments',
                 'extrapolated_q17_longest_alignment',
                 'extrapolated_q17_mean_alignment_length',
                 'extrapolated_q20_alignments',
                 'extrapolated_q20_longest_alignment',
                 'extrapolated_q20_mean_alignment_length',
                 'extrapolated_q47_alignments',
                 'extrapolated_q47_longest_alignment',
                 'extrapolated_q47_mean_alignment_length',
                 'extrapolated_q7_alignments',
                 'extrapolated_q7_longest_alignment',
                 'extrapolated_q7_mean_alignment_length',
                 'genomelength',
                 'genomesize',
                 'i100Q10_reads',
                 'i100Q17_reads',
                 'i100Q20_reads',
                 'i100Q47_reads',
                 'i100Q7_reads',
                 'i150Q10_reads',
                 'i150Q17_reads',
                 'i150Q20_reads',
                 'i150Q47_reads',
                 'i150Q7_reads',
                 'i200Q10_reads',
                 'i200Q17_reads',
                 'i200Q20_reads',
                 'i200Q47_reads',
                 'i200Q7_reads',
                 'i250Q10_reads',
                 'i250Q17_reads',
                 'i250Q20_reads',
                 'i250Q47_reads',
                 'i250Q7_reads',
                 'i300Q10_reads',
                 'i300Q17_reads',
                 'i300Q20_reads',
                 'i300Q47_reads',
                 'i300Q7_reads',
                 'i350Q10_reads',
                 'i350Q17_reads',
                 'i350Q20_reads',
                 'i350Q47_reads',
                 'i350Q7_reads',
                 'i400Q10_reads',
                 'i400Q17_reads',
                 'i400Q20_reads',
                 'i400Q47_reads',
                 'i400Q7_reads',
                 'i450Q10_reads',
                 'i450Q17_reads',
                 'i450Q20_reads',
                 'i450Q47_reads',
                 'i450Q7_reads',
                 'i500Q10_reads',
                 'i500Q17_reads',
                 'i500Q20_reads',
                 'i500Q47_reads',
                 'i500Q7_reads',
                 'i50Q10_reads',
                 'i50Q17_reads',
                 'i50Q20_reads',
                 'i50Q47_reads',
                 'i50Q7_reads',
                 'i550Q10_reads',
                 'i550Q17_reads',
                 'i550Q20_reads',
                 'i550Q47_reads',
                 'i550Q7_reads',
                 'i600Q10_reads',
                 'i600Q17_reads',
                 'i600Q20_reads',
                 'i600Q47_reads',
                 'i600Q7_reads',
                 'q10_alignments',
                 'q10_longest_alignment',
                 'q10_mapped_bases',
                 'q10_mean_alignment_length',
                 'q10_qscore_bases',
                 'q17_alignments',
                 'q17_longest_alignment',
                 'q17_mapped_bases',
                 'q17_mean_alignment_length',
                 'q17_qscore_bases',
                 'q20_alignments',
                 'q20_longest_alignment',
                 'q20_mapped_bases',
                 'q20_mean_alignment_length',
                 'q20_qscore_bases',
                 'q47_alignments',
                 'q47_longest_alignment',
                 'q47_mapped_bases',
                 'q47_mean_alignment_length',
                 'q47_qscore_bases',
                 'q7_alignments',
                 'q7_longest_alignment',
                 'q7_mapped_bases',
                 'q7_mean_alignment_length',
                 'q7_qscore_bases',
                 'r100Q10',
                 'r100Q17',
                 'r100Q20',
                 'r200Q10',
                 'r200Q17',
                 'r200Q20',
                 'r50Q10',
                 'r50Q17',
                 'r50Q20',
                 'rLongestAlign',
                 'rMeanAlignLen',
                 'rNumAlignments',
                 's100Q10',
                 's100Q17',
                 's100Q20',
                 's200Q10',
                 's200Q17',
                 's200Q20',
                 's50Q10',
                 's50Q17',
                 's50Q20',
                 'sLongestAlign',
                 'sMeanAlignLen',
                 'sNumAlignments',
                 'sampled_100q10_reads',
                 'sampled_100q17_reads',
                 'sampled_100q20_reads',
                 'sampled_100q47_reads',
                 'sampled_100q7_reads',
                 'sampled_200q10_reads',
                 'sampled_200q17_reads',
                 'sampled_200q20_reads',
                 'sampled_200q47_reads',
                 'sampled_200q7_reads',
                 'sampled_300q10_reads',
                 'sampled_300q17_reads',
                 'sampled_300q20_reads',
                 'sampled_300q47_reads',
                 'sampled_300q7_reads',
                 'sampled_400q10_reads',
                 'sampled_400q17_reads',
                 'sampled_400q20_reads',
                 'sampled_400q47_reads',
                 'sampled_400q7_reads',
                 'sampled_50q10_reads',
                 'sampled_50q17_reads',
                 'sampled_50q20_reads',
                 'sampled_50q47_reads',
                 'sampled_50q7_reads',
                 'sampled_mapped_bases_in_q10_alignments',
                 'sampled_mapped_bases_in_q17_alignments',
                 'sampled_mapped_bases_in_q20_alignments',
                 'sampled_mapped_bases_in_q47_alignments',
                 'sampled_mapped_bases_in_q7_alignments',
                 'sampled_q10_alignments',
                 'sampled_q10_longest_alignment',
                 'sampled_q10_mean_alignment_length',
                 'sampled_q17_alignments',
                 'sampled_q17_longest_alignment',
                 'sampled_q17_mean_alignment_length',
                 'sampled_q20_alignments',
                 'sampled_q20_longest_alignment',
                 'sampled_q20_mean_alignment_length',
                 'sampled_q47_alignments',
                 'sampled_q47_longest_alignment',
                 'sampled_q47_mean_alignment_length',
                 'sampled_q7_alignments',
                 'sampled_q7_longest_alignment',
                 'sampled_q7_mean_alignment_length',
                 'totalNumReads',
                 'total_number_of_sampled_reads',
                 ]

    for field in qmetrics:
        if check(field):
            q("""ALTER TABLE rundb_libmetrics """
              """ADD COLUMN "%s" integer NOT NULL DEFAULT 0;""" % field)
            log.write("Added %s to database\n" % field)

def convert_int_to_float(log):
    qmetrics = ["q47_coverage_percentage"]
    for field in qmetrics:
        q("""ALTER TABLE rundb_libmetrics ALTER "%s" TYPE double precision;""" % field)

def resize_varchar(table, col,length):
    q("""ALTER TABLE %s ALTER COLUMN "%s" TYPE varchar(%s);""" % (table, col, length))

def convert_int_to_bigint(log):
    """Human Genome is large enough to need a bigint column"""
    qmetrics = ["genomesize"]
    for field in qmetrics:
        q("""ALTER TABLE rundb_libmetrics ALTER COLUMN "%s" TYPE bigint;""" % field)

def add_lib_metrics_float(log):
    def check(tablename): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_libmetrics limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == tablename:
                return False
        return True
    qmetrics = ["rCoverage","sCoverage","q10_coverage_percentage",
                "q17_coverage_percentage","q20_coverage_percentage",
                "q7_coverage_percentage","q47_coverage_percentage",
                "cf","ie","dr",
                "sampled_q7_coverage_percentage",
                "sampled_q7_mean_coverage_depth",
                "sampled_q10_coverage_percentage",
                "sampled_q10_mean_coverage_depth",
                "sampled_q17_coverage_percentage",
                "sampled_q17_mean_coverage_depth",
                "sampled_q20_coverage_percentage",
                "sampled_q20_mean_coverage_depth",
                "sampled_q47_coverage_percentage",
                "sampled_q47_mean_coverage_depth",
                "extrapolated_q7_coverage_percentage",
                "extrapolated_q7_mean_coverage_depth",
                "extrapolated_q10_coverage_percentage",
                "extrapolated_q10_mean_coverage_depth",
                "extrapolated_q17_coverage_percentage",
                "extrapolated_q17_mean_coverage_depth",
                "extrapolated_q20_coverage_percentage",
                "extrapolated_q20_mean_coverage_depth",
                "extrapolated_q47_coverage_percentage",
                "extrapolated_q47_mean_coverage_depth"
                ]

    for field in qmetrics:
        if check(field):
            q("""ALTER TABLE rundb_libmetrics """
              """ADD COLUMN "%s" double precision NOT NULL DEFAULT 0;""" % field)
            log.write("Added %s to database\n" % field)
            
def add_lib_metrics_512char(log):
    """this will add a column for the additional alignment metrics"""
    def check(tablename): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_libmetrics limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == tablename:
                return False
        return True
    
    qmetrics = ["Genome_Version","Index_Version","genome"]

    for field in qmetrics:
        if check(field):
            q("""ALTER TABLE rundb_libmetrics """
              """ADD COLUMN "%s" character varying(512);""" % field)
            log.write("Added %s to database\n" % field)
            
            
def add_exp_char(log):
    def check(tablename): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_experiment limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    collist = ['storageHost','barcodeId']
    for field in collist:
        if check(field):
            q("""ALTER TABLE rundb_experiment """
            """ADD COLUMN "%s" character varying(128) NOT NULL DEFAULT '';""" % field)
            log.write("Added %s to database\n" % field)

def add_library_key(log):
    def check(tablename): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_experiment limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = 'libraryKey'
    if check(colname):
        q("""ALTER TABLE rundb_experiment """
          """ADD COLUMN "libraryKey" character varying(64);""")

def add_backup_name(log):
    def check(tablename): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_backup limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = 'backupName'
    if check(colname):
        q("""ALTER TABLE rundb_backup """
          """ADD COLUMN "backupName" character varying(256);""")

def add_analysis_metrics_int(log):
    def check(tablename): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_analysismetrics limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    ametrics = ['ignored','excluded','lib_pass_basecaller',
                'lib_pass_cafie','washout','washout_dud',
                'washout_ambiguous','washout_live',
                'washout_test_fragment','washout_library',
                'keypass_all_beads'
                ]

    for field in ametrics:
        if check(field):
            q("""ALTER TABLE rundb_analysismetrics """
              """ADD COLUMN "%s" integer NOT NULL DEFAULT 0;""" % field)
            log.write("Added %s to database\n" % field)

def add_experiment_int (log):
    def check(tablename): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_experiment limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    exp_int = ['flows']

    for field in exp_int:
        if check(field):
            q("""ALTER TABLE rundb_experiment """
              """ADD COLUMN "%s" integer NOT NULL DEFAULT 0;""" % field)
            log.write("Added %s to database\n" % field)

def add_globalconfig_char(log):
    def check(tablename): 
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_globalconfig limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = ['web_root','default_storage_options','site_name','sfftrim_args']
    for field in colname:
        if check(field):
            q("""ALTER TABLE rundb_globalconfig """
              """ADD COLUMN "%s" character varying(500);""" % field)
            log.write('Added %s to database\n' % field)

def add_plugin_char(log):
    def check(tablename):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_plugin limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = ["project","sample","libraryName","chipType"]
    for field in colname:
        if check(field):
            q("""ALTER TABLE rundb_plugin """
              """ADD COLUMN "%s" character varying(512) NOT NULL DEFAULT '';""" % field)
            log.write('Added %s to database\n' % field)

def add_dnabarcode_char(log):
    def check(tablename):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_dnabarcode limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = ["name","floworder"]
    for field in colname:
        if check(field):
            q("""ALTER TABLE rundb_dnabarcode """
              """ADD COLUMN "%s" character varying(128) NOT NULL DEFAULT '';""" % field)
            log.write('Added %s to database table %s\n' % (field, 'dnabarcode'))

def add_plugin_bool(log):
    def check(tablename):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_plugin limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = ['autorun']
    for field in colname:
        if check(field):
            q("""ALTER TABLE rundb_plugin """
              """ADD COLUMN "%s" boolean NOT NULL DEFAULT TRUE;""" % field)
            log.write('Added %s to database\n' % field)

def add_plugin_config(log):
    def check(tablename):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_plugin limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = 'config'
    if check(colname):
        q("""ALTER TABLE rundb_plugin """
          """ADD COLUMN "config" text;""")

def add_globalconfig_bool(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_globalconfig limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if 'sfftrim' == col:
                return False
        return True

    if check():
        q("""ALTER TABLE rundb_globalconfig """
          """ADD COLUMN "sfftrim" boolean DEFAULT False;""")
        log.write("Added 'sfftrim' to database\n")
        
def add_report_pluginStore(log):
    def check(tablename):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_results limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = 'pluginStore'
    if check(colname):
        q("""ALTER TABLE rundb_results """
          """ADD COLUMN "pluginStore" text;""")

def add_report_pluginState(log):
    def check(tablename):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_results limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = 'pluginState'
    if check(colname):
        q("""ALTER TABLE rundb_results """
          """ADD COLUMN "pluginState" text;""")

def add_report_metaData(log):
    def check(tablename):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_results limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = 'metaData'
    if check(colname):
        q("""ALTER TABLE rundb_results """
          """ADD COLUMN "metaData" text;""")

def experimentFlowOrder(log):
        """allow longer flow orders"""
        # This was original code to increase field from 64 to 512 bytes.
        #q("""ALTER TABLE rundb_experiment ALTER column "flowsInOrder" TYPE char(512); """)
        # Convert field type to character varying (512)
        q("""ALTER TABLE rundb_experiment ALTER column "flowsInOrder" TYPE character varying(512); """)

def add_experiment_metaData(log):
    def check(tablename):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_experiment limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col.lower() == tablename.lower():
                return False
        return True
    colname = 'metaData'
    if check(colname):
        q("""ALTER TABLE rundb_experiment """
          """ADD COLUMN "metaData" text;""")

def add_ref_genome_status(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_referencegenome limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == 'status':
                return False
        return True

    if check():
        q("""ALTER TABLE rundb_referencegenome """
          """ADD COLUMN "status" character varying(512);""")
        log.write("Added 'status' to genome ref table\n")

def add_ref_genome_version(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_referencegenome limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == 'index_version':
                return False
        return True

    if check():
        q("""ALTER TABLE rundb_referencegenome """
          """ADD COLUMN "index_version" character varying(512);""")
        log.write("Added 'index_version' to genome ref table\n")

def remove_ref_genome_pretty_name(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_referencegenome limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == 'pretty_name':
                return True
        return False

    if check():
        q("""ALTER TABLE rundb_referencegenome """
          """DROP COLUMN "pretty_name";""")
        log.write("Added 'pretty_name' to genome ref table\n")

def add_ref_genome_short_name(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_referencegenome limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == 'short_name':
                return False
        return True

    if check():
        q("""ALTER TABLE rundb_referencegenome """
          """ADD COLUMN "short_name" character varying(512);""")
        log.write("Added 'short_name' to genome ref table\n")

def add_ref_genome_error(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_referencegenome limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == 'verbose_error':
                return False
        return True

    if check():
        q("""ALTER TABLE rundb_referencegenome """
          """ADD COLUMN "verbose_error" character varying(512);""")
        log.write("Added 'verbose_error' to genome ref table\n")

#def add_foreign_key (log):
#    def check():
#        cursor = db.connection.cursor()
#       cursor.execute("SELECT * FROM rundb_results limit 1")
#        columnNames = [d[0] for d in cursor.description]
#        cursor.close()
#        for col in columnNames:
#            if col == 'reportstorage_id':
#               return False
#        return True
#
#    if check():
#        q("""ALTER TABLE rundb_results ADD COLUMN reportstorage_id INTEGER;"""
#          """ALTER TABLE rundb_results ADD FOREIGN KEY ("reportstorage_id") REFERENCES "rundb_reportstorage" ("id");""")
#        log.write("Added 'reportstorage_id' to rundb_results table\n")
        
def add_foreign_key (log, tableName, colName, referenceName, fieldType):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM %s limit 1" % (tableName))
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == '%s_id' % colName:
                return False
        return True

    if check():
        q("""ALTER TABLE %s ADD COLUMN "%s_id" %s;"""
          """ALTER TABLE %s ADD FOREIGN KEY ("%s_id") REFERENCES %s (id);""" % (tableName, colName, fieldType, tableName, colName, referenceName))
        log.write("Added '%s_id' to %s table\n" % (colName, tableName))

def add_table_dnabarcode(log):
    try:
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_dnabarcode limit 1")
    except: #TODO: Need specific error that indicates table does not exist
        # Add non existent table
        q("""CREATE TABLE "rundb_dnabarcode () WITH ( OIDS + FALSE ) TABLESPACE pg_default; ALTER TABLE dnabarcode OWNER TO ion;""")
        log.write("Added table rundb_dnabarcode\n")
        

def modify_chip_slot(log):
    chips = models.Chip.objects.all()
    for chip in chips:
        if chip.name == '316':
            chip.slots = 1
            chip.save()
            log.write("316 chip slot set to 1\n")

def add_results_indexes(log):
    # Generate with manage.py sqlindexes
    indexes = [
        """CREATE INDEX "rundb_results_timeStamp" ON "rundb_results" ("timeStamp");""",
        """CREATE INDEX "rundb_results_reportstorage_id" ON "rundb_results" ("reportstorage_id");""",
        """CREATE INDEX "rundb_tfmetrics_name" ON "rundb_tfmetrics" ("name");""",
        """CREATE INDEX "rundb_tfmetrics_name_like" ON "rundb_tfmetrics" ("name" varchar_pattern_ops);""",
    ]
    for idx in indexes:
        try:
            q(idx)
            log.write("Created index: '%s'\n" % idx)
        except (IntegrityError, DatabaseError):
            transaction.rollback()
            log.write("Index: '%s' already exists\n" % idx)
            # Already exists
        except:
            log.write("Failed to create index:\n")
            log.write(traceback.format_exc())

def add_content_file(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM rundb_content limit 1")
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        return "file" not in columnNames

    if check():
        q("""ALTER TABLE rundb_content """
          """ADD COLUMN "file" character varying(255);""")
        log.write("Added 'file' to Content table\n")


def add_upload_to_content(log, tableName, colName, referenceName, fieldType):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM %s limit 1" % (tableName))
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == '%s_id' % colName:
                return False
        return True

    if check():
        q("""ALTER TABLE %s ADD COLUMN "%s_id" %s;"""
          """ALTER TABLE %s ADD FOREIGN KEY ("%s_id") REFERENCES %s (id);""" % (tableName, colName, fieldType, tableName, colName, referenceName))
        log.write("Added '%s_id' to %s table\n" % (colName, tableName))
        q("""UPDATE rundb_content SET contentupload_id = CAST (substring(file from '/results/uploads/BED/([0-9]+)/.*') as integer);""")
        log.write("Set initial Content-ContentUpload foregin keys.")


def new_default_site_name(log):
    def check():
        cursor = db.connection.cursor()
        cursor.execute("SELECT site_name FROM rundb_globalconfig WHERE id=1 limit 1")
        site_name = cursor.fetchone()[0]
        log.write("Site name is '%s'\n" % site_name)
        cursor.close()
        return site_name == '<Set site_name in Global Configs on Admin Tab>'

    if check():
        q("""UPDATE rundb_globalconfig SET site_name='Torrent Server' WHERE id=1""")
        log.write("Changed the Site name from '<Set site_name in Global Configs on Admin Tab>' to 'Torrent Server'\n")


def update_userprofile_names(log, tablename, columnname, col_size, default=""):
    def check(tablename, field):
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM %s limit 1" % tablename)
        columnNames = [d[0] for d in cursor.description]
        cursor.close()
        for col in columnNames:
            if col == field:
                return False
        return True

    if check(tablename, columnname):
        q("""ALTER TABLE %s ADD COLUMN %s character varying(%d) DEFAULT '%s';
        """ % (tablename, columnname, col_size, default))
        log.write("Added %s to %s with default value %s\n" % (columnname,
                                                              tablename,
                                                              default))
        q("""UPDATE rundb_userprofile SET name = first_name
        FROM auth_user WHERE user_id = auth_user.id;""")


def move_plugin_reports(log):
    # Plugin table added by syncdb
    def hasPluginStateStore():
        cursor = db.connection.cursor()
        cursor.execute("""SELECT * FROM rundb_results limit 1""")
        columnNames = [d[0].lower() for d in cursor.description]
        cursor.close()
        return ('pluginstate' in columnNames and 'pluginstore' in columnNames)

    if not hasPluginStateStore():
        return
    # Monkey patch Results class
    from iondb.rundb import json_field
    models.Results.add_to_class('pluginState', json_field.JSONField(blank=True))
    models.Results.add_to_class('pluginStore', json_field.JSONField(blank=True))

    from distutils.version import StrictVersion as V
    def getLatestPlugin(pluginName):
        latestVersion = '0.0'
        latestPlugin = None
        for p in models.Plugin.objects.filter(name=pluginName):
            if latestPlugin:
                if V(p.version) > V(latestPlugin.version):
                    latestPlugin = p
            else:
                latestPlugin = p
        return latestPlugin

    # hash by name of latest version of every plugin 
    # (assumes plugin result came from latest version)
    plugins = {}
    for p in models.Plugin.objects.all():
        if p.name in plugins:
            # skip if we already have a plugin and this one is inactive
            if not p.active:
                continue
            # Skip if plugin we already have is newer
            if V(p.version) < V(plugins[p.name].version):
                continue
            log.write("Duplicate Plugin Name: '%s' Version='%s' replaces Version='%s'\n" % (p.name, p.version, plugins[p.name].version))
        plugins[p.name] = p

    count_new = 0
    count_update = 0
    count_errors = 0

    # Fixup invalid states - tuple of tuples
    ALLOWED_STATES = [ x[0] for x in models.PluginResult.ALLOWED_STATES ]

    re_error = re.compile(r"error|fail|false", re.I)
    re_complete = re.compile(r"complete|true|success", re.I)

    # Move status and results over
    for result in models.Results.objects.all():
        # django parses the json to dict for us
        pluginState = result.pluginState
        pluginStore = result.pluginStore

        for pluginName in set.union(set(pluginState.keys()), set(pluginStore.keys())):
            try:
                state = pluginState[pluginName]
            except KeyError:
                state = 'Error'
            try:
                store = pluginStore[pluginName]
            except KeyError:
                store = None

            # Restrict state field to a limited set of values
            if state not in ALLOWED_STATES:
                state = str(state)
                if state == '0':
                    state = 'Error'
                elif state == '1':
                    state = 'Complete'
                elif re_error.search(state):
                    state = 'Error'
                elif re_complete.search(state):
                    state = 'Complete'
                elif state == '':
                    """ Unknown is now a valid state """
                    state = 'Unknown'
                else:
                    log.write("Unknown State: '%s' = '%s'\n" % (pluginName, state))
                    # Truncate long strings to fit in 20 char field
                    if len(state) > 20:
                        state = (string[:18] + "+")

            if pluginName not in plugins:
                log.write("No plugin record found for: '" + str(pluginName) + "' Creating placeholder.\n")
                new_plugin = models.Plugin(name=pluginName,version='0',
                                           path='',date=datetime.now(),
                                           active=False,selected=False,autorun=False)
                new_plugin.save()
                plugins[pluginName] = new_plugin

            plugin = plugins[pluginName]

            # Update existing records
            with transaction.commit_on_success():
                (pluginreport,created) = models.PluginResult.objects.get_or_create(plugin=plugin,result=result)
                if created:
                    count_new += 1
                else:
                    count_update += 1

                pluginreport.state = state
                pluginreport.store = store

                try:
                    pluginreport.save()
                except:
                    log.write("ERROR: %s => '%s' : '%s'\n" % (pluginName, state, store))
                    log.write(traceback.format_exc())
                    count_errors += 1

    if count_new or count_update:
        log.write('Migrated %d plugin results from Results to PluginReport. (%d new, %d updated)\n' % (count_new + count_update, count_new, count_update))

    if count_errors:
        log.write('ERROR: Found %d invalid records. Original values recorded in migrate.log.' % count_errors)

    ## Remove columns!
    q("""ALTER TABLE rundb_results DROP COLUMN "pluginState", DROP COLUMN "pluginStore";""")

if __name__ == '__main__':
    hasdb = False
    try:
        cursor = db.connection.cursor()
        hasdb = True
        cursor.close()
    except:
        hasdb=False
        print 'No database found'
    if hasdb:
        print "There is a database, trying to do the migration"

        try:
            f = open('/var/log/ion/migrate.log', 'w')
        except:
            f = open('migrate.log', 'w')
        #add default location check box
        add_bool_field(f, "rundb_location", ["defaultlocation"], False)
        add_ftp_status(f)
        add_prebead_status(f)
        add_lib_metrics_int(f)
        add_lib_metrics_float(f)
        add_lib_metrics_512char(f)
        add_library_key(f)
        add_backup_name(f)
        add_analysis_metrics_int(f)
        add_globalconfig_char(f)
        add_globalconfig_bool(f)
        convert_int_to_float(f)
        add_exp_char(f)
        #add_report_pluginStore(f) ## Note removed below
        #add_report_pluginState(f) ## Note removed below
        add_report_metaData(f)
        add_experiment_metaData(f)
        add_ref_genome_status(f)
        add_ref_genome_version(f)
        add_ref_genome_error(f)
        remove_ref_genome_pretty_name(f)
        add_ref_genome_short_name(f)
        add_experiment_int(f)
        add_plugin_char(f)
        add_plugin_bool(f)
        add_plugin_config(f)
        add_table_dnabarcode(f)
        add_dnabarcode_char(f)
        experimentFlowOrder(f)
        add_content_file(f)
        new_default_site_name(f)

        add_int_field(f, "rundb_backupconfig", ["grace_period"], 72)
        add_char_field(f, "rundb_rig", "ftpserver", 128, 'ts')
        add_char_field(f, "rundb_rig", "ftpusername", 64, 'ionguest')
        add_char_field(f, "rundb_rig", "ftppassword", 64, 'ionguest')
        add_char_field(f, "rundb_rig", "updatehome", 256, 'ts')
        add_char_field(f, "rundb_rig", "ftprootdir", 64, '/results')
        add_bool_field(f, "rundb_rig", ["updateflag"], False)


        add_int_field(f, "rundb_dnabarcode", ["index"], 0)
        add_char_field(f, "rundb_dnabarcode", "annotation", 512, '')
        add_char_field(f, "rundb_dnabarcode", "adapter", 128, '')
        add_int_field(f,"rundb_dnabarcode",['score_mode'],0)
        add_float_field(f,"rundb_dnabarcode",['score_cutoff'],0)
        add_char_field(f, "rundb_dnabarcode", "id_str", 128, '')
        
        add_char_field(f, "rundb_experiment", "reverse_primer", 128, '')
        add_char_field(f, "rundb_experiment", "user_ack", 24, 'U')
        
        #add rig status data
        rig_status_char = ["state", "last_init_date", "last_clean_date", "last_experiment"]
        rig_status_text = ["version","alarms"]

        for rig_field in rig_status_char:
            add_char_field(f,"rundb_rig",rig_field, 512)

        for rig_field in rig_status_text:
            add_text_field(f,"rundb_rig",rig_field)

        #add rig serial number
        add_char_field(f,"rundb_rig","serial", 24)

        add_int_field(f, "rundb_threeprimeadapter", ["qual_cutoff","qual_window","adapter_cutoff"], 0)

        add_char_field (f, "rundb_experiment", "rawdatastyle", 24, 'single')
        
        add_foreign_key(f, 'rundb_results', 'reportstorage', 'rundb_reportstorage', 'INTEGER')
        add_bool_field(f,"rundb_globalconfig", ["auto_archive_ack"], False)

        add_bool_field(f,"rundb_plannedexperiment", ["preAnalysis"], False)

        #add bed file
        add_char_field(f,"rundb_plannedexperiment", "bedfile", 1024 )
        #add region file
        add_char_field(f,"rundb_plannedexperiment", "regionfile", 1024 )
        add_char_field(f,"rundb_plannedexperiment", "libkit", 512 )
        add_char_field(f,"rundb_plannedexperiment", "variantfrequency", 512 )

        #expand charvar lengths
        resize_varchar("rundb_plannedexperiment", "project", "127")
        resize_varchar("rundb_plannedexperiment", "sample", "127")
        resize_varchar("rundb_plannedexperiment", "runname", "255")
        resize_varchar("rundb_plannedexperiment", "notes", "255")

        add_char_field(f,"rundb_experiment", "sequencekitname", 512)
        add_char_field(f,"rundb_experiment", "sequencekitbarcode", 512)
        add_char_field(f,"rundb_experiment", "librarykitname", 512)
        add_char_field(f,"rundb_experiment", "librarykitbarcode", 512)

        # UserProfile
        update_userprofile_names(f, "rundb_userprofile", "name", 93, "")

        #add status field to plugins this will house data about the download status
        #making this JSON to store any error messages from celery
        add_text_field(f,"rundb_plugin", "status")

        #add runtype meta
        add_text_field(f,"rundb_runtype", "meta")

        modify_chip_slot(f)

        # Plugin inactivation and feed url
        add_char_field(f, "rundb_plugin", "url", 256)
        add_bool_field(f, "rundb_plugin", ["active"], True)
        # Mark any missing plugins as inactive
        q("""UPDATE rundb_plugin SET active=False WHERE path='';""");

        # Extend Content to track it's upload
        add_upload_to_content(f, 'rundb_content', 'contentupload', 'rundb_contentupload', 'INTEGER')
        add_text_field(f,"rundb_contentupload", "meta")


        # Call ./manage.py syncdb
        from django.core import management
        management.call_command('syncdb')

        #change int to bigint for human genome size
        convert_int_to_bigint(f)

        move_plugin_reports(f)

        add_results_indexes(f)

        # Should run VACUUM ANALYZE to use indexes after this...
        # But it cannot be run in a transaction,
        # and django connections are # automatically wrapped in transactions...

        f.close()
        
    sys.exit(0)

