#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
from django import db
from django.db import transaction
import sys
import os
import subprocess
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

from django.db import models
from iondb.rundb import models
from socket import gethostname
from django.contrib.auth.models import User

int_test_file = "/opt/ion/.ion-internal-server"

def add_user(username,password):
    
    try:
        user_exists = User.objects.get(username=username)
        print "User", username, "already existed"
        return None
    except:
        print username, "added"
        user = User.objects.create_user(username,"ionuser@iontorrent.com",password)
        user.is_staff = True
        user.save()
        return user

def addScript(filePath):
    '''Adds a new top level script `TLScript.py` to the database.
    This function does not care if one exists.  The reason is that 
    we continually update the `TLScript.py` and want to over write 
    the existing one at every update'''
    f = open(filePath)
    data = f.read()
    f.close()
    script = models.RunScript.objects.all()
    if len(script) > 0:
        script = script[0]
        script.script = data
        script.save()
    else:
        script = models.RunScript(name="TopLevelScript", script=data)
        script.save()
    print "Saved New Top Level Script"

def mungeScript(filePath):
    '''We hack up the script here for R&D purposes'''
    # We are including the -p 1 option to alignmentQC.sh
    #com = "sed -i 's/alignmentQC.pl --input/alignmentQC.pl -p 1 --input/g' %s" % filePath
    com = "sed -i 's/sam_parsed = False/sam_parsed = True/' %s" % filePath
    try:
        os.system(com)
        print "R&D mod to TLScript carried out"
    except:
        print "Could not hack up the TLScript.py.  Sorry Rob."
                
    
def add_location():
    '''Checks if a location exists and creates a default location
    called `Home` if none exist. '''
    loc = models.Location.objects.all()
    if len(loc) > 0:
        print "Location exists: %s" % loc[0].name
    else:
        loc = models.Location(name="Home",defaultlocation=True)
        loc.save()
        print "Location Saved"

def add_fileserver():
    '''Creates a default fileserver called `Home` with location
    `/results/` if one does not exist'''
    fs = models.FileServer.objects.all()
    if len(fs) > 0:
        print "FileServer exists: %s" % fs[0].name
    else:
        fs = models.FileServer(name="Home", 
                               filesPrefix='/results/', 
                               location=models.Location.objects.all()[0])
        fs.save()
        print "Fileserver added"

def reset_report_path(path):
    """ NOT USED"""
    def remove_postfix(url):
        return "".join(url.strip().split('//')[1].strip().split('/')[1:])
        
    report_list = models.Results.objects.all()
    for report in report_list:
        report.reportLink = path.join(path, remove_webpath(report.reportLink))
        report.sffLink = path.join(path, remove_webpath(report.sffLink))
        report.fastqLink = path.join(path, remove_webpath(report.fastqLink))
        report.tfFastq = path.join(path, remove_webpath(report.tfFastq))
        report.log = path.join(path, remove_webpath(report.log))
        report.save()
    print 'Updated all Reports to %s' % path

def add_reportstorage():
    '''Adds a generic-default report storage location.  Also, for legacy 
    installs this function would strip the original full path (http://somehting...)
    and make it a relative path.'''
    rs = models.ReportStorage.objects.all()
    if len(rs)>0:
        rs = rs[0]
        print "ReportStorage exists: %s" % rs.name
        if 'http' in rs.webServerPath:
            rs.webServerPath = '/' + rs.webServerPath.strip().split('/')[-1]
            rs.save()
            print 'Webserver path set to: %s' % rs.webServerPath
    else:
        hoststring = "/output"
        rs = models.ReportStorage(name="Home",
                                  webServerPath=hoststring,
                                  dirPath="/results/analysis/output")
        rs.save()
        print "Reportstorage added"

def add_backupconfig():
    '''Creates a backup configuration with default values
    if one doesn't exist.'''
    bk = models.BackupConfig.objects.all()
    if len(bk)>0:
        print 'BackupConfig exists: %s' % bk[0].name
    else:
        kwargs = {'name':'Archive', 
                  'location': models.Location.objects.all()[0],
                  'backup_directory': '',
                  'backup_threshold': 90,
                  'number_to_backup': 10,
                  'grace_period':72,
                  'timeout': 60,
                  'bandwidth_limit': 0,
                  'status': '-',
                  'online': True,
                  'comments': '',
                  'email': ''
                  }
        bk = models.BackupConfig(**kwargs)
        bk.save()
        print 'BackupConfig added'

def add_chips():
    '''Sets the per chip default analysis args into the 
    `chips` table in the database.  Also creates a chip 
    called `takeover` that will force 2 slots to be used
    and therefore using an entire server.  '''
    try:
        # Determine slots by number of cpu sockets installed
        p1 = subprocess.Popen("/usr/bin/lscpu",stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["grep","^CPU socket"],stdin=p1.stdout,stdout=subprocess.PIPE)
        sockets = p2.stdout.read().strip().split(":")[1]
        sockets = int(sockets)
    except:
        sockets = 2
        print traceback.format_exc()
        
    chip_to_slots = (('318',sockets,''),
                     ('316',1,''),
                     ('314',1,''),
                     ('takeover',sockets,''),
                     )
    chips = [c.name for c in models.Chip.objects.all()]
    # Add any chips that might not be in the database
    # (this case is only on TS initialization)
    for (name,slots,args) in chip_to_slots:
        if name not in chips:
            c = models.Chip(name=name,
                            slots=slots,
                            args=args
                            )
            c.save()
    # make sure all chips in database have the above settings
    # (this case when updating TS typically)
    for (name,slots,args) in chip_to_slots:
        try:
            #print "Chip: %s Slots: %d" % (name,slots)
            c = models.Chip.objects.get(name=name)
            c.slots = slots
            c.args  = args
            c.save()
        except:
            print "Could not find a chip named %s.  This is rubbish.  Should never get here." % name
    return

def add_global_config():
    gc = models.GlobalConfig.objects.all()
    defaultArg = 'Analysis'
    defaultStore = 'A'
    if len(gc)>0:
        gc = gc[0]
        print 'GlobalConfig exists: %s' % gc.name
        if os.path.isfile (int_test_file):
            if gc.default_command_line != defaultArg:
                gc.default_command_line = defaultArg
                gc.save()
                print "Updated default arg to %s" % defaultArg
            else:
                pass

        else:
            print "Skipping Command Line default update"

        if gc.sfftrim_args == None:
            gc.sfftrim_args = '' 
            gc.save()
            print "Updated sfftrim_args to blank default" 

    else:
        kwargs = {'name':'Config', 
                  'selected':False,
                  'plugin_folder':'plugins',
                  'default_command_line':defaultArg,
                  'fasta_path':'',
                  'reference_path':'',
                  'records_to_display':20,
                  'default_test_fragment_key':'ATCG',
                  'default_library_key':'TCAG',
                  'default_flow_order':'TACG',
                  'plugin_output_folder':'plugin_out',
                  'default_plugin_script':'launch.sh',
                  'web_root':'',
                  'site_name':'Torrent Server',
                  'default_storage_options':defaultStore,
                  'sfftrim': False,
                  'sfftrim_args': '',
                  'auto_archive_ack':False,
                  }
        gc = models.GlobalConfig(**kwargs)
        gc.save()
        print 'GlobalConfig added'


def runtype_add(type,description):
    """Helper function to add runtypes"""

    rt = models.RunType.objects.filter(runType=type)

    if rt:
        print "RunType" , type, "exists"
    else:
        rt = models.RunType(runType=type,description=description)
        rt.save()
        print type, " RunType added"

def VariantFrequencies_add(name):
    """Helper function to add VariantFrequencies"""

    vf = models.VariantFrequencies.objects.filter(name=name)

    if vf:
        print "VariantFrequency" , vf[0],  "exists"
    else:
        vf = models.VariantFrequencies(name=name)
        vf.save()
        print name, " VariantFrequency added"

def add_runtype():
    """Create a generic runtype if it does not exist"""""

    #blow away any existing run types
    runtypes = models.RunType.objects.all()
    for runtype in runtypes:
        runtype.delete()

    runtype_add("GENS","Generic Sequencing")
    runtype_add("AMPS","AmpliSeq")
    runtype_add("TARS","TargetSeq")
    runtype_add("WGNM","Whole Genome")
    VariantFrequencies_add("Germ Line")
    VariantFrequencies_add("Somatic")


def create_default_pgm(pgmname,comment=''):
    pgms = models.Rig.objects.all()
    for pgm in pgms:
        if pgm.name == pgmname:
            print "PGM named %s already exists" % pgmname
            return True
    locs = models.Location.objects.all()
    if locs:
        loc = locs[locs.count() - 1]
        pgm = models.Rig(name=pgmname,
                         location=loc,
                         comments=comment)
        pgm.save()
    else:
        print "Error: No Location object exists in database!"
        return 1

def add_ion_xpress_dnabarcode_set():
    '''List from jira wiki page:
    https://iontorrent.jira.com/wiki/display/MOLBIO/Ion+Xpress+Barcode+Adapters+1-16+Kit+-+Lot+XXX
    '''
    blist=['CTAAGGTAAC',    #1
           'TAAGGAGAAC',    #2
           'AAGAGGATTC',    #3
           'TACCAAGATC',    #4
           'CAGAAGGAAC',    #5
           'CTGCAAGTTC',    #6
           'TTCGTGATTC',    #7
           'TTCCGATAAC',    #8
           'TGAGCGGAAC',    #9
           'CTGACCGAAC',    #10
           'TCCTCGAATC',    #11
           'TAGGTGGTTC',    #12
           'TCTAACGGAC',    #13
           'TTGGAGTGTC',    #14
           'TCTAGAGGTC',    #15
           'TCTGGATGAC',    #16
           'TCTATTCGTC',    #17
           'AGGCAATTGC',    #18
           'TTAGTCGGAC',    #19
           'CAGATCCATC',    #20
           'TCGCAATTAC',    #21
           'TTCGAGACGC',    #22
           'TGCCACGAAC',    #23
           'AACCTCATTC',    #24
           'CCTGAGATAC',    #25
           'TTACAACCTC',    #26
           'AACCATCCGC',    #27
           'ATCCGGAATC',    #28
           'TCGACCACTC',    #29
           'CGAGGTTATC',    #30
           'TCCAAGCTGC',    #31
           'TCTTACACAC',    #32
           ]
    btype='none'
    name='IonXpress'
    adapter = 'GAT'
    
    # Attempt to read dnabarcode set named 'IonXpress' from dbase
    dnabcs = models.dnaBarcode.objects.filter(name=name)
    if len(dnabcs) > 0:
        print '%s dnaBarcode Set exists in database' % name
        # make sure we have all the sequences we expect
        for index,sequence in enumerate(blist,start=1):
            # Search for this sequence in the list of barcode records
            bc_found = dnabcs.filter(sequence=sequence)
            if len(bc_found) > 1:
                print "ERROR: More than one entry with sequence %s" % sequence
                print "TODO: Fix this situation, Mr. Programmer!"
            if len(bc_found) == 1:
                
                # Make sure floworder field is not 'none'
                if bc_found[0].floworder == 'none':
                    bc_found[0].floworder = ''
                    
                # Make sure id string has zero padded index field
                bc_found[0].id_str = '%s_%03d' % (name,index)
                
                # Save changes to database
                bc_found[0].save()
                
            else:   # array length is zero
                print "Adding entry for %s" % sequence
                kwargs = {
                    'name':name,
                    'id_str':'%s_%03d' % (name,index),
                    'sequence':sequence,
                    'type':btype,
                    'length':len(sequence),
                    'floworder':'',
                    'index':index,
                    'annotation':'',
                    'adapter':adapter,
                    'score_mode':1,
                    'score_cutoff':2.0,
                }
                ret = models.dnaBarcode(**kwargs)
                ret.save()
    else:
        # Add the barcodes because they do not exist.
        # NOTE: name for id_str
        for index,sequence in enumerate(blist,start=1):
            kwargs = {
                'name':name,
                'id_str':'%s_%03d' % (name,index),
                'sequence':sequence,
                'type':btype,
                'length':len(sequence),
                'floworder':'',
                'index':index,
                'annotation':'',
                'adapter':adapter,
                'score_mode':1,
                'score_cutoff':2.0,
            }
            ret = models.dnaBarcode(**kwargs)
            ret.save()
        print '%s dnaBarcode Set added to database' % name
    return
    
def add_ion_dnabarcode_set():
    '''List from TS-1517 or, file Barcodes_052611.xlsx
    Added extra T to start of each sequence'''
    blist=[
        "TACTCACGATA",
        "TCGTGTCGCAC",
        "TGATGATTGCC",
        "TCGATAATCTT",
        "TCTTACACCAC",
        "TAGCCAAGTAC",
        "TGACATTACTT",
        "TGCCTTACCGC",
        "TACCGAGGCAC",
        "TGCAAGCCTTC",
        "TACATTACATC",
        "TCAAGCACCGC",
        "TAGCTTACCGC",
        "TCATGATCAAC",
        "TGACCGCATCC",
        "TGGTGTAGCAC"]
    btype='none'
    name='IonSet1'
    adapter = 'CTGCTGTACGGCCAAGGCGT'
    
    # Check for barcode set named 'ionSet1' and remove it
    # this is slightly different than desired name: 'IonSet1'
    allbarcodes = models.dnaBarcode.objects.filter(name='ionSet1')
    if allbarcodes:
        allbarcodes.all().delete()
        
    # Check for barcode set named 'IonSet1'
    allbarcodes = models.dnaBarcode.objects.filter(name=name)
    if allbarcodes:
        allbarcodes.all().delete()
    
    # Add the IonSet1 default barcodes
    index = 1
    for i in blist:
        kwargs = {
            'name':name,
            'id_str':'%s_%02d' % (name,index),
            'sequence':i,
            'type':btype,
            'length':len(i),
            'floworder':'',
            'index':index,
            'annotation':'',
            'adapter':adapter,
            'score_mode':0,
            'score_cutoff':0.90,
        }
        ret = models.dnaBarcode(**kwargs)
        ret.save()
        index += 1

def ensure_dnabarcodes_have_id_str():
    #For the 1.5 release, we are adding the id_str field to each dnabarcode record.
    allbarcodes = models.dnaBarcode.objects.all()
    if not allbarcodes:
    	return
    for bc in allbarcodes:
        if bc.id_str == None or bc.id_str == "":
            bc.id_str = "%s_%s" % (bc.name,str(bc.index))
            bc.save()
    return
    
def add_ThreePrimeadapter ():
    '''Ion default 3' adapter'''
    name = 'Ion Kit'
    sequence = 'ATCACCGACTGCCCATAGAGAGGCTGAGAC'
    description = 'Default adapter'
    qual_cutoff = 9
    qual_window = 30
    adapter_cutoff = 16
    
    # There should only be one query result object
    try:
        adapter = models.ThreePrimeadapter.objects.get(name=name)
    except:
        adapter = None
    if not adapter:
        print "Adding %s adapter" % name
        kwargs = {
            'name':name,
            'sequence':sequence,
            'description':description,
            'qual_cutoff':qual_cutoff,
            'qual_window':qual_window,
            'adapter_cutoff':adapter_cutoff
        }
        ret = models.ThreePrimeadapter(**kwargs)
        ret.save()
    else:
        #print "We have %s sequence %s" % (adapter.name,adapter.sequence)
        adapter.sequence = sequence
        adapter.description = description
        adapter.qual_cutoff = qual_cutoff
        adapter.qual_window = qual_window
        adapter.adapter_cutoff = adapter_cutoff
        adapter.save()
    
def add_library_kit(name,description,sap):
    try:
        kit = models.LibraryKit.objects.get(name=name)
    except:
        kit = None
    if not kit:
        kwargs = {
            'name':name,
            'description':description,
            'sap':sap
        }
        obj = models.LibraryKit(**kwargs)
        obj.save()
    else:
        kit.description = description
        kit.sap = sap
        kit.save()
        
def add_sequencing_kit(name,description,sap):
    try:
        kit = models.SequencingKit.objects.get(name=name)
    except:
        kit = None
    if not kit:
        kwargs = {
            'name':name,
            'description':description,
            'sap':sap
        }
        obj = models.SequencingKit(**kwargs)
        obj.save()
    else:
        kit.description = description
        kit.sap = sap
        kit.save()
    
    
if __name__=="__main__":
    import sys
    import traceback
    def check_bool(value):
        if value.lower() == 'true':
            return True
        else:
            return False
    print 'Install Script run with command args %s' % ' '.join(sys.argv)
    try:
        cursor = db.connection.cursor()
        cursor.close()
    except:
        print 'No database found'
        print traceback.format_exc()
        sys.exit(1)

    try:
        if os.path.isfile (int_test_file):
            # Modify the TLScript.py for R&D internal use
            mungeScript(sys.argv[1])
            
        addScript(sys.argv[1])
    except:
        print 'Top Level script Failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        add_chips()
    except:
        print 'Adding Chips Failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        add_location()
    except:
        print 'Adding Location Failed'
        print traceback.format_exc()
        sys.exit(1)
    try:
        add_fileserver()
    except:
        print 'Adding File Server Failed'
        print traceback.format_exc()
        sys.exit(1)
    try:
        add_reportstorage()
    except:
        print 'Adding Report Storage Failed'
        print traceback.format_exc()
        sys.exit(1)
    try:
        add_global_config()
    except:
        print 'Adding Global Config Failed'
        print traceback.format_exc()
        sys.exit(1)
    try:
        add_user("ionuser","ionuser")
    except:
        print 'Adding ionuser failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        # TODO: for these users, set_unusable_password()
        # These users exists only to uniformly store records of their contact
        # information for customer support.
        lab = add_user("lab_contact","lab_contact")
        if lab is not None:
            lab_profile = lab.get_profile()
            lab_profile.title = "Lab Contact"
            lab_profile.save()
        it = add_user("it_contact","it_contact")
        if it is not None:
            it_profile = it.get_profile()
            it_profile.title = "IT Contact"
            it_profile.save()
    except:
        print 'Adding special users failed'
        print traceback.format_exc()
        sys.exit(1)

    #try to add GENS runType
    try:
        add_runtype()
    except:
        print 'Adding GENS runType failed'
        print traceback.format_exc()
        sys.exit(1)


    #try to add PGMs
    try:
        create_default_pgm('default',comment = "This is a model PGM.  Do not delete.")
        #causes errors if system is not completely configured and auto-analysis
        # kicks off.
        #create_default_pgm('PGM_test',comment = "This is a test pgm for staging sample datasets.")
    except:
        print 'Adding default PGM failed'
        print traceback.format_exc()
        sys.exit(1)
        
    try:
        add_ion_dnabarcode_set()
        ensure_dnabarcodes_have_id_str()	# for existing barcode records
    except:
        print 'Adding dnaBarcodeset failed'
        print traceback.format_exc()
        sys.exit(1)
        
    try:
    	add_ion_xpress_dnabarcode_set()        
    except:
        print 'Adding dnaBarcodeset: IonXpress failed'
        print traceback.format_exc()
        sys.exit(1)
        
    try:
    	add_ThreePrimeadapter()
    except:
        print "Adding 3' adapter failed"
        print traceback.format_exc()
        sys.exit(1)
        
    try:
    	add_backupconfig()
    except:
        print "Adding backup configuration failed"
        print traceback.format_exc()
        sys.exit(1)
    
    try:
        add_library_kit('IonLibKit','Ion Plus Fragment Library Kit','4471252')
    except:
        print "Adding library_kit failed"
        print traceback.format_exc()
        sys.exit(1)
    
    try:
        add_sequencing_kit('IonSeqKit','Ion Sequencing Kit','4468997')
        add_sequencing_kit('IonSeq200Kit','Ion Sequencing 200 Kit','4471258')
    except:
        print "Adding sequencing_kit failed"
        print traceback.format_exc()
        sys.exit(1)
        