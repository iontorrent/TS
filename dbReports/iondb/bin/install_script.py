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

from django.core.exceptions import ValidationError

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
    `chips` table in the database.  '''
    try:
        # Determine slots by number of cpu sockets installed
        p1 = subprocess.Popen("/usr/bin/lscpu",stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["grep","^CPU socket"],stdin=p1.stdout,stdout=subprocess.PIPE)
        sockets = p2.stdout.read().strip().split(":")[1]
        sockets = int(sockets)
    except:
        sockets = 2
        print traceback.format_exc()
        
    chip_to_slots = (('318',1,''),
                     ('316',1,''),
                     ('314',1,''),
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
    
    # Remove the special chip labelled 'takeover'; no longer used.
    try:
        c = models.Chip.objects.get(name='takeover')
        c.delete()
        print "Deleted Chip object named 'takeover'"
    except:
        pass
    return

def add_global_config():
    gc = models.GlobalConfig.objects.all()
    #defaultArg = 'Analysis --wellsfileonly'
    defaultArg = 'Analysis'
    defaultBaseCallerArg = 'BaseCaller'
    defaultStore = 'A'
    if len(gc)>0:
        gc = gc[0]
        print 'GlobalConfig exists: %s' % gc.name
        if not os.path.isfile (int_test_file):
            if gc.default_command_line != defaultArg:
                gc.default_command_line = defaultArg
                gc.save()
                print "Updated default arg to %s" % defaultArg
            else:
                pass

        if gc.basecallerargs != defaultBaseCallerArg:
            gc.basecallerargs = defaultBaseCallerArg
            gc.save()
            print "Updated default basecallerargs to %s" % defaultBaseCallerArg


    else:
        kwargs = {'name':'Config', 
                  'selected':False,
                  'plugin_folder':'plugins',
                  'default_command_line':defaultArg,
                  'basecallerargs':'BaseCaller',
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

def add_or_update_barcode_set(blist,btype,name,adapter):
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
           'TTCTCATTGAAC',    #33
           'TCGCATCGTTC',    #34
           'TAAGCCATTGTC',    #35
           'AAGGAATCGTC',    #36
           'CTTGAGAATGTC',    #37
           'TGGAGGACGGAC',    #38
           'TAACAATCGGC',    #39
           'CTGACATAATC',    #40
           'TTCCACTTCGC',    #41
           'AGCACGAATC',    #42
           'CTTGACACCGC',    #43
           'TTGGAGGCCAGC',    #44           
           'TGGAGCTTCCTC',    #45
           'TCAGTCCGAAC',    #46
           'TAAGGCAACCAC',    #47
           'TTCTAAGAGAC',    #48
           'TCCTAACATAAC',    #49
           'CGGACAATGGC',    #50
           'TTGAGCCTATTC',    #51
           'CCGCATGGAAC',    #52
           'CTGGCAATCCTC',    #53
           'CCGGAGAATCGC',    #54
           'TCCACCTCCTC',    #55
           'CAGCATTAATTC',    #56
           'TCTGGCAACGGC',    #57
           'TCCTAGAACAC',    #58
           'TCCTTGATGTTC',    #59
           'TCTAGCTCTTC',    #60           
           'TCACTCGGATC',    #61
           'TTCCTGCTTCAC',    #62
           'CCTTAGAGTTC',    #63
           'CTGAGTTCCGAC',    #64
           'TCCTGGCACATC',    #65
           'CCGCAATCATC',    #66
           'TTCCTACCAGTC',    #67
           'TCAAGAAGTTC',    #68
           'TTCAATTGGC',    #69
           'CCTACTGGTC',    #70           
           'TGAGGCTCCGAC',    #71
           'CGAAGGCCACAC',    #72
           'TCTGCCTGTC',    #73
           'CGATCGGTTC',    #74
           'TCAGGAATAC',    #75
           'CGGAAGAACCTC',    #76
           'CGAAGCGATTC',    #77
           'CAGCCAATTCTC',    #78
           'CCTGGTTGTC',    #79
           'TCGAAGGCAGGC',    #80           
           'CCTGCCATTCGC',    #81
           'TTGGCATCTC',    #82
           'CTAGGACATTC',    #83
           'CTTCCATAAC',    #84
           'CCAGCCTCAAC',    #85
           'CTTGGTTATTC',    #86
           'TTGGCTGGAC',    #87
           'CCGAACACTTC',    #88
           'TCCTGAATCTC',    #89
           'CTAACCACGGC',    #90           
           'CGGAAGGATGC',    #91
           'CTAGGAACCGC',    #92
           'CTTGTCCAATC',    #93
           'TCCGACAAGC',    #94
           'CGGACAGATC',    #95
           'TTAAGCGGTC',    #96
           
           ]
    btype='none'
    name='IonXpress'
    adapter = 'GAT'
    
    add_or_update_barcode_set(blist,btype,name,adapter)    
    return

def add_ion_xpress_rna_dnabarcode_set():
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
           ]
    btype='none'
    name='IonXpressRNA'
    adapter = 'GGCCAAGGCG'
    
    add_or_update_barcode_set(blist,btype,name,adapter)  
    return
    
def add_ion_xpress_rna_adapter_dnabarcode_set():
    blist=['GGCCAAGGCG',    #adapter only                
           ]
    btype='none'
    name='RNA_Barcode_None'
    adapter = ''
    
    add_or_update_barcode_set(blist,btype,name,adapter)  
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
    

def add_library_kit_info(name, description, flowCount):
    print "Adding library kit info"
    try:
        kit = models.KitInfo.objects.get(kitType='LibraryKit', name=name)
    except:
        kit = None
    if not kit:
        kwargs = {
            'kitType' : 'LibraryKit',
            'name' : name,
            'description' : description,
            'flowCount' : flowCount
        }
        obj = models.KitInfo(**kwargs)
        obj.save()
    else:
        kit.description = description
        kit.flowCount = flowCount
        kit.save()
        
def add_ThreePrimeadapter (direction, name, sequence, description, isDefault):
    print "Adding 3' adapter"
        
    '''Ion default 3' adapter'''
    qual_cutoff = 9
    qual_window = 30
    adapter_cutoff = 16
    
    # name is unique. There should only be one query result object
    try:
        adapter = models.ThreePrimeadapter.objects.get(name=name)
    except:
        adapter = None
    if not adapter:
        print "Going to add %s adapter" % name
        print "Adding 3' adapter: name=", name, "; sequence=", sequence
        
        kwargs = {
            'direction':direction,
            'name':name,
            'sequence':sequence,
            'description':description,
            'qual_cutoff':qual_cutoff,
            'qual_window':qual_window,
            'adapter_cutoff':adapter_cutoff,
            'isDefault':isDefault
            
        }
        ret = models.ThreePrimeadapter(**kwargs)
        ret.save()
    else:
        ##print "Going to update 3' adapter %s for direction %s \n" % (adapter.name, adapter.direction)
        adapter.direction = direction
        adapter.sequence = sequence
        adapter.description = description
        adapter.qual_cutoff = qual_cutoff
        adapter.qual_window = qual_window
        adapter.adapter_cutoff = adapter_cutoff

        #do not blindly update the isDefault flag since user might have chosen his own
        #adapter as the default
        if isDefault:
            defaultAdapterCount = models.ThreePrimeadapter.objects.filter(direction=direction, isDefault = True).count()            
            if defaultAdapterCount == 0:
                adapter.isDefault = isDefault
        else:
            adapter.isDefault = isDefault
            
        adapter.save()


def add_libraryKey (direction, name, sequence, description, isDefault):
    print "Adding library key"
    
    # There should only be one query result object
    try:
        libKey = models.LibraryKey.objects.get(name=name)
    except:
        libKey = None
    if not libKey:
        print "Going to add %s library key" % name
        print "Adding library key: name=", name, "; sequence=", sequence

        kwargs = {
            'direction':direction,
            'name':name,
            'sequence':sequence,
            'description':description,
            'isDefault':isDefault
            
        }
        ret = models.LibraryKey(**kwargs)
        ret.save()
    else:
        ##print "Going to update library key %s sequence %s for direction %s \n" % (libKey.name, libKey.sequence, libKey.direction)

        libKey.sequence = sequence
        libKey.description = description

        #do not blindly update the isDefault flag since user might have chosen his own
        #adapter as the default
        if isDefault:
            defaultKeyCount = models.LibraryKey.objects.filter(direction=direction, isDefault = True).count()            
            if defaultKeyCount == 0:
                libKey.isDefault = isDefault
        else:
            libKey.isDefault = isDefault        

        libKey.save()

    
def add_barcode_args():
    print "Adding barcode_args"
    try:
        gc = models.GlobalConfig.objects.all()[0]
        try:
            barcode_args = gc.barcode_args
        except AttributeError:
            print "barcode_args field does not exist"
            return 1
        # Only add the argument if it does not exist
        if "doesnotexist" in str(gc.barcode_args.get('filter',"doesnotexist")):
            gc.barcode_args['filter'] = 0.01
            gc.save()
            print "added barcodesplit:filter"
    except:
        print "There is no GlobalConfig object in database"
        print traceback.format_exc()
    

def add_sequencing_kit_info(name, description, flowCount):
    print "Adding sequencing kit info"
    try:
        kit = models.KitInfo.objects.get(kitType='SequencingKit', name=name)
    except:
        kit = None
    if not kit:
        kwargs = {
            'kitType':'SequencingKit',
            'name':name,
            'description':description,
            'flowCount':flowCount
        }
        obj = models.KitInfo(**kwargs)
        obj.save()
    else:
        kit.description = description
        kit.flowCount = flowCount
        kit.save()
    

def add_sequencing_kit_part_info(kitName, barcode):
    print "Adding parts for sequencing kit"
    try:
        kit = models.KitInfo.objects.get(kitType='SequencingKit', name=kitName)
    except:
        kit = None
    if kit:
        ##print "sequencing kit found. Id:", kit.id, " kit name:", kit.name
        
        try:
            entry = models.KitPart.objects.get(barcode=barcode)
        except:
            entry = None
          
        if not entry:      
            kwargs = {
                'kit':kit,
                'barcode':barcode
            }
            obj = models.KitPart(**kwargs)
            obj.save()
        ##else:
          ##  print "Barcode ", barcode, " already exists"
    else:
        print "Kit:", kitName, " not found. Barcode:", barcode, " is not added"

    
       
def add_library_kit_info(name, description, flowCount):
    print "Adding library kit info"
    try:
        kit = models.KitInfo.objects.get(kitType='LibraryKit', name=name)
    except:
        kit = None
    if not kit:
        kwargs = {
            'kitType':'LibraryKit',
            'name':name,
            'description':description,
            'flowCount':flowCount
        }
        obj = models.KitInfo(**kwargs)
        obj.save()
    else:
        kit.description = description
        kit.flowCount = flowCount
        kit.save()
    

def add_library_kit_part_info(kitName, barcode):
    print "Adding parts for library kit"
    try:
        kit = models.KitInfo.objects.get(kitType='LibraryKit', name=kitName)
    except:
        kit = None
    if kit:
        ##print "library kit found. Id:", kit.id, " kit name:", kit.name
                
        try:
            entry = models.KitPart.objects.get(barcode=barcode)
        except:
            entry = None
          
        if not entry:      
            kwargs = {
                'kit':kit,
                'barcode':barcode
            }
            obj = models.KitPart(**kwargs)
            obj.save()
        ##else:
          ##  print "Barcode:", barcode, " already exists"
    else:
        print "Kit:", kitName, " not found. Barcode:", barcode, " is not added"

    
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
    	add_ion_xpress_rna_dnabarcode_set()        
    except:
        print 'Adding dnaBarcodeset: IonXpress_RNA failed'
        print traceback.format_exc()
        sys.exit(1)
        
    try:
    	add_ion_xpress_rna_adapter_dnabarcode_set()        
    except:
        print 'Adding dnaBarcodeset: IonXpress_RNA failed'
        print traceback.format_exc()
        sys.exit(1)            
        
    try:
        add_ThreePrimeadapter('Forward', 'Ion P1B', 'ATCACCGACTGCCCATAGAGAGGCTGAGAC', 'Default forward adapter', True)
        add_ThreePrimeadapter('Reverse', 'Ion Paired End Rev', 'CTGAGTCGGAGACACGCAGGGATGAGATGG', 'Default reverse adapter', True)
        add_ThreePrimeadapter('Forward', 'Ion B', 'CTGAGACTGCCAAGGCACACAGGGGATAGG', 'Ion B', False)
        add_ThreePrimeadapter('Forward', 'Ion Truncated Fusion', 'ATCACCGACTGCCCATCTGAGACTGCCAAG', 'Ion Truncated Fusion', False)
        add_ThreePrimeadapter('Forward', 'Ion Paired End Fwd', 'GCTGAGGATCACCGACTGCCCATAGAGAGG', 'Ion Paired End Fwd', False)        
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
        add_barcode_args()
    except:
        print "Modifying barcode-args list failed"
        print traceback.format_exc()
        sys.exit(1)
        
    try:
        add_sequencing_kit_info('IonSeqKit','(100bp) Ion Sequencing Kit','260')
        add_sequencing_kit_info('IonSeq200Kit','(200bp) Ion Sequencing 200 Kit','520')
        add_sequencing_kit_info('IonPGM200Kit','(200bp) Ion PGM(tm) 200 Sequencing Kit','520')
        add_sequencing_kit_info('IonPGM200Kit-v2','(200bp) Ion PGM(tm) 200 Sequencing Kit v2','520')

    except:
        print "Adding sequencing_kit info failed"
        print traceback.format_exc()
        sys.exit(1)

    try:
        add_sequencing_kit_part_info('IonSeqKit','4468997')
        add_sequencing_kit_part_info('IonSeqKit','4468996')
        add_sequencing_kit_part_info('IonSeqKit','4468995')
        add_sequencing_kit_part_info('IonSeqKit','4468994')
        add_sequencing_kit_part_info('IonSeq200Kit','4471258')
        add_sequencing_kit_part_info('IonSeq200Kit','4471257')
        add_sequencing_kit_part_info('IonSeq200Kit','4471259')
        add_sequencing_kit_part_info('IonSeq200Kit','4471260')
        add_sequencing_kit_part_info('IonPGM200Kit','4474004')
        add_sequencing_kit_part_info('IonPGM200Kit','4474005')
        add_sequencing_kit_part_info('IonPGM200Kit','4474006')
        add_sequencing_kit_part_info('IonPGM200Kit','4474007')
        add_sequencing_kit_part_info('IonPGM200Kit-v2','4478321')
        add_sequencing_kit_part_info('IonPGM200Kit-v2','4478322')
        add_sequencing_kit_part_info('IonPGM200Kit-v2','4478323')
        add_sequencing_kit_part_info('IonPGM200Kit-v2','4478324') 
               
    except:
        print "Adding sequencing_kit part info failed"
        print traceback.format_exc()
        sys.exit(1)    
   
    try:
        add_library_kit_info('IonFragmentLibKit','Ion Fragment Library Kit','0')
        add_library_kit_info('IonFragmentLibKit2','Ion Fragment Library Kit','0')
        add_library_kit_info('IonPlusFragmentLibKit','Ion Plus Fragment Library Kit','0')
        add_library_kit_info('Ion Xpress Plus Fragment Library Kit','Ion Xpress Plus Fragment Library Kit','0')
        add_library_kit_info('Ion Xpress Plus Paired-End Library Kit','Ion Xpress Plus Paired-End Library Kit','0')
        add_library_kit_info('Ion Plus Paired-End Library Kit','Ion Plus Paired-End Library Kit','0')
        add_library_kit_info('Ion AmpliSeq 2.0 Beta Kit','Ion AmpliSeq 2.0 Beta Kit','0')
        add_library_kit_info('Ion AmpliSeq 2.0 Library Kit','Ion AmpliSeq 2.0 Library Kit','0')
        add_library_kit_info('Ion Total RNA Seq Kit','Ion Total RNA Seq Kit','0')
        add_library_kit_info('Ion Total RNA Seq Kit v2','Ion Total RNA Seq Kit v2','0')
    except:
        print "Adding library_kit info failed"
        print traceback.format_exc()
        sys.exit(1)
        
    try:
        add_library_kit_part_info('IonFragmentLibKit','4462907')
        add_library_kit_part_info('IonFragmentLibKit2','4466464')
        add_library_kit_part_info('IonPlusFragmentLibKit','4471252')
        add_library_kit_part_info('Ion Xpress Plus Fragment Library Kit','4471269')
        add_library_kit_part_info('Ion Xpress Plus Paired-End Library Kit','4477109')  
        add_library_kit_part_info('Ion Plus Paired-End Library Kit','4477110')
        add_library_kit_part_info('Ion AmpliSeq 2.0 Beta Kit','4467226')  
        add_library_kit_part_info('Ion AmpliSeq 2.0 Library Kit','4475345')
        add_library_kit_part_info('Ion Total RNA Seq Kit','4466666')  
        add_library_kit_part_info('Ion Total RNA Seq Kit v2','4475936')              
    except:
        print "Adding library_kit part info failed"
        print traceback.format_exc()
        sys.exit(1)       
       
    try:
        add_libraryKey('Forward', 'Ion TCAG', 'TCAG', 'Default forward library key', True)
        add_libraryKey('Reverse', 'Ion Paired End', 'TCAGC', 'Default reverse library key', True)
        add_libraryKey('Forward', 'Ion TCAGT', 'TCAGT', 'Ion TCAGT', False)
    except ValidationError:
        print "Info: Validation error due to the pre-existence of library key"
    except:
        print "Adding library key failed"
        print traceback.format_exc()
        sys.exit(1)
