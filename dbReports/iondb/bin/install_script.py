#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
from django import db, shortcuts
from django.db import transaction
import sys
import os
import traceback

from iondb.rundb import models
from socket import gethostname
from django.contrib.auth.models import User

from django.core.exceptions import ValidationError
from django.core.exceptions import ObjectDoesNotExist

int_test_file = "/opt/ion/.ion-internal-server"

def add_user(username,password):
    
    try:
        user_exists = User.objects.get(username=username)
        #print "User", username, "already existed"
        return None
    except:
        #print username, "added"
        user = User.objects.create_user(username,"ionuser@iontorrent.com",password)
        user.is_staff = True
        user.save()
        return user

def create_user_profiles():
    for user in User.objects.all():
        (profile, created) = models.UserProfile.objects.get_or_create(user=user)
        if created:
            print "Added missing userprofile for: %s" % user.username

def add_location():
    '''Checks if a location exists and creates a default location
    called `Home` if none exist. '''
    loc = models.Location.objects.all()
    if len(loc) > 0:
        #print "Location exists: %s" % loc[0].name
        pass
    else:
        loc = models.Location(name="Home",defaultlocation=True)
        loc.save()
        #print "Location Saved"
        
def add_fileserver(_name,_path):
    fs = models.FileServer.objects.all()
    if len(fs) == 0:
        exists = False
        #print "DEBUG:There are no objects"
    else:
        #print "DEBUG:There is one or more objects"
        exists = False
        for f in fs:
            #print "DEBUG:%s:%s" % (f.name,f.filesPrefix)
            if f.filesPrefix in _path:
                exists = True
                
    # If fileserver name/path does not exist, add it
    if not exists:
        fs = models.FileServer(name=_name, 
                               filesPrefix=_path, 
                               location=models.Location.objects.all()[0])
        fs.save()
        
    else:
        #print "DEBUG:Fileserver %s:%s exists" % (_name,_path)
        pass

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
        #rs = rs[0]
        ##print "ReportStorage exists: %s" % rs.name
        #if 'http' in rs.webServerPath:
        #    rs.webServerPath = '/' + rs.webServerPath.strip().split('/')[-1]
        #    rs.save()
        #    #print 'Webserver path set to: %s' % rs.webServerPath
            
        '''If there is no default set, mark newest report storage location as default'''
        defaultSR = rs.exclude(default=False)
        if len(defaultSR) == 0:
            '''find newest Report Storage and mark it default'''
            rs = rs.order_by('pk')
            rs[len(rs)-1].default = True
            rs[len(rs)-1].save()
            
    else:
        hoststring = "/output"
        rs = models.ReportStorage(name="Home",
                                  webServerPath=hoststring,
                                  dirPath="/results/analysis/output",
                                  default=True)
        rs.save()

def add_backupconfig():
    '''Creates a backup configuration with default values
    if one doesn't exist.'''
    bk = models.BackupConfig.objects.all()
    if len(bk)>0:
        #print 'BackupConfig exists: %s' % bk[0].name
        pass
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
    from iondb.utils.default_chip_args import default_chip_args
    '''Sets the per chip default analysis args into the `chips` table in the database.  '''
        
    chips = (('314','314'),
             ('316','316'),
             ('318','318'),
             ('318v2','318v2'),
             ('316v2','316v2'),
             ('314v2','314v2'),
             ('P1.0.19','P0'),
             ('900','P1'),
             ('900v2','P1v2'),
             ('P1.1.16','P1'),
             ('P1.1.17','P1'),
             ('P1.2.18','P1'),
             ('P2.0.16','P2'),
             ('P2.1.16','P2'),
             ('P2.2.16','P2'),
            )

    for (name,description) in chips:
        
        # get default args for this chip
        args = default_chip_args(name)
 
        try:
            # (this case when updating TS typically)
            c = models.Chip.objects.get(name=name)
            c.slots = args['slots']
            c.beadfindargs       = args['beadfindArgs']
            c.analysisargs       = args['analysisArgs']
            c.prebasecallerargs  = args['prebasecallerArgs']
            c.basecallerargs     = args['basecallerArgs']
            c.thumbnailbeadfindargs       = args['thumbnailBeadfindArgs']
            c.thumbnailanalysisargs       = args['thumbnailAnalysisArgs']
            c.prethumbnailbasecallerargs  = args['prethumbnailBasecallerArgs']
            c.thumbnailbasecallerargs     = args['thumbnailBasecallerArgs']
            c.save()
        except ObjectDoesNotExist:
            # (this case is only on TS initialization or when new chips added)
            c = models.Chip(name=name,
                            slots=args['slots'],
                            description = description,
                            analysisargs    = args['analysisArgs'],
                            basecallerargs  = args['basecallerArgs'],
                            beadfindargs    = args['beadfindArgs'],
                            thumbnailanalysisargs    = args['thumbnailAnalysisArgs'],
                            thumbnailbasecallerargs  = args['thumbnailBasecallerArgs'],
                            thumbnailbeadfindargs    = args['thumbnailBeadfindArgs']
                            )
            c.save()
            print "Added Chip object named %s." % name
    
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
    defaultStore = 'A'
    if not len(gc)>0:
        kwargs = {'name':'Config', 
                  'selected':False,
                  'plugin_folder':'plugins',
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
                  'base_recalibrate':True,
                  }
        gc = models.GlobalConfig(**kwargs)
        gc.save()
        print 'GlobalConfig added'


def runtype_add_obsolete(type,description):
    """Helper function to add runtype if it does not exist """

    rt = models.RunType.objects.filter(runType=type)

    if rt:
        #print "RunType" , type, "exists"
        pass
    else:
        rt = models.RunType(runType=type,description=description)
        rt.save()
        #print type, " RunType added"

def VariantFrequencies_add(name):
    """Helper function to add VariantFrequencies"""

    vf = models.VariantFrequencies.objects.filter(name=name)

    if vf:
        #print "VariantFrequency" , vf[0],  "exists"
        pass
    else:
        vf = models.VariantFrequencies(name=name)
        vf.save()
        #print name, " VariantFrequency added"


def create_default_pgm(pgmname,comment=''):
    pgms = models.Rig.objects.all()
    for pgm in pgms:
        if pgm.name == pgmname:
            #print "PGM named %s already exists" % pgmname
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
        #print '%s dnaBarcode Set exists in database' % name
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
                #print "Adding entry for %s" % sequence
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

def add_or_update_barcode_set2(blist,btype,name,adapter, scoreMode, scoreCutoff ):
# Attempt to read dnabarcode set named 'IonXpress' from dbase
    dnabcs = models.dnaBarcode.objects.filter(name=name)
    if len(dnabcs) > 0:
        #print '%s dnaBarcode Set exists in database' % name
        # make sure we have all the sequences we expect
        for index,sequence in enumerate(blist,start=1):
            # Search for this sequence in the list of barcode records
            bc_found = dnabcs.filter(sequence=sequence)
            if len(bc_found) > 1:
                print "ERROR: More than one entry with sequence %s" % sequence
                print "TODO: Fix this situation, Mr. Programmer!"
            if len(bc_found) == 1:

                #print "%s dnaBarcode sequence %s already in the database" % (name, sequence)  
                              
                # Make sure floworder field is not 'none'
                if bc_found[0].floworder == 'none':
                    bc_found[0].floworder = ''
                    
                # Make sure id string has zero padded index field
                bc_found[0].id_str = '%s_%02d' % (name,index)
                
                # Save changes to database
                bc_found[0].save()
                
            else:   # array length is zero
                #print "Adding entry for %s" % sequence
                kwargs = {
                    'name':name,
                    'id_str':'%s_%02d' % (name,index),
                    'sequence':sequence,
                    'type':btype,
                    'length':len(sequence),
                    'floworder':'',
                    'index':index,
                    'annotation':'',
                    'adapter':adapter,
                    'score_mode':scoreMode,
                    'score_cutoff':scoreCutoff,
                }
                ret = models.dnaBarcode(**kwargs)
                ret.save()
    else:
        # Add the barcodes because they do not exist.
        # NOTE: name for id_str
        for index,sequence in enumerate(blist,start=1):
            kwargs = {
                'name':name,
                'id_str':'%s_%02d' % (name,index),
                'sequence':sequence,
                'type':btype,
                'length':len(sequence),
                'floworder':'',
                'index':index,
                'annotation':'',
                'adapter':adapter,
                'score_mode':scoreMode,
                'score_cutoff':scoreCutoff,
            }
            ret = models.dnaBarcode(**kwargs)
            ret.save()
        print '%s dnaBarcode Set added to database' % name
         

def add_or_update_ion_dnabarcode_set():
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

    #now that we use the id as a reference key, we can't drop and create every time dbReports is installed        
    add_or_update_barcode_set2(blist,btype,name,adapter, 0, 0.90)  


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
    #print "Adding library kit info"
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
    #print "Adding 3' adapter"
    
    # name is unique. There should only be one query result object
    try:
        adapter = models.ThreePrimeadapter.objects.get(name=name)
    except:
        adapter = None
    if not adapter:
        #print "Going to add %s adapter" % name
        #print "Adding 3' adapter: name=", name, "; sequence=", sequence
        
        kwargs = {
            'direction':direction,
            'name':name,
            'sequence':sequence,
            'description':description,
            'isDefault':isDefault
            
        }
        ret = models.ThreePrimeadapter(**kwargs)
        ret.save()
    else:
        ##print "Going to update 3' adapter %s for direction %s \n" % (adapter.name, adapter.direction)
        adapter.direction = direction
        adapter.sequence = sequence
        adapter.description = description

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
    #print "Adding library key"
    
    # There should only be one query result object
    try:
        libKey = models.LibraryKey.objects.get(name=name)
    except:
        libKey = None
    if not libKey:
        #print "Going to add %s library key" % name
        #print "Adding library key: name=", name, "; sequence=", sequence

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
    #print "Adding barcode_args"
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
    #print "Adding sequencing kit info"
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
    #print "Adding parts for sequencing kit"
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
    #print "Adding library kit info"
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
    #print "Adding parts for library kit"
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
        
def add_prune_rule(_rule):
    try:
        obj = models.dm_prune_field.objects.get(rule=_rule)
    except:
        obj = models.dm_prune_field()
        obj.rule = _rule
        obj.save()
    return obj.pk
        
def add_prune_group(_item):
    '''This redefines(or creates) the prune group object as described by the _item variable'''
    
    def getRuleNums(_list):
        '''Returns list of pk for given rule pattern strings'''
        ruleNums = []
        for pattern in _list:
            try:
                rule = models.dm_prune_field.objects.get(rule=pattern)
                ruleNums.append(int(rule.pk))
            except:
                ruleNums.append(add_prune_rule(pattern))
        # The ruleNums field is a CommaSeparatedIntegerField so we convert array to comma separated integers
        ruleStr = ','.join(['%d' % i for i in ruleNums])
        return ruleStr
    
    try:
        obj = models.dm_prune_group.objects.get(name=_item['groupName'])
    except:
        obj = models.dm_prune_group()
        
    obj.name = _item['groupName']
    obj.rules = ''
    obj.editable = _item['editable']
    obj.ruleNums = getRuleNums(_item['ruleList'])
    obj.save()
        
    
if __name__=="__main__":

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
        add_fileserver("Home","/results/")
        if os.path.isdir ("/rawdata"):
            add_fileserver("Raw Data","/rawdata/")  # T620 support
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

    create_user_profiles()

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


#    #try to add runTypes
#    try:
#        runtype_add("GENS","Generic Sequencing")
#        runtype_add("AMPS","AmpliSeq DNA")
#        runtype_add("TARS","TargetSeq")
#        runtype_add("WGNM","Whole Genome")
#        runtype_add("AMPS_RNA", "AmpliSeq RNA")
#    except:
#        print 'Adding runType failed'
#        print traceback.format_exc()
#        sys.exit(1)

    #try to add variant frequencies
    try:
        VariantFrequencies_add("Germ Line")
        VariantFrequencies_add("Somatic")
    except:
        print 'Adding VariantFrequencies failed'
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
        add_or_update_ion_dnabarcode_set()
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
        ###add_ThreePrimeadapter('Forward', 'Finnzyme', 'TGAACTGACGCACGAAATCACCGACTGCCC', 'Finnzyme', False)
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
        add_libraryKey('Forward', 'Ion TCAG', 'TCAG', 'Default forward library key', True)
        add_libraryKey('Reverse', 'Ion Paired End', 'TCAGC', 'Default reverse library key', True)
        add_libraryKey('Forward', 'Ion TCAGT', 'TCAGT', 'Ion TCAGT', False)
        ###add_libraryKey('Forward', 'Finnzyme', 'TCAGTTCA', 'Finnzyme', False)
    except ValidationError:
        print "Info: Validation error due to the pre-existence of library key"
    except:
        print "Adding library key failed"
        print traceback.format_exc()
        sys.exit(1)
    
    #
    # Create Report Data Management Configuration object.
    # There will only ever be one object in the table
    if models.dm_reports.objects.count() == 0:
        rdmobj = models.dm_reports()
        rdmobj.save()
       
    #
    # Cleanup 3.0 release default prune groups
    #
    rdmobj = models.dm_reports.objects.all().order_by('pk').reverse()[0]
    nameList = ['light prune', 'moderate prune', 'heavy prune']
    for groupname in nameList:
        try:
            prunegroup = models.dm_prune_group.objects.get(name=groupname)
            prunegroup.delete()
            # Reset default prune group if it was set to the deleted group
            if rdmobj.pruneLevel == groupname:
                rdmobj.pruneLevel = "No-op"
                rdmobj.save()
        except ObjectDoesNotExist:
            pass
    #
    # Cleanup 3.0 release default rules that are dangerous
    #
    nameList = ['*']
    for rulename in nameList:
        try:
            rule = models.dm_prune_field.objects.get(rule=rulename)
            rule.delete()
        except:
            pass
        
    # Add Rules Objects.  Not strictly necessary to add rules since rules are added when prune_groups are defined, below.
    checkList = [
        'MaskBead.mask',
        '1.wells',
        '1.tau',
        '1.lmres',
        '1.cafie-residuals',
        'bg_param.h5',
        'separator.h5',
        'BkgModel*.txt',
        'separator*.txt',
        '*.bam',
        '*.bai',
        '*.sff',
        '*.fastq',
        '*.zip',
        ]
    for item in checkList:
        add_prune_rule(item)
    #
    # Define the Default Prune Groups:  These are the default groups and we overwrite existing groups with same name.
    # Note that the No-op group is the default, which is set in numerous places in the code.
    #
    groupList = [
        {'groupName':'No-op',
         'ruleList':[],
         'editable':False},
        {'groupName':'logs-dev',
        'ruleList':['BkgModel*.txt',
                    'separator*.txt'],
        'editable':False},
        {'groupName':'diag-dev',
        'ruleList':['BkgModel*.txt',
                    'separator*.txt',
                    'MaskBead.mask',
                    '1.tau',
                    '1.lmres',
                    '1.cafie-residuals',
                    'bg_param.h5',
                    'separator.h5'],
        'editable':False},
        {'groupName':'deliverables',
        'ruleList':['BkgModel*.txt',
                    'separator*.txt',
                    'MaskBead.mask',
                    '1.tau',
                    '1.lmres',
                    '1.cafie-residuals',
                    'bg_param.h5',
                    'separator.h5',
                    '*.bam',
                    '*.bai',
                    '*.sff',
                    '*.fastq',
                    '*.zip',
                    '!pgm_logs.zip',],
        'editable':False},
    ]
    for item in groupList:
        add_prune_group(item)
        
