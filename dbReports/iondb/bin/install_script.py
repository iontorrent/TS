#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.bin.djangoinit import *
from django import db
import sys
import os
import traceback

from iondb.rundb import models
from django.contrib.auth.models import User
from django.contrib.auth.models import Group

from django.core.exceptions import ValidationError
from django.core.exceptions import ObjectDoesNotExist

from django.core import management

int_test_file = "/opt/ion/.ion-internal-server"


def add_user(username, password):
    is_newly_added = False
    try:
        user_exists = User.objects.get(username=username)
        # print "User", username, "already existed"
        return user_exists, is_newly_added
    except:
        # print username, "added"
        user = User.objects.create_user(username, "ionuser@iontorrent.com", password)
        # user.is_staff = True # demoted to block use of admin interface
        user.save()
        is_newly_added = True
        return user, is_newly_added


def create_user_profiles():
    for user in User.objects.all():
        (profile, created) = models.UserProfile.objects.get_or_create(user=user)
        if created:
            print "Added missing userprofile for: %s" % user.username


def default_location():
    loc = models.Location.objects.filter(defaultlocation=True) or models.Location.objects.filter(name='Home')
    if loc:
        loc = loc[0]
    else:
        loc = models.Location.objects.all()[0]
    return loc
    

def add_fileserver(_name, _path):
    fs = models.FileServer.objects.all()
    if len(fs) == 0:
        exists = False
        # print "DEBUG:There are no objects"
    else:
        # print "DEBUG:There is one or more objects"
        exists = False
        for f in fs:
            # print "DEBUG:%s:%s" % (f.name,f.filesPrefix)
            if f.filesPrefix in _path:
                exists = True

    # If fileserver name/path does not exist, add it
    if not exists:
        fs = models.FileServer(name=_name,
                               filesPrefix=_path,
                               location=default_location() )
        fs.save()

    else:
        # print "DEBUG:Fileserver %s:%s exists" % (_name,_path)
        pass


def add_reportstorage():
    '''Adds a generic-default report storage location.  Also, for legacy
    installs this function would strip the original full path (http://somehting...)
    and make it a relative path.'''
    rs = models.ReportStorage.objects.all()
    if len(rs) > 0:
        # rs = rs[0]
        # print "ReportStorage exists: %s" % rs.name
        # if 'http' in rs.webServerPath:
        #    rs.webServerPath = '/' + rs.webServerPath.strip().split('/')[-1]
        #    rs.save()
        # print 'Webserver path set to: %s' % rs.webServerPath

        '''If there is no default set, mark newest report storage location as default'''
        defaultSR = rs.exclude(default=False)
        if len(defaultSR) == 0:
            '''find newest Report Storage and mark it default'''
            rs = rs.order_by('pk')
            rs[len(rs) - 1].default = True
            rs[len(rs) - 1].save()

    else:
        hoststring = "/output"
        rs = models.ReportStorage(name="Home",
                                  webServerPath=hoststring,
                                  dirPath="/results/analysis/output",
                                  default=True)
        rs.save()


def add_chips_obsolete():
    from iondb.utils.default_chip_args import default_chip_args
    '''Sets the per chip default analysis args into the `chips` table in the database.  '''

    chips = (('314', '314'),
             ('316', '316'),
             ('318', '318'),
             ('318v2', '318v2'),
             ('316v2', '316v2'),
             ('314v2', '314v2'),
             ('P1.0.19', 'P0'),
             ('900', 'P1'),
             ('900v2', 'P1v2'),
             ('P1.1.16', 'P1'),
             ('P1.1.17', 'P1'),
             ('P1.2.18', 'P1'),
             ('P2.0.16', 'P2'),
             ('P2.1.16', 'P2'),
             ('P2.2.16', 'P2'),
             )

    for (name, description) in chips:

        # get default args for this chip
        args = default_chip_args(name)

        try:
            # (this case when updating TS typically)
            c = models.Chip.objects.get(name=name)
            c.slots = args['slots']
            c.beadfindargs = args['beadfindArgs']
            c.analysisargs = args['analysisArgs']
            c.prebasecallerargs = args['prebasecallerArgs']
            c.basecallerargs = args['basecallerArgs']
            c.thumbnailbeadfindargs = args['thumbnailBeadfindArgs']
            c.thumbnailanalysisargs = args['thumbnailAnalysisArgs']
            c.prethumbnailbasecallerargs = args['prethumbnailBasecallerArgs']
            c.thumbnailbasecallerargs = args['thumbnailBasecallerArgs']
            c.save()
        except ObjectDoesNotExist:
            # (this case is only on TS initialization or when new chips added)
            c = models.Chip(name=name,
                            slots=args['slots'],
                            description=description,
                            analysisargs=args['analysisArgs'],
                            basecallerargs=args['basecallerArgs'],
                            beadfindargs=args['beadfindArgs'],
                            thumbnailanalysisargs=args['thumbnailAnalysisArgs'],
                            thumbnailbasecallerargs=args['thumbnailBasecallerArgs'],
                            thumbnailbeadfindargs=args['thumbnailBeadfindArgs']
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


def add_or_update_global_config():
    gc = models.GlobalConfig.objects.all()
    defaultStore = 'A'
    if not len(gc) > 0:
        add_global_config(gc)
    else:
        try:
            gc = models.GlobalConfig.objects.get(name="Config")
            gc.enable_compendia_OCP = True
            gc.save()
            print "GloblConfig updated"
        except:
            print 'GlobalConfig for Config not found or update failed. Try to fix missing config'
            add_global_config(gc)


def add_global_config(configs):
    defaultStore = 'A'    
    kwargs = {'name': 'Config',
              'selected': False,
              'records_to_display': 20,
              'default_test_fragment_key': 'ATCG',
              'default_library_key': 'TCAG',
              'default_flow_order': 'TACG',
              'plugin_output_folder': 'plugin_out',
              'web_root': '',
              'site_name': 'Torrent Server',
              'default_storage_options': defaultStore,
              'auto_archive_ack': False,
              'base_recalibration_mode': 'standard_recal',
              'enable_compendia_OCP' : True
              }
    configs = models.GlobalConfig(**kwargs)
    configs.save()
    print 'GlobalConfig added'


def runtype_add_obsolete(type, description):
    """Helper function to add runtype if it does not exist """

    rt = models.RunType.objects.filter(runType=type)

    if rt:
        # print "RunType" , type, "exists"
        pass
    else:
        rt = models.RunType(runType=type, description=description)
        rt.save()
        # print type, " RunType added"


def add_or_update_barcode_set(blist, btype, name, adapter):
# Attempt to read dnabarcode set named 'IonXpress' from dbase
    dnabcs = models.dnaBarcode.objects.filter(name=name)
    if len(dnabcs) > 0:
        # print '%s dnaBarcode Set exists in database' % name
        # make sure we have all the sequences we expect
        for index, sequence in enumerate(blist, start=1):
            # Search for this sequence in the list of barcode records
            bc_found = dnabcs.filter(sequence=sequence)
            if len(bc_found) > 1:
                print "ERROR: More than one entry with sequence %s" % sequence
                print "Fix this situation, Mr. Programmer!"
            if len(bc_found) == 1:

                # Make sure floworder field is not 'none'
                if bc_found[0].floworder == 'none':
                    bc_found[0].floworder = ''

                # Make sure id string has zero padded index field
                bc_found[0].id_str = '%s_%03d' % (name, index)

                # update type
                bc_found[0].type = btype

                # Save changes to database
                bc_found[0].save()

            else:   # array length is zero
                # print "Adding entry for %s" % sequence
                kwargs = {
                    'name': name,
                    'id_str': '%s_%03d' % (name, index),
                    'sequence': sequence,
                    'type': btype,
                    'length': len(sequence),
                    'floworder': '',
                    'index': index,
                    'annotation': '',
                    'adapter': adapter,
                    'score_mode': 1,
                    'score_cutoff': 2.0,
                }
                ret = models.dnaBarcode(**kwargs)
                ret.save()
    else:
        # Add the barcodes because they do not exist.
        # NOTE: name for id_str
        for index, sequence in enumerate(blist, start=1):
            kwargs = {
                'name': name,
                'id_str': '%s_%03d' % (name, index),
                'sequence': sequence,
                'type': btype,
                'length': len(sequence),
                'floworder': '',
                'index': index,
                'annotation': '',
                'adapter': adapter,
                'score_mode': 1,
                'score_cutoff': 2.0,
            }
            ret = models.dnaBarcode(**kwargs)
            ret.save()
        print '%s dnaBarcode Set added to database' % name


def add_ion_xpress_dnabarcode_set():
    '''List from jira wiki page:
    https://iontorrent.jira.com/wiki/display/MOLBIO/Ion+Xpress+Barcode+Adapters+1-16+Kit+-+Lot+XXX
    '''
    blist = ['CTAAGGTAAC',  # 1
             'TAAGGAGAAC',  # 2
             'AAGAGGATTC',  # 3
             'TACCAAGATC',  # 4
             'CAGAAGGAAC',  # 5
             'CTGCAAGTTC',  # 6
             'TTCGTGATTC',  # 7
             'TTCCGATAAC',  # 8
             'TGAGCGGAAC',  # 9
             'CTGACCGAAC',  # 10
             'TCCTCGAATC',  # 11
             'TAGGTGGTTC',  # 12
             'TCTAACGGAC',  # 13
             'TTGGAGTGTC',  # 14
             'TCTAGAGGTC',  # 15
             'TCTGGATGAC',  # 16
             'TCTATTCGTC',  # 17
             'AGGCAATTGC',  # 18
             'TTAGTCGGAC',  # 19
             'CAGATCCATC',  # 20
             'TCGCAATTAC',  # 21
             'TTCGAGACGC',  # 22
             'TGCCACGAAC',  # 23
             'AACCTCATTC',  # 24
             'CCTGAGATAC',  # 25
             'TTACAACCTC',  # 26
             'AACCATCCGC',  # 27
             'ATCCGGAATC',  # 28
             'TCGACCACTC',  # 29
             'CGAGGTTATC',  # 30
             'TCCAAGCTGC',  # 31
             'TCTTACACAC',  # 32
             'TTCTCATTGAAC',  # 33
             'TCGCATCGTTC',  # 34
             'TAAGCCATTGTC',  # 35
             'AAGGAATCGTC',  # 36
             'CTTGAGAATGTC',  # 37
             'TGGAGGACGGAC',  # 38
             'TAACAATCGGC',  # 39
             'CTGACATAATC',  # 40
             'TTCCACTTCGC',  # 41
             'AGCACGAATC',  # 42
             'CTTGACACCGC',  # 43
             'TTGGAGGCCAGC',  # 44
             'TGGAGCTTCCTC',  # 45
             'TCAGTCCGAAC',  # 46
             'TAAGGCAACCAC',  # 47
             'TTCTAAGAGAC',  # 48
             'TCCTAACATAAC',  # 49
             'CGGACAATGGC',  # 50
             'TTGAGCCTATTC',  # 51
             'CCGCATGGAAC',  # 52
             'CTGGCAATCCTC',  # 53
             'CCGGAGAATCGC',  # 54
             'TCCACCTCCTC',  # 55
             'CAGCATTAATTC',  # 56
             'TCTGGCAACGGC',  # 57
             'TCCTAGAACAC',  # 58
             'TCCTTGATGTTC',  # 59
             'TCTAGCTCTTC',  # 60
             'TCACTCGGATC',  # 61
             'TTCCTGCTTCAC',  # 62
             'CCTTAGAGTTC',  # 63
             'CTGAGTTCCGAC',  # 64
             'TCCTGGCACATC',  # 65
             'CCGCAATCATC',  # 66
             'TTCCTACCAGTC',  # 67
             'TCAAGAAGTTC',  # 68
             'TTCAATTGGC',  # 69
             'CCTACTGGTC',  # 70
             'TGAGGCTCCGAC',  # 71
             'CGAAGGCCACAC',  # 72
             'TCTGCCTGTC',  # 73
             'CGATCGGTTC',  # 74
             'TCAGGAATAC',  # 75
             'CGGAAGAACCTC',  # 76
             'CGAAGCGATTC',  # 77
             'CAGCCAATTCTC',  # 78
             'CCTGGTTGTC',  # 79
             'TCGAAGGCAGGC',  # 80
             'CCTGCCATTCGC',  # 81
             'TTGGCATCTC',  # 82
             'CTAGGACATTC',  # 83
             'CTTCCATAAC',  # 84
             'CCAGCCTCAAC',  # 85
             'CTTGGTTATTC',  # 86
             'TTGGCTGGAC',  # 87
             'CCGAACACTTC',  # 88
             'TCCTGAATCTC',  # 89
             'CTAACCACGGC',  # 90
             'CGGAAGGATGC',  # 91
             'CTAGGAACCGC',  # 92
             'CTTGTCCAATC',  # 93
             'TCCGACAAGC',  # 94
             'CGGACAGATC',  # 95
             'TTAAGCGGTC',  # 96

             ]
    btype = ''
    name = 'IonXpress'
    adapter = 'GAT'

    add_or_update_barcode_set(blist, btype, name, adapter)
    return


def add_ion_xpress_rna_dnabarcode_set():
    blist = ['CTAAGGTAAC',  # 1
             'TAAGGAGAAC',  # 2
             'AAGAGGATTC',  # 3
             'TACCAAGATC',  # 4
             'CAGAAGGAAC',  # 5
             'CTGCAAGTTC',  # 6
             'TTCGTGATTC',  # 7
             'TTCCGATAAC',  # 8
             'TGAGCGGAAC',  # 9
             'CTGACCGAAC',  # 10
             'TCCTCGAATC',  # 11
             'TAGGTGGTTC',  # 12
             'TCTAACGGAC',  # 13
             'TTGGAGTGTC',  # 14
             'TCTAGAGGTC',  # 15
             'TCTGGATGAC',  # 16
             ]
    btype = 'rna'
    name = 'IonXpressRNA'
    adapter = 'GGCCAAGGCG'

    add_or_update_barcode_set(blist, btype, name, adapter)
    return


def add_ion_xpress_rna_adapter_dnabarcode_set():
    blist = ['GGCCAAGGCG',  # adapter only
             ]
    btype = 'rna'
    name = 'RNA_Barcode_None'
    adapter = ''

    add_or_update_barcode_set(blist, btype, name, adapter)
    return


def add_or_update_barcode_set2(blist, btype, name, adapter, scoreMode,
                               scoreCutoff, alt_barcode_prefix=None,
                               skip_barcode_leading_zero=False,
                               barcode_num_digits=2, start_id_str_value=1, index=0, isActive=True):
    print("add_or_update_barcode_set2... name=%s; alt_barcode_prefix=%s" %
          (name, alt_barcode_prefix))

    digitCount = str(barcode_num_digits)

# Attempt to read dnabarcode set named 'IonXpress' from dbase
    dnabcs = models.dnaBarcode.objects.filter(name=name)
    if len(dnabcs) > 0:
        # print '%s dnaBarcode Set exists in database' % name
        # make sure we have all the sequences we expect
        for id_index, sequence in enumerate(blist, start=start_id_str_value):
            # Search for this sequence in the list of barcode records
            bc_found = dnabcs.filter(sequence=sequence)
            if len(bc_found) > 1:
                print "ERROR: More than one entry with sequence %s" % sequence
                print "Fix this situation, Mr. Programmer!"

            index += 1
            # Make sure id string has zero padded index field if skip_barcode_leading_zero is not true
            if alt_barcode_prefix:
                if skip_barcode_leading_zero:
                    barcode_name = '%s%d' % (alt_barcode_prefix, id_index)
                else:
                    format_string = '%s%0' + digitCount + 'd'
                    barcode_name = format_string % (alt_barcode_prefix, id_index)
            else:
                if skip_barcode_leading_zero:
                    barcode_name = '%s_%d' % (name, id_index)
                else:
                    format_string = '%s_%0' + digitCount + 'd'
                    barcode_name = format_string % (name, id_index)

            if len(bc_found) == 1:

                # print "%s dnaBarcode sequence %s already in the database" % (name, sequence)

                # Make sure floworder field is not 'none'
                if bc_found[0].floworder == 'none':
                    bc_found[0].floworder = ''

                bc_found[0].id_str = barcode_name

                # update type
                bc_found[0].type = btype

                # update index
                bc_found[0].index = index

                # update active
                bc_found[0].active = isActive

                # Save changes to database
                bc_found[0].save()

            else:   # array length is zero
                # print "Adding entry for %s" % sequence
                kwargs = {
                    'name': name,
                    'id_str': barcode_name,
                    'sequence': sequence,
                    'type': btype,
                    'length': len(sequence),
                    'floworder': '',
                    'index': index,
                    'annotation': '',
                    'adapter': adapter,
                    'score_mode': scoreMode,
                    'score_cutoff': scoreCutoff,
                    'active': isActive
                }
                ret = models.dnaBarcode(**kwargs)
                ret.save()
    else:
        # Add the barcodes because they do not exist.
        # NOTE: name for id_str

        format_string = '%s%0' + digitCount + 'd'

        for id_index, sequence in enumerate(blist, start=start_id_str_value):
            index += 1
            if alt_barcode_prefix:
                if skip_barcode_leading_zero:
                    barcode_name = '%s%d' % (alt_barcode_prefix, id_index)
                else:
                    barcode_name = format_string % (alt_barcode_prefix, id_index)
            else:
                if skip_barcode_leading_zero:
                    barcode_name = '%s_%d' % (name, id_index)
                else:
                    barcode_name = format_string % (name, id_index)

            kwargs = {
                'name': name,
                'id_str': barcode_name,
                'sequence': sequence,
                'type': btype,
                'length': len(sequence),
                'floworder': '',
                'index': index,
                'annotation': '',
                'adapter': adapter,
                'score_mode': scoreMode,
                'score_cutoff': scoreCutoff,
                'active': isActive
            }
            ret = models.dnaBarcode(**kwargs)
            ret.save()
        print '%s dnaBarcode Set added to database' % name

    return index


def add_or_update_ion_dnabarcode_set():
    '''List from TS-1517 or, file Barcodes_052611.xlsx
    Added extra T to start of each sequence'''
    blist = [
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
    btype = ''
    name = 'IonSet1'
    adapter = 'CTGCTGTACGGCCAAGGCGT'

    # Check for barcode set named 'ionSet1' and remove it
    # this is slightly different than desired name: 'IonSet1'
    allbarcodes = models.dnaBarcode.objects.filter(name='ionSet1')
    if allbarcodes:
        allbarcodes.all().delete()

    # now that we use the id as a reference key, we can't drop and create every time dbReports is installed
    add_or_update_barcode_set2(blist, btype, name, adapter, 0, 0.90)


def add_or_update_ion_select_dnabarcode_set():
    '''Add barcode kit Ion Select BC Set-1 List for TS-10595
    '''
    blist = [
        "CTAAGGTAAC",
        "TTACAACCTC",
        "CCTGCCATTCGC",
        "TGGAGGACGGAC",
        "TGAGCGGAAC",
        "CCTTAGAGTTC",
        "TCCTCGAATC",
        "AACCTCATTC",
        "CGGACAATGGC",
        "TCCTGAATCTC",
        "TAAGCCATTGTC",
        "CTGAGTTCCGAC",
        "CGGAAGAACCTC",
        "TCTTACACAC",
        "AAGGAATCGTC",
        "TAGGTGGTTC"]
    btype = ''
    name = 'Ion Select BC Set-1'
    adapter = 'GAT'
    barcode_prefix = "IonSelect-"

    # This Select barcode kit has been released with no leading zero padding
    add_or_update_barcode_set2(blist, btype, name, adapter, 1, 2.00, barcode_prefix, True)


def add_or_update_singleSeq_dnabarcode_set():
    '''Add barcode kit SingleSeq List for TS-10680
    '''
    blist = [
        "TAGGTGGTTC",
        "TCTATTCGTC",
        "TCGCAATTAC",
        "TTGAGCCTATTC",
        "CTGGCAATCCTC",
        "CCGGAGAATCGC",
        "TCTAGCTCTTC",
        "TCACTCGGATC",
        "TCCTGGCACATC",
        "TTCCTACCAGTC",
        "TCAAGAAGTTC",
        "CCTACTGGTC",
        "TCAGGAATAC",
        "CGGAAGAACCTC",
        "CGAAGCGATTC",
        "CAGCCAATTCTC",
        "TTGGCATCTC",
        "CTAGGACATTC",
        "CCAGCCTCAAC",
        "CTTGGTTATTC",
        "CTAGGAACCGC",
        "CTTGTCCAATC",
        "TCCGACAAGC",
        "TTAAGCGGTC"]
    btype = ''
    name = 'Ion SingleSeq Barcode set 1'
    adapter = 'GAT'
    barcode_prefix = "SingleSeq_"

    add_or_update_barcode_set2(blist, btype, name, adapter, 1, 2.00, barcode_prefix, False, 3)


def add_or_update_ionCode_dnabarcode_set():
    '''Add barcode kit IonCode List for TS-10614
    '''
    blist = [
        "CTAAGGTAAC",
        "TAAGGAGAAC",
        "AAGAGGATTC",
        "TACCAAGATC",
        "CAGAAGGAAC",
        "CTGCAAGTTC",
        "TTCGTGATTC",
        "TTCCGATAAC",
        "TGAGCGGAAC",
        "CTGACCGAAC",
        "TCCTCGAATC",
        "TAGGTGGTTC",
        "TCTAACGGAC",
        "TTGGAGTGTC",
        "TCTAGAGGTC",
        "TCTGGATGAC",
        "TCTATTCGTC",
        "AGGCAATTGC",
        "TTAGTCGGAC",
        "CAGATCCATC",
        "TCGCAATTAC",
        "TTCGAGACGC",
        "TGCCACGAAC",
        "AACCTCATTC",
        "CCTGAGATAC",
        "TTACAACCTC",
        "AACCATCCGC",
        "ATCCGGAATC",
        "CGAGGTTATC",
        "TCCAAGCTGC",
        "TCTTACACAC",
        "TTCTCATTGAAC",
        "TCGCATCGTTC",
        "TAAGCCATTGTC",
        "AAGGAATCGTC",
        "CTTGAGAATGTC",
        "TGGAGGACGGAC",
        "TAACAATCGGC",
        "CTGACATAATC",
        "TTCCACTTCGC",
        "AGCACGAATC",
        "TTGGAGGCCAGC",
        "TGGAGCTTCCTC",
        "TCAGTCCGAAC",
        "TAAGGCAACCAC",
        "TTCTAAGAGAC",
        "TCCTAACATAAC",
        "CGGACAATGGC",
        "TTGAGCCTATTC",
        "CCGCATGGAAC",
        "CTGGCAATCCTC",
        "TCCACCTCCTC",
        "CAGCATTAATTC",
        "TCCTTGATGTTC",
        "TCTAGCTCTTC",
        "TCACTCGGATC",
        "TTCCTGCTTCAC",
        "CCTTAGAGTTC",
        "CTGAGTTCCGAC",
        "TCCTGGCACATC",
        "CCGCAATCATC",
        "CCAACATTATC",
        "TCAAGAAGTTC",
        "TTCAATTGGC",
        "CCTACTGGTC",
        "TGAGGCTCCGAC",
        "CGAAGGCCACAC",
        "TCTGCCTGTC",
        "CGATCGGTTC",
        "TCAGGAATAC",
        "CGGAAGAACCTC",
        "CGAAGCGATTC",
        "CAGCCAATTCTC",
        "TCGAAGGCAGGC",
        "CCTGCCATTCGC",
        "CTAGGACATTC",
        "CTTCCATAAC",
        "CCAGCCTCAAC",
        "CTTGGTTATTC",
        "TTGGCTGGAC",
        "CCGAACACTTC",
        "TCCTGAATCTC",
        "CTAACCACGGC",
        "CGGAAGGATGC",
        "CTTGTCCAATC",
        "TCCGACAAGC",
        "CGGACAGATC",
        "CCTTGAGGCGGC",
        "TTCTTCCTCTTC",
        "TTCTTCAAGATC",
        "CTTGGAACTGTC",
        "TCGGCCGGAATC",
        "TGGAGATAATTC",
        "TGAATTCCGGAC",
        "CTTGCCACCGTC",
        "CTAACAATTCAC",
        "TTCGCAATGAAC",
        "TTCCGCACGGC",
        "TTGGCCAATTGC",
        "TCTAGTTCAAC",
        "TGAGAAGAATTC",
        "CCTCAACCATC",
        "CCTGCTGGATTC",
        "TGGCAGGAATTC",
        "CGCTTCGATTC",
        "TTCCAGATTGC",
        "TCCGGAGTCTTC",
        "TACATCCATC",
        "GCAACACGAC",
        "TAAGCAATTCTC",
        "CTGATCCATTC",
        "TAGGAACAATC",
        "AACCGGAATTC",
        "CCGGAGGTAATC",
        "TTCAGGACCTTC",
        "TCTAACCAATGC",
        "TCCGAGCTGATC",
        "TTACCATGTTC",
        "CTCATTCCGGTC",
        "TCGAGGCCTGGC",
        "TGGAAGGTTGC",
        "TAGGATTCCGAC",
        "TTGAAGCTCCGC",
        "TTCAACTTCTTC",
        "TTAGGCTCAAC",
        "CCAAGGCGAATC",
        "CTTAGATCGGTC",
        "CCGGTCCGATTC",
        "TTGGAGCGAC",
        "CTTGTTCCGGC",
        "TCCGGCAAGATC",
        "TTCCTATCCGAC",
        "CTAATTGAATC",
        "TTCGACACCAC",
        "TCCGCCATGC",
        "CCAGTTCCTC",
        "TAACAATAATTC",
        "TGCCTGGATC",
        "CTGAAGTCGGAC",
        "AAGGAATGGAAC",
        "TTCCGAACCGAC",
        "TTCACCAGGATC",
        "CTACAACTTC",
        "CTGAGGCATCAC",
        "CCAGCATCATTC",
        "CCGGCTTGAAC",
        "TCAGGCAGATTC",
        "TTCTGCACGATC",
        "TCCGAAGATAAC",
        "CCTCATCGTTC",
        "TGCAACCAAC",
        "CGGAATCCGGTC",
        "TCTTGAGGAAGC",
        "CCGCCACCAATC",
        "AAGGTTATTC",
        "TGGAGATTGGTC",
        "TCTCCATCAATC",
        "TGGAGCCAACAC",
        "TCTAATCGATTC",
        "CCACCAATAC",
        "CTTGGATTCGAC",
        "TTCTGGATTATC",
        "TTCTTCTGGC",
        "TCCTGAGACTC",
        "CTGGAACAAGAC",
        "TCTTGCTTAATC",
        "CTCCAATTGGAC",
        "CTAAGGAAGGTC",
        "TGAAGGCACCTC",
        "ACAATCCGGTTC",
        "TCCTTACAGAAC",
        "TGAATCGAAC",
        "CTTGAAGCCGTC",
        "TTGAGATCAATC",
        "CAGCAATTCGAC",
        "CGAAGCTAATC",
        "CTTAAGGCTGAC",
        "CTGGAGAACCAC",
        "TACTTGGAATC",
        "CTAGGCCTCCTC",
        "CCGAGAACAAC",
        "TTAAGACGTC",
        "CTAAGATCCGC",
        "TGGCTTCATC",
        "CGAACAATTGTC",
        "TTCAAGGTGTTC",
        "CTTAACCACCAC",
        "TCCGGACCGTTC",
        "CCTTGAGCATGC",
        "TCTTAGATATTC",
        "CCTGAATTAC",
        "AAGCCAACCAAC",
        "TCTGGCAACGGC",
        "CTAGGAACCGC",
        "TTAAGCGGTC",
        "TTGGCATCTC",
        "TTGGTTCCAAC",
        "TTAGGCTGATTC",
        "TGGAACCACGTC",
        "AGGCAACGGAAC",
        "TCCTCCTCCAC",
        "TCTCATTCATTC",
        "TTCGGAACGTTC",
        "TTAAGATTATC",
        "CCTTATGGATTC",
        "CTTGAACAGGTC",
        "CCGAACCTATC",
        "CCAGGACGTC",
        "CCTTGTCGTC",
        "CTAGCCAATGAC",
        "CCTGGCTCAATC",
        "CTGAGGCTTGTC",
        "CAAGAATAATTC",
        "CCGAACCAACGC",
        "CCTAGATTAATC",
        "TCAACCACAAC",
        "AGGCCATTGATC",
        "TCGAGAATCGGC",
        "TTCTGCCACTTC",
        "CTAAGCCATCTC",
        "CCTTAGCTCGGC",
        "TCTTAGGACGGC",
        "CTTGCAATGGAC",
        "TCTAATGGTC",
        "CCTCCACGATC",
        "CCTAAGGCAGGC",
        "TCTGGAAGTCGC",
        "TCTTAGGTAATC",
        "TCCAGGCTTATC",
        "TGAATTCTTC",
        "CTTGAGAATTAC",
        "AAGGCCTCGAAC",
        "TTCCTTCAACAC",
        "TGGTTGGATTC",
        "CGGCAACATTC",
        "TGCATTCCGGTC",
        "CGGAAGCATCAC",
        "CTTGGAGTCCTC",
        "CGGACCACGGAC",
        "TTGCCAACCGGC",
        "CTGTTCGAAC",
        "CCGAGTGGTC",
        "CAAGGCTTCCAC",
        "TCTTCATGAATC",
        "TTGACATTAATC",
        "TCAGGCCGAAC",
        "TTCCGCATTGAC",
        "TGGAAGGTCCAC",
        "TTAGCAACATTC",
        "TCTTAGCGATC",
        "AAGCAATCCATC",
        "CCAAGTTGTTC",
        "TGGACTCAATTC",
        "TCTGTAATTC",
        "TTGAAGGATCGC",
        "TTCTACCGGC",
        "TGGAAGAAGGAC",
        "CACCATCCGGTC",
        "CCTGCCGGAATC",
        "CGGCCTTCGGTC",
        "CCTTGGCCTGGC",
        "CTAGTCGAATTC",
        "TAGACGGAATTC",
        "TCCTCCAAGTTC",
        "CCTAAGCTAC",
        "CTAACCGATTC",
        "CCACATCGAAC",
        "TCTTCCTTCCGC",
        "TGGACAATTGAC",
        "TTAGCCTTAAC",
        "TCACCTCGTTC",
        "TCTGACATTCGC",
        "TCCGCTCGGAC",
        "TTCTTGATCATC",
        "CTGACTCCGGC",
        "GAAGATCTTC",
        "TTCCGAAGTCAC",
        "TCCTGGAGTAAC",
        "TGACCAATCCAC",
        "TGGAGGCCGGTC",
        "TAGCATCCGGC",
        "TAAGTCGATC",
        "TCGGCCATAC",
        "TTCCAATCCGTC",
        "CTTGAAGACGAC",
        "TAGAATTCCGTC",
        "CGAGATGAAC",
        "TTCTACAATATC",
        "CCTGGTTGTC",
        "TTCTGAGCGTTC",
        "TCCAAGGACAC",
        "CCAAGCAACGGC",
        "TTCTTAAGTATC",
        "TTCGACTCGTC",
        "TCCGGAACTAC",
        "CGGCCTCAATTC",
        "CAGCCTCCGGAC",
        "TCTGACGGAATC",
        "TCCTTATTGAC",
        "TTCTCCATTATC",
        "AACTTCCGGATC",
        "TCCACAATTAAC",
        "AAGGCTCGGTTC",
        "CCGAGTCGATTC",
        "TTCTTCACCAAC",
        "CGGAACCTTGC",
        "CTAGGCCAGGAC",
        "CCTGAAGGCTGC",
        "CTTGAAGGTCTC",
        "CTCTCCGATTC",
        "CTTAACATCCTC",
        "TTGAATGGTC",
        "TGAGGAATTCAC",
        "TGGAAGCAAGTC",
        "CTGAGCAATCTC",
        "TCCAGCCATATC",
        "CCTTACTCATC",
        "TCCTTCCACTTC",
        "TGAACCATTGAC",
        "TTAGGATCATTC",
        "TTCCAATTCCAC",
        "TTCTGGTTCTTC",
        "TTCTGTCCGC",
        "TCCGAAGAGATC",
        "TTACACGGAC",
        "TAAGTCCAATTC",
        "TAAGATTCGGC",
        "AGGCCTAATTC",
        "TCTTCAGGATTC",
        "TTCCTTGGTCAC",
        "TTCCTTGCCGTC",
        "CTTAGGCAAGTC",
        "AACAATTCGAAC",
        "CGAGGATCCGTC",
        "TCCTCTTCCTC",
        "TCGAAGCTTCGC",
        "CTAAGGTTCGAC",
        "TTAGGAATCCGC",
        "TGGCCAATCGAC",
        "CTTAAGCATTAC",
        "TTCCTCTAATTC",
        "CTTCCATTCGAC",
        "CCTACAAGATTC",
        "TCTTGAAGATGC",
        "TGAAGCCATCTC",
        "CTAAGCTTGGTC",
        "CTTGGATAAC",
        "CAAGCCACCGTC",
        "CCTAGGTCTTC",
        "TTGACTTCCGGC",
        "CCAGACCGAAC",
        "TCTAGCCATCGC",
        "CGCCAAGAATC",
        "TGAGGCATGGTC",
        "CGAGATTCGGAC",
        "TCAACCTGATC",
        "TCCTGAGGTATC",
        "AAGATGAATC",
        "CCTGGAAGACGC",
        "TTGCACCGTTC",
        "TCTAAGACTTC",
        "CGAACATATTC",
        "TTGGACTTATTC",
        "CGAGGCAATGAC",
        "TCGAGATTAATC",
        "TTCGCCAACAC",
        "TCGGCACGAATC",
        "TCCGTTCGGTC",
        "CCTTAGGATGGC",
        "CACCACCAATTC",
        "TTGAAGCCAGGC",
        "CCTCCAATCGGC",
        "TTACAATGAATC",
        "TCCTGCATGATC",
        "CGCTTCCAAC",
        "TGACAACTTC",
        "CTTGGCCAACTC",
        "TCTTGGCAATGC",
        "AAGGCATCCAAC",
        "CCGACCGGATTC",
        "TTCATTCCGTTC",
        "CTGGCATCGGAC",
        "CTTGACCTGGTC",
        "TTCTACCTCAAC"]

    btype = ''
    name = 'IonCode'
    adapter = 'GGTGAT'
    barcode_prefix = "IonCode_"

    # IonCode barcodes come in 96-well plates.
    # The first two numerical digits are plate number, and the last two digits
    # mark barcodes 1-96 within a plate.
    index = 0
    index = add_or_update_barcode_set2(
        blist[0: 96], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 101, index)
    index = add_or_update_barcode_set2(
        blist[96: 96 * 2], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 201, index)
    index = add_or_update_barcode_set2(
        blist[96 * 2: 96 * 3], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 301, index)
    index = add_or_update_barcode_set2(
        blist[96 * 3: 96 * 4], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 401, index)
    index = add_or_update_barcode_set2(
        blist[96 * 4: 96 * 5], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 501, index)
    index = add_or_update_barcode_set2(
        blist[96 * 5: 96 * 6], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 601, index)


def add_or_update_ionCode1_32_dnabarcode_set():
    '''Add barcode kit IonCode List for TS-10873
    '''
    blist = [
        "CTAAGGTAAC",
        "TAAGGAGAAC",
        "AAGAGGATTC",
        "TACCAAGATC",
        "CAGAAGGAAC",
        "CTGCAAGTTC",
        "TTCGTGATTC",
        "TTCCGATAAC",
        "TGAGCGGAAC",
        "CTGACCGAAC",
        "TCCTCGAATC",
        "TAGGTGGTTC",
        "TCTAACGGAC",
        "TTGGAGTGTC",
        "TCTAGAGGTC",
        "TCTGGATGAC",
        "TCTATTCGTC",
        "AGGCAATTGC",
        "TTAGTCGGAC",
        "CAGATCCATC",
        "TCGCAATTAC",
        "TTCGAGACGC",
        "TGCCACGAAC",
        "AACCTCATTC",
        "CCTGAGATAC",
        "TTACAACCTC",
        "AACCATCCGC",
        "ATCCGGAATC",
        "CGAGGTTATC",
        "TCCAAGCTGC",
        "TCTTACACAC",
        "TTCTCATTGAAC"]

    btype = ''
    name = 'IonCode Barcodes 1-32'
    adapter = 'GGTGAT'
    barcode_prefix = "IonCode_"

    # IonCode barcodes come in 96-well plates.
    # The first two numerical digits are plate number, and the last two digits
    # mark barcodes 1-96 within a plate.
    index = 0
    index = add_or_update_barcode_set2(
        blist, btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 101, index)


def delete_ioncode_tagseq_dnabarcode_set():
    # Delete barcode kit to allow barcode name renaming with embedded spaces
    allbarcodes = models.dnaBarcode.objects.filter(name='IonCode-TagSequencing')
    if allbarcodes:
        allbarcodes.all().delete()


def add_or_update_ioncode_tagseq_dnabarcode_set():
    blist = [
        "CTAAGGTAAC",
        "TAAGGAGAAC",
        "AAGAGGATTC",
        "TACCAAGATC",
        "CAGAAGGAAC",
        "CTGCAAGTTC",
        "TTCGTGATTC",
        "TTCCGATAAC",
        "TGAGCGGAAC",
        "CTGACCGAAC",
        "TCCTCGAATC",
        "TAGGTGGTTC",
        "TCTAACGGAC",
        "TTGGAGTGTC",
        "TCTAGAGGTC",
        "TCTGGATGAC",
        "TCTATTCGTC",
        "AGGCAATTGC",
        "TTAGTCGGAC",
        "CAGATCCATC",
        "TCGCAATTAC",
        "TTCGAGACGC",
        "TGCCACGAAC",
        "AACCTCATTC",
        "CCTGAGATAC",
        "TTACAACCTC",
        "AACCATCCGC",
        "ATCCGGAATC",
        "CGAGGTTATC",
        "TCCAAGCTGC",
        "TCTTACACAC",
        "TTCTCATTGAAC",
        "TCGCATCGTTC",
        "TAAGCCATTGTC",
        "AAGGAATCGTC",
        "CTTGAGAATGTC",
        "TGGAGGACGGAC",
        "TAACAATCGGC",
        "CTGACATAATC",
        "TTCCACTTCGC",
        "AGCACGAATC",
        "TTGGAGGCCAGC",
        "TGGAGCTTCCTC",
        "TCAGTCCGAAC",
        "TAAGGCAACCAC",
        "TTCTAAGAGAC",
        "TCCTAACATAAC",
        "CGGACAATGGC",
        "TTGAGCCTATTC",
        "CCGCATGGAAC",
        "CTGGCAATCCTC",
        "TCCACCTCCTC",
        "CAGCATTAATTC",
        "TCCTTGATGTTC",
        "TCTAGCTCTTC",
        "TCACTCGGATC",
        "TTCCTGCTTCAC",
        "CCTTAGAGTTC",
        "CTGAGTTCCGAC",
        "TCCTGGCACATC",
        "CCGCAATCATC",
        "CCAACATTATC",
        "TCAAGAAGTTC",
        "TTCAATTGGC",
        "CCTACTGGTC",
        "TGAGGCTCCGAC",
        "CGAAGGCCACAC",
        "TCTGCCTGTC",
        "CGATCGGTTC",
        "TCAGGAATAC",
        "CGGAAGAACCTC",
        "CGAAGCGATTC",
        "CAGCCAATTCTC",
        "TCGAAGGCAGGC",
        "CCTGCCATTCGC",
        "CTAGGACATTC",
        "CTTCCATAAC",
        "CCAGCCTCAAC",
        "CTTGGTTATTC",
        "TTGGCTGGAC",
        "CCGAACACTTC",
        "TCCTGAATCTC",
        "CTAACCACGGC",
        "CGGAAGGATGC",
        "CTTGTCCAATC",
        "TCCGACAAGC",
        "CGGACAGATC",
        "CCTTGAGGCGGC",
        "TTCTTCCTCTTC",
        "TTCTTCAAGATC",
        "CTTGGAACTGTC",
        "TCGGCCGGAATC",
        "TGGAGATAATTC",
        "TGAATTCCGGAC",
        "CTTGCCACCGTC",
        "CTAACAATTCAC",
        "TTCGCAATGAAC",
        "TTCCGCACGGC",
        "TTGGCCAATTGC",
        "TCTAGTTCAAC",
        "TGAGAAGAATTC",
        "CCTCAACCATC",
        "CCTGCTGGATTC",
        "TGGCAGGAATTC",
        "CGCTTCGATTC",
        "TTCCAGATTGC",
        "TCCGGAGTCTTC",
        "TACATCCATC",
        "GCAACACGAC",
        "TAAGCAATTCTC",
        "CTGATCCATTC",
        "TAGGAACAATC",
        "AACCGGAATTC",
        "CCGGAGGTAATC",
        "TTCAGGACCTTC",
        "TCTAACCAATGC",
        "TCCGAGCTGATC",
        "TTACCATGTTC",
        "CTCATTCCGGTC",
        "TCGAGGCCTGGC",
        "TGGAAGGTTGC",
        "TAGGATTCCGAC",
        "TTGAAGCTCCGC",
        "TTCAACTTCTTC",
        "TTAGGCTCAAC",
        "CCAAGGCGAATC",
        "CTTAGATCGGTC",
        "CCGGTCCGATTC",
        "TTGGAGCGAC",
        "CTTGTTCCGGC",
        "TCCGGCAAGATC",
        "TTCCTATCCGAC",
        "CTAATTGAATC",
        "TTCGACACCAC",
        "TCCGCCATGC",
        "CCAGTTCCTC",
        "TAACAATAATTC",
        "TGCCTGGATC",
        "CTGAAGTCGGAC",
        "AAGGAATGGAAC",
        "TTCCGAACCGAC",
        "TTCACCAGGATC",
        "CTACAACTTC",
        "CTGAGGCATCAC",
        "CCAGCATCATTC",
        "CCGGCTTGAAC",
        "TCAGGCAGATTC",
        "TTCTGCACGATC",
        "TCCGAAGATAAC",
        "CCTCATCGTTC",
        "TGCAACCAAC",
        "CGGAATCCGGTC",
        "TCTTGAGGAAGC",
        "CCGCCACCAATC",
        "AAGGTTATTC",
        "TGGAGATTGGTC",
        "TCTCCATCAATC",
        "TGGAGCCAACAC",
        "TCTAATCGATTC",
        "CCACCAATAC",
        "CTTGGATTCGAC",
        "TTCTGGATTATC",
        "TTCTTCTGGC",
        "TCCTGAGACTC",
        "CTGGAACAAGAC",
        "TCTTGCTTAATC",
        "CTCCAATTGGAC",
        "CTAAGGAAGGTC",
        "TGAAGGCACCTC",
        "ACAATCCGGTTC",
        "TCCTTACAGAAC",
        "TGAATCGAAC",
        "CTTGAAGCCGTC",
        "TTGAGATCAATC",
        "CAGCAATTCGAC",
        "CGAAGCTAATC",
        "CTTAAGGCTGAC",
        "CTGGAGAACCAC",
        "TACTTGGAATC",
        "CTAGGCCTCCTC",
        "CCGAGAACAAC",
        "TTAAGACGTC",
        "CTAAGATCCGC",
        "TGGCTTCATC",
        "CGAACAATTGTC",
        "TTCAAGGTGTTC",
        "CTTAACCACCAC",
        "TCCGGACCGTTC",
        "CCTTGAGCATGC",
        "TCTTAGATATTC",
        "CCTGAATTAC",
        "AAGCCAACCAAC",
        "TCTGGCAACGGC",
        "CTAGGAACCGC",
        "TTAAGCGGTC",
        "TTGGCATCTC",
        "TTGGTTCCAAC",
        "TTAGGCTGATTC",
        "TGGAACCACGTC",
        "AGGCAACGGAAC",
        "TCCTCCTCCAC",
        "TCTCATTCATTC",
        "TTCGGAACGTTC",
        "TTAAGATTATC",
        "CCTTATGGATTC",
        "CTTGAACAGGTC",
        "CCGAACCTATC",
        "CCAGGACGTC",
        "CCTTGTCGTC",
        "CTAGCCAATGAC",
        "CCTGGCTCAATC",
        "CTGAGGCTTGTC",
        "CAAGAATAATTC",
        "CCGAACCAACGC",
        "CCTAGATTAATC",
        "TCAACCACAAC",
        "AGGCCATTGATC",
        "TCGAGAATCGGC",
        "TTCTGCCACTTC",
        "CTAAGCCATCTC",
        "CCTTAGCTCGGC",
        "TCTTAGGACGGC",
        "CTTGCAATGGAC",
        "TCTAATGGTC",
        "CCTCCACGATC",
        "CCTAAGGCAGGC",
        "TCTGGAAGTCGC",
        "TCTTAGGTAATC",
        "TCCAGGCTTATC",
        "TGAATTCTTC",
        "CTTGAGAATTAC",
        "AAGGCCTCGAAC",
        "TTCCTTCAACAC",
        "TGGTTGGATTC",
        "CGGCAACATTC",
        "TGCATTCCGGTC",
        "CGGAAGCATCAC",
        "CTTGGAGTCCTC",
        "CGGACCACGGAC",
        "TTGCCAACCGGC",
        "CTGTTCGAAC",
        "CCGAGTGGTC",
        "CAAGGCTTCCAC",
        "TCTTCATGAATC",
        "TTGACATTAATC",
        "TCAGGCCGAAC",
        "TTCCGCATTGAC",
        "TGGAAGGTCCAC",
        "TTAGCAACATTC",
        "TCTTAGCGATC",
        "AAGCAATCCATC",
        "CCAAGTTGTTC",
        "TGGACTCAATTC",
        "TCTGTAATTC",
        "TTGAAGGATCGC",
        "TTCTACCGGC",
        "TGGAAGAAGGAC",
        "CACCATCCGGTC",
        "CCTGCCGGAATC",
        "CGGCCTTCGGTC",
        "CCTTGGCCTGGC",
        "CTAGTCGAATTC",
        "TAGACGGAATTC",
        "TCCTCCAAGTTC",
        "CCTAAGCTAC",
        "CTAACCGATTC",
        "CCACATCGAAC",
        "TCTTCCTTCCGC",
        "TGGACAATTGAC",
        "TTAGCCTTAAC",
        "TCACCTCGTTC",
        "TCTGACATTCGC",
        "TCCGCTCGGAC",
        "TTCTTGATCATC",
        "CTGACTCCGGC",
        "GAAGATCTTC",
        "TTCCGAAGTCAC",
        "TCCTGGAGTAAC",
        "TGACCAATCCAC",
        "TGGAGGCCGGTC",
        "TAGCATCCGGC",
        "TAAGTCGATC",
        "TCGGCCATAC",
        "TTCCAATCCGTC",
        "CTTGAAGACGAC",
        "TAGAATTCCGTC",
        "CGAGATGAAC",
        "TTCTACAATATC",
        "CCTGGTTGTC",
        "TTCTGAGCGTTC",
        "TCCAAGGACAC",
        "CCAAGCAACGGC",
        "TTCTTAAGTATC",
        "TTCGACTCGTC",
        "TCCGGAACTAC",
        "CGGCCTCAATTC",
        "CAGCCTCCGGAC",
        "TCTGACGGAATC",
        "TCCTTATTGAC",
        "TTCTCCATTATC",
        "AACTTCCGGATC",
        "TCCACAATTAAC",
        "AAGGCTCGGTTC",
        "CCGAGTCGATTC",
        "TTCTTCACCAAC",
        "CGGAACCTTGC",
        "CTAGGCCAGGAC",
        "CCTGAAGGCTGC",
        "CTTGAAGGTCTC",
        "CTCTCCGATTC",
        "CTTAACATCCTC",
        "TTGAATGGTC",
        "TGAGGAATTCAC",
        "TGGAAGCAAGTC",
        "CTGAGCAATCTC",
        "TCCAGCCATATC",
        "CCTTACTCATC",
        "TCCTTCCACTTC",
        "TGAACCATTGAC",
        "TTAGGATCATTC",
        "TTCCAATTCCAC",
        "TTCTGGTTCTTC",
        "TTCTGTCCGC",
        "TCCGAAGAGATC",
        "TTACACGGAC",
        "TAAGTCCAATTC",
        "TAAGATTCGGC",
        "AGGCCTAATTC",
        "TCTTCAGGATTC",
        "TTCCTTGGTCAC",
        "TTCCTTGCCGTC",
        "CTTAGGCAAGTC",
        "AACAATTCGAAC",
        "CGAGGATCCGTC",
        "TCCTCTTCCTC",
        "TCGAAGCTTCGC",
        "CTAAGGTTCGAC",
        "TTAGGAATCCGC",
        "TGGCCAATCGAC",
        "CTTAAGCATTAC",
        "TTCCTCTAATTC",
        "CTTCCATTCGAC",
        "CCTACAAGATTC",
        "TCTTGAAGATGC",
        "TGAAGCCATCTC",
        "CTAAGCTTGGTC",
        "CTTGGATAAC",
        "CAAGCCACCGTC",
        "CCTAGGTCTTC",
        "TTGACTTCCGGC",
        "CCAGACCGAAC",
        "TCTAGCCATCGC",
        "CGCCAAGAATC",
        "TGAGGCATGGTC",
        "CGAGATTCGGAC",
        "TCAACCTGATC",
        "TCCTGAGGTATC",
        "AAGATGAATC",
        "CCTGGAAGACGC",
        "TTGCACCGTTC",
        "TCTAAGACTTC",
        "CGAACATATTC",
        "TTGGACTTATTC",
        "CGAGGCAATGAC",
        "TCGAGATTAATC",
        "TTCGCCAACAC",
        "TCGGCACGAATC",
        "TCCGTTCGGTC",
        "CCTTAGGATGGC",
        "CACCACCAATTC",
        "TTGAAGCCAGGC",
        "CCTCCAATCGGC",
        "TTACAATGAATC",
        "TCCTGCATGATC",
        "CGCTTCCAAC",
        "TGACAACTTC",
        "CTTGGCCAACTC",
        "TCTTGGCAATGC",
        "AAGGCATCCAAC",
        "CCGACCGGATTC",
        "TTCATTCCGTTC",
        "CTGGCATCGGAC",
        "CTTGACCTGGTC",
        "TTCTACCTCAAC"
        ]

    btype = ''
    name = 'IonCode - TagSequencing'
    adapter = 'TCTGTACGGTGACAAGGCG'
    barcode_prefix = "IonCodeTag_"

    # IonCode barcodes come in 96-well plates.
    # The first two numerical digits are plate number, and the last two digits
    # mark barcodes 1-96 within a plate.
    index = 0
    index = add_or_update_barcode_set2(
        blist[0: 96], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 101, index, False)
    index = add_or_update_barcode_set2(
        blist[96: 96 * 2], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 201, index, False)
    index = add_or_update_barcode_set2(
        blist[96 * 2: 96 * 3], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 301, index, False)
    index = add_or_update_barcode_set2(
        blist[96 * 3: 96 * 4], btype, name, adapter, 1, 2.00, barcode_prefix, False, 4, 401, index, False)

    return


def add_or_update_ion_xpress_museek_dnabarcode_set():
    '''Add barcode kit Ion Xpress MuSeek Barcode set 1, TS-11970
    '''
    blist = [
        "CTAAGGTAAC",    # IonXMuSeek_001
        "TAAGGAGAAC",    # IonXMuSeek_002
        "AAGAGGATTC",    # IonXMuSeek_003
        "TACCAAGATC",    # IonXMuSeek_004
        "CAGAAGGAAC",    # IonXMuSeek_005
        "CTGCAAGTTC",    # IonXMuSeek_006
        "TTCGTGATTC",    # IonXMuSeek_007
        "TGAGCGGAAC",    # IonXMuSeek_008
        "CTGACCGAAC",    # IonXMuSeek_009
        "TCCTCGAATC",    # IonXMuSeek_010
        "TAGGTGGTTC",    # IonXMuSeek_011
        "TCTAACGGAC",    # IonXMuSeek_012
        "TTGGAGTGTC",    # IonXMuSeek_013
        "TCTAGAGGTC",    # IonXMuSeek_014
        "TCTGGATGAC",    # IonXMuSeek_015
        "TCTATTCGTC",    # IonXMuSeek_016
        "AGGCAATTGC",    # IonXMuSeek_017
        "TTAGTCGGAC",    # IonXMuSeek_018
        "CAGATCCATC",    # IonXMuSeek_019
        "TCGCAATTAC",    # IonXMuSeek_020
        "TTCGAGACGC",    # IonXMuSeek_021
        "TGCCACGAAC",    # IonXMuSeek_022
        "AACCTCATTC",    # IonXMuSeek_023
        "AACCATCCGC",    # IonXMuSeek_024
        "ATCCGGAATC",    # IonXMuSeek_025
        "CGAGGTTATC",    # IonXMuSeek_026
        "TCCAAGCTGC",    # IonXMuSeek_027
        "TCTTACACAC",    # IonXMuSeek_028
        "TTCTCATTGAAC",    # IonXMuSeek_029
        "TCGCATCGTTC",    # IonXMuSeek_030
        "TAAGCCATTGTC",    # IonXMuSeek_031
        "AAGGAATCGTC",    # IonXMuSeek_032
        "TAACAATCGGC",    # IonXMuSeek_033
        "CTGACATAATC",    # IonXMuSeek_034
        "TTGGAGGCCAGC",    # IonXMuSeek_035
        "TGGAGCTTCCTC",    # IonXMuSeek_036
        "TCAGTCCGAAC",    # IonXMuSeek_037
        "TTCTAAGAGAC",    # IonXMuSeek_038
        "TCCTAACATAAC",    # IonXMuSeek_039
        "TTGAGCCTATTC",    # IonXMuSeek_040
        "CCGCATGGAAC",    # IonXMuSeek_041
        "CTGGCAATCCTC",    # IonXMuSeek_042
        "TCCACCTCCTC",    # IonXMuSeek_043
        "CAGCATTAATTC",    # IonXMuSeek_044
        "TCCTTGATGTTC",    # IonXMuSeek_045
        "TCTAGCTCTTC",    # IonXMuSeek_046
        "TCACTCGGATC",    # IonXMuSeek_047
        "TTCCTGCTTCAC",    # IonXMuSeek_048
        "CCTTAGAGTTC",    # IonXMuSeek_049
        "CTGAGTTCCGAC",    # IonXMuSeek_050
        "CCAACATTATC",    # IonXMuSeek_051
        "TCAAGAAGTTC",    # IonXMuSeek_052
        "TTCAATTGGC",    # IonXMuSeek_053
        "CCTACTGGTC",    # IonXMuSeek_054
        "TGAGGCTCCGAC",    # IonXMuSeek_055
        "CGAAGGCCACAC",    # IonXMuSeek_056
        "TCTGCCTGTC",    # IonXMuSeek_057
        "CGATCGGTTC",    # IonXMuSeek_058
        "TCAGGAATAC",    # IonXMuSeek_059
        "CGGAAGAACCTC",    # IonXMuSeek_060
        "CGAAGCGATTC",    # IonXMuSeek_061
        "TCGAAGGCAGGC",    # IonXMuSeek_062
        "CTAGGACATTC",    # IonXMuSeek_063
        "CCAGCCTCAAC",    # IonXMuSeek_064
        "CTTGGTTATTC",    # IonXMuSeek_065
        "TTGGCTGGAC",    # IonXMuSeek_066
        "TCCTGAATCTC",    # IonXMuSeek_067
        "CTAACCACGGC",    # IonXMuSeek_068
        "CGGAAGGATGC",    # IonXMuSeek_069
        "CTTGTCCAATC",    # IonXMuSeek_070
        "TCCGACAAGC",    # IonXMuSeek_071
        "TTCTTCCTCTTC",    # IonXMuSeek_072
        "TTCTTCAAGATC",    # IonXMuSeek_073
        "CTTGGAACTGTC",    # IonXMuSeek_074
        "TGGAGATAATTC",    # IonXMuSeek_075
        "TGAATTCCGGAC",    # IonXMuSeek_076
        "CTAACAATTCAC",    # IonXMuSeek_077
        "TTCGCAATGAAC",    # IonXMuSeek_078
        "TTCCGCACGGC",    # IonXMuSeek_079
        "TTGGCCAATTGC",    # IonXMuSeek_080
        "TCTAGTTCAAC",    # IonXMuSeek_081
        "TGAGAAGAATTC",    # IonXMuSeek_082
        "CGCTTCGATTC",    # IonXMuSeek_083
        "TTCCAGATTGC",    # IonXMuSeek_084
        "TCCGGAGTCTTC",    # IonXMuSeek_085
        "TACATCCATC",    # IonXMuSeek_086
        "GCAACACGAC",    # IonXMuSeek_087
        "TAAGCAATTCTC",    # IonXMuSeek_088
        "CTGATCCATTC",    # IonXMuSeek_089
        "TAGGAACAATC",    # IonXMuSeek_090
        "AACCGGAATTC",    # IonXMuSeek_091
        "CCGGAGGTAATC",    # IonXMuSeek_092
        "TTCAGGACCTTC",    # IonXMuSeek_093
        "TCTAACCAATGC",    # IonXMuSeek_094
        "TCCGAGCTGATC",    # IonXMuSeek_095
        "TTACCATGTTC",    # IonXMuSeek_096
    ]
    btype = ''
    name = 'Ion Xpress MuSeek Barcode set 1'
    adapter = 'TTCGTGCGTCAGTTCA'
    barcode_prefix = "IonXMuSeek_"

    # This Select barcode kit has been released with no leading zero padding
    add_or_update_barcode_set2(blist, btype, name, adapter, 1, 2.00, barcode_prefix, False, 3)


def ensure_dnabarcodes_have_id_str():
    # For the 1.5 release, we are adding the id_str field to each dnabarcode record.
    allbarcodes = models.dnaBarcode.objects.all()
    if not allbarcodes:
        return
    for bc in allbarcodes:
        if bc.id_str == None or bc.id_str == "":
            bc.id_str = "%s_%s" % (bc.name, str(bc.index))
            bc.save()
    return


def add_library_kit_info(name, description, flowCount):
    # print "Adding library kit info"
    try:
        kit = models.KitInfo.objects.get(kitType='LibraryKit', name=name)
    except:
        kit = None
    if not kit:
        kwargs = {
            'kitType': 'LibraryKit',
            'name': name,
            'description': description,
            'flowCount': flowCount
        }
        obj = models.KitInfo(**kwargs)
        obj.save()
    else:
        kit.description = description
        kit.flowCount = flowCount
        kit.save()


def add_ThreePrimeadapter(direction, name, sequence, description, isDefault):
    # print "Adding 3' adapter"

    # name is unique. There should only be one query result object
    try:
        adapter = models.ThreePrimeadapter.objects.get(name=name)
    except:
        adapter = None
    if not adapter:
        # print "Going to add %s adapter" % name
        # print "Adding 3' adapter: name=", name, "; sequence=", sequence

        kwargs = {
            'direction': direction,
            'name': name,
            'sequence': sequence,
            'description': description,
            'isDefault': isDefault

        }
        ret = models.ThreePrimeadapter(**kwargs)
        ret.save()
    else:
        # print "Going to update 3' adapter %s for direction %s \n" % (adapter.name, adapter.direction)
        adapter.direction = direction
        adapter.sequence = sequence
        adapter.description = description

        # do not blindly update the isDefault flag since user might have chosen his own
        # adapter as the default
        if isDefault:
            defaultAdapterCount = models.ThreePrimeadapter.objects.filter(
                direction=direction, isDefault=True).count()
            if defaultAdapterCount == 0:
                adapter.isDefault = isDefault
        else:
            adapter.isDefault = isDefault

        adapter.save()


def add_libraryKey(direction, name, sequence, description, isDefault):
    # print "Adding library key"

    # There should only be one query result object
    try:
        libKey = models.LibraryKey.objects.get(name=name)
    except:
        libKey = None
    if not libKey:
        # print "Going to add %s library key" % name
        # print "Adding library key: name=", name, "; sequence=", sequence

        kwargs = {
            'direction': direction,
            'name': name,
            'sequence': sequence,
            'description': description,
            'isDefault': isDefault

        }
        ret = models.LibraryKey(**kwargs)
        ret.save()
    else:
        # print "Going to update library key %s sequence %s for direction %s \n" %
        # (libKey.name, libKey.sequence, libKey.direction)

        libKey.sequence = sequence
        libKey.description = description

        # do not blindly update the isDefault flag since user might have chosen his own
        # adapter as the default
        if isDefault:
            defaultKeyCount = models.LibraryKey.objects.filter(direction=direction, isDefault=True).count()
            if defaultKeyCount == 0:
                libKey.isDefault = isDefault
        else:
            libKey.isDefault = isDefault

        libKey.save()


def add_sequencing_kit_info(name, description, flowCount):
    # print "Adding sequencing kit info"
    try:
        kit = models.KitInfo.objects.get(kitType='SequencingKit', name=name)
    except:
        kit = None
    if not kit:
        kwargs = {
            'kitType': 'SequencingKit',
            'name': name,
            'description': description,
            'flowCount': flowCount
        }
        obj = models.KitInfo(**kwargs)
        obj.save()
    else:
        kit.description = description
        kit.flowCount = flowCount
        kit.save()


def add_sequencing_kit_part_info(kitName, barcode):
    # print "Adding parts for sequencing kit"
    try:
        kit = models.KitInfo.objects.get(kitType='SequencingKit', name=kitName)
    except:
        kit = None
    if kit:
        # print "sequencing kit found. Id:", kit.id, " kit name:", kit.name

        try:
            entry = models.KitPart.objects.get(barcode=barcode)
        except:
            entry = None

        if not entry:
            kwargs = {
                'kit': kit,
                'barcode': barcode
            }
            obj = models.KitPart(**kwargs)
            obj.save()
        # else:
          # print "Barcode ", barcode, " already exists"
    else:
        print "Kit:", kitName, " not found. Barcode:", barcode, " is not added"


def add_library_kit_info(name, description, flowCount):
    # print "Adding library kit info"
    try:
        kit = models.KitInfo.objects.get(kitType='LibraryKit', name=name)
    except:
        kit = None
    if not kit:
        kwargs = {
            'kitType': 'LibraryKit',
            'name': name,
            'description': description,
            'flowCount': flowCount
        }
        obj = models.KitInfo(**kwargs)
        obj.save()
    else:
        kit.description = description
        kit.flowCount = flowCount
        kit.save()


def add_library_kit_part_info(kitName, barcode):
    # print "Adding parts for library kit"
    try:
        kit = models.KitInfo.objects.get(kitType='LibraryKit', name=kitName)
    except:
        kit = None
    if kit:
        # print "library kit found. Id:", kit.id, " kit name:", kit.name

        try:
            entry = models.KitPart.objects.get(barcode=barcode)
        except:
            entry = None

        if not entry:
            kwargs = {
                'kit': kit,
                'barcode': barcode
            }
            obj = models.KitPart(**kwargs)
            obj.save()
        # else:
          # print "Barcode:", barcode, " already exists"
    else:
        print "Kit:", kitName, " not found. Barcode:", barcode, " is not added"


def load_dbData(file_name):
    """
    load system data to db
    """
    print "Loading data to iondb..."
    management.call_command('loaddata', file_name)


if __name__ == "__main__":

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
        add_fileserver("Home", "/results/")
        if os.path.isdir("/rawdata"):
            add_fileserver("Raw Data", "/rawdata/")  # T620 support
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
        add_or_update_global_config()
    except:
        print 'Adding Global Config Failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        user, is_newly_added = add_user("ionuser", "ionuser")
        if user:
            try:
                group = Group.objects.get(name='ionusers')
                if group and user.groups.count() == 0:
                    user.groups.add(group)
                    user.save()
            except:
                print 'Assigning user group to ionuser failed'
                print traceback.format_exc()
    except:
        print 'Adding ionuser failed'
        print traceback.format_exc()
        sys.exit(1)

    create_user_profiles()

    try:
        # for these users, set_unusable_password()
        # These users exists only to uniformly store records of their contact
        # information for customer support.
        lab, is_newly_added = add_user("lab_contact", "lab_contact")
        if lab is not None:
            lab_profile = lab.userprofile
            lab_profile.title = "Lab Contact"
            lab_profile.save()
        it, is_newly_added = add_user("it_contact", "it_contact")
        if it is not None:
            it_profile = it.userprofile
            it_profile.title = "IT Contact"
            it_profile.save()
    except:
        print 'Adding special users failed'
        print traceback.format_exc()
        sys.exit(1)

# try to add runTypes
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

    # try to add PGMs
    try:
        models.Rig.objects.get_or_create(name = 'default', defaults={'location': default_location(),
                         'comments': "This is a model PGM.  Do not delete." })
    except:
        print 'Adding default PGM failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        add_or_update_ion_dnabarcode_set()
        ensure_dnabarcodes_have_id_str()  # for existing barcode records
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
        add_or_update_ion_select_dnabarcode_set()
        ensure_dnabarcodes_have_id_str()    # for existing barcode records
    except:
        print 'Adding dnaBarcodeset: IonSelect failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        add_or_update_singleSeq_dnabarcode_set()
        ensure_dnabarcodes_have_id_str()    # for existing barcode records
    except:
        print 'Adding dnaBarcodeset: SingleSeq failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        add_or_update_ionCode_dnabarcode_set()
        ensure_dnabarcodes_have_id_str()    # for existing barcode records
    except:
        print 'Adding dnaBarcodeset: IonCode failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        add_or_update_ionCode1_32_dnabarcode_set()
        ensure_dnabarcodes_have_id_str()    # for existing barcode records
    except:
        print 'Adding dnaBarcodeset: IonCode1-32 failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        delete_ioncode_tagseq_dnabarcode_set()
        ensure_dnabarcodes_have_id_str()    # for existing barcode records
    except:
        print 'Deleting dnaBarcodeset: IonCode-TagSequencing failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        add_or_update_ioncode_tagseq_dnabarcode_set()
        ensure_dnabarcodes_have_id_str()    # for existing barcode records
    except:
        print 'Adding dnaBarcodeset: IonCode - TagSequencing failed'
        print traceback.format_exc()
        sys.exit(1)

    try:
        add_or_update_ion_xpress_museek_dnabarcode_set()
        ensure_dnabarcodes_have_id_str()    # for existing barcode records
    except:
        print 'Adding dnaBarcodeset: Ion Xpress MuSeek Barcode set 1 failed'
        print traceback.format_exc()
        sys.exit(1)


#    try:
#        add_ThreePrimeadapter('Forward', 'Ion P1B', 'ATCACCGACTGCCCATAGAGAGGCTGAGAC', 'Default forward adapter', True)
#        add_ThreePrimeadapter('Reverse', 'Ion Paired End Rev', 'CTGAGTCGGAGACACGCAGGGATGAGATGG', 'Default reverse adapter', True)
#        add_ThreePrimeadapter('Forward', 'Ion B', 'CTGAGACTGCCAAGGCACACAGGGGATAGG', 'Ion B', False)
#        add_ThreePrimeadapter('Forward', 'Ion Truncated Fusion', 'ATCACCGACTGCCCATCTGAGACTGCCAAG', 'Ion Truncated Fusion', False)
# add_ThreePrimeadapter('Forward', 'Finnzyme', 'TGAACTGACGCACGAAATCACCGACTGCCC', 'Finnzyme', False)
#        add_ThreePrimeadapter('Forward', 'Ion Paired End Fwd', 'GCTGAGGATCACCGACTGCCCATAGAGAGG', 'Ion Paired End Fwd', False)
#    except:
#        print "Adding 3' adapter failed"
#        print traceback.format_exc()
#        sys.exit(1)

    try:
        add_libraryKey('Forward', 'Ion TCAG', 'TCAG', 'Default forward library key', True)
        add_libraryKey('Reverse', 'Ion Paired End', 'TCAGC', 'Default reverse library key', True)
        add_libraryKey('Forward', 'Ion TCAGT', 'TCAGT', 'Ion TCAGT', False)
        # add_libraryKey('Forward', 'Finnzyme', 'TCAGTTCA', 'Finnzyme', False)
    except ValidationError:
        print "Info: Validation error due to the pre-existence of library key"
    except:
        print "Adding library key failed"
        print traceback.format_exc()
        sys.exit(1)

    # Allow re-ordering of analysisArgs entries in ts_dbData.json
    models.AnalysisArgs.objects.filter(isSystem=True).delete()

    # This is necessary to be able to re-order chip entries in ts_dbData.json
    for chip in models.Chip.objects.all():
        chip.delete()

    load_dbData("rundb/fixtures/ts_dbData_chips_kits.json")
    load_dbData("rundb/fixtures/ts_dbData.json")
    load_dbData("rundb/fixtures/ts_dbData_analysisargs.json")    
    load_dbData("rundb/fixtures/ionusers_group.json")

    # Setup an ion mesh user for mesh authed api calls to use
    load_dbData("rundb/fixtures/ionmesh_group.json")
    try:
        user, is_newly_added = add_user("ionmesh", "ionmesh")
        if is_newly_added:
            user.set_unusable_password()
            group = Group.objects.get(name='ionmesh')
            if group and user.groups.count() == 0:
                user.groups.add(group)
            user.save()
    except:
        print 'Adding ionmesh user failed'
        print traceback.format_exc()
        sys.exit(1)
