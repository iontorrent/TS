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
        fs = models.FileServer(name=_name, filesPrefix=_path, location=default_location() )
        fs.save()


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
    print('Install Script run with command args %s' % ' '.join(sys.argv))
    try:
        cursor = db.connection.cursor()
        cursor.close()

        add_fileserver("Home", "/results/")
        if os.path.isdir("/rawdata"):
            add_fileserver("Raw Data", "/rawdata/")  # T620 support

        add_reportstorage()

        add_or_update_global_config()

        user, is_newly_added = add_user("ionuser", "ionuser")
        if user:
            try:
                group = Group.objects.get(name='ionusers')
                if group and user.groups.count() == 0:
                    user.groups.add(group)
                    user.save()
            except:
                print('Assigning user group to ionuser failed')
                print(traceback.format_exc())

        create_user_profiles()

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

        # try to add PGMs
        models.Rig.objects.get_or_create(name='default', defaults={'location': default_location(), 'comments': "This is a model PGM.  Do not delete."})

        try:
            add_libraryKey('Forward', 'Ion TCAG', 'TCAG', 'Default forward library key', True)
            add_libraryKey('Reverse', 'Ion Paired End', 'TCAGC', 'Default reverse library key', True)
            add_libraryKey('Forward', 'Ion TCAGT', 'TCAGT', 'Ion TCAGT', False)  # add_libraryKey('Forward', 'Finnzyme', 'TCAGTTCA', 'Finnzyme', False)
        except ValidationError:
            print("Info: Validation error due to the pre-existence of library key")

        # Allow re-ordering of analysisArgs entries in ts_dbData.json
        models.AnalysisArgs.objects.filter(isSystem=True).delete()

        # This is necessary to be able to re-order chip entries in ts_dbData.json
        for chip in models.Chip.objects.all():
            chip.delete()

        load_dbData("rundb/fixtures/ts_dbData_chips_kits.json")
        load_dbData("rundb/fixtures/ts_dbData_chips_kits_rnd.json")
        load_dbData("rundb/fixtures/ts_dbData.json")
        load_dbData("rundb/fixtures/ts_dbData_analysisargs.json")
        load_dbData("rundb/fixtures/ts_dbData_analysisargs_rnd.json")
        load_dbData("rundb/fixtures/ionusers_group.json")

        # Setup an ion mesh user for mesh authed api calls to use
        load_dbData("rundb/fixtures/ionmesh_group.json")

        user, is_newly_added = add_user("ionmesh", "ionmesh")
        if is_newly_added:
            user.set_unusable_password()
            group = Group.objects.get(name='ionmesh')
            if group and user.groups.count() == 0:
                user.groups.add(group)
            user.save()

    except Exception:
        print(traceback.format_exc())
