# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# encoding: utf-8
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models

#20120627 change log:
# 1) modified libraryKey & 3' adapter run mode values 

#20120724
# south migration will run before install_script.py.  For a non-upgrade brand new TS installation, db data will not be present while migration scripts are run.
# need to add in the minimum set of such data so south migration scripts can do their intended tasks.
#
# Fix: Pre-load v2.2 data in
# 1) rundb_runType
# 2) rundb_kitInfo
# 3) rundb_kitPart
# 4) rundb_libraryKey
# 5) rundb_threePrimeAdapter
# before such data is referenced in subsequent migration scripts

#20120927 fix
#user could have changed the contents of the keys or adapters. Do not attempt to create library key or 3' adapter when key/adapter with the same name already exists

class Migration(SchemaMigration):

    def forwards(self, orm):
        #20120724 fix
        # 1) rundb_runType
        #runtype_add("GENS","Generic Sequencing")
        #runtype_add("AMPS","AmpliSeq")
        #runtype_add("TARS","TargetSeq")
        #runtype_add("WGNM","Whole Genome")
        appl, isCreated =orm.runtype.objects.get_or_create(runType = "GENS",
            description = "Generic Sequencing")
                    
        print "*** AFTER get_or_create runType for GENS - isCreated=%s " % (str(isCreated)) 
        
        appl, isCreated =orm.runtype.objects.get_or_create(runType = "AMPS",
            description = "AmpliSeq")
                    
        print "*** AFTER get_or_create runType for AMPS - isCreated=%s " % (str(isCreated)) 
        
        appl, isCreated =orm.runtype.objects.get_or_create(runType = "TARS",
            description = "TargetSeq")
                    
        print "*** AFTER get_or_create runType for TARS - isCreated=%s " % (str(isCreated)) 
        
        appl, isCreated =orm.runtype.objects.get_or_create(runType = "WGNM",
            description = "Whole Genome")
                    
        print "*** AFTER get_or_create runType for WGNM - isCreated=%s " % (str(isCreated)) 


        # 2 & 3) rundb_kitInfo & rundb_kitPart
        #add_sequencing_kit_info('IonSeqKit','(100bp) Ion Sequencing Kit','260')
        #add_sequencing_kit_part_info('IonSeqKit','4468997')
        #add_sequencing_kit_part_info('IonSeqKit','4468996')
        #add_sequencing_kit_part_info('IonSeqKit','4468995')
        #add_sequencing_kit_part_info('IonSeqKit','4468994')
        seqKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "SequencingKit", name = "IonSeqKit", 
            description = "(100bp) Ion Sequencing Kit", flowCount = 260)

        print "*** AFTER get_or_create SequencingKit  IonSeqKit - isCreated=%s " % (str(isCreated)) 
        
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4468997")

        print "*** AFTER get_or_create seqKit Part 4468997 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);
            seqKit.save();
                   
        #this part number should be new
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4468996")

        print "*** AFTER get_or_create seqKit Part 4468996 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);        
            seqKit.save();
                    
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4468995")

        print "*** AFTER get_or_create seqKit Part 4468995 isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);
            seqKit.save();
                   
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4468994")

        print "*** AFTER get_or_create seqKit Part 4468994 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);                
            seqKit.save()
        
        #add_sequencing_kit_info('IonSeq200Kit','(200bp) Ion Sequencing 200 Kit','520')
        #add_sequencing_kit_part_info('IonSeq200Kit','4471258')
        #add_sequencing_kit_part_info('IonSeq200Kit','4471257')
        #add_sequencing_kit_part_info('IonSeq200Kit','4471259')
        #add_sequencing_kit_part_info('IonSeq200Kit','4471260')        
        seqKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "SequencingKit", name = "IonSeq200Kit", 
        description = "(200bp) Ion Sequencing 200 Kit", flowCount = 520)

        print "*** AFTER get_or_create SequencingKit IonSeq200Kit - isCreated=%s " % (str(isCreated)) 
        
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4471258")

        print "*** AFTER get_or_create seqKit Part 4471258 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);
            seqKit.save();
                   
        #this part number should be new
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4471257")

        print "*** AFTER get_or_create seqKit Part 4471257 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);        
            seqKit.save();
                    
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4471259")

        print "*** AFTER get_or_create seqKit Part 4471259 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);
            seqKit.save();
                   
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4471260")

        print "*** AFTER get_or_create seqKit Part 4471260 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);                
            seqKit.save();
        
        #add_sequencing_kit_info('IonPGM200Kit','(200bp) Ion PGM(tm) 200 Sequencing Kit','520')
        #add_sequencing_kit_part_info('IonPGM200Kit','4474004')
        #add_sequencing_kit_part_info('IonPGM200Kit','4474005')
        #add_sequencing_kit_part_info('IonPGM200Kit','4474006')
        #add_sequencing_kit_part_info('IonPGM200Kit','4474007')        
        seqKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "SequencingKit", name = "IonPGM200Kit", 
        description = "(200bp) Ion PGM(tm) 200 Sequencing Kit", flowCount = 520)

        print "*** AFTER get_or_create SequencingKit  IonPGM200Kit - isCreated=%s " % (str(isCreated)) 
        
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4474004")

        print "*** AFTER get_or_create seqKit Part 4474004 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);
            seqKit.save();
                   
        #this part number should be new
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4474005")

        print "*** AFTER get_or_create seqKit Part 4474005 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);        
            seqKit.save();
        
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4474006")

        print "*** AFTER get_or_create seqKit Part 4474006 -  isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);
            seqKit.save();
       
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4474007")

        print "*** AFTER get_or_create seqKit Part 4474007 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);                
            seqKit.save();

        #add_sequencing_kit_info('IonPGM200Kit-v2','(200bp) Ion PGM(tm) 200 Sequencing Kit v2','520')
        #add_sequencing_kit_part_info('IonPGM200Kit-v2','4478321')
        #add_sequencing_kit_part_info('IonPGM200Kit-v2','4478322')
        #add_sequencing_kit_part_info('IonPGM200Kit-v2','4478323')
        #add_sequencing_kit_part_info('IonPGM200Kit-v2','4478324')         
        seqKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "SequencingKit", name = "IonPGM200Kit-v2", 
        description = "(200bp) Ion PGM(tm) 200 Sequencing Kit v2", flowCount = 520)

        print "*** AFTER get_or_create SequencingKit  IonPGM200Kit-v2 - isCreated=%s " % (str(isCreated)) 
        
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4478321")

        print "*** AFTER get_or_create seqKit Part 4478321 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);
            seqKit.save();
                   
        #this part number should be new
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4478322")

        print "*** AFTER get_or_create seqKit Part 4478322 isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);        
            seqKit.save();
                    
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4478323")

        print "*** AFTER get_or_create seqKit Part 4478323 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);
            seqKit.save();
                   
        seqKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = seqKit, barcode = "4478324")

        print "*** AFTER get_or_create seqKit Part 4478324 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            seqKit.kitpart_set.add(seqKitPart);                
            seqKit.save();
        
        #add_library_kit_info('IonFragmentLibKit','Ion Fragment Library Kit','0')
        #add_library_kit_part_info('IonFragmentLibKit','4462907')       
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "IonFragmentLibKit", 
        description = "Ion Fragment Library Kit", flowCount = 0)

        print "*** AFTER get_or_create Library Kit IonFragmentLibKit - isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4462907")

        print "*** AFTER get_or_create libKit Part 4462907 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
        
        #add_library_kit_info('IonFragmentLibKit2','Ion Fragment Library Kit','0')
        #add_library_kit_part_info('IonFragmentLibKit2','4466464')        
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "IonFragmentLibKit2", 
        description = "Ion Fragment Library Kit", flowCount = 0)

        print "*** AFTER get_or_create Library Kit IonFragmentLibKit2 - isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4466464")

        print "*** AFTER get_or_create libKit Part 4466464 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
        
        #add_library_kit_info('IonPlusFragmentLibKit','Ion Plus Fragment Library Kit','0')
        #add_library_kit_part_info('IonPlusFragmentLibKit','4471252')        
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "IonPlusFragmentLibKit", 
        description = "Ion Plus Fragment Library Kit", flowCount = 0)

        print "*** AFTER get_or_create Library Kit IonPlusFragmentLibKit - isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4471252")

        print "*** AFTER get_or_create libKit Part 4471252 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
        
        #add_library_kit_info('Ion Xpress Plus Fragment Library Kit','Ion Xpress Plus Fragment Library Kit','0')
        #add_library_kit_part_info('Ion Xpress Plus Fragment Library Kit','4471269')
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "Ion Xpress Plus Fragment Library Kit", 
        description = "Ion Xpress Plus Fragment Library Kit", flowCount = 0)

        print "*** AFTER get_or_create Library Kit Ion Xpress Plus Fragment Library Kit - isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4471269")

        print "*** AFTER get_or_create libKit Part 4471269 -  isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
        
        #add_library_kit_info('Ion Xpress Plus Paired-End Library Kit','Ion Xpress Plus Paired-End Library Kit','0')
        #add_library_kit_part_info('Ion Xpress Plus Paired-End Library Kit','4477109')  
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "Ion Xpress Plus Paired-End Library Kit", 
        description = "Ion Xpress Plus Paired-End Library Kit", flowCount = 0)

        print "*** AFTER get_or_create Library Kit Ion Xpress Plus Paired-End Library Kit - isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4477109")

        print "*** AFTER get_or_create libKit Part 4477109 -  isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
        
        #add_library_kit_info('Ion Plus Paired-End Library Kit','Ion Plus Paired-End Library Kit','0')
        #add_library_kit_part_info('Ion Plus Paired-End Library Kit','4477110')
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "Ion Plus Paired-End Library Kit", 
        description = "Ion Plus Paired-End Library Kit", flowCount = 0)

        print "*** AFTER get_or_create Library Kit Ion Plus Paired-End Library Kit isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4477110")

        print "*** AFTER get_or_create libKit Part 4477110 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
       
        #add_library_kit_info('Ion AmpliSeq 2.0 Beta Kit','Ion AmpliSeq 2.0 Beta Kit','0')
        #add_library_kit_part_info('Ion AmpliSeq 2.0 Beta Kit','4467226')  
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "Ion AmpliSeq 2.0 Beta Kit", 
        description = "Ion AmpliSeq 2.0 Beta Kit", flowCount = 0)

        print "*** AFTER get_or_create Library Kit Ion AmpliSeq 2.0 Beta Kit - isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4467226")

        print "*** AFTER get_or_create libKit Part 4467226 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
          
        #add_library_kit_info('Ion AmpliSeq 2.0 Library Kit','Ion AmpliSeq 2.0 Library Kit','0')
        #add_library_kit_part_info('Ion AmpliSeq 2.0 Library Kit','4475345')
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "Ion AmpliSeq 2.0 Library Kit", 
        description = "Ion AmpliSeq 2.0 Library Kit", flowCount = 0)

        print "*** AFTER get_or_create Library Kit Ion AmpliSeq 2.0 Library Kit - isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4475345")

        print "*** AFTER get_or_create libKit Part 4475345 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
        
        #add_library_kit_info('Ion Total RNA Seq Kit','Ion Total RNA Seq Kit','0')
        #add_library_kit_part_info('Ion Total RNA Seq Kit','4466666')  
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "Ion Total RNA Seq Kit", 
        description = "Ion Total RNA Seq Kit", flowCount = 0)

        print "*** AFTER get_or_create Library Kit Ion Total RNA Seq Kit isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4466666")

        print "*** AFTER get_or_create libKit Part 4466666 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
        
        #add_library_kit_info('Ion Total RNA Seq Kit v2','Ion Total RNA Seq Kit v2','0')
        #add_library_kit_part_info('Ion Total RNA Seq Kit v2','4475936')        
        libKit, isCreated = orm.kitinfo.objects.get_or_create(kitType = "LibraryKit", name = "Ion Total RNA Seq Kit v2", 
        description = "Ion Total RNA Seq Kit v2", flowCount = 0)

        print "*** AFTER get_or_create Library Kit Ion Total RNA Seq Kit v2 isCreated=%s " % (str(isCreated)) 
        
        libKitPart, isCreated = orm.kitpart.objects.get_or_create(kit = libKit, barcode = "4475936")

        print "*** AFTER get_or_create libKit Part 4475936 - isCreated=%s " % (str(isCreated)) 
        if (isCreated):
            libKit.kitpart_set.add(libKitPart);
            libKit.save();
        

        #===========================
        # 4) rundb_libraryKey
        #add_libraryKey('Forward', 'Ion TCAG', 'TCAG', 'Default forward library key', True)
        try:
            libKey = orm.librarykey.objects.get(direction="Forward", name="Ion TCAG")            
            print "*** Library Key Ion TCAG already exists" 
        except orm.librarykey.DoesNotExist:
            libKey, isCreated = orm.librarykey.objects.get_or_create(direction="Forward", name="Ion TCAG", description="Default forward library key", sequence="TCAG", isDefault=True)

            print "*** AFTER get_or_create Library Key Ion TCAG - isCreated=%s " % (str(isCreated)) 

        
        #add_libraryKey('Reverse', 'Ion Paired End', 'TCAGC', 'Default reverse library key', True)
        try:
            libKey = orm.librarykey.objects.get(direction="Reverse", name="Ion Paired End")
            print "*** Library Key Ion Paired End already exists" 
        except orm.librarykey.DoesNotExist:
            libKey, isCreated = orm.librarykey.objects.get_or_create(direction="Reverse", name="Ion Paired End", description="Default reverse library key", sequence="TCAGC", isDefault=True)

            print "*** AFTER get_or_create Library Key Ion Paired End - isCreated=%s " % (str(isCreated)) 
        
        #add_libraryKey('Forward', 'Ion TCAGT', 'TCAGT', 'Ion TCAGT', False) 
        try:       
            libKey = orm.librarykey.objects.get(direction="Forward", name="Ion TCAGT")
            print "*** Library Key Ion TCAG already exists"
        except orm.librarykey.DoesNotExist:               
            libKey, isCreated = orm.librarykey.objects.get_or_create(direction="Forward", name="Ion TCAGT", description="Ion TCAGT", sequence="TCAGT", isDefault=False)

            print "*** AFTER get_or_create Library Key Ion TCAG - isCreated=%s " % (str(isCreated)) 
        

        #===========================        
        # 5) rundb_threePrimeAdapter        
        #add_ThreePrimeadapter('Forward', 'Ion P1B', 'ATCACCGACTGCCCATAGAGAGGCTGAGAC', 'Default forward adapter', True)
        try:
            adapter = orm.threeprimeadapter.objects.get(direction="Forward", name="Ion P1B")
            print "*** 3 prime adapter Ion P1B already exists"
        except orm.threeprimeadapter.DoesNotExist:
            adapter, isCreated = orm.threeprimeadapter.objects.get_or_create(direction="Forward", name="Ion P1B", description="Default forward adapter", sequence="ATCACCGACTGCCCATAGAGAGGCTGAGAC", isDefault=True, qual_cutoff=9, qual_window=30, adapter_cutoff=16)
            print "*** AFTER get_or_create 3 prime adapter Ion P1B - isCreated=%s " % (str(isCreated)) 

        #add_ThreePrimeadapter('Reverse', 'Ion Paired End Rev', 'CTGAGTCGGAGACACGCAGGGATGAGATGG', 'Default reverse adapter', True)
        try:
            adapter = orm.threeprimeadapter.objects.get(direction="Reverse", name="Ion Paired End Rev")
            print "*** 3 prime adapter Ion Paired End Rev already exists" 
        except orm.threeprimeadapter.DoesNotExist:  
            adapter, isCreated = orm.threeprimeadapter.objects.get_or_create(direction="Reverse", name="Ion Paired End Rev", description="Default reverse adapter", sequence="CTGAGTCGGAGACACGCAGGGATGAGATGG", isDefault=True, qual_cutoff=9, qual_window=30, adapter_cutoff=16)

            print "*** AFTER get_or_create 3 prime adapter Ion Paired End Rev - isCreated=%s " % (str(isCreated)) 
         
        #add_ThreePrimeadapter('Forward', 'Ion B', 'CTGAGACTGCCAAGGCACACAGGGGATAGG', 'Ion B', False)
        try:
            adapter = orm.threeprimeadapter.objects.get(direction="Forward", name="Ion B");        
            print "*** 3 prime adapter Ion B already exists"
        except orm.threeprimeadapter.DoesNotExist:
            adapter, isCreated = orm.threeprimeadapter.objects.get_or_create(direction="Forward", name="Ion B", description="Ion B", sequence="CTGAGACTGCCAAGGCACACAGGGGATAGG", isDefault=False, qual_cutoff=9, qual_window=30, adapter_cutoff=16)

            print "*** AFTER get_or_create 3 prime adapter Ion B - isCreated=%s " % (str(isCreated)) 
       
        
        #add_ThreePrimeadapter('Forward', 'Ion Truncated Fusion', 'ATCACCGACTGCCCATCTGAGACTGCCAAG', 'Ion Truncated Fusion', False)
        try:
            adapter = orm.threeprimeadapter.objects.get(direction="Forward", name="Ion Truncated Fusion")       
            print "*** 3 prime adapter Ion Truncated Fusion already exists"
        except orm.threeprimeadapter.DoesNotExist:
            adapter, isCreated = orm.threeprimeadapter.objects.get_or_create(direction="Forward", name="Ion Truncated Fusion", description="Ion Truncated Fusion", sequence="ATCACCGACTGCCCATCTGAGACTGCCAAG", isDefault=False, qual_cutoff=9, qual_window=30, adapter_cutoff=16)

            print "*** AFTER get_or_create 3 prime adapter Ion Truncated Fusion - isCreated=%s " % (str(isCreated)) 

               
        #add_ThreePrimeadapter('Forward', 'Ion Paired End Fwd', 'GCTGAGGATCACCGACTGCCCATAGAGAGG', 'Ion Paired End Fwd', False) 
        try:     
            adapter = orm.threeprimeadapter.objects.get(direction="Forward", name="Ion Paired End Fwd")        
            print "*** 3 prime adapter Ion Paired End Fwd already exists"
        except orm.threeprimeadapter.DoesNotExist:       
            adapter, isCreated = orm.threeprimeadapter.objects.get_or_create(direction="Forward", name="Ion Paired End Fwd", description="Ion Paired End Fwd", sequence="GCTGAGGATCACCGACTGCCCATAGAGAGG", isDefault=False, qual_cutoff=9, qual_window=30, adapter_cutoff=16)

            print "*** AFTER get_or_create 3 prime adapter Ion Paired End Fwd - isCreated=%s " % (str(isCreated))         


        #===========================                
        # update runMode for pairedEnd
        for libKey in orm.librarykey.objects.all():
            if libKey.name == 'Ion Paired End':
                libKey.runMode = 'pe'
                libKey.save()
                
 
        for adapter in orm.threeprimeadapter.objects.all():
            if adapter.name == 'Ion Paired End Fwd' or adapter.direction == 'Reverse':
                adapter.runMode = 'pe'
                adapter.save()


    def backwards(self, orm):
        #20120724-fix
        for appl in orm.runtype.objects.filter(runType__in=['GENS', 'AMPS', 'TARS', 'WGNM']):
            appl.delete()
            print "*** AFTER runType %s is deleted " % (appl.runType)

        for seqKit in orm.kitinfo.objects.filter(kitType = "SequencingKit",  name__in=["IonSeqKit", "IonSeq200Kit", "IonPGM200Kit", "IonPGM200Kit-v2"]):
            for seqKitPart in orm.kitpart.objects.filter(kit = seqKit):
                seqKitPart.delete()
                
            seqKit.delete()
            print "*** AFTER seqKit %s is deleted " % (seqKit.name)

        for libKit in orm.kitinfo.objects.filter(name__in=["IonFragmentLibKit", "IonFragmentLibKit2", "IonPlusFragmentLibKit", 
                                                                  "Ion Xpress Plus Fragment Library Kit", "Ion Xpress Plus Paired-End Library Kit", 
                                                                  "Ion Plus Paired-End Library Kit", "Ion AmpliSeq 2.0 Beta Kit","Ion AmpliSeq 2.0 Library Kit", 
                                                                  "Ion Total RNA Seq Kit", "Ion Total RNA Seq Kit v2"]):
            for libKitPart in orm.kitpart.objects.filter(kit = libKit):
                libKitPart.delete()
                            
            libKit.delete()
            print "*** AFTER libKit %s is deleted " % (libKit.name)

        for libKey in orm.librarykey.objects.filter(name__in=["Ion TCAG", "Ion Paired End", "Ion TCAGT"]):
            libKey.delete()
            print "*** AFTER libKey %s is deleted " % (libKey.name)

        for adapter in orm.threeprimeadapter.objects.filter(name__in=["Ion P1B", "Ion Paired End Rev", "Ion B", "Ion Truncated Fusion", "Ion Paired End Fwd"]):
            adapter.delete()
            print "*** AFTER adapter %s is deleted " % (adapter.name)


        #===========================                       
        # update runMode for pairedEnd
        for libKey in orm.librarykey.objects.all():
            if libKey.name == 'Ion Paired End':
                libKey.runMode = 'single'
                libKey.save()
         
 
        for adapter in orm.threeprimeadapter.objects.all():
            if adapter.name == 'Ion Paired End Fwd' or adapter.direction == 'Reverse':
                adapter.runMode = 'single'
                adapter.save()

 

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
            'date_joined': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2012, 6, 18, 15, 56, 32, 990491)'}),
            'email': ('django.db.models.fields.EmailField', [], {'max_length': '75', 'blank': 'True'}),
            'first_name': ('django.db.models.fields.CharField', [], {'max_length': '30', 'blank': 'True'}),
            'groups': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['auth.Group']", 'symmetrical': 'False', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'is_active': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'is_staff': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'is_superuser': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'last_login': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2012, 6, 18, 15, 56, 32, 990395)'}),
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
            'report': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'analysismetrics_set'", 'to': "orm['rundb.Results']"}),
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
        'rundb.eventlog': {
            'Meta': {'object_name': 'EventLog'},
            'content_type': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'content_type_set_for_eventlog'", 'to': "orm['contenttypes.ContentType']"}),
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'object_pk': ('django.db.models.fields.PositiveIntegerField', [], {}),
            'text': ('django.db.models.fields.TextField', [], {'max_length': '3000'})
        },
        'rundb.experiment': {
            'Meta': {'object_name': 'Experiment'},
            'autoAnalyze': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'barcodeId': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'baselineRun': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'chipBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'blank': 'True'}),
            'chipType': ('django.db.models.fields.CharField', [], {'max_length': '32'}),
            'cycles': ('django.db.models.fields.IntegerField', [], {}),
            'date': ('django.db.models.fields.DateTimeField', [], {'db_index': 'True'}),
            'expCompInfo': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'expDir': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'expName': ('django.db.models.fields.CharField', [], {'max_length': '128', 'db_index': 'True'}),
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
            'analysisthumbnailargs': ('django.db.models.fields.CharField', [], {'max_length': '5000', 'blank': 'True'}),
            'auto_archive_ack': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'barcode_args': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'basecallerargs': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'basecallerthumbnailargs': ('django.db.models.fields.CharField', [], {'max_length': '5000', 'blank': 'True'}),
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
            'report': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'libmetrics_set'", 'to': "orm['rundb.Results']"}),
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
            'runMode' : ('django.db.models.fields.CharField', [], {'max_length': '64', 'default' : "'single'", 'blank': 'True'}),
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
            'planShortID': ('django.db.models.fields.CharField', [], {'db_index': 'True', 'max_length': '5', 'null': 'True', 'blank': 'True'}),
            'planStatus': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'preAnalysis': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'projects': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'plans'", 'symmetrical': 'False', 'to': "orm['rundb.Project']"}),
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
            'Meta': {'unique_together': "(('name', 'version'),)", 'object_name': 'Plugin'},
            'active': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'autorun': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'chipType': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'config': ('django.db.models.fields.TextField', [], {'default': "''", 'null': 'True', 'blank': 'True'}),
            'date': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2012, 6, 18, 15, 56, 32, 697365)'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'libraryName': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '512', 'db_index': 'True'}),
            'path': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'project': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'sample': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'selected': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'status': ('django.db.models.fields.TextField', [], {'default': "''", 'null': 'True', 'blank': 'True'}),
            'url': ('django.db.models.fields.URLField', [], {'default': "''", 'max_length': '256', 'blank': 'True'}),
            'version': ('django.db.models.fields.CharField', [], {'max_length': '256'})
        },
        'rundb.pluginresult': {
            'Meta': {'unique_together': "(('plugin', 'result'),)", 'object_name': 'PluginResult'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'plugin': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Plugin']"}),
            'result': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'pluginresult_set'", 'to': "orm['rundb.Results']"}),
            'state': ('django.db.models.fields.CharField', [], {'max_length': '20'}),
            'store': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'})
        },
        'rundb.project': {
            'Meta': {'object_name': 'Project'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'creator': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['auth.User']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '32'}),
            'public': ('django.db.models.fields.BooleanField', [], {'default': 'True'})
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
            'report': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'qualitymetrics_set'", 'to': "orm['rundb.Results']"})
        },
        'rundb.referencegenome': {
            'Meta': {'ordering': "['short_name']", 'object_name': 'ReferenceGenome'},
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
            'analysisVersion': ('django.db.models.fields.CharField', [], {'max_length': '256'}),
            'experiment': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Experiment']"}),
            'fastqLink': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'framesProcessed': ('django.db.models.fields.IntegerField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'log': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'metaData': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'parentIDs': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'processedCycles': ('django.db.models.fields.IntegerField', [], {}),
            'processedflows': ('django.db.models.fields.IntegerField', [], {}),
            'projects': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'results'", 'symmetrical': 'False', 'to': "orm['rundb.Project']"}),
            'reference': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'reportLink': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'reportstorage': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'storage'", 'null': 'True', 'to': "orm['rundb.ReportStorage']"}),
            'resultsName': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'resultsType': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
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
            'genetics': ('django.db.models.fields.CharField', [], {'max_length': '64', 'default': "'dna'", 'blank': 'True'}),
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
            'report': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'tfmetrics_set'", 'to': "orm['rundb.Results']"}),
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
            'runMode' : ('django.db.models.fields.CharField', [], {'max_length': '64', 'default' : "'single'", 'blank': 'True'}),
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
