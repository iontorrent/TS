# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# -*- coding: utf-8 -*-

import datetime
from south.db import db
from south.v2 import DataMigration
from django.db import models
import json


#20130115 change log:
# Plan data decentralization (PDD): 
# 1) for each used plan with corresponding experiment already deleted, delete the plan as well
# 2) delete experiments with blank unique
# 3) for each unused plan and template, create an Experiment, Sample and EAS record

#20130129 change log:
# 4) fixed plans that are associated with experiment but their planExecuted flag not set to true 

#20130202 change log:
# 5) fixed barcoded sample names that exceeded db attribute length.
# 6) fixed corresponding barcodedSample json blob when barcoded sample name is cropped

#20130226 change log:
# 7) enhanced 4 above to fix plans' planStatus and non-system templates as well

#201300301 change log:
# 8) cropped plan's sample name longer than 64-chars for experiiment creation

class Migration(DataMigration):

    def forwards(self, orm):
        #4), 
        #7)
        preMigration_hasErrors = False

        plans = orm.plannedexperiment.objects.filter(planExecuted = False).exclude(experiment__isnull = True)
        if plans:
            preMigration_hasErrors = True
            print "*** 0124_ Data error found: ", plans.count(), " plan(s) have experiment associated with them but planExecuted flag is not set to True: "
        for plan in plans:
            if plan.isReusable:
                if plan.isSystem:
                    print "WARNING: 0124_ CANNOT AUTO-FIX SYSTEM TEMPLATE!  NEED TO FIX MANUALLY UPON INSPECTION! - Plan name=%s: pk=%d; planStatus=%s; planExecuted=%s; isTemplate=%s" %(plan.planName, plan.id, plan.planStatus, str(plan.planExecuted), str(plan.isReusable))
                else:
                    print "WARNING: 0124_ A REUSABLE PLAN SHOULD NOT BE ASSOCIATED WITH A SEQUENCING RUN!! Marking it as non-reusable - Plan name=%s: pk=%d; planStatus=%s; planExecuted=%s; isTemplate=%s" %(plan.planName, plan.id, plan.planStatus, str(plan.planExecuted), str(plan.isReusable))
                    plan.isReusable = False
                
                    plan.planExecuted =  True
                    plan.save()                    
            else:
                if (plan.planStatus == "planned"):
                    print "0124_ Going to fix used plan with incorrect planStatus and planExecuted values.  Plan name=%s: pk=%d; planStatus=%s; planExecuted=%s;" %(plan.planName, plan.id, plan.planStatus, str(plan.planExecuted))             
                    plan.planStatus = "run"
                else:
                    print "0124_ Going to fix used plan with incorrect planExecuted value.  Plan name=%s: pk=%d; planStatus=%s; planExecuted=%s;" %(plan.planName, plan.id, plan.planStatus, str(plan.planExecuted))              
                
                plan.planExecuted =  True
                plan.save()
    
        if not preMigration_hasErrors:
            print "*** Good! No planExecuted data issue found in the plans. Ready to proceed with 0124_ migration. "

        
        # 1)
        leftOverPlans = orm.plannedexperiment.objects.filter(isReusable = False, planExecuted = True, experiment__id__isnull=True)

        if leftOverPlans:
            print ">>> migration script 0124 going to remove %d used plan(s) with the corresponding experiments previously deleted. " %(leftOverPlans.count())
        else:
            print ">>> migration script 0124 finds no leftOver plans to clean up."
        
        for leftOverPlan in leftOverPlans:
            print "*** migration script 0124 going to delete leftOver plan=%s; id=%d; guid=%s" %(leftOverPlan.planName, leftOverPlan.pk, leftOverPlan.planGUID)              
            leftOverPlan.delete()
        
        # 2)
        leftOverExps = orm.experiment.objects.filter(unique = "").exclude(expName="NONE_ReportOnly_NONE")

        if leftOverExps:
            print ">>> migration script 0124 going to remove %d experiment(s) with no unique info. " %(leftOverExps.count())
        else:
            print ">>> migration script 0124 finds no leftOver experiments to clean up."
        
        for leftOverExp in leftOverExps:
            print "*** migration script 0124 going to delete leftOver exp=%s; id=%d; guid=%s" %(leftOverExp.expName, leftOverExp.pk, leftOverExp.unique)              
            leftOverExp.delete()
        
        # 3)
        plans = orm.plannedexperiment.objects.filter(planExecuted = False).exclude(planStatus = "planned")
        
        if plans:
            print ">>> migration script 0124 going to create Experiment, Sample and EAS records for %d plans. " %(plans.count())
        else:
            print ">>> migration script 0124 finds no plans meeting the criteria to create Experiment, Sample or EAS records."
            
        for plan in plans:

            ##print "*** going to processing plan=%s; id=%d; guid=%s" %(plan.planName, plan.pk, plan.planGUID)  
                            
            plan.planStatus = "planned"
            
            plan.save()
            
            #this should not happen
            ##for exp in plan.experiment_set.all():
            ##    print "!!!plan already has associated experiment.id=%d" %(exp.pk)
            
            ##print "!!!plan.id=%d" %(plan.pk)
            
            currentDate = datetime.datetime.now()
            
            # 8)
            #deal with sample name discrepancy between plan and experiment
            sampleName = "" if not plan.sample else plan.sample
            if sampleName.__len__() > 64:
                print "WARNING: 0124_ Sample name=%s for plan=%s; plan.id=%s exceeds database 64-char limit to be set for experiment and will be cropped. " %(sampleName, plan.planName, str(plan.pk))
                croppedSampleName = sampleName[:64]
            else:
                croppedSampleName = sampleName
                      
            ##getattr(plan, 'autoAnalyze', True),     
            exp_kwargs = {
                          'autoAnalyze' : True if not plan.autoAnalyze else plan.autoAnalyze,
                          'barcodeId' : "" if not plan.barcodeId else plan.barcodeId,
                          'chipType' : "" if not plan.chipType else plan.chipType,
                          'date' : currentDate,
                          'flows' : 0 if not plan.flows else plan.flows,
                          'forward3primeadapter' : "" if not plan.forward3primeadapter else plan.forward3primeadapter,
                          'isReverseRun' : False if not plan.isReverseRun else plan.isReverseRun,
                          'library' : "" if not plan.library else plan.library,
                          'libraryKey' : "" if not plan.libraryKey else plan.libraryKey,
                          'librarykitname' : "" if not plan.librarykitname else plan.librarykitname,
                          'notes' : "" if not plan.notes else plan.notes,
                          'plan' : plan,
                          'reverse3primeadapter' : "" if not plan.reverse3primeadapter else plan.reverse3primeadapter,
                          'reverselibrarykey' : "" if not plan.reverselibrarykey else plan.reverselibrarykey,
                          'runMode' : plan.runMode,
                          'sample' : croppedSampleName,
                          'sequencekitname' : "" if not plan.sequencekitname else plan.sequencekitname,
                          'status' : plan.planStatus,
            
                          'expDir' : '',
                          #temp experiment name value below will be replaced in crawler
                          'expName' : plan.planGUID,
                          'displayName' : plan.planShortID,
                          'pgmName' : '',
                          'log' : '',
                          #db constraint requires a unique value for experiment. temp unique value below will be replaced in crawler
                          'unique' : plan.planGUID,
                          'chipBarcode' : '',
                          'seqKitBarcode' : '',
                          'sequencekitbarcode' : '',
                          'reagentBarcode' : '',
                          'cycles' : 0,
                          'expCompInfo' : '',
                          'baselineRun' : '',
                          'flowsInOrder' : '',
                          'ftpStatus' : '',
            
                          }
    

            ##experiment = Experiment.objects.create(**exp_kwargs)
            experiment = orm.experiment(**exp_kwargs) 
            experiment.save()
            ##print "*** after creating experiment oid=%s for plan=%s;" %(str(experiment.pk), plan.planName)     
            
            barcodedSamples_dict = plan.barcodedSamples
        
            #if this is a plan
            if not plan.isReusable:
                #Note: sample-driven planning is not yet supported. all is plan-driven planning 
                                
                # add single sample 
                if plan.sample:              

                    sample_kwargs = {
                                     'name' : plan.sample,
                                     'displayedName' : "" if not plan.sampleDisplayedName else plan.sampleDisplayedName,
                                     'date' : plan.date,
                                     'status' : plan.planStatus
                                     }

                    sample = orm.sample.objects.get_or_create(name=plan.sample, defaults=sample_kwargs)[0]
                    sample.experiments.add(experiment)
                    sample.save()
          
                    ##print "*** after creating sample oid=%s for plan=%s; plan.id=%s;" %(str(sample.pk), plan.planName, str(plan.pk))                   
                
                # add multiple barcoded samples
                hasAltered = False
                
                if plan.barcodedSamples:                    
                    barcodedSamples_dict = json.loads(plan.barcodedSamples)

                    #workaround - build a new dictionary of barcodedSamples to get around
                    #TypeError: 'unicode' object does not support item deletion at del
                    newBarcodedSamples_dict = {}
                    
                    for displayedName, value in barcodedSamples_dict.items():
                        name = displayedName.replace(' ', '_')
                        
                        if displayedName.__len__() > 127:
                            print "WARNING: 0124_ barcodedSample name=%s for plan=%s; plan.id=%s exceeds database 127-char limit and will be cropped. " %(displayedName, plan.planName, str(plan.pk))
                            
                            croppedDisplayedName = displayedName[:127]
                            croppedName = name[:127]                           
                            hasAltered = True
                            
                            ##barcodedSamples_dict[str(croppedDisplayedName)] = value
                            ##del plan.barcodedSamples[displayedName]

                        else:
                            croppedDisplayedName = displayedName
                            croppedName = name
                        
                        newBarcodedSamples_dict[croppedDisplayedName] = value
                            
                        sample_kwargs = {
                                     'name' : croppedName,
                                     'displayedName' : croppedDisplayedName,
                                     'date' : plan.date,
                                     'status' : plan.planStatus
                                     }

                        sample = orm.sample.objects.get_or_create(name=croppedName, defaults=sample_kwargs)[0]
                        sample.experiments.add(experiment)
                        sample.save()
                        
                        ##print "*** after creating barcoded sample oid=%s for plan=%s; plan.id=%s;" %(str(sample.pk), plan.planName, str(plan.pk))
                    
                    if hasAltered:                   
                        print "*** after fixing barcoded samples plan=%s; plan.id=%s; original barcodedSamples=%s; new barcodedSamples=%s" %(plan.planName, str(plan.pk), barcodedSamples_dict, newBarcodedSamples_dict)                    
                    barcodedSamples_dict = json.dumps(newBarcodedSamples_dict)  

            #20121024PDD-TODO: if we want to support sample-level genome reference, need to persist reference and barcode for each barcoded sample in barcodedSamples            
            #20121024PDD-TODO: need to persist variantFrequency in selectedPlugins
       
            eas_3primeAdapter = "" if not plan.forward3primeadapter else plan.forward3primeadapter
            eas_libraryKey = "" if not plan.libraryKey else plan.libraryKey

            if plan.isReverseRun:
                eas_3primeAdapter = "" if not plan.reverse3primeadapter else plan.reverse3primeadapter
                eas_libraryKey = "" if not plan.reverselibrarykey else plan.reverselibrarykey
                ##print "Hello, I AM A PAIRED-END REVERSE PLAN!!!"
        
        
            eas_kwargs = {
                          ##'barcodedSamples' : "" if not plan.barcodedSamples else plan.barcodedSamples,
                          'barcodedSamples' : "" if not barcodedSamples_dict else barcodedSamples_dict,
                          'barcodeKitName' : "" if not plan.barcodeId else plan.barcodeId,
                          'date' : plan.date,
                          'experiment' : experiment,
                          'hotSpotRegionBedFile' : "" if not plan.regionfile else plan.regionfile,
                          'isEditable' : True,
                          'isOneTimeOverride' : False,
                          'libraryKey' : eas_libraryKey,
                          'libraryKitName' : "" if not plan.librarykitname else plan.librarykitname,
                          'reference' : "" if not plan.library else plan.library,
                          'selectedPlugins' : "" if not plan.selectedPlugins else plan.selectedPlugins,
                          'status' : plan.planStatus,
                          'targetRegionBedFile' : "" if not plan.bedfile else plan.bedfile,
                          'threePrimeAdapter' : eas_3primeAdapter
                          }

            ##eas = ExperimentAnalysisSettings.objects.create(**eas_kwargs)
            eas = orm.experimentAnalysisSettings(**eas_kwargs)
            eas.save()
            ##print "*** after creating eas oid=%s for plan=%s; plan.id=%s;" %(str(eas.pk), plan.planName, str(plan.pk))      
            
            ##print "*** after updating plan=%s; oid=%s; plan.id=%s;" %(plan.planName, str(plan.pk), str(plan.pk))


            
    def backwards(self, orm):
        plans = orm.plannedexperiment.objects.filter(planExecuted = False, planStatus = "planned")
        
        for plan in plans:
            plan.planStatus = ""
                    
            experiments = orm.experiment.objects.filter(plan = plan)
            
            # repopulate Plan fields
            if experiments:
                experiment = experiments[0]
                plan.autoAnalyze = experiment.autoAnalyze
                plan.chipBarcode = experiment.chipBarcode
                plan.chipType = experiment.chipType
                plan.flows = experiment.flows
                plan.flowsInOrder = experiment.flowsInOrder
                plan.notes =  experiment.notes
                plan.planPGM = experiment.pgmName
                plan.sequencekitname = experiment.sequencekitname
                
                eas_set = experiment.eas_set.all().order_by("-date").filter(isOneTimeOverride=False)
                if eas_set:
                    eas = eas_set[0]
                    #print "---0124 rollback: exp.pk=%s; eas.pk=%s; " %(str(experiment.pk), str(eas.pk))
                    #AttributeError: 'ExperimentAnalysisSettings' object has no attribute 'barcodedSamples'
                    plan.barcodeId = eas.barcodeKitName   
                    ##plan.barcodedSamples = eas.barcodedSamples                 
                    plan.bedfile = eas.targetRegionBedFile
                    plan.forward3primeadapter = eas.threePrimeAdapter
                    plan.library = eas.reference
                    plan.libraryKey = eas.libraryKey
                    plan.libraryKitName = eas.libraryKitName
                    plan.regionfile = eas.hotSpotRegionBedFile
                    plan.selectedPlugins = eas.selectedPlugins
                                    
                samples = experiment.samples.all()
                if samples:
                    plan.sample = samples[0].name
                    plan.sampleDisplayedName = samples[0].displayedName
                
            plan.save()                
            
            for experiment in experiments:
                for sample in experiment.samples.all():
                    experiment.samples.remove(sample)
                    experiment.save()

                    #currently, it is an 1-1 between experiment and sample
                    if not (sample.status == "loaded"): 
                        #print "*** about to rollback PDD addition of sample=%s in experiment=%s" %(sample.name, experiment.expName)                         
                        sample.delete()
                    
                eas_set = experiment.eas_set.all()
                for eas in eas_set:
                    eas.delete()
            
                experiment.delete()
        
            #print "*** after rollback PDD data model change for plan=%s; oid=%s" %(plan.planName, str(plan.pk))      



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
        'rundb.applproduct': {
            'Meta': {'object_name': 'ApplProduct'},
            'applType': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.RunType']"}),
            'defaultChipType': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'defaultControlSeqKit': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'controlSeqKit_applProduct_set'", 'null': 'True', 'to': "orm['rundb.KitInfo']"}),
            'defaultFlowCount': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'defaultGenomeRefName': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'null': 'True', 'blank': 'True'}),
            'defaultHotSpotRegionBedFileName': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'null': 'True', 'blank': 'True'}),
            'defaultLibraryKit': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'libKit_applProduct_set'", 'null': 'True', 'to': "orm['rundb.KitInfo']"}),
            'defaultPairedEndAdapterKit': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'peAdapterKit_applProduct_set'", 'null': 'True', 'to': "orm['rundb.KitInfo']"}),
            'defaultPairedEndLibraryKit': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'peLibKit_applProduct_set'", 'null': 'True', 'to': "orm['rundb.KitInfo']"}),
            'defaultPairedEndSequencingKit': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'peSeqKit_applProduct_set'", 'null': 'True', 'to': "orm['rundb.KitInfo']"}),
            'defaultSamplePrepKit': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'samplePrepKit_applProduct_set'", 'null': 'True', 'to': "orm['rundb.KitInfo']"}),
            'defaultSequencingKit': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'seqKit_applProduct_set'", 'null': 'True', 'to': "orm['rundb.KitInfo']"}),
            'defaultTargetRegionBedFileName': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'null': 'True', 'blank': 'True'}),
            'defaultTemplateKit': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'templateKit_applProduct_set'", 'null': 'True', 'to': "orm['rundb.KitInfo']"}),
            'defaultVariantFrequency': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'description': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isActive': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'isDefault': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'isDefaultPairedEnd': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'isHotspotRegionBEDFileSuppported': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'isPairedEndSupported': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'isVisible': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'productCode': ('django.db.models.fields.CharField', [], {'default': "'any'", 'unique': 'True', 'max_length': '64'}),
            'productName': ('django.db.models.fields.CharField', [], {'max_length': '128'})
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
            'keepTN': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'location': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Location']"}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '64'}),
            'number_to_backup': ('django.db.models.fields.IntegerField', [], {'blank': 'True'}),
            'online': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'status': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'}),
            'timeout': ('django.db.models.fields.IntegerField', [], {'blank': 'True'})
        },
        'rundb.chip': {
            'Meta': {'object_name': 'Chip'},
            'analysisargs': ('django.db.models.fields.CharField', [], {'max_length': '5000', 'blank': 'True'}),
            'basecallerargs': ('django.db.models.fields.CharField', [], {'max_length': '5000', 'blank': 'True'}),
            'beadfindargs': ('django.db.models.fields.CharField', [], {'max_length': '5000', 'blank': 'True'}),
            'description': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '128'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'slots': ('django.db.models.fields.IntegerField', [], {}),
            'thumbnailanalysisargs': ('django.db.models.fields.CharField', [], {'max_length': '5000', 'blank': 'True'}),
            'thumbnailbasecallerargs': ('django.db.models.fields.CharField', [], {'max_length': '5000', 'blank': 'True'}),
            'thumbnailbeadfindargs': ('django.db.models.fields.CharField', [], {'max_length': '5000', 'blank': 'True'})
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
        'rundb.dm_prune_field': {
            'Meta': {'object_name': 'dm_prune_field'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'rule': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '64'})
        },
        'rundb.dm_prune_group': {
            'Meta': {'object_name': 'dm_prune_group'},
            'editable': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '128'}),
            'ruleNums': ('django.db.models.fields.CommaSeparatedIntegerField', [], {'default': "''", 'max_length': '128', 'blank': 'True'})
        },
        'rundb.dm_reports': {
            'Meta': {'object_name': 'dm_reports'},
            'autoAge': ('django.db.models.fields.IntegerField', [], {'default': '90'}),
            'autoPrune': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'autoType': ('django.db.models.fields.CharField', [], {'default': "'P'", 'max_length': '32'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'location': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'pruneLevel': ('django.db.models.fields.CharField', [], {'default': "'No-op'", 'max_length': '128'})
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
            'text': ('django.db.models.fields.TextField', [], {'max_length': '3000'}),
            'username': ('django.db.models.fields.CharField', [], {'default': "'ION'", 'max_length': '32', 'blank': 'True'})
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
            'diskusage': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'displayName': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '128'}),
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
            'notes': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'null': 'True', 'blank': 'True'}),
            'pgmName': ('django.db.models.fields.CharField', [], {'max_length': '64'}),
            'plan': ('django.db.models.fields.related.OneToOneField', [], {'blank': 'True', 'related_name': "'experiment'", 'unique': 'True', 'null': 'True', 'to': "orm['rundb.PlannedExperiment']"}),
            'rawdatastyle': ('django.db.models.fields.CharField', [], {'default': "'single'", 'max_length': '24', 'null': 'True', 'blank': 'True'}),
            'reagentBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'blank': 'True'}),
            'resultDate': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'null': 'True', 'db_index': 'True', 'blank': 'True'}),
            'reverse3primeadapter': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'reverse_primer': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'reverselibrarykey': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'runMode': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '64', 'blank': 'True'}),
            'sample': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'seqKitBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'blank': 'True'}),
            'sequencekitbarcode': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'sequencekitname': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'star': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'status': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'storageHost': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'storage_options': ('django.db.models.fields.CharField', [], {'default': "'A'", 'max_length': '200'}),
            'unique': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '512'}),
            'usePreBeadfind': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'user_ack': ('django.db.models.fields.CharField', [], {'default': "'U'", 'max_length': '24'})
        },
        'rundb.experimentanalysissettings': {
            'Meta': {'object_name': 'ExperimentAnalysisSettings'},
            'barcodeKitName': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'barcodedSamples': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'null': 'True', 'blank': 'True'}),
            'date': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'experiment': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'eas_set'", 'null': 'True', 'to': "orm['rundb.Experiment']"}),
            'hotSpotRegionBedFile': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isEditable': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'isOneTimeOverride': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'libraryKey': ('django.db.models.fields.CharField', [], {'max_length': '64', 'blank': 'True'}),
            'libraryKitBarcode': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'libraryKitName': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'reference': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'selectedPlugins': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'null': 'True', 'blank': 'True'}),
            'status': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'targetRegionBedFile': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'threePrimeAdapter': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'})
        },
        'rundb.fileserver': {
            'Meta': {'object_name': 'FileServer'},
            'comments': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'filesPrefix': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'location': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Location']"}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'percentfull': ('django.db.models.fields.FloatField', [], {'default': '0.0', 'null': 'True', 'blank': 'True'})
        },
        'rundb.globalconfig': {
            'Meta': {'object_name': 'GlobalConfig'},
            'auto_archive_ack': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'barcode_args': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'base_recalibrate': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
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
            'Meta': {'unique_together': "(('kitType', 'name'),)", 'object_name': 'KitInfo'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '3024', 'blank': 'True'}),
            'flowCount': ('django.db.models.fields.PositiveIntegerField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'instrumentType': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '64', 'blank': 'True'}),
            'isActive': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'kitType': ('django.db.models.fields.CharField', [], {'max_length': '20'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '512'}),
            'nucleotideType': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '64', 'blank': 'True'}),
            'runMode': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '64', 'blank': 'True'}),
            'uid': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '10'})
        },
        'rundb.kitpart': {
            'Meta': {'unique_together': "(('barcode',),)", 'object_name': 'KitPart'},
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
            'total_mapped_reads': ('django.db.models.fields.BigIntegerField', [], {}),
            'total_mapped_target_bases': ('django.db.models.fields.BigIntegerField', [], {}),
            'total_number_of_sampled_reads': ('django.db.models.fields.IntegerField', [], {})
        },
        'rundb.librarykey': {
            'Meta': {'object_name': 'LibraryKey'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'direction': ('django.db.models.fields.CharField', [], {'default': "'Forward'", 'max_length': '20'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isDefault': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '256'}),
            'runMode': ('django.db.models.fields.CharField', [], {'default': "'single'", 'max_length': '64', 'blank': 'True'}),
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
            'Meta': {'ordering': "['-id']", 'object_name': 'PlannedExperiment'},
            'adapter': ('django.db.models.fields.CharField', [], {'max_length': '256', 'null': 'True', 'blank': 'True'}),
            'autoAnalyze': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'autoName': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'barcodeId': ('django.db.models.fields.CharField', [], {'max_length': '256', 'null': 'True', 'blank': 'True'}),
            'barcodedSamples': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'null': 'True', 'blank': 'True'}),
            'bedfile': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'chipBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'chipType': ('django.db.models.fields.CharField', [], {'max_length': '32', 'null': 'True', 'blank': 'True'}),
            'controlSequencekitname': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'cycles': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'date': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'expName': ('django.db.models.fields.CharField', [], {'max_length': '128', 'blank': 'True'}),
            'flows': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'flowsInOrder': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'forward3primeadapter': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'irworkflow': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'isFavorite': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'isPlanGroup': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'isReusable': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'isReverseRun': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'isSystem': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'isSystemDefault': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'libkit': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'library': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'libraryKey': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'librarykitname': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'metaData': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'}),
            'notes': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'null': 'True', 'blank': 'True'}),
            'pairedEndLibraryAdapterName': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'parentPlan': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'childPlan_set'", 'null': 'True', 'to': "orm['rundb.PlannedExperiment']"}),
            'planDisplayedName': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'planExecuted': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'planExecutedDate': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'planGUID': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'planName': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'planPGM': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'planShortID': ('django.db.models.fields.CharField', [], {'db_index': 'True', 'max_length': '5', 'null': 'True', 'blank': 'True'}),
            'planStatus': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'}),
            'preAnalysis': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'projects': ('django.db.models.fields.related.ManyToManyField', [], {'symmetrical': 'False', 'related_name': "'plans'", 'blank': 'True', 'to': "orm['rundb.Project']"}),
            'qcValues': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['rundb.QCType']", 'null': 'True', 'through': "orm['rundb.PlannedExperimentQC']", 'symmetrical': 'False'}),
            'regionfile': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'reverse3primeadapter': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'reverse_primer': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'reverselibrarykey': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'runMode': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '64', 'blank': 'True'}),
            'runType': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'runname': ('django.db.models.fields.CharField', [], {'max_length': '255', 'null': 'True', 'blank': 'True'}),
            'sample': ('django.db.models.fields.CharField', [], {'max_length': '127', 'null': 'True', 'blank': 'True'}),
            'sampleDisplayedName': ('django.db.models.fields.CharField', [], {'max_length': '127', 'null': 'True', 'blank': 'True'}),
            'samplePrepKitName': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'selectedPlugins': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'null': 'True', 'blank': 'True'}),
            'seqKitBarcode': ('django.db.models.fields.CharField', [], {'max_length': '64', 'null': 'True', 'blank': 'True'}),
            'sequencekitname': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'storageHost': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'storage_options': ('django.db.models.fields.CharField', [], {'default': "'A'", 'max_length': '200'}),
            'templatingKitName': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'}),
            'usePostBeadfind': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'usePreBeadfind': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'username': ('django.db.models.fields.CharField', [], {'max_length': '128', 'null': 'True', 'blank': 'True'}),
            'variantfrequency': ('django.db.models.fields.CharField', [], {'max_length': '512', 'null': 'True', 'blank': 'True'})
        },
        'rundb.plannedexperimentqc': {
            'Meta': {'unique_together': "(('plannedExperiment', 'qcType'),)", 'object_name': 'PlannedExperimentQC'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'plannedExperiment': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.PlannedExperiment']"}),
            'qcType': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.QCType']"}),
            'threshold': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'})
        },
        'rundb.plugin': {
            'Meta': {'unique_together': "(('name', 'version'),)", 'object_name': 'Plugin'},
            'active': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'autorun': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'autorunMutable': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'config': ('django.db.models.fields.TextField', [], {'default': "''", 'null': 'True', 'blank': 'True'}),
            'date': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime.now'}),
            'description': ('django.db.models.fields.TextField', [], {'default': "''", 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'majorBlock': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '512', 'db_index': 'True'}),
            'path': ('django.db.models.fields.CharField', [], {'max_length': '512'}),
            'pluginsettings': ('django.db.models.fields.TextField', [], {'default': "''", 'null': 'True', 'blank': 'True'}),
            'script': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '256', 'blank': 'True'}),
            'selected': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'status': ('django.db.models.fields.TextField', [], {'default': "''", 'null': 'True', 'blank': 'True'}),
            'url': ('django.db.models.fields.URLField', [], {'default': "''", 'max_length': '256', 'blank': 'True'}),
            'userinputfields': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'null': 'True', 'blank': 'True'}),
            'version': ('django.db.models.fields.CharField', [], {'max_length': '256'})
        },
        'rundb.pluginresult': {
            'Meta': {'ordering': "['-id']", 'unique_together': "(('plugin', 'result'),)", 'object_name': 'PluginResult'},
            'apikey': ('django.db.models.fields.CharField', [], {'max_length': '256', 'null': 'True', 'blank': 'True'}),
            'config': ('django.db.models.fields.TextField', [], {'default': "''", 'blank': 'True'}),
            'endtime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'jobid': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'plugin': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['rundb.Plugin']"}),
            'result': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'pluginresult_set'", 'to': "orm['rundb.Results']"}),
            'size': ('django.db.models.fields.BigIntegerField', [], {'default': '-1'}),
            'starttime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'state': ('django.db.models.fields.CharField', [], {'max_length': '20'}),
            'store': ('django.db.models.fields.TextField', [], {'default': "'{}'", 'blank': 'True'})
        },
        'rundb.project': {
            'Meta': {'object_name': 'Project'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'creator': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['auth.User']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '64'}),
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
        'rundb.qctype': {
            'Meta': {'object_name': 'QCType'},
            'defaultThreshold': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'description': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'maxThreshold': ('django.db.models.fields.PositiveIntegerField', [], {'default': '100'}),
            'minThreshold': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'qcName': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '512'})
        },
        'rundb.qualitymetrics': {
            'Meta': {'object_name': 'QualityMetrics'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'q0_100bp_reads': ('django.db.models.fields.IntegerField', [], {}),
            'q0_150bp_reads': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
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
            'default': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'dirPath': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'webServerPath': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'rundb.results': {
            'Meta': {'object_name': 'Results'},
            'analysisVersion': ('django.db.models.fields.CharField', [], {'max_length': '256'}),
            'autoExempt': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'diskusage': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'eas': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'results_set'", 'null': 'True', 'to': "orm['rundb.ExperimentAnalysisSettings']"}),
            'experiment': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'results_set'", 'to': "orm['rundb.Experiment']"}),
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
            'reportStatus': ('django.db.models.fields.CharField', [], {'default': "'Nothing'", 'max_length': '64', 'null': 'True'}),
            'reportstorage': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'storage'", 'null': 'True', 'to': "orm['rundb.ReportStorage']"}),
            'representative': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
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
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'meta': ('django.db.models.fields.TextField', [], {'default': "''", 'null': 'True', 'blank': 'True'}),
            'nucleotideType': ('django.db.models.fields.CharField', [], {'default': "'dna'", 'max_length': '64', 'blank': 'True'}),
            'runType': ('django.db.models.fields.CharField', [], {'max_length': '512'})
        },
        'rundb.sample': {
            'Meta': {'object_name': 'Sample'},
            'date': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'description': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'null': 'True', 'blank': 'True'}),
            'displayedName': ('django.db.models.fields.CharField', [], {'max_length': '127', 'null': 'True', 'blank': 'True'}),
            'experiments': ('django.db.models.fields.related.ManyToManyField', [], {'symmetrical': 'False', 'related_name': "'samples'", 'null': 'True', 'to': "orm['rundb.Experiment']"}),
            'externalId': ('django.db.models.fields.CharField', [], {'max_length': '127', 'null': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '127', 'null': 'True', 'blank': 'True'}),
            'status': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '512', 'blank': 'True'})
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
            'HPAccuracy': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'Meta': {'object_name': 'TFMetrics'},
            'Q10Histo': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'Q10Mean': ('django.db.models.fields.FloatField', [], {}),
            'Q10ReadCount': ('django.db.models.fields.FloatField', [], {}),
            'Q17Histo': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'Q17Mean': ('django.db.models.fields.FloatField', [], {}),
            'Q17ReadCount': ('django.db.models.fields.FloatField', [], {}),
            'SysSNR': ('django.db.models.fields.FloatField', [], {}),
            'aveKeyCount': ('django.db.models.fields.FloatField', [], {}),
            'corrHPSNR': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'keypass': ('django.db.models.fields.FloatField', [], {}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '128', 'db_index': 'True'}),
            'number': ('django.db.models.fields.FloatField', [], {}),
            'report': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'tfmetrics_set'", 'to': "orm['rundb.Results']"}),
            'sequence': ('django.db.models.fields.CharField', [], {'max_length': '512'})
        },
        'rundb.threeprimeadapter': {
            'Meta': {'object_name': 'ThreePrimeadapter'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'direction': ('django.db.models.fields.CharField', [], {'default': "'Forward'", 'max_length': '20'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isDefault': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '256'}),
            'runMode': ('django.db.models.fields.CharField', [], {'default': "'single'", 'max_length': '64', 'blank': 'True'}),
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
            'user': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['auth.User']", 'unique': 'True'})
        },
        'rundb.variantfrequencies': {
            'Meta': {'object_name': 'VariantFrequencies'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '3024', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '512', 'blank': 'True'})
        }
    }

    complete_apps = ['rundb']
    
