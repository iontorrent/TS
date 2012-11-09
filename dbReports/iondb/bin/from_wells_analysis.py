#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#
# PURPOSE: Command line utility which initiates from-wells analysis given a directory
# path to 1.wells data.
#
# USE CASE: Importing sig proc results to a different TS.
# Requires that the given argument is a filesystem path to a directory containing:
#
# explog.txt
# sigproc_results/1.wells
# sigproc_results/bfmask.bin
# sigproc_results/bfmask.stats
# sigproc_results/Bead_density_contour.png
# sigproc_results/avgNukeTrace_*.txt
#
################################################################################
import os
import sys
import json
import glob
import time
import copy
import string
import random
import urllib
import argparse
import datetime
import traceback
import logging
logging.basicConfig(level=logging.DEBUG)
from djangoinit import *
from iondb.rundb import models
from ion.utils.explogparser import load_log_path
from ion.utils.explogparser import parse_log
from iondb.utils.crawler_utils import getFlowOrder
from iondb.utils.crawler_utils import folder_mtime

TIMESTAMP_RE = models.Experiment.PRETTY_PRINT_RE
def extract_rig(folder):
    """Given the name of a folder storing experiment data, return the name
    of the PGM from which the date came."""
    #return os.path.basename(os.path.dirname(folder))
    return "uploads"

def exp_kwargs(d,folder):
    """Converts the output of `parse_log` to the a dictionary of
    keyword arguments needed to create an ``Experiment`` database object. """
    identical_fields = ("sample","library","cycles","flows",)
    simple_maps = (
        ("experiment_name","expName"),
        ("chiptype", "chipType"),
        ("chipbarcode", "chipBarcode"),
        ("user_notes", "notes"),
        ##("seqbarcode", "seqKitBarcode"),
        ("autoanalyze", "autoAnalyze"),
        ("prebeadfind", "usePreBeadfind"),
        ##("librarykeysequence", "libraryKey"),
        ("barcodeid", "barcodeId"),
        ("isReverseRun", "isReverseRun"),
        )
    full_maps = (
        ("pgmName",d.get('devicename',extract_rig(folder))),
        ("log", json.dumps(d, indent=4)),
        ("expDir", folder),
        ("unique", folder),
        ("baselineRun", d.get("runtype") == "STD" or d.get("runtype") == "Standard"),
        ("date", folder_mtime(folder)),
        ("storage_options",models.GlobalConfig.objects.all()[0].default_storage_options),
        ("flowsInOrder",getFlowOrder(d.get("image_map", ""))),
        ("reverse_primer",d.get('reverse_primer', 'Ion Kit')),
        )

    derive_attribute_list = [ "libraryKey", "reverselibrarykey", "forward3primeadapter", 
                             "reverse3primeadapter", "sequencekitname", "seqKitBarcode", 
                             "sequencekitbarcode", "librarykitname", "librarykitbarcode",
                             "runMode", "selectedPlanShortID","selectedPlanGUID"]
    
    ret = {}
    for f in identical_fields:
        ret[f] = d.get(f,'')
    for k1,k2 in simple_maps:
        ret[k2] = d.get(k1,'')
    for k,v in full_maps:
        ret[k] = v

    for attribute in derive_attribute_list:
        ret[attribute] = ''

    #N.B. this field is not used
    ret['storageHost'] = 'localhost'

    # If Flows keyword is defined in explog.txt...
    if ret['flows'] != "":
        # Cycles should be based on number of flows, not cycles published in log file
        # (Use of Cycles is deprecated in any case! We should be able to enter a random number here)
        ret['cycles'] = int( int(ret['flows']) / len(ret['flowsInOrder']) )
    else:
        # ...if Flows is not defined in explog.txt:  (Very-old-dataset support)
        ret['flows'] = len(ret['flowsInOrder']) * int(ret['cycles'])
        logging.warn ("Flows keyword missing: Calculated Flows is %d" % int(ret['flows']))

    if ret['barcodeId'].lower() == 'none':
        ret['barcodeId'] = ''
        
    if len(d.get('blocks',[])) > 0:
        ret['rawdatastyle'] = 'tiled'
        ret['autoAnalyze'] = False
        for bs in d['blocks']:
            # Hack alert.  Watch how explogparser.parse_log munges these strings when detecting which one is the thumbnail entry
            # Only thumbnail will have 0,0 as first and second element of the string.
            if '0' in bs.split(',')[0] and '0' in bs.split(',')[1]:
                continue
            if auto_analyze_block(bs):
                ret['autoAnalyze'] = True
                logging.debug ("Block Run. Detected at least one block to auto-run analysis")
                break
        if ret['autoAnalyze'] == False:
            logging.debug ("Block Run. auto-run whole chip has not been specified")
    else:
        ret['rawdatastyle'] = 'single'

    planShortId = d.get("planned_run_short_id", '')
    
    #fix [TS-3064] for PGM backward compatibility
    if (planShortId == None or len(planShortId) == 0):
        planShortId = d.get("pending_run_short_id", '')
    
    selectedPlanGUId = d.get("planned_run_guid", '')
                         
    ret["selectedPlanShortID"] = planShortId
    ret["selectedPlanGUID"] = selectedPlanGUId
                          
    logging.debug ("...planShortId=%s; selectedPlanGUId=%s" % (planShortId, selectedPlanGUId))
    print 'crawler: plannedRunShortId=', planShortId        
    print 'crawler: plannedRunGUId=', selectedPlanGUId

    sequencingKitName = d.get("seqkitname", '')
    if sequencingKitName != "NOT_SCANNED":
        ret['sequencekitname'] = sequencingKitName
        
    #in rundb_experiment, there are 2 attributes for sequencingKitBarcode!!
    sequencingKitBarcode = d.get("seqkitpart", '')
    if sequencingKitBarcode != "NOT_SCANNED":
        ret['seqKitBarcode'] = sequencingKitBarcode
        ret['sequencekitbarcode'] = sequencingKitBarcode

    libraryKitName = d.get('libkit', '')
    if libraryKitName != "NOT_SCANNED":
        ret['librarykitname'] = libraryKitName

    libraryKitBarcode = d.get("libbarcode", '')
    if libraryKitBarcode != "NOT_SCANNED":
        ret['librarykitbarcode'] = libraryKitBarcode
        

    #Rules for applying the library key overrides: 
    #1) If plan is used and library key is specified, use that value
    #2) Otherwise, if user has specified one on PGM's advanced page 
    #   Validation required:
    #   Why: It could be left-over from a previous run and is not compatible with current run)
    #   How: It has to be pre-defined in db and is in the direction of the the new run.
    #   What: If it passes validation, use it
    #3) Otherwise, use system default for that direction
    #4) If plan is NOT used, and user has specified one in PGM's advanced page, do validation as above
    #5) If it passes validation, use it
    #6) Otherwise, use system default for that direction as defined in db 
    #7) If the system default somehow has no value, we'll use the library key from 
    #   PGM's advanced setup page 

    isPlanFound = False
    planObj = None
    if selectedPlanGUId:
        try:
            planObj = models.PlannedExperiment.objects.get(planGUID=selectedPlanGUId)
            isPlanFound = True
            ret["runMode"] = planObj.runMode
   
            expName = d.get("experiment_name", '')
            #fix TS-4714: fail-safe measure, mark the plan executed if instrument somehow does not mark it as executed
            if (not planObj.planExecuted):
                logging.warn("REPAIR: marking plan %s as executed for experiment %s" % (planObj.planGUID, expName))
                planObj.planExecuted = True

            planObj.expName = expName                
            planObj.save()
                                
        except models.PlannedExperiment.DoesNotExist:
            logging.warn("No plan with GUId %s found in database " % selectedPlanGUId )
        except models.PlannedExperiment.MultipleObjectsReturned:
            logging.warn("Multiple plan with GUId %s found in database " % selectedPlanGUId)
    else:        
        if (planShortId and len(planShortId) > 0):  
            try:
                #PGM should have set the plan as executed already
                #note: if instrument does not include plan GUID for the plan and does not mark the plan as executed,
                #crawler will not do any repair (to mark a plan as executed) and actually indexError will likely happen
                planObj = models.PlannedExperiment.objects.filter(planShortID=planShortId, planExecuted=True).order_by("-date")[0]
                isPlanFound = True
                ret["runMode"] = planObj.runMode
   
                planObj.expName = d.get("experiment_name", '')                
                planObj.save()
                               
                logging.debug("...planShortId=%s is for runMode=%s; reverse=%s" % (planShortId, planObj.runMode, planObj.isReverseRun))
            except IndexError:
                logging.warn("No plan with short id %s found in database " % planShortId )

    #v3.0 if the plan is a paired-end plan, find the parent plan. 
    #if parent plan is found and current status is NOT "reserved", mark it as "reserved"
    #if parent plan is found and current status is already "reserved", mark it as planExecuted
    if isPlanFound:
        if not planObj.parentPlan == None:
            logging.debug("crawler planObj.parentPlan.guid=%s" % (planObj.parentPlan.planGUID))
            
            isNeedUpdate = False
            if (planObj.parentPlan.planStatus == "reserved"):
                logging.debug("crawler GOING TO CHECK CHILDREN... planObj.parentPlan.guid=%s" % (planObj.parentPlan.planGUID))
                
                isChildPlanAllDone = True
                childPlans = planObj.parentPlan.childPlan_set.all()
                for childPlan in childPlans:         
                    if (childPlan.planExecuted or childPlan.planStatus == "voided"):
                        continue
                    else:
                        logging.debug("crawler NOT ALL child plans are DONE... planObj.parentPlan.guid=%s" % (planObj.parentPlan.planGUID))
                        isChildPlanAllDone = False
                
                if isChildPlanAllDone and planObj.parentPlan.planExecuted == False:
                    logging.debug("crawler GOING TO SET PLANEXECUTED to TRUE... planObj.parentPlan.guid=%s" % (planObj.parentPlan.planGUID))
                    planObj.parentPlan.planExecuted = "True"
                    isNeedUpdate = True
                
            else:
                planObj.parentPlan.planStatus = "reserved"
                isNeedUpdate = True
                
            if isNeedUpdate:
                logging.debug("crawler GOING TO SAVE... planObj.parentPlan.guid=%s" % (planObj.parentPlan.planGUID))
                planObj.parentPlan.save()
            
        else:
            logging.debug("crawler NO PARENT PLAN... planObj.name=%s" % (planObj.planName))
    else:
        #if user does not use a plan for the run, fetch the system default plan template, and clone it for this run

        isNeedSystemDefaultTemplate = False        
        experiment = None
        
        #if we have already saved the experiment to db, and is using the cloned system default plan, explog will
        #not know that
        try:
            experiment = models.Experiment.objects.get(expName = d.get("experiment_name", ''))

            planShortId = experiment.selectedPlanShortID
            selectedPlanGUId = experiment.selectedPlanGUID
 
            #if experiment has already been saved in db with no plan associated with it, don't bother to fix it up
            if (planShortId == None or len(planShortId) == 0):            
                ##don't bother to create plans for old runs that don't use plans
                isNeedSystemDefaultTemplate = False
            else:
                #watch out: if the run is using a system default plan template clone, explog will not have all the info                                                        
                ##logging.info("DFLTTEST...#0 explog.planShortId=%s explog.selectedPlanGUId=%s" % (ret["selectedPlanShortID"], ret["selectedPlanGUID"]))

                if (ret["selectedPlanShortID"] == planShortId):                                       
                    ##logging.info("DFLTTEST #0 exp from DB...NOTHING TO FIX... planShortId=%s selectedPlanGUId=%s" % (planShortId, selectedPlanGUId))
                    #this case should have been handled by the code above and should not happen here
                    pass
                else:
                    ret["selectedPlanShortID"] = planShortId
                    ret["selectedPlanGUID"] = selectedPlanGUId
            
                    if (selectedPlanGUId and len(selectedPlanGUId) > 0):
                   
                        ##logging.info("DFLTTEST #1 exp from DB...planShortId=%s selectedPlanGUId=%s" % (planShortId, selectedPlanGUId))                        
                        try:
                            planObj = models.PlannedExperiment.objects.get(planGUID=selectedPlanGUId)
                            isPlanFound = True
                            ret["runMode"] = planObj.runMode
            
                        except models.PlannedExperiment.DoesNotExist:
                            logging.warn("No plan with GUId %s found in database " % selectedPlanGUId )
                        except models.PlannedExperiment.MultipleObjectsReturned:
                            logging.warn("Multiple plan with GUId %s found in database " % selectedPlanGUId)
                    else:        
                        if (planShortId and len(planShortId) > 0):  
                            try:
                                #PGM should have set the plan as executed already
                                planObj = models.PlannedExperiment.objects.filter(planShortID=planShortId, planExecuted=True).order_by("-date")[0]
                                isPlanFound = True
                                ret["runMode"] = planObj.runMode
                    
                                ##logging.info("DFLTTEST #2 exp from DB...planShortId=%s is for runMode=%s; reverse=%s" % (planShortId, planObj.runMode, planObj.isReverseRun))
                            except IndexError:
                                logging.warn("No plan with short id %s found in database " % planShortId )
      
        except models.Experiment.DoesNotExist:
            logging.warn("expName: %s not yet in database and may need a sys default plan" % d.get("experiment_name", ''))   
            
            #fix TS-4713 if a system default has somehow been cloned for this run, don't bother to clone again
            try:
                sysDefaultClones = models.PlannedExperiment.objects.filter(expName = d.get("experiment_name", ''))
                
                if sysDefaultClones:
                    #logging.debug("SKIP cloning system default plan for %s since one already exists " % (d.get("experiment_name", '')))
                    isNeedSystemDefaultTemplate = False
                else:
                    isNeedSystemDefaultTemplate = True
            except:
                logging.warn(traceback.format_exc())                
                isNeedSystemDefaultTemplate = False
        except models.Experiment.MultipleObjectsReturned:
            #this should not happen since instrument assign uniques run name. But if it happens, don't bother to apply the
            #system default plan template to the experiment 
            logging.warn("multiple expName: %s found in database" % d.get("experiment_name", ''))        
            isNeedSystemDefaultTemplate = False            
            
 
        if isNeedSystemDefaultTemplate == True:
            try:
                systemDefaultPlanTemplate = models.PlannedExperiment.objects.filter(isReusable=True, isSystem=True, isSystemDefault=True).order_by("-date")[0]
            
                planObj = copy.copy(systemDefaultPlanTemplate)
                planObj.pk = None
                planObj.planGUID = None
                planObj.planShortID = None
                planObj.isReusable = False
                planObj.isSystem = False
                planObj.isSystemDefault = False

                #fix TS-4664: include experiment name to system default clone
                expName = d.get("experiment_name", '')
                planObj.planName = "CopyOfSystemDefault_" +  expName
                planObj.expName = expName
                
                planObj.planExecuted = True

                planObj.save()

                #clone the qc thresholds as well
                qcValues = systemDefaultPlanTemplate.plannedexperimentqc_set.all()
                
                for qcValue in qcValues:
                    qcObj  = copy.copy(qcValue)

                    qcObj.pk = None
                    qcObj.plannedExperiment = planObj
                    qcObj.save()
                    
                planShortId = planObj.planShortID
                selectedPlanGUId = planObj.planGUID
                ret["selectedPlanShortID"] = planShortId
                ret["selectedPlanGUID"] = selectedPlanGUId
            
                logging.info("crawler AFTER SAVING SYSTEM DEFAULT CLONE %s for experiment=%s;"  % (planObj.planName, expName))                                  
                isPlanFound = True
                ret["runMode"] = planObj.runMode
                
            except IndexError:
                logging.warn("No system default plan template found in database ")
            except:
                logging.warn(traceback.format_exc())
                logging.warn("Error in trying to use system default plan template for experiment=%s" % (d.get("experiment_name", '')))        


    # planObj is initialized as None, which is an acceptable foreign key value
    # for Experiment.plan; however, by this point, we have either found a plan
    # or created one from the default plan template, so there should be a plan
    # and in all three cases, None, found plan, and default plan, we're ready 
    # to commit the plan object to the Experiment.plan foreign key relationship.
    ret["plan"] = planObj

    isReverseRun = d.get("isreverserun", '')
    
    #if PGM is running the old version, there is no isReverseRun in explog.txt. Check the plan if used
    if not isReverseRun:
        if isPlanFound:
            if planObj.isReverseRun:
                isReverseRun = "Yes"
            else:
                isReverseRun = "No"

    if isReverseRun == "Yes":
        ret['isReverseRun'] = True
        ret['reverselibrarykey'] = ''

        if isPlanFound == False:
            ret["runMode"] = "pe"
            
        try:
            defaultReverseLibraryKey = models.LibraryKey.objects.get(direction='Reverse', isDefault=True)
            defaultReverse3primeAdapter = models.ThreePrimeadapter.objects.get(direction='Reverse', isDefault=True)
            
            validatedPgmLibraryKey = None
            dbPgmLibraryKey = None
            pgmLibraryKey = d.get("librarykeysequence", '')
            
            hasPassed = False
            if pgmLibraryKey == None or len(pgmLibraryKey) == 0:
                #logging.debug("...pgmLibraryKey not specified. ")
                hasPassed = False
            else:
                dbPgmLibraryKeys = models.LibraryKey.objects.filter(sequence=pgmLibraryKey)
        
                if dbPgmLibraryKeys:
                    for dbKey in dbPgmLibraryKeys:
                        if dbKey.direction == "Reverse":
                            #logging.debug("...pgmLibraryKey %s has been validated for reverse run" % pgmLibraryKey)
                            validatedPgmLibraryKey = dbKey
                            hasPassed = True
                            break
                else:
                    hasPassed = False
            
            #set default in case plan is not used or not found in db
            if hasPassed:
                #logging.debug("...Default for reverse run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)

                ret['reverselibrarykey'] = validatedPgmLibraryKey.sequence
            else:
                #logging.debug("...Default for reverse run. Use default library key=%s " % defaultReverseLibraryKey.sequence)               
                ret['reverselibrarykey'] = defaultReverseLibraryKey.sequence
                
            ret['reverse3primeadapter'] = defaultReverse3primeAdapter.sequence

            if isPlanFound:
                #logging.debug("...REVERSE plan is FOUND for planShortId=%s " % planShortId)
                    
                if planObj.reverselibrarykey:
                    #logging.debug("...Plan used for reverse run. Use plan library key=%s " % planObj.reverselibrarykey)
    
                    ret['reverselibrarykey'] = planObj.reverselibrarykey
                else:
                    if hasPassed:
                        #logging.debug("...Plan used for reverse run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)
    
                        ret['reverselibrarykey'] = validatedPgmLibraryKey.sequence
                    else:
                        #logging.debug("...Plan used for reverse run. Use default library key=%s " % defaultReverseLibraryKey.sequence)
    
                        ret['reverselibrarykey'] = defaultReverseLibraryKey.sequence
                        
                    if planObj.reverse3primeadapter:
                        ret['reverse3primeadapter'] = planObj.reverse3primeadapter
                    else:
                        ret['reverse3primeadapter'] = defaultReverse3primeAdapter.sequence                    
                         
            else:
                if hasPassed:
                    #logging.debug("...Plan used but not on db for reverse run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)
    
                    ret['reverselibrarykey'] = validatedPgmLibraryKey.sequence
                else:
                    logging.debug("...Plan used but not on db for reverse run. Use default library key=%s " % defaultReverseLibraryKey.sequence)
                        
                    ret['reverselibrarykey'] = defaultReverseLibraryKey.sequence
    
                ret['reverse3primeadapter'] = defaultReverse3primeAdapter.sequence
                 
            #this should never happen                 
            if ret['reverselibrarykey']== None or len(ret['reverselibrarykey']) == 0:
                #logging.debug("...A library key cannot be determined for this REVERSE run  Use PGM default. ")
                ret['reverselibraryKey'] = d.get("librarykeysequence", '')

        except models.LibraryKey.DoesNotExist:
            logging.warn("No default reverse library key in database for experiment %s" % ret['expName'])
            return ret, False
        except models.LibraryKey.MultipleObjectsReturned:
            logging.warn("Multiple default reverse library keys found in database for experiment %s" % ret['expName'])
            return ret, False
        except models.ThreePrimeadapter.DoesNotExist:
            logging.warn("No default reverse 3' adapter in database for experiment %s" % ret['expName'])
            return ret, False
        except models.ThreePrimeadapter.MultipleObjectsReturned:
            logging.warn("Multiple default reverse 3' adapters found in database for experiment %s" % ret['expName'])
            return ret, False
        except:
            logging.warn("Experiment %s" % ret['expName'])
            logging.warn(traceback.format_exc())
            return ret, False
    else:
        ret['isReverseRun'] = False
        ret['libraryKey'] = ''
        
        if isPlanFound == False:
            ret["runMode"] = "single"
        
        defaultPairedEndForward3primeAdapter = None
        try:
            #Note: In v3.0, plan has concept of "runMode". 
            #TS-4524: allow crawler to be more tolerant, especially PE 3' adapter is only used for PE run
            defaultPairedEndForward3primeAdapter = models.ThreePrimeadapter.objects.get(direction="Forward", name__iexact = "Ion Paired End Fwd")
            
        except models.ThreePrimeadapter.DoesNotExist:
            logging.warn("No default pairedEnd forward 3' adapter in database for experiment %s" % ret['expName'])
        except models.ThreePrimeadapter.MultipleObjectsReturned:
            logging.warn("Multiple default pairedEnd forward 3' adapters found in database for experiment %s" % ret['expName'])

        try:
            #NOTE: In v2.2, there is no way to tell if a run (aka an experiment) is part of a paired-end forward run or not
            defaultForwardLibraryKey = models.LibraryKey.objects.get(direction='Forward', isDefault=True)
            defaultForward3primeAdapter = models.ThreePrimeadapter.objects.get(direction='Forward', isDefault=True)
                          
            validatedPgmLibraryKey = None
            dbPgmLibraryKey = None
            pgmLibraryKey = d.get("librarykeysequence", '')
            #logging.debug("...pgmLibraryKey is %s " % pgmLibraryKey)
                        
            hasPassed = False
            if pgmLibraryKey == None or len(pgmLibraryKey) == 0:
                #logging.debug("...pgmLibraryKey not specified. ")
                hasPassed = False
            else:
                dbPgmLibraryKeys = models.LibraryKey.objects.filter(sequence=pgmLibraryKey)
        
                if dbPgmLibraryKeys:
                    for dbKey in dbPgmLibraryKeys:
                        if dbKey.direction == "Forward":
                            #logging.debug("...pgmLibraryKey %s has been validated for forward run" % pgmLibraryKey)
                            validatedPgmLibraryKey = dbKey
                            hasPassed = True
                            break
                else:
                    hasPassed = False

            #set default in case plan is not used or not found in db           
            if hasPassed:
                #logging.debug("...Default for forward run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)

                ret['libraryKey'] = validatedPgmLibraryKey.sequence
            else:
                #logging.debug("...Default for forward run. Use default library key=%s " % defaultForwardLibraryKey.sequence)

                ret['libraryKey'] = defaultForwardLibraryKey.sequence
                
            ret['forward3primeadapter'] = defaultForward3primeAdapter.sequence     
                    
            if isPlanFound:
                #logging.debug("...FORWARD plan is FOUND for planShortId=%s " % planShortId)
                                    
                if planObj.libraryKey:
                    #logging.debug("...Plan used for forward run. Use plan library key=%s " % planObj.libraryKey)
    
                    ret['libraryKey'] = planObj.libraryKey
                else:
                    if hasPassed:
                        #logging.debug("...Plan used for forward run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)
    
                        ret['libraryKey'] = validatedPgmLibraryKey.sequence
                    else:
                        #logging.debug("...Plan used for forward run. Use default library key=%s " % defaultForwardLibraryKey.sequence)
    
                        ret['libraryKey'] = defaultForwardLibraryKey.sequence
                        
                if planObj.forward3primeadapter:
                    ret['forward3primeadapter'] = planObj.forward3primeadapter
                else:
                    if (planObj.runMode == "pe"):
                        if defaultPairedEndForward3primeAdapter:
                            ret['forward3primeadapter'] = defaultPairedEndForward3primeAdapter.sequence
                        else:
                            ret['forward3primeadapter'] = ""
                    else:
                        ret['forward3primeadapter'] = defaultForward3primeAdapter.sequence
            else:
                if hasPassed:
                    #logging.debug("...Plan used but not on db for forward run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)
    
                    ret['libraryKey'] = validatedPgmLibraryKey.sequence
                else:
                    #logging.debug("...Plan used but not on db for forward run. Use default library key=%s " % defaultForwardLibraryKey.sequence)
                        
                    ret['libraryKey'] = defaultForwardLibraryKey.sequence
    
                ret['forward3primeadapter'] = defaultForward3primeAdapter.sequence
                
                        
            if ret['libraryKey'] == None or ret['libraryKey'] == "":
                #logging.debug("...A library key cannot be determined for this FORWARD run  Use PGM default. ")
                ret['libraryKey'] = d.get("librarykeysequence", '')

        except models.LibraryKey.DoesNotExist:
            logging.warn("No default forward library key in database for experiment %s" % ret['expName'])
            return ret, False
        except models.LibraryKey.MultipleObjectsReturned:
            logging.warn("Multiple default forward library keys found in database for experiment %s" % ret['expName'])
            return ret, False
        except models.ThreePrimeadapter.DoesNotExist:
            logging.warn("No default forward 3' adapter in database for experiment %s" % ret['expName'])
            return ret, False
        except models.ThreePrimeadapter.MultipleObjectsReturned:
            logging.warn("Multiple default forward 3' adapters found in database for experiment %s" % ret['expName'])
            return ret, False                 
        except:
            logging.warn("Experiment %s" % ret['expName'])
            logging.warn(traceback.format_exc())
            return ret, False
                 
    # Limit input sizes to defined field widths in models.py
    ret['notes'] = ret['notes'][:1024]
    ret['expDir'] = ret['expDir'][:512]
    ret['expName'] = ret['expName'][:128]
    ret['pgmName'] = ret['pgmName'][:64]
    ret['unique'] = ret['unique'][:512]
    ret['storage_options'] = ret['storage_options'][:200]
 #   ret['project'] = ret['project'][:64]
    ret['sample'] = ret['sample'][:64]
    ret['library'] = ret['library'][:64]
    ret['chipBarcode'] = ret['chipBarcode'][:64]
    ret['seqKitBarcode'] = ret['seqKitBarcode'][:64]
    ret['chipType'] = ret['chipType'][:32]
    ret['flowsInOrder'] = ret['flowsInOrder'][:512]
    ret['libraryKey'] = ret['libraryKey'][:64]
    ret['barcodeId'] = ret['barcodeId'][:128]
    ret['reverse_primer'] = ret['reverse_primer'][:128]
    ret['reverselibrarykey'] = ret['reverselibrarykey'][:64]    
    ret['reverse3primeadapter'] = ret['reverse3primeadapter'][:512]
    ret['forward3primeadapter'] = ret['forward3primeadapter'][:512]
    ret['sequencekitbarcode'] = ret['sequencekitbarcode'][:512]
    ret['librarykitbarcode'] = ret['librarykitbarcode'][:512]
    ret['sequencekitname'] = ret['sequencekitname'][:512]    
    ret['sequencekitbarcode'] = ret['sequencekitbarcode'][:512]
    ret['librarykitname'] = ret['librarykitname'][:512]    
    ret['librarykitbarcode'] = ret['librarykitbarcode'][:512]
    ret['runMode'] = ret['runMode'][:64]
    ret['selectedPlanShortID'] = ret['selectedPlanShortID'][:5]
    ret['selectedPlanGUID'] = ret['selectedPlanGUID'][:512]

    logging.debug("For experiment %s" % ret['expName'])        
    logging.debug("...Ready to save run: isReverseRun=%s;" % ret['isReverseRun'])
    logging.debug("...Ready to save run: libraryKey=%s;" % ret['libraryKey']) 
    logging.debug("...Ready to save run: forward3primeadapter=%s;" % ret['forward3primeadapter'])     
    logging.debug("...Ready to save run: reverselibrarykey=%s;" % ret['reverselibrarykey'])     
    logging.debug("...Ready to save run: reverse3primeadapter=%s;" % ret['reverse3primeadapter'])  
        
    return ret, True

# Extracted from crawler.py and stem modified to be Reanalysis instead of Auto
def get_name_from_json(exp, key, thumbnail_analysis):
    data = exp.log
    name = data.get(key, False)
    twig = ''
    if thumbnail_analysis:
        twig = '_tn'
    # also ignore name if it has the string value "None"
    if not name or name == "None":
        uniq = ''.join(random.choice(string.letters + string.digits) for i in xrange(4))
        return 'Reanalysis_%s_%s_%s%s' % (exp.pretty_print().replace(" ","_"),exp.pk,uniq,twig)
    else:
        return '%s_%s%s' % (str(name),exp.pk,twig)
        
# Extracted from crawler.py and modified to launch fromWells analysis
def generate_http_post(exp,projectName,data_path,thumbnail_analysis=False,pe_forward="",pe_reverse="",report_name=""):
    def get_default_command(chipName):
        gc = models.GlobalConfig.objects.all()
        ret = None
        if len(gc)>0:
            for i in gc:
                ret = i.default_command_line
        else:
            ret = 'Analysis'
        
        # Force the from-wells option here
        ret = ret + " --from-wells %s" % os.path.join(data_path,"1.wells")
        
        # Need to also get chip specific arguments from dbase
        #print "chipType is %s" % chipName
        chips = models.Chip.objects.all()
        for chip in chips:
            if chip.name in chipName:
                ret = ret + " " + chip.args
        return ret

    try:
        GC = models.GlobalConfig.objects.all().order_by('pk')[0]
        basecallerArgs = GC.basecallerargs
        base_recalibrate = GC.base_recalibrate
    except models.GlobalConfig.DoesNotExist:
        basecallerArgs = "BaseCaller"
        base_recalibrate = False

    if pe_forward == "":
        pe_forward = "None"
    if pe_reverse == "":
        pe_reverse = "None"
    if report_name == "":
        report_name = get_name_from_json(exp,'autoanalysisname',thumbnail_analysis)

    params = urllib.urlencode({'report_name':report_name,
                            'tf_config':'',
                            'path':exp.expDir,
                            'args':get_default_command(exp.chipType),
                            'basecallerArgs':basecallerArgs,
                            'submit': ['Start Analysis'],
                            'do_thumbnail':"%r" % thumbnail_analysis,
                            'blockArgs':'fromWells',
                            'previousReport':os.path.join(data_path),
                            'forward_list':pe_forward,
                            'reverse_list':pe_reverse,
                            'project_names':projectName,
                            'do_base_recal':base_recalibrate})
    #logging.debug (params)
    headers = {"Content-type": "text/html",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    
    status_msg = report_name
    try:
        f = urllib.urlopen('http://127.0.0.1/rundb/newanalysis/%s/0' % (exp.pk), params)
        response = f.read()
        #print(response)
    except:
        print('could not autostart %s' % exp.expName)
        print(traceback.format_exc())
        try:
            f = urllib.urlopen('https://127.0.0.1/rundb/newanalysis/%s/0' % (exp.pk), params)
            response = f.read()
            #logging.debug(response)
        except:
            print('could not autostart %s' % exp.expName)
            print(traceback.format_exc())
            status_msg = None
    return status_msg

def newExperiment(explog_path):
    '''Create Experiment record'''
    # Parse the explog.txt file
    text = load_log_path(explog_path)
    dict = parse_log(text)
    
    # Create the Experiment Record
    folder = os.path.dirname(explog_path)
    
    # Test if Experiment object already exists
    try:
        newExp = models.Experiment.objects.get(unique=folder)
    except:
        newExp = None
        
    if newExp is None:
        try:
            expArgs,st = exp_kwargs(dict,folder)
            newExp = models.Experiment(**expArgs)
            newExp.save()
        except:
            newExp = None
            print traceback.format_exc()
    
    return newExp

def newReport(exp, data_path):
    '''Submit analysis job'''
    projectName = ''
    output = generate_http_post(exp,projectName,data_path)
    return output

def getReportURL(report_name):
    URLString = None
    try:
        report = models.Results.objects.get(resultsName=report_name)
        URLString = report.reportLink
    except models.Results.DoesNotExist:
        URLString = "Not found"
    except:
        print traceback.format_exc()
    finally:
        return URLString

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initiate from-wells analysis Report")
    parser.add_argument("directory",metavar="directory",help="Path to data to analyze")
    
    # If no arguments, print help and exit
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
        
    # Parse command line
    args = parser.parse_args()
    
    # Test inputs
    if not os.path.isdir(args.directory):
        print "Does not exist: %s" % args.directory
        sys.exit(1)
    src_dir = args.directory
    
    # Validate existence of prerequisite files
    explog_path = os.path.join(src_dir,'explog.txt')
    if not os.path.isfile(explog_path):
        print "Does not exist: %s" % explog_path
        print "Cannot create environment for re-analysis to take place"
        print "STATUS: Error"
        sys.exit(1)
        
    wells_path = os.path.join(src_dir,'sigproc_results','1.wells')
    if not os.path.isfile(wells_path):
        print "Does not exist: %s" % wells_path
        print "Cannot basecall without output from signal processing"
        print "STATUS: Error"
        sys.exit(1)
        
    testpath = os.path.join(src_dir,'sigproc_results','analysis.bfmask.bin')
    if not os.path.isfile(testpath):
        testpath = os.path.join(src_dir,'sigproc_results','bfmask.bin')
        if not os.path.isfile(testpath):
            print "Does not exist: %s" % testpath
            print "Cannot basecall without bfmask.bin from signal processing"
            print "STATUS: Error"
            sys.exit(1)
    
    testpath = os.path.join(src_dir,'sigproc_results','analysis.bfmask.stats')
    if not os.path.isfile(testpath):
        testpath = os.path.join(src_dir,'sigproc_results','bfmask.stats')
        if not os.path.isfile(testpath):
            print "Does not exist: %s" % testpath
            print "Cannot basecall without bfmask.stats from signal processing"
            print "STATUS: Error"
            sys.exit(1)
    
    # Missing these files just means key signal graph will not be generated
    testpath = os.path.join(src_dir,'sigproc_results','avgNukeTrace_ATCG.txt')
    if not os.path.isfile(testpath):
        print "Does not exist: %s" % testpath
        print "Cannot create TF key signal graph without %s file" % 'avgNukeTrace_ATCG.txt'
        
    testpath = os.path.join(src_dir,'sigproc_results','avgNukeTrace_TCAG.txt')
    if not os.path.isfile(testpath):
        print "Does not exist: %s" % testpath
        print "Cannot create Library key signal graph without %s file" % 'avgNukeTrace_TACG.txt'
    
    # Create Experiment record
    newExp = newExperiment(explog_path)
    if newExp is None:
        print "STATUS: Error"
        sys.exit(1)
    
    # Submit analysis job URL
    report_name = newReport(newExp, src_dir)
    if report_name is None:
        print "STATUS: Error"
        sys.exit(1)
    
    # Test for Report Object
    count = 0
    delay = 1
    retries = 60
    while count < retries:
        count += 1
        reportURL = getReportURL(report_name)
        if reportURL is None:
            print "STATUS: Error"
            sys.exit(1)
        elif reportURL == "Not found":
            print "Retry %d of %d in %d second" % (count,retries, delay)
            time.sleep(delay)
        else:
            count = retries
        
    print "STATUS: Success"
    print "REPORT-URL: %s" % reportURL
    
    