#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("upgrade_analysisArgs_templates_n_unused_plans")
#log.setLevel(logging.DEBUG)

from django.db import transaction

import iondb.bin.djangoinit
from iondb.rundb import models

# import pytz
# from datetime import datetime

from traceback import format_exc


#
# PURPOSE:
#
# This script is to list or upgrade all the analysis arguments based for either
#
#    All the system templates
#    All system & user-created templates and ALL the user-created unused plans [default and list]
#    All the user-created templates and user-created unused plans
#
# Before you upgrade the db records, you can set the INTENT to list what plans and/or templates have been detected to upgrade.
#
# Usage: Change the INTENT value to whatever option desired and run this script


class Intents:
    LIST_ALL = 1
    UPGRADE_ALL = 2 
    LIST_SYS_TEMPLATES_ONLY = 3 
    UPGRADE_SYS_TEMPLATES_ONLY = 4
    LIST_USER_TEMPLATES_PLANS_ONLY = 5
    UPGRADE_USER_TEMPLATES_PLANS_ONLY = 6
    

def do_upgrade(plan, best_match_analysis_args):
    """
    Upgrade analysisArgs in plan or template
    """
    plan.alignmentargs = best_match_analysis_args.alignmentargs 
    plan.analysisargs = best_match_analysis_args.analysisargs
    plan.basecallerargs = best_match_analysis_args.basecallerargs
    plan.beadfindargs = best_match_analysis_args.beadfindargs
    plan.calibrateargs = best_match_analysis_args.calibrateargs
    plan.prebasecallerargs = best_match_analysis_args.prebasecallerargs
    plan.ionstatsargs = best_match_analysis_args.ionstatsargs
    
    plan.prethumbnailbasecallerargs = best_match_analysis_args.prethumbnailbasecallerargs
    plan.thumbnailalignmentargs = best_match_analysis_args.thumbnailalignmentargs
    plan.thumbnailanalysisargs = best_match_analysis_args.thumbnailanalysisargs
    plan.thumbnailbasecallerargs = best_match_analysis_args.thumbnailbasecallerargs
    plan.thumbnailbeadfindargs = best_match_analysis_args.thumbnailbeadfindargs
    plan.thumbnailcalibrateargs = best_match_analysis_args.thumbnailcalibrateargs
    plan.thumbnailionstatsargs = best_match_analysis_args.thumbnailionstatsargs

def is_upgrade_needed(eas, best_match_analysis_args):
    """
    Return a boolean to indicate if upgrade is needed.
    """
    has_differences = False

    alignmentargs = eas.alignmentargs
    analysisargs = eas.analysisargs
    basecallerargs = eas.basecallerargs
    beadfindargs = eas.beadfindargs
    calibrateargs = eas.calibrateargs
    prebasecallerargs = eas.prebasecallerargs
    ionstatsargs = eas.ionstatsargs
    
    prethumbnailbasecallerargs = eas.prethumbnailbasecallerargs
    
    thumbnailalignmentargs = eas.thumbnailalignmentargs
    thumbnailanalysisargs = eas.thumbnailanalysisargs
    thumbnailbasecallerargs = eas.thumbnailbasecallerargs
    thumbnailbeadfindargs = eas.thumbnailbeadfindargs
    thumbnailcalibrateargs = eas.thumbnailcalibrateargs
    thumbnailionstatsargs = eas.thumbnailionstatsargs
    
    if (alignmentargs != best_match_analysis_args.alignmentargs):
        log.info("DIFF...orig alignmentargs=%s" %(alignmentargs))
        log.info("DIFF...new alignmentargs=%s" %(best_match_analysis_args.alignmentargs))
        
    if (analysisargs != best_match_analysis_args.analysisargs):
        log.info("DIFF...orig alignmentargs=%s" %(analysisargs))
        log.info("DIFF...new alignmentargs=%s" %(best_match_analysis_args.analysisargs))
        
    if (alignmentargs != best_match_analysis_args.alignmentargs or 
        analysisargs != best_match_analysis_args.analysisargs or
        basecallerargs != best_match_analysis_args.basecallerargs or
        beadfindargs != best_match_analysis_args.beadfindargs or
        calibrateargs != best_match_analysis_args.calibrateargs or
        prebasecallerargs != best_match_analysis_args.prebasecallerargs or
        ionstatsargs != best_match_analysis_args.ionstatsargs or
        prethumbnailbasecallerargs != best_match_analysis_args.prethumbnailbasecallerargs or
        thumbnailalignmentargs != best_match_analysis_args.thumbnailalignmentargs or
        thumbnailanalysisargs != best_match_analysis_args.thumbnailanalysisargs or
        thumbnailbasecallerargs != best_match_analysis_args.thumbnailbasecallerargs or
        thumbnailbeadfindargs != best_match_analysis_args.thumbnailbeadfindargs or
        thumbnailcalibrateargs != best_match_analysis_args.thumbnailcalibrateargs or
        thumbnailionstatsargs != best_match_analysis_args.thumbnailionstatsargs):
        has_differences = True
            
    return has_differences


def check_analysis_args(plan_or_template, latest_analysisArgs_objs, intent):
    """
    Find the latest analysis arguments for this plan or template.
    Return a boolean to indicate if upgrade is needed and the best_match_analysis_args.
    """
    error = None
    
    chipType = plan_or_template.experiment.chipType
    sequenceKitName = plan_or_template.experiment.sequencekitname
    templatingKitName = plan_or_template.templatingKitName
    libraryKitName = plan_or_template.latestEAS.libraryKitName
    samplePrepKitName = plan_or_template.samplePrepKitName
            
    # chip name backwards compatibility
    chipType = chipType.replace('"','')
    if models.Chip.objects.filter(name=chipType).count() == 0:
        chipType = chipType[:3]

    latest_analysisArgs_objs_by_chip = latest_analysisArgs_objs.filter(chipType=chipType).order_by('-pk')

    runType = plan_or_template.runType
    applicationGroupName = plan_or_template.applicationGroup.name if plan_or_template.applicationGroup else ""
                                                                                     
    best_match_analysis_args = models.AnalysisArgs.best_match(chipType, sequenceKitName, templatingKitName, libraryKitName, samplePrepKitName, latest_analysisArgs_objs_by_chip, runType, applicationGroupName)

    if best_match_analysis_args:
        log.info("for chipType=%s; sequenceKitName=%s; =%s; libraryKitName=%s; samplePrepKitName=%s; applicationType=%s; applicationGroupName=%s - best_match_analysis_args.pk=%d" %(chipType, sequenceKitName, templatingKitName, libraryKitName, samplePrepKitName, runType, applicationGroupName, best_match_analysis_args.pk))       
        is_to_upgrade = is_upgrade_needed(plan_or_template.latestEAS, best_match_analysis_args)
    else:
        is_to_upgrade = False
        error = "Unsupported chipType:%s" %(chipType)  
        
    return is_to_upgrade, best_match_analysis_args, error
    
    

@transaction.commit_manually()  
def process_data(plans_or_templates, intent):
    """
    Loop through the data to determine if any of the analysis arguments need to be upgraded.
    If intent is to list, just print the plan's pk and name that are upgrade candidates.
    If intent is to upgrade, upgrade the analysis arguments only of the plans that need to be upgraded.
    """
    
    latest_analysisArgs_objs = models.AnalysisArgs.objects.all().exclude(active = False).order_by('-pk')
            
    count_upgrade = 0
    count_error = 0
    
    try:
        if (len(latest_analysisArgs_objs) > 0):
            for existing_data in plans_or_templates:
                is_to_upgrade, best_match_analysis_args, error = check_analysis_args(existing_data, latest_analysisArgs_objs, intent)
        
                if is_to_upgrade:
                    if existing_data.isReusable:
                       log.info("Template: pk=%d; name=%s" %(existing_data.pk, existing_data.planDisplayedName))
                    else:
                       log.info("Plan: pk=%d; name=%s" %(existing_data.pk, existing_data.planDisplayedName))
    
                    count_upgrade += 1
                                        
                    if (intent in [Intents.UPGRADE_ALL, Intents.UPGRADE_SYS_TEMPLATES_ONLY, Intents.UPGRADE_USER_TEMPLATES_PLANS_ONLY]):
                        do_upgrade(existing_data, best_match_analysis_args)
                        
                        existing_data.save()
    
                        #record changes in journal log
                        if existing_data.isReusable:
                            msg = 'Analysis arguments upgraded for template: %s (%s)' % (existing_data.planDisplayedName, existing_data.pk)
                        else:
                            msg = 'Analysis arguments upgraded for plan: %s (%s)' % (existing_data.planDisplayedName, existing_data.pk)
                            
                        models.EventLog.objects.add_entry(existing_data, msg)
    
                elif error:
                    count_error += 1
                    if existing_data.isReusable:
                       log.warn("Skipped upgrading template: pk=%d; name=%s due to unsupported chipType=%s" %(existing_data.pk, existing_data.planDisplayedName, existing_data.experiment.chipType))
                    else:
                       log.warn("Skipped upgrading plan: pk=%d; name=%s due to unsupported chipType=%s" %(existing_data.pk, existing_data.planDisplayedName, existing_data.experiment.chipType))
                    
            if (intent in [Intents.UPGRADE_ALL, Intents.UPGRADE_SYS_TEMPLATES_ONLY, Intents.UPGRADE_USER_TEMPLATES_PLANS_ONLY]):        
                log.info(">> SUMMARY >> analysis arguments of %d plans and templates have been upgraded. " %(count_upgrade))
            else:
                log.info(">> SUMMARY >> analysis arguments of %d plans and templates are candidates to be upgraded. " %(count_upgrade))
                
            if count_error > 0:
                log.warn(">> SUMMARY >> Skipped upgrading %d plans and templates due to unsupported chipType" %(count_error))
               
 
        else:
            log.warning(">> SUMMARY >> No analysisArgs entries found in DB. Something is not right!  This upgrade script has nothing to do here.")
  
    except:
        print format_exc()
        transaction.rollback()
        print "*** Exceptions found. Upgrade(s) rolled back. ***"
    else: 
        ##if count_upgrade > 0:
        transaction.commit()
        if count_upgrade > 0:
            print "*** Template/plan upgrades committed. ***" 
        else:
            print "*** Done ***" 


def filter_querySet_for_analysisArgs(querySet):
   querySet2 = querySet.exclude(latestEAS__beadfindargs = "",
                                 latestEAS__thumbnailbeadfindargs = "",
                                 latestEAS__analysisargs = "",
                                 latestEAS__thumbnailanalysisargs = "",
                                 latestEAS__prebasecallerargs = "",
                                 latestEAS__prethumbnailbasecallerargs = "" ,
                                 latestEAS__calibrateargs = "",
                                 latestEAS__thumbnailcalibrateargs = "",
                                 latestEAS__basecallerargs = "",
                                 latestEAS__thumbnailbasecallerargs = "",
                                 latestEAS__alignmentargs = "",
                                 latestEAS__thumbnailalignmentargs = "",
                                 latestEAS__ionstatsargs = "",
                                 latestEAS__thumbnailionstatsargs = "")    
   return querySet2



if __name__ == '__main__':
    
    INTENT = Intents.LIST_ALL

#     INTENT = Intents.LIST_ALL
#     INTENT = Intents.UPGRADE_ALL
#     INTENT = Intents.LIST_SYS_TEMPLATES_ONLY
#     INTENT = Intents.UPGRADE_SYS_TEMPLATES_ONLY
#     INTENT = Intents.LIST_USER_TEMPLATES_PLANS_ONLY
#     INTENT = Intents.UPGRADE_USER_TEMPLATES_PLANS_ONLY


    if INTENT in [Intents.LIST_ALL, Intents.UPGRADE_ALL]: 
        querySet = models.PlannedExperiment.objects.filter(planExecuted = False).exclude(experiment__chipType = "").exclude(latestEAS__isEditable = False).order_by("isReusable")
        
        querySet2 = filter_querySet_for_analysisArgs(querySet)
                
        #log.info(">> ALL - querySet.count=%d; querySet2.count=%d" %(len(querySet), len(querySet2)))
        log.info(">> ALL - Among %d plans and templates " %(len(querySet2)))

        process_data(querySet2, INTENT)
        
    elif INTENT in [Intents.LIST_SYS_TEMPLATES_ONLY, Intents.UPGRADE_SYS_TEMPLATES_ONLY]:
        querySet = models.PlannedExperiment.objects.filter(isReusable = True, isSystem = True).exclude(experiment__chipType = "").exclude(latestEAS__isEditable = False).order_by("isReusable")
        querySet2 = filter_querySet_for_analysisArgs(querySet)        
        #log.info(">> SYS_TEMPLATES_ONLY - querySet.count=%d" %(len(querySet)))
        log.info(">> SYS_TEMPLATES_ONLY - Among %d plans and templates " %(len(querySet2)))

        process_data(querySet2, INTENT) 
               
    elif INTENT in [Intents.LIST_USER_TEMPLATES_PLANS_ONLY, Intents.UPGRADE_USER_TEMPLATES_PLANS_ONLY]:
        querySet = models.PlannedExperiment.objects.filter(planExecuted = False, isSystem = False).exclude(experiment__chipType = "").exclude(latestEAS__isEditable = False).order_by("isReusable")
        querySet2 = filter_querySet_for_analysisArgs(querySet)        
        #log.info(">> USER_TEMPLATES_PLANS_ONLY - querySet.count=%d" %(len(querySet)))        
        log.info(">> USER_TEMPLATES_PLANS_ONLY - Among %d plans and templates " %(len(querySet2)))

        process_data(querySet2, INTENT)        
        

