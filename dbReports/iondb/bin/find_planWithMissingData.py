#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
import sys
import os
from iondb.rundb import models

#20130114 change log:
# scanned db for plans that have no experiment
# scanned db for experiments with no experiementAnalysisSettings

hasErrors = False

plans = models.PlannedExperiment.objects.filter(experiment__id__isnull=True)
if plans:
    hasErrors = True
    print "*** Error: ", plans.count(), " plan(s) have no experiment associated with them: "
for plan in plans:
    print "Plan name=%s: database_id=%d; planStatus=%s; planExecuted=%s; isTemplate=%s" %(plan.planName, plan.id, plan.planStatus, str(plan.planExecuted), str(plan.isReusable))
    
exps = models.Experiment.objects.filter(eas_set__isnull = True)
if exps:
    hasErrors = True
        
    if plans:
        print ""
    print "*** Error: ", exps.count(), " experiment(s) have no experimentAnalysisSettings associated with them: "
for exp in exps:
    print "Experiment name=%s: database_id=%d; status=%s" %(exp.expName, exp.id, str(exp.status))
    
if not hasErrors:
    print "*** Good! No missing data found in the plans or experiments."

