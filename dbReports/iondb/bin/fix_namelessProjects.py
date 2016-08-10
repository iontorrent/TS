#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
import sys
import os
from iondb.rundb import models

# 20121228 change log:
# dissociated link between plan and project with name ""
# deleted project from db with name ""


projects = models.Project.objects.filter(name="")

if projects:
    for project in projects:
        plans = models.PlannedExperiment.objects.filter(projects__id=project.pk)
        if plans:
            for plan in plans:

                print "*** Going to dissociate plan pk=%s from nameless project pk=%s" % (str(plan.pk), str(project.pk))
                plan.projects.remove(project.id)

        print "*** Going to delete nameless project pk=%s" % (str(project.pk))
        project.delete()
else:
    print "No nameless projects found. Nothing to fix."
