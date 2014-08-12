#!/usr/bin/python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import json
import traceback
import argparse
from iondb.bin import djangoinit
from iondb.rundb.models import Experiment, FileServer
from django.db import transaction

import iondb.rundb.data.dmactions_types as dmactions_types
from iondb.rundb.data.data_import import find_data_to_import, create_obj

def restore_from_json(data, path_to_data):
    # similar to data_import.load_serialized_json function
    # creates Plan, Experiment and ExperimentAnalysisSettings objects from serialized json
    saved_objs = {}
    create_sequence = ['planned experiment', 'experiment', 'experiment analysis settings']
    
    exp = Experiment.objects.filter(expDir=path_to_data)
    if exp:
        print "Experiment %s already exists, skipping." % exp[0].expName
        return saved_objs
    
    for d in data:
        if d['model'] == 'rundb.experiment':
            d['fields']['unique'] = path_to_data
            d['fields']['expDir'] = path_to_data
    
    try:
        with transaction.commit_on_success():
            for model_name in create_sequence:
                obj = create_obj(model_name, data, saved_objs)
                saved_objs[model_name] = obj
    except:
        print 'Failed to create database objects.'
        print traceback.format_exc()
    
    return saved_objs

def find_files():
    # search FileServer paths for serialized json files
    json_paths = []
    dirs = []
    for fs_path in FileServer.objects.all().order_by('pk').values_list('filesPrefix', flat=True):
        for found in find_data_to_import(fs_path, 3):
            for json_path in found['categories'].values():
                json_dir = os.path.dirname(json_path)
                if json_dir not in dirs:
                    dirs.append(json_dir)
                    json_paths.append(json_path)
    return json_paths

if __name__ == '__main__':
    '''
    Restores Experiment to database from serialized_*.json
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filepath', dest='filepath', action='store', default="", help="serialized json path to import from")
    args = parser.parse_args()
    
    if args.filepath:
        if not os.path.exists(args.filepath):
            print "Error: did not find file %s, exiting." % args.filepath
            sys.exit(1)
        else:
            json_paths = [args.filepath]
    else:
        json_paths = find_files()
    
    print "Processing %d datasets." % len(json_paths)
    
    for path in json_paths:
        try:
            with open(path) as f:
                data = json.load(f)
                print "Restoring from %s." % path
        except:
            print "Error: Unable to load %s." % path
            print traceback.format_exc()
        else:
            restore_from_json(data, os.path.dirname(path))
        