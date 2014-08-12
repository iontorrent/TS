# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#!/usr/bin/env python
from fabric.api import env, local, sudo, require, run, put, settings, cd
from fabric.utils import abort
import subprocess
from xml.etree import ElementTree as ET
import os
import sys
import time
try:
    from iondb import settings as base_settings
except ImportError, e:
    raise e


# globals
env.prj_name = 'ion-dbreports'
env.user = 'ionadmin'
env.password = 'ionadmin'
env.reject_unknown_hosts = False    
env.svn_base_url = 'https://iontorrent.jira.com/svn/TS'
env.svn_path = 'dbReports'

def nifty():
    ''' show nifty dev cmds'''
    #TODO:
    
    
def _ensure_virtualenv():
    if "VIRTUAL_ENV" not in os.environ:
        sys.stderr.write("$VIRTUAL_ENV not found. Make sure to activate virtualenv first\n\n")
        sys.exit(-1)
    env.virtualenv = os.environ["VIRTUAL_ENV"]


def dev_setup():
    '''Prepares the virtulenv by installing requirements using PIP''' 
    _ensure_virtualenv()
    _verify_local_settings()
    local('mkdir -p reports')
    local('pip install -q -r requirements/env.txt')
    local('pip install -q -r requirements/runtime.txt')
    local('pip install -q -r requirements/analysis.txt')
    local('pip install -q -r requirements/test.txt')


def _filecheck():
    '''Run the bamboo-filecheck.sh to ensure files have proper line endings and copyright text'''
    local('cd ..; MODULES="pipeline dbReports" ./buildTools/bamboo-filecheck.sh')

    
def _pep8():
    '''Run the pep8 tool to check code style'''
    local('pep8 --config=.pep8 iondb > reports/pep8.out')


def _pylint():
    '''Run the pep8 tool to check more code style and some basic flow checks'''
    local('pylint --rcfile=.pylintrc --ignore=migrations,tests.py iondb > reports/pylint.out')


def _pyflakes():
    '''Run the pyflakes tool to analysis the code'''
    local('pyflakes iondb > reports/pyflakes.out')


def _clonedigger():
    '''Run the clonedigger to analysis for duplicated code'''
    local('clonedigger --ignore-dir=migrations -o reports/clonedigger.html iondb/ ')

def _jshint():
    '''Inspecting JS using jshint'''
    local('cd iondb/media; make inspect > ../../reports/jshint.txt')

def jshint():
    '''Inspecting JS using jshint'''
    _jshint()
    
def check(warn_only=True, extended = False):
    '''Run the static analysis tools'''
    dev_setup()
    _filecheck()
    env.warn_only = warn_only  #setting env.warn_only to True doesn't cause the fab task to abort on shell failures returning non-zero
    _pep8()
    _pylint()
    _jshint()
    if extended:
        _clonedigger()
        _pyflakes()
    env.warn_only = False


def _verify_coverage(packages = ['.','iondb.rundb'], line_rate_threshold = 85, branch_rate_threshold = 85, fail = True):
    coverage_data = ET.parse('reports/coverage.xml')
    _packages = coverage_data.getroot().find('packages').findall('package')
    errors = []
    for package in _packages:
        if (package.attrib['name'] in packages):
            print 'Verifying coverage for package %s' %  package.attrib['name']
            line_rate = float(package.attrib['line-rate']) * 100
            if (line_rate < line_rate_threshold):
                errors.append('Line rate of %f%% is below %d%%' % (line_rate, line_rate_threshold))
            branch_rate = float(package.attrib['branch-rate']) * 100
            if (branch_rate < branch_rate_threshold):
                errors.append('Branch rate of %f%% is below %d%%' % (branch_rate, branch_rate_threshold))
    if errors and fail:
        abort('\n\t' + '\n\t'.join(errors))

def _grant_createdb_permission():
    local('psql -U postgres -c "ALTER USER ion WITH CREATEDB;"')


def _verify_local_settings():
    errors = []
    if not 'django_nose' in base_settings.INSTALLED_APPS or base_settings.INSTALLED_APPS.index('south') > base_settings.INSTALLED_APPS.index('django_nose'): 
        errors.append('!!! Please add \'django_nose\' to the INSTALLED APPS after the entry for \'south\' in your local_settings.py' )
        errors.append('!!! .TODO: can we simplify this? auto-gen local_settings; auto-update existing local_settings?? ' )
        
    if not 'iondb.test_runner.IonTestSuiteRunner' == base_settings.TEST_RUNNER:
        errors.append('!!! Please set TEST_RUNNER=\'iondb.test_runner.IonTestSuiteRunner\' in your settings.py or local_settings.py' )
    
    if errors:
        abort('\n\t' + '\n\t'.join(errors))


def test(reusedb=0, verbosity=2, coverage=1, viewcoverage=1, fail=1):
    '''Run the test suite and bail out if it fails
    
        :reusedb=[0|1] 1 will reuse an existing test db (test_iondb)
        :verbosity=[1|2|3] 1 is least, 2 is more, 3 is most verbose (test_iondb)
        :coverage=[0|1] 1 turns coverage on, 0 is off (test_iondb)
        :viewcoverage=[0|1] opens the html coverage report automatically, 0 is off (test_iondb)
        :fail=[0|1] 0 will ignore coverage check failures, 1 (default) will fail on coverage check failures 
    '''
    # dev_setup()
    _verify_local_settings()
    _grant_createdb_permission()
    options = ""
    if int(coverage) == 1: 
        options += "--cover-erase \
            --cover-package=iondb.rundb \
            --with-coverage --cover-xml --cover-xml-file=reports/coverage.xml \
            --cover-branches \
            --cover-html --cover-html-dir=reports/coverage/"

    cmd = "export REUSE_DB=%s; export PYTHONPATH=`pwd`; \
            python manage.py test --verbosity=%s \
            %s \
            iondb" % (reusedb, verbosity, options)
    flag = subprocess.call(cmd, shell=True)

    if int(coverage) == 1 and int(viewcoverage) == 1:
        local('firefox file://$PWD/reports/coverage/index.html')
    if int(coverage) == 1:
        _verify_coverage(fail=fail==1)
    
    if not flag and fail:
        sys.exit(1)
    else:
        sys.exit(0)
        pass
    # 


def clean():
    '''Cleans up generated files from the local file system'''
    local('rm -rf reports')

def _build_iondbreports():
    local('cd ..; rm -rf build/dbReports')
    local('cd ..; MODULES=dbReports ./buildTools/build.sh')

def _build_pipeline():
    local('cd ..; rm -rf build/pipeline')
    local('cd ..; MODULES=pipeline ./buildTools/build.sh')

def build():
    '''Builds ion-dbreports using CMake'''
    _filecheck()
    compilepy()
    cleanpyc()
    _build_pipeline()
    _build_iondbreports()
    
def install():
    '''Builds & Installs the ion-dbreports and dependencies from local'''
    build()
    local('sudo dpkg -i ../build/pipeline/*.deb')
    local('sudo gdebi ../build/dbReports/*.deb')

def precommit():
    '''Runs precommit checks - static analysis, unit tests, builds & installs .deb packages'''
    check()
    test()
    install()

def ci():
    '''Runs Continuous Integration build locally'''
    precommit()
    
    
def runserver():
    '''Starts the Django runserver '''
    local('export PYTHONPATH=`pwd`; \
            python manage.py runserver 0.0.0.0:8000')

def cleanpyc():
    '''Removes iondb/**/*.pyc files'''
    local('rm -f `/usr/bin/find "./iondb/" -name \'*.pyc\' -print`')
    
def compilepy():
    '''Compiles python code in iondb/**'''
    cleanpyc()
    local('python -m compileall -q -f "./iondb/"')
    
def shell():
    local('export PYTHONPATH=`pwd`; \
            python manage.py shell')
    
#def setup():
#    """
#    Setup a fresh virtualenv as well as a few useful directories, then run
#    a full deployment
#    """
#    require('hosts')
#    require('path')
#
#    # ensure apt is up to date
#    sudo('apt-get update')
#    # install Python environment
#    sudo('apt-get install -y build-essential python-dev python-setuptools python-virtualenv')
#    # install subversion
#    sudo('apt-get install -y subversion')
#
#    sudo('easy_install pip')

