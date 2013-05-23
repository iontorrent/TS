# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import subprocess
import socket
import logging
import sys
from django import forms
from django.forms import widgets as djangoWidget
from django.forms.extras import widgets
from django.forms.widgets import PasswordInput
from django import shortcuts
from iondb.rundb import models
import datetime
from iondb.utils import devices
from iondb.rundb import tasks
from django.forms.util import flatatt
from django.utils.safestring import mark_safe
from django.utils.encoding import StrAndUnicode, force_unicode
from itertools import chain
from django.utils.html import escape, conditional_escape
import string
from django.conf import settings

logger = logging.getLogger(__name__)

from django.utils.safestring import mark_safe
class Plugins_SelectMultiple(forms.CheckboxSelectMultiple):
    def render(self, name, value, attrs=None): 
        csm = super(Plugins_SelectMultiple, self).render(name, value, attrs=None)
        csm = csm.replace(u'<ul>', u'').replace(u'</ul>', u'').replace(u'<label>',u'').replace(u'</label>',u'')
        # add 'configure' button for plugins
        btn_html = '&nbsp; <button type="button" class="configure_plugin" id="configure_plugin_XXX" data-plugin_pk=XXX style="display: none;"> Configure </button>'
        output = ''
        for line in csm.split('<li>'):
            if line.find('value="') > 0:
                output += '<p>' + line.split('</li>')[0]
                pk = line.split('value="')[1].split('"')[0]
                plugin = models.Plugin.objects.get(pk=pk)
                if plugin.isPlanConfig():
                    output += btn_html.replace('XXX',pk)
                output += '</p>'            
        
        return mark_safe(output)

class DataSelect(djangoWidget.Widget):
    """this is added to be able to have data attribs to the options"""

    allow_multiple_selected = False

    def __init__(self, attrs=None, choices=()):
        super(DataSelect, self).__init__(attrs)
        # choices can be any iterable, but we may need to render this widget
        # multiple times. Thus, collapse it into a list so it can be consumed
        # more than once.
        self.choices = list(choices)

    def render(self, name, value, attrs=None, choices=()):
        if value is None: value = ''
        final_attrs = self.build_attrs(attrs, name=name)
        output = [u'<select%s>' % flatatt(final_attrs)]
        options = self.render_options(choices, [value])
        if options:
            output.append(options)
        output.append(u'</select>')
        return mark_safe(u'\n'.join(output))

    def render_option(self, selected_choices, option_value, option_label, option_data):
        option_value = force_unicode(option_value)
        if option_value in selected_choices:
            selected_html = u' selected="selected"'
            if not self.allow_multiple_selected:
                # Only allow for a single selection.
                selected_choices.remove(option_value)
        else:
            selected_html = ''
        return u'<option data-version="%s" value="%s"%s>%s</option>' % (
            option_data, escape(option_value), selected_html,
            conditional_escape(force_unicode(option_label)))

    def render_options(self, choices, selected_choices):
        # Normalize to strings.
        selected_choices = set(force_unicode(v) for v in selected_choices)
        output = []
        for option_value, option_label, option_data in chain(self.choices, choices):
                output.append(self.render_option(selected_choices, option_value, option_label, option_data))
        return u'\n'.join(output)

class RunParamsForm(forms.Form):

    report_name = forms.CharField(max_length=128,
                                widget=forms.TextInput(attrs={'size':'60','class':'textInput input-xlarge'}) )
    path = forms.CharField(max_length=512,widget=forms.HiddenInput)

    beadfindArgs = forms.CharField(max_length=1024,
                           required=False,
                           widget=forms.Textarea(attrs={'size':'512','class':'textInput input-xlarge','rows':4,'cols':50}))

    analysisArgs = forms.CharField(max_length=1024,
                           required=False,
                           widget=forms.Textarea(attrs={'size':'512','class':'textInput input-xlarge','rows':4,'cols':50}))

    prebasecallerArgs = forms.CharField(max_length=1024,
                                     required=False,
                                     widget=forms.Textarea(attrs={'size':'512','class':'textInput input-xlarge','rows':4,'cols':50}))

    basecallerArgs = forms.CharField(max_length=1024,
                           required=False,
                           widget=forms.Textarea(attrs={'size':'512','class':'textInput input-xlarge','rows':4,'cols':50}))

    thumbnailBeadfindArgs = forms.CharField(max_length=1024,
                                     required=False,
                                     widget=forms.Textarea(attrs={'size':'512','class':'textInput input-xlarge','rows':4,'cols':50}))

    thumbnailAnalysisArgs = forms.CharField(max_length=1024,
                                          required=False,
                                          widget=forms.Textarea(attrs={'size':'512','class':'textInput input-xlarge','rows':4,'cols':50}))

    prethumbnailBasecallerArgs = forms.CharField(max_length=1024,
                                              required=False,
                                              widget=forms.Textarea(attrs={'size':'512','class':'textInput input-xlarge','rows':4,'cols':50}))

    thumbnailBasecallerArgs = forms.CharField(max_length=1024,
                                     required=False,
                                     widget=forms.Textarea(attrs={'size':'512','class':'textInput input-xlarge','rows':4,'cols':50}))

    blockArgs = forms.CharField(max_length=128,
                           required=False,
                           widget=forms.HiddenInput)

    libraryKey = forms.CharField(max_length=128,
                           required=False,
                           initial="TCAG",
                           widget=forms.TextInput(attrs={'size':'60', 'class': 'input-xlarge'}))

    tfKey= forms.CharField(max_length=128,
                                 required=False,
                                 initial="ATCG",
                                 widget=forms.TextInput(attrs={'size':'60', 'class': 'input-xlarge'}))

    tf_config = forms.FileField(required=False,
                                widget=forms.FileInput(attrs={'size':'60', 'class':'input-file'}))
    align_full = forms.BooleanField(required=False, initial=False)
    do_thumbnail = forms.BooleanField(required=False, initial=True, label="Thumbnail only")
    do_base_recal = forms.BooleanField(required=False, label="Enable Base Recalibration")

    realign = forms.BooleanField(required=False)

    aligner_opts_extra = forms.CharField(max_length=100000,
                                         required=False,
                                         widget=forms.Textarea(attrs={'cols': 50, 'rows': 4, 'class': 'input-xlarge'}))
    mark_duplicates = forms.BooleanField(required=False, initial=False)

    previousReport = forms.CharField(required=False,widget=DataSelect(attrs={'class': 'input-xlarge'}
                                                                      ))

    previousThumbReport = forms.CharField(required=False,widget=DataSelect(
        attrs={'class': 'input-xlarge'}))

    project_names = forms.CharField(max_length=1024,
                           required=False,
                           widget=forms.TextInput(attrs={'size':'60','class':'textInput input-xlarge'}))
    
    def clean_report_name(self):
        """
        Verify that the user input doesn't have chars that we don't want
        """
        reportName = self.cleaned_data.get("report_name")
        if reportName[0] == "-":
            logger.error("That Report name can not begin with '-'")
            raise forms.ValidationError(("The Report name can not begin with '-'"))
        if len(reportName) > 128:
            raise forms.ValidationError(("Report Name needs to be less than 128 characters long"))
        if not set(reportName).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.- "):
            logger.error("That Report name has invalid characters. The valid values are letters, numbers, underscore and period.")
            raise forms.ValidationError(("That Report name has invalid characters. The valid values are letters, numbers, underscore and period."))
        else:
            return reportName

    def clean_analysisArgs(self):
        """
        Verify that the user input doesn't have chars that we don't want
        """
        analysisArgs = self.cleaned_data.get("analysisArgs")
        if not set(analysisArgs).issubset(string.printable):
            logger.error("The Analysis command contains non ascii characters.")
            raise forms.ValidationError(("The Analysis command contains non ascii characters."))
        else:
            return analysisArgs

    def clean_basecallerArgs(self):
        """
        Verify that the user input doesn't have chars that we don't want
        """
        basecallerArgs = self.cleaned_data.get("basecallerArgs")
        if not set(basecallerArgs).issubset(string.printable):
            logger.error("The basecaller command contains non ascii characters.")
            raise forms.ValidationError(("The basecaller command contains non ascii characters."))
        else:
            return basecallerArgs


    def clean_beadfindArgs(self):
        """
        Verify that the user input doesn't have chars that we don't want
        """
        beadfindArgs = self.cleaned_data.get("beadfindArgs")
        if not set(beadfindArgs).issubset(string.printable):
            logger.error("The beadfind command contains non ascii characters.")
            raise forms.ValidationError(("The beadfind command contains non ascii characters."))
        else:
            return beadfindArgs


    def clean_libraryKey(self):
        #TODO: note that because this is a hidden advanced field it will not be clear if it fails
        key = self.cleaned_data.get('libraryKey')
        if not set(key).issubset("ATCG"):
            logger.error("This key has invalid characters. The valid values are TACG.")
            raise forms.ValidationError(("This key has invalid characters. The valid values are TACG."))
        else:
            return key

    def clean_tfKey(self):
        key = self.cleaned_data.get('tfKey')
        if not set(key).issubset("ATCG"):
            logger.error("This key has invalid characters. The valid values are TACG.")
            raise forms.ValidationError(("This key has invalid characters. The valid values are TACG."))
        else:
            return key

    def clean_project_names(self):
        """
        Verify that the user input doesn't have chars that we don't want
        """
        projectNames = self.cleaned_data.get("project_names")
        names = []
        for name in projectNames.split(','):
            if name:
              names.append(name)
              if len(name) > 64:
                  raise forms.ValidationError(("Project Name needs to be less than 64 characters long. Please separate different projects with a comma."))
              if not set(name).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.- "):
                  logger.error("Project name has invalid characters. The valid values are letters, numbers, underscore and period.")
                  raise forms.ValidationError(("Project name has invalid characters. The valid values are letters, numbers, underscore and period."))
        return ','.join(names)
            

from iondb.rundb.plan.views_helper import dict_bed_hotspot

class bedfiles_Select(forms.Select):
    # class to allow filtering bedfiles options by reference
    def __init__(self, attrs=None, choices=()):
        super(bedfiles_Select, self).__init__(attrs=attrs, choices=choices)
        bedfiles = dict_bed_hotspot()
        self.bedfiles = bedfiles.get(attrs['name'],[])
    
    def render(self, name, value, attrs=None):
        output = '<select id="%s" class="input-xlarge" name="%s">' % (attrs['id'], name)
        output+= '<option value=""></option>'
        for bedfile in self.bedfiles:
            if value and value == bedfile.file:
                output+='<option selected class="%s" value="%s">%s</option>' % (bedfile.meta['reference'], bedfile.file, bedfile.path)
            else:
                output+='<option class="%s" value="%s">%s</option>' % (bedfile.meta['reference'], bedfile.file, bedfile.path)
        output+='</select>'
        return mark_safe(output)

class AnalysisSettingsForm(forms.ModelForm):

    reference = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}) )
    targetRegionBedFile = forms.ChoiceField(required=False, widget=bedfiles_Select(attrs={'name':'bedFiles'}) )
    hotSpotRegionBedFile = forms.ChoiceField(required=False, widget=bedfiles_Select(attrs={'name':'hotspotFiles'}) )
    plugins = forms.ModelMultipleChoiceField(required=False, widget=Plugins_SelectMultiple(),
        queryset=models.Plugin.objects.filter(selected=True, active=True).order_by('name', '-version'))
    pluginsUserInput = forms.CharField(required=False, widget=forms.HiddenInput())
    barcodeKitName = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}) )
    threePrimeAdapter = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}) )

    def __init__(self, *args, **kwargs):
        super(AnalysisSettingsForm, self).__init__(*args, **kwargs)
        # initialize choices when form instance created
        references = models.ReferenceGenome.objects.filter(index_version=settings.TMAP_VERSION, enabled=True)
        self.fields['reference'].choices= [('none','none')] + [(v[0],"%s (%s)" % (v[0],v[1])) for v in references.values_list('short_name', 'name')]
        bedfiles = dict_bed_hotspot()
        self.fields['targetRegionBedFile'].choices= [('','')] + [(v.file,v.path) for v in bedfiles.get('bedFiles',[])]
        self.fields['hotSpotRegionBedFile'].choices= [('','')] + [(v.file,v.path) for v in bedfiles.get('hotspotFiles',[])]
        self.fields['barcodeKitName'].choices= [('','')]+list(models.dnaBarcode.objects.order_by('name').distinct('name').values_list('name','name'))
        adapters = models.ThreePrimeadapter.objects.all().order_by('-isDefault', 'name')
        self.fields['threePrimeAdapter'].choices= [(v[1],"%s (%s)" % (v[0],v[1])) for v in adapters.values_list('name', 'sequence')]
    
    class Meta:
        model = models.ExperimentAnalysisSettings
        fields = ('reference', 'targetRegionBedFile', 'hotSpotRegionBedFile', 'barcodeKitName', 'threePrimeAdapter')
        
class ExperimentSettingsForm(forms.ModelForm):

    sample = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}) )
    barcodedSamples = forms.CharField(required=False, widget=forms.HiddenInput())
    runtype = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'})) 
    libraryKitname = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    sequencekitname = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))    
    chipBarcode = forms.CharField(required=False, max_length=64, widget=forms.TextInput(attrs={'class':'textInput input-xlarge validateAlphaNumNoSpace'}) )
    libraryKey = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}) )
    notes = forms.CharField(required=False, max_length=128, widget=forms.TextInput(attrs={'class':'textInput input-xlarge'}) )

    def __init__(self, *args, **kwargs):
        super(ExperimentSettingsForm, self).__init__(*args, **kwargs)
        # initialize sample and key choices when form instance created
        self.fields['sample'].choices = [('','')]+ list(models.Sample.objects.filter().values_list('id', 'displayedName'))
        self.fields['libraryKey'].choices = [(v[0],"%s (%s)" % (v[0],v[1])) for v in models.LibraryKey.objects.filter().values_list('sequence', 'description')]
        self.fields['runtype'].choices = [(v[0], "%s (%s)" % (v[1],v[0])) for v in models.RunType.objects.all().order_by("id").values_list('runType','description')] 
        self.fields['libraryKitname'].choices = [('','')] + list(models.KitInfo.objects.filter(kitType='LibraryKit').values_list('name','name'))
        self.fields['sequencekitname'].choices = [('','')]+list(models.KitInfo.objects.filter(kitType='SequencingKit').values_list('name','name')) 
    
    class Meta:
        model = models.Experiment
        fields = ('sample', 'sequencekitname', 'chipBarcode', 'notes')
        

class EditBackup(forms.Form):
    def get_dir_choices():
        basicChoice = [(None, 'None')]
        for choice in devices.to_media(devices.disk_report()):
            basicChoice.append(choice)
        return tuple(basicChoice)
    def get_loc_choices():
        basicChoice = []
        for loc in models.Location.objects.all():
            basicChoice.append((loc,loc))
        return tuple(basicChoice)
    def make_throttle_choices():
        choice = [(0,'Unlimited'), (10000,'10MB/Sec')]
        return tuple(choice)

    archive_directory = forms.ChoiceField(choices=())
    number_to_archive = forms.IntegerField()
    timeout = forms.IntegerField()
    percent_full_before_archive = forms.IntegerField(max_value=99,min_value = 1)
    bandwidth_limit = forms.ChoiceField(make_throttle_choices())
    email = forms.EmailField(required=False)
    enabled = forms.BooleanField(required=False)
    grace_period = forms.IntegerField()

    def __init__(self,*args,**kwargs):
        super(EditBackup,self).__init__(*args,**kwargs)
        def get_dir_choices():
            basicChoice = [(None, 'None')]
            for choice in devices.to_media(devices.disk_report()):
                basicChoice.append(choice)
            return tuple(basicChoice)
        self.fields['archive_directory'].choices = get_dir_choices()

def getLevelChoices():
    #try:
    #    numPG = models.dm_prune_group.objects.count()
    #except:
    #    return [""]
    tupleToBe = []
    #for i in range(1,numPG+1):
    #    tupleToBe.append(("%02d" % (i), '%d'%(i)))
    pgrps = models.dm_prune_group.objects.all()
    for i,pgrp in enumerate(pgrps,start=1):
        tupleToBe.append(("%s" % pgrp.name, '%d' % i))

    return tuple(tupleToBe)

class EditReportBackup(forms.Form):
    def get_dir_choices():
        basicChoice = [(None, 'None')]
        for choice in devices.to_media(devices.disk_report()):
            basicChoice.append(choice)
        return tuple(basicChoice)
    def get_loc_choices():
        basicChoice = []
        for loc in models.Location.objects.all():
            basicChoice.append((loc,loc))
        return tuple(basicChoice)

    location = forms.ChoiceField(choices=(), label='Backup Location')
    days = forms.ChoiceField(required=False, choices=tuple([(10, '10'), (30, '30'), (60, '60'), (90, '90'), (0, '-')]))
    #autoDays = forms.ChoiceField(choices=tuple([(30, '30'), (60, '60'), (90, '90'), (180, '180'), (365, '365')]))
    autoDays = forms.IntegerField(label="Auto-action delay (days)", min_value=0, max_value=9999)
    autoPrune = forms.BooleanField(required=False, label="Enable Auto-action")
    pruneLevel = forms.ChoiceField(choices=getLevelChoices(), widget=forms.RadioSelect())
    #autoAction = forms.ChoiceField(choices=(tuple([('P', 'Prune'), ('A', 'Archive'), ('D', 'Delete')])), label='Act Automatically')
    autoAction = forms.ChoiceField(choices=(tuple([('P', 'Prune'), ('A', 'Archive')])), label='Auto-action')

    def __init__(self,*args,**kwargs):
        super(EditReportBackup,self).__init__(*args,**kwargs)
        def get_dir_choices():
            basicChoice = [(None, 'None')]
            for choice in devices.to_media(devices.disk_report()):
                basicChoice.append(choice)
            return tuple(basicChoice)

        self.fields['pruneLevel'].choices = getLevelChoices()
        self.fields['location'].choices = get_dir_choices()

class bigPruneEdit(forms.Form):

    checkField = forms.MultipleChoiceField(choices=tuple([('x','y')]), widget=forms.CheckboxSelectMultiple())
    newField = forms.CharField(widget=forms.TextInput(attrs = {'class':'textInput validateFilenameRegex'}))
    remField = forms.MultipleChoiceField(choices=tuple([('x', 'y')]),widget=forms.CheckboxSelectMultiple())

    def __init__(self,*args,**kwargs):
        target = kwargs.pop("pk")
        super(bigPruneEdit,self).__init__(*args,**kwargs)

        def get_selections(pkTarget):
            ruleList = models.dm_prune_field.objects.all().order_by('pk')
            choices = []
            for rule in ruleList:
                choices.append(('%s:'%target+'%s'%rule.pk, rule.rule))
            logger.error(choices)
            return tuple(choices)

        def get_removeIDs():
            ruleList = models.dm_prune_field.objects.all().order_by('pk')
            choices = []
            for rule in ruleList:
                choices.append(('%s'%rule.pk, rule.rule))
            return tuple(choices)

        self.fields['checkField'].choices = get_selections(target)
        self.fields['remField'].choices = get_removeIDs()

class EditPruneLevels(forms.Form):
    def get_dir_choices():
        basicChoice = [(None, 'None')]
        for choice in devices.to_media(devices.disk_report()):
            basicChoice.append(choice)
        return tuple(basicChoice)

    def getLevelChoices():
        #try:
        #    numPG = models.dm_prune_group.objects.count()
        #except:
        #    return [""]
        tupleToBe = []
        #for i in range(1,numPG+1):
        #    tupleToBe.append(("%02d" % (i), '%d'%(i)))
        pgrps = models.dm_prune_group.objects.all()
        for i,pgrp in enumerate(pgrps,start=1):
            tupleToBe.append(("%s" % pgrp.name, '%d' % i))
        tupleToBe.append(('add', 'Add...'))
        tupleToBe.append(('rem', 'Remove...'))
        return tuple(tupleToBe)

    location = forms.ChoiceField(choices=())
    autoPrune = forms.BooleanField(required=False)
    pruneLevel = forms.MultipleChoiceField(choices=getLevelChoices(), widget=forms.CheckboxSelectMultiple())
    editChoice = forms.MultipleChoiceField(choices=getLevelChoices(), widget=forms.RadioSelect())
    name = forms.SlugField(label="Group Name", widget=forms.TextInput(attrs = {'class':'textInput required validatePhrase'}))

    def __init__(self,*args,**kwargs):
        super(EditPruneLevels,self).__init__(*args,**kwargs)
        def get_dir_choices():
            basicChoice = [(None, 'None')]
            for choice in devices.to_media(devices.disk_report()):
                basicChoice.append(choice)
            return tuple(basicChoice)
        def getLevelChoices():
            numPG = models.dm_prune_group.objects.count()
            tupleToBe = []
            for i in range(1,numPG+1):
                if i < 10:
                    tupleToBe.append(('%s'%i, '%s'%('%s: '%i)))
                else:
                    tupleToBe.append((i, '%s'%('%s'%i)))
            tupleToBe.append(('add', 'Add...'))
            tupleToBe.append(('rem', 'Remove...'))
            return tuple(tupleToBe)
        #self.fields['pruneLevel'].choices = getLevelChoices()
        self.fields['editChoice'].choices = getLevelChoices()
        self.fields['location'].choices = get_dir_choices()


class EmailAddress(forms.ModelForm):
    "Made to have full symetry with the EmailAddress model fields"
    class Meta:
        model = models.EmailAddress


class EditReferenceGenome(forms.Form):
    name = forms.CharField(max_length=512,required=True)
    NCBI_name = forms.CharField(max_length=512,required=True)
    read_sample_size = forms.CharField(max_length=512,required=False)
    notes = forms.CharField(max_length=1048,required=False,widget=forms.Textarea(attrs={'cols': 50, 'rows': 4}))
    enabled = forms.BooleanField(required=False)
    genome_key = forms.IntegerField(widget=forms.HiddenInput(), required=True)
    index_version = forms.CharField(widget=forms.HiddenInput(), required=True)

    def clean_name(self):
        """don't allow duplicate names
        make an exception for the genome we are working with
        """

        index_version = self.data.getlist("index_version")[0]
        genome_key = self.data.getlist("genome_key")[0]
        sample_size = self.data.getlist("read_sample_size")[0]
        get_name = self.cleaned_data.get('name')
        genomes = models.ReferenceGenome.objects.filter(short_name=get_name).exclude(pk=genome_key)

        if not set(get_name).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"):
            raise forms.ValidationError(("The short name has invalid characters. The valid values are letters, numbers, and underscores."))

        if not sample_size.isdigit():
            raise forms.ValidationError(("The read sample size must be a positive number."))

        for g in genomes:
            if get_name == g.name and g.index_version == index_version :
                error_str = "A reference with the name" + get_name + " and index version" + index_version + " already exists"
                raise forms.ValidationError(error_str)
        return get_name


class UserProfileForm(forms.ModelForm):
    # validate name length against the longest possible name
    #  Source: http://www.independent.co.uk/news/uk/this-britain/captain-fantastic-claims-worlds-longest-name-993957.html
    # Captain Fantastic Faster Than Superman Spiderman Batman Wolverine Hulk And The Flash Combined
    name = forms.CharField(max_length=93)
    email = forms.EmailField()

    def __init__(self, *args, **kw):
        super(UserProfileForm, self).__init__(*args, **kw)
        self.fields['email'].initial = self.instance.user.email

        self.fields.keyOrder = ['name', 'email', 'phone_number']

    def save(self, *args, **kw):
        super(UserProfileForm, self).save(*args, **kw)
        self.instance.user.email = self.cleaned_data.get('email')
        self.instance.user.save()

    class Meta:
        model = models.UserProfile


class NetworkConfigForm(forms.Form):
    modes = (("dhcp","DHCP",), ("static", "Static"))
    mode = forms.ChoiceField(widget=forms.widgets.RadioSelect, choices=modes)
    address = forms.IPAddressField(label="IP Address", required=False)
    subnet = forms.IPAddressField(required=False, label="Subnet")
    gateway = forms.IPAddressField(required=False)
    nameservers = forms.CharField(required=False, max_length=256)
    proxy_address = forms.CharField(required=False, max_length=256)
    proxy_port = forms.CharField(required=False)
    proxy_username = forms.CharField(required=False)
    proxy_password = forms.CharField(required=False)
    collab_ip = forms.IPAddressField(required=False)

    def get_network_settings(self):
        """Usage: ./TSquery [options]
             --all, -a           Output all information (default)
             --proxy_info        Output just proxy http info
             --eth_info          Output just ethernet info
             --eth-dev           Specify eth device to query (default = eth0)
             --json              Output json format
             --outputfile, -o    Write output to file
        Outputs:
        http_proxy:not set
        network_device:eth0
        network_mode:dhcp
        network_address:10.0.2.15
        network_subnet:255.255.255.0
        network_gateway:10.0.2.2
        """
        settings = {
                "mode":"",
                "address":"",
                "subnet":"",
                "gateway":"",
                "nameservers":"",
                "proxy_address": "",
                "proxy_port": "",
                "proxy_username": "",
                "proxy_password": "",
                "collab_ip": "",
                }
        cmd = ["/usr/sbin/TSquery"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            out_lines = stdout.split("\n")
            formatted_output = "\n".join("    %s" % l for l in out_lines)
            logger.info("TSquery output:\n%s" % formatted_output)
            for line in out_lines:
                if line:
                    key, value = line.split(":", 1)
                    if key.startswith("network_"):
                        key = key.replace("network_", '', 1)
                    settings[key] = value
        except Exception as error:
            logger.error("When attempting to run TSquery:\n%s" % error)
        return settings

    def new_config(self, one, two, keys):
        return not all(k in one and k in two and one[k] == two[k] for k in keys)

    def set_to_current_values(self):
        settings = self.get_network_settings()
        self.fields['mode'].initial = settings["mode"]
        self.fields['address'].initial = settings["address"]
        self.fields['subnet'].initial = settings["subnet"]
        self.fields['gateway'].initial = settings["gateway"]
        self.fields['nameservers'].initial = settings["nameservers"]
        self.fields['proxy_address'].initial = settings["proxy_address"]
        self.fields['proxy_port'].initial = settings["proxy_port"]
        self.fields['proxy_username'].initial = settings["proxy_username"]
        self.fields['proxy_password'].initial = settings["proxy_password"]
        self.fields['collab_ip'].initial = settings["collab_ip"]

    def __init__(self, *args, **kw):
        super(NetworkConfigForm, self).__init__(*args, **kw)
        self.set_to_current_values()

    def save(self, *args, **kw):
        host_task, proxy_task, dns_task = None, None, None
        settings = self.get_network_settings()
        host_config = ["mode", "address", "subnet", "gateway"]
        if self.new_config(self.cleaned_data, settings, host_config):
            logger.info("User changed the host network settings.")
            if self.cleaned_data['mode'] == "dhcp":
                host_task = tasks.dhcp.delay()
            elif self.cleaned_data['mode'] == "static":
                address = self.cleaned_data['address']
                subnet = self.cleaned_data['subnet']
                gateway = self.cleaned_data['gateway']
                host_task = tasks.static_ip.delay(address, subnet, gateway)
        proxy_config = ["proxy_address", "proxy_port", "proxy_username", "proxy_password"]
        if self.new_config(self.cleaned_data, settings, proxy_config):
            logger.info("User changed the proxy settings.")
            if self.cleaned_data['proxy_address'] and self.cleaned_data['proxy_port']:
                address = self.cleaned_data['proxy_address']
                port = self.cleaned_data['proxy_port']
                user = self.cleaned_data['proxy_username']
                password = self.cleaned_data['proxy_password']
                proxy_task = tasks.proxyconf.delay(address, port, user, password)
            else:
                proxy_task = tasks.ax_proxy.delay()
        if self.new_config(self.cleaned_data, settings, ["nameservers"]):
            logger.info("User changed the DNS settings.")
            if self.cleaned_data['nameservers']:
                dns_task = tasks.dnsconf.delay(self.cleaned_data['nameservers'])
        if host_task:
            host_task.get()
        if proxy_task:
            proxy_task.get()
        if dns_task:
            dns_task.get()
        self.set_to_current_values()


class AmpliseqLogin(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=PasswordInput)
