# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.utils import toBoolean
import json
import uuid
from django.core.exceptions import ValidationError
from traceback import format_exc
from iondb.rundb.plan.page_plan.step_helper import StepHelperType
from django.contrib.auth.models import User
from iondb.rundb.models import PlannedExperiment, PlannedExperimentQC, QCType,\
    Project, Plugin, RunType, ApplicationGroup

from iondb.rundb.plan.page_plan.step_names import StepNames
from django.db import transaction
from iondb.rundb.plan.page_plan.plugins_step_data import PluginFieldNames
from iondb.rundb.plan.page_plan.output_step_data import OutputFieldNames
from iondb.rundb.plan.page_plan.save_plan_by_sample_step_data import SavePlanBySampleFieldNames
from iondb.rundb.plan.page_plan.barcode_by_sample_step_data import BarcodeBySampleFieldNames

import logging
logger = logging.getLogger(__name__)

class StepHelperDbSaver():

    def getProjectObjectList(self, step_helper, username):
        retval = []
        if step_helper.steps[StepNames.OUTPUT].savedFields[OutputFieldNames.PROJECTS]:
            for project_id in step_helper.steps[StepNames.OUTPUT].savedFields[OutputFieldNames.PROJECTS]:
                retval.append(Project.objects.get(id=int(project_id)))
        
        newProjectNames = step_helper.steps[StepNames.OUTPUT].savedFields[OutputFieldNames.NEW_PROJECTS]
        if newProjectNames:
            newProjectNames = newProjectNames.split(',')
            retval.extend(Project.bulk_get_or_create(newProjectNames, User.objects.get(username=username)))
        return retval

    def __get_universal_params(self, step_helper, username):
        if not step_helper.isPlan():
            if step_helper.isTemplateBySample():
                save_template_step_data = step_helper.steps[StepNames.SAVE_TEMPLATE_BY_SAMPLE]
            else:
                save_template_step_data = step_helper.steps[StepNames.SAVE_TEMPLATE]
                
        
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        kits_step_data = step_helper.steps[StepNames.KITS]
        reference_step_data = step_helper.steps[StepNames.REFERENCE]
        plugins_step_data = step_helper.steps[StepNames.PLUGINS]
        ionreporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        
        if not step_helper.isPlan():
            isFavorite = save_template_step_data.savedFields['setAsFavorite']
        else:
            isFavorite = False
        
        runType = ''
        application_group = None
        sampleGrouping = None
        if application_step_data.savedObjects['runType']:
            runType = application_step_data.savedObjects['runType'].runType
            application_group = application_step_data.savedObjects['runType'].applicationGroups.all()[0:1][0]
        
        if application_step_data.savedFields['applicationGroup']:
            application_group = ApplicationGroup.objects.get(pk=application_step_data.savedFields['applicationGroup'])

        if ionreporter_step_data.savedFields['sampleGrouping']:
            sampleGrouping = ionreporter_step_data.savedFields['sampleGrouping']

        templatingKitName = kits_step_data.savedFields['templatekitname']
        controlSequencekitname = kits_step_data.savedFields['controlsequence']
        samplePrepKitName = kits_step_data.savedFields['samplePreparationKit']
        x_barcodeId = kits_step_data.savedFields['barcodeId']
        x_chipType = kits_step_data.savedFields['chipType']
        if not x_chipType:
            x_chipType = ''
        
        x_flows = kits_step_data.savedFields['flows']
        x_forward3primeadapter = kits_step_data.savedFields['forward3primeAdapter']
        x_libraryKey = kits_step_data.savedFields['libraryKey']
        x_librarykitname = kits_step_data.savedFields['librarykitname']
        x_sequencekitname = kits_step_data.savedFields['sequencekitname']
        x_isDuplicateReads = kits_step_data.savedFields['isDuplicateReads']
        
        x_bedfile = reference_step_data.savedFields['targetBedFile']
        x_library = reference_step_data.savedFields["reference"]
        x_regionfile = reference_step_data.savedFields['hotSpotBedFile']
        
        selectedPluginsValue = plugins_step_data.getSelectedPluginsValue()
        
        planStatus = "planned"
        if not step_helper.isEditRun():
            if step_helper.isIonChef():
                if step_helper.isPlan() or step_helper.isPlanBySample():
                    planStatus = "pending"
        else:
            planStatus = "run"
            
        retval = {
            'applicationGroup': application_group,
            'sampleGrouping' : sampleGrouping, 
            'usePreBeadfind': True,
            'usePostBeadfind': True,
            'preAnalysis': True,
            'runType': runType,
            'templatingKitName': templatingKitName,
            'controlSequencekitname': controlSequencekitname,
            'runMode': 'single',
            'isSystem': False,
            'isPlanGroup': False,
            'username': username,
            'isFavorite': toBoolean(isFavorite, False),
            'pairedEndLibraryAdapterName': '',
            'samplePrepKitName': samplePrepKitName,
            'planStatus' : planStatus,

            'x_autoAnalyze': True,
            'x_barcodeId': x_barcodeId,
            'x_bedfile': x_bedfile,
            'x_chipType': x_chipType,
            'x_flows': x_flows,
            'x_forward3primeadapter': x_forward3primeadapter,
            'x_library': x_library,
            'x_libraryKey': x_libraryKey,
            'x_librarykitname': x_librarykitname,
            'x_regionfile': x_regionfile,
            'x_selectedPlugins': selectedPluginsValue,
            'x_sequencekitname': x_sequencekitname,
            'x_variantfrequency': '',
            'x_isDuplicateReads': False if x_isDuplicateReads is None else x_isDuplicateReads
        }
        return retval

    def __get_specific_params_by_sample(self, step_helper, index=0, sample_set_item_display_Name=None, sample_external_id='', sample_description='', sampleSet_uid=None, planTotal=1):        
        save_step = step_helper.steps[StepNames.SAVE_PLAN_BY_SAMPLE]
        barcoding_step = step_helper.steps[StepNames.BARCODE_BY_SAMPLE]
        sampleset = save_step.savedObjects[SavePlanBySampleFieldNames.SAMPLESET]
        plugins_step_data = step_helper.steps[StepNames.PLUGINS]
        ionreporter_step_data = step_helper.steps[StepNames.IONREPORTER]

        isReusable = False
        barcodedSamples = None

        planDisplayedName = save_step.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME]

        if step_helper.isBarcoded():
            if step_helper.previous_plan_id == -1:
                planDisplayedName = '%s %s' % (planDisplayedName, sampleset.displayedName)
        else:
            planDisplayedName = '%s %s' % (planDisplayedName, sample_set_item_display_Name)

        selectedPluginsValue = plugins_step_data.getSelectedPluginsValue()
        
        if step_helper.isBarcoded():
            barcodedSamples = json.dumps(\
                step_helper.steps[StepNames.BARCODE_BY_SAMPLE].savedObjects[BarcodeBySampleFieldNames.SAMPLE_TO_BARCODE]\
                )
            
        retval = {'planDisplayedName': planDisplayedName,
                  'planName': planDisplayedName.replace(' ', '_'),
                  'isReusable': isReusable,
                  'sampleSet': sampleset,
                  'sampleSet_planIndex' : index,
                  'sampleSet_uid' : sampleSet_uid,
                  'x_barcodedSamples': barcodedSamples,
                  # 'x_numberOfChips' : barcoding_step.savedFields[BarcodeBySampleFieldNames.NUMBER_OF_CHIPS],
                  'x_selectedPlugins': selectedPluginsValue,
                  'x_isSaveBySample' : True
                }
        if ionreporter_step_data.savedFields['irworkflow'] != None:
            retval.update({'irworkflow' : ionreporter_step_data.savedFields['irworkflow']})
        if not step_helper.isBarcoded():
            retval.update({
                    'sampleSet_planTotal' : planTotal,
                    'x_sample_external_id': sample_external_id,
                    'x_sample_description': sample_description,
                    'x_sampleDisplayedName': sample_set_item_display_Name,
                    })
        else:
            retval.update({
                    'sampleSet_planTotal' : planTotal,
                    })

        return retval

    def __get_specific_params(self, step_helper, username, sample_name='', tube_label='', 
                              sample_external_id='', sample_description='',
                              is_multi_sample=False):
        ionreporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        if step_helper.isTemplateBySample():
            planDisplayedName = step_helper.steps[StepNames.SAVE_TEMPLATE_BY_SAMPLE].savedFields['templateName']
        else:
            planDisplayedName = step_helper.steps[StepNames.SAVE_TEMPLATE].savedFields['templateName']
        sampleTubeLabel = ''
        isReusable = True
        note = ''
        sample = ''
        sample_display_name = ''
        barcodedSamples = None
        if step_helper.isPlan():
            planDisplayedName = step_helper.steps[StepNames.SAVE_PLAN].savedFields['planName']
            if is_multi_sample:
                planDisplayedName += '_' + sample_name.strip()
            sampleTubeLabel = tube_label
            isReusable = False
            note = step_helper.steps[StepNames.SAVE_PLAN].savedFields['note']
            sample = sample_name.strip().replace(' ', '_')
            sample_display_name = sample_name.strip()
            if step_helper.isBarcoded():
                barcodedSamples = json.dumps(step_helper.steps[StepNames.SAVE_PLAN].savedObjects['sampleToBarcode'])
                sampleTubeLabel = step_helper.steps[StepNames.SAVE_PLAN].savedFields['barcodeSampleTubeLabel']
        retval = {'planDisplayedName': planDisplayedName,
                  'planName': planDisplayedName.replace(' ', '_'),
                  'sampleTubeLabel' : sampleTubeLabel,
                  'isReusable': isReusable,
                  'x_notes': note,
                  'x_sample': sample,
                  'x_sample_external_id': sample_external_id,
                  'x_sample_description': sample_description,
                  'x_sampleDisplayedName': sample_display_name,
                  'x_barcodedSamples': barcodedSamples
                  }
        if ionreporter_step_data.savedFields['irworkflow'] != None:
            retval.update({'irworkflow' : ionreporter_step_data.savedFields['irworkflow']})
        return retval
        
    
    def __update_non_barcode_plugins_with_ir(self, step_helper, parentDict, sampleValueDict):
        save_plan_step_data = step_helper.steps[StepNames.SAVE_PLAN]
        # if save_plan_step_data.prepopulatedFields['selectedIr']:
        if step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId'] and step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId'] != '0':
            # ir_plugin = save_plan_step_data.prepopulatedFields['selectedIr']
            ir_plugin = step_helper.steps[StepNames.IONREPORTER].prepopulatedFields['IR_PLUGIN']
            
            user_input_dict = {}
            user_input_dict['Workflow'] = sampleValueDict['irWorkflow']
            user_input_dict['Gender'] = sampleValueDict['irGender']
            user_input_dict['sample'] = sampleValueDict['sampleName']
            user_input_dict['sampleName'] = sampleValueDict['sampleName'].strip().replace(' ', '_')
            user_input_dict['sampleExternalId'] = sampleValueDict["sampleExternalId"]
            user_input_dict['sampleDescription'] = sampleValueDict["sampleDescription"]
            user_input_dict['Relation'] = sampleValueDict['irRelation'] or ""
            user_input_dict['RelationRole'] = sampleValueDict['irRelationRole'] or ""

            if 'irSetID' in sampleValueDict:
                if step_helper.isEdit() or step_helper.isCopy():
                    try:
                       user_input_dict['setid'] = sampleValueDict['irSetID'] + step_helper.steps[StepNames.SAVE_PLAN].prepopulatedFields['setid_suffix']
                    except:
                        user_input_dict['setid'] = sampleValueDict['irSetID'] + step_helper.steps[StepNames.BARCODE_BY_SAMPLE].prepopulatedFields['setid_suffix']
                else:
                    user_input_dict['setid'] = str(sampleValueDict['irSetID']) + '__' + str(uuid.uuid4())

            
            # parentDict['x_selectedPlugins'][ir_plugin.name] = self.__get_ir_plugins_entry(ir_plugin, [user_input_dict])
            accountId = step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId']
            accountName = step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountName']
            applicationType = step_helper.steps[StepNames.IONREPORTER].savedFields['applicationType']
            if not applicationType:
                try:
                    applicationType = step_helper.steps[StepNames.SAVE_PLAN].savedFields['applicationType']
                except:
                    applicationType = step_helper.steps[StepNames.BARCODE_BY_SAMPLE].savedFields['applicationType']
            parentDict['x_selectedPlugins']['IonReporterUploader'] = self.__get_ir_plugins_entry(ir_plugin, [user_input_dict],
                                                                                                 accountId, accountName, applicationType)
        
    def __update_barcode_plugins_with_ir(self, step_helper, parentDict, userInputList, suffix=None):
        # save_plan_step_data = step_helper.steps[StepNames.SAVE_PLAN]
        # if save_plan_step_data.prepopulatedFields['selectedIr']:
        if step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId'] and step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId'] != '0':
            # ir_plugin = save_plan_step_data.prepopulatedFields['selectedIr']
            ir_plugin = step_helper.steps[StepNames.IONREPORTER].prepopulatedFields['IR_PLUGIN']
            # parentDict['x_selectedPlugins'][ir_plugin.name] = self.__get_ir_plugins_entry(ir_plugin, userInputList)
            accountId = step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId']
            accountName = step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountName']
            applicationType = step_helper.steps[StepNames.IONREPORTER].savedFields['applicationType']
            if not applicationType:
                try:
                    applicationType = step_helper.steps[StepNames.SAVE_PLAN].savedFields['applicationType']
                except:
                    applicationType = step_helper.steps[StepNames.BARCODE_BY_SAMPLE].savedFields['applicationType']

            for item in userInputList:
                if 'setid' in item:
                    if step_helper.isEdit() or step_helper.isCopy():
                        try:
                            item['setid'] = item['setid'] + step_helper.steps[StepNames.SAVE_PLAN].prepopulatedFields['setid_suffix']
                        except:
                            item['setid'] = item['setid'] + step_helper.steps[StepNames.BARCODE_BY_SAMPLE].prepopulatedFields['setid_suffix']
                    else:
                        item['setid'] = item['setid'] + '__' + suffix
            parentDict['x_selectedPlugins']['IonReporterUploader'] = self.__get_ir_plugins_entry(ir_plugin, userInputList,
                                                                                                 accountId, accountName, applicationType)
            
    def __get_ir_plugins_entry(self, ir_plugin, userInputList, accountId, accountName, applicationType):
        # version = 1.0 if ir_plugin.name == 'IonReporterUploader_V1_0' else ir_plugin.version
        version = ir_plugin.version
        ir_plugin_dict = {
                PluginFieldNames.PL_ID: ir_plugin.id,
                PluginFieldNames.NAME: ir_plugin.name,
                PluginFieldNames.VERSION: version,
                PluginFieldNames.FEATURES: [PluginFieldNames.EXPORT],
                }

        for userInput in userInputList:
            userInput['ApplicationType'] = applicationType

        user_input = {PluginFieldNames.ACCOUNT_ID: accountId,
                      PluginFieldNames.ACCOUNT_NAME: accountName,
                      "userInputInfo": userInputList}
        
        ir_plugin_dict[PluginFieldNames.USER_INPUT] = user_input
        return ir_plugin_dict

    @transaction.commit_manually
    def save(self, step_helper, username):
        try:
            if step_helper.isPlanBySample() or step_helper.isTemplateBySample():
                planTemplate = self.__innser_save_by_sample(step_helper, username)
            else:
                planTemplate = self.__innser_save(step_helper, username)
            transaction.commit()

            return planTemplate
        except ValidationError, err:
            transaction.rollback()
            logger.exception(format_exc())
            message = "Internal error while trying to save the plan. "
            for msg in err.messages:                
                message += str(msg)
                message += " "
            raise ValueError(message)
        except Exception as excp:
            transaction.rollback()
            logger.exception(format_exc())
            message = "Internal error while trying to save the plan. %s" %(excp.message)
            raise ValueError(message)
        except:
            transaction.rollback()
            logger.exception(format_exc())
            raise ValueError("A completely unexpected error has occurred while trying to save plan.")


            
    def __innser_save_by_sample(self, step_helper, username):        
        projectObjList = self.getProjectObjectList(step_helper, username)
        sampleSet_uid = str(uuid.uuid4())
        if step_helper.isPlanBySample():
            
            if step_helper.isBarcoded():
                kwargs = self.__get_universal_params(step_helper, username)
                kwargs.update(self.__get_specific_params_by_sample(step_helper, sampleSet_uid=sampleSet_uid))
                suffix = str(uuid.uuid4())
                self.__update_barcode_plugins_with_ir(step_helper, kwargs, step_helper.steps[StepNames.BARCODE_BY_SAMPLE].savedObjects['barcodedIrPluginEntires'], suffix=suffix)

                if step_helper.sh_type == StepHelperType.EDIT_PLAN_BY_SAMPLE and step_helper.previous_plan_id > 0:
                    self.savePlannedExperiment(step_helper, kwargs, projectObjList, step_helper.previous_plan_id)
                elif step_helper.sh_type == StepHelperType.CREATE_NEW_TEMPLATE_BY_SAMPLE:
                    self.saveTemplate(step_helper, username, projectObjList)
                else:
                    self.savePlannedExperiment(step_helper, kwargs, projectObjList)
            else:
                sampleDicts = []
                for chip, values in step_helper.steps[StepNames.BARCODE_BY_SAMPLE].savedObjects['chipToSample'].items():
                    sampleName = values['sampleName']
                    if sampleName:
                        sampleDicts.append(values)
                
                isMultiSample = len(sampleDicts) > 1
                
                firstIteration = True
                index = 0
                suffix = str(uuid.uuid4())
                for valueDict in sampleDicts:
                    kwargs = self.__get_universal_params(step_helper, username)
                    kwargs.update(self.__get_specific_params_by_sample(step_helper, index=index, \
                            sample_set_item_display_Name=valueDict['sampleName'], sample_external_id=valueDict['sampleExternalId'], sample_description=valueDict['sampleDescription'], \
                            sampleSet_uid=sampleSet_uid, planTotal=len(sampleDicts)))
                    self.__update_barcode_plugins_with_ir(step_helper, kwargs, [step_helper.steps[StepNames.BARCODE_BY_SAMPLE].savedObjects['barcodedIrPluginEntires'][index]], suffix=suffix)
                    index += 1
                    #self.__update_non_barcode_plugins_with_ir(step_helper, kwargs, valueDict)
                    
                    if step_helper.sh_type == StepHelperType.EDIT_PLAN_BY_SAMPLE and step_helper.previous_plan_id > 0 and firstIteration:
                        self.savePlannedExperiment(step_helper, kwargs, projectObjList, step_helper.previous_plan_id)
                    else:
                        self.savePlannedExperiment(step_helper, kwargs, projectObjList)
                    firstIteration = False 
        else:
            return self.saveTemplate(step_helper, username, projectObjList)

    def __innser_save(self, step_helper, username):
        projectObjList = self.getProjectObjectList(step_helper, username)
        if step_helper.isPlan():
            save_plan_step_data = step_helper.steps[StepNames.SAVE_PLAN]
            
            if step_helper.isBarcoded():
                kwargs = self.__get_universal_params(step_helper, username)
                kwargs.update(self.__get_specific_params(step_helper, username, '', '', False))
                self.__update_barcode_plugins_with_ir(step_helper, kwargs, save_plan_step_data.savedObjects['barcodedIrPluginEntires'], suffix=str(uuid.uuid4()))
                if step_helper.sh_type in [StepHelperType.EDIT_PLAN, StepHelperType.EDIT_RUN] and step_helper.previous_plan_id > 0:
                    self.savePlannedExperiment(step_helper, kwargs, projectObjList, step_helper.previous_plan_id)
                else:
                    self.savePlannedExperiment(step_helper, kwargs, projectObjList)
            else:
                sampleDicts = []
                for chip, values in save_plan_step_data.savedObjects['chipToSamples'].items():
                    sampleName = values['sampleName']
                    if sampleName:
                        sampleDicts.append(values)
                
                isMultiSample = len(sampleDicts) > 1
                
                firstIteration = True
                for valueDict in sampleDicts:
                    kwargs = self.__get_universal_params(step_helper, username)
                    kwargs.update(self.__get_specific_params(step_helper, username, valueDict['sampleName'], valueDict['tubeLabel'], 
                                                             valueDict['sampleExternalId'], valueDict['sampleDescription'], isMultiSample))
                    self.__update_non_barcode_plugins_with_ir(step_helper, kwargs, valueDict)
                    
                    if step_helper.sh_type in [StepHelperType.EDIT_PLAN, StepHelperType.EDIT_RUN] and step_helper.previous_plan_id > 0 and firstIteration:
                        self.savePlannedExperiment(step_helper, kwargs, projectObjList, step_helper.previous_plan_id)
                    else:
                        self.savePlannedExperiment(step_helper, kwargs, projectObjList)
                    firstIteration = False
        else:
            return self.saveTemplate(step_helper, username, projectObjList)

    def saveTemplate(self, step_helper, username, projectObjList):
        #logger.debug("step_helper_db_saver.saveTemplate() isCopy=%s; isCreate=%s; isEdit=%s; isEditRun=%s" %(str(step_helper.isCopy()), str(step_helper.isCreate()), str(step_helper.isEdit()), str(step_helper.isEditRun())))
        
        kwargs = self.__get_universal_params(step_helper, username)
        kwargs.update(self.__get_specific_params(step_helper, username))
        # ir_step_data = step_helper.steps[StepNames.IONREPORTER]
        
        # ir_plugin = None
        # if ir_step_data.savedFields['uploaders']:
        #     ir_qs = Plugin.objects.filter(pk__in=ir_step_data.savedFields['uploaders'], name__icontains='IonReporter')
        #     irExists = ir_qs.count() > 0
        #     if irExists:
        #         ir_plugin = ir_qs[0:1][0]
        #         kwargs['x_selectedPlugins'][ir_plugin.name] = self.__get_ir_plugins_entry(ir_plugin, '')

        if step_helper.steps[StepNames.IONREPORTER].savedFields['irOptions'] != '0':
            accountId = step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId']
            accountName = step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountName']
            applicationType = step_helper.steps[StepNames.IONREPORTER].savedFields['applicationType']
            kwargs['x_selectedPlugins']['IonReporterUploader'] = self.__get_ir_plugins_entry(step_helper.steps[StepNames.IONREPORTER].prepopulatedFields['IR_PLUGIN'],
                                                                                             '', accountId, accountName, applicationType)

        if step_helper.sh_type == StepHelperType.EDIT_TEMPLATE and step_helper.previous_template_id > 0:
            return self.savePlannedExperiment(step_helper, kwargs, projectObjList, step_helper.previous_template_id)
        else:
            return self.savePlannedExperiment(step_helper, kwargs, projectObjList)

    def savePlannedExperiment(self, step_helper, param_dict, projectObjList, pe_id_to_update=None):
        planTemplate = None
        #if we're changing a plan from having 1 sample to say 2 samples, we need to UPDATE 1 plan and CREATE 1 plan!!
        logger.debug("About to save, KWARGS ARE: %s" % str(param_dict))
        for key,value in sorted(param_dict.items()):
            logger.debug('KWARG %s: %s' % (str(key), str(value)))
        if pe_id_to_update:
            planTemplate, extra_kwargs = PlannedExperiment.objects.save_plan(pe_id_to_update, **param_dict)
        else:
            planTemplate, extra_kwargs = PlannedExperiment.objects.save_plan(-1, **param_dict)
        
        self.saveQc(planTemplate, step_helper.steps[StepNames.MONITORING])
        self.saveProjects(planTemplate, projectObjList)
        return planTemplate
            

    def saveQc(self, planTemplate, monitoring_step):
        # Update QCtype thresholds
        qcTypes = QCType.objects.all()
        for qcType in qcTypes:
            qc_threshold = monitoring_step.savedFields[qcType.qcName]
            if qc_threshold:
                # get existing PlannedExperimentQC if any
                plannedExpQcs = PlannedExperimentQC.objects.filter(plannedExperiment=planTemplate.id, qcType=qcType.id)
                if len(plannedExpQcs) > 0:
                    for plannedExpQc in plannedExpQcs:
                        plannedExpQc.threshold = qc_threshold
                        plannedExpQc.save()
                else:
                    kwargs = {
                        'plannedExperiment': planTemplate,
                        'qcType': qcType,
                        'threshold': qc_threshold
                    }
                    plannedExpQc = PlannedExperimentQC(**kwargs)
                    plannedExpQc.save()

    def saveProjects(self, planTemplate, projectObjList):
        # add/remove projects
        if projectObjList:
            #TODO: refactor this logic to simplify using django orm
            projectNameList = [project.name for project in projectObjList]
            for currentProject in planTemplate.projects.all():
                if currentProject.name not in projectNameList:
                    planTemplate.projects.remove(currentProject)
            for projectObj in projectObjList:
                planTemplate.projects.add(projectObj)
        else:
            planTemplate.projects.clear()