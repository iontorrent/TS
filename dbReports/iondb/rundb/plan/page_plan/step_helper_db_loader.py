# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import types
from iondb.rundb.models import PlannedExperiment, PlannedExperimentQC,\
    RunType, dnaBarcode, Plugin, ApplProduct, SampleSet, ThreePrimeadapter, Chip, KitInfo

from iondb.rundb.plan.page_plan.step_helper import StepHelper, StepHelperType
from iondb.rundb.plan.views_helper import getPlanDisplayedName, getPlanBarcodeCount

import json
from iondb.rundb.json_field import JSONDict, JSONEncoder
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.kits_step_data import KitsFieldNames
from iondb.rundb.plan.page_plan.reference_step_data import ReferenceFieldNames
from iondb.rundb.plan.page_plan.plugins_step_data import PluginFieldNames
from iondb.rundb.plan.page_plan.output_step_data import OutputFieldNames 
from iondb.rundb.plan.page_plan.barcode_by_sample_step_data import BarcodeBySampleFieldNames
from iondb.rundb.plan.page_plan.save_plan_by_sample_step_data import SavePlanBySampleFieldNames
from iondb.rundb.plan.page_plan.save_plan_step_data import SavePlanFieldNames


import logging
logger = logging.getLogger(__name__)

class StepHelperDbLoader():
    
    def getStepHelperForRunType(self, run_type_id, step_helper_type=StepHelperType.CREATE_NEW_TEMPLATE):
        '''
            Creates a step helper for the specified runtype, this can be a plan or a template step helper.
        '''
        #logger.debug("ENTER step_helper_db_loader.getStepHelperForRunType() run_type_id=%s" %(str(run_type_id)))
        
        step_helper = StepHelper(sh_type=step_helper_type)
        ionReporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        
        runType = RunType.objects.get(pk=run_type_id)
        step_helper.parentName = runType.description
        
        self._updateApplicationStepData(runType, step_helper, application_step_data)
        
        kits_step_data = step_helper.steps[StepNames.KITS]
        self._updateKitsStepData(runType, step_helper, kits_step_data)
        
        return step_helper


    def _updateApplicationStepData(self, runTypeObj, step_helper, application_step_data):
        application_step_data.savedFields['runType'] = runTypeObj.pk
        application_step_data.savedFields['applicationGroup'] = runTypeObj.applicationGroups.all()[0:1][0].pk
        application_step_data.savedFields['applicationGroupName'] = runTypeObj.applicationGroups.all()[0:1][0].name

        ##application_step_data.savedObjects["runType"] = runTypeObj
        ##application_step_data.savedObjects["applProduct"] = ApplProduct.objects.get(isActive=True, isDefault=True, isVisible=True,
        ##                                                               applType__runType = runTypeObj.runType)
            
        application_step_data.updateSavedObjectsFromSavedFields()
        
        step_helper.update_dependent_steps(application_step_data)


    def _updateKitsStepData(self, runTypeObj, step_helper, kits_step_data):
        kits_step_data.prepopulatedFields['is_chipType_required'] = step_helper.isPlan()
        
   
    def getStepHelperForTemplateRunType(self, run_type_id, step_helper_type=StepHelperType.CREATE_NEW_TEMPLATE, template_id = -1):
        '''
            Creates a template step helper for the specified runty.
        '''
        #logger.debug("ENTER step_helper_db_loader.getStepHelperForRunType() run_type_id=%s" %(str(run_type_id)))
        
        step_helper = StepHelper(sh_type=step_helper_type, previous_template_id = template_id)
        ionReporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        
        runType = RunType.objects.get(pk=run_type_id)
        step_helper.parentName = runType.description
        
        self._updateApplicationStepData(runType, step_helper, application_step_data)
                
        kits_step_data = step_helper.steps[StepNames.KITS]
        self._updateKitsStepData(runType, step_helper, kits_step_data)
                
        return step_helper
    
    def getStepHelperForNewTemplateBySample(self, run_type_id, step_helper_type=StepHelperType.CREATE_NEW_TEMPLATE_BY_SAMPLE):
        '''
            
        '''
        #logger.debug("ENTER step_helper_db_loader.getStepHelperForNewTemplateBySample()")
                
        step_helper = StepHelper(sh_type=step_helper_type)
        ionReporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        
        runType = RunType.objects.get(pk=run_type_id)
        step_helper.parentName = runType.description
        
        self._updateApplicationStepData(runType, step_helper, application_step_data)
        
        return step_helper
    
    def updateTemplateSpecificStepHelper(self, step_helper, planned_experiment):
        '''
            Updates the template specific step helper with template specific info from the planned experiment.
        '''
        #logger.debug("ENTER step_helper_db_loader.updateTemplateSpecificStepHelper()")                
        
        save_template_step_data = step_helper.steps[StepNames.SAVE_TEMPLATE]
        
        planDisplayedName = getPlanDisplayedName(planned_experiment)
        
        if step_helper.sh_type == StepHelperType.COPY_TEMPLATE:
            save_template_step_data.savedFields['templateName'] = "Copy of " + planDisplayedName
        else:
            save_template_step_data.savedFields['templateName'] = planDisplayedName
            
        save_template_step_data.savedFields['setAsFavorite'] = planned_experiment.isFavorite

        save_template_step_data.savedFields['note'] = planned_experiment.get_notes()        

   
    def updatePlanSpecificStepHelper(self, step_helper, planned_experiment, set_template_name=False):
        '''
            Updates the plan specific step helper with plan specific info from the planned experiment.
            
            If the planned experiment is a template and you'd like the originating template name to show up
            in the save plan page pass in set_template_name=True
        '''
        #logger.debug("ENTER step_helper_db_loader.updatePlanSpecificStepHelper()")

        planDisplayedName = getPlanDisplayedName(planned_experiment)

        if set_template_name:
            step_helper.steps[StepNames.SAVE_TEMPLATE].savedFields['templateName'] = planDisplayedName
        
        save_plan_step_data = step_helper.steps[StepNames.SAVE_PLAN]

        #Add a "copy of" if we're copying.
        if step_helper.isCopy():
            save_plan_step_data.savedFields['planName'] = "Copy of " + planDisplayedName
        else:
            save_plan_step_data.savedFields['planName'] = planDisplayedName
        
        save_plan_step_data.savedFields['note'] = planned_experiment.get_notes()
        save_plan_step_data.savedFields['barcodeSet'] = planned_experiment.get_barcodeId()
        
        save_plan_step_data.prepopulatedFields["plan_reference"] = planned_experiment.get_library()
        save_plan_step_data.prepopulatedFields["plan_targetRegionBedFile"] = planned_experiment.get_bedfile()
        save_plan_step_data.prepopulatedFields["plan_hotSpotRegionBedFile"] = planned_experiment.get_regionfile()
        save_plan_step_data.prepopulatedFields["runType"] = planned_experiment.runType
          
        isOncoSameSample = False

        if (planned_experiment.runType == "AMPS_DNA_RNA"):
            sample_count = planned_experiment.get_sample_count()
            barcode_count =  getPlanBarcodeCount(planned_experiment)
            isOncoSameSample =  (sample_count < barcode_count) or ("oncomine" in planned_experiment.categories.lower())               

        save_plan_step_data.savedFields["isOncoSameSample"] = isOncoSameSample               

        #logger.debug("step_helper_db_loader.updatePlanSpecificStepHelper isOncoSameSample=%s" %(isOncoSameSample))

        
        # add IonReporter parameters
        irInfo = self._getIRinfo(planned_experiment)
        if irInfo:
            save_plan_step_data.prepopulatedFields['selectedIr'] = irInfo['selectedIr']
            save_plan_step_data.prepopulatedFields['irConfigJson'] = irInfo['irConfigJson']
            save_plan_step_data.prepopulatedFields['setid_suffix'] = irInfo.get('setid_suffix')

            logger.debug("step_helper_db_loader.updatePlanSpecificStepHelper() irInfo=%s" %(irInfo))  
       
        samplesTable = self._getSamplesTable_from_plan(planned_experiment, step_helper, irInfo)
        if samplesTable:
            if planned_experiment.runType == "AMPS_DNA_RNA":
                samplesTable = sorted(samplesTable, key=lambda item: item['nucleotideType'])
            else:
                samplesTable = sorted(samplesTable, key=lambda item: item['sampleName'])

            save_plan_step_data.savedFields['samplesTable'] = json.dumps(samplesTable)
        
        if step_helper.isBarcoded():
            #do not copy sampleTubeLabel since a sample tube is meant for 1 run only
            save_plan_step_data.savedFields['barcodeSampleTubeLabel'] = "" if step_helper.isCopy() else planned_experiment.sampleTubeLabel
        
        save_plan_step_data.updateSavedObjectsFromSavedFields()

    def updateUniversalStepHelper(self, step_helper, planned_experiment):
        '''
            Update a step helper with info from planned experiment that applies to both plans and templates.
        '''
        #logger.debug("ENTER step_helper_db_loader.updateUniversalStepHelper()")
                
        # export_step_data = step_helper.steps[StepNames.EXPORT]
        ionreporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        kits_step_data = step_helper.steps[StepNames.KITS]
        reference_step_data = step_helper.steps[StepNames.REFERENCE]
        plugins_step_data = step_helper.steps[StepNames.PLUGINS]
        # if not step_helper.isPlanBySample():
        #     ionreporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        appl_product = None
        
        # application_step_data.updateFromStep(export_step_data)
        
        ionreporter_step_data.savedFields['sampleGrouping'] = planned_experiment.sampleGrouping.pk if planned_experiment.sampleGrouping else None 
        ionreporter_step_data.savedObjects['sampleGrouping'] = planned_experiment.sampleGrouping if planned_experiment.sampleGrouping else None 

        appl_product = self._updateUniversalStep_applicationData(step_helper, planned_experiment, application_step_data)

        logger.debug("step_helper_db_loader.updateUniversalStepHelper() planned_experiment.id=%d; applProduct.productCode=%s" %(planned_experiment.id, appl_product.productCode))

        self._updateUniversalStep_kitData(step_helper, planned_experiment, appl_product, application_step_data, kits_step_data)

        
        if step_helper.isEdit() or step_helper.isEditRun():
             # During plan editing, kits_step_data.updateFromStep() is executed before step_helper_db_loader.updateUniversalStepHelper().
            # This results in savedObjects[ApplicationFieldNames.APPL_PRODUCT] not getting set.
            # WORKAROUND: The following is a workaround to ensure prepopulatedFields are set for the Kits chevron 
            self._updateUniversalStep_kitData_for_edit(step_helper, planned_experiment, appl_product, application_step_data, kits_step_data)

        self._updateUniversalStep_referenceData(step_helper, planned_experiment, appl_product, reference_step_data)
        self._updateUniversalStep_pluginData_ionreporterData(step_helper, planned_experiment, appl_product, plugins_step_data, ionreporter_step_data)
            
        logger.debug("PLUGINS ARE: %s" % str(plugins_step_data.savedFields[StepNames.PLUGINS]))
        
        qc_values = planned_experiment.qcValues.all()
        for qc_value in qc_values:
            step_helper.steps['Monitoring'].savedFields[qc_value.qcName] = PlannedExperimentQC.objects.get(plannedExperiment__pk=planned_experiment.pk,
                                                                                                           qcType__pk=qc_value.pk).threshold
        logger.debug("QCs ARE: %s" % str(step_helper.steps['Monitoring'].savedFields))
        
        step_helper.steps[StepNames.OUTPUT].savedFields[OutputFieldNames.PROJECTS] = []
        projects = planned_experiment.projects.all()
        for project in projects:
            step_helper.steps[StepNames.OUTPUT].savedFields[OutputFieldNames.PROJECTS].append(project.pk)

    
    def _updateUniversalStep_applicationData(self, step_helper, planned_experiment, application_step_data):                
        selectedRunType = RunType.objects.filter(runType="GENS")[0:1][0]
        if selectedRunType.applicationGroups.all().count() > 0:
            application_step_data.savedFields['applicationGroup'] = selectedRunType.applicationGroups.all()[0:1][0].pk
            application_step_data.savedFields['applicationGroupName'] = selectedRunType.applicationGroups.all()[0:1][0].name
            
        #only set the runtype if its still a valid one, otherwise keep the default that was set in the constructor.
        if RunType.objects.filter(runType=planned_experiment.runType).count() > 0:
            selectedRunType = RunType.objects.filter(runType=planned_experiment.runType)[0:1][0]
            if hasattr(planned_experiment, 'applicationGroup') and planned_experiment.applicationGroup:
                application_step_data.savedFields['applicationGroup'] = planned_experiment.applicationGroup.pk
                application_step_data.savedFields['applicationGroupName'] = planned_experiment.applicationGroup.name
            else:
                #if no application group is selected, pick the first one associated with the runType
                application_step_data.savedFields['applicationGroup'] = selectedRunType.applicationGroups.all()[0:1][0].pk 
                application_step_data.savedFields['applicationGroupName'] = selectedRunType.applicationGroups.all()[0:1][0].name                
                #logger.debug("step_helper_db_loader.updateUniversalStepHelper() planned_experiment.id=%d; PICKING applicationGroup.id=%d" %(planned_experiment.id, application_step_data.savedFields['applicationGroup'])) 

        logger.debug("step_helper_db_loader._updateUniversalStep_applicationData() planned_experiment.id=%d; PICKING applicationGroup.id=%d; applicationGroupName=%s" %(planned_experiment.id, application_step_data.savedFields['applicationGroup'], application_step_data.savedFields['applicationGroupName'])) 

        application_step_data.savedFields['runType'] = selectedRunType.pk
        application_step_data.savedObjects['runType'] = selectedRunType
        
        #TODO: need to consider application group (based on planned_experiment's)
        appl_product = ApplProduct.objects.get(applType__runType = selectedRunType.runType, isDefault = True, isActive = True, isVisible = True)

        application_step_data.savedFields['categories'] =  planned_experiment.categories
        
        #logger.debug("step_helper_db_loader_updateUniversalStep_applicationData() helper.sh_type=%s   application_step_data.categories=%s" %(step_helper.sh_type, application_step_data.savedFields['categories']))
            
        return appl_product 


    def _updateUniversalStep_kitData(self, step_helper, planned_experiment, appl_product, application_step_data, kits_step_data):
        application_step_data.savedObjects['applProduct'] = appl_product
                
        kits_step_data.savedFields['templatekitType'] = "OneTouch"
        if planned_experiment.is_ionChef():
            kits_step_data.savedFields['templatekitType'] = "IonChef"
        
        kits_step_data.savedFields['templatekitname'] = planned_experiment.templatingKitName
        kits_step_data.savedFields['controlsequence'] = planned_experiment.controlSequencekitname
        kits_step_data.savedFields['samplePreparationKit'] = planned_experiment.samplePrepKitName
        kits_step_data.savedFields['barcodeId'] = planned_experiment.get_barcodeId()

        chipType = planned_experiment.get_chipType()
        kits_step_data.savedFields['chipType'] = 'P1.1.17' if chipType == '900v2' else chipType
        kits_step_data.prepopulatedFields['is_chipType_required'] = step_helper.isPlan()

        kits_step_data.savedFields['flows'] = planned_experiment.get_flows()
        kits_step_data.savedFields['forward3primeAdapter'] = planned_experiment.get_forward3primeadapter()
        kits_step_data.savedFields['libraryKey'] = planned_experiment.get_libraryKey()
        tfKey = planned_experiment.get_tfKey()
        if tfKey:
            kits_step_data.savedFields['tfKey'] = tfKey
        kits_step_data.savedFields['librarykitname'] = planned_experiment.get_librarykitname()
        kits_step_data.savedFields['sequencekitname'] = planned_experiment.get_sequencekitname()
        kits_step_data.savedFields['isDuplicateReads'] = planned_experiment.is_duplicateReads()
        kits_step_data.savedFields['base_recalibrate'] = planned_experiment.do_base_recalibrate()
        kits_step_data.savedFields['realign'] = planned_experiment.do_realign()

        avalanche3PrimeAdapters = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single', chemistryType = 'avalanche').order_by('-isDefault', 'name')
        kits_step_data.savedFields['avalancheForward3PrimeAdapter'] = avalanche3PrimeAdapters[0].sequence
        if appl_product.defaultAvalancheTemplateKit:
            kits_step_data.savedFields['avalancheTemplateKitName'] = appl_product.defaultAvalancheTemplateKit.name
        if appl_product.defaultAvalancheSequencingKit:
            kits_step_data.savedFields['avalancheSequencekitname'] = appl_product.defaultAvalancheSequencingKit.name
        
        nonAvalanche3PrimeAdapters = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').exclude(chemistryType = 'avalanche').order_by('-isDefault', 'name')
        kits_step_data.savedFields['nonAvalancheForward3PrimeAdapter'] = nonAvalanche3PrimeAdapters[0].sequence
        if appl_product.defaultTemplateKit:
            kits_step_data.savedFields['nonAvalancheTemplateKitName'] = appl_product.defaultTemplateKit.name
        if appl_product.defaultSequencingKit:
            kits_step_data.savedFields['nonAvalancheSequencekitname'] = appl_product.defaultSequencingKit.name
            
        kits_step_data.prepopulatedFields[KitsFieldNames.IS_BARCODE_KIT_SELECTION_REQUIRED] = appl_product.isBarcodeKitSelectionRequired        


    def _updateUniversalStep_kitData_for_edit(self, step_helper, planned_experiment, appl_product, application_step_data, kits_step_data):                
        # if editing a sequenced run old/obsolete chipType and kits must be included
        if step_helper.isEditRun():
            kits_step_data.prepopulatedFields['chipTypes'] = Chip.objects.filter(name=kits_step_data.savedFields['chipType'])
            kits_step_data.prepopulatedFields['controlSeqKits'] |= KitInfo.objects.filter(name=kits_step_data.savedFields['controlsequence'])
            kits_step_data.prepopulatedFields['samplePrepKits'] |= KitInfo.objects.filter(name=kits_step_data.savedFields['samplePreparationKit'])
            kits_step_data.prepopulatedFields['libKits'] |= KitInfo.objects.filter(name=kits_step_data.savedFields['librarykitname'])
            kits_step_data.prepopulatedFields['seqKits'] |= KitInfo.objects.filter(name=kits_step_data.savedFields['sequencekitname'])
            
            savedtemplatekit = KitInfo.objects.filter(name=kits_step_data.savedFields['templatekitname'])
            kits_step_data.prepopulatedFields['templateKits'] |= savedtemplatekit
            oneTouchKits = kits_step_data.prepopulatedFields['templateKitTypes']['OneTouch']['kit_values']
            ionChefKits = kits_step_data.prepopulatedFields['templateKitTypes']['IonChef']['kit_values']
            avalancheKits = kits_step_data.prepopulatedFields['templateKitTypes']['Avalanche']['kit_values']
            kits_step_data.prepopulatedFields['templateKitTypes']['OneTouch']['kit_values'] |= savedtemplatekit.filter(kitType__in=oneTouchKits.values_list('kitType',flat=True))
            kits_step_data.prepopulatedFields['templateKitTypes']['IonChef']['kit_values'] |= savedtemplatekit.filter(kitType__in=ionChefKits.values_list('kitType',flat=True))
            kits_step_data.prepopulatedFields['templateKitTypes']['Avalanche']['kit_values'] |= savedtemplatekit.filter(kitType__in=avalancheKits.values_list('kitType',flat=True))
            
        if step_helper.isEdit():
            if appl_product.applType.runType in ["AMPS", "AMPS_EXOME"]:
                kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in =["", "DNA", "AMPS_ANY"], isActive=True).order_by("name")            
            elif appl_product.applType.runType in ["AMPS_RNA"]:
                kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in =["", "RNA", "AMPS_ANY"], isActive=True).order_by("name")
            elif appl_product.applType.runType in ["RNA"]:
                kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType = "RNA", isActive=True).order_by("name")            
            elif appl_product.applType.runType in ["AMPS_DNA_RNA"]:
                kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in =["", "DNA", "RNA", "AMPS_ANY"], isActive=True).order_by("name")
            else:
                kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in =["", "DNA"], isActive=True).order_by("name")
                
            kits_step_data.prepopulatedFields[KitsFieldNames.BARCODES] = list(dnaBarcode.objects.values('name').distinct().order_by('name'))                              
            if appl_product.barcodeKitSelectableType == "":              
                kits_step_data.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type__in =["", "none"]).distinct().order_by('name'))   
            elif appl_product.barcodeKitSelectableType == "dna":
                kits_step_data.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type = "dna").distinct().order_by('name'))            
            elif appl_product.barcodeKitSelectableType == "rna":
                kits_step_data.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type = "rna").distinct().order_by('name'))            
            elif appl_product.barcodeKitSelectableType == "dna+":
                kits_step_data.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type__in =["dna", "", "none"]).distinct().order_by('name'))            
            elif appl_product.barcodeKitSelectableType == "rna+":
                kits_step_data.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type__in =["rna", "", "none"]).distinct().order_by('name'))            
            else:
                kits_step_data.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').distinct().order_by('name')) 

                       
        
    def _updateUniversalStep_referenceData(self, step_helper, planned_experiment, appl_product, reference_step_data):        
        reference_step_data.savedFields['targetBedFile'] = planned_experiment.get_bedfile()
        reference_step_data.savedFields["reference"] = planned_experiment.get_library()
        reference_step_data.savedFields['hotSpotBedFile'] = planned_experiment.get_regionfile()

        #logger.debug("step_helper_db_loader._updateUniversalStep_referenceData() REFERENCE plan_reference=%s" %(reference_step_data.savedFields["reference"]))
        
        reference_step_data.prepopulatedFields['showHotSpotBed'] = False
        if appl_product and appl_product.isHotspotRegionBEDFileSuppported:
            reference_step_data.prepopulatedFields['showHotSpotBed'] = True
    
        #if the plan or template has pre-selected reference info, it is possible that it is not found in db in this TS instance
        #a plan's or template's pre-selected reference info trumps applProducts default selection values!
        if reference_step_data.savedFields["reference"]:            
            reference_step_data.prepopulatedFields["referenceMissing"] = True
            if reference_step_data.savedFields["reference"] in [ref.short_name for ref in reference_step_data.prepopulatedFields["references"]]:
                reference_step_data.prepopulatedFields["referenceMissing"] = False
            else:
                logger.debug("at step_helper_db_loader.updateUniversalStepHelper() RERERENCE_MISSING saved reference=%s" %(reference_step_data.savedFields["reference"]));
        else:
            reference_step_data.prepopulatedFields["referenceMissing"] = False
    
        if  reference_step_data.savedFields["targetBedFile"]:
            
            reference_step_data.prepopulatedFields["targetBedFileMissing"] = True
            if reference_step_data.savedFields["targetBedFile"] in reference_step_data.file_dict["bedFileFullPaths"] or\
               reference_step_data.savedFields["targetBedFile"] in reference_step_data.file_dict["bedFilePaths"]:
                reference_step_data.prepopulatedFields["targetBedFileMissing"] = False
            else:
                logger.debug("at step_helper_db_loader.updateUniversalStepHelper() TARGED_BED_FILE_MISSING saved target=%s" %(reference_step_data.savedFields["targetBedFile"]));
        else:         
            reference_step_data.prepopulatedFields["targetBedFileMissing"] = False                      

        if reference_step_data.savedFields["hotSpotBedFile"]:
            
            reference_step_data.prepopulatedFields["hotSpotBedFileMissing"] = True
            if reference_step_data.savedFields["hotSpotBedFile"] in reference_step_data.file_dict["bedFileFullPaths"] or\
               reference_step_data.savedFields["hotSpotBedFile"] in reference_step_data.file_dict["bedFilePaths"]:
                reference_step_data.prepopulatedFields["hotSpotBedFileMissing"] = False
            else:
                logger.debug("at step_helper_db_loader.updateUniversalStepHelper() HOT_SPOT_BED_FILE_MISSING saved hotSpot=%s" %(reference_step_data.savedFields["hotSpotBedFile"]));
        else:        
            reference_step_data.prepopulatedFields["hotSpotBedFileMissing"] = False            

        stepHelper_type = step_helper.sh_type
        
        logger.debug("step_helper_db_loader._updateUniversalStep_referenceData() stepHelper_type=%s; reference_step_data.savedFields=%s" %(stepHelper_type, reference_step_data.savedFields))
        
        if stepHelper_type == StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE or stepHelper_type == StepHelperType.EDIT_PLAN_BY_SAMPLE or stepHelper_type == StepHelperType.COPY_PLAN_BY_SAMPLE:
            barcoding_step = step_helper.steps[StepNames.BARCODE_BY_SAMPLE]
            save_plan_step = step_helper.steps[StepNames.SAVE_PLAN_BY_SAMPLE]

            barcoding_step.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = reference_step_data.savedFields.get(ReferenceFieldNames.REFERENCE, "")
            barcoding_step.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
            barcoding_step.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.HOT_SPOT_BED_FILE, "")
        
            #logger.debug("step_helper_db_loader._updateUniversalStep_referenceData() stepHelper_type=%s; barcoding_step.savedFields=%s" %(stepHelper_type, barcoding_step.savedFields))
            #logger.debug("step_helper_db_loader._updateUniversalStep_referenceData() stepHelper_type=%s; step_helper=%s; barcoding_step=%s" %(stepHelper_type, step_helper, barcoding_step))
            
            save_plan_step.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = reference_step_data.savedFields.get(ReferenceFieldNames.REFERENCE, "")
            save_plan_step.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
            save_plan_step.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.HOT_SPOT_BED_FILE, "")

            barcoding_step.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = reference_step_data
            save_plan_step.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = reference_step_data
            
        elif stepHelper_type == StepHelperType.CREATE_NEW_PLAN or stepHelper_type == StepHelperType.COPY_PLAN or stepHelper_type == StepHelperType.EDIT_PLAN or stepHelper_type == StepHelperType.EDIT_RUN:
            save_plan_step_data = step_helper.steps[StepNames.SAVE_PLAN]
                        
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = reference_step_data.savedFields.get(ReferenceFieldNames.REFERENCE, "")
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.HOT_SPOT_BED_FILE, "")

            save_plan_step_data.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = reference_step_data

    
    
    def _updateUniversalStep_pluginData_ionreporterData(self, step_helper, planned_experiment, appl_product, plugins_step_data, ionreporter_step_data):
        plugins_step_data.savedFields[StepNames.PLUGINS] = []
        plugins = planned_experiment.get_selectedPlugins()
        pluginIds = []
        for plugin_name, plugin_dict in plugins.items():
            # find existing plugin by plugin_name (handles plugins that were reinstalled or uninstalled)
            try:
                plugin = Plugin.objects.filter(name=plugin_name, active=True)[0]
            except:
                continue

            ##we now need to show all non-IRU export plugins on the Plugins chevron
            if "ionreporter" in plugin_name.lower():
            ##if PluginFieldNames.EXPORT in plugin.pluginsettings.get(PluginFieldNames.FEATURES,[]):
                if not step_helper.isPlanBySample():
                    #ionreporter_step_data.savedFields['uploaders'].append(plugin.id)
                    pass
            else:
                pluginIds.append(plugin.id)
                plugins_step_data.savedFields[PluginFieldNames.PLUGIN_CONFIG % plugin.id] = json.dumps(plugin_dict.get(PluginFieldNames.USER_INPUT,''), cls=JSONEncoder, separators=(',', ':'))
            
            if 'accountId' in plugin_dict:
                ionreporter_step_data.savedFields['irAccountId'] = plugin_dict.get('accountId')
                ionreporter_step_data.savedFields['irAccountName'] = plugin_dict.get('accountName')
                ionreporter_step_data.savedFields['irVersion'] = plugin_dict.get('version')
            elif PluginFieldNames.USER_INPUT in plugin_dict and 'accountId' in plugin_dict[PluginFieldNames.USER_INPUT]:
                ionreporter_step_data.savedFields['irAccountId'] = plugin_dict[PluginFieldNames.USER_INPUT].get('accountId')
                ionreporter_step_data.savedFields['irAccountName'] = plugin_dict[PluginFieldNames.USER_INPUT].get('accountName')
                
                if 'userconfigs' in plugin.config:
                    if 'ionadmin' in plugin.config.get('userconfigs'):
                        _list = plugin.config.get('userconfigs').get('ionadmin')
                        for l in _list:
                            if l.get('id') == ionreporter_step_data.savedFields['irAccountId']:
                                ionreporter_step_data.savedFields['irVersion'] = l.get('version')

        if 'IonReporterUploader' not in plugins:
            ionreporter_step_data.savedFields['irAccountId'] = '0'
            ionreporter_step_data.savedFields['irAccountName'] = 'None'

        
        step_helper.steps[StepNames.IONREPORTER].savedFields['irworkflow'] = planned_experiment.irworkflow
        plugins_step_data.savedFields[PluginFieldNames.PLUGIN_IDS] = ', '.join(str(v) for v in pluginIds)
        plugins_step_data.updateSavedObjectsFromSavedFields()

            
    def get_ir_fields_dict_from_user_input_info(self, user_input_info, sample_name, index):
        #logger.debug("ENTER step_helper_db_loader.get_ir_fields_dict_from_user_input_info()")
                
        if sample_name == 'barcoded--Sample':
            if index >= len(user_input_info[0].keys()):
                return dict(
                    sample = "",
                    sampleDescription = "",
                    sampleExternalId = "",
                    barcodeId = "",
                    Gender = None,
                    RelationRole = None,
                    Workflow = None,
                    setid = None,
                    cancerType = None,
                    cellularityPct = None
                )

            sample_name = user_input_info[0].keys()[index]
            
            #do not re-invent what has already been persisted in the JSON blob!
            barcodeSampleInfo = user_input_info[0].get(sample_name).get('barcodeSampleInfo', {})
            barcode_id_strs = sorted(user_input_info[0].get(sample_name).get('barcodeSampleInfo').keys())

            barcode_id_str = user_input_info[0].get(sample_name).get('barcodeSampleInfo').keys()[0]
            sampleDescription = user_input_info[0].get(sample_name).get('barcodeSampleInfo').get(barcode_id_str).get('description')
            externalId = user_input_info[0].get(sample_name).get('barcodeSampleInfo').get(barcode_id_str).get('externalId')
            
            return dict(
                sample = sample_name,
                sampleDescription = sampleDescription,
                sampleExternalId = externalId,
                barcodeSampleInfo = barcodeSampleInfo,
                barcode_id_strs = barcode_id_strs,
                Gender = None,
                RelationRole = None,
                Workflow = None,
                setid = None,
                cancerType = None,
                cellularityPct = None                
            )
        else:
            return user_input_info[index]

    def updatePlanBySampleSpecificStepHelper(self, step_helper, planned_experiment, sampleset_id):
        """

        """
        #logger.debug("ENTER step_helper_db_loader.updatePlanBySampleSpecificStepHelper() planned_experiment.id=%d; step_helper=%s" %(planned_experiment.id, step_helper))
  
        barcoding_step = step_helper.steps[StepNames.BARCODE_BY_SAMPLE]
        save_plan_step = step_helper.steps[StepNames.SAVE_PLAN_BY_SAMPLE]
                        
        planDisplayedName = getPlanDisplayedName(planned_experiment)

        if step_helper.isCopy():
            save_plan_step.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = "Copy of " + planDisplayedName
        else:
            save_plan_step.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = planDisplayedName
        
        existing_plan = step_helper.isEdit() or step_helper.isCopy()

        barcoding_step.prepopulatedFields["runType"] = planned_experiment.runType
        save_plan_step.prepopulatedFields["runType"] = planned_experiment.runType
                
        isOncoSameSample = False

        if (planned_experiment.runType == "AMPS_DNA_RNA"):
            if existing_plan:
                sample_count = planned_experiment.get_sample_count()
                barcode_count =  getPlanBarcodeCount(planned_experiment)
                #logger.debug("step_helper_db_loader.updatePlanBySampleSpecificStepHelper() sample_count=%d; barcode_count=%d" %(sample_count, barcode_count))
                
                isOncoSameSample = (sample_count < barcode_count)  
            else:
                isOncoSameSample =  ("oncomine" in planned_experiment.categories.lower())               

        barcoding_step.savedFields["isOncoSameSample"] = isOncoSameSample
        save_plan_step.savedFields["isOncoSameSample"] = isOncoSameSample               
        
        if sampleset_id:
            sampleset = SampleSet.objects.get(pk=sampleset_id)
            if sampleset.SampleGroupType_CV:
                step_helper.steps[StepNames.APPLICATION].savedFields['sampleGrouping'] = sampleset.SampleGroupType_CV.pk
        else:
            sampleset = planned_experiment.sampleSet

        save_plan_step.savedObjects[SavePlanBySampleFieldNames.SAMPLESET] = sampleset
        
        sorted_sampleSetItems = list(sampleset.samples.all().order_by("sample__displayedName"))
        barcoding_step.prepopulatedFields['samplesetitems'] = sorted_sampleSetItems

        # Pick barcode set to use:
        #   1. Edit/Copy - get from plan
        #   2. Create - get from sampleSetItems or, if none, the barcode set selected in the plan template
        barcodeSet = planned_experiment.get_barcodeId()
        if not existing_plan:
            for item in sorted_sampleSetItems:
                if item.dnabarcode:
                    barcodeSet = item.dnabarcode.name
                    break
        barcoding_step.savedFields['barcodeSet'] = step_helper.steps[StepNames.KITS].savedFields['barcodeId'] = barcodeSet
        
        if barcodeSet:
            barcoding_step.prepopulatedFields['planned_dnabarcodes'] = list(dnaBarcode.objects.filter(name=barcodeSet).order_by('id_str'))

        # IonReporter parameters
        irInfo = self._getIRinfo(planned_experiment)
        if irInfo:
            barcoding_step.prepopulatedFields['selectedIr'] = irInfo['selectedIr']
            barcoding_step.prepopulatedFields['setid_suffix'] = irInfo.get('setid_suffix')

        # Populate samples table
        if existing_plan:
            samplesTable = self._getSamplesTable_from_plan(planned_experiment, step_helper, irInfo)
            #need to sort collection by nucleotideType
            samplesTable = sorted(samplesTable, key=lambda item: item['nucleotideType'])         
            samplesTable = sorted(samplesTable, key=lambda item: item['sampleName'])             
        else:
            samplesTable = []
            for item in sorted_sampleSetItems:
                sampleDict = {
                    "barcodeId"         : item.dnabarcode.id_str if item.dnabarcode else '', 
                    "sampleName"        : item.sample.displayedName,
                    "sampleExternalId"  : item.sample.externalId, 
                    "sampleDescription" : item.sample.description,
                    "nucleotideType"       : "",
                    "controlSequenceType"  : "",                    
                    "reference"            : "",
                    "targetRegionBedFile"  : "",
                    "hotSpotRegionBedFile" : "",
                    "cancerType"        : "",
                    "cellularityPct"   : "",
                    "irWorkflow"        : planned_experiment.irworkflow,
                    "irGender"          : item.gender,
                    "irRelationRole"    : item.relationshipRole,
                    "irSetID"           : item.relationshipGroup,
                    "ircancerType"      : item.cancerType,
                    "ircellularityPct"  : item.cellularityPct
                }
                
                #logger.debug("step_helper_db_loader.updatePlanBySampleSpecificStepHelper() sampleDict=%s" %(sampleDict))
                
                samplesTable.append(sampleDict)
        
        if samplesTable:
            barcoding_step.savedObjects['samplesTableList'] = samplesTable
            barcoding_step.savedFields['samplesTable'] = json.dumps(samplesTable)



    def _updatePlanBySampleSpecificStepHelper_barcodeKit(self, step_helper, planned_experiment, sampleset_id):
        """
            Hack: For plan by sample set, make sure the barcode kit is pre-set before processing at barcode_by_sample_step_data
            TODO: Revisit db_loader for plan by sample set!
        """
        logger.debug("ENTER step_helper_db_loader._updatePlanBySampleSpecificStepHelper_barcodeKit() planned_experiment.id=%d; sampleset_id=%s" %(planned_experiment.id, sampleset_id))
  
        barcoding_step = step_helper.steps[StepNames.BARCODE_BY_SAMPLE]
        save_plan_step = step_helper.steps[StepNames.SAVE_PLAN_BY_SAMPLE]
                        
        planDisplayedName = getPlanDisplayedName(planned_experiment)
        
        existing_plan = step_helper.isEdit() or step_helper.isCopy()
        
        if sampleset_id:
            sampleset = SampleSet.objects.get(pk=sampleset_id)
        else:
            sampleset = planned_experiment.sampleSet

        save_plan_step.savedObjects[SavePlanBySampleFieldNames.SAMPLESET] = sampleset
        
        sorted_sampleSetItems = list(sampleset.samples.all().order_by("sample__displayedName"))

        # Pick barcode set to use:
        #   1. Edit/Copy - get from plan
        #   2. Create - get from sampleSetItems or, if none, the barcode set selected in the plan template
        barcodeSet = planned_experiment.get_barcodeId()
        if not existing_plan:
            for item in sorted_sampleSetItems:
                if item.dnabarcode:
                    barcodeSet = item.dnabarcode.name
                    break
                
        logger.debug("step_helper_db_loader._updatePlanBySampleSpecificStepHelper_barcodeKit() planned_experiment.id=%d; barcodeSet=%s" %(planned_experiment.id, barcodeSet))

        ##TODO-uncomment-after-4.2.x-patch-to-use-constants barcoding_step.savedFields[SavePlanFieldNames.BARCODE_SET] = step_helper.steps[StepNames.KITS].savedFields[KitsFieldNames.BARCODE_ID] = barcodeSet
        barcoding_step.savedFields["barcodeSet"] = step_helper.steps[StepNames.KITS].savedFields["barcodeId"] = barcodeSet
        
        #crucial step
        barcoding_step.updateSavedFieldsForSamples()


    def _getIRinfo(self, planned_experiment):
        #logger.debug("ENTER step_helper_db_loader._getIRinfo()")
        
        # get IonReporterUploader parameters, if any
        for plugin_name, plugin_dict in planned_experiment.get_selectedPlugins().items():
            if 'IonReporter' in plugin_name:
                try:
                    plugin = Plugin.objects.filter(name=plugin_name, active=True)[0]
                except:
                    continue

                irInfo = {
                    'selectedIr': plugin,
                    'irConfigJson': json.dumps(plugin.userinputfields),
                    'userInputInfo': None
                }
                if PluginFieldNames.USER_INPUT in plugin_dict:
                    #Handle the old and the new style userinput in the plugin dictionary
                    if isinstance(plugin_dict[PluginFieldNames.USER_INPUT], dict):
                        userInputInfo = plugin_dict[PluginFieldNames.USER_INPUT].get("userInputInfo",[])
                        if userInputInfo and len(userInputInfo)>0:
                            irInfo['userInputInfo'] = userInputInfo
                            irInfo['setid_suffix'] = userInputInfo[0]['setid'][userInputInfo[0]['setid'].find('__'):]
                    elif isinstance(plugin_dict[PluginFieldNames.USER_INPUT], list) and len(plugin_dict[PluginFieldNames.USER_INPUT])>0:
                        irInfo['userInputInfo'] = plugin_dict[PluginFieldNames.USER_INPUT]
        
                return irInfo
        return None

    def _getSamplesTable_from_plan(self, planned_experiment, step_helper, irInfo=None):
        #logger.debug("ENTER step_helper_db_loader._getSamplesTable_from_plan()")
        
        samplesTable = []
            
        planNucleotideType = self._get_nucleotideType_byRunTypeOrApplGroup(planned_experiment)
        runType = planned_experiment.runType
        
        logger.debug("step_helper_db_loader._getSamplesTable_from_plan() planNucleotideType=%s; runType=%s" %(planNucleotideType, runType))
        
        if step_helper.isBarcoded():
            # build samples table from barcodedSamples
            
            sample_to_barcode = planned_experiment.get_barcodedSamples()

            #WORKAROUND FOR HUB: plan from HUB can have barcodeKit selected but with empty barcodedSamples JSON blob
            application_group_name = "" if not planned_experiment.applicationGroup else planned_experiment.applicationGroup.name
            #logger.debug("step_helper_db_loader._getSamplesTable_from_plan() application_group_name=%s" %(application_group_name))               
            
            
            if not sample_to_barcode:
                #logger.debug("step_helper_db_loader._getSamplesTable_from_plan() NO existing barcodedSamples for plan.pk=%d; planName=%s" %(planned_experiment.id, planned_experiment.planDisplayedName))               

                sampleInfo = None
                experiment = planned_experiment.experiment
                latest_eas = planned_experiment.latestEAS
                if experiment and experiment.samples.count() > 0:
                    sampleInfo = experiment.samples.values()[0]

                sampleDict = {"barcodeId"            : "", 
                              "sampleName"           : sampleInfo['displayedName'] if sampleInfo else "", 
                              "sampleExternalId"     : sampleInfo['externalId'] if sampleInfo else "", 
                              "sampleDescription"    : sampleInfo['description'] if sampleInfo else "", 
                              "nucleotideType"       : planNucleotideType,
                              "controlSequenceType"  : None,
                              "reference"            : planned_experiment.get_library() if planned_experiment.get_library() else "",
                              "hotSpotRegionBedFile" : planned_experiment.get_regionfile() if planned_experiment.get_regionfile() else "",
                              "targetRegionBedFile"  : planned_experiment.get_bedfile() if planned_experiment.get_bedfile() else "",                                      
                              }                        
                samplesTable.append(sampleDict)
                
                logger.debug("step_helper_db_loader._getSamplesTable_from_plan() NO existing barcodedSamples for plan.pk=%d; planName=%s; sampleDict=%s"  %(planned_experiment.id, planned_experiment.planDisplayedName, sampleDict))               
                        
            else:
                for sample, value in sample_to_barcode.items():                
                    if 'barcodeSampleInfo' in value:
 
                        for barcode, sampleInfo in value['barcodeSampleInfo'].items():
                            sampleReference = sampleInfo.get("reference", "")
                            ##if not sampleReference and planned_experiment.get_library():
                            if runType != "AMPS_DNA_RNA" and not sampleReference:
                                sampleReference = planned_experiment.get_library()
                                
                            sampleHotSpotRegionBedFile = sampleInfo.get("hotSpotRegionBedFile", "")
                            ##if (not planHotSpotRegionBedFile and planNucleotideType != "RNA"):
                            if runType != "AMPS_DNA_RNA"  and not sampleHotSpotRegionBedFile:
                                sampleHotSpotRegionBedFile = planned_experiment.get_regionfile()
                                
                            sampleTargetRegionBedFile = sampleInfo.get("targetRegionBedFile", "")
                            ##if (not planTargetRegionBedFile and planNucleotideType != "RNA"):
                            if runType != "AMPS_DNA_RNA"  and not sampleTargetRegionBedFile:
                                sampleTargetRegionBedFile = planned_experiment.get_bedfile()
                            
                            sampleDict = {
                                "barcodeId"            : barcode,
                                "sampleName"           : sample,
                                "sampleExternalId"     : sampleInfo.get('externalId',''),
                                "sampleDescription"    : sampleInfo.get('description',''),
                                "nucleotideType"       : sampleInfo.get("nucleotideType", planNucleotideType),  
                                "controlSequenceType"  : sampleInfo.get("controlSequnceType", ""),
                                "reference"            : sampleReference,
                                "hotSpotRegionBedFile" : sampleHotSpotRegionBedFile,
                                "targetRegionBedFile"  : sampleTargetRegionBedFile
                                 
                            }
                            samplesTable.append(sampleDict)
                            logger.debug("step_helper_db_loader._getSamplesTable_from_plan() barcodeSampleInfo plan.pk=%d; planName=%s; sampleName=%s; sampleDict=%s" %(planned_experiment.id, planned_experiment.planDisplayedName, sample, sampleDict))               

                    else:
                        #logger.debug("step_helper_db_loader._getSamplesTable_from_plan() NO barcodeSampleInfo plan.pk=%d; planName=%s; reference=%s " %(planned_experiment.id, planned_experiment.planDisplayedName, planned_experiment.get_library()))   
                        
                        for barcode in value.get('barcodes',[]):
                            sampleDict = {"barcodeId"            : barcode, 
                                          "sampleName"           : sample, 
                                          "sampleExternalId"     : None, 
                                          "sampleDescription"    : None,
                                          "nucleotideType"       : planNucleotideType,
                                          "controlSequenceType"  : None,
                                          "reference"            : planned_experiment.get_library(),
                                          "hotSpotRegionBedFile" : "" if planNucleotideType == "RNA" else planned_experiment.get_regionfile(),
                                          "targetRegionBedFile"  : "" if planNucleotideType == "RNA" else planned_experiment.get_bedfile(),                              
                                          }                  

                            samplesTable.append(sampleDict)

            # add IR values
            if irInfo and irInfo['userInputInfo']:
                barcodeToIrValues = {}
                for irvalues in irInfo['userInputInfo']:
                    barcodeId = irvalues.get('barcodeId')
                    if barcodeId:
                        barcodeToIrValues[barcodeId] = irvalues
                
                for sampleDict in samplesTable:
                    for irkey, irvalue in barcodeToIrValues.get(sampleDict['barcodeId'],{}).items():
                        if irkey == 'Relation':
                            sampleDict['irRelationshipType'] = irvalue
                        elif irkey == 'setid':
                            sampleDict['irSetID'] = irvalue.split('__')[0]
                        else:
                            sampleDict['ir'+irkey] = irvalue

        else:
            #when we load a non-barcoded run for editing/copying we know it will only have a single sample.
            sampleTubeLabel = "" if step_helper.isCopy() else planned_experiment.sampleTubeLabel
            if sampleTubeLabel is None:
                sampleTubeLabel = ""
                
            sampleDict = {
                'sampleName':  planned_experiment.get_sampleDisplayedName(),
                'sampleExternalId' : planned_experiment.get_sample_external_id(),
                'sampleDescription': planned_experiment.get_sample_description(),
                'tubeLabel':   sampleTubeLabel,
                "nucleotideType"       : planNucleotideType,                  
            }
            
            # add IR values
            if irInfo and irInfo['userInputInfo']:
                for irkey, irvalue in irInfo['userInputInfo'][0].items():
                    if irkey == 'Relation':
                        sampleDict['irRelationshipType'] = irvalue
                    elif irkey == 'setid':
                        sampleDict['irSetID'] = irvalue.split('__')[0]
                    else:
                        sampleDict['ir'+irkey] = irvalue
            
            samplesTable = [sampleDict]
        
        return samplesTable
        

    def _get_nucleotideType_byRunTypeOrApplGroup(self, plan):
        #logger.debug("ENTER step_helper_db_loader._get_nucleotideType_byRunTypeOrApplGroup()")
         
        if plan and plan.runType:
            runTypeObjs = RunType.objects.filter(runType = plan.runType)
            if runTypeObjs:
                runTypeObj = runTypeObjs[0]
                if (runTypeObj.nucleotideType):
                    return runTypeObj.nucleotideType.upper()
                else:
                    if plan.applicationGroup:
                        value = plan.applicationGroup.name
                        return value if value in ["DNA", "RNA"] else ""
                       
        return ""
    

    def getStepHelperForTemplatePlannedExperiment(self, pe_id, step_helper_type=StepHelperType.EDIT_TEMPLATE, sampleset_id=None):
        '''
            Get a step helper from a template planned experiment.
        '''
        #logger.debug("ENTER step_helper_db_loader.getStepHelperForTemplatePlannedExperiment() step_helper_type=%s; pe_id=%s" %(step_helper_type, str(pe_id)))

        planned_experiment = PlannedExperiment.objects.get(pk=pe_id)
        if not planned_experiment.isReusable:
            raise ValueError("You must pass in a template id, not a plan id.")
        
                    
        runType = planned_experiment.runType
        if runType:
            runTypeObjs = RunType.objects.filter(runType = runType)
            if runTypeObjs.count > 0:
                #logger.debug("step_helper_db_loader.getStepHelperForTemplatePlannedExperiment() runType_id=%d" %(runTypeObjs[0].id)) 
                step_helper = self.getStepHelperForTemplateRunType(runTypeObjs[0].id, step_helper_type, pe_id)
            else:
                step_helper = StepHelper(sh_type=step_helper_type, previous_template_id = pe_id)                
        else:
            step_helper = StepHelper(sh_type=step_helper_type, previous_template_id = pe_id)

        planDisplayedName = getPlanDisplayedName(planned_experiment)
        
        step_helper.parentName = planDisplayedName

        #retrieve and set barcode kit info before the rest of the logic is executed
        if step_helper.isPlan() and step_helper.isPlanBySample():
            self._updatePlanBySampleSpecificStepHelper_barcodeKit(step_helper, planned_experiment, sampleset_id)

        
        self.updateUniversalStepHelper(step_helper, planned_experiment)

        if step_helper.isPlan() and step_helper.isPlanBySample():
            self.updatePlanBySampleSpecificStepHelper(step_helper, planned_experiment, sampleset_id)
        elif step_helper.isPlan():
            self.updatePlanSpecificStepHelper(step_helper, planned_experiment, True)
        else:
            self.updateTemplateSpecificStepHelper(step_helper, planned_experiment)
        
        return step_helper
    
    def getStepHelperForPlanPlannedExperiment(self, pe_id, step_helper_type=StepHelperType.EDIT_PLAN):
        '''
            Get a plan step helper from a plan planned experiment.
        '''
        logger.debug("ENTER step_helper_db_loader.getStepHelperForPlanPlannedExperiment() step_helper_type=%s; pe_id=%s" %(step_helper_type, str(pe_id)))
        
        step_helper = StepHelper(sh_type=step_helper_type, previous_plan_id=pe_id)
        planned_experiment = PlannedExperiment.objects.get(pk=pe_id)

        planDisplayedName = getPlanDisplayedName(planned_experiment)

        step_helper.parentName = planDisplayedName
        
        if planned_experiment.isReusable:
            raise ValueError("You must pass in a plan id, not a template id.")

        self.updateUniversalStepHelper(step_helper, planned_experiment)
        if step_helper.isPlan() and step_helper.isPlanBySample():
            self.updatePlanBySampleSpecificStepHelper(step_helper, planned_experiment, planned_experiment.sampleSet.pk)
        elif step_helper.isPlan():
            self.updatePlanSpecificStepHelper(step_helper, planned_experiment)
        else:
            raise ValueError("Can not create templates from plans.")
        
        return step_helper
