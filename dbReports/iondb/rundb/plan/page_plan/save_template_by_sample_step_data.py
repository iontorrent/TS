# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from django.utils.translation import ugettext as _
from django.core.urlresolvers import reverse
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.plan_validator import (
    validate_plan_name,
    validate_notes,
    validate_QC,
)
from iondb.rundb.plan.page_plan.save_template_step_data import (
    SaveTemplateStepDataFieldNames,
)
from iondb.rundb.models import QCType
from iondb.rundb.labels import (
    ModelsQcTypeToLabelsQcTypeAsDict,
    ModelsQcTypeToLabelsQcType,
)
from collections import OrderedDict


class MonitoringFieldNames:
    QC_TYPES = "qcTypes"


class SaveTemplateBySampleStepData(AbstractStepData):
    def __init__(self, sh_type):
        super(SaveTemplateBySampleStepData, self).__init__(sh_type)
        self.resourcePath = "rundb/plan/page_plan/page_plan_by_sample_save_plan.html"
        self.prev_step_url = reverse("page_plan_output")
        self.next_step_url = reverse("page_plan_save")
        self.savedFields = OrderedDict()

        self.savedFields[SaveTemplateStepDataFieldNames.TEMPLATE_NAME] = None
        self.savedFields[SaveTemplateStepDataFieldNames.SET_AS_FAVORITE] = None
        self.savedFields[SaveTemplateStepDataFieldNames.NOTE] = None

        self.savedFields[SaveTemplateStepDataFieldNames.LIMS_META] = None
        self.savedObjects[SaveTemplateStepDataFieldNames.META] = {}

        self.sh_type = sh_type

        # Monitoring
        self.qcNames = []
        self.ModelsQcTypeToLabelsQcTypeAsDict = ModelsQcTypeToLabelsQcTypeAsDict
        all_qc_types = list(QCType.objects.all().order_by("qcName"))
        self.prepopulatedFields[MonitoringFieldNames.QC_TYPES] = all_qc_types
        for qc_type in all_qc_types:
            self.savedFields[qc_type.qcName] = qc_type.defaultThreshold
            self.qcNames.append(qc_type.qcName)

    def getStepName(self):
        return StepNames.SAVE_TEMPLATE_BY_SAMPLE

    def validateField(self, field_name, new_field_value):
        self.validationErrors.pop(field_name, None)

        if field_name == SaveTemplateStepDataFieldNames.TEMPLATE_NAME:
            errors = validate_plan_name(
                new_field_value,
                field_label=_(
                    "workflow.step.savetemplate.bysample.fields.templateName.label"
                ),
            )  #'Template Name'
            if errors:
                self.validationErrors[field_name] = "\n".join(errors)

        elif field_name == SaveTemplateStepDataFieldNames.NOTE:
            errors = validate_notes(
                new_field_value,
                field_label=_(
                    "workflow.step.savetemplate.bysample.fields.templateName.label"
                ),
            )
            if errors:
                self.validationErrors[field_name] = "\n".join(errors)

        # validate all qc thresholds must be positive integers
        elif field_name in self.qcNames:
            errors = validate_QC(
                new_field_value, ModelsQcTypeToLabelsQcType(field_name)
            )
            if errors:
                self.validationErrors[field_name] = errors[0]

    def updateSavedObjectsFromSavedFields(self):
        pass
