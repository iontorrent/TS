# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved


class StepHelperType():
    CREATE_NEW_TEMPLATE = "create_new_template"
    CREATE_NEW_TEMPLATE_BY_SAMPLE = "create_new_template_by_sample"

    EDIT_TEMPLATE = "edit_template"
    COPY_TEMPLATE = "copy_template"

    CREATE_NEW_PLAN = "create_new_plan"
    EDIT_PLAN = "edit_plan"
    COPY_PLAN = "copy_plan"

    CREATE_NEW_PLAN_BY_SAMPLE = "create_new_plan_by_sample"
    EDIT_PLAN_BY_SAMPLE = "edit_plan_by_sample"
    COPY_PLAN_BY_SAMPLE = "copy_plan_by_sample"

    EDIT_RUN = "edit_run"

    TEMPLATE_TYPES = [CREATE_NEW_TEMPLATE, EDIT_TEMPLATE, COPY_TEMPLATE, CREATE_NEW_TEMPLATE_BY_SAMPLE]
    PLAN_TYPES = [CREATE_NEW_PLAN, EDIT_PLAN, COPY_PLAN, EDIT_RUN,
                  CREATE_NEW_PLAN_BY_SAMPLE, EDIT_PLAN_BY_SAMPLE, COPY_PLAN_BY_SAMPLE]
    PLAN_BY_SAMPLE_TYPES = [CREATE_NEW_PLAN_BY_SAMPLE, EDIT_PLAN_BY_SAMPLE, COPY_PLAN_BY_SAMPLE]
