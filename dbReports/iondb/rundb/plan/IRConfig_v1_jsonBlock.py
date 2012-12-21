# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#!/usr/bin/python
# Ion Plugin Ion Reporter Uploader


def sample_relationship_fields():
    # Webservice Call For Workflow Data Structure goes here. Replace later
    webservice_call = [{"Workflow": "TargetSeq Germline"}, {"Workflow": "myTGWorkflow"}, {"Workflow": "myDirectorsTGWorkflow"}, {"Workflow": "DepartmentTGWorkflow"}, {"Workflow": "Whole Genome"}, {"Workflow": "Annotate Variants"}, {"Workflow": "TargetSeq Somatic"}, {"Workflow": "AmpliSeq Germline"}, {"Workflow": "AmpliSeq Somatic"}, {"Workflow": "AmpliSeq2Wfl"}, {"Workflow": "TumorNormal"}, {"Workflow": "myTNWorkflow"}, {"Workflow": "myIR_v1 Test Workflow"}]
    jsonfile = {}  # Represents JSON Object
    workflow_list = []

    # Add Workflows to List
    for x in webservice_call:
        workflow_list.append(x["Workflow"])

    # "columns" key in JSON
    columns_list = []
    order1 = {"Name": "Workflow", "Order": "1", "Type": "list", "ValueType": "String", "Values": workflow_list}
    columns_list.append(order1)
    jsonfile["columns"] = columns_list

    # "column-map" key in JSON. Directly get from webservice call
    jsonfile["column-map"] = webservice_call

    ##20120821-orig print json.dumps(jsonfile, sort_keys=True, indent=4)
    return jsonfile


def main():
    sample_relationship_fields()

# For development use
if __name__ == "__main__":
    main()
