# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#!/usr/bin/python
# Ion Plugin Ion Reporter Uploader


def sample_relationship_fields():
    # Webservice Call For Workflow/ApplicationType Data Structure goes here. Replace later
    webservice_call = [{"Workflow": "TargetSeq Germline", "ApplicationType": "TargetSeq Germline"}, {"Workflow": "myTGWorkflow", "ApplicationType": "TargetSeq Germline"}, {"Workflow": "myDirectorsTGWorkflow", "ApplicationType": "TargetSeq Germline"}, {"Workflow": "DepartmentTGWorkflow", "ApplicationType": "TargetSeq Germline"}, {"Workflow": "Whole Genome", "ApplicationType": "Whole Genome"}, {"Workflow": "Annotate Variants", "ApplicationType": "Annotate Variants"}, {"Workflow": "TargetSeq Somatic", "ApplicationType": "TargetSeq Somatic"}, {"Workflow": "AmpliSeq Germline", "ApplicationType": "AmpliSeq Germline"}, {"Workflow": "AmpliSeq Somatic", "ApplicationType": "AmpliSeq Somatic"}, {"Workflow": "AmpliSeq2Wfl", "ApplicationType": "AmpliSeq Somatic"}, {"Workflow": "TumorNormal", "ApplicationType": "TumorNormal"}, {"Workflow": "myTNWorkflow", "ApplicationType": "TumorNormal"}]
    jsonfile = {}  # Represents JSON Object
    workflow_list = []

    # Add Workflows to List
    for x in webservice_call:
        workflow_list.append(x["Workflow"])

    # "columns" key in JSON
    columns_list = []
    order1 = {"Name": "Workflow", "Order": "1", "Type": "list", "ValueType": "String", "Values": workflow_list}
    order2 = {"Name": "RelationshipType", "Order": "2", "Type": "list", "ValueType": "String", "Values": ["Self", "TumorNormal", "Trio"]}
    order3 = {"Name": "SetId", "Order": "3", "Type": "input", "ValueType": "Integer"}
    order4 = {"Name": "Relation", "Order": "4", "Type": "list", "ValueType": "String", "Values": ["Tumor", "Normal", "Father", "Mother", "Child"]}
    columns_list.append(order1)
    columns_list.append(order2)
    columns_list.append(order3)
    columns_list.append(order4)
    jsonfile["columns"] = columns_list

    # "column-map" key in JSON. Directly get from webservice call
    jsonfile["column-map"] = webservice_call

    # "restrictionRules" key in JSON. These are hardcoded for now
    restrictionRules_list = []
    e1 = {"Name": "ApplicationType", "Value": "TumorNormal"}
    e2 = {"Name": "RelationshipType", "Values": ["TumorNormal"]}
    entree3 = {}
    entree3["For"] = e1
    entree3["Valid"] = e2
    restrictionRules_list.append(entree3)

    e1 = {"Name": "RelationshipType", "Value": "TumorNormal"}
    e2 = {"Name": "Relation", "Values": ["Tumor", "Normal"]}
    entree3 = {}
    entree3["For"] = e1
    entree3["Valid"] = e2
    restrictionRules_list.append(entree3)

    e1 = {"Name": "RelationshipType", "Value": "Trio"}
    e2 = {"Name": "Relation", "Values": ["Father", "Mother", "Child"]}
    entree3 = {}
    entree3["For"] = e1
    entree3["Valid"] = e2
    restrictionRules_list.append(entree3)

    e1 = {"Name": "RelationshipType", "Value": "Self"}
    e2 = {"Name": "SetID"}
    entree3 = {}
    entree3["For"] = e1
    entree3["Disabled"] = e2
    restrictionRules_list.append(entree3)

    e1 = {"Name": "RelationshipType", "Value": "Self"}
    e2 = {"Name": "Relation"}
    entree3 = {}
    entree3["For"] = e1
    entree3["Disabled"] = e2
    restrictionRules_list.append(entree3)

    jsonfile["restrictionRules"] = restrictionRules_list
    # 20120730-orig print json.dumps(jsonfile, sort_keys=True, indent=4)
    return jsonfile


def main():
    sample_relationship_fields()

# For development use
if __name__ == "__main__":
    main()
