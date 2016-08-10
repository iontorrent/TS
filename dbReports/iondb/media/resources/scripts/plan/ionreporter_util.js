
function get_decorated_workflow_name(workflowObj, workflowName, irReference, ionLabel, separator) {
    var decorated_workflow_name = workflowName;
    var isOCP = false;
    
    if ("OCP_Workflow" in workflowObj && workflowObj["OCP_Workflow"].toLowerCase() == "true") {
        isOCP = true;
        var displayedWorkflow =  workflowName + " (" + workflowObj["DNA_RNA_Workflow"] + ")";
        decorated_workflow_name += " (" + workflowObj["DNA_RNA_Workflow"];
    }     
    if (irReference !== "") {
        decorated_workflow_name += separator + irReference;
    }
    if ("tag_isFactoryProvidedWorkflow" in workflowObj && workflowObj["tag_isFactoryProvidedWorkflow"].toLowerCase() == "true") {
        decorated_workflow_name +=  separator + ionLabel;
    }
    if (isOCP) {
        decorated_workflow_name += ")";
    }

    return decorated_workflow_name;
}
